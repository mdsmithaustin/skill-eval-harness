from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from ablation_model import scorable_run
from cli_contracts import die
from experimental_pairs import ExperimentalPairKey, ExperimentalPopulation
from json_contracts import strict_json_loads
from telemetry import (
    measurement_from_envelope_or_cost,
    measurement_from_envelope_or_nonnegative,
    measurement_from_envelope_or_usage,
)
from trace_contracts import (
    EventStateSource,
    InvalidEventLog,
    parse_event_log,
    parse_event_state,
)


def stable_id(*parts: Any) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(["skill-eval-harness/langsmith/v1", *parts], sort_keys=True, allow_nan=False),
        )
    )


def identifiers(data: Mapping, keys: tuple[str, ...]) -> dict:
    return {k: v for k in keys if isinstance(v := data.get(k), (str, bool, float, int))}


def native_event_file(runs: Path, row: dict) -> Path:
    recorded = PurePosixPath(str(row["run_base"]).replace("\\", "/"))
    case, model, variant = row["case_id"], row.get("model"), row["variant"]
    layouts = [(case, variant)]
    if model:
        layouts.insert(0, (case, model, variant))
    for layout in layouts:
        if any(part in (".", "..", "") or "/" in part or "\\" in part for part in layout):
            continue
        for suffix in ((f"run-{row['run_number']}",), ()):
            if not suffix and row["run_number"] != 1:
                continue
            parts = (*layout, *suffix)
            if recorded.parts[-len(parts) :] == parts:
                artifact = runs.joinpath(*parts, "events.json").resolve()
                if not artifact.is_relative_to(runs.resolve()):
                    raise ValueError("events.json falls outside --runs")
                return artifact
    raise ValueError("recorded run_base does not match the benchmark row")


def native_attempts(report: dict, runs: Path | None) -> list[dict]:
    attempts = []
    for row in report["results"]:
        if row["split"] != "tune":
            continue
        pair = ExperimentalPairKey.from_row(row, population=ExperimentalPopulation.ANSWER)
        identity = {**pair.to_dict(), "variant": row["variant"]}
        metadata = row.get("metadata", {})
        details = {
            **identifiers(row, ("split", "missing_output", "execution_valid", "grading_availability")),
            **identifiers(
                metadata,
                (
                    "skill_tree_hash",
                    "fixture_tree_hash",
                    "served_model",
                    "requested_model",
                    "effort",
                    "operation_observation_complete",
                    "trace_observation_complete",
                    "answer_design_sha256",
                    "answer_task_sha256",
                    "answer_instruction_sha256",
                ),
            ),
            "comparison_group": pair.to_dict(),
            "report_availability": (
                report.get("availability")
                if report.get("availability") in ("partial", "complete")
                else "unknown"
            ),
            "skill_revisions": [
                identifiers(item, ("name", "digest", "digest_scheme"))
                for item in metadata.get("skill_revisions", [])
            ],
        }
        if isinstance(metadata.get("effort"), dict):
            details["effort"] = identifiers(metadata["effort"], ("requested", "applied_by"))
        if metadata.get("operation_observation_complete") is True and isinstance(
            metadata.get("skill_invoked"), bool
        ):
            details["skill_invoked"] = metadata["skill_invoked"]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage = measurement_from_envelope_or_usage(metadata, key)
            if usage.value is not None:
                details[key] = usage.value
        elapsed = measurement_from_envelope_or_nonnegative(metadata, "elapsed_ms")
        if elapsed.value is not None:
            details["elapsed_ms"] = elapsed.value
        cost = measurement_from_envelope_or_cost(metadata)
        if cost.value is not None:
            details["cost"] = cost.value.to_dict()
            details["cost_provenance"] = cost.provenance
            details["cost_basis"] = identifiers(
                dict(cost.basis), ("provider", "billing_scope", "pricing_model", "pricing_table_version")
            )
        events = []
        if runs:
            artifact = native_event_file(runs, row)
            if artifact.exists():
                raw = strict_json_loads(artifact.read_bytes())
                if isinstance(raw, dict):
                    raw = {key: raw[key] for key in ("schema_version", "events") if key in raw}
                log = parse_event_log(raw)
                if isinstance(log, InvalidEventLog):
                    raise ValueError(log.reason)
                events = [
                    {
                        "kind": event["type"],
                        **identifiers(event, ("name", "status", "state_source", "timestamp")),
                    }
                    for event in log.events
                ]
        outcomes = []
        if scorable_run(row):
            for key in (
                "objective_pass_rate",
                "combined_pass_rate",
                "outcome_pass_rate",
                "process_pass_rate",
                "efficiency_pass_rate",
            ):
                if isinstance(row.get(key), (float, int)):
                    outcomes.append({"key": key, "value": row[key], "basis": "benchmark"})
            for basis, field in (("assertion", "assertions"), ("judge", "qualitative_assertions")):
                for index, assertion in enumerate(row.get(field, []), 1):
                    if (
                        isinstance(assertion.get("passed"), bool)
                        and not assertion.get("skipped")
                        and assertion.get("availability", "complete") == "complete"
                    ):
                        outcomes.append(
                            {
                                "key": f"{index}.{assertion['type']}",
                                "value": assertion["passed"],
                                "basis": basis,
                            }
                        )
        attempts.append({"identity": identity, "metadata": details, "events": events, "outcomes": outcomes})
    return attempts


def moment(value: Any, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    text = str(value)
    if re.fullmatch(r"\d{10}(?:\.\d+)?", text):
        return datetime.fromtimestamp(float(text), timezone.utc)
    if re.fullmatch(r"\d{13}", text):
        return datetime.fromtimestamp(int(text) / 1000, timezone.utc)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_export(document: dict, *, project: str, study: str, runs: Path | None = None) -> list[dict]:
    if not isinstance(document, dict):
        raise TypeError("expected a result document")
    project = str(uuid.UUID(project))
    if document.get("source") == "scbench":
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError("unsupported SCBench index version")
        if runs:
            raise ValueError("--runs applies only to a native benchmark")
        source, attempts = "scbench", document["attempts"]
    else:
        if document.get("population") != "answer" or "results" not in document:
            raise ValueError("expected a native answer benchmark or a v1 SCBench result index")
        source, attempts = "skill-harness", native_attempts(document, runs)

    requests: list[dict] = []
    root_ids = set()
    for attempt_index, attempt in enumerate(attempts):
        meta = attempt["metadata"]
        if meta.get("split") in {"holdout", "holdback"}:
            continue
        if source == "scbench":
            identity = {k: attempt["identity"][k] for k in ("arm", "run_id", "problem", "checkpoint")}
            if any(not isinstance(identity[k], str) or not identity[k] for k in ("arm", "run_id", "problem")):
                raise ValueError("invalid SCBench attempt identity")
            if type(identity["checkpoint"]) is not int or identity["checkpoint"] < 1:
                raise ValueError("invalid SCBench checkpoint")
            details = identifiers(
                meta,
                (
                    "model_name",
                    "seed",
                    "state",
                    "agent_type",
                    "agent_version",
                    "comparison",
                    "cost_basis",
                    "skill_revision",
                    "skill_revision_source",
                    "effort",
                    "skill_tree_hash",
                    "starting_snapshot",
                ),
            )
            for key in (
                "cost",
                "input",
                "output",
                "cache_read",
                "cache_write",
                "reasoning",
                "duration",
                "steps",
            ):
                if type(meta.get(key)) in (int, float):
                    details[key] = meta[key]
            details["thread_id"] = stable_id(
                project, study, source, {k: identity[k] for k in ("arm", "run_id", "problem")}
            )
        else:
            identity, details = attempt["identity"], dict(meta)
        trace_id = stable_id(project, study, source, identity)
        if trace_id in root_ids:
            raise ValueError("duplicate attempt identity in export")
        root_ids.add(trace_id)
        details.update({"source": source, "study": study, **identity})
        events = attempt["events"]
        fallback = datetime.fromtimestamp(document.get("generated_at", 0), timezone.utc)
        observed = [moment(e["timestamp"], fallback) for e in events if e.get("timestamp")]
        started, ended = meta.get("started"), meta.get("ended")
        if observed and started and datetime.fromisoformat(started.replace("Z", "+00:00")).tzinfo is None:
            started = ended = None
        start = moment(started, min(observed, default=fallback))
        times = [
            moment(e.get("timestamp"), start + timedelta(microseconds=i)) for i, e in enumerate(events, 1)
        ]
        end = max([start, moment(ended, start), *times])
        details["time_basis"] = "recorded" if started or observed else "report_time"

        def order(t: datetime, uid: str) -> str:
            return t.strftime("%Y%m%dT%H%M%S%fZ") + uid

        root_order = order(start, trace_id)
        common = {"session_id": project, "trace_id": trace_id}
        root = {
            **common,
            "id": trace_id,
            "name": source,
            "run_type": "chain",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "dotted_order": root_order,
            "inputs": identity,
            "outputs": {},
            "extra": {"metadata": details},
        }
        requests.append({"path": "runs", "body": root})
        for i, (event, timestamp) in enumerate(zip(events, times), 1):
            event_id = stable_id(trace_id, "event", i)
            info = identifiers(event, ("kind", "name", "actor", "author"))
            if "status" in event:
                info["status"] = parse_event_state(event["status"], status_present=True).state.value
            if event.get("state_source") in tuple(state.value for state in EventStateSource):
                info["state_source"] = event["state_source"]
            if source == "scbench":
                info["index_ref"] = f"#/attempts/{attempt_index}/events/{i - 1}"
                if event.get("spec_verdict") in ("pass", "fail"):
                    info.update({"spec_verdict": event["spec_verdict"], "verdict_basis": "reviewer_report"})
            if "name" in info and not re.fullmatch(r"[\w.:-]{1,120}", str(info["name"])):
                info.pop("name")
            requests.append(
                {
                    "path": "runs",
                    "body": {
                        **common,
                        "id": event_id,
                        "parent_run_id": trace_id,
                        "dotted_order": root_order + "." + order(timestamp, event_id),
                        "name": info.get("name") or info["kind"],
                        "run_type": "tool" if info["kind"] in {"tool", "tool_call", "command"} else "chain",
                        "start_time": timestamp.isoformat(),
                        "end_time": timestamp.isoformat(),
                        "inputs": {},
                        "outputs": {},
                        "extra": {"metadata": {**info, "event_number": i}},
                    },
                }
            )
        for metric_index, metric in enumerate(attempt["outcomes"]):
            value = metric["value"]
            if value is None:
                continue
            key = metric["key"]
            basis = metric["basis"]
            metric_id = stable_id(trace_id, "feedback", basis, key)
            if isinstance(value, str):
                value = stable_id(trace_id, "label", value)
                metric_id = stable_id(metric_id, value)
            elif not isinstance(value, (bool, float, int)):
                raise TypeError("expected scalar outcome")
            rating = {"value": value}
            if isinstance(value, bool):
                rating["score"] = value
            elif isinstance(value, (int, float)) and abs(value) <= 99999.9999:
                rating["score"] = round(value, 4)
            requests.append(
                {
                    "path": "feedback",
                    "body": {
                        **common,
                        "id": metric_id,
                        "run_id": trace_id,
                        "start_time": start.isoformat(),
                        "key": f"{basis}.{key}",
                        "comment": (
                            f"{basis} #/attempts/{attempt_index}/outcomes/{metric_index}"
                            if source == "scbench"
                            else basis
                        ),
                        **rating,
                    },
                }
            )
    return requests


def upload(requests: list[dict], endpoint: str, api_key: str) -> None:
    if not re.fullmatch(r"[!-~]+", api_key):
        raise ValueError("LANGSMITH_API_KEY contains invalid header characters")
    url = urlsplit(endpoint)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("LANGSMITH_ENDPOINT must be an HTTPS API base URL")
    host = url.hostname.rstrip(".")
    if host == "api.smith.langchain.com" or host.endswith(".api.smith.langchain.com"):
        now = datetime.now(timezone.utc)
        for request in requests:
            if request["path"] == "runs":
                run = request["body"]
                if abs(datetime.fromisoformat(run["start_time"]) - now) > timedelta(hours=24):
                    raise ValueError(
                        f"Run {run['id']} starts at {run['start_time']}, outside the hosted LangSmith "
                        "24-hour ingestion window. No records sent. Upload fresh runs as they finish, "
                        "or omit --upload to inspect the archive locally."
                    )
    connection = http.client.HTTPSConnection(url.hostname, url.port, timeout=30)
    try:
        for request in requests:
            body = request["body"]
            path = f"{url.path.rstrip('/')}/{request['path']}"
            payload = json.dumps(body, allow_nan=False).encode()
            headers = {"Content-Type": "application/json", "X-Api-Key": api_key}
            connection.request("POST", path, body=payload, headers=headers)
            response = connection.getresponse()
            response.read()
            if response.status == 409:
                update = (
                    identifiers(body, ("score", "value", "comment"))
                    if request["path"] == "feedback"
                    else body
                )
                connection.request(
                    "PATCH", f"{path}/{body['id']}", body=json.dumps(update).encode(), headers=headers
                )
                response = connection.getresponse()
                response.read()
            if not 200 <= response.status < 300:
                raise ValueError(
                    f"LangSmith returned HTTP {response.status} for {request['path']} {body['id']}"
                )
    finally:
        connection.close()


def export_langsmith(args: argparse.Namespace) -> int:
    try:
        path = Path(args.source)
        if path.is_dir():
            path = path / "benchmark.json"
        document = strict_json_loads(path.read_bytes())
        requests = build_export(
            document, project=args.project, study=args.study, runs=Path(args.runs) if args.runs else None
        )
        if args.upload:
            key = os.environ.get("LANGSMITH_API_KEY")
            if not key:
                raise ValueError("set LANGSMITH_API_KEY to upload")
            upload(requests, os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"), key)
            print(f"Uploaded {len(requests)} requests", file=sys.stderr)
        else:
            print(json.dumps(requests, ensure_ascii=False, indent=2, allow_nan=False))
    except (KeyError, TypeError, ValueError, OSError, http.client.HTTPException) as exc:
        die(str(exc))
    return 0
