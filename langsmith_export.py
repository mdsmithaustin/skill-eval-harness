from __future__ import annotations

import argparse
import hashlib
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
from claude_inspection import inspect_stdout
from cli_contracts import die
from experimental_pairs import ExperimentalPairKey, ExperimentalPopulation
from json_contracts import strict_json_loads
from scbench_codex import explicit_skill_names, inspect_exec, inspect_rollouts
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
from trace_normalization import observed_skill_name, parse_trace_jsonl_text_with_lines


def stable_id(*parts: Any) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(
                ["skill-eval-harness/langsmith/v1", *parts],
                sort_keys=True,
                allow_nan=False,
            ),
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
        if any(
            part in (".", "..", "") or "/" in part or "\\" in part for part in layout
        ):
            continue
        for suffix in ((f"run-{row['run_number']}",), ()):
            if not suffix and row["run_number"] != 1:
                continue
            parts = (*layout, *suffix)
            if recorded.parts[-len(parts) :] == parts:
                artifact = runs.joinpath(*parts, "events.json").resolve()
                if not artifact.is_relative_to(runs.resolve()):
                    raise ValueError("events.json falls outside --runs")
                if artifact != runs.resolve().joinpath(*parts, "events.json"):
                    raise ValueError("events.json is aliased to another artifact")
                return artifact
    raise ValueError("recorded run_base does not match the benchmark row")


def native_inspection(
    runs: Path, row: dict
) -> tuple[list[dict], list[dict], list[dict], str]:
    artifact = native_event_file(runs, row)
    if not artifact.exists():
        return [], [], [], "report_only"
    raw = strict_json_loads(artifact.read_bytes())
    dialect = raw.get("source") if isinstance(raw, dict) else None
    if isinstance(raw, dict):
        raw = {key: raw[key] for key in ("schema_version", "events") if key in raw}
    log = parse_event_log(raw)
    if isinstance(log, InvalidEventLog):
        raise TypeError(log.reason)
    normalized = [dict(event) for event in log.events]
    events = []
    for event in normalized:
        kind = event["type"]
        if kind in {"file_read", "file_write", "skill_load"}:
            kind = "tool_call"
        item = {
            "kind": kind,
            **identifiers(
                event,
                (
                    "name",
                    "status",
                    "state_source",
                    "timestamp",
                    "role",
                    "synthetic",
                    "exit_code",
                    "is_error",
                ),
            ),
        }
        ref = event.get("raw_ref")
        if (
            isinstance(ref, dict)
            and ref.get("file") == "trace.jsonl"
            and type(ref.get("line")) is int
            and ref["line"] > 0
        ):
            item["source"] = f"trace.jsonl:{ref['line']}"
        if observed := observed_skill_name(event):
            item.update(skill_name=observed, evidence="successful_skill_operation")
        events.append(item)
    if dialect not in {"codex", "claude"}:
        return events, [], normalized, "normalized_events_only"

    def read(path: Path) -> tuple[list[tuple[int, dict]], bool]:
        if path.resolve() != path or not path.is_relative_to(artifact.parent):
            raise ValueError("native trace falls outside its run")
        records, errors, lines, _ = parse_trace_jsonl_text_with_lines(
            path.read_text(encoding="utf-8", errors="replace"),
            strict_json_errors=False,
            strict=False,
        )
        return list(zip(lines, records, strict=True)), bool(errors)

    paths = (
        sorted((artifact.parent / "codex-rollouts").glob("rollout-*.jsonl"))
        if dialect == "codex"
        else []
    )
    if paths:
        files = []
        partial = False
        for path in paths:
            rows, errors = read(path)
            files.append((path.relative_to(artifact.parent).as_posix(), rows))
            partial |= errors
        observations, spans = inspect_rollouts(files)
    else:
        path = artifact.with_name("trace.jsonl")
        if not path.exists():
            return events, [], normalized, "normalized_events_only"
        rows, partial = read(path)
        if dialect == "claude":
            observations, spans = inspect_stdout(rows, source="trace.jsonl")
        elif any(record.get("type") == "session_meta" for _, record in rows):
            observations, spans = inspect_rollouts([("trace.jsonl", rows)])
        else:
            observations, spans = inspect_exec(rows, events)
            return (
                observations,
                spans,
                normalized,
                "partial_raw_trace" if partial else "turn_aggregate_only",
            )
    basis = (
        "provider_requests"
        if any(s["kind"] == "llm" for s in [*observations, *spans])
        else "provider_events_only"
    )
    return observations, spans, normalized, "partial_raw_trace" if partial else basis


def native_attempts(report: dict, runs: Path | None) -> list[dict]:
    attempts = []
    for row in report["results"]:
        if row["split"] != "tune":
            continue
        pair = ExperimentalPairKey.from_row(
            row, population=ExperimentalPopulation.ANSWER
        )
        identity = {**pair.to_dict(), "variant": row["variant"]}
        metadata = row.get("metadata", {})
        details = {
            **identifiers(
                row,
                ("split", "missing_output", "execution_valid", "grading_availability"),
            ),
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
            details["effort"] = identifiers(
                metadata["effort"], ("requested", "applied_by")
            )
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
                dict(cost.basis),
                ("provider", "billing_scope", "pricing_model", "pricing_table_version"),
            )
        events, spans = [], []
        details["model_evidence"] = "report_only"
        if runs:
            events, spans, normalized, details["model_evidence"] = native_inspection(
                runs, row
            )
            artifact = native_event_file(runs, row)
            prompt = artifact.with_name("task-prompt.txt")
            if prompt.is_file():
                if prompt.resolve() != prompt:
                    raise ValueError("native task prompt falls outside its run")
                contents = prompt.read_bytes()
                text = contents.decode("utf-8")
                task = {
                    "id": "prepared-task-prompt",
                    "kind": "prompt",
                    "name": "Task prompt",
                    "evidence": "prepared_task_prompt",
                    "role": "user",
                    "digest": hashlib.sha256(contents).hexdigest(),
                    "prompt_text": text,
                }
                if requested := explicit_skill_names(text):
                    task.update(
                        skill_name=requested[0],
                        skill_names=requested,
                        evidence="explicit_skill_request",
                    )
                spans.append(task)
            targets = {r.get("name") for r in details["skill_revisions"]} | {
                report.get("skill_name")
            }
            if isinstance(names := metadata.get("target_skill_names"), list):
                targets.update(name for name in names if isinstance(name, str))
            targets.discard(None)
            if targets and details["model_evidence"] != "report_only":
                skill_observations = {
                    item.get("evidence")
                    for item in [*events, *spans]
                    if item.get("skill_name") in targets
                    or any(
                        isinstance(name, str) and name in targets
                        for name in item.get("skill_names", [])
                    )
                }
                if skill_observations & {"explicit_skill_request", "explicit_command"}:
                    details["skill_requested"] = True
                if skill_observations & {"injected_skill", "skill_directory_injection"}:
                    details["skill_injected"] = True
                matched = any(
                    observed_skill_name(event) in targets for event in normalized
                ) or bool(skill_observations & {"skill_completed", "skill_file_read"})
                if matched or (
                    metadata.get("operation_observation_complete") is True
                    and details["model_evidence"] != "partial_raw_trace"
                ):
                    details["skill_invoked"] = matched
                    details["skill_invocation_basis"] = (
                        "target_name_in_observed_operations"
                    )
        elif (
            isinstance(metadata.get("skill_invoked"), bool)
            and metadata.get("operation_observation_complete") is True
        ):
            details["reported_skill_invoked"] = metadata["skill_invoked"]
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
                    outcomes.append(
                        {"key": key, "value": row[key], "basis": "benchmark"}
                    )
            for basis, field in (
                ("assertion", "assertions"),
                ("judge", "qualitative_assertions"),
            ):
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
        attempts.append(
            {
                "identity": identity,
                "metadata": details,
                "events": events,
                "spans": spans,
                "outcomes": outcomes,
            }
        )
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


def token_usage(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}

    def counts(data, keys):
        return {k: data[k] for k in keys if type(data.get(k)) is int and data[k] >= 0}

    result = counts(value, ("input_tokens", "output_tokens", "total_tokens"))
    for key, fields in (
        ("input_token_details", ("cache_read", "cache_creation")),
        ("output_token_details", ("reasoning",)),
    ):
        if isinstance(value.get(key), dict) and (details := counts(value[key], fields)):
            result[key] = details
    return result


def model_content(event: dict, directory: Path, identity: dict) -> tuple[dict, dict]:
    root = directory.resolve()
    if (identity["arm"], identity["run_id"]) != (root.parent.name, root.name):
        raise ValueError("model-content archive does not match the SCBench attempt")
    checkpoint = (
        root
        / identity["problem"]
        / f"checkpoint_{identity['checkpoint']}"
        / "agent/workspace/raw-api"
    ).resolve()
    if not checkpoint.is_relative_to(root):
        raise ValueError("model capture directory escapes archive")
    bodies = []
    for side in ("request", "response"):
        location = event.get(f"{side}_body_source")
        if location is None:
            bodies.append({})
            continue
        path = (root / location).resolve()
        if path.parent.parent != checkpoint or not path.name.endswith(f".{side}.json"):
            raise ValueError("invalid archived model body reference")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != event.get(f"{side}_body_digest"):
            raise ValueError("archived model body digest mismatch")
        value = strict_json_loads(data)
        if not isinstance(value, dict):
            raise TypeError("expected archived model body object")
        bodies.append(value)
    return bodies[0], bodies[1]


def order(t: datetime, uid: str) -> str:
    return t.strftime("%Y%m%dT%H%M%S%fZ") + uid


def nested_run_requests(
    root: dict, nodes: list, bodies: dict, by_key: dict
) -> list[dict]:
    ordered: list[int] = []
    pending: set[int] = set()

    def visit(i: int) -> None:
        body = bodies[i]
        if "parent_run_id" in body:
            return
        if i in pending:
            raise ValueError("cyclic trace parent reference")
        pending.add(i)
        parent = root
        parent_key = nodes[i - 1].get("parent_id")
        if parent_key is not None:
            if parent_key not in by_key:
                raise ValueError("missing trace parent")
            parent_index = by_key[parent_key]
            visit(parent_index)
            parent = bodies[parent_index]
        body["parent_run_id"] = parent["id"]
        ordered.append(i)
        pending.remove(i)

    for i in bodies:
        visit(i)
    for i in reversed(ordered):
        key = nodes[i - 1].get("parent_id")
        parent = bodies[by_key[key]] if key is not None else root
        body = bodies[i]
        if parent is root or parent["extra"]["metadata"]["kind"] == "agent":
            parent["start_time"] = min(parent["start_time"], body["start_time"])
            parent["end_time"] = max(parent["end_time"], body["end_time"])
    orders = {root["id"]: root["dotted_order"]}
    for i in ordered:
        body = bodies[i]
        dotted = (
            orders[body["parent_run_id"]]
            + "."
            + order(datetime.fromisoformat(body["start_time"]), body["id"])
        )
        orders[body["id"]] = body["dotted_order"] = dotted
    return [{"path": "runs", "body": bodies[i]} for i in ordered]


def build_export(
    document: dict,
    *,
    project: str,
    study: str,
    runs: Path | None = None,
    include_prompts: bool = False,
    tool_content_from: Path | None = None,
    model_content_from: Path | None = None,
) -> list[dict]:
    if not isinstance(document, dict):
        raise TypeError("expected a result document")
    project = str(uuid.UUID(project))
    if document.get("source") == "scbench":
        if (
            type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
        ):
            raise ValueError("unsupported SCBench index version")
        if runs:
            raise ValueError("--runs applies only to a native benchmark")
        source, attempts = "scbench", document["attempts"]
    else:
        if document.get("population") != "answer" or "results" not in document:
            raise ValueError(
                "expected a native answer benchmark or a v1 SCBench result index"
            )
        if model_content_from or tool_content_from:
            raise ValueError("content archive options require an SCBench index")
        source, attempts = "skill-harness", native_attempts(document, runs)

    requests: list[dict] = []
    root_ids = set()
    for attempt_index, attempt in enumerate(attempts):
        meta = attempt["metadata"]
        if meta.get("split") in {"holdout", "holdback"}:
            continue
        if source == "scbench":
            identity = {
                k: attempt["identity"][k]
                for k in ("arm", "run_id", "problem", "checkpoint")
            }
            if any(
                not isinstance(identity[k], str) or not identity[k]
                for k in ("arm", "run_id", "problem")
            ):
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
                project,
                study,
                source,
                {k: identity[k] for k in ("arm", "run_id", "problem")},
            )
        else:
            identity, details = attempt["identity"], dict(meta)
        trace_id = stable_id(project, study, source, identity)
        if trace_id in root_ids:
            raise ValueError("duplicate attempt identity in export")
        root_ids.add(trace_id)
        details.update({"source": source, "study": study, **identity})
        events = attempt["events"]
        if tool_content_from:
            from scbench_results import attach_tool_content

            attempt = {
                **attempt,
                "events": [dict(e) for e in events],
                "spans": [dict(e) for e in attempt.get("spans", [])],
            }
            attach_tool_content(attempt, tool_content_from)
            events = attempt["events"]
        nodes = [*events, *attempt.get("spans", [])]
        fallback = datetime.fromtimestamp(document.get("generated_at", 0), timezone.utc)
        observed = [
            moment(e["timestamp"], fallback) for e in nodes if e.get("timestamp")
        ]
        started, ended = meta.get("started"), meta.get("ended")
        if (
            observed
            and started
            and datetime.fromisoformat(started.replace("Z", "+00:00")).tzinfo is None
        ):
            started = ended = None
        start = min([moment(started, min(observed, default=fallback)), *observed])
        details["time_basis"] = "recorded" if started or observed else "report_time"
        details["observed_model_calls"] = sum(e["kind"] == "llm" for e in nodes)
        if source == "scbench":
            details["model_evidence"] = (
                "provider_requests"
                if details["observed_model_calls"]
                else "no_observed_requests"
            )
        details["captured_request_bodies"] = sum(
            e.get("request_body_available") is True for e in nodes
        )

        root_order = order(start, trace_id)
        common = {"session_id": project, "trace_id": trace_id}
        root = {
            **common,
            "id": trace_id,
            "name": source,
            "run_type": "chain",
            "start_time": start.isoformat(),
            "end_time": moment(ended, start).isoformat(),
            "dotted_order": root_order,
            "inputs": identity,
            "outputs": {},
            "extra": {"metadata": details},
        }
        requests.append({"path": "runs", "body": root})
        by_key = {}
        bodies = {}
        for i, event in enumerate(nodes, 1):
            key = event.get("id")
            if key is not None:
                if not isinstance(key, str) or not key or key in by_key:
                    raise ValueError("duplicate or invalid trace observation ID")
                by_key[key] = i
            uid = (
                stable_id(trace_id, "observation", key)
                if key
                else stable_id(trace_id, "event", i)
            )
            timestamp = moment(
                event.get("timestamp"), start + timedelta(microseconds=i)
            )
            finish = max(timestamp, moment(event.get("end_time"), timestamp))
            info = identifiers(
                event,
                (
                    "kind",
                    "role",
                    "synthetic",
                    "actor",
                    "author",
                    "skill_name",
                    "evidence",
                    "digest",
                    "exit_code",
                    "is_error",
                    "effort",
                    "input_digest",
                    "output_digest",
                    "result_observed",
                    "content_complete",
                    "model_time_basis",
                    "ancestry",
                    "request_body_available",
                    "response_body_available",
                    "skill_evidence",
                    "request_body_digest",
                    "response_body_digest",
                ),
            )
            if isinstance(sender := event.get("sender_thread"), str) and re.fullmatch(
                r"[a-f0-9]{64}", sender
            ):
                info["sender_thread"] = sender
            if isinstance(receivers := event.get("receiver_threads"), list):
                info["receiver_threads"] = [
                    v
                    for v in receivers
                    if isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)
                ]
            if (
                source == "skill-harness"
                and isinstance(result_ref := event.get("result_source"), str)
                and re.fullmatch(r"trace\.jsonl:[1-9][0-9]*", result_ref)
            ):
                info["raw_result_ref"] = result_ref
            if isinstance(names := event.get("skill_names"), list):
                info["skill_names"] = [
                    name
                    for name in names
                    if isinstance(name, str) and re.fullmatch(r"[\w.:-]{1,160}", name)
                ]
            info["time_basis"] = (
                "recorded" if event.get("timestamp") else "attempt_fallback"
            )
            ref = event.get("source")
            if (
                source == "skill-harness"
                and isinstance(ref, str)
                and re.fullmatch(
                    r"(?:trace\.jsonl|codex-rollouts/rollout-[\w-]+\.jsonl):[1-9][0-9]*",
                    ref,
                )
            ):
                info["raw_ref"] = ref
            if "status" in event:
                info["status"] = parse_event_state(
                    event["status"], status_present=True
                ).state.value
            if event.get("state_source") in tuple(
                state.value for state in EventStateSource
            ):
                info["state_source"] = event["state_source"]
            if source == "scbench":
                collection, offset = (
                    ("events", i - 1)
                    if i <= len(events)
                    else ("spans", i - len(events) - 1)
                )
                info["index_ref"] = f"#/attempts/{attempt_index}/{collection}/{offset}"
                if event.get("spec_verdict") in ("pass", "fail"):
                    info.update(
                        {
                            "spec_verdict": event["spec_verdict"],
                            "verdict_basis": "reviewer_report",
                        }
                    )
            if i <= len(events):
                info["event_number"] = i
            kind = info["kind"]
            name = event.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[\w. :-]{1,120}", name):
                name = kind
            run_type = {
                "llm": "llm",
                "prompt": "prompt",
                "tool": "tool",
                "tool_call": "tool",
                "command": "tool",
            }.get(kind, "chain")
            inputs = {}
            outputs = {}
            if (
                include_prompts
                and kind == "prompt"
                and event.get("role") in ("user", "agent_brief")
                and not event.get("synthetic")
                and isinstance(event.get("prompt_text"), str)
            ):
                inputs["messages"] = [{"role": "user", "content": event["prompt_text"]}]
            if tool_content_from and run_type == "tool":
                if "tool_input" in event:
                    inputs["arguments"] = event["tool_input"]
                if "tool_output" in event:
                    outputs["result"] = event["tool_output"]
            if kind == "llm":
                if model_content_from:
                    inputs, outputs = model_content(event, model_content_from, identity)
                for field, target in (
                    ("model", "ls_model_name"),
                    ("provider", "ls_provider"),
                ):
                    value = event.get(field)
                    if isinstance(value, str) and re.fullmatch(
                        r"[\w./:-]{1,160}", value
                    ):
                        info[target] = value
                if usage := token_usage(event.get("usage")):
                    outputs["usage_metadata"] = usage
            body = {
                **common,
                "id": uid,
                "name": name,
                "run_type": run_type,
                "start_time": timestamp.isoformat(),
                "end_time": finish.isoformat(),
                "inputs": inputs,
                "outputs": outputs,
                "extra": {"metadata": info},
            }
            if info.get("status") == "failed" or info.get("is_error") is True:
                body["error"] = "Recorded operation failed"
            bodies[i] = body
        requests.extend(nested_run_requests(root, nodes, bodies, by_key))
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
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("LANGSMITH_ENDPOINT must be an HTTPS API base URL")
    host = url.hostname.rstrip(".")
    if host == "api.smith.langchain.com" or host.endswith(".api.smith.langchain.com"):
        now = datetime.now(timezone.utc)
        for request in requests:
            if request["path"] == "runs":
                run = request["body"]
                if abs(datetime.fromisoformat(run["start_time"]) - now) > timedelta(
                    hours=24
                ):
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
                    "PATCH",
                    f"{path}/{body['id']}",
                    body=json.dumps(update).encode(),
                    headers=headers,
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
            document,
            project=args.project,
            study=args.study,
            runs=Path(args.runs) if args.runs else None,
            include_prompts=args.include_prompts,
            tool_content_from=args.tool_content_from,
            model_content_from=args.model_content_from,
        )
        if args.upload:
            key = os.environ.get("LANGSMITH_API_KEY")
            if not key:
                raise ValueError("set LANGSMITH_API_KEY to upload")
            upload(
                requests,
                os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
                key,
            )
            print(f"Uploaded {len(requests)} requests", file=sys.stderr)
        else:
            print(json.dumps(requests, ensure_ascii=False, indent=2, allow_nan=False))
    except (KeyError, TypeError, ValueError, OSError, http.client.HTTPException) as exc:
        die(str(exc))
    return 0
