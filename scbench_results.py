from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from json_contracts import stream_json_loads, strict_json_loads

_CHECKPOINT_SCORES = (
    "strict_pass_rate", "core_pass_rate", "isolated_pass_rate", "total_tests",
    "passed_tests", "regression_total", "regression_passed", "erosion", "verbosity",
)
_STUDY_SCORES = (
    "pass_rate", "strict_pass_rate", "solved", "n_excluded", "pass_rate_excl",
    "strict_pass_rate_excl", "solved_excl", "erosion_rescored", "verbosity_rescored",
)


@dataclass(frozen=True)
class _Archive:
    root: Path
    output: Path | None

    def input(self, relative: Path | str) -> Path:
        path = self.root / relative
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"input escapes the evidence directory: {path}")
        if self.output is not None and (
            self.output == resolved or self.output.exists() and path.exists() and path.samefile(self.output)
        ):
            raise ValueError(f"output aliases an SCBench input: {path}")
        return path


def _rows(path: Path, *, agent_stream: bool = False) -> Iterator[tuple[int, dict[str, Any]]]:
    loads = stream_json_loads if agent_stream else strict_json_loads
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = loads(line)
                    if not isinstance(value, dict):
                        raise TypeError("expected an object")
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                yield line_number, value


def _identity(row: dict, checkpoint: Any) -> tuple[str, int]:
    problem = row.get("problem")
    if not isinstance(problem, str) or problem in {"", ".", ".."} or Path(problem).name != problem:
        raise ValueError(f"invalid SCBench problem name {problem!r}")
    if type(checkpoint) is not int or checkpoint < 1:
        raise ValueError(f"invalid checkpoint number {checkpoint!r} for {problem}")
    return problem, checkpoint


def _scores(row: dict, keys: tuple[str, ...], basis: str, source: str) -> list[dict]:
    result = []
    for key in keys:
        value = row.get(key)
        if value is not None:
            if not isinstance(value, (int, float, bool)):
                raise ValueError(f"{source}: expected numeric or boolean {key}")
            result.append({"key": key, "value": value, "basis": basis, "source": source})
    return result


def _codex_events(archive: _Archive, checkpoint: Path) -> list[dict]:
    events = []
    for path in sorted(archive.input(checkpoint / "agent").glob("rollout-*.jsonl")):
        path = archive.input(path)
        actor = "unknown"
        for number, row in _rows(path, agent_stream=True):
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            if row.get("type") == "session_meta":
                actor = payload.get("agent_role") or ("lead" if isinstance(payload.get("source"), str) else "subagent")
            if row.get("type") != "response_item":
                continue
            kind = payload.get("type")
            if kind in {"function_call", "custom_tool_call"}:
                name = payload.get("name")
                if name == "exec" and "tools.apply_patch(" in str(payload.get("input", "")):
                    name = "exec_with_apply_patch"
                event = {"kind": "tool_call", "name": name}
            elif kind == "agent_message" and payload.get("recipient") == "/root":
                event = {"kind": "agent_message", "name": "message_to_lead", "author": payload.get("author")}
                event["text"] = "\n".join(
                    block["text"] for block in payload.get("content", [])
                    if isinstance(block, dict) and isinstance(block.get("text"), str)
                )
                verdicts = {
                    match.group(1).lower() for match in re.finditer(
                        r"(?im)^(?:- )?(?:\*\*)?(?:spec verdict: |specification )(pass|fail)(?:ed)?(?:[.!](?:\*\*)?(?=\s|$)|\*\*(?=\s|$)|$)",
                        event["text"],
                    )
                }
                if len(verdicts) == 1:
                    event["spec_verdict"] = verdicts.pop()
            else:
                continue
            events.append({**event, "actor": actor, "timestamp": row.get("timestamp"),
                           "source": f"{path.relative_to(archive.root)}:{number}"})
    return sorted(events, key=lambda item: item["timestamp"] or "")


def _claude_events(archive: _Archive, checkpoint: Path) -> list[dict]:
    path = archive.input(checkpoint / "agent" / "stdout.jsonl")
    if not path.is_file():
        return []
    events = []
    for number, row in _rows(path, agent_stream=True):
        if row.get("type") != "assistant":
            continue
        for block in row.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                events.append({"kind": "tool_call", "name": block["name"],
                               "actor": "subagent" if row.get("parent_tool_use_id") else "lead",
                               "timestamp": row.get("timestamp"), "source": f"{path.relative_to(archive.root)}:{number}"})
    return events


def _join_study(study: _Archive, root: Path, attempts: dict) -> None:
    seen = set()
    for number, row in _rows(study.input("checkpoints.jsonl")):
        if (row.get("arm"), row.get("run_id")) != (root.parent.name, root.name):
            continue
        key = _identity(row, row.get("checkpoint"))
        if key in seen or key not in attempts:
            raise ValueError(f"ambiguous or unknown study checkpoint {key!r}")
        seen.add(key)
        attempt = attempts[key]
        attempt["metadata"]["study"] = study.root.name
        attempt["outcomes"].extend(_scores(row, _STUDY_SCORES, "study_analysis",
                                            f"study/checkpoints.jsonl:{number}"))
    if not seen:
        raise ValueError(f"no study rows match {root.parent.name}/{root.name}")


def _join_replay(directory: _Archive, archive: _Archive, attempts: dict) -> None:
    replay_file = directory.input("review-value-replay.json")
    confirmations_file = directory.input("review-value-confirmations.json")
    bound = set()
    for index, row in enumerate(_object_array(replay_file)):
        key = _identity(row, row.get("checkpoint"))
        if key not in attempts:
            continue
        rollout = row.get("lead_rollout")
        if not isinstance(rollout, str) or Path(rollout).name != rollout:
            raise ValueError(f"invalid lead_rollout at {replay_file}:{index}")
        agent_dir = Path(key[0]) / f"checkpoint_{key[1]}" / "agent"
        if not archive.input(agent_dir / rollout).is_file():
            continue
        if key in bound:
            raise ValueError(f"ambiguous replay checkpoint {key!r}")
        bound.add(key)
        attempts[key]["outcomes"].extend(_scores(
            row, ("end_state_equals_snapshot", "edits_after_first_reviewer_message"),
            "snapshot_replay", f"review/{replay_file.name}#/{index}"))
    if not bound:
        raise ValueError("no replay rows match archived rollouts")
    seen = set()
    for index, row in enumerate(_object_array(confirmations_file)):
        key = _identity(row, row.get("checkpoint"))
        if key not in bound or row.get("ran") is not True:
            continue
        if key in seen:
            raise ValueError(f"ambiguous replay confirmation {key!r}")
        seen.add(key)
        source = f"review/{confirmations_file.name}#/{index}"
        results = _scores(row, ("pre_review_passed", "final_passed", "tests"),
                          "hidden_test_replay", source)
        for change in ("gained", "lost"):
            for test in row.get(change, []):
                if not isinstance(test, dict) or not isinstance(test.get("test"), str):
                    raise TypeError(f"{source}: invalid {change} test")
                results.append({"key": f"test_{change}", "value": test["test"],
                                "basis": "hidden_test_replay", "source": source})
        attempts[key]["outcomes"].extend(results)


def _object_array(path: Path) -> list[dict]:
    rows = strict_json_loads(path.read_bytes())
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected an array of objects")
    return rows


def _settings(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a mapping")
    return value


def import_run(root: Path, study: Path | None = None, review: Path | None = None,
               skill_revision: str | None = None, *, output: Path | None = None) -> dict:
    root = root.resolve()
    output = output.resolve() if output is not None else None
    archive = _Archive(root, output)
    attempts = {}
    problem_info = {}
    config = _settings(archive.input("config.yaml"))
    for number, row in _rows(archive.input("checkpoint_results.jsonl")):
        problem, idx = key = _identity(row, row.get("idx"))
        checkpoint = f"checkpoint_{idx}"
        if row.get("checkpoint") != checkpoint or key in attempts:
            raise ValueError(f"inconsistent or duplicate checkpoint {key!r}")
        if problem not in problem_info:
            info = _settings(archive.input(Path(problem) / "run_info.yaml"))
            problem_info[problem] = {k: info[k] for k in ("agent_type", "agent_version", "model_name", "seed") if k in info}
        source = f"checkpoint_results.jsonl:{number}"
        metadata = {**problem_info[problem], "comparison": "arm", "source": source}
        if isinstance(config.get("thinking"), str):
            metadata["effort"] = config["thinking"]
        if skill_revision:
            metadata["skill_revision"] = skill_revision
            metadata["skill_revision_source"] = "operator"
        metadata.update({k: row[k] for k in (
            "state", "started", "ended", "cost", "steps", "input", "output", "cache_read",
            "cache_write", "reasoning", "duration",
        ) if row.get(k) is not None})
        if "cost" in metadata:
            metadata["cost_basis"] = "as_recorded_by_scbench"
        attempts[key] = {
            "identity": {"arm": root.parent.name, "run_id": root.name, "problem": problem, "checkpoint": idx},
            "metadata": metadata,
            "events": (
                _claude_events(archive, Path(problem) / checkpoint)
                if metadata.get("agent_type") == "claude_code"
                else _codex_events(archive, Path(problem) / checkpoint)
            ),
            "outcomes": _scores(row, _CHECKPOINT_SCORES, "checkpoint_results", source),
        }
    if not attempts:
        raise ValueError("SCBench run has no checkpoint results")
    if study:
        _join_study(_Archive(study.resolve(), output), root, attempts)
    if review:
        _join_replay(_Archive(review.resolve(), output), archive, attempts)
    return {"schema_version": 1, "source": "scbench", "attempts": [attempts[key] for key in sorted(attempts)]}
