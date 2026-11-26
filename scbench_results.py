from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from claude_inspection import _digest as claude_digest
from claude_inspection import inspect_archive
from json_contracts import stream_json_loads, strict_json_loads
from scbench_codex import _digest as codex_digest
from scbench_codex import inspect_rollouts

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


def _codex_events(archive: _Archive, checkpoint: Path) -> tuple[list[dict], list[dict]]:
    paths = sorted(archive.input(checkpoint / "agent").glob("rollout-*.jsonl"))
    return inspect_rollouts(
        (path.relative_to(archive.root).as_posix(), _rows(archive.input(path), agent_stream=True))
        for path in paths
    )


def _claude_events(archive: _Archive, checkpoint: Path) -> tuple[list[dict], list[dict]]:
    return inspect_archive(archive, checkpoint, _rows)


def _tool_value(row: dict, event: dict, side: str, ref: str):
    uid, kind = event.get("id"), event.get("kind")
    if not isinstance(uid, str):
        raise TypeError("missing tool observation ID")
    if uid.startswith("claude-tool:") and kind == "tool_call":
        call_id = uid.removeprefix("claude-tool:")
        incoming = side == "input"
        if row.get("type") == ("assistant" if incoming else "user"):
            for block in row.get("message", {}).get("content", []):
                if not isinstance(block, dict):
                    continue
                if (block.get("type") == ("tool_use" if incoming else "tool_result")
                    and block.get("id" if incoming else "tool_use_id") == call_id):
                    return block["input" if incoming else "content"], claude_digest
    p = row.get("payload")
    if isinstance(p, dict):
        if kind == "command" and row.get("type") == "event_msg" and p.get("type") == "item_completed":
            item = p.get("item", {})
            if item.get("type") == "CommandExecution" and uid.endswith(f":command:{item.get('id') or ref}"):
                return item["command" if side == "input" else "aggregated_output"], codex_digest
        if kind == "tool_call" and row.get("type") == "response_item":
            key = p.get("call_id") or p.get("id") or ref
            if uid.endswith(f":tool:{key}"):
                packet = p.get("type")
                if side == "input" and packet in ("custom_tool_call", "function_call"):
                    return p["input" if packet == "custom_tool_call" else "arguments"], codex_digest
                if side == "output" and packet in ("custom_tool_call_output", "function_call_output"):
                    return p["output"], codex_digest
    raise ValueError("SCBench tool source does not match the observation ID")


def attach_tool_content(attempt: dict, run_root: Path) -> None:
    """Resolve selected tool inputs and outputs against verified archive rows."""
    archive = _Archive(run_root.resolve(), None)
    identity = attempt["identity"]
    problem, index = _identity(identity, identity["checkpoint"])
    if (identity["arm"], identity["run_id"]) != (archive.root.parent.name, archive.root.name):
        raise ValueError("tool-content archive does not match the SCBench attempt")
    agent = archive.input(Path(problem) / f"checkpoint_{index}" / "agent")
    if agent != agent.resolve():
        raise ValueError("tool-content checkpoint directory is a symlink")
    reads: dict[Path, dict[int, list]] = {}
    for event in [*attempt["events"], *attempt.get("spans", [])]:
        event.pop("tool_input", None)
        event.pop("tool_output", None)
        if event.get("kind") not in ("tool_call", "command"):
            continue
        for side in ("input", "output"):
            if not event.get(f"{side}_digest"):
                continue
            key = "source" if side == "input" or event["kind"] == "command" else "result_source"
            ref = event.get(key)
            if not isinstance(ref, str):
                raise TypeError("missing tool source")
            name, _, number = ref.rpartition(":")
            if Path(name).is_absolute() or not number.isdecimal() or int(number) < 1:
                raise ValueError("invalid tool source")
            path = archive.input(name).resolve()
            relative = path.relative_to(agent.resolve())
            if not (relative.match("*.jsonl") and (
                len(relative.parts) == 1 and (relative.name == "stdout.jsonl" or relative.match("rollout-*.jsonl"))
                or relative.parts[:2] == ("workspace", "projects")
            )):
                raise ValueError("invalid tool source file")
            reads.setdefault(path, {}).setdefault(int(number), []).append((event, side, ref))
    for path, wanted in reads.items():
        for line, row in _rows(path, agent_stream=True):
            for event, side, ref in wanted.pop(line, []):
                value, digest = _tool_value(row, event, side, ref)
                if digest(value) != event[f"{side}_digest"]:
                    raise ValueError("SCBench tool content digest mismatch")
                event[f"tool_{side}"] = value
            if not wanted:
                break
        if wanted:
            raise ValueError("SCBench tool source line missing")


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
        reader = _claude_events if metadata.get("agent_type") == "claude_code" else _codex_events
        events, spans = reader(archive, Path(problem) / checkpoint)
        attempts[key] = {
            "identity": {"arm": root.parent.name, "run_id": root.name, "problem": problem, "checkpoint": idx},
            "metadata": metadata,
            "events": events,
            "spans": spans,
            "outcomes": _scores(row, _CHECKPOINT_SCORES, "checkpoint_results", source),
        }
    if not attempts:
        raise ValueError("SCBench run has no checkpoint results")
    if study:
        _join_study(_Archive(study.resolve(), output), root, attempts)
    if review:
        _join_replay(_Archive(review.resolve(), output), archive, attempts)
    return {"schema_version": 1, "source": "scbench", "attempts": [attempts[key] for key in sorted(attempts)]}
