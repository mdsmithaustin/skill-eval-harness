from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import skill_benchmark as sb

ACTIVITY_RECORDS = {
    "command": {"type": "command", "command": "pytest -q"},
    "tool_call": {"type": "tool_call", "name": "lookup", "arguments": {"query": "needle"}},
    "file_read": {"type": "file_read", "name": "Read", "path": "/ws/notes.md"},
    "file_write": {"type": "file_write", "name": "Write", "path": "/ws/out.txt"},
    "skill_load": {"type": "file_read", "name": "Read", "path": "/ws/SKILL.md"},
}


def grade(base: Path, maximum: int) -> bool:
    verdict = sb.assertion_result(
        {"name": "calls", "type": "tool_count_le", "max": maximum},
        "done", base / "output.md", run_base=base, manifest_dir=base)
    return verdict["passed"]


def write_activity(base: Path, kind: str) -> dict:
    trace = json.dumps({**ACTIVITY_RECORDS[kind], "status": "completed"}) + "\n"
    sb.write_runner_outcome(base, sb.RunnerOutcome(
        provider="subagent", answer="done", returncode=0, trace_text=trace))
    return sb.read_metrics_base(base)


def assert_accounted(case: unittest.TestCase, base: Path, kind: str) -> None:
    metrics = write_activity(base, kind)
    events = json.loads((base / "events.json").read_text(encoding="utf-8"))["events"]
    case.assertEqual([(e["type"], e["status"]) for e in events], [(kind, "completed")])
    case.assertEqual(metrics["tool_calls"], 1)
    case.assertTrue(grade(base, 1))
    case.assertFalse(grade(base, 0))


def claude_tools(order: tuple[str, ...], *, failed: bool = False) -> list[dict]:
    tools = [
        {"type": "tool_use", "id": "read", "name": "Read", "input": {"file_path": "/ws/notes.md"}},
        {"type": "tool_use", "id": "write", "name": "Edit", "input": {"file_path": "/ws/out.txt"}},
        {"type": "tool_use", "id": "shell", "name": "Bash", "input": {"command": "pytest -q"}},
    ]
    return [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"role": "assistant", "content": tools}},
        *[{"type": "user", "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": name,
            "content": f"result-{name}", "is_error": failed and name == "shell",
        }]}} for name in order],
        {"type": "result", "subtype": "success", "is_error": False, "result": "done"},
        {"type": "system", "subtype": "task_summary"},
    ]


def assert_claude_conserved(case: unittest.TestCase, order: tuple[str, ...], *, failed: bool = False) -> None:
    doc, metrics = sb.normalize_trace_records(claude_tools(order, failed=failed), source="claude")
    activity = [event for event in doc["events"] if event["type"] in {"file_read", "file_write", "command"}]
    case.assertEqual([(e["type"], e["status"]) for e in activity[:3]],
                     [("file_read", "in_progress"), ("file_write", "in_progress"), ("command", "in_progress")])
    kinds = {"read": "file_read", "write": "file_write", "shell": "command"}
    case.assertEqual([(e["type"], e["status"], e["output_summary"],
                       e["raw_ref"]["line"], e["raw_result_ref"]["line"]) for e in activity[3:]],
                     [(kinds[name], "completed", f"result-{name}", 2, line)
                      for line, name in enumerate(order, 3)])
    case.assertEqual({key: metrics[key] for key in ("tool_calls", "file_reads", "file_writes", "commands", "errors")},
                     {"tool_calls": 3, "file_reads": 1, "file_writes": 1, "commands": 1, "errors": int(failed)})
    case.assertNotIn("trace_protocol_errors", metrics)


class TraceConservationTests(unittest.TestCase):
    def test_every_tool_activity_kind_counts_and_agrees_with_grading(self):
        for kind in ACTIVITY_RECORDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                assert_accounted(self, Path(td) / "run", kind)

    def test_an_accounting_hole_is_detected_for_each_activity_kind(self):
        for kind in ACTIVITY_RECORDS:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                with mock.patch.object(sb, "TRAJECTORY_STEP_TYPES", sb.TRAJECTORY_STEP_TYPES - {kind}), \
                     self.assertRaisesRegex(AssertionError, "0 != 1"):
                    assert_accounted(self, Path(td) / "broken", kind)
                assert_accounted(self, Path(td) / "restored", kind)

    def test_file_tool_grading_cannot_ignore_activity_that_metrics_count(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "run"
            metrics = write_activity(base, "file_write")
            self.assertEqual(metrics["tool_calls"], 1)
            self.assertFalse(grade(base, 0))
            with mock.patch.object(sb, "TRAJECTORY_STEP_TYPES", {"command", "tool_call"}):
                with self.assertRaisesRegex(AssertionError, "True is not false"):
                    self.assertFalse(grade(base, 0))
            self.assertFalse(grade(base, 0))

    def test_interleaved_claude_calls_keep_result_identity_and_count_once(self):
        for order, failed in itertools.product(itertools.permutations(("read", "write", "shell")), (False, True)):
            with self.subTest(order=order, failed=failed):
                assert_claude_conserved(self, order, failed=failed)

    def test_unfinished_claude_calls_remain_visible_without_completed_credit(self):
        for order in ((), ("read",), ("write", "shell")):
            with self.subTest(completed=order), tempfile.TemporaryDirectory() as td:
                records = claude_tools(order)
                base = Path(td) / "run"
                sb.write_runner_outcome(base, sb.RunnerOutcome(
                    provider="claude", answer="done", returncode=0,
                    trace_text="\n".join(map(json.dumps, records)) + "\n"))
                metrics = sb.read_metrics_base(base)
                events = json.loads((base / "events.json").read_text(encoding="utf-8"))["events"]
                activity = [event for event in events
                            if event["type"] in {"file_read", "file_write", "command"}]
                self.assertEqual(sum(e["status"] == "in_progress" for e in activity), 3)
                self.assertEqual(sum(e["status"] == "completed" for e in activity), len(order))
                self.assertEqual(metrics["tool_calls"], len(order))
                self.assertFalse(metrics["trace_observation_complete"])
                self.assertFalse(grade(base, 3))

    def test_dropped_claude_result_is_detected(self):
        dialect = sb.CLAUDE_TRACE_DIALECT

        def drop_completion(records, *, record_lines=None):
            return [(line, record) for line, record in dialect.flatten(records, record_lines=record_lines)
                    if not (record.get("type") == "file_write" and record.get("status") == "completed")]

        with mock.patch.dict(sb.TRACE_DIALECTS, {"claude": sb.TraceDialect(flatten=drop_completion)}):
            with self.assertRaisesRegex(AssertionError, "Lists differ"):
                assert_claude_conserved(self, ("read", "write", "shell"))
        assert_claude_conserved(self, ("read", "write", "shell"))

    def test_codex_completed_file_changes_keep_each_path_in_graded_count(self):
        records = [{"type": "item.completed", "item": {
            "id": "edit", "type": "file_change", "status": "completed",
            "changes": [{"path": "one.txt", "kind": "add"}, {"path": "two.txt", "kind": "update"}],
        }}, {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}}]
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "run"
            sb.write_runner_outcome(base, sb.RunnerOutcome(
                provider="codex", answer="done", returncode=0,
                trace_text="\n".join(map(json.dumps, records)) + "\n"))
            events = json.loads((base / "events.json").read_text(encoding="utf-8"))["events"]
            self.assertEqual([(e["input_summary"], e["raw_ref"]["line"]) for e in events if e["type"] == "file_write"],
                             [("one.txt", 1), ("two.txt", 1)])
            self.assertEqual(sb.read_metrics_base(base)["file_writes"], 2)
            self.assertTrue(grade(base, 2))
            self.assertFalse(grade(base, 1))
