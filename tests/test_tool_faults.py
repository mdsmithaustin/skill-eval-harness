import json
import tempfile
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli


class ToolFaultPreparationTests(unittest.TestCase):
    def test_prepare_rejects_boolean_fault_budget_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root, cases=[{
                "id": "c", "split": "tune", "prompt": "Do the task.",
                "assertions": [{"name": "answer", "type": "contains", "value": "alpha"}],
                "tool_faults": [{"tool": "search", "output": "denied", "times": True}],
            }])
            output = root / "tasks.jsonl"
            code, stdout, stderr = run_cli("prepare", str(manifest), "--out", str(output))
            self.assertEqual(code, 2)
            self.assertIn("tool_faults", stderr)
            self.assertFalse(output.exists())

    def test_prepare_carries_same_ordered_declarations_across_arms_and_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root, cases=[{
                "id": "c", "split": "tune", "prompt": "Do the task.",
                "assertions": [{"name": "answer", "type": "contains", "value": "alpha"}],
                "tool_faults": [{"tool": "search", "output": {"text": "denied"},
                                 "is_error": True, "times": 1}],
            }])
            code, stdout, stderr = run_cli("prepare", str(manifest), "--runs-per-variant", "2")
            self.assertEqual(code, 0, stderr)
            rows = [json.loads(line) for line in stdout.splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual([row.get("tool_faults") for row in rows], [[{
                "tool": "search", "output": {"text": "denied"},
                "is_error": True, "times": 1,
            }]] * 4)
