import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli


class BridgeCliBoundaryTests(unittest.TestCase):
    def test_bridge_requires_explicit_command_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root)
            tasks = root / "tasks.jsonl"
            self.assertEqual(run_cli("prepare", str(manifest), "--out", str(tasks))[0], 0)
            runs = root / "runs"
            code, _, stderr = run_cli("run-subagent", "--tasks", str(tasks), "--runs", str(runs),
                                      "--tool-bridge", "--tool-replay", "strict")
            self.assertEqual(code, 1)
            self.assertIn("tool bridge requires --agent-cmd", stderr)
            self.assertFalse(runs.exists())


@unittest.skipUnless(hasattr(os, "waitid") and hasattr(os, "WNOWAIT"),
                     "bridge requires nonreaping waitid")
class BridgeProcessTests(unittest.TestCase):
    def test_known_error_crosses_real_cli_and_saved_reader(self):
        import skill_benchmark as sb
        import tool_fault_contracts as domain
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root, cases=[{
                "id": "c", "split": "tune", "prompt": "Return alpha.",
                "assertions": [{"name": "answer", "type": "contains", "value": "alpha"}],
                "tool_faults": [{"tool": "search", "output": "alpha", "is_error": True}],
            }])
            tasks = root / "tasks.jsonl"
            self.assertEqual(run_cli("prepare", str(manifest), "--out", str(tasks))[0], 0)
            runs = root / "runs"
            fixture = Path(__file__).parent / "fixtures" / "tool_bridge_child.py"
            child = shlex.join([sys.executable, str(fixture.resolve()), "success"])
            process = subprocess.run([sys.executable, sb.__file__, "run-subagent", "--tasks", str(tasks),
                                      "--runs", str(runs), "--tool-bridge", "--agent-cmd", child,
                                      "--tool-replay", "strict", "--timeout", "2"],
                                     capture_output=True, text=True, timeout=15)
            self.assertEqual(process.returncode, 0, process.stderr)
            row = json.loads(tasks.read_text().splitlines()[0])
            base = runs / row["run_dir"]
            self.assertEqual((base / "output.md").read_text(), "alpha known True\n")
            observation = sb.read_tool_observation_base(base)
            self.assertEqual(observation.coverage, "closed")
            self.assertEqual(observation.assess(domain.Count("errors")).value, 1)
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_call", "tool": "search", "is_error": True}, base, {})[0], True)
