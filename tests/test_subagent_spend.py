import json
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import run_cli, write_with_skill_task


class SubagentSpendTests(unittest.TestCase):
    def test_paid_claude_error_preserves_cost_and_actual_zero_exit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, tasks, run_dir = write_with_skill_task(root)
            marker = root / "launches"
            envelope = {"type": "result", "is_error": True, "api_error_status": 529,
                        "result": "API Error: overloaded", "total_cost_usd": 0.06,
                        "usage": {"input_tokens": 1, "output_tokens": 0}}
            stub = root / "claude"
            stub.write_text(f'''#!{sys.executable}
import sys
from pathlib import Path
sys.stdin.read()
Path({str(marker)!r}).write_text("started\\n")
sys.stdout.write({json.dumps(envelope)!r})
''')
            stub.chmod(0o755)
            code, _, stderr = run_cli("run-subagent", "--tasks", tasks,
                                      "--runs", root / "runs", "--claude-bin", stub)
            base = root / "runs" / run_dir
            metadata = json.loads((base / "metadata.json").read_text())
            self.assertEqual(code, 0, stderr)
            self.assertEqual(marker.read_text(), "started\n")
            self.assertEqual(metadata["cost_normalized"].get("total_cost"), 0.06)
            self.assertEqual(metadata["returncode"], 0)
            self.assertFalse(metadata["provider_response_complete"])
            self.assertIn("Claude provider error", (base / "output.md").read_text())
