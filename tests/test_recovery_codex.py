import json
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import claude_stream_records, run_cli, write_with_skill_task


class RecoveryCompatibilityTests(unittest.TestCase):
    def test_public_codex_and_claude_recovery_preserve_effort_and_fresh_sessions(self):
        for agent in ("codex", "claude"):
            with self.subTest(agent=agent), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, tasks, relative = write_with_skill_task(root)
                row = json.loads(tasks.read_text())
                row["recovery"] = {
                    "checkpoint_path": "checkpoint.json", "expected_content": '{"phase":1}',
                    "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE",
                    "forbidden_path": "forbidden.txt"}
                tasks.write_text(json.dumps(row) + "\n")
                script = root / "fake_provider.py"
                script.write_text(
                    f"#!{sys.executable}\n"
                    "import json, os, sys, time\n"
                    "from pathlib import Path\n"
                    f"agent = {agent!r}\n"
                    "prompt = sys.stdin.read()\n"
                    "phase = prompt if prompt in {'RECOVER', 'REFUSE'} else 'INITIAL'\n"
                    "if agent == 'codex':\n"
                    "    assert 'model_reasoning_effort=high' in sys.argv\n"
                    "    assert '--ephemeral' in sys.argv\n"
                    "else:\n"
                    "    assert sys.argv[sys.argv.index('--effort') + 1] == 'high'\n"
                    "    assert '--no-session-persistence' in sys.argv\n"
                    "with Path('calls.jsonl').open('a') as handle:\n"
                    "    handle.write(json.dumps({'phase': phase, 'pid': os.getpid(), 'argv': sys.argv}) + '\\n')\n"
                    "if phase == 'INITIAL':\n"
                    "    print(json.dumps({'type': 'system', 'subtype': 'init', 'model': 'fake-model'}), flush=True)\n"
                    "    Path('checkpoint.json').write_bytes(b'{\"phase\":1}')\n"
                    "    time.sleep(60)\n"
                    "if phase == 'RECOVER':\n"
                    "    assert Path('checkpoint.json').read_bytes() == b'{\"phase\":1}'\n"
                    "    Path('recovered').write_bytes(b'done')\n"
                    "if phase == 'REFUSE':\n"
                    "    assert Path('recovered').read_bytes() == b'done'\n"
                    "if agent == 'codex':\n"
                    "    Path(sys.argv[sys.argv.index('--output-last-message') + 1]).write_text('done')\n"
                    "    print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 2, 'output_tokens': 3}}))\n"
                    "else:\n"
                    f"    records = {claude_stream_records(answer='done', served_model='fake-model', stop_reason='end_turn')!r}\n"
                    "    for record in records:\n"
                    "        print(json.dumps(record))\n")
                script.chmod(0o755)
                command = ("--codex-cmd", f"{sys.executable} {script}") if agent == "codex" else ("--claude-bin", str(script))
                code, _, stderr = run_cli(
                    "run-agent", "--agent", agent, "--tasks", tasks, "--runs", root / "runs",
                    *command, "--model", "fake-model", "--effort", "high", "--timeout", "4")
                self.assertEqual(code, 0, stderr)
                base = root / "runs" / relative
                record = json.loads((base / "recovery.json").read_text())
                self.assertEqual([phase["state"] for phase in record["phases"]], ["checkpoint_stop", "complete", "complete"])
                snapshot = base / "recovery/final"
                files = json.loads((snapshot / "files.json").read_text())
                calls = [json.loads(line) for line in (snapshot / files["calls.jsonl"]["blob"]).read_text().splitlines()]
                self.assertEqual([call["phase"] for call in calls], ["INITIAL", "RECOVER", "REFUSE"])
                self.assertEqual(len({call["pid"] for call in calls}), 3)
                self.assertEqual(record["requested_effort"], "high")
                self.assertFalse(Path(record["workspace"]).exists())

    def test_public_ordinary_codex_row_preserves_effort_argv_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, tasks, relative = write_with_skill_task(root)
            probe = root / "probe.json"
            script = root / "fake_codex.py"
            script.write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "prompt = sys.stdin.read()\n"
                f"Path({str(probe)!r}).write_text(json.dumps({{'argv': sys.argv, 'cwd': os.getcwd(), 'prompt': prompt}}))\n"
                "Path(sys.argv[sys.argv.index('--output-last-message') + 1]).write_text('ordinary answer')\n"
                "print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 2, 'output_tokens': 3}}))\n")
            runs = root / "runs"
            code, _, stderr = run_cli(
                "run-agent", "--agent", "codex", "--tasks", tasks, "--runs", runs,
                "--codex-cmd", f"{sys.executable} {script}", "--model", "fake-model",
                "--effort", "high", "--timeout", "4")
            self.assertEqual(code, 0, stderr)
            observed = json.loads(probe.read_text())
            self.assertIn("model_reasoning_effort=high", observed["argv"])
            self.assertEqual(observed["argv"][observed["argv"].index("--model") + 1], "fake-model")
            base = runs / relative
            metadata = json.loads((base / "metadata.json").read_text())
            self.assertEqual(metadata["returncode"], 0)
            self.assertEqual(metadata["effort"], {"requested": "high", "applied_by": "codex -c model_reasoning_effort"})
            self.assertEqual((base / "output.md").read_text().strip(), "ordinary answer")
            self.assertTrue((base / "metrics.json").is_file())
            self.assertTrue((base / "environment.json").is_file())
            self.assertTrue((base / "workspace-changes.json").is_file())
            self.assertFalse((base / "recovery.json").exists())
            self.assertFalse(Path(observed["cwd"]).exists())
