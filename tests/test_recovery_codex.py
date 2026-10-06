import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import (
    claude_stream_records,
    run_cli,
    stub_claude_stream,
    write_with_skill_task,
)

import skill_benchmark as sb


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
                for phase, returncode in zip(record["phases"], (-signal.SIGTERM, 0, 0)):
                    facts = phase["adapter_environment"]["recovery_process"]
                    self.assertEqual(facts, json.loads((base / "recovery" / phase["phase"] / "process.json").read_text()))
                    self.assertEqual((facts["state"], facts["os_returncode"], facts["compatibility_returncode"]),
                                     (phase["state"], returncode, returncode))
                    self.assertEqual((facts["leader_reaped"], facts["pipes_drained"], facts["group_stopped"]),
                                     (True, True, True))
                    self.assertEqual(phase["adapter_environment"]["runner"], agent)
                self.assertFalse(Path(record["workspace"]).exists())

    def test_public_claude_capture_failures_preserve_process_and_timeout_facts(self):
        write_json = sb.write_json
        write_bytes = Path.write_bytes
        for mode, os_returncode, compatibility_returncode in (
                ("stop", -signal.SIGTERM, -signal.SIGTERM),
                ("timeout", -signal.SIGKILL, 124), ("exit-124", 124, 124)):
            for artifact in ("process.json", "stdout.bin"):
                with self.subTest(mode=mode, artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    _, tasks, relative = write_with_skill_task(root)
                    row = json.loads(tasks.read_text())
                    row["recovery"] = {
                        "checkpoint_path": "checkpoint.json", "expected_content": '{"phase":1}',
                        "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE",
                        "forbidden_path": "forbidden.txt"}
                    tasks.write_text(json.dumps(row) + "\n")
                    script = root / "fake_claude.py"
                    script.write_text(
                        f"#!{sys.executable}\n"
                        "import os, sys, time\n"
                        "from pathlib import Path\n"
                        "sys.stdin.read()\n"
                        "os.write(1, b'{\"type\":\"system\",\"subtype\":\"init\"}\\n')\n"
                        "os.write(2, b'raw-stderr\\xff')\n"
                        f"mode = {mode!r}\n"
                        "if mode == 'exit-124':\n"
                        "    raise SystemExit(124)\n"
                        "if mode == 'stop':\n"
                        "    Path('checkpoint.json').write_bytes(b'{\"phase\":1}')\n"
                        "time.sleep(60)\n")
                    script.chmod(0o755)

                    def fail_process(path, content, artifact=artifact):
                        if path.name == artifact:
                            raise OSError("capture storage failed")
                        return write_json(path, content)

                    def fail_raw(path, content, artifact=artifact):
                        if path.name == artifact:
                            raise OSError("capture storage failed")
                        return write_bytes(path, content)

                    with mock.patch.object(sb, "write_json", side_effect=fail_process), \
                            mock.patch.object(Path, "write_bytes", autospec=True, side_effect=fail_raw):
                        code, _, stderr = run_cli(
                            "run-agent", "--agent", "claude", "--tasks", tasks, "--runs", root / "runs",
                            "--claude-bin", script, "--model", "fake-model", "--timeout", "1")
                    self.assertEqual(code, 1, stderr)
                    base = root / "runs" / relative
                    record = json.loads((base / "recovery.json").read_text())
                    self.assertEqual((record["status"], record["failure"], len(record["phases"])),
                                     ("failed", "capture_failed", 1))
                    phase = record["phases"][0]
                    facts = phase["adapter_environment"]["recovery_process"]
                    self.assertEqual((phase["state"], facts["state"], facts["error"]),
                                     ("capture_failed", "capture_failed", "OSError: capture storage failed"))
                    self.assertEqual((facts["os_returncode"], facts["compatibility_returncode"],
                                      phase["compatibility_returncode"]),
                                     (os_returncode, compatibility_returncode, compatibility_returncode))
                    self.assertEqual(phase["outcome"], "TimedOut" if mode == "timeout" else "ProviderFailed")
                    self.assertEqual((facts["leader_reaped"], facts["pipes_drained"], facts["group_stopped"]),
                                     (True, True, True))
                    process = base / "recovery/initial/process.json"
                    if artifact == "process.json":
                        self.assertFalse(process.exists())
                    else:
                        self.assertEqual(json.loads(process.read_text()), facts)
                    self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(), b"raw-stderr\xff")
                    self.assertIsNone(record["certificate"])
                    self.assertFalse(Path(record["workspace"]).exists())

    def test_public_ordinary_claude_row_preserves_environment_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, tasks, relative = write_with_skill_task(root)
            script = stub_claude_stream(root / "claude", answer="ordinary answer", served_model="fake-model")
            code, _, stderr = run_cli(
                "run-agent", "--agent", "claude", "--tasks", tasks, "--runs", root / "runs",
                "--claude-bin", script, "--model", "fake-model", "--effort", "high", "--timeout", "4")
            self.assertEqual(code, 0, stderr)
            base = root / "runs" / relative
            metadata = json.loads((base / "metadata.json").read_text())
            self.assertEqual(metadata["returncode"], 0)
            self.assertEqual(metadata["effort"], {"requested": "high", "applied_by": "claude --effort"})
            self.assertEqual((base / "output.md").read_text().strip(), "ordinary answer")
            environment = json.loads((base / "environment.json").read_text())
            self.assertEqual(set(environment), {"runner", "command", "context_isolation", "cwd", "stdout_utf8_valid", "variant"})
            self.assertEqual((environment["runner"], environment["cwd"], environment["stdout_utf8_valid"]),
                             ("claude", "<isolated workspace>", True))
            self.assertIn("--effort high", environment["command"])
            self.assertTrue((base / "metrics.json").is_file())
            self.assertTrue((base / "artifact-commit.json").is_file())
            self.assertFalse((base / "recovery.json").exists())

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
