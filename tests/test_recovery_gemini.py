import hashlib
import json
import signal
import tempfile
import unittest
from pathlib import Path

from helpers import run_cli, write_with_skill_task
from test_gemini_backend import _success_stream, _write_executable


class GeminiRecoveryTests(unittest.TestCase):
    def test_public_gemini_recovery_preserves_fixture_and_fresh_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.bin"
            fixture.write_bytes(b"fixture\x00\xff\r\n")
            _, tasks, _ = write_with_skill_task(root)
            row = json.loads(tasks.read_text())
            row.update(variant="without_skill", skill_paths=[], skill_root_keys=[],
                       input_files=[str(fixture)], run_dir="case-1/without_skill",
                       prompt="INITIAL", instruction="")
            row.pop("skill_tree_hash", None)
            row["recovery"] = {
                "checkpoint_path": "checkpoint.json", "expected_content": '{"phase":1}',
                "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE",
                "forbidden_path": "forbidden.txt"}
            tasks.write_text(json.dumps(row) + "\n")
            probes = root / "probes.jsonl"
            script = root / "fake_gemini.py"
            _write_executable(script,
                "import json, os, sys, time\n"
                "from pathlib import Path\n"
                "version = '--version' in sys.argv\n"
                f"with Path({str(probes)!r}).open('a') as handle:\n"
                "    handle.write(json.dumps({'version': version, 'pid': os.getpid(), 'cwd': os.getcwd()}) + '\\n')\n"
                "if version:\n"
                "    print('0.55.0-recovery-fake')\n"
                "    raise SystemExit(0)\n"
                "prompt = next(argument.split('=', 1)[1] for argument in sys.argv if argument.startswith('--prompt='))\n"
                "assert sys.argv[sys.argv.index('--output-format') + 1] == 'stream-json'\n"
                "assert '--model=gemini-test' in sys.argv\n"
                "assert '--skip-trust' in sys.argv and '--sandbox' in sys.argv\n"
                "assert Path(sys.argv[sys.argv.index('--policy') + 1]).is_file()\n"
                "home = Path(os.environ['GEMINI_CLI_HOME']).resolve()\n"
                "assert home.is_dir() and not home.is_relative_to(Path.cwd().resolve())\n"
                "assert Path('inputs/fixture.bin').read_bytes() == b'fixture\\x00\\xff\\r\\n'\n"
                "phase = prompt if prompt in {'RECOVER', 'REFUSE'} else 'INITIAL'\n"
                "with Path('calls.jsonl').open('a') as handle:\n"
                "    handle.write(json.dumps({'phase': phase, 'pid': os.getpid(), 'cwd': os.getcwd(), 'argv': sys.argv}) + '\\n')\n"
                f"records = [json.loads(line) for line in {_success_stream()!r}.splitlines()]\n"
                "records[0]['session_id'] = 'fake-session-' + str(os.getpid())\n"
                "records[1]['content'] = 'answer ' + phase\n"
                "if phase == 'REFUSE':\n"
                "    records[1]['content'] = 'fake attempted_write forbidden.txt; fake denial'\n"
                "for record in records[:2]:\n"
                "    print(json.dumps(record), flush=True)\n"
                "if phase == 'INITIAL':\n"
                "    Path('checkpoint.json').write_bytes(b'{\"phase\":1}')\n"
                "    time.sleep(60)\n"
                "if phase == 'RECOVER':\n"
                "    assert Path('checkpoint.json').read_bytes() == b'{\"phase\":1}'\n"
                "    Path('recovered.bin').write_bytes(b'recovered\\x00\\xff')\n"
                "if phase == 'REFUSE':\n"
                "    assert Path('recovered.bin').read_bytes() == b'recovered\\x00\\xff'\n"
                "print(json.dumps(records[2]), flush=True)\n")
            code, _, stderr = run_cli(
                "run-agent", "--agent", "gemini", "--tasks", tasks, "--runs", root / "runs",
                "--gemini-cmd", script, "--model", "gemini-test", "--timeout", "4")
            base = root / "runs" / row["run_dir"]
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(code, 0, f"{stderr}\n{record}")
            self.assertEqual((record["status"], record["failure"]), ("complete", None))
            self.assertEqual([phase["phase"] for phase in record["phases"]],
                             ["initial", "recovery", "refusal"])
            self.assertEqual([phase["state"] for phase in record["phases"]],
                             ["checkpoint_stop", "complete", "complete"])
            evidence = base / "recovery"
            files = json.loads((evidence / "final/files.json").read_text())
            calls = [json.loads(line) for line in
                     (evidence / "final" / files["calls.jsonl"]["blob"]).read_text().splitlines()]
            self.assertEqual([call["phase"] for call in calls], ["INITIAL", "RECOVER", "REFUSE"])
            self.assertEqual(len({call["pid"] for call in calls}), 3)
            self.assertEqual({Path(call["cwd"]).resolve() for call in calls},
                             {Path(record["workspace"]).resolve()})
            observed_probes = [json.loads(line) for line in probes.read_text().splitlines()]
            self.assertEqual([probe["version"] for probe in observed_probes],
                             [True, False, True, False, True, False])
            self.assertEqual([probe["pid"] for probe in observed_probes if not probe["version"]],
                             [call["pid"] for call in calls])
            sessions = []
            for phase, call, os_returncode in zip(record["phases"], calls, [-signal.SIGTERM, 0, 0]):
                directory = evidence / phase["phase"]
                facts = json.loads((directory / "process.json").read_text())
                self.assertEqual((facts["pid"], facts["process_group"], facts["os_returncode"]),
                                 (call["pid"], call["pid"], os_returncode))
                self.assertEqual(facts["state"], phase["state"])
                self.assertEqual(facts["compatibility_returncode"], os_returncode)
                self.assertEqual(phase["compatibility_returncode"], os_returncode)
                self.assertEqual((facts["leader_reaped"], facts["pipes_drained"], facts["group_stopped"]),
                                 (True, True, True))
                self.assertIsNone(facts["error"])
                self.assertEqual(facts["signal_sent"], signal.SIGTERM if phase["phase"] == "initial" else None)
                self.assertEqual(facts["checkpoint_observed_live"], phase["phase"] == "initial")
                environment = phase["adapter_environment"]
                self.assertEqual(environment["recovery_process"], facts)
                self.assertEqual(environment["gemini_cli_version"], "0.55.0-recovery-fake")
                self.assertEqual(environment["gemini_cli_version_status"], "reported")
                self.assertTrue(environment["config_isolated"])
                self.assertTrue(environment["gemini_home_outside_workdir"])
                invocation = json.loads((directory / "invocation.json").read_text())
                self.assertEqual(invocation["client_argv"], call["argv"])
                raw = (directory / "stdout.bin").read_bytes()
                self.assertEqual(facts["raw_sha256"]["stdout.bin"], hashlib.sha256(raw).hexdigest())
                self.assertEqual(facts["raw_sha256"]["stderr.bin"], hashlib.sha256(b"").hexdigest())
                self.assertEqual((directory / "stderr.bin").read_bytes(), b"")
                sessions.append(json.loads(raw.splitlines()[0])["session_id"])
                for snapshot in ("before", "after"):
                    manifest = json.loads((directory / snapshot / "files.json").read_text())
                    self.assertEqual((directory / snapshot / manifest["inputs/fixture.bin"]["blob"]).read_bytes(),
                                     b"fixture\x00\xff\r\n")
            self.assertEqual(len(set(sessions)), 3)
            self.assertEqual((evidence / "initial/checkpoint-observed.bin").read_bytes(), b'{"phase":1}')
            self.assertIn(b"fake attempted_write forbidden.txt; fake denial", (evidence / "refusal/stdout.bin").read_bytes())
            self.assertFalse(record["phases"][-1]["forbidden_path_present"])
            self.assertEqual((record["runtime_model"], record["runtime_effort"],
                              record["enforcing_denial"], record["certificate"]), (None, None, None, None))
            self.assertFalse(Path(record["workspace"]).exists())
