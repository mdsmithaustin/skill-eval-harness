import json
import os
import signal
import sys
import tempfile
import unittest
from contextlib import ExitStack
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
                    "if agent == 'claude':\n"
                    "    for kind in ('initialize', 'get_settings'):\n"
                    "        request = json.loads(sys.stdin.readline())\n"
                    "        assert request['request']['subtype'] == kind\n"
                    "        applied = {'model': 'fake-model', 'effort': None}\n"
                    "        print(json.dumps({'type':'control_response','response':{'subtype':'success','request_id':request['request_id'],'response':{'applied':applied,'settings':{'secret':'CANARY'}}}}), flush=True)\n"
                    "    user = json.loads(sys.stdin.readline())\n"
                    "    prompt = user['message']['content']\n"
                    "    assert sys.stdin.read() == ''\n"
                    "else:\n"
                    "    prompt = sys.stdin.read()\n"
                    "phase = prompt if prompt in {'RECOVER', 'REFUSE'} else 'INITIAL'\n"
                    "if agent == 'codex': print(json.dumps({'type':'thread.started','thread_id':str(os.getpid())}), flush=True)\n"
                    "if agent == 'codex':\n"
                    "    assert 'model_reasoning_effort=high' in sys.argv\n"
                    "    assert '--ephemeral' not in sys.argv\n"
                    "else:\n"
                    "    assert sys.argv[sys.argv.index('--effort') + 1] == 'high'\n"
                    "    assert '--no-session-persistence' in sys.argv\n"
                    "with Path('calls.jsonl').open('a') as handle:\n"
                    "    handle.write(json.dumps({'phase': phase, 'pid': os.getpid(), 'argv': sys.argv}) + '\\n')\n"
                    "if phase == 'INITIAL':\n"
                    "    print(json.dumps({'type': 'system', 'subtype': 'init', 'model': 'fake-model', 'session_id': str(os.getpid())} if agent == 'claude' else {'type':'turn.started'}), flush=True)\n"
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
                    "        record['session_id'] = str(os.getpid())\n"
                    "        if record['type'] == 'result': record['is_error'] = False\n"
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
        atomic_write = sb._atomic_write_text
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
                        "import json, os, signal, sys, time\n"
                        "from pathlib import Path\n"
                        "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
                        "for kind in ('initialize', 'get_settings'):\n"
                        "    request = json.loads(sys.stdin.readline())\n"
                        "    print(json.dumps({'type':'control_response','response':{'subtype':'success','request_id':request['request_id'],'response':{}}}), flush=True)\n"
                        "json.loads(sys.stdin.readline())\n"
                        "assert sys.stdin.read() == ''\n"
                        "print(json.dumps({'type':'system','subtype':'init','session_id':str(os.getpid())}), flush=True)\n"
                        "os.write(2, b'raw-stderr\\xff')\n"
                        f"mode = {mode!r}\n"
                        "if mode == 'exit-124':\n"
                        "    raise SystemExit(124)\n"
                        "if mode == 'stop':\n"
                        "    Path('checkpoint.pending').write_bytes(b'{\"phase\":1}')\n"
                        "    Path('checkpoint.pending').rename('checkpoint.json')\n"
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

                    def fail_publication(path, content, artifact=artifact):
                        if path.name == artifact:
                            raise OSError("capture storage failed")
                        return atomic_write(path, content)

                    with mock.patch.object(sb, "write_json", side_effect=fail_process), \
                            mock.patch.object(sb, "_atomic_write_text", side_effect=fail_publication), \
                            mock.patch.object(Path, "write_bytes", autospec=True, side_effect=fail_raw):
                        code, _, stderr = run_cli(
                            "run-agent", "--agent", "claude", "--tasks", tasks, "--runs", root / "runs",
                            "--claude-bin", script, "--model", "fake-model",
                            "--timeout", "1" if mode == "timeout" else "4")
                    self.assertEqual(code, 1, stderr)
                    base = root / "runs" / relative
                    record = json.loads((base / "recovery.json").read_text())
                    self.assertEqual((record["status"], record["failure"], len(record["phases"])),
                                     ("failed", "capture_failed", 1))
                    if artifact == "process.json":
                        self.assertEqual(record["phases"][0]["state"], "capture_failed")
                        self.assertFalse((base / "recovery/initial/process.json").exists())
                        self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(), b"")
                        continue
                    phase = record["phases"][0]
                    facts = phase["adapter_environment"]["recovery_process"]
                    self.assertEqual((phase["state"], facts["state"], facts["error"]),
                                     ("capture_failed", "capture_failed", "native_artifact_failed"))
                    self.assertEqual((facts["os_returncode"], facts["compatibility_returncode"],
                                      phase["compatibility_returncode"]),
                                     (os_returncode, compatibility_returncode, compatibility_returncode), facts)
                    self.assertEqual(phase["outcome"], "TimedOut" if mode == "timeout" else "ProviderFailed")
                    self.assertEqual((facts["leader_reaped"], facts["pipes_drained"], facts["group_stopped"]),
                                     (True, True, True))
                    process = base / "recovery/initial/process.json"
                    if artifact == "process.json":
                        self.assertFalse(process.exists())
                    else:
                        self.assertEqual(json.loads(process.read_text()), facts)
                    self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(), b"")
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


class NativeRecoveryStopTests(unittest.TestCase):
    def run_native_case(self, root, mode):
        _, tasks, relative = write_with_skill_task(root)
        row = json.loads(tasks.read_text())
        row["recovery"] = {
            "checkpoint_path": "checkpoint.json", "expected_content": '{"phase":1}',
            "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE",
            "forbidden_path": "forbidden.txt", "match": "json"}
        tasks.write_text(json.dumps(row) + "\n")
        script = root / "native.py"
        script.write_text(f'''import json, os, signal, sys, time
from pathlib import Path
mode = {mode!r}
prompt = sys.stdin.read()
thread = str(os.getpid())
print(json.dumps({{"type":"thread.started","thread_id":thread}}), flush=True)
home = Path(os.environ["CODEX_HOME"])
sessions = home / "sessions"
sessions.mkdir(parents=True, exist_ok=True)
meta = {{"id":thread,"session_id":"root-"+thread,"cwd":os.getcwd(),"cli_version":"0.160.1","base_instructions":"ROLLOUT_CANARY"}}
context = {{"model":"context-model","effort":"high","cwd":os.getcwd(),"workspace_roots":[os.getcwd()],
           "approval_policy":"never","sandbox_policy":{{"type":"read-only","network_access":False}},
           "permission_profile":{{"secret":"ROLLOUT_CANARY"}},"developer_instructions":"ROLLOUT_CANARY"}}
rollout = sessions / ("rollout-date-"+thread+".jsonl")
if mode == "metadata-mismatch": meta["id"] = "other"
if mode == "metadata-version": meta["cli_version"] = "newer"
records = [{{"timestamp":"2026-10-06","type":"session_meta","payload":meta}},
           {{"timestamp":"2026-10-06","ordinal":1,"type":"turn_context","payload":context}},
           {{"timestamp":"2026-10-06","ordinal":2,"type":"turn_context","payload":dict(context,turn_id="turn-1",effort=None)}},
           {{"timestamp":"2026-10-06","type":"response_item","payload":{{"secret":"ROLLOUT_CANARY"}}}}]
wire = "".join(json.dumps(record)+"\\n" for record in records)
if mode != "metadata-missing": rollout.write_text(wire)
if mode == "metadata-ambiguous": (sessions / ("other-"+thread+".jsonl")).write_text(wire)
if mode == "metadata-unflushed": rollout.write_text(wire.rstrip())
if mode == "metadata-symlink":
    rollout.unlink()
    outside = Path(__file__).parent / "outside-rollout"
    outside.write_text(wire)
    rollout.symlink_to(outside)
if prompt in {{"RECOVER", "REFUSE"}}:
    Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text("done")
    print(json.dumps({{"type":"turn.completed"}}), flush=True)
    raise SystemExit(0)
def handle(number, frame):
    if mode == "changed":
        Path("checkpoint.json").write_bytes(b"changed")
    if mode == "equivalent":
        Path("checkpoint.json").write_bytes(b'{{ "phase":1.0 }}')
    if mode == "removed":
        Path("checkpoint.json").unlink()
    if mode == "handler-complete":
        print(json.dumps({{"type":"turn.completed"}}), flush=True)
    if mode == "failure":
        print(json.dumps({{"type":"turn.failed"}}), flush=True)
    if mode == "malformed-after-complete":
        print(json.dumps({{"type":"turn.completed"}}), flush=True)
        print("{{bad", flush=True)
    print(json.dumps({{"type":"item.completed","item":{{"type":"agent_message","text":"handler-ready"}}}}),
          end="" if mode == "unterminated" else "\\n", flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, handle)
print(json.dumps({{"type":"turn.started"}}), flush=True)
if mode == "before-complete":
    print(json.dumps({{"type":"turn.completed"}}), flush=True)
if mode == "escaped-pipe":
    child = os.fork()
    if child == 0:
        os.setsid()
        (Path(__file__).parent / "escaped.pid").write_text(str(os.getpid()))
        time.sleep(60)
        os._exit(0)
    while not (Path(__file__).parent / "escaped.pid").exists(): time.sleep(.001)
Path("checkpoint.json").write_bytes(b'{{"phase":1}}')
time.sleep(60)
''')
        code, stdout, stderr = run_cli(
            "run-agent", "--agent", "codex", "--tasks", tasks, "--runs", root / "runs",
            "--codex-cmd", f"{sys.executable} {script}", "--timeout", "3")
        return (code, stdout, stderr), root / "runs" / relative

    def test_public_native_zero_stop_and_terminal_checkpoint_vetoes(self):
        popen, killpg = sb.subprocess.Popen, sb.os.killpg
        for mode, expected in (("stop", "checkpoint_stop"), ("changed", "checkpoint_mismatch"),
                               ("equivalent", "checkpoint_mismatch"), ("removed", "checkpoint_mismatch"),
                               ("handler-complete", "natural_completion"), ("before-complete", "natural_completion"),
                               ("failure", "process_failed"), ("malformed-after-complete", "natural_completion"),
                               ("unterminated", "capture_failed")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                processes = []
                def spawn(*args, processes=processes, **kwargs):
                    process = popen(*args, **kwargs)
                    processes.append(process)
                    return process
                def signal_group(group, number, processes=processes):
                    result = killpg(group, number)
                    if number == signal.SIGTERM:
                        self.assertEqual(processes[0].wait(timeout=2), 0)
                    return result
                with mock.patch.object(sb.subprocess, "Popen", side_effect=spawn), \
                        mock.patch.object(sb.os, "killpg", side_effect=signal_group):
                    result, base = self.run_native_case(Path(temporary), mode)
                self.assertEqual(result[0], 0 if mode == "stop" else 1, result[2])
                record = json.loads((base / "recovery.json").read_text())
                facts = json.loads((base / "recovery/initial/process.json").read_text())
                self.assertEqual(facts["state"], expected)
                self.assertEqual(facts["os_returncode"], 0)
                self.assertEqual(facts["compatibility_returncode"], 0)
                self.assertIsNone(facts["termination_cause"])
                self.assertTrue(facts["guards"]["live_after_capture"])
                self.assertTrue(facts["guards"]["live_before_term"])
                self.assertTrue(facts["guards"]["term_requested"])
                self.assertTrue(facts["pipes_drained"])
                self.assertTrue(facts["group_stopped"])
                self.assertEqual(record["phases"][0]["adapter_environment"]["recovery_process"], facts)
                self.assertEqual(len(record["phases"]), 3 if mode == "stop" else 1)
                self.assertIn(b'handler-ready', (base / "recovery/initial/stdout.bin").read_bytes())
                self.assertIsNone(record["certificate"])


class ClaudeNativeProtocolTests(unittest.TestCase):
    def run_protocol_case(self, root, mode, *, large_prompt=False, scope_frame=None):
        _, tasks, relative = write_with_skill_task(root)
        row = json.loads(tasks.read_text())
        row["recovery"] = {
            "checkpoint_path": "checkpoint.json", "expected_content": "{}",
            "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE", "forbidden_path": "target"}
        if large_prompt:
            row["prompt"] = "ORIGINAL " + "x" * 100000
        tasks.write_text(json.dumps(row) + "\n")
        script = root / "claude"
        script.write_text(f'''#!{sys.executable}
import json, os, signal, sys, time
from pathlib import Path
mode = {mode!r}
assert sys.argv[sys.argv.index("--input-format") + 1] == "stream-json"
assert "--permission-prompt-tool" not in sys.argv
os.write(2, b"STDERR_CANARY" * 20000)
for kind in ("initialize", "get_settings"):
    request = json.loads(sys.stdin.readline())
    assert request["request"]["subtype"] == kind
    if kind == "initialize":
        assert request["request"] == {{"subtype":"initialize","hooks":None}}
    response = {{"type":"control_response","response":{{"subtype":"success","request_id":request["request_id"],
        "response":{{"applied":{{"model":"resolved-model","effort":None}},"settings":{{"secret":"SETTINGS_CANARY"}}}}}}}}
    if kind == "get_settings":
        if mode == "wrong-id": response["response"]["request_id"] = "wrong"
        if mode == "settings-error": response["response"] = {{"subtype":"error","request_id":request["request_id"],"error":"SETTINGS_CANARY"}}
        if mode == "invalid-effort": response["response"]["response"]["applied"]["effort"] = ["SETTINGS_CANARY"]
        if mode == "malformed":
            os.write(1, b'{{"SETTINGS_CANARY":oops}}\\n')
            time.sleep(60)
        if mode == "duplicate":
            os.write(1, ('{{"type":"control_response","type":"SETTINGS_CANARY"}}\\n').encode())
            time.sleep(60)
        if mode == "oversized":
            os.write(1, b'{{"secret":"SETTINGS_CANARY' + b'x' * 1100000)
            time.sleep(60)
        if mode == "truncated":
            os.write(1, b'{{"secret":"SETTINGS_CANARY')
            raise SystemExit(0)
        if mode == "incoming-control":
            print(json.dumps({{"type":"control_request","request_id":"incoming","request":{{"subtype":"can_use_tool","secret":"SETTINGS_CANARY"}}}}), flush=True)
            time.sleep(60)
    wire = (json.dumps(response) + "\\n").encode()
    for i in range(0, len(wire), 19): os.write(1, wire[i:i+19])
user = json.loads(sys.stdin.readline())
assert user["type"] == "user" and user["session_id"] == "default"
assert user["parent_tool_use_id"] is None
prompt = user["message"]["content"]
assert sys.stdin.read() == ""
session = "default" if mode == "default-session" else str(os.getpid())
print(json.dumps({{"type":"system","subtype":"init","session_id":session,"model":"init-model"}}), flush=True)
if mode == "event-controls":
    print(json.dumps({{"type":"stream_event","session_id":session,"event":{{"type":"message_start","message":{{"role":"assistant","content":[],"model":"served-model","settings":{{"secret":"SETTINGS_CANARY"}}}}}}}}), flush=True)
if mode == "invalid-event":
    print(json.dumps({{"type":"stream_event","session_id":session,"event":{{"type":"message_start","message":{{"type":"control_response","settings":{{"secret":"SETTINGS_CANARY"}}}}}}}}), flush=True)

if mode == "ambiguous-session":
    print(json.dumps({{"type":"system","subtype":"init","session_id":"other"}}), flush=True)
print(json.dumps({{"type":"assistant","session_id":session,"message":{{"role":"assistant","model":"served-model","content":[{{"type":"text","text":"task"}}]}}}}), flush=True)
if prompt not in {{"RECOVER", "REFUSE"}}:
    def handle(number, frame):
        if mode == "parent-scope":
            record = {scope_frame!r}
            parent = record.get("parent_tool_use_id")
            record["session_id"] = "nested-session" if isinstance(parent, str) else session
            print(json.dumps(record), flush=True)
            if parent is not None and not isinstance(parent, str):
                print(json.dumps({{"type":"assistant","session_id":session,"message":{{"role":"assistant",
                    "model":"served-model","content":[{{"type":"text","text":"AFTER_SCOPE_CANARY"}}]}}}}), flush=True)
        if mode in {{"assistant-error", "unknown-assistant-error", "malformed-after-assistant-error"}}:
            error = {{"detail":"SETTINGS_CANARY"}} if mode == "unknown-assistant-error" else "authentication_failed"
            print(json.dumps({{"type":"assistant","session_id":session,"error":error,
                "error_detail":"SETTINGS_CANARY","message":{{"role":"assistant","model":"served-model","content":[]}}}}), flush=True)
            if mode == "malformed-after-assistant-error": os.write(1, b'{{"secret":"SETTINGS_CANARY"}}invalid\\n')
        if mode == "handler-complete":
            print(json.dumps({{"type":"result","session_id":session,"is_error":False,"result":"done"}}), flush=True)
            os.write(1, b'{{"secret":"SETTINGS_CANARY"}}invalid\\n')
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, handle)
    Path("checkpoint.pending").write_bytes(b"{{}}")
    Path("checkpoint.pending").rename("checkpoint.json")
    time.sleep(60)
print(json.dumps({{"type":"result","session_id":session,"is_error":False,"result":"done","subtype":"success"}}), flush=True)
''')
        script.chmod(0o755)
        result = run_cli("run-agent", "--agent", "claude", "--tasks", tasks,
                         "--runs", root / "runs", "--claude-bin", script, "--timeout", "4")
        return result, root / "runs" / relative

    def test_public_malformed_parent_scope_blocks_continuation_and_omits_secrets(self):
        records = (
            {"type": "result", "is_error": False, "result": "SCOPE_CANARY"},
            {"type": "assistant", "error": "authentication_failed", "error_detail": "SCOPE_CANARY",
             "message": {"role": "assistant", "model": "nested-model",
                         "content": [{"type": "text", "text": "SCOPE_CANARY"}]}},
            {"type": "result", "is_error": True, "result": "SCOPE_CANARY"},
        )
        for parent in (False, True, 0, 1, -1, 0.0, 1.5, {}, [],
                       {"secret": "SCOPE_CANARY"}, ["SCOPE_CANARY"]):
            for record in records:
                with self.subTest(parent=parent, record=record), tempfile.TemporaryDirectory() as temporary:
                    result, base = self.run_protocol_case(
                        Path(temporary), "parent-scope", scope_frame=dict(record, parent_tool_use_id=parent))
                    self.assertEqual(result[0], 1, result[2])
                    recovery = json.loads((base / "recovery.json").read_text())
                    self.assertEqual((recovery["failure"], len(recovery["phases"])), ("capture_failed", 1))
                    facts = json.loads((base / "recovery/initial/process.json").read_text())
                    self.assertEqual((facts["state"], facts["os_returncode"], facts["error"]),
                                     ("capture_failed", 0, "native_frame_invalid"))
                    self.assertEqual(facts["terminal"], {"session_id": str(facts["pid"]), "readable": False,
                                                        "completion_seen": False, "failure_seen": False})
                    self.assertTrue(facts["leader_reaped"] and facts["pipes_drained"] and facts["group_stopped"])
                    native = json.loads((base / "recovery/initial/native.json").read_text())
                    self.assertEqual(native["availability"], "unavailable")
                    self.assertEqual([item["model"] for item in native["served_models"]], ["served-model"])
                    for path in base.rglob("*"):
                        if path.is_file():
                            self.assertNotIn(b"CANARY", path.read_bytes(), str(path))
                    self.assertNotIn("CANARY", str(result))

    def test_public_valid_parent_scope_preserves_terminal_vetoes_and_nested_continuation(self):
        records = (
            ({"type": "result", "is_error": False, "result": "done"}, "natural_completion", True, False),
            ({"type": "assistant", "error": "authentication_failed",
              "message": {"role": "assistant", "model": "nested-model", "content": []}},
             "process_failed", False, True),
            ({"type": "result", "is_error": True}, "process_failed", False, True),
        )
        for scope, main in (({}, True), ({"parent_tool_use_id": None}, True),
                            ({"parent_tool_use_id": "tool-parent"}, False),
                            ({"parent_tool_use_id": ""}, False), ({"parent_tool_use_id": " "}, False)):
            for record, state, completed, failed in records:
                with self.subTest(scope=scope, record=record), tempfile.TemporaryDirectory() as temporary:
                    result, base = self.run_protocol_case(Path(temporary), "parent-scope", scope_frame=dict(record, **scope))
                    self.assertEqual(result[0], 1 if main else 0, result[2])
                    recovery = json.loads((base / "recovery.json").read_text())
                    self.assertEqual([phase["state"] for phase in recovery["phases"]],
                                     [state] if main else ["checkpoint_stop", "complete", "complete"])
                    facts = json.loads((base / "recovery/initial/process.json").read_text())
                    self.assertEqual(facts["terminal"], {"session_id": str(facts["pid"]), "readable": True,
                                                        "completion_seen": completed and main, "failure_seen": failed and main})
                    frames = list(map(json.loads, (base / "recovery/initial/stdout.bin").read_text().splitlines()))
                    expected = dict(record, session_id=str(facts["pid"]) if main else "nested-session", **scope)
                    self.assertEqual(frames[-1], expected)

    def test_public_controls_filter_secrets_and_preserve_unknown_effort(self):
        for mode in ("safe", "settings-error", "event-controls", "invalid-event", "wrong-id", "malformed", "invalid-effort",
                     "duplicate", "oversized", "truncated", "incoming-control", "default-session", "ambiguous-session", "handler-complete",
                     "assistant-error", "unknown-assistant-error", "malformed-after-assistant-error"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                result, base = self.run_protocol_case(Path(temporary), mode)
                self.assertEqual(result[0], 0 if mode in {"safe", "settings-error", "event-controls"} else 1, result[2])
                for path in base.rglob("*"):
                    if path.is_file():
                        self.assertNotIn(b"SETTINGS_CANARY", path.read_bytes(), str(path))
                        self.assertNotIn(b"STDERR_CANARY", path.read_bytes(), str(path))
                self.assertNotIn("CANARY", str(result))
                native = json.loads((base / "recovery/initial/native.json").read_text())
                facts = json.loads((base / "recovery/initial/process.json").read_text())
                self.assertEqual(facts["stderr_contract"], "omitted-v2")
                self.assertEqual(facts["raw_sha256"]["stderr.bin"], sb.hashlib.sha256(b"").hexdigest())
                self.assertGreater(facts["omitted_stderr_bytes"], 0)
                if mode in {"safe", "settings-error", "event-controls"}:
                    self.assertEqual(native["settings"][0]["availability"], "unavailable" if mode == "settings-error" else "complete")
                    self.assertIsNone(native["settings"][0]["effort"])
                    self.assertEqual(native["settings"][0]["effort_available"], mode != "settings-error")
                    self.assertEqual(native["settings"][0]["scope"], "next_request")
                    self.assertEqual(native["served_models"][0]["model"], "served-model")
                    self.assertNotEqual(native["session_id"], "default")
                    self.assertEqual(facts["os_returncode"], 0)
                if mode == "handler-complete":
                    self.assertTrue(facts["terminal"]["completion_seen"])
                    self.assertFalse(facts["terminal"]["readable"])
                if mode in {"assistant-error", "unknown-assistant-error", "malformed-after-assistant-error"}:
                    self.assertTrue(facts["terminal"]["failure_seen"], facts)
                    self.assertFalse(facts["terminal"]["completion_seen"])
                    self.assertEqual(facts["state"], "capture_failed" if mode == "malformed-after-assistant-error" else "process_failed")
                    self.assertEqual(len(json.loads((base / "recovery.json").read_text())["phases"]), 1)
                    errors = [frame["error"] for frame in map(json.loads, (base / "recovery/initial/stdout.bin").read_text().splitlines())
                              if "error" in frame]
                    self.assertEqual(errors, ["unknown" if mode == "unknown-assistant-error" else "authentication_failed"])

    def test_partial_writes_backpressure_multiplexing_and_eof(self):
        write = sb.os.write
        attempts = 0
        def partial_write(descriptor, content):
            nonlocal attempts
            attempts += 1
            if attempts % 3 == 0:
                raise BlockingIOError()
            return write(descriptor, content[:117])
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(sb.os, "write", side_effect=partial_write):
            result, base = self.run_protocol_case(Path(temporary), "safe", large_prompt=True)
            self.assertEqual(result[0], 0, result[2])
            self.assertGreater(attempts, 100)
            facts = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertTrue(facts["pipes_drained"])
            self.assertEqual(facts["state"], "checkpoint_stop")


class CodexNativeObservationTests(unittest.TestCase):
    run_native_case = NativeRecoveryStopTests.run_native_case

    def test_public_scoped_projection_precedes_cleanup_and_preserves_repeated_scopes(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb, "locate_codex_rollout", side_effect=AssertionError("ambient lookup used")):
            result, base = self.run_native_case(Path(temporary), "metadata-valid")
            self.assertEqual(result[0], 0, result[2])
            native = json.loads((base / "recovery/initial/native.json").read_text())
            self.assertEqual(native["availability"], "partial")
            self.assertNotEqual(native["thread_id"], native["root_session_id"])
            self.assertEqual(native["cli_version"], "0.160.1")
            self.assertEqual(len(native["contexts"]), 2)
            first, second = native["contexts"]
            self.assertEqual((first["scope"], first["turn_id"], first["effort"]),
                             ("session_unknown_turn", None, "high"))
            self.assertEqual((second["scope"], second["turn_id"], second["effort"], second["effort_available"]),
                             ("turn", "turn-1", None, True))
            self.assertEqual((first["approval_policy"], first["sandbox_type"], first["network_access"]),
                             ("never", "read-only", False))
            retained = (base / "recovery/initial/codex-context.jsonl").read_bytes()
            for context in native["contexts"]:
                self.assertEqual(sb.hashlib.sha256(retained.splitlines(keepends=True)[context["line"] - 1]).hexdigest(),
                                 context["retained_sha256"])
            self.assertEqual(sb.hashlib.sha256(retained).hexdigest(), native["retained_sha256"])
            for path in base.rglob("*"):
                if path.is_file():
                    self.assertNotIn(b"ROLLOUT_CANARY", path.read_bytes(), str(path))
            invocation = json.loads((base / "recovery/initial/invocation.json").read_text())
            self.assertNotIn("--ephemeral", invocation["client_argv"])
            identity = invocation["workspace_identity"]
            self.assertTrue(Path(identity["canonical_cwd"]).is_absolute())
            self.assertGreater(identity["inode"], 0)
            self.assertEqual({entry["module"] for entry in invocation["loaded_files"]},
                             {"skill_benchmark", "invocation_contracts"})
            self.assertEqual(invocation["installed_distribution"]["name"], "skill-eval-harness-ext")
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(record["schema_version"], 2)
            self.assertFalse(Path(record["workspace"]).exists())
            environment = record["phases"][0]["adapter_environment"]
            self.assertEqual(environment["temporary_home_cleanup"]["status"], "removed")
            self.assertIsNone(record["runtime_model"])
            self.assertIsNone(record["runtime_effort"])
            self.assertIsNone(record["certificate"])

    def test_public_missing_mismatched_ambiguous_and_unflushed_rollouts_remain_unknown(self):
        for mode in ("metadata-missing", "metadata-mismatch", "metadata-ambiguous",
                     "metadata-version", "metadata-unflushed", "metadata-symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                result, base = self.run_native_case(Path(temporary), mode)
                self.assertEqual(result[0], 0, result[2])
                native = json.loads((base / "recovery/initial/native.json").read_text())
                self.assertEqual(native["availability"], "unavailable")
                self.assertEqual(native["contexts"], [])
                self.assertEqual((base / "recovery/initial/codex-context.jsonl").read_bytes(), b"")

    def test_explicit_ephemeral_is_preserved_and_metadata_is_unavailable(self):
        original = sb.codex_cli_invoke
        def supplied(prompt, **options):
            options["codex_cmd"] += " --ephemeral"
            return original(prompt, **options)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(sb, "codex_cli_invoke", side_effect=supplied):
            result, base = self.run_native_case(Path(temporary), "stop")
            self.assertEqual(result[0], 0, result[2])
            native = json.loads((base / "recovery/initial/native.json").read_text())
            self.assertEqual((native["availability"], native["reason"]), ("unavailable", "explicit_ephemeral"))
            invocation = json.loads((base / "recovery/initial/invocation.json").read_text())
            self.assertEqual(invocation["client_argv"].count("--ephemeral"), 1)

    def test_native_artifact_failures_block_continuation_and_publication_is_last(self):
        write_bytes, write_json = Path.write_bytes, sb.write_json
        atomic_write = sb._atomic_write_text
        for artifact in ("checkpoint-observed.bin", "checkpoint-final.bin", "codex-context.jsonl", "native.json", "stdout.bin", "process.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                def fail_bytes(path, content, artifact=artifact):
                    if path.name == artifact:
                        raise OSError("artifact failed")
                    return write_bytes(path, content)
                def fail_json(path, content, artifact=artifact):
                    if path.name == artifact:
                        raise OSError("artifact failed")
                    return write_json(path, content)
                def fail_publication(path, content, artifact=artifact):
                    if path.name == artifact:
                        raise OSError("artifact failed")
                    return atomic_write(path, content)
                with mock.patch.object(Path, "write_bytes", autospec=True, side_effect=fail_bytes), \
                        mock.patch.object(sb, "_atomic_write_text", side_effect=fail_publication), \
                        mock.patch.object(sb, "write_json", side_effect=fail_json):
                    result, base = self.run_native_case(Path(temporary), "stop")
                self.assertEqual(result[0], 1, result[2])
                record = json.loads((base / "recovery.json").read_text())
                self.assertEqual((record["failure"], len(record["phases"])), ("capture_failed", 1))
                self.assertEqual(record["phases"][0]["state"], "capture_failed")
                process = base / "recovery/initial/process.json"
                if artifact == "process.json":
                    self.assertFalse(process.exists())
                else:
                    facts = json.loads(process.read_text())
                    self.assertEqual(facts["state"], "capture_failed")
                    self.assertEqual(facts, record["phases"][0]["adapter_environment"]["recovery_process"])

    def test_failed_publication_removes_a_partially_committed_marker(self):
        atomic_write = sb._atomic_write_text
        def fail_after_replace(path, content):
            atomic_write(path, content)
            if path.name == "process.json":
                raise OSError("publication acknowledgment failed")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb, "_atomic_write_text", side_effect=fail_after_replace):
            result, base = self.run_native_case(Path(temporary), "stop")
            self.assertEqual(result[0], 1, result[2])
            self.assertFalse((base / "recovery/initial/process.json").exists())
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual((record["failure"], record["phases"][0]["state"]),
                             ("capture_failed", "capture_failed"))

    def test_native_snapshot_failure_blocks_the_run_after_process_publication(self):
        snapshot = sb.snapshot_recovery_workspace
        for destination, expected_phases in (("after", 1), ("final", 3)):
            with self.subTest(destination=destination), tempfile.TemporaryDirectory() as temporary:
                def fail_after_capture(workspace, directory, destination=destination):
                    snapshot(workspace, directory)
                    if directory.name == destination:
                        raise OSError("native snapshot failed")
                with mock.patch.object(sb, "snapshot_recovery_workspace", side_effect=fail_after_capture):
                    result, base = self.run_native_case(Path(temporary), "stop")
                self.assertEqual(result[0], 1, result[2])
                record = json.loads((base / "recovery.json").read_text())
                self.assertEqual(record["status"], "failed")
                self.assertIn("native snapshot failed", record["failure"])
                self.assertEqual(len(record["phases"]), expected_phases)
                facts = json.loads((base / "recovery/initial/process.json").read_text())
                self.assertEqual(facts["state"], "checkpoint_stop")
                self.assertEqual(record["phases"][0]["state"], "checkpoint_stop")
                self.assertEqual(facts, record["phases"][0]["adapter_environment"]["recovery_process"])
                self.assertIsNone(record["certificate"])


class NativeLifecycleRaceTests(unittest.TestCase):
    run_native_case = NativeRecoveryStopTests.run_native_case

    def test_native_group_cleanup_preserves_leader_until_after_handler_output(self):
        for eof_on_term, terminal, expected in (
                (True, b'{"type":"turn.completed"}\n', "natural_completion"),
                (False, b'{"type":"turn.failed"}\n', "process_failed")):
            with self.subTest(eof_on_term=eof_on_term), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                root = Path(temporary)
                evidence = root / "evidence"
                evidence.mkdir()
                pipes = []
                for _ in range(3):
                    reader, writer = os.pipe()
                    pipes.append((stack.enter_context(os.fdopen(reader, "rb", buffering=0)),
                                  stack.enter_context(os.fdopen(writer, "wb", buffering=0))))
                (stdin_reader, stdin_writer), (stdout_reader, stdout_writer), (stderr_reader, stderr_writer) = pipes
                process = mock.Mock(pid=424242, returncode=None, stdin=stdin_writer,
                                    stdout=stdout_reader, stderr=stderr_reader)
                state = {"exited": False}
                cleanup_returncodes = []
                stdout_writer.write(b'{"type":"thread.started","thread_id":"ordering-control"}\n')
                write = os.write

                def deliver_prompt(fd, content, root=root, stdin_writer=stdin_writer, write=write):
                    written = write(fd, content)
                    if fd == stdin_writer.fileno():
                        (root / "checkpoint.json").write_bytes(b'{"phase":1}')
                    return written

                def observe_exit(kind, pid, flags, state=state):
                    return object() if state["exited"] else None

                def signal_group(group, number, process=process, state=state,
                                 stdout_writer=stdout_writer, stderr_writer=stderr_writer,
                                 terminal=terminal, eof_on_term=eof_on_term,
                                 cleanup_returncodes=cleanup_returncodes):
                    self.assertEqual(group, process.pid)
                    if number == signal.SIGTERM:
                        stdout_writer.write(terminal)
                        stdout_writer.close()
                        if eof_on_term:
                            stderr_writer.close()
                        state["exited"] = True
                    elif number == signal.SIGKILL:
                        cleanup_returncodes.append(process.returncode)
                        stderr_writer.close()

                def reap(timeout, process=process):
                    process.returncode = 0
                    return 0

                process.wait.side_effect = reap
                capture = sb.RecoveryCapture(evidence, sb.RecoveryCase.parse({
                    "checkpoint_path": "checkpoint.json", "expected_content": '{"phase":1}',
                    "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE",
                    "forbidden_path": "forbidden.txt"}))
                plan = sb.ProcessInvocationPlan.from_values(
                    [sys.executable], input_text="INITIAL", cwd=root, timeout_s=3,
                    recovery_capture=capture, native_recovery=sb.NativeRecoveryConfig("codex"))
                with mock.patch.object(sb.subprocess, "Popen", return_value=process), \
                        mock.patch.object(sb.os, "write", side_effect=deliver_prompt), \
                        mock.patch.object(sb.os, "waitid", side_effect=observe_exit, create=True), \
                        mock.patch.object(sb.os, "killpg", side_effect=signal_group), \
                        mock.patch.object(sb, "recovery_group_stopped", return_value=(True, "control_group_stopped")):
                    result = sb.run_argv_capture(plan)
                self.assertEqual(stdin_reader.read(), b"INITIAL")
                facts = json.loads((evidence / "process.json").read_text())
                self.assertEqual(facts["state"], expected)
                self.assertEqual(capture.state.value, expected)
                self.assertEqual(facts["os_returncode"], 0)
                self.assertEqual(result.returncode, 0)
                self.assertEqual((facts["leader_reaped"], facts["pipes_drained"], facts["group_stopped"]),
                                 (True, True, True))
                self.assertTrue(facts["guards"]["term_requested"])
                self.assertEqual(result.stdout.encode(), (evidence / "stdout.bin").read_bytes())
                self.assertTrue(result.stdout.encode().endswith(terminal))
                self.assertEqual(cleanup_returncodes, [None])

    def test_both_live_and_deadline_rechecks_prevent_signaling(self):
        popen, write_bytes, monotonic, kill = sb.subprocess.Popen, Path.write_bytes, sb.time.monotonic, sb.os.kill
        for boundary in (1, 2):
            for event in ("leader_exit", "deadline"):
                with self.subTest(boundary=boundary, event=event), tempfile.TemporaryDirectory() as temporary:
                    processes = []
                    state = {"checking": False, "checks": 0}
                    def spawn(*args, processes=processes, **kwargs):
                        process = popen(*args, **kwargs)
                        processes.append(process)
                        return process
                    def observe(path, content, state=state):
                        result = write_bytes(path, content)
                        if path.name == "checkpoint-observed.bin":
                            state["checking"] = True
                        return result
                    def clock(boundary=boundary, event=event, state=state, processes=processes):
                        value = monotonic()
                        if state["checking"]:
                            state["checks"] += 1
                            if state["checks"] == boundary:
                                kill(processes[0].pid, signal.SIGTERM)
                                self.assertEqual(processes[0].wait(timeout=2), 0)
                                if event == "deadline":
                                    return value + 100
                        return value
                    with mock.patch.object(sb.subprocess, "Popen", side_effect=spawn), \
                            mock.patch.object(Path, "write_bytes", autospec=True, side_effect=observe), \
                            mock.patch.object(sb.time, "monotonic", side_effect=clock):
                        result, base = self.run_native_case(Path(temporary), "stop")
                    self.assertEqual(result[0], 1, result[2])
                    facts = json.loads((base / "recovery/initial/process.json").read_text())
                    self.assertEqual(facts["state"], "natural_completion" if event == "leader_exit" else "deadline")
                    self.assertEqual(facts["os_returncode"], 0)
                    self.assertIsNone(facts["signal_sent"])
                    self.assertFalse(facts["guards"]["term_requested"])
                    record = json.loads((base / "recovery.json").read_text())
                    self.assertEqual(len(record["phases"]), 1)

    def test_failed_term_with_actual_zero_and_failed_group_observation_are_not_rescued(self):
        popen, killpg = sb.subprocess.Popen, sb.os.killpg
        for mode in ("term", "group"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                processes = []
                def spawn(*args, processes=processes, **kwargs):
                    process = popen(*args, **kwargs)
                    processes.append(process)
                    return process
                def signal_group(group, number, mode=mode, processes=processes):
                    result = killpg(group, number)
                    if mode == "term" and number == signal.SIGTERM:
                        self.assertEqual(processes[0].wait(timeout=2), 0)
                        raise OSError("synthetic TERM failure")
                    return result
                observe = sb.recovery_group_stopped
                def group_status(group, mode=mode, observe=observe):
                    return (False, "observation_unavailable") if mode == "group" else observe(group)
                with mock.patch.object(sb.subprocess, "Popen", side_effect=spawn), \
                        mock.patch.object(sb.os, "killpg", side_effect=signal_group), \
                        mock.patch.object(sb, "recovery_group_stopped", side_effect=group_status):
                    result, base = self.run_native_case(Path(temporary), "stop")
                self.assertEqual(result[0], 1, result[2])
                facts = json.loads((base / "recovery/initial/process.json").read_text())
                self.assertEqual(facts["state"], "signal_failed" if mode == "term" else "cleanup_failed")
                self.assertEqual(facts["os_returncode"], 0)
                if mode == "term":
                    self.assertIsNone(facts["signal_sent"])
                    self.assertFalse(facts["guards"]["term_requested"])

    def test_escaped_pipe_holder_is_not_observed_as_eof_and_is_cleaned_up_by_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            try:
                result, base = self.run_native_case(root, "escaped-pipe")
                self.assertEqual(result[0], 1, result[2])
                facts = json.loads((base / "recovery/initial/process.json").read_text())
                self.assertEqual(facts["state"], "capture_failed")
                self.assertFalse(facts["pipes_drained"])
                self.assertTrue(facts["leader_reaped"])
                self.assertTrue(facts["group_stopped"])
                self.assertEqual(facts["os_returncode"], 0)
            finally:
                sb.os.kill(int((root / "escaped.pid").read_text()), signal.SIGKILL)
