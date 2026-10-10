import errno
import hashlib
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import run_cli, write_with_skill_task

import skill_benchmark as sb
from invocation_contracts import RecoveryCase


class RecoveryTests(unittest.TestCase):
    def setup_case(self, root, *, mode="stop", match="json", recovery=True):
        fixture = root / "fixture.bin"
        fixture.write_bytes(b"fixture\x00\xff\r\n")
        case = {"checkpoint_path": "checkpoint.json", "expected_content": '{"phase":1}',
                "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE",
                "forbidden_path": "forbidden.txt", "match": match}
        _, tasks, _ = write_with_skill_task(root)
        row = json.loads(tasks.read_text())
        row.update(variant="without_skill", skill_paths=[], skill_root_keys=[],
                   input_files=[str(fixture)], run_dir="case-1/without_skill",
                   prompt="INITIAL", instruction="")
        row.pop("skill_tree_hash", None)
        if recovery:
            row["recovery"] = case
        tasks.write_text(json.dumps(row) + "\n")
        script = root / "fake.py"
        script.write_text(f'''import hashlib, json, os, signal, sys, time
from pathlib import Path
mode = {mode!r}
prompt = sys.argv[sys.argv.index("--prompt") + 1]
phase = prompt if prompt in {{"RECOVER", "REFUSE"}} else "INITIAL"
fixture = Path("inputs/fixture.bin").read_bytes()
assert fixture == b"fixture\\x00\\xff\\r\\n"
with Path("turns.jsonl").open("a") as handle:
    handle.write(json.dumps({{"phase": phase, "pid": os.getpid(), "cwd": os.getcwd(),
                             "fixture": hashlib.sha256(fixture).hexdigest()}}) + "\\n")
if phase == "RECOVER":
    assert json.loads(Path("checkpoint.json").read_bytes()) == {{"phase": 1}}
    Path("recovered.txt").write_bytes(b"recovered")
    if mode == "recovery-fails":
        print(json.dumps({{"role": "assistant", "content": "failed recovery"}}), flush=True)
        raise SystemExit(7)
elif phase == "REFUSE":
    assert Path("recovered.txt").read_bytes() == b"recovered"
    print(json.dumps({{"attempted_write": "forbidden.txt", "denial": "fake denial"}}), flush=True)
    if mode == "forbidden":
        Path("forbidden.txt").write_bytes(b"not enforced")
elif mode != "one-shot":
    if mode == "exit-124":
        raise SystemExit(124)
    if mode in {{"early", "terminal"}}:
        if mode == "terminal":
            Path("checkpoint.json").write_bytes(b'{{"phase":1}}')
        print(json.dumps({{"role": "assistant", "content": "early"}}), flush=True)
        raise SystemExit(0)
    if mode in {{"descendant", "exit-before-signal-descendant"}}:
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if mode == "exit-before-signal-descendant":
                Path("child-ready").touch()
            time.sleep(60)
            os._exit(0)
        Path("child.pid").write_text(str(child))
        if mode == "exit-before-signal-descendant":
            while not Path("child-ready").exists():
                time.sleep(0.005)
    os.write(1, (json.dumps({{"role": "assistant", "content": "initial"}}) + "\\n").encode())
    os.write(1, (json.dumps({{"raw": "z" * 16000}}) + "\\r\\n").encode())
    os.write(2, b"raw-secret\\xff" + b"x" * 16000)
    if mode == "timeout":
        time.sleep(60)
    if mode == "changed":
        def change_checkpoint(number, frame):
            Path("checkpoint.json").write_bytes(b"changed")
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)
        signal.signal(signal.SIGTERM, change_checkpoint)
    if mode == "buffered":
        def flush_evidence(number, frame):
            os.write(1, b'{{"buffered":"after signal"}}\\n')
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)
        signal.signal(signal.SIGTERM, flush_evidence)
    if mode == "exit-handler":
        signal.signal(signal.SIGTERM, lambda number, frame: sys.exit(0))
    if mode == "removed":
        def remove_checkpoint(number, frame):
            Path("checkpoint.json").unlink()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)
        signal.signal(signal.SIGTERM, remove_checkpoint)
    if mode == "bad":
        Path("checkpoint.json").write_bytes(b"{{")
    elif mode == "wrong":
        Path("checkpoint.json").write_bytes(b'{{"phase":2}}')
    elif mode == "bytes":
        Path("checkpoint.json").write_bytes(b'{{"phase":1}}')
    elif mode == "number":
        Path("checkpoint.json").write_bytes(b'{{"phase":1.0}}')
    elif mode == "boolean":
        Path("checkpoint.json").write_bytes(b'{{"phase":true}}')
    elif mode == "escape":
        outside = Path(__file__).resolve().parent / "outside.json"
        outside.write_bytes(b'{{"phase":1}}')
        Path("checkpoint.json").symlink_to(outside)
    else:
        Path("checkpoint.json").write_bytes(b'{{ "phase": 1 }}\\n')
    if mode in {{"exit-before-signal", "exit-before-signal-descendant"}}:
        while not Path("release").exists():
            time.sleep(0.005)
        raise SystemExit(0)
    time.sleep(60)
print(json.dumps({{"role": "assistant", "content": "answer " + phase,
                  "usage": {{"input_tokens": 2, "output_tokens": 3}}}}), flush=True)
''')
        return tasks, script, row

    def run_case(self, root, *, mode="stop", match="json", recovery=True, timeout=4):
        tasks, script, row = self.setup_case(root, mode=mode, match=match, recovery=recovery)
        runs = root / "runs"
        result = run_cli("run-agent", "--agent", "vibe", "--tasks", tasks, "--runs", runs,
                         "--model", "fake-model", "--vibe-cmd", f"{sys.executable} {script}",
                         "--timeout", str(timeout))
        base = runs / row["run_dir"]
        return result, base

    def read_snapshot(self, directory, relative):
        files = json.loads((directory / "files.json").read_text())
        entry = files[relative]
        content = (directory / entry["blob"]).read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
        return content

    def test_public_fixed_phases_preserve_bytes_and_fresh_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, base = self.run_case(root)
            self.assertEqual(result[0], 0, result[2])
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(record["status"], "complete")
            self.assertEqual([phase["phase"] for phase in record["phases"]],
                             ["initial", "recovery", "refusal"])
            self.assertEqual([phase["state"] for phase in record["phases"]],
                             ["checkpoint_stop", "complete", "complete"])
            evidence = base / "recovery"
            process = json.loads((evidence / "initial/process.json").read_text())
            self.assertEqual(process["os_returncode"], -signal.SIGTERM)
            self.assertEqual(process["signal_sent"], signal.SIGTERM)
            self.assertEqual((process["leader_reaped"], process["pipes_drained"],
                              process["group_stopped"], process["checkpoint_observed_live"]),
                             (True, True, True, True))
            stderr = (evidence / "initial/stderr.bin").read_bytes()
            self.assertEqual(stderr, b"raw-secret\xff" + b"x" * 16000)
            self.assertEqual(process["raw_sha256"]["stderr.bin"], hashlib.sha256(stderr).hexdigest())
            stdout = (evidence / "initial/stdout.bin").read_bytes()
            self.assertEqual(stdout, b'{"role": "assistant", "content": "initial"}\n'
                             + b'{"raw": "' + b"z" * 16000 + b'"}\r\n')
            for name in ("initial", "recovery", "refusal"):
                self.assertEqual(self.read_snapshot(evidence / name / "before", "inputs/fixture.bin"),
                                 b"fixture\x00\xff\r\n")
                self.assertEqual(self.read_snapshot(evidence / name / "after", "inputs/fixture.bin"),
                                 b"fixture\x00\xff\r\n")
            turns = [json.loads(line) for line in self.read_snapshot(
                evidence / "final", "turns.jsonl").splitlines()]
            self.assertEqual([turn["phase"] for turn in turns], ["INITIAL", "RECOVER", "REFUSE"])
            self.assertEqual(len({turn["pid"] for turn in turns}), 3)
            self.assertEqual(len({turn["cwd"] for turn in turns}), 1)
            self.assertEqual(record["requested_model"], "fake-model")
            self.assertEqual((record["runtime_model"], record["runtime_effort"],
                              record["enforcing_denial"], record["certificate"]), (None, None, None, None))
            invocation = json.loads((evidence / "refusal/invocation.json").read_text())
            self.assertEqual(invocation["client_argv"][:2], [sys.executable, str(root / "fake.py")])
            self.assertEqual(invocation["client_argv"][invocation["client_argv"].index("--prompt") + 1], "REFUSE")
            self.assertFalse(Path(record["workspace"]).exists())
            self.assertFalse(record["phases"][-1]["forbidden_path_present"])
            self.assertIn(b"attempted_write", (evidence / "refusal/stdout.bin").read_bytes())

    def test_failures_block_fresh_phases_and_retain_partial_evidence(self):
        for mode, match, state in (("bad", "json", "checkpoint_mismatch"),
                                   ("boolean", "json", "checkpoint_mismatch"),
                                   ("wrong", "json", "checkpoint_mismatch"),
                                   ("stop", "bytes", "checkpoint_mismatch"),
                                   ("early", "json", "natural_completion"),
                                   ("changed", "json", "checkpoint_mismatch"),
                                   ("removed", "json", "checkpoint_mismatch"),
                                   ("exit-handler", "json", "natural_completion"),
                                   ("timeout", "json", "deadline")):
            with self.subTest(mode=mode, match=match), tempfile.TemporaryDirectory() as temporary:
                result, base = self.run_case(Path(temporary), mode=mode, match=match, timeout=1)
                self.assertEqual(result[0], 1, result[2])
                record = json.loads((base / "recovery.json").read_text())
                self.assertEqual((record["status"], record["failure"]), ("failed", state))
                self.assertEqual(len(record["phases"]), 1)
                process = json.loads((base / "recovery/initial/process.json").read_text())
                self.assertEqual(process["state"], state)
                self.assertTrue(process["leader_reaped"])
                self.assertTrue(process["group_stopped"])
                self.assertFalse(Path(record["workspace"]).exists())
                if mode == "timeout":
                    self.assertEqual(process["os_returncode"], -signal.SIGKILL)

    def test_natural_exit_with_matching_checkpoint_blocks_recovery(self):
        popen = sb.subprocess.Popen
        children = []
        def spawn_exited(*arguments, **keywords):
            child = popen(*arguments, **keywords)
            children.append(child)
            try:
                self.assertEqual(child.wait(timeout=2), 0)
            except sb.subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
                raise
            return child
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb.subprocess, "Popen", side_effect=spawn_exited):
            result, base = self.run_case(Path(temporary), mode="terminal")
            self.assertEqual(result[0], 1, result[2])
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual((record["status"], record["failure"]),
                             ("failed", "natural_completion"))
            self.assertEqual([phase["phase"] for phase in record["phases"]], ["initial"])
            evidence = base / "recovery"
            process = json.loads((evidence / "initial/process.json").read_text())
            self.assertEqual((process["state"], process["os_returncode"], process["signal_sent"]),
                             ("natural_completion", 0, None))
            self.assertEqual((process["leader_reaped"], process["pipes_drained"],
                              process["group_stopped"], process["checkpoint_observed_live"]),
                             (True, True, True, False))
            self.assertEqual((evidence / "initial/stdout.bin").read_bytes(),
                             b'{"role": "assistant", "content": "early"}\n')
            self.assertEqual((evidence / "initial/stderr.bin").read_bytes(), b"")
            self.assertEqual(self.read_snapshot(evidence / "final", "checkpoint.json"),
                             b'{"phase":1}')
            turns = [json.loads(line) for line in self.read_snapshot(
                evidence / "final", "turns.jsonl").splitlines()]
            self.assertEqual([turn["phase"] for turn in turns], ["INITIAL"])
            self.assertEqual(len(children), 1)
            self.assertEqual(children[0].returncode, 0)
            with self.assertRaises(ProcessLookupError):
                os.kill(children[0].pid, 0)
            with self.assertRaises(ProcessLookupError):
                os.killpg(children[0].pid, 0)
            self.assertFalse(Path(record["workspace"]).exists())

    def test_matching_external_symlink_checkpoint_cannot_authorize_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, base = self.run_case(Path(temporary), mode="escape")
            self.assertEqual(result[0], 1, result[2])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual(process["state"], "observer_failed")
            self.assertIn("escapes workspace", process["error"])
            self.assertIsNone(process["signal_sent"])
            self.assertTrue(process["group_stopped"])
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(len(record["phases"]), 1)

    def test_checkpoint_directory_swap_cannot_read_external_bytes(self):
        open_file = os.open
        for swap_at in ("workspace", "file"):
            with self.subTest(swap_at=swap_at), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                workspace = root / "workspace"
                directory = workspace / "nested"
                directory.mkdir(parents=True)
                (directory / "checkpoint.json").write_bytes(b"inside")
                outside = root / "outside"
                outside.mkdir()
                (outside / "checkpoint.json").write_bytes(b"outside")
                self.assertEqual(sb.recovery_file_bytes(workspace, "nested/checkpoint.json"), b"inside")
                descriptors = []
                swapped = False

                def replace_directory(path, flags, *arguments, swap_at=swap_at,
                                      directory=directory, workspace=workspace,
                                      outside=outside, descriptors=descriptors, **keywords):
                    nonlocal swapped
                    if not swapped and (swap_at == "workspace" or Path(path).name == "checkpoint.json"):
                        directory.rename(workspace / "original")
                        directory.symlink_to(outside, target_is_directory=True)
                        swapped = True
                    descriptor = open_file(path, flags, *arguments, **keywords)
                    descriptors.append(descriptor)
                    return descriptor

                with mock.patch.object(sb.os, "open", side_effect=replace_directory):
                    if swap_at == "workspace":
                        with self.assertRaises((OSError, ValueError)):
                            sb.recovery_file_bytes(workspace, "nested/checkpoint.json")
                    else:
                        self.assertEqual(sb.recovery_file_bytes(workspace, "nested/checkpoint.json"), b"inside")
                self.assertTrue(swapped)
                for descriptor in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)

    def test_checkpoint_reads_require_posix_flags_before_opening_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "checkpoint.json").write_bytes(b"inside")
            for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"):
                with self.subTest(flag=flag), mock.patch.object(sb.os, flag), mock.patch.object(
                        sb.os, "open", side_effect=AssertionError("opened before resolving required flags")) as open_file:
                    delattr(sb.os, flag)
                    with self.assertRaisesRegex(AttributeError, flag):
                        sb.recovery_file_bytes(workspace, "checkpoint.json")
                    open_file.assert_not_called()

    def test_checkpoint_reads_reject_symlink_components_and_nonregular_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            directory = workspace / "nested"
            directory.mkdir()
            (directory / "checkpoint.json").write_bytes(b"inside")
            (workspace / "alias").symlink_to(directory, target_is_directory=True)
            (workspace / "checkpoint.json").symlink_to(directory / "checkpoint.json")
            os.mkfifo(workspace / "pipe")
            self.assertEqual(sb.recovery_file_bytes(workspace, "nested/checkpoint.json"), b"inside")
            for relative in ("alias/checkpoint.json", "checkpoint.json", "nested", "pipe"):
                with self.subTest(relative=relative), self.assertRaises((OSError, ValueError)):
                    sb.recovery_file_bytes(workspace, relative)

    def test_descendant_held_pipes_are_drained_and_group_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, base = self.run_case(Path(temporary), mode="descendant")
            self.assertEqual(result[0], 0, result[2])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertTrue(process["group_stopped"])
            self.assertTrue(process["pipes_drained"])
            child = int(self.read_snapshot(base / "recovery/initial/after", "child.pid"))
            self.assertNotEqual(child, process["pid"])
            self.assertEqual(sb.recovery_group_stopped(process["process_group"])[0], True)

    def test_evidence_buffered_until_signal_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, base = self.run_case(Path(temporary), mode="buffered")
            self.assertEqual(result[0], 0, result[2])
            self.assertIn(b'{"buffered":"after signal"}\n',
                          (base / "recovery/initial/stdout.bin").read_bytes())

    def test_forbidden_presence_is_a_fact_not_a_certificate(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, base = self.run_case(Path(temporary), mode="forbidden")
            self.assertEqual(result[0], 0, result[2])
            record = json.loads((base / "recovery.json").read_text())
            self.assertTrue(record["phases"][-1]["forbidden_path_present"])
            self.assertIsNone(record["certificate"])
            self.assertEqual(self.read_snapshot(base / "recovery/final", "forbidden.txt"), b"not enforced")

    def test_invalid_prepared_recovery_fails_before_any_spawn(self):
        for field, value in (("recovery_prompt", ""), ("expected_content", None),
                             ("checkpoint_path", "../escape"), ("forbidden_path", "/escape"),
                             ("match", "guess"), ("expected_content", "{")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                tasks, script, row = self.setup_case(root)
                row["recovery"][field] = value
                tasks.write_text(json.dumps(row) + "\n")
                code, _, stderr = run_cli("run-agent", "--agent", "vibe", "--tasks", tasks,
                                           "--runs", root / "runs", "--vibe-cmd", str(script))
                self.assertEqual(code, 1)
                self.assertIn("recovery", stderr)
                self.assertFalse((root / "runs").exists())

    def test_case_changes_design_fingerprints_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, row = self.setup_case(Path(temporary))
            task = sb.PreparedTask.from_row(row)
            encoded = task.harness_record()
            self.assertEqual(sb.PreparedTask.from_row(encoded).recovery.expected_content, b'{"phase":1}')
            first = (sb.answer_case_input_fingerprint(row, task),
                     sb.answer_task_fingerprint(row, task, "fake-model"))
            row["recovery"]["refusal_prompt"] = "another refusal"
            changed = sb.PreparedTask.from_row(row)
            self.assertNotEqual(first[0], sb.answer_case_input_fingerprint(row, changed))
            self.assertNotEqual(first[1], sb.answer_task_fingerprint(row, changed, "fake-model"))

    def test_occupied_artifacts_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, base = self.run_case(root)
            self.assertEqual(result[0], 0, result[2])
            original = (base / "recovery.json").read_bytes()
            result, _ = self.run_case(root)
            self.assertEqual(result[0], 1)
            self.assertIn("artifacts already exist", result[2])
            self.assertEqual((base / "recovery.json").read_bytes(), original)

    def test_observer_exception_stops_group_and_retains_raw_capture(self):
        original = sb.recovery_file_bytes
        def failing_observer(workspace, relative):
            if relative == "checkpoint.json" and (workspace / relative).exists():
                raise OSError("observer broke")
            return original(workspace, relative)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb, "recovery_file_bytes", side_effect=failing_observer):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual(process["state"], "observer_failed")
            self.assertTrue(process["leader_reaped"])
            self.assertTrue(process["group_stopped"])
            self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(),
                             b"raw-secret\xff" + b"x" * 16000)

    def test_ordinary_public_row_keeps_one_shot_artifacts_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, base = self.run_case(Path(temporary), mode="one-shot", recovery=False)
            self.assertEqual(result[0], 0, result[2])
            self.assertIn("answer INITIAL", (base / "output.md").read_text())
            metadata = json.loads((base / "metadata.json").read_text())
            self.assertEqual((metadata["provider"], metadata["returncode"]), ("vibe", 0))
            self.assertEqual(metadata["effort"], {"requested": None, "applied_by": "backend_default"})
            self.assertFalse((base / "recovery.json").exists())
            self.assertTrue((base / "workspace-changes.json").exists())

    def test_missing_case_fields_and_null_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "recovery must be an object"):
            RecoveryCase.parse(None)
        with self.assertRaisesRegex(ValueError, "recovery requires"):
            RecoveryCase.parse({})

    def test_deadline_wins_a_matching_checkpoint_before_signal(self):
        clock = sb.time.monotonic
        read = sb.recovery_file_bytes
        expired = False
        def checkpoint_at_deadline(workspace, relative):
            nonlocal expired
            content = read(workspace, relative)
            if relative == "checkpoint.json":
                expired = True
            return content
        def deadline_clock():
            return clock() + (100 if expired else 0)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb, "recovery_file_bytes", side_effect=checkpoint_at_deadline), mock.patch.object(
                sb.time, "monotonic", side_effect=deadline_clock):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual(process["state"], "deadline")
            self.assertIsNone(process["signal_sent"])
            self.assertEqual(process["compatibility_returncode"], 124)
            self.assertEqual(process["os_returncode"], -signal.SIGKILL)
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(len(record["phases"]), 1)

    def test_capture_exception_keeps_other_raw_files_and_stops_phases(self):
        write = Path.write_bytes
        def fail_stdout(path, content):
            if path.name == "stdout.bin":
                raise OSError("capture storage failed")
            return write(path, content)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                Path, "write_bytes", autospec=True, side_effect=fail_stdout):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual(process["state"], "capture_failed")
            self.assertTrue(process["group_stopped"])
            self.assertTrue(process["leader_reaped"])
            self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(),
                             b"raw-secret\xff" + b"x" * 16000)
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual((record["failure"], len(record["phases"])), ("capture_failed", 1))

    def test_artifact_failures_keep_process_facts_and_outcomes_consistent(self):
        write_json = sb.write_json
        write_bytes = Path.write_bytes
        invoke = sb.invoke_argv_with_timeout
        for mode, os_returncode, timed_out in (("stop", -signal.SIGTERM, False),
                                              ("timeout", -signal.SIGKILL, True),
                                              ("exit-124", 124, False)):
            for artifact in ("process.json", "stdout.bin", None):
                with self.subTest(mode=mode, artifact=artifact), tempfile.TemporaryDirectory() as temporary:
                    outcomes = []
                    def observe_outcome(plan, outcomes=outcomes):
                        outcome = invoke(plan)
                        outcomes.append(outcome)
                        return outcome
                    def fail_process(path, content, artifact=artifact):
                        if path.name == artifact:
                            raise OSError("process storage failed")
                        return write_json(path, content)
                    def fail_raw(path, content, artifact=artifact):
                        if path.name == artifact:
                            raise OSError("raw storage failed")
                        return write_bytes(path, content)
                    with mock.patch.object(sb, "write_json", side_effect=fail_process), \
                            mock.patch.object(Path, "write_bytes", autospec=True, side_effect=fail_raw), \
                            mock.patch.object(sb, "invoke_argv_with_timeout", side_effect=observe_outcome):
                        result, base = self.run_case(Path(temporary), mode=mode, timeout=1)
                    expected_state = ("capture_failed" if artifact else
                                      {"stop": "checkpoint_stop", "timeout": "deadline",
                                       "exit-124": "natural_completion"}[mode])
                    record = json.loads((base / "recovery.json").read_text())
                    phase = record["phases"][0]
                    facts = phase["adapter_environment"]["recovery_process"]
                    self.assertEqual(phase["state"], expected_state)
                    self.assertEqual(facts["state"], expected_state)
                    expected_error = ({"process.json": "OSError: process storage failed",
                                       "stdout.bin": "OSError: raw storage failed"}.get(artifact))
                    self.assertEqual(facts["error"], expected_error)
                    self.assertEqual(facts["os_returncode"], os_returncode)
                    compatibility_returncode = 124 if timed_out else os_returncode
                    self.assertEqual(facts["compatibility_returncode"], compatibility_returncode)
                    self.assertEqual(phase["compatibility_returncode"], compatibility_returncode)
                    self.assertEqual(outcomes[0].returncode, compatibility_returncode)
                    self.assertIs(outcomes[0].timed_out, timed_out)
                    self.assertEqual(dict(outcomes[0].metadata["recovery_process"]), facts)
                    self.assertEqual((facts["leader_reaped"], facts["pipes_drained"], facts["group_stopped"]),
                                     (True, True, True))
                    if artifact != "process.json":
                        self.assertEqual(json.loads((base / "recovery/initial/process.json").read_text()), facts)
                    else:
                        self.assertFalse((base / "recovery/initial/process.json").exists())
                    if artifact or mode != "stop":
                        self.assertEqual(result[0], 1, result[2])
                        self.assertEqual((record["status"], record["failure"], len(record["phases"])),
                                         ("failed", expected_state, 1))
                    else:
                        self.assertEqual(result[0], 0, result[2])
                        self.assertEqual(record["status"], "complete")
                    if mode != "exit-124":
                        self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(),
                                         b"raw-secret\xff" + b"x" * 16000)
                    self.assertFalse(Path(record["workspace"]).exists())

    def test_failed_signal_is_not_an_intentional_stop(self):
        killpg = os.killpg
        def fail_term(process_group, number):
            if number == signal.SIGTERM:
                raise PermissionError("signal refused")
            return killpg(process_group, number)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb.os, "killpg", side_effect=fail_term):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual((process["state"], process["os_returncode"]), ("signal_failed", -signal.SIGKILL))
            self.assertTrue(process["group_stopped"])
            self.assertEqual(json.loads((base / "recovery.json").read_text())["failure"], "signal_failed")

    def test_exit_before_checkpoint_signal_blocks_recovery_and_cleans_group(self):
        popen, killpg = sb.subprocess.Popen, os.killpg
        processes = {}
        signal_errors = []
        def spawn(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes[process.pid] = (process, Path(kwargs["cwd"]))
            return process
        def exit_before_term(process_group, number):
            if number == signal.SIGTERM:
                process, workspace = processes[process_group]
                (workspace / "release").touch()
                self.assertEqual(process.wait(timeout=2), 0)
            try:
                return killpg(process_group, number)
            except ProcessLookupError as exc:
                if number == signal.SIGTERM:
                    signal_errors.append(exc.errno)
                raise
        for descendant in (False, True):
            with self.subTest(descendant=descendant), tempfile.TemporaryDirectory() as temporary:
                processes.clear()
                signal_errors.clear()
                mode = "exit-before-signal-descendant" if descendant else "exit-before-signal"
                with mock.patch.object(sb.subprocess, "Popen", side_effect=spawn), \
                        mock.patch.object(sb.os, "killpg", side_effect=exit_before_term):
                    result, base = self.run_case(Path(temporary), mode=mode)
                self.assertEqual(signal_errors, [] if descendant else [errno.ESRCH])
                self.assertEqual(result[0], 1, result[2])
                process = json.loads((base / "recovery/initial/process.json").read_text())
                self.assertEqual(process["os_returncode"], 0)
                self.assertEqual(process["signal_sent"], signal.SIGTERM if descendant else None)
                self.assertEqual((process["leader_reaped"], process["pipes_drained"],
                                  process["group_stopped"], process["checkpoint_observed_live"]),
                                 (True, True, True, True))
                self.assertEqual(process["process_group_cleanup"]["status"],
                                 "kill_sent" if descendant else "not_needed")
                self.assertEqual(sb.recovery_group_stopped(process["process_group"])[0], True)
                record = json.loads((base / "recovery.json").read_text())
                self.assertEqual((record["status"], [phase["phase"] for phase in record["phases"]]),
                                 ("failed", ["initial"]))
                turns = [json.loads(line) for line in self.read_snapshot(
                    base / "recovery/final", "turns.jsonl").splitlines()]
                self.assertEqual([turn["phase"] for turn in turns], ["INITIAL"])
                self.assertEqual((process["state"], record["failure"]),
                                 ("natural_completion", "natural_completion"))
                self.assertIsNone(process["error"])

    def test_unconfirmed_group_cleanup_blocks_recovery(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb, "recovery_group_stopped", return_value=(False, "observation_unavailable")):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual(process["state"], "cleanup_failed")
            self.assertTrue(process["leader_reaped"])
            self.assertFalse(process["group_stopped"])
            self.assertEqual(json.loads((base / "recovery.json").read_text())["failure"], "cleanup_failed")

    def test_preexisting_checkpoint_never_spawns_initial_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks, script, row = self.setup_case(root)
            row["recovery"]["checkpoint_path"] = "inputs/fixture.bin"
            tasks.write_text(json.dumps(row) + "\n")
            result = run_cli("run-agent", "--agent", "vibe", "--tasks", tasks,
                             "--runs", root / "runs", "--vibe-cmd", f"{sys.executable} {script}")
            self.assertEqual(result[0], 1, result[2])
            base = root / "runs" / row["run_dir"]
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(record["failure"], "preexisting_checkpoint")
            self.assertEqual(record["phases"], [])
            self.assertEqual(self.read_snapshot(base / "recovery/final", "inputs/fixture.bin"),
                             b"fixture\x00\xff\r\n")

    def test_snapshot_error_retains_process_evidence_and_partial_manifest(self):
        capture = sb.snapshot_recovery_workspace
        def fail_after(workspace, destination):
            capture(workspace, destination)
            if destination.name == "after":
                raise OSError("snapshot failed after capture")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb, "snapshot_recovery_workspace", side_effect=fail_after):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            record = json.loads((base / "recovery.json").read_text())
            self.assertIn("snapshot failed after capture", record["failure"])
            self.assertEqual(len(record["phases"]), 1)
            self.assertEqual(self.read_snapshot(base / "recovery/initial/after", "inputs/fixture.bin"),
                             b"fixture\x00\xff\r\n")
            self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(),
                             b"raw-secret\xff" + b"x" * 16000)

    def test_failed_recovery_process_never_starts_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, base = self.run_case(Path(temporary), mode="recovery-fails")
            self.assertEqual(result[0], 1, result[2])
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(record["failure"], "process_failed")
            self.assertEqual([phase["phase"] for phase in record["phases"]], ["initial", "recovery"])
            process = json.loads((base / "recovery/recovery/process.json").read_text())
            self.assertEqual(process["os_returncode"], 7)
            self.assertIn(b"failed recovery", (base / "recovery/recovery/stdout.bin").read_bytes())

    def test_adapter_exception_retains_stopped_process_evidence(self):
        invoke = sb.VibeBackend.invoke_answer
        def fail_parse(backend, request, **options):
            invoke(backend, request, **options)
            raise RuntimeError("adapter decoding failed")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb.VibeBackend, "invoke_answer", autospec=True, side_effect=fail_parse):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            record = json.loads((base / "recovery.json").read_text())
            self.assertIn("adapter decoding failed", record["failure"])
            self.assertEqual([phase["state"] for phase in record["phases"]], ["checkpoint_stop"])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual(process["os_returncode"], -signal.SIGTERM)
            self.assertTrue(process["group_stopped"])

    def test_exact_byte_checkpoint_match_is_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, base = self.run_case(Path(temporary), mode="bytes", match="bytes")
            self.assertEqual(result[0], 0, result[2])
            self.assertEqual((base / "recovery/initial/checkpoint-observed.bin").read_bytes(), b'{"phase":1}')

    def test_semantic_json_checkpoint_accepts_equivalent_numeric_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, base = self.run_case(Path(temporary), mode="number")
            self.assertEqual(result[0], 0, result[2])
            self.assertEqual((base / "recovery/initial/checkpoint-observed.bin").read_bytes(), b'{"phase":1.0}')

    def test_pipe_capture_exception_still_drains_and_reaps_real_child(self):
        communicate = sb.subprocess.Popen.communicate
        read = sb.recovery_file_bytes
        ready = False
        injected = False
        def observe_ready(workspace, relative):
            nonlocal ready
            content = read(workspace, relative)
            if relative == "checkpoint.json" and content == b'{ "phase": 1 }\n':
                ready = True
            return content
        def fail_read(process, *arguments, **keywords):
            nonlocal injected
            if ready and not injected:
                injected = True
                raise OSError("pipe read failed")
            return communicate(process, *arguments, **keywords)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb.subprocess.Popen, "communicate", autospec=True, side_effect=fail_read), \
                mock.patch.object(sb, "recovery_file_bytes", side_effect=observe_ready):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            self.assertTrue(injected)
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual(process["state"], "capture_failed")
            self.assertEqual(process["error"], "OSError: pipe read failed")
            self.assertTrue(process["leader_reaped"])
            self.assertTrue(process["group_stopped"])
            self.assertEqual((base / "recovery/initial/stdout.bin").read_bytes(),
                             b'{"role": "assistant", "content": "initial"}\n' +
                             json.dumps({"raw": "z" * 16000}).encode() + b"\r\n")
            self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(),
                             b"raw-secret\xff" + b"x" * 16000)

    def test_wrapper_identity_uses_child_cwd_for_relative_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks, script, row = self.setup_case(root)
            script.write_text(f"#!{sys.executable}\n" + script.read_text())
            script.chmod(0o755)
            row["input_files"].append(str(script))
            tasks.write_text(json.dumps(row) + "\n")
            result = run_cli("run-agent", "--agent", "vibe", "--tasks", tasks,
                             "--runs", root / "runs", "--vibe-cmd", "./inputs/fake.py", "--timeout", "4")
            self.assertEqual(result[0], 0, result[2])
            base = root / "runs" / row["run_dir"]
            invocation = json.loads((base / "recovery/initial/invocation.json").read_text())
            self.assertEqual(invocation["client_argv"][0], "./inputs/fake.py")
            self.assertEqual(invocation["executable"], str(Path(invocation["cwd"]) / "inputs/fake.py"))
            self.assertEqual(invocation["executable_sha256"], hashlib.sha256(script.read_bytes()).hexdigest())
            self.assertIsNone(invocation["effective_exec_argv"])

    def test_final_snapshot_failure_does_not_report_completed_capability(self):
        capture = sb.snapshot_recovery_workspace
        def fail_final(workspace, destination):
            capture(workspace, destination)
            if destination.name == "final":
                raise OSError("final snapshot failed")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb, "snapshot_recovery_workspace", side_effect=fail_final):
            result, base = self.run_case(Path(temporary))
            self.assertEqual(result[0], 1, result[2])
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("final snapshot failed", record["failure"])
            self.assertEqual(len(record["phases"]), 3)
            self.assertIn(b"attempted_write", (base / "recovery/refusal/stdout.bin").read_bytes())
            self.assertFalse(Path(record["workspace"]).exists())

    def test_cancellation_reaps_group_and_retains_partial_evidence_before_reraising(self):
        read = sb.recovery_file_bytes
        def cancel_observer(workspace, relative):
            if relative == "checkpoint.json" and (workspace / relative).exists():
                raise KeyboardInterrupt("cancelled recovery")
            return read(workspace, relative)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
                sb, "recovery_file_bytes", side_effect=cancel_observer):
            root = Path(temporary)
            with self.assertRaisesRegex(KeyboardInterrupt, "cancelled recovery"):
                self.run_case(root)
            base = root / "runs/case-1/without_skill"
            record = json.loads((base / "recovery.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("KeyboardInterrupt", record["failure"])
            process = json.loads((base / "recovery/initial/process.json").read_text())
            self.assertEqual(process["state"], "observer_failed")
            self.assertTrue(process["leader_reaped"])
            self.assertTrue(process["group_stopped"])
            self.assertEqual((base / "recovery/initial/stderr.bin").read_bytes(),
                             b"raw-secret\xff" + b"x" * 16000)
            self.assertFalse(Path(record["workspace"]).exists())
