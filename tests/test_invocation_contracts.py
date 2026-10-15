import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import process_runtime as pr
import skill_benchmark as sb
from invocation_contracts import (
    InvocationRequest,
    InvocationResult,
    InvocationState,
    ProcessInvocationPlan,
    RawProcessCapture,
    RecoveryCapture,
    TimeoutSeconds,
)
from manifest_contracts import ModelId
from trigger_contracts import InvocationOutcome


class InvocationPlanContractTests(unittest.TestCase):
    def test_answer_request_parses_precise_model_and_timeout_values(self):
        request = InvocationRequest.parse(
            prompt="do the task",
            workspace=Path("workspace"),
            model="model-a",
            timeout_s=30,
        )
        self.assertIsInstance(request.model, ModelId)
        self.assertIsInstance(request.timeout_s, TimeoutSeconds)
        self.assertEqual(request.model, "model-a")
        self.assertEqual(request.timeout_s, 30)

    def test_process_plan_freezes_every_spawn_input(self):
        environment = {"TOKEN": "secret"}
        plan = ProcessInvocationPlan.from_values(
            ["provider", "--json"],
            input_text="prompt",
            cwd="workspace",
            timeout_s=20,
            environment=environment,
        )
        environment["TOKEN"] = "changed"
        self.assertEqual(plan.argv, ("provider", "--json"))
        self.assertEqual(plan.cwd, Path("workspace"))
        self.assertEqual(plan.environment, {"TOKEN": "secret"})
        with self.assertRaises(TypeError):
            plan.environment["NEW"] = "value"  # type: ignore[index]

    def test_legacy_wrapper_preserves_inherited_stdin(self):
        observed: list[ProcessInvocationPlan] = []

        def invoke(plan: ProcessInvocationPlan) -> InvocationOutcome:
            observed.append(plan)
            return InvocationOutcome.from_process(
                stdout="", stderr="", returncode=0, elapsed_ms=0)

        with mock.patch.object(pr, "invoke_argv_with_timeout", side_effect=invoke):
            sb.run_argv_with_timeout(["provider"], timeout=1, input_text=None)

        self.assertIsNone(observed[0].input_text)

    def test_process_plan_rejects_values_that_cannot_be_spawned(self):
        invalid = [
            ([], {}, 1),
            ("provider", {}, 1),
            ([""], {}, 1),
            (["provider\x00bad"], {}, 1),
            (["provider"], {"BAD=KEY": "value"}, 1),
            (["provider"], {"KEY": "bad\x00value"}, 1),
            (["provider"], {}, 0),
            (["provider"], {}, True),
        ]
        for argv, environment, timeout in invalid:
            with self.subTest(argv=argv, environment=environment, timeout=timeout), \
                    self.assertRaises((TypeError, ValueError)):
                ProcessInvocationPlan.from_values(
                    argv,
                    input_text="",
                    cwd=Path("workspace"),
                    timeout_s=timeout,
                    environment=environment,
                )

    def test_process_result_closes_lifecycle_state(self):
        completed = InvocationResult(
            stdout="answer",
            stderr="",
            returncode=0,
            elapsed_ms=12,
            invocation_state=InvocationState.COMPLETE,
            stdout_utf8_valid=True,
            stderr_utf8_valid=True,
        )
        self.assertFalse(completed.timed_out)
        with self.assertRaisesRegex(ValueError, "requires returncode"):
            InvocationResult(
                stdout="",
                stderr="",
                returncode=0,
                elapsed_ms=12,
                invocation_state=InvocationState.TIMED_OUT,
                stdout_utf8_valid=True,
                stderr_utf8_valid=True,
                timed_out=False,
            )

        with self.assertRaisesRegex(ValueError, "contradicts stdout_utf8_valid"):
            InvocationResult(
                stdout="",
                stderr="",
                returncode=0,
                elapsed_ms=12,
                invocation_state=InvocationState.COMPLETE,
                stdout_utf8_valid=True,
                stderr_utf8_valid=True,
                adapter_metadata={"stdout_utf8_valid": False},
            )

    def test_subprocess_owner_accepts_only_a_validated_plan(self):
        with self.assertRaisesRegex(TypeError, "ProcessInvocationPlan"):
            sb.invoke_argv_with_timeout(["provider"])  # type: ignore[arg-type]

    def test_harness_reexports_the_contract_types(self):
        self.assertIs(sb.InvocationRequest, InvocationRequest)
        self.assertIs(sb.InvocationResult, InvocationResult)
        self.assertIs(sb.ProcessInvocationPlan, ProcessInvocationPlan)


class ReachedExitTests(unittest.TestCase):
    def test_only_a_process_that_exited_on_its_own_counts(self):
        # One definition for the trigger contract, the Pi trace writer and the
        # answer runners: a timeout, a spawn failure or a harness failure never
        # observed the provider exit.
        expected = {
            InvocationState.COMPLETE: True,
            InvocationState.PROCESS_FAILED: True,
            InvocationState.PROVIDER_FAILED: True,
            InvocationState.TIMED_OUT: False,
            InvocationState.SPAWN_FAILED: False,
            InvocationState.HARNESS_FAILED: False,
        }
        self.assertEqual(set(expected), set(InvocationState))
        for state, reached in expected.items():
            with self.subTest(state=state):
                self.assertIs(state.reached_exit, reached)
        outcome = InvocationOutcome.from_process(stdout="", stderr="", returncode=0, elapsed_ms=0)
        self.assertIs(outcome.process_observation_complete, outcome.state.reached_exit)


class ProcessCaptureTests(unittest.TestCase):
    def test_delayed_reader_receives_complete_utf8_prompt_and_eof(self):
        prompt = "Καλημέρα 🌱 café\n" * 32768
        for recovery, exec_wrapper in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(recovery=recovery, exec_wrapper=exec_wrapper), \
                    tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                evidence = root / "evidence"
                evidence.mkdir()
                argv = [sys.executable, "-c",
                     ("import hashlib, json, sys, time; time.sleep(0.2); "
                     "data = sys.stdin.buffer.read(); "
                     "print(json.dumps([len(data), hashlib.sha256(data).hexdigest()]))")]
                if exec_wrapper:
                    argv = [sys.executable, "-c",
                            ("import os, sys, time; time.sleep(0.2); "
                             "os.execv(sys.executable, sys.argv[1:])"), *argv]
                outcome = pr.invoke_argv_with_timeout(ProcessInvocationPlan.from_values(argv,
                    input_text=prompt, cwd=root, timeout_s=2,
                    recovery_capture=RecoveryCapture(evidence) if recovery else None))
                self.assertEqual(outcome.returncode, 0, outcome.stderr)
                self.assertFalse(outcome.timed_out)
                self.assertEqual(json.loads(outcome.stdout), [
                    917504, "38a798d25dc73830fc5fbbc7acec36e9380be47b6fdfcdb4e5e6861e0deda748"])

    def test_empty_stdin_reaches_eof(self):
        with tempfile.TemporaryDirectory() as temporary:
            outcome = pr.invoke_argv_with_timeout(ProcessInvocationPlan.from_values(
                [sys.executable, "-c", "import sys; print(repr(sys.stdin.buffer.read()))"],
                input_text="", cwd=temporary, timeout_s=2))
        self.assertEqual((outcome.returncode, outcome.stdout, outcome.stderr), (0, "b''\n", ""))

    def test_delayed_input_does_not_block_large_output_capture(self):
        capture = RawProcessCapture()
        with tempfile.TemporaryDirectory() as temporary:
            outcome = pr.invoke_argv_with_timeout(ProcessInvocationPlan.from_values(
                [sys.executable, "-c",
                 ("import sys, time; sys.stdout.buffer.write(b'x' * 262144); sys.stdout.flush(); "
                 "sys.stderr.buffer.write(b'y' * 262144); sys.stderr.flush(); time.sleep(0.2); "
                 "data = sys.stdin.buffer.read(); print(len(data))")],
                input_text="z" * 917504, cwd=temporary, timeout_s=2, raw_capture=capture))
        self.assertEqual(outcome.returncode, 0, outcome.stderr)
        self.assertEqual(outcome.stdout, "x" * 262144 + "917504\n")
        self.assertEqual(outcome.stderr, "y" * 4000)
        self.assertEqual(capture.facts.stderr, b"y" * 262144)
        self.assertTrue(capture.facts.communication_complete)

    def test_nonreading_child_times_out_with_partial_output_and_is_reaped(self):
        capture = RawProcessCapture()
        with tempfile.TemporaryDirectory() as temporary:
            outcome = pr.invoke_argv_with_timeout(ProcessInvocationPlan.from_values(
                [sys.executable, "-c",
                 ("import os, sys, time; print(os.getpid(), flush=True); "
                 "print('started', file=sys.stderr, flush=True); time.sleep(30)")],
                input_text="z" * 917504, cwd=temporary, timeout_s=1, raw_capture=capture))
        self.assertEqual(outcome.returncode, 124)
        self.assertTrue(outcome.timed_out)
        self.assertEqual(outcome.stderr, "started\n")
        self.assertEqual(capture.facts.stdout, outcome.stdout.encode())
        self.assertEqual(capture.facts.stderr, b"started\n")
        pid = int(outcome.stdout)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        if hasattr(os, "killpg"):
            with self.assertRaises(ProcessLookupError):
                os.killpg(pid, 0)

    def test_natural_exit_after_poll_timeout_preserves_captured_stderr(self):
        communicate = sb.subprocess.Popen.communicate
        for stderr in ("", "provider stderr\n"):
            for returncode in (0, 7):
                with self.subTest(stderr=stderr, returncode=returncode), \
                        tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    release = root / "release"
                    child = root / "child.py"
                    child.write_text(
                        "import pathlib, sys, time\n"
                        f"while not pathlib.Path({str(release)!r}).exists():\n"
                        "    time.sleep(0.005)\n"
                        "sys.stdout.write('fixture stdout\\n')\n"
                        f"sys.stderr.write({stderr!r})\n"
                        f"raise SystemExit({returncode})\n", encoding="utf-8")
                    owned = []

                    def exit_after_poll(process, *arguments, owned=owned,
                                        release=release, **keywords):
                        try:
                            return communicate(process, *arguments, **keywords)
                        except sb.subprocess.TimeoutExpired:
                            if not owned:
                                owned.append(process)
                                release.touch()
                                process.wait(timeout=3)
                            raise

                    with mock.patch.object(sb.subprocess.Popen, "communicate",
                                           autospec=True, side_effect=exit_after_poll):
                        outcome = sb.invoke_argv_with_timeout(
                            ProcessInvocationPlan.from_values(
                                [sys.executable, str(child)], input_text=None,
                                cwd=root, timeout_s=5))

                    self.assertEqual(len(owned), 1)
                    self.assertEqual(owned[0].returncode, returncode)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(owned[0].pid, 0)
                    if hasattr(os, "killpg"):
                        with self.assertRaises(ProcessLookupError):
                            os.killpg(owned[0].pid, 0)
                    self.assertEqual(outcome.stdout, "fixture stdout\n")
                    self.assertEqual(outcome.returncode, returncode)
                    self.assertFalse(outcome.timed_out)
                    self.assertIs(outcome.state, InvocationState.COMPLETE if returncode == 0
                                  else InvocationState.PROCESS_FAILED)
                    self.assertTrue(outcome.process_observation_complete)
                    self.assertEqual(outcome.stderr, stderr)


if __name__ == "__main__":
    unittest.main()
