import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import skill_benchmark as sb
from invocation_contracts import (
    InvocationRequest,
    InvocationResult,
    InvocationState,
    ProcessInvocationPlan,
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

        with mock.patch.object(sb, "invoke_argv_with_timeout", side_effect=invoke):
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
