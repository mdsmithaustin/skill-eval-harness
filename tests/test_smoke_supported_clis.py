"""Offline contract tests for the opt-in supported-CLI live-smoke runner."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import ROOT, load_example_module

import skill_benchmark as sb
from agent_capabilities import AGENT_CAPABILITIES, SMOKE_TARGETS, SmokeTarget
from trigger_contracts import InvocationOutcome

SCRIPT = ROOT / "scripts" / "smoke_supported_clis.py"
smoke = load_example_module("smoke_supported_clis", "scripts/smoke_supported_clis.py")


class SupportedCliSmokeTests(unittest.TestCase):
    def test_minimal_disposable_manifest_has_one_answer_and_both_trigger_polarities(self):
        with tempfile.TemporaryDirectory() as td:
            manifest_path = smoke.make_smoke_repo(Path(td))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            answer = [case for case in manifest["cases"] if case["kind"] != "trigger"]
            triggers = [case for case in manifest["cases"] if case["kind"] == "trigger"]
            self.assertEqual(len(answer), 1)
            self.assertEqual(len(triggers), 2)
            self.assertEqual({case["id"] for case in triggers}, {"trig-pos-review-change", "trig-neg-unrelated-question"})
            self.assertTrue((manifest_path.parent.parent / "skills" / "demo" / "SKILL.md").exists())

    def test_smoke_targets_are_capability_qualified_and_own_model_defaults(self):
        expected = {
            "claude": ("SMOKE_CLAUDE_MODEL", "haiku", "answer"),
            "codex": ("SMOKE_CODEX_MODEL", "gpt-5.4-mini", "answer"),
            "gemini": ("SMOKE_GEMINI_MODEL", "gemini-2.5-flash", "answer"),
            "vibe": ("SMOKE_VIBE_MODEL", "devstral-small-latest", "answer"),
            "pi": ("SMOKE_PI_MODEL", "openai-codex/gpt-5.4-mini", "trigger"),
        }
        self.assertEqual(
            {name: (target.model_env, target.fallback_model, target.population)
             for name, target in SMOKE_TARGETS.items()},
            expected,
        )
        for name, target in SMOKE_TARGETS.items():
            capability = AGENT_CAPABILITIES[name]
            self.assertTrue(capability.answer_runner if target.population == "answer" else capability.autonomous_trigger)
            self.assertEqual(target.resolved_model({target.model_env: "  "}), target.fallback_model)
            self.assertEqual(target.resolved_model({target.model_env: "custom/model"}), "custom/model")
        # The CLI owns one --<agent>-model option per registry target, which
        # defaults to that target's environment-resolved model.
        argv = ["smoke_supported_clis.py", "--out-dir", "out"]
        with mock.patch.object(sys, "argv", argv):
            defaults = smoke.parse_args()
        for name, target in SMOKE_TARGETS.items():
            with self.subTest(target=name):
                self.assertEqual(getattr(defaults, f"{name}_model"), target.resolved_model(os.environ))
                with mock.patch.object(sys, "argv", [*argv, f"--{name}-model", "custom/model"]):
                    self.assertEqual(getattr(smoke.parse_args(), f"{name}_model"), "custom/model")

    def test_smoke_target_rejects_invalid_registry_states(self):
        for args in (("", "ENV", "model", "answer"), ("pi", "", "model", "trigger"),
                     ("pi", "ENV", "", "trigger"), ("pi", "ENV", "model", "other")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                SmokeTarget(*args)

    def test_answer_assessment_rejects_trace_zeros_when_trace_is_incomplete(self):
        def row(availability):
            trace_keys = ("tool_calls", "commands", "file_reads", "file_writes",
                          "errors", "retries", "repeated_command_max", "skill_invoked")
            return {
                "variant": "with_skill", "execution_valid": True,
                "missing_output": False, "objective_pass_rate": 1.0,
                "metadata": {
                    "observation_complete": True,
                    "trace_observation_complete": False,
                    "telemetry": {
                        "schema_version": 3,
                        "observation_evidence": {
                            "schema_version": 1,
                            "process": {"state": "complete"},
                            "provider_response": {"state": "complete"},
                            "trace": {"state": "incomplete"},
                            "artifact_set": {"state": "complete"},
                            "operation_evidence_complete": False,
                        },
                        "measurements": {
                            key: {"availability": availability} for key in trace_keys
                        },
                    },
                },
            }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "benchmark.json"
            for availability, expected in (("available", False), ("unavailable", True)):
                with self.subTest(availability=availability):
                    path.write_text(json.dumps({"results": [row(availability)]}), encoding="utf-8")
                    report = {"checks": []}
                    self.assertIs(smoke.assess_answer_benchmark(path, "claude", report), expected)
                    telemetry = next(check for check in report["checks"]
                                     if check["label"] == "claude:telemetry-contract")
                    self.assertIs(telemetry["passed"], expected)

    def test_trigger_assessment_requires_the_exact_positive_and_negative_fixture_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pi-trigger.json"
            report = {"checks": []}
            path.write_text(json.dumps({"results": [{
                "query": smoke.SMOKE_TRIGGER_EXPECTATIONS[0][0], "should_trigger": True,
                "observation_complete": True, "returncode": 0, "pass": True,
            }]}), encoding="utf-8")
            self.assertFalse(smoke.assess_trigger_report(path, report))
            self.assertFalse(report["checks"][0]["passed"])

    def test_gemini_live_smoke_requires_recorded_cli_version(self):
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / "runs"
            environment = runs / "case" / "with_skill" / "run-1" / "environment.json"
            environment.parent.mkdir(parents=True)
            report = {"checks": []}
            environment.write_text(json.dumps({
                "gemini_cli_version_status": "unavailable",
            }), encoding="utf-8")
            self.assertFalse(smoke.assess_gemini_version_evidence(runs, report))
            environment.write_text(json.dumps({
                "gemini_cli_version_status": "reported",
                "gemini_cli_version": "0.55.0-test",
            }), encoding="utf-8")
            report = {"checks": []}
            self.assertTrue(smoke.assess_gemini_version_evidence(runs, report))
            self.assertEqual(report["cli_versions"]["gemini"], ["0.55.0-test"])

    def test_trigger_assessment_rejects_a_persisted_pass_that_contradicts_provider_failure(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pi-trigger.json"
            rows = []
            for query, should_trigger in smoke.SMOKE_TRIGGER_EXPECTATIONS:
                rows.append({
                    "agent": "pi", "model": "bad", "query": query,
                    "should_trigger": should_trigger, "triggered": should_trigger,
                    "pass": True, "observation_complete": True,
                    "returncode": 0, "timed_out": False, "elapsed_ms": 1,
                    "evidence": ["/tmp/skills/demo/SKILL.md"] if should_trigger else [],
                    "usage_normalized": {"source": "missing"},
                    "cost_normalized": {"source": "missing"},
                    "stderr": "", "provider_error": "provider rejected model",
                })
            path.write_text(json.dumps({"results": rows}), encoding="utf-8")
            report = {"checks": []}
            self.assertFalse(smoke.assess_trigger_report(path, report))
            self.assertFalse(report["checks"][0]["passed"])

    def test_trigger_assessment_turns_malformed_rows_into_a_failed_check(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad-trigger.json"
            path.write_text(json.dumps({"results": [None, None]}), encoding="utf-8")
            report = {"checks": []}
            self.assertFalse(smoke.assess_trigger_report(path, report))
            self.assertFalse(report["checks"][0]["passed"])

    def test_failed_prepare_short_circuits_before_any_answer_call(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [str(SCRIPT), "--out-dir", str(Path(td) / "out"), "--live", "--agents", "claude",
                    "--claude-model", "haiku", "--timeout", "1"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(smoke.shutil, "which", return_value="/mock/claude"), \
                 mock.patch.object(smoke, "run", return_value=False) as run:
                self.assertEqual(smoke.main(), 1)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs["label"], "claude:prepare")

    def test_smoke_subprocess_result_is_derived_from_typed_invocation_state(self):
        outcomes = [
            (InvocationOutcome.from_process(stdout="x" * 5000, stderr="y" * 5000, returncode=0, elapsed_ms=1234), True),
            (InvocationOutcome.from_process(stdout="", stderr="bad", returncode=1, elapsed_ms=2), False),
            (InvocationOutcome.from_process(stdout="", stderr="timeout", returncode=124, elapsed_ms=3), False),
            (InvocationOutcome.from_process(stdout="", stderr="missing", returncode=127, elapsed_ms=4), False),
        ]
        for outcome, expected in outcomes:
            with self.subTest(state=outcome.state), mock.patch.object(
                smoke, "invoke_argv_with_timeout", return_value=outcome,
            ):
                report = {"commands": []}
                self.assertIs(smoke.run(["cmd"], cwd=ROOT, report=report, label="x"), expected)
                entry = report["commands"][0]
                self.assertEqual(entry["state"], outcome.state.value)
                self.assertEqual(entry["returncode"], outcome.returncode)
                self.assertEqual(entry["elapsed_seconds"], round(outcome.elapsed_ms / 1000, 3))
                self.assertLessEqual(len(entry["stdout"]), 4000)
                self.assertLessEqual(len(entry["stderr"]), 4000)

    def test_registry_population_dispatches_pi_to_trigger_runner(self):
        with tempfile.TemporaryDirectory() as td:
            argv = [str(SCRIPT), "--out-dir", str(Path(td) / "out"), "--live", "--agents", "pi",
                    "--pi-model", "model", "--timeout", "1"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(smoke.shutil, "which", return_value="/mock/pi"), \
                 mock.patch.object(smoke, "run", return_value=True) as run, \
                 mock.patch.object(smoke, "assess_trigger_report", return_value=True):
                self.assertEqual(smoke.main(), 0)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs["label"], "pi:trigger")
            self.assertIn("run_trigger_matrix.py", run.call_args.args[0][1])

    def test_registry_dispatch_supports_a_non_pi_trigger_target(self):
        synthetic = SmokeTarget("other", "SMOKE_OTHER_MODEL", "cheap", "trigger")
        with tempfile.TemporaryDirectory() as td:
            argv = [str(SCRIPT), "--out-dir", str(Path(td) / "out"), "--live", "--agents", "other",
                    "--other-model", "cheap", "--timeout", "1"]
            with mock.patch.object(smoke, "SMOKE_TARGETS", {"other": synthetic}), \
                 mock.patch.dict(smoke.DEFAULT_MODELS, {"other": "unused-default"}), \
                 mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(smoke.shutil, "which", return_value="/mock/other"), \
                 mock.patch.object(smoke, "run", return_value=True) as run, \
                 mock.patch.object(smoke, "assess_trigger_report", return_value=True) as assess:
                self.assertEqual(smoke.main(), 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--agent") + 1], "other")
            self.assertEqual(command[command.index("--model") + 1], "cheap")
            self.assertEqual(run.call_args.kwargs["label"], "other:trigger")
            self.assertEqual(assess.call_args.args[2], "other")

    def test_live_acknowledgement_is_required_before_any_cli_call(self):
        with tempfile.TemporaryDirectory() as td:
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--out-dir", str(Path(td) / "out")],
                text=True, capture_output=True, check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--live", completed.stderr)

    def test_live_smoke_rejects_an_empty_agent_selection(self):
        with tempfile.TemporaryDirectory() as td:
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--live", "--agents", " , ", "--out-dir", str(Path(td) / "out")],
                text=True, capture_output=True, check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("at least one", completed.stderr)


class PermissionEditSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="permission-edit-assessment-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.directory = Path(cls.tmp.name)
        cls.runs = cls.directory / "runs"
        tasks = cls.directory / "tasks.jsonl"
        demo = ROOT / "examples/edited-file-demo"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
        for args in (
            ["prepare", str(demo / "evals/shared-benchmark.json"), "--out", str(tasks)],
            ["run-codex", "--tasks", str(tasks), "--runs", str(cls.runs), "--codex-cmd",
             shlex.join([sys.executable, "-B", str(demo / "stub_runner.py")])],
        ):
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "skill_benchmark.py"), *args],
                                       cwd=ROOT, env=env, text=True, capture_output=True, check=False, timeout=60)
            if completed.returncode != 0:
                raise AssertionError(f"{args[0]} exited {completed.returncode}\n{completed.stderr}")
        cls.treatment = cls.runs / "normalize-name/with_skill"

    def copied_runs(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        runs = Path(tmp.name) / "runs"
        shutil.copytree(self.runs, runs)
        return runs, runs / "normalize-name/with_skill"

    def test_permission_assessment_requires_the_verified_edit_and_native_command(self):
        report = {"checks": []}
        self.assertTrue(smoke.assess_permission_edit(self.runs, report))
        self.assertEqual([check["passed"] for check in report["checks"]], [True])

    def test_permission_assessment_accepts_native_shell_test_command(self):
        for shell in ("/bin/zsh", "/bin/bash", "/bin/sh", "zsh", "bash", "sh"):
            for option in ("-lc", "-c"):
                command = f"{shell} {option} 'python3 -B inputs/test_name_tools.py'"
                with self.subTest(command=command):
                    runs, run_dir = self.copied_runs()
                    path = run_dir / "events.json"
                    trace = json.loads(path.read_text())
                    event = next(event for event in trace["events"]
                                 if event["type"] == "command" and event["status"] == "completed")
                    event["input_summary"] = command
                    path.write_text(json.dumps(trace))
                    sb.write_artifact_commit(run_dir)
                    report = {"checks": []}
                    self.assertTrue(smoke.assess_permission_edit(runs, report))
                    self.assertEqual([check["passed"] for check in report["checks"]], [True])

    def test_permission_assessment_rejects_shell_commands_outside_exact_test_contract(self):
        commands = (
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py; true'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py && true'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py || true'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py | cat'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py & wait'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py > result.txt'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py\ntrue'",
            "/bin/zsh -lc 'python3\n-B inputs/test_name_tools.py'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py'\ntrue",
            "python3\n-B inputs/test_name_tools.py",
            "/bin/zsh -lc 'python3 -B $(echo inputs/test_name_tools.py)'",
            "/bin/zsh -lc 'python3 -B `echo inputs/test_name_tools.py`'",
            "/bin/zsh -lc 'python3 -B ${TEST_FILE}'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py extra'",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py' extra",
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py' ; true",
            "/bin/zsh -x -lc 'python3 -B inputs/test_name_tools.py'",
            "/bin/zsh -ic 'python3 -B inputs/test_name_tools.py'",
            "/bin/fish -c 'python3 -B inputs/test_name_tools.py'",
            "/custom/bin/zsh -lc 'python3 -B inputs/test_name_tools.py'",
            '/bin/zsh -lc "sh -c \'python3 -B inputs/test_name_tools.py\'"',
            "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py",
            "/bin/zsh -lc 'python3 -B -c print(1)'",
        )
        for command in commands:
            with self.subTest(command=command):
                runs, run_dir = self.copied_runs()
                path = run_dir / "events.json"
                trace = json.loads(path.read_text())
                event = next(event for event in trace["events"]
                             if event["type"] == "command" and event["status"] == "completed")
                event["input_summary"] = command
                path.write_text(json.dumps(trace))
                sb.write_artifact_commit(run_dir)
                report = {"checks": []}
                self.assertFalse(smoke.assess_permission_edit(runs, report))
                self.assertEqual([check["passed"] for check in report["checks"]], [False])

    def test_permission_assessment_requires_successful_completed_shell_test_command(self):
        mutations = (
            {"exit_code": 1}, {"exit_code": None}, {"exit_code": False},
            {"status": "in_progress"}, {"status": "failed"},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                runs, run_dir = self.copied_runs()
                path = run_dir / "events.json"
                trace = json.loads(path.read_text())
                event = next(event for event in trace["events"]
                             if event["type"] == "command" and event["status"] == "completed")
                event["input_summary"] = "/bin/zsh -lc 'python3 -B inputs/test_name_tools.py'"
                event.update(mutation)
                path.write_text(json.dumps(trace))
                sb.write_artifact_commit(run_dir)
                self.assertFalse(smoke.assess_permission_edit(runs, {"checks": []}))

    def test_permission_assessment_rejects_denied_partial_missing_exit_and_false_claims(self):
        mutations = {
            "denied": lambda event: event.update(exit_code=1),
            "missing-exit": lambda event: event.pop("exit_code"),
            "boolean-exit": lambda event: event.update(exit_code=False),
            "started": lambda event: event.update(status="in_progress"),
            "failed": lambda event: event.update(status="failed"),
            "unrelated": lambda event: event.update(input_summary="python3 -B -c 'print(1)'"),
            "extra-command": lambda event: event.update(input_summary="python3 -B inputs/test_name_tools.py; true"),
            "assistant-claim": lambda event: event.update(type="message", role="assistant"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                runs, run_dir = self.copied_runs()
                path = run_dir / "events.json"
                trace = json.loads(path.read_text())
                event = next(event for event in trace["events"]
                             if event["type"] == "command" and event["status"] == "completed")
                mutate(event)
                path.write_text(json.dumps(trace))
                sb.write_artifact_commit(run_dir)
                self.assertFalse(smoke.assess_permission_edit(runs, {"checks": []}))

    def test_permission_assessment_rejects_an_incomplete_observation(self):
        runs, run_dir = self.copied_runs()
        metadata = json.loads((run_dir / "metadata.json").read_text())
        metadata["trace_observation_complete"] = False
        (run_dir / "metadata.json").write_text(json.dumps(metadata))
        sb.write_artifact_commit(run_dir)
        self.assertFalse(smoke.assess_permission_edit(runs, {"checks": []}))

    def test_permission_assessment_rejects_native_command_success_without_an_edit(self):
        runs, run_dir = self.copied_runs()
        baseline = runs / "normalize-name/without_skill"
        (run_dir / "workspace-changes.json").write_bytes((baseline / "workspace-changes.json").read_bytes())
        (run_dir / "candidate.patch").unlink()
        sb.write_artifact_commit(run_dir)
        self.assertFalse(smoke.assess_permission_edit(runs, {"checks": []}))

    def test_permission_assessment_rejects_missing_treatment(self):
        self.assertFalse(smoke.assess_permission_edit(self.directory / "absent", {"checks": []}))

    def test_permission_mode_requires_live_and_exactly_codex_before_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flags in (["--permission-edit", "--agents", "codex"],
                          ["--live", "--permission-edit", "--agents", "claude"],
                          ["--live", "--permission-edit", "--agents", "codex,claude"],
                          ["--live", "--permission-edit", "--agents", "codex,codex"]):
                with self.subTest(flags=flags):
                    completed = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", tmp, *flags],
                                               text=True, capture_output=True, check=False)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_permission_mode_records_the_explicit_workspace_write_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(out_dir=tmp, live=True, permission_edit=True, agents="codex", timeout=1,
                                      **{f"{agent}_model": "test-model" for agent in SMOKE_TARGETS})
            with mock.patch.object(smoke, "parse_args", return_value=args), \
                 mock.patch.object(smoke.shutil, "which", return_value="/mock/codex"), \
                 mock.patch.object(smoke, "run", return_value=True) as run, \
                 mock.patch.object(smoke, "assess_permission_edit", return_value=True):
                self.assertEqual(smoke.main(), 0)
            commands = [call.args[0] for call in run.call_args_list]
            answer = next(command for command in commands if "run-agent" in command)
            self.assertEqual(shlex.split(answer[answer.index("--codex-cmd") + 1]),
                             ["/mock/codex", "exec", "--sandbox", "workspace-write"])
            benchmark = next(command for command in commands if "benchmark" in command)
            self.assertIn("--allow-scripts", benchmark)
            report = json.loads((Path(tmp) / "smoke.json").read_text())
            self.assertEqual(report["permission_policy"], {
                "agent": "codex", "sandbox": "workspace-write",
                "command_prefix": ["/mock/codex", "exec", "--sandbox", "workspace-write"],
            })


if __name__ == "__main__":
    unittest.main()
