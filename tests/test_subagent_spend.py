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

    def fixture(self, root, *, turns=()):
        manifest, tasks, run_dir = write_with_skill_task(root)
        row = json.loads(tasks.read_text())
        if turns:
            row["turns"] = list(turns)
            tasks.write_text(json.dumps(row) + "\n")
        return manifest, tasks, root / "runs", run_dir

    def backend(self, root, route, outputs, *, exit_code=0, edit=False):
        marker = root / "launches"
        replies = root / "replies.json"
        replies.write_text(json.dumps(outputs))
        script = root / ("claude" if route == "claude" else "agent.py")
        script.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
text = sys.stdin.read()
request = {{"prompt": text}} if {route!r} == "claude" else json.loads(text)
marker = Path({str(marker)!r})
n = len(marker.read_text().splitlines()) if marker.exists() else 0
with marker.open("a") as handle:
    handle.write(json.dumps(request) + "\\n")
if {edit!r}:
    workspace = Path.cwd() if {route!r} == "claude" else Path(request["workspace"])
    (workspace / "candidate.txt").write_text("paid edit\\n")
sys.stdout.write(json.loads(Path({str(replies)!r}).read_text())[n])
sys.exit({exit_code})
''')
        script.chmod(0o755)
        return ("--claude-bin", script) if route == "claude" else ("--agent-cmd", f"{sys.executable} {script}")

    def ledgers(self, runs):
        return [json.loads(path.read_text()) for path in sorted((runs / "spend").glob("*/spend-ceiling.json"))]

    def invoke(self, tasks, runs, backend, *flags):
        return run_cli("run-subagent", "--tasks", tasks, "--runs", runs, *backend, *flags)

    def snapshot(self, base):
        return {str(path.relative_to(base)): path.read_bytes()
                for path in base.rglob("*") if path.is_file()}

    def assert_terminal(self, base, state):
        import ablation_model as am
        import skill_benchmark as sb
        from artifact_contracts import CompleteArtifactSet, observe_artifact_set
        metadata = json.loads((base / "metadata.json").read_text())
        body = (base / "output.md").read_text()
        self.assertEqual(metadata["artifact_terminal_state"], state)
        self.assertIsNone(metadata["returncode"])
        self.assertNotIn("invocation_state", metadata)
        self.assertFalse(metadata["provider_response_complete"])
        self.assertFalse(metadata["process_observation_complete"])
        self.assertFalse(am.execution_valid(metadata, body))
        self.assertIn("[CLAUDE FAILURE", body)
        self.assertIsInstance(observe_artifact_set(base, declared_contract_version=metadata["artifact_contract_version"]), CompleteArtifactSet)
        self.assertIsNone(am.metadata_lifecycle_error(sb._with_committed_artifact_state(base, metadata)))
        return metadata

    def test_zero_cap_both_routes_preserve_absent_and_prior_destinations(self):
        from helpers import claude_stream_records
        for route in ("claude", "shell"):
            for prior in ("absent", "committed", "uncommitted"):
                with self.subTest(route=route, prior=prior), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    _, tasks, runs, run_dir = self.fixture(root, turns=("one", "two"))
                    response = ("\n".join(json.dumps(row) for row in claude_stream_records(cost=0.06))
                                if route == "claude" else json.dumps({"answer": "prior answer", "usage": {"cost_usd": 0.06},
                                                                      "telemetry_scope": "turn_delta"}))
                    backend = self.backend(root, route, [response, response])
                    base = runs / run_dir
                    if prior == "committed":
                        self.assertEqual(self.invoke(tasks, runs, backend)[0], 0)
                        (root / "launches").unlink()
                    elif prior == "uncommitted":
                        base.mkdir(parents=True)
                        (base / "operator.txt").write_text("preserve me")
                        (base / "tool-replay.json").write_text("incompatible prior replay")
                        (base / "empty-directory").mkdir()
                        (base / "operator-link").symlink_to("operator.txt")
                    before = self.snapshot(base) if prior != "absent" else None
                    code, _, stderr = self.invoke(tasks, runs, backend, "--max-cost-usd", "0", "--tool-replay", "auto")
                    self.assertEqual(code, 2, stderr)
                    self.assertFalse((root / "launches").exists())
                    ledger = self.ledgers(runs)[0]
                    self.assertEqual([row["state"] for row in ledger["calls"]], ["not_started", "not_started"])
                    self.assertEqual([row["call"]["external_turn"] for row in ledger["calls"]], [1, 2])
                    if prior == "absent":
                        metadata = self.assert_terminal(base, "budget_stopped")
                        self.assertEqual(len(metadata["subagent_refusals"]), 2)
                        self.assertFalse(any(base.glob("turn-*")))
                    else:
                        self.assertEqual(self.snapshot(base), before)
                        if prior == "uncommitted":
                            self.assertTrue((base / "empty-directory").is_dir())
                            self.assertEqual((base / "operator-link").readlink(), Path("operator.txt"))

    def test_crossing_three_turns_retains_new_evidence_and_grades_incomplete(self):
        from helpers import claude_stream_records
        for route in ("claude", "shell"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                manifest, tasks, runs, run_dir = self.fixture(root, turns=("one", "two", "three"))
                response = ("\n".join(json.dumps(row) for row in claude_stream_records(cost=0.6))
                            if route == "claude" else json.dumps({"answer": "token-XYZ", "usage": {"cost_usd": 0.6},
                                                                  "telemetry_scope": "turn_delta"}))
                backend = self.backend(root, route, [response] * 3, edit=True)
                self.assertEqual(self.invoke(tasks, runs, backend)[0], 0)
                base = runs / run_dir
                (base / "unrelated-prior.txt").write_text("old")
                (root / "launches").unlink()
                code, _, stderr = self.invoke(tasks, runs, backend, "--max-cost-usd", "0.5", "--tool-replay", "record")
                self.assertEqual(code, 2, stderr)
                self.assertEqual(len((root / "launches").read_text().splitlines()), 1)
                metadata = self.assert_terminal(base, "budget_stopped")
                self.assertEqual(metadata["multi_turn_telemetry"]["attempted_turns"], 1)
                self.assertEqual(metadata["multi_turn_telemetry"]["cost"]["observed_delta_total"], 0.6)
                self.assertIsNone(metadata["cost_normalized"].get("total_cost"))
                self.assertTrue((base / "turn-1" / "artifact-commit.json").is_file())
                self.assertFalse((base / "turn-2").exists())
                self.assertFalse((base / "turn-3").exists())
                self.assertFalse((base / "unrelated-prior.txt").exists())
                self.assertIn("+paid edit", (base / "candidate.patch").read_text())
                changes = json.loads((base / "workspace-changes.json").read_text())
                self.assertEqual(changes["changes"][0]["path"], "candidate.txt")
                ledger = self.ledgers(runs)[0]
                self.assertEqual(ledger["spent_usd"], "0.6")
                self.assertEqual([row["state"] for row in ledger["calls"]], ["settled", "not_started", "not_started"])
                report = root / "report.json"
                code, _, stderr = run_cli("benchmark", manifest, "--runs", runs, "--out", report)
                self.assertEqual(code, 0, stderr)
                result = json.loads(report.read_text())
                row = next(row for row in result["results"] if row["variant"] == "with_skill")
                self.assertFalse(row["execution_valid"])
                self.assertEqual(result["spend_invocations"][0]["spent_usd"], "0.6")
                code, output, stderr = run_cli("cost-summary", "--manifest", manifest, "--runs", runs)
                self.assertEqual(code, 0, stderr)
                self.assertEqual(json.loads(output)["spend_invocations"][0]["spent_usd"], "0.6")

    def test_paid_failures_keep_safe_dollars_and_actual_process_codes(self):
        cases = (
            ("claude", {"type": "result", "is_error": True, "result": "API Error", "total_cost_usd": 0.06}, 0),
            ("claude", {"type": "result", "result": {"bad": "answer"}, "total_cost_usd": 0.07}, 0),
            ("shell", {"answer": "untrusted", "usage": {"cost_usd": 0.08}}, 7),
            ("shell", {"answer": 3, "usage": {"cost_usd": 0.09}}, 0),
            ("shell", {"answer": "untrusted", "trace": "bad", "usage": {"cost_usd": 0.1}}, 0),
            ("shell", {"answer": "untrusted", "usage": {"cost_usd": 0.11, "total_tokens": "bad"}}, 0),
        )
        for route, response, process_code in cases:
            with self.subTest(route=route, response=response), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, tasks, runs, run_dir = self.fixture(root)
                backend = self.backend(root, route, [json.dumps(response)], exit_code=process_code)
                code, _, stderr = self.invoke(tasks, runs, backend, "--max-cost-usd", "1")
                self.assertEqual(code, 0, stderr)
                cost = response.get("total_cost_usd", response.get("usage", {}).get("cost_usd"))
                ledger = self.ledgers(runs)[0]
                self.assertEqual(ledger["calls"][0]["charge"], {"basis": "observed", "amount_usd": str(cost),
                                                             "provenance": "provider_reported"})
                base = runs / run_dir
                metadata = json.loads((base / "metadata.json").read_text())
                self.assertEqual(metadata["returncode"], process_code)
                self.assertEqual(metadata["cost_normalized"]["total_cost"], cost)
                self.assertFalse(metadata["provider_response_complete"])
                self.assertEqual(metadata["subagent_rejected_calls"][0]["reported_cost_usd"], str(cost))
                self.assertIn("[CLAUDE FAILURE", (base / "output.md").read_text())
                self.assertEqual(metadata["subagent_rejected_calls"][0]["raw_response"], json.dumps(response))

    def test_scope_zero_unknown_and_assumed_charges_remain_distinct(self):
        cases = (
            ({"cost_usd": 0}, "turn_delta", None, ["settled", "settled"], "observed", "0"),
            ({}, "turn_delta", None, ["settled", "not_started"], "unpriced", "0"),
            ({}, "turn_delta", "0.2", ["settled", "settled"], "assumed", "0.4"),
            ({"cost_usd": 0.9}, "conversation_cumulative", None, ["settled", "not_started"], "unpriced", "0"),
            ({"cost_usd": 0.9}, None, None, ["settled", "not_started"], "unpriced", "0"),
            ({"cost_usd": 0.9}, "conversation_cumulative", "0.2", ["settled", "settled"], "assumed", "0.4"),
        )
        for usage, scope, assumption, states, basis, spent in cases:
            with self.subTest(usage=usage, scope=scope, assumption=assumption), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, tasks, runs, run_dir = self.fixture(root, turns=("one", "two"))
                response = {"answer": "token-XYZ", "usage": usage}
                if scope is not None:
                    response["telemetry_scope"] = scope
                backend = self.backend(root, "shell", [json.dumps(response)] * 2)
                flags = ("--assumed-cost-per-run-usd", assumption) if assumption is not None else ()
                code, _, stderr = self.invoke(tasks, runs, backend, "--max-cost-usd", "1", *flags)
                self.assertEqual(code, 2 if basis == "unpriced" else 0, stderr)
                ledger = self.ledgers(runs)[0]
                self.assertEqual([row["state"] for row in ledger["calls"]], states)
                self.assertEqual(ledger["calls"][0]["charge"]["basis"], basis)
                self.assertEqual(ledger["spent_usd"], spent)
                turn_meta = json.loads((runs / run_dir / "turn-1" / "metadata.json").read_text())
                self.assertEqual(turn_meta["cost_normalized"].get("total_cost"), usage.get("cost_usd"))
                if basis == "unpriced":
                    self.assert_terminal(runs / run_dir, "budget_stopped")

    def test_direct_rejection_retains_diagnostics_without_inventing_process_evidence(self):
        import skill_benchmark as sb
        from spend_contracts import SpendPolicy
        for scope, assumption, basis in (("turn_delta", None, "observed"),
                                         ("turn_delta", "0.2", "observed"),
                                         ("conversation_cumulative", "0.2", "assumed"),
                                         (None, None, "unpriced")):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, tasks, runs, run_dir = self.fixture(root, turns=("one", "two"))
                calls = []

                def agent(calls=calls, scope=scope, **kwargs):
                    calls.append(kwargs["prompt"])
                    return {"answer": 3, "usage": {"cost_usd": 0.9}, "telemetry_scope": scope, "returncode": 7}

                code = sb.run_subagent_tasks(sb.load_prepared_tasks(tasks), runs, agent,
                                             spend_policy=SpendPolicy.from_raw("0.1", assumption))
                self.assertEqual(code, 2)
                self.assertEqual(len(calls), 1)
                ledger = self.ledgers(runs)[0]
                self.assertEqual(ledger["calls"][0]["charge"]["basis"], basis)
                self.assertEqual(ledger["calls"][1]["state"], "not_started")
                metadata = self.assert_terminal(runs / run_dir, "budget_stopped")
                diagnostic = metadata["subagent_rejected_calls"][0]
                self.assertEqual(diagnostic["reported_cost_usd"], "0.9")
                self.assertEqual(diagnostic["telemetry_scope"], scope or "unavailable")
                self.assertFalse(any((runs / run_dir).glob("turn-*")))

        for raised in (False, True):
            with self.subTest(raised=raised), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, tasks, runs, run_dir = self.fixture(root)

                def agent(raised=raised, **kwargs):
                    if raised:
                        raise RuntimeError("opaque callback broke")
                    return {"answer": "invalid", "trace": "bad", "usage": {"cost_usd": 0.9}}

                code = sb.run_subagent_tasks(sb.load_prepared_tasks(tasks), runs, agent,
                                             spend_policy=SpendPolicy.from_raw("1"))
                self.assertEqual(code, 2 if raised else 0)
                metadata = self.assert_terminal(runs / run_dir, "response_rejected")
                self.assertNotIn("not_started", json.dumps(metadata["subagent_rejection"]))
                charge = self.ledgers(runs)[0]["calls"][0]["charge"]
                self.assertEqual(charge["basis"], "unpriced" if raised else "observed")
                if raised:
                    self.assertEqual(charge["reason"], "invocation_raised")
                else:
                    self.assertEqual(metadata["cost_normalized"]["total_cost"], 0.9)

    def test_invalid_or_ambiguous_dollars_do_not_certify_a_price(self):
        invalid = ('{"answer":"token-XYZ","usage":{"cost_usd":0.1},"usage":{"cost_usd":0.2}}',
                   '{"answer":"token-XYZ","usage":{"cost_usd":0.1}} trailing',
                   '{"answer":"token-XYZ","usage":{"cost_usd":NaN}}',
                   json.dumps({"answer": 4, "usage": {"cost_usd": True}}),
                   json.dumps({"answer": 4, "usage": {"cost_usd": -1}}),
                   json.dumps({"answer": 4, "usage": {"cost_usd": float("inf")}}))
        for response in invalid:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, tasks, runs, run_dir = self.fixture(root)
                backend = self.backend(root, "shell", [response])
                code, _, stderr = self.invoke(tasks, runs, backend, "--max-cost-usd", "1")
                self.assertEqual(code, 2, stderr)
                ledger = self.ledgers(runs)[0]
                self.assertEqual(ledger["calls"][0]["charge"]["basis"], "unpriced")
                metadata = json.loads((runs / run_dir / "metadata.json").read_text())
                self.assertIsNone(metadata["cost_normalized"].get("total_cost"))
                self.assertEqual(metadata["returncode"], 0)
                self.assertFalse(metadata["provider_response_complete"])

    def test_publication_failure_retains_settlement_and_prior_root(self):
        from unittest import mock

        import skill_benchmark as sb
        for operation in ("write_artifact_commit", "_install_staged_run"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, tasks, runs, run_dir = self.fixture(root)
                backend = self.backend(root, "shell", [json.dumps({"answer": "prior", "usage": {"cost_usd": 0.2}})] * 2)
                self.assertEqual(self.invoke(tasks, runs, backend)[0], 0)
                before = self.snapshot(runs / run_dir)
                with mock.patch.object(sb, operation, side_effect=OSError("artifact write failed")):
                    with self.assertRaisesRegex(OSError, "artifact write failed"):
                        self.invoke(tasks, runs, backend, "--max-cost-usd", "1")
                self.assertEqual(self.ledgers(runs)[0]["calls"][0]["charge"],
                                 {"basis": "observed", "amount_usd": "0.2", "provenance": "provider_reported"})
                self.assertEqual(self.snapshot(runs / run_dir), before)
                self.assertFalse(any((runs / run_dir).parent.glob(".*sidecars-*")))
                self.assertFalse(any((runs / run_dir).parent.glob(".*artifact-stage-*")))

    def test_ledger_settlement_publication_failure_propagates(self):
        from unittest import mock

        import spend_runtime
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, tasks, runs, run_dir = self.fixture(root)
            backend = self.backend(root, "shell", [json.dumps({"answer": "token-XYZ", "usage": {"cost_usd": 0.2}})])
            replace = spend_runtime.os.replace

            def fail_settlement(source, destination):
                if Path(destination).name == "spend-ceiling.json":
                    raw = json.loads(Path(source).read_text())
                    if raw["calls"][0]["state"] == "settled":
                        raise OSError("settlement write failed")
                return replace(source, destination)

            with mock.patch.object(spend_runtime.os, "replace", side_effect=fail_settlement):
                with self.assertRaisesRegex(OSError, "settlement write failed"):
                    self.invoke(tasks, runs, backend, "--max-cost-usd", "1")
            self.assertEqual(len((root / "launches").read_text().splitlines()), 1)
            self.assertEqual(self.ledgers(runs)[0]["calls"][0]["state"], "in_flight")
            self.assertFalse((runs / run_dir / "output.md").exists())
            self.assertFalse(any((runs / run_dir).parent.glob(".*sidecars-*")))

    def test_base_exception_propagates_after_unavailable_settlement_and_cleanup(self):
        import skill_benchmark as sb
        from spend_contracts import SpendPolicy
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, tasks, runs, run_dir = self.fixture(root)
            seen_workspaces = []

            def agent(**kwargs):
                seen_workspaces.append(kwargs["workspace"])
                raise KeyboardInterrupt("operator interrupted callback")

            with self.assertRaisesRegex(KeyboardInterrupt, "operator interrupted callback"):
                sb.run_subagent_tasks(sb.load_prepared_tasks(tasks), runs, agent, spend_policy=SpendPolicy.from_raw("1"))
            self.assertEqual(self.ledgers(runs)[0]["calls"][0]["charge"],
                             {"basis": "unpriced", "reason": "invocation_raised", "observed_subtotal_usd": None})
            self.assertEqual(len(seen_workspaces), 1)
            self.assertFalse(seen_workspaces[0].exists())
            self.assertFalse((runs / run_dir / "output.md").exists())
            self.assertFalse(any((runs / run_dir).parent.glob(".*sidecars-*")))

    def test_capped_recovery_anywhere_rejects_batch_without_writes(self):
        for prior in (False, True):
            with self.subTest(prior=prior), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, tasks, runs, run_dir = self.fixture(root)
                row = json.loads(tasks.read_text())
                second = dict(row, run_number=2, run_dir=str(Path(run_dir) / "run-2"), recovery={
                    "checkpoint_path": "checkpoint.json", "expected_content": "{}",
                    "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE",
                    "forbidden_path": "forbidden.txt", "match": "json"})
                tasks.write_text("\n".join(json.dumps(item) for item in (row, second)) + "\n")
                if prior:
                    runs.mkdir()
                    (runs / "operator.txt").write_text("untouched")
                before = self.snapshot(runs) if prior else None
                backend = self.backend(root, "shell", [json.dumps({"answer": "token-XYZ", "usage": {"cost_usd": 0.2}})])
                code, _, stderr = self.invoke(tasks, runs, backend, "--max-cost-usd", "1")
                self.assertEqual(code, 2)
                self.assertIn("capped recovery", stderr)
                self.assertFalse((root / "launches").exists())
                if prior:
                    self.assertEqual(self.snapshot(runs), before)
                else:
                    self.assertFalse(runs.exists())

    def test_allowed_capped_and_uncapped_calls_keep_prompt_and_artifact_bytes(self):
        from unittest import mock

        from helpers import claude_stream_records

        import skill_benchmark as sb
        for route in ("claude", "shell"):
            for turns in ((), ("one", "second prompt")):
                with self.subTest(route=route, turns=turns), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    _, tasks, runs, run_dir = self.fixture(root, turns=turns)
                    response = ("\n".join(json.dumps(row) for row in claude_stream_records(cost=0.06))
                                if route == "claude" else json.dumps({"answer": "token-XYZ", "usage": {"cost_usd": 0.06},
                                                                      "trace": [{"type": "command", "command": "ls", "status": "completed"}],
                                                                      "telemetry_scope": "turn_delta"}))
                    backend = self.backend(root, route, [response] * (len(turns) or 1), edit=True)
                    with mock.patch.object(sb.time, "time", return_value=1000):
                        self.assertEqual(self.invoke(tasks, runs, backend, "--tool-replay", "record")[0], 0)
                    first = self.snapshot(runs / run_dir)
                    design = (runs / "answer-design.json").read_bytes()
                    prompts = [json.loads(row) for row in (root / "launches").read_text().splitlines()]
                    (root / "launches").unlink()
                    with mock.patch.object(sb.time, "time", return_value=1000):
                        self.assertEqual(self.invoke(tasks, runs, backend, "--tool-replay", "record", "--max-cost-usd", "1")[0], 0)
                    second = self.snapshot(runs / run_dir)
                    capped_prompts = [json.loads(row) for row in (root / "launches").read_text().splitlines()]
                    for row in prompts + capped_prompts:
                        row.pop("workspace", None)
                    self.assertEqual(capped_prompts, prompts)
                    self.assertEqual((runs / "answer-design.json").read_bytes(), design)
                    self.assertEqual(set(second), set(first))
                    for path, old in first.items():
                        if path.endswith("metadata.json"):
                            new_metadata = json.loads(second[path])
                            new_metadata.pop("spend_ledger_path", None)
                            new_metadata.pop("spend_call_id", None)
                            self.assertEqual(new_metadata, json.loads(old), path)
                        elif path.endswith("workspace-changes.json"):
                            old_changes, new_changes = json.loads(old), json.loads(second[path])
                            for changes in (old_changes, new_changes):
                                changes.pop("workspace_root", None)
                                changes.pop("workspace_root_realpath", None)
                            self.assertEqual(new_changes, old_changes, path)
                        elif not path.endswith("artifact-commit.json"):
                            self.assertEqual(second[path], old, path)
                    self.assertEqual(self.ledgers(runs)[0]["spent_usd"], "0.12" if turns else "0.06")

    def test_response_is_immutable_after_settlement_and_thaws_at_owner_boundary(self):
        from unittest import mock

        import skill_benchmark as sb
        from spend_contracts import SpendPolicy
        from spend_runtime import SpendAdmission
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, tasks, runs, run_dir = self.fixture(root)
            response = {"answer": "token-XYZ", "usage": {"cost_usd": 0.2},
                        "trace": [{"type": "command", "command": "ls", "status": "completed"}]}
            run = SpendAdmission.run

            def mutate_after_settlement(admission, call, invoke):
                result = run(admission, call, invoke)
                response["answer"] = "changed"
                response["usage"]["cost_usd"] = 8
                response["trace"][0]["command"] = "rm"
                with self.assertRaises(TypeError):
                    result.value.response["usage"]["cost_usd"] = 9
                return result

            with mock.patch.object(SpendAdmission, "run", mutate_after_settlement):
                code = sb.run_subagent_tasks(sb.load_prepared_tasks(tasks), runs, lambda **kwargs: response,
                                             spend_policy=SpendPolicy.from_raw("1"))
            self.assertEqual(code, 0)
            metadata = json.loads((runs / run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["cost_normalized"]["total_cost"], 0.2)
            self.assertEqual((runs / run_dir / "output.md").read_text(), "token-XYZ")
            self.assertIn('"command": "ls"', (runs / run_dir / "trace.jsonl").read_text())

    def test_terminal_metadata_cannot_override_or_contradict_derived_evidence(self):
        import ablation_model as am
        import skill_benchmark as sb
        from runner_contracts import OutcomeContext, Provider
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, tasks, runs, run_dir = self.fixture(root)
            backend = self.backend(root, "shell", [json.dumps({"answer": "token-XYZ"})])
            self.assertEqual(self.invoke(tasks, runs, backend, "--max-cost-usd", "0")[0], 2)
            metadata = sb._with_committed_artifact_state(runs / run_dir, json.loads((runs / run_dir / "metadata.json").read_text()))
            self.assertTrue(metadata["artifact_set_complete"])
            for key, value in (("returncode", 0), ("invocation_state", "complete"),
                               ("provider_response_complete", True), ("process_observation_complete", True),
                               ("observation_complete", True), ("timed_out", True),
                               ("artifact_terminal_state", "unknown"), ("provider", "claude"),
                               ("subagent_refusals", []), ("subagent_rejection", {})):
                with self.subTest(key=key):
                    changed = {**metadata, key: value}
                    self.assertIsNotNone(am.metadata_lifecycle_error(changed))
                    self.assertFalse(am.execution_valid(changed, "token-XYZ"))
            for key, value in (("artifact_terminal_state", "budget_stopped"), ("subagent_refusals", []),
                               ("subagent_rejection", {}), ("invocation_state", "complete"),
                               ("provider_response_complete", True)):
                for field in ("metadata_extra", "metrics_extra"):
                    with self.subTest(key=key, field=field), self.assertRaisesRegex(ValueError, "derived evidence"):
                        OutcomeContext(provider=Provider.SUBAGENT, **{field: {key: value}})

    def test_started_conversation_retains_tool_replay_when_later_turns_are_refused(self):
        import skill_benchmark as sb
        from spend_contracts import SpendPolicy
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, tasks, runs, run_dir = self.fixture(root, turns=("one", "two", "three"))
            histories = []

            def agent(**kwargs):
                histories.append(kwargs["history"])
                result = kwargs["tool_executor"]("search", {"q": "paid"})
                return {"answer": result, "usage": {"cost_usd": 0.6}, "telemetry_scope": "turn_delta"}

            code = sb.run_subagent_tasks(sb.load_prepared_tasks(tasks), runs, agent,
                                         spend_policy=SpendPolicy.from_raw("0.5"), replay_mode="record",
                                         live_tools={"search": lambda payload: "retained tool answer"})
            self.assertEqual(code, 2)
            self.assertEqual(histories, [[]])
            base = runs / run_dir
            self.assert_terminal(base, "budget_stopped")
            self.assertIn("retained tool answer", (base / "tool-replay.json").read_text())
            self.assertEqual((base / "turn-1" / "output.md").read_text(), "retained tool answer")
