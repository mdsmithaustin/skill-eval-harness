import json
import tempfile
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli


class ToolFaultPreparationTests(unittest.TestCase):
    def test_prepare_rejects_boolean_fault_budget_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root, cases=[{
                "id": "c", "split": "tune", "prompt": "Do the task.",
                "assertions": [{"name": "answer", "type": "contains", "value": "alpha"}],
                "tool_faults": [{"tool": "search", "output": "denied", "times": True}],
            }])
            output = root / "tasks.jsonl"
            code, _stdout, stderr = run_cli("prepare", str(manifest), "--out", str(output))
            self.assertEqual(code, 1)
            self.assertIn("tool_faults", stderr)
            self.assertFalse(output.exists())

    def test_prepare_carries_same_ordered_declarations_across_arms_and_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root, cases=[{
                "id": "c", "split": "tune", "prompt": "Do the task.",
                "assertions": [{"name": "answer", "type": "contains", "value": "alpha"}],
                "tool_faults": [{"tool": "search", "output": {"text": "denied"},
                                 "is_error": True, "times": 1}],
            }])
            code, stdout, stderr = run_cli("prepare", str(manifest), "--runs-per-variant", "2")
            self.assertEqual(code, 0, stderr)
            rows = [json.loads(line) for line in stdout.splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual([row.get("tool_faults") for row in rows], [[{
                "tool": "search", "output": {"text": "denied"},
                "is_error": True, "times": 1,
            }]] * 4)

    def test_representative_invalid_declarations_and_absent_projection(self):
        import skill_benchmark as sb
        import tool_fault_contracts as faults
        for declaration in ({"tool": "search", "output": float("inf")},
                            {"tool": "search", "output": "x", "is_error": 1},
                            {"tool": "search", "output": "x", "match": "["},
                            {"tool": "search", "output": "x", "unknown": True}):
            with self.subTest(declaration=declaration), self.assertRaises((TypeError, ValueError)):
                faults.parse_faults([declaration])
        with tempfile.TemporaryDirectory() as directory:
            manifest = make_eval_repo(Path(directory))
            rows = sb.prepared_task_rows(manifest, sb.validate_manifest(manifest))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("tool_faults" not in row and "tool_fault_task_sha256" not in row for row in rows))
            self.assertEqual(sb.ToolReplayStore.call_key("search", {"q": "alpha"}), "248fdf6c9c2e1efc704cd7525d030943")


class LegacyToolAssertionTests(unittest.TestCase):
    def test_saved_count_accepts_value_alias(self):
        import skill_benchmark as sb
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            sb.write_json(base / "events.json", {"schema_version": 2, "events": [{
                "type": "tool_call", "name": "search", "status": "completed",
            }]})
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_count_le", "max": 1}, base, {})[0], True)
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_count_le", "value": 1}, base, {})[0], True)

    def test_error_filter_requires_all_selected_named_and_command_errors(self):
        import skill_benchmark as sb
        for tool, event_type, event_name in (("search", "tool_call", "search"),
                                           ("bash", "command", "")):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                known = {"type": event_type, "status": "completed", "command": "known",
                         "is_error": False}
                rejected = {**known, "command": "rejected", "is_error": True}
                unknown = {"type": event_type, "status": "completed", "command": "unknown"}
                if event_name:
                    known["name"] = rejected["name"] = unknown["name"] = event_name
                assertion = {"type": "tool_call", "tool": tool, "is_error": False,
                             "min_count": 1, "max_count": 1}
                sb.write_json(base / "events.json", {"schema_version": 2, "events": [known, rejected]})
                self.assertEqual(sb.process_or_efficiency_assertion_result(assertion, base, {})[0], True)
                sb.write_json(base / "events.json", {"schema_version": 2, "events": [known, unknown]})
                self.assertEqual(sb.process_or_efficiency_assertion_result(assertion, base, {}),
                                 (None, "delivered_error_unavailable"))
                sb.write_json(base / "events.json", {"schema_version": 2, "events": [known, {
                    "type": "tool_call", "name": "other", "status": "completed",
                }]})
                self.assertEqual(sb.process_or_efficiency_assertion_result(assertion, base, {})[0], True)


class DirectCallableFaultTests(unittest.TestCase):
    def prepare(self, root, *, faults=None, turns=False, repeats=1):
        import skill_benchmark as sb
        case = {
            "id": "c", "split": "tune",
            "assertions": [{"name": "answer", "type": "contains", "value": "alpha"}],
            "tool_faults": faults or [{"tool": "search", "output": {"text": "denied"},
                                      "is_error": True, "times": 1}],
        }
        case.update({"turns": [{"prompt": "first"}, {"prompt": "second"}]} if turns else {"prompt": "Do the task."})
        manifest = make_eval_repo(root, cases=[case])
        rows = sb.prepared_task_rows(manifest, sb.validate_manifest(manifest), runs_per_variant=repeats, models=["fixture"])
        return manifest, rows

    def test_saved_closed_count_accepts_validated_value_alias(self):
        import skill_benchmark as sb
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root, cases=[{
                "id": "c", "split": "tune", "prompt": "Return alpha.",
                "assertions": [{"name": "limit", "type": "tool_count_le", "value": 1}],
                "tool_faults": [{"tool": "search", "output": "found"}],
            }])
            rows = sb.prepared_task_rows(manifest, sb.validate_manifest(manifest))
            def callback(*, tool_executor, **kwargs):
                self.assertEqual(tool_executor("search", {}), "found")
                return {"answer": "alpha"}
            runs = root / "runs"
            self.assertEqual(sb.run_subagent_tasks(rows, runs, callback, replay_mode="strict"), 0)
            base = runs / rows[0]["run_dir"]
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_count_le", "max": 1}, base, {})[0], True)
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_count_le", "value": 1}, base, {})[0], True)

    def test_direct_callback_budgets_reset_and_saved_error_is_unavailable(self):
        import skill_benchmark as sb
        import tool_fault_contracts as faults
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root, turns=True, repeats=2)
            outputs = []
            def callback(*, tool_executor, **kwargs):
                output = tool_executor("search", {"q": "alpha"})
                outputs.append(output)
                if isinstance(output, dict):
                    output["text"] = "mutated by callback"
                return {"answer": "alpha", "telemetry_scope": "turn_delta"}
            runs = root / "runs"
            self.assertEqual(sb.run_subagent_tasks(rows, runs, callback, replay_mode="record",
                live_tools={"search": lambda payload: "live"}), 0)
            self.assertEqual(outputs, [{"text": "mutated by callback"}, "live"] * 4)
            for row in rows:
                base = runs / row["run_dir"]
                observation = sb.read_tool_observation_base(base)
                self.assertEqual(observation.coverage, "closed")
                self.assertEqual(observation.assess(faults.ExactOutput(1)).value, {"text": "denied"})
                self.assertEqual(observation.assess(faults.Count("tool_calls")).value, 2)
                self.assertEqual(observation.assess(faults.Count("errors")).reason, "output_only_api")
                self.assertEqual(sb.process_or_efficiency_assertion_result(
                    {"type": "tool_call", "tool": "search"}, base, {})[0], True)
                self.assertEqual(sb.process_or_efficiency_assertion_result(
                    {"type": "tool_call", "tool": "search", "is_error": True}, base, {}),
                    (None, "output_only_api"))
                self.assertEqual(sb.read_metrics_base(base).get("errors"), None)
                self.assertEqual(sb.read_metrics_base(base)["tool_calls"], 2)
                sequence = observation.assess(faults.ToolClaim({"type": "tool_sequence", "expected": ["search", "search"]}))
                self.assertEqual(sequence.value[2], 1.0)
                self.assertIn("precision=1.0000; recall=1.0000; f1=1.0000", sequence.value[1])
                replay = json.loads((base / "tool-replay.json").read_text())
                self.assertEqual([record["output"] for record in replay["records"]], ["live"])
                self.assertEqual([event["type"] for event in sb.read_events_base(base)[0]], ["error"])

    def test_direct_open_prefix_proves_presence_and_withholds_exact_population(self):
        import skill_benchmark as sb
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root, faults=[{"tool": "search", "output": "denied"}])
            def callback(*, tool_executor, **kwargs):
                self.assertEqual(tool_executor("search", "first"), "denied")
                self.assertEqual(tool_executor("search", "second"), "denied")
                raise RuntimeError("callback tail failed")
            runs = root / "runs"
            sb.run_subagent_tasks(rows, runs, callback, replay_mode="strict")
            base = runs / rows[0]["run_dir"]
            self.assertEqual(sb.read_tool_observation_base(base).coverage, "open")
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_call", "tool": "search"}, base, {})[0], True)
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_call", "tool": "other", "expected_no_call": True}, base, {})[0], None)
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_count_le", "max": 2}, base, {})[0], None)
            sequence = sb.process_or_efficiency_assertion_result(
                {"type": "tool_sequence", "mode": "superset", "expected": ["search"]}, base, {})
            self.assertEqual(sequence[0], True)
            self.assertEqual(len(sequence), 2)

    def test_saved_closed_turn_survives_later_incomplete_journal_tail(self):
        import skill_benchmark as sb
        import tool_fault_contracts as faults
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root, turns=True, faults=[{"tool": "search", "output": "fixture"}])
            def callback(*, tool_executor, history, **kwargs):
                tool_executor("search", {})
                if history:
                    raise RuntimeError("later callback failed")
                return {"answer": "alpha", "telemetry_scope": "turn_delta"}
            runs = root / "runs"
            sb.run_subagent_tasks(rows, runs, callback, replay_mode="strict")
            base = runs / rows[0]["run_dir"]
            turn = base / "turn-1"
            self.assertEqual(sb.read_tool_observation_base(turn).coverage, "closed")
            member = json.loads((turn / "events.json").read_text())["mediated_tools"]
            with Path(member["journal_path"]).open("a") as handle:
                handle.write('{"version":1,')
            observation = sb.read_tool_observation_base(turn)
            self.assertEqual(observation.coverage, "closed")
            self.assertEqual(observation.assess(faults.Count("tool_calls")).value, 1)
            self.assertEqual(observation.assess(faults.ExactOutput(1)).value, "fixture")
            self.assertEqual(sb.read_tool_observation_base(base).coverage, "open")

    def test_direct_closed_empty_differs_from_missing_expected_member(self):
        import skill_benchmark as sb
        import tool_fault_contracts as faults
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root)
            runs = root / "runs"
            sb.run_subagent_tasks(rows, runs, lambda **kwargs: {"answer": "alpha"}, replay_mode="strict")
            base = runs / rows[0]["run_dir"]
            self.assertEqual(sb.read_tool_observation_base(base).assess(faults.Count("tool_calls")).value, 0)
            events = json.loads((base / "events.json").read_text())
            del events["mediated_tools"]
            (base / "events.json").write_text(json.dumps(events))
            self.assertEqual(sb.read_tool_observation_base(base).coverage, "missing")
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_call", "expected_no_call": True}, base, {})[0], None)

    def test_direct_live_mutable_identity_is_preserved_with_evidence_gap(self):
        import skill_benchmark as sb
        import tool_fault_contracts as faults
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root)
            aliased = {"is_error": False, "text": "live"}
            def callback(*, tool_executor, **kwargs):
                self.assertIs(tool_executor("other", {}), aliased)
                return {"answer": "alpha"}
            runs = root / "runs"
            sb.run_subagent_tasks(rows, runs, callback, replay_mode="record", live_tools={"other": lambda value: aliased})
            observation = sb.read_tool_observation_base(runs / rows[0]["run_dir"])
            self.assertEqual(observation.coverage, "open")
            self.assertIsInstance(observation.assess(faults.Count("tool_calls")), faults.Unavailable)

    def test_direct_paid_return_survives_later_journal_failure(self):
        from unittest import mock

        import skill_benchmark as sb
        from spend_contracts import SpendPolicy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root)
            original = sb._ToolReceiptJournal._commit
            def commit(journal, transition):
                if transition["type"] == "confirmed":
                    raise OSError("fixture journal confirmation failure")
                return original(journal, transition)
            def callback(*, tool_executor, **kwargs):
                tool_executor("search", {})
                return {"answer": "alpha", "usage": {"cost_usd": 0.125}}
            runs = root / "runs"
            with mock.patch.object(sb._ToolReceiptJournal, "_commit", commit):
                sb.run_subagent_tasks(rows, runs, callback, replay_mode="strict",
                    spend_policy=SpendPolicy.from_raw(1))
            ledgers = list((runs / "spend").glob("*/spend-ceiling.json"))
            self.assertEqual(len(ledgers), 1)
            ledger = json.loads(ledgers[0].read_text())
            self.assertEqual(ledger["spent_usd"], "0.250")
            metrics = sb.read_metrics_base(runs / rows[0]["run_dir"])
            self.assertEqual(metrics["cost_usd"], 0.125)
            self.assertEqual(metrics["timed_out"], False)

    def test_direct_fifo_and_output_dictionary_are_unchanged(self):
        import skill_benchmark as sb
        import tool_fault_contracts as faults
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tool-replay.json"
            payload = {"q": "alpha"}
            key = sb.ToolReplayStore.call_key("search", payload)
            path.write_text(json.dumps({"version": 1, "sanitize": [], "records": [
                {"tool": "search", "key": key, "output": {"is_error": True, "value": "one"}},
                {"tool": "search", "key": key, "output": "two"},
            ]}))
            store = sb.ToolReplayStore(path, "strict", faults=faults.parse_faults([
                {"tool": "search", "output": "fault-first", "times": 1},
            ]), case_input_sha256="sha256:" + "a" * 64)
            self.assertEqual(store.resolve("search", payload), "fault-first")
            expected_object = store.recorded[key][0]
            result, resolution = store._resolve_mediated("search", payload)
            self.assertIs(result, expected_object)
            result["value"] = "mutated"
            self.assertEqual(resolution.output, {"is_error": True, "value": "one"})
            self.assertEqual(store.resolve("search", payload), "two")
            self.assertEqual(store.new_records, [])
            self.assertEqual(json.loads(path.read_text())["version"], 1)

    def test_direct_caught_wrapper_failure_cannot_close_or_confirm_failed_output(self):
        from unittest import mock

        import skill_benchmark as sb
        import tool_fault_contracts as faults
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root, faults=[{"tool": "search", "output": "denied"}])
            original = sb._ToolReceiptJournal._commit
            failed = set()
            def commit(journal, transition):
                result = original(journal, transition)
                if transition["type"] == "prepared" and str(journal.path) not in failed:
                    failed.add(str(journal.path))
                    raise OSError("wrapper failed after preparation")
                return result
            def callback(*, tool_executor, **kwargs):
                with self.assertRaisesRegex(OSError, "wrapper failed"):
                    tool_executor("search", "first")
                return {"answer": "alpha"}
            runs = root / "runs"
            with mock.patch.object(sb._ToolReceiptJournal, "_commit", commit):
                sb.run_subagent_tasks(rows, runs, callback, replay_mode="strict")
            observation = sb.read_tool_observation_base(runs / rows[0]["run_dir"])
            self.assertEqual(observation.coverage, "open")
            self.assertIsInstance(observation.assess(faults.ExactOutput(1)), faults.Unavailable)

    def test_saved_wrong_identity_and_corrupt_tail_do_not_promote_claims(self):
        import skill_benchmark as sb
        import tool_fault_contracts as faults
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root, faults=[{"tool": "search", "output": "denied"}])
            def callback(*, tool_executor, **kwargs):
                tool_executor("search", "first")
                tool_executor("search", "second")
                return {"answer": "alpha"}
            runs = root / "runs"
            sb.run_subagent_tasks(rows, runs, callback, replay_mode="strict")
            base = runs / rows[0]["run_dir"]
            events = json.loads((base / "events.json").read_text())
            member = events["mediated_tools"]
            journal = Path(member["journal_path"])
            with journal.open("a") as handle:
                handle.write('{"broken":')
            observation = sb.read_tool_observation_base(base)
            self.assertEqual(observation.coverage, "open")
            self.assertEqual(observation.assess(faults.ExactOutput(1)).value, "denied")
            self.assertIsInstance(observation.assess(faults.Count("tool_calls")), faults.Unavailable)
            member["identity"]["run"]["variant"] = "without_skill"
            (base / "events.json").write_text(json.dumps(events))
            observation = sb.read_tool_observation_base(base)
            self.assertEqual(observation.coverage, "invalid")
            self.assertIsInstance(observation.assess(faults.ExactOutput(1)), faults.Unavailable)

    def test_direct_foreign_thread_cannot_confirm_owner_output(self):
        import threading

        import skill_benchmark as sb
        import tool_fault_contracts as faults
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root)
            def callback(*, tool_executor, **kwargs):
                tool_executor("search", "first")
                errors = []
                def foreign():
                    try:
                        tool_executor("search", "foreign")
                    except RuntimeError as exc:
                        errors.append(str(exc))
                thread = threading.Thread(target=foreign)
                thread.start()
                thread.join(2)
                self.assertEqual(errors, ["mediated callable requires synchronous nonreentrant tool execution"])
                return {"answer": "alpha"}
            runs = root / "runs"
            sb.run_subagent_tasks(rows, runs, callback, replay_mode="strict")
            observation = sb.read_tool_observation_base(runs / rows[0]["run_dir"])
            self.assertIsInstance(observation.assess(faults.ExactOutput(1)), faults.Unavailable)

    def test_cli_refuses_faults_and_attests_all_rows_before_output(self):
        import skill_benchmark as sb
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root)
            tasks = root / "tasks.jsonl"
            tasks.write_text("\n".join(json.dumps(row) for row in rows))
            runs = root / "runs"
            code, _, stderr = run_cli("run-subagent", "--tasks", tasks, "--runs", runs,
                                      "--agent-cmd", "fixture-must-not-launch", "--only-fault-cases")
            self.assertEqual(code, 1)
            self.assertIn("direct Python callable", stderr)
            self.assertFalse(runs.exists())
            rows[-1]["tool_faults"][0]["output"] = "altered"
            tasks.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "attestation mismatch"):
                run_cli("run-subagent", "--tasks", tasks, "--runs", runs,
                        "--agent-cmd", "fixture-must-not-launch", "--skip-fault-cases")
            self.assertFalse(runs.exists())
            self.assertEqual(sb.PreparedTask.from_row(rows[0]).tool_faults[0].tool, "search")
            del rows[-1]["tool_faults"]
            tasks.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "attestation mismatch"):
                run_cli("run-subagent", "--tasks", tasks, "--runs", runs,
                        "--agent-cmd", "fixture-must-not-launch", "--skip-fault-cases")
            self.assertFalse(runs.exists())

    def test_direct_saved_pairing_retains_unknown_operation_metrics(self):
        from helpers import result_row

        import skill_benchmark as sb
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, rows = self.prepare(root)
            def callback(*, tool_executor, **kwargs):
                tool_executor("search", {})
                return {"answer": "alpha"}
            runs = root / "runs"
            sb.run_subagent_tasks(rows, runs, callback, replay_mode="strict")
            results = [result_row("c", row["variant"], run_number=1, run_base=str(runs / row["run_dir"])) for row in rows]
            diff = sb.build_trajectory_diff(results)
            self.assertEqual(diff["pairs_compared"], 1)
            case = diff["cases"][0]
            self.assertEqual(case["mean_deltas"]["tool_calls"], 0.0)
            self.assertIsNone(case["mean_deltas"]["commands"])
            self.assertEqual(case["metric_pairing"]["commands"]["expected_pairs"], 1)
            self.assertEqual(case["metric_pairing"]["commands"]["observed_pairs"], 0)


class ToolObservationSnapshotTests(unittest.TestCase):
    def journal(self, *, close_run=True):
        import tool_fault_contracts as domain

        identity = {"task_sha256": "sha256:" + "a" * 64,
                    "case_input_sha256": "sha256:" + "b" * 64,
                    "run": {"case_id": "c", "variant": "with_skill", "run_number": 1,
                            "model": None, "population": "answer"},
                    "faults": [{"tool": "search", "output": {"items": [{"text": "denied"}]},
                                "is_error": True}], "expected_turns": 2}
        rows = []
        def append(transition):
            rows.append(domain.journal_record(rows, transition))
        append({"type": "opened", "identity": identity})
        for turn in (1, 2):
            append({"type": "turn_opened", "turn": turn})
            resolution = domain._Resolution("fault", domain.fault_identity(identity["case_input_sha256"], 1),
                {"items": [{"text": "denied"}]}, {"availability": "known", "value": True})
            append(domain.callable_preparation(identity, turn, turn, "search",
                                               {"queries": [{"text": f"query-{turn}"}]}, resolution))
            append({"type": "confirmed", "turn": turn, "exchange": turn,
                    "prepared_sha256": rows[-1]["record_sha256"]})
            append({"type": "turn_closed", "turn": turn})
        if close_run:
            append({"type": "closed"})
        return identity, rows

    def test_nested_constructor_input_mutation_cannot_change_assessments(self):
        import tool_fault_contracts as domain

        identity, rows = self.journal()
        observation = domain.ToolObservation(identity, tuple(rows))
        identity["run"]["variant"] = "without_skill"
        identity["faults"][0]["output"]["items"][0]["text"] = "changed declaration"
        rows[2]["input"]["queries"][0]["text"] = "changed input"
        rows[2]["payload"]["output"]["items"][0]["text"] = "changed output"
        rows.clear()
        self.assertEqual(observation.coverage, "closed")
        self.assertEqual(observation.assess(domain.Count("tool_calls")), domain.Observed(2))
        self.assertEqual(observation.assess(domain.ExactOutput(1)),
                         domain.Observed({"items": ({"text": "denied"},)}))
        first = observation.canonical_rows()[0]
        self.assertEqual(first["identity"]["run"]["variant"], "with_skill")
        self.assertEqual(first["input"], {"queries": [{"text": "query-1"}]})

    def test_canonical_rows_are_detached_on_every_projection(self):
        import tool_fault_contracts as domain

        identity, rows = self.journal()
        observation = domain.ToolObservation(identity, tuple(rows))
        projected = observation.canonical_rows()
        projected[0]["identity"]["faults"][0]["output"]["items"][0]["text"] = "changed fault"
        projected[0]["input"]["queries"].append({"text": "changed input"})
        projected[0]["payload"]["output"]["items"][0]["text"] = "changed output"
        projected[0]["payload"]["delivered_error"]["reason"] = "changed reason"
        projected.clear()
        fresh = observation.canonical_rows()
        self.assertEqual([row["exchange"] for row in fresh], [1, 2])
        self.assertEqual(fresh[0]["identity"]["faults"][0]["output"], {"items": [{"text": "denied"}]})
        self.assertEqual(fresh[0]["input"], {"queries": [{"text": "query-1"}]})
        self.assertEqual(fresh[0]["payload"]["output"], {"items": [{"text": "denied"}]})
        self.assertEqual(observation.assess(domain.Count("errors")), domain.Unavailable("output_only_api"))

    def test_turn_selection_and_value_semantics_preserve_source_fields(self):
        import dataclasses
        import inspect

        import tool_fault_contracts as domain

        identity, rows = self.journal(close_run=False)
        observation = domain.ToolObservation(identity, tuple(rows))
        first = observation.for_turn(1)
        second = observation.for_turn(2)
        self.assertEqual(observation.coverage, "open")
        self.assertEqual(first.coverage, "closed")
        self.assertEqual(second.coverage, "closed")
        self.assertEqual(first.assess(domain.Count("tool_calls")), domain.Observed(1))
        self.assertEqual(second.canonical_rows()[0]["input"], {"queries": [{"text": "query-2"}]})
        self.assertEqual(first, domain.ToolObservation(identity, tuple(rows), 1))
        self.assertNotEqual(first, second)
        public_fields = ("_identity", "_rows", "turn", "reason")
        self.assertEqual(tuple(inspect.signature(domain.ToolObservation).parameters), public_fields)
        expected_repr = "ToolObservation(" + ", ".join(
            f"{name}={getattr(first, name)!r}" for name in public_fields) + ")"
        self.assertEqual(repr(first), expected_repr)
        self.assertEqual(tuple(field.name for field in dataclasses.fields(first) if field.compare), public_fields)
        self.assertEqual(tuple(field.name for field in dataclasses.fields(first) if field.hash is not False),
                         public_fields)
        with self.assertRaises(TypeError):
            hash(first)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            observation.for_turn(True)

    def test_gap_reason_and_constructor_rejection_survive_turn_projection(self):
        import tool_fault_contracts as domain

        identity, rows = self.journal(close_run=False)
        observation = domain.ToolObservation(identity, tuple(rows), reason="invalid_tool_observation_tail")
        self.assertEqual(observation.for_turn(1).coverage, "open")
        self.assertEqual(observation.for_turn(1).assess(domain.Count("tool_calls")),
                         domain.Unavailable("invalid_tool_observation_tail"))
        self.assertEqual(observation.for_turn(1).assess(domain.ExactOutput(1)),
                         domain.Observed({"items": ({"text": "denied"},)}))
        rows[2]["payload"]["output"]["items"][0]["text"] = "forged output"
        with self.assertRaisesRegex(ValueError, "invalid tool journal chain"):
            domain.ToolObservation(identity, tuple(rows))
