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
