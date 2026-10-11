import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import run_trigger_matrix as matrix
from spend_contracts import SpendPolicy

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples/demo-skill/evals/shared-benchmark.json"


class MatrixSpendTests(unittest.TestCase):
    def run_matrix(self, root, agents, policy):
        rows = matrix.cases_from_manifest(matrix.load_manifest(MANIFEST), "tune")
        return matrix.run_matrix(
            MANIFEST, rows, agents, [None], 2, 10, 2,
            spend_policy=policy, spend_root=root / "spend", trace_runs=root / "traces")

    def test_zero_cap_retains_every_paid_repetition_without_process_claims(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = self.run_matrix(root, ["claude"], SpendPolicy.from_raw(0))
            self.assertEqual(report["summary"]["total"], 4)
            self.assertEqual(report["summary"]["incomplete_reasons"], {"cost_ceiling": 4})
            self.assertNotIn("pass_rate", report["summary"])
            for row in report["results"]:
                self.assertEqual(row["execution_state"], "not_started")
                self.assertIsNone(row["pass"])
                self.assertNotIn("returncode", row)
            self.assertFalse((root / "traces").exists())

    def test_zero_cap_runs_concrete_stub_and_retains_paid_population(self):
        with tempfile.TemporaryDirectory() as td:
            report = self.run_matrix(Path(td), ["claude", "stub"], SpendPolicy.from_raw(0))
            self.assertEqual(report["summary"]["total"], 8)
            self.assertEqual(report["summary"]["observed"], 4)
            self.assertEqual(len(report["spend"]["calls"]), 4)
            self.assertEqual(report["spend"]["scope"], "paid_matrix_calls")

    def test_detection_failure_retains_paid_dollars_and_complete_process(self):
        from trigger_contracts import InvocationOutcome, matrix_observation_from_row
        returned = InvocationOutcome.from_process(
            stdout='{"type":"result","subtype":"success","total_cost_usd":0.6}\n',
            stderr="", returncode=0, elapsed_ms=1)
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(matrix.ClaudeAdapter, "_run_argv", return_value=returned) as invoke, \
             mock.patch.object(matrix.ClaudeAdapter, "detect", side_effect=RuntimeError("detection")):
            root = Path(td)
            rows = matrix.cases_from_manifest(matrix.load_manifest(MANIFEST), "tune")[:1]
            report = matrix.run_matrix(MANIFEST, rows, ["claude"], [None], 3, 5, 1,
                                       spend_policy=SpendPolicy.from_raw("0.5"), spend_root=root / "spend")
        invoke.assert_called_once()
        self.assertEqual(report["spend"]["spent_usd"], "0.6")
        self.assertEqual(report["results"][0]["original_invocation"]["returncode"], 0)
        self.assertEqual(report["results"][0]["incomplete_reason"], "discovery")
        self.assertEqual(report["summary"]["total"], 3)
        self.assertEqual(matrix.matrix_exit_code(report), 2)
        for row in report["results"]:
            matrix_observation_from_row(row)
        import skill_benchmark as benchmark
        self.assertEqual(len(benchmark._trigger_report_rows(report, "paid").observations), 3)

    def test_positive_known_missing_route_preflight_has_no_persistent_writes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(SystemExit, "requires --assumed"):
                self.run_matrix(root, ["claude", "codex"], SpendPolicy.from_raw(1))
            self.assertEqual(list(root.iterdir()), [])

    def test_substituted_stub_gets_no_offline_authority(self):
        class Replaced(matrix.StubAdapter):
            def invoke(self, *args):
                raise AssertionError("must not run")
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(matrix.ADAPTERS, {"stub": Replaced}):
            report = self.run_matrix(Path(td), ["stub"], SpendPolicy.from_raw(0))
        self.assertEqual(report["summary"]["observed"], 0)
        self.assertEqual(len(report["spend"]["calls"]), 4)

    def test_permitted_cap_keeps_offline_protocol_and_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            uncapped = self.run_matrix(root / "uncapped", ["stub"], None)
            capped = self.run_matrix(root / "capped", ["stub"], SpendPolicy.from_raw(1))
        self.assertEqual(uncapped["protocol"], capped["protocol"])
        self.assertEqual(uncapped["design"], capped["design"])
        self.assertEqual(uncapped["summary"], capped["summary"])
        self.assertEqual(capped["spend"]["calls"], [])



class BoundedAdmissionTests(unittest.TestCase):
    def population(self, offline=()):
        from spend_contracts import MatrixCall, MatrixPopulation, MatrixScheduleCell
        digest = "sha256:" + "a" * 64
        return MatrixPopulation(tuple(MatrixScheduleCell(
            MatrixCall("stub" if index in offline else "claude", None, str(index), digest,
                       1, digest, digest), index in offline) for index in range(5)))

    def test_unknown_drains_active_peer_and_continues_offline(self):
        from spend_runtime import NotStarted, Priced, SpendAdmission, Started
        from telemetry import Measurement, Money
        population = self.population(offline=(4,))
        started = []
        barrier = threading.Barrier(2, timeout=5)
        def invoke(call):
            started.append(call.query_id)
            if call.query_id in {"0", "1"}:
                barrier.wait()
            cost = (Measurement.unavailable("missing") if call.query_id == "0"
                    else Measurement.available(Money.from_raw("0.2"), provenance="provider_reported"))
            return Priced(call.query_id, cost)
        with tempfile.TemporaryDirectory() as td:
            with SpendAdmission.open(SpendPolicy.from_raw(1), population.paid_plan,
                                     root=Path(td)) as admission:
                results = admission.bounded_map(population, invoke, workers=2)
                self.assertEqual(started, ["0", "1", "4"])
                self.assertEqual([type(value) for value in results],
                                 [Started, Started, NotStarted, NotStarted, Started])
                self.assertEqual(admission.ledger.spent.amount, Money.from_raw("0.2").amount)
                from spend_contracts import SpendLedger
                self.assertEqual(SpendLedger.from_dict(json.loads(admission.path.read_text())),
                                 admission.ledger)

    def test_admission_persistence_failure_never_launches(self):
        from spend_runtime import BoundedRunError, SpendAdmission
        population = self.population()
        with tempfile.TemporaryDirectory() as td:
            with SpendAdmission.open(SpendPolicy.from_raw(1), population.paid_plan,
                                     root=Path(td)) as admission:
                with mock.patch.object(admission, "_persist", side_effect=OSError("disk")):
                    invoke = mock.Mock()
                    with self.assertRaises(BoundedRunError) as caught:
                        admission.bounded_map(population, invoke, workers=2)
                    self.assertIsInstance(caught.exception.cause, OSError)
                    invoke.assert_not_called()

    def test_supported_directory_sync_failure_is_fatal_but_unsupported_is_allowed(self):
        import errno
        import os
        import stat

        from spend_runtime import BoundedRunError, Priced, SpendAdmission
        from telemetry import Measurement, Money
        population = self.population()
        real_sync = os.fsync
        for number in (errno.EIO, errno.EINVAL):
            with self.subTest(errno=number), tempfile.TemporaryDirectory() as td:
                with SpendAdmission.open(SpendPolicy.from_raw(1), population.paid_plan,
                                         root=Path(td)) as admission:
                    def sync(fd, error_number=number):
                        if stat.S_ISDIR(os.fstat(fd).st_mode):
                            raise OSError(error_number, "directory sync")
                        real_sync(fd)
                    invoke = mock.Mock(return_value=Priced("done", Measurement.available(
                        Money.from_raw(1), provenance="provider_reported")))
                    with mock.patch("spend_runtime.os.fsync", side_effect=sync):
                        if number == errno.EIO:
                            with self.assertRaises(BoundedRunError):
                                admission.bounded_map(population, invoke, workers=1)
                            invoke.assert_not_called()
                        else:
                            admission.bounded_map(population, invoke, workers=1)
                            invoke.assert_called_once()



class RetainedEvidenceTests(unittest.TestCase):
    def test_redactor_failure_retains_original_process_and_dollars(self):
        import sys

        import skill_benchmark as benchmark
        from invocation_contracts import ProcessInvocationPlan, RawProcessCapture
        capture = RawProcessCapture()
        stream = '{"type":"result","total_cost_usd":0.6,"usage":{"input_tokens":true}}\n'
        def fail(text):
            raise ValueError("redaction failed")
        with tempfile.TemporaryDirectory() as td:
            plan = ProcessInvocationPlan.from_values(
                [sys.executable, "-c", f"print({stream!r}, end='')"], input_text=None,
                cwd=td, timeout_s=5, redact_output=fail, raw_capture=capture)
            with self.assertRaisesRegex(ValueError, "redaction failed"):
                benchmark.invoke_argv_with_timeout(plan)
        self.assertEqual(capture.facts.returncode, 0)
        raw = matrix._raw_invocation(matrix._CellCapture(capture))
        cost, floor = matrix._matrix_dollars("claude", raw)
        self.assertEqual(str(cost.value.amount), "0.6")
        self.assertIsNone(floor)

    def test_timeout_price_is_a_floor_and_codex_dollars_are_unknown(self):
        from trigger_contracts import InvocationOutcome
        invocation = InvocationOutcome.from_timeout(
            stdout='{"type":"result","total_cost_usd":0.7}\n', stderr="", elapsed_ms=1)
        cost, floor = matrix._matrix_dollars("claude", invocation)
        self.assertIsNone(cost.value)
        self.assertEqual(str(floor.amount), "0.7")
        cost, floor = matrix._matrix_dollars("codex", invocation)
        self.assertIsNone(cost.value)
        self.assertIsNone(floor)

    def test_not_started_and_failure_rows_round_trip_with_full_denominator(self):
        from trigger_contracts import (
            FailedTriggerObservation,
            InvocationOutcome,
            NotStartedTriggerObservation,
            TriggerExpectation,
            TriggerFailureStage,
            TriggerNotStartedReason,
            TriggerRepetitionIdentity,
            matrix_observation_from_row,
        )
        from trigger_reporting import summarize_trigger_cohort, trigger_cohort_as_dict
        common = ("claude", None, "query", TriggerExpectation.TRIGGER,
                  TriggerRepetitionIdentity("id", 1), None, {})
        refused = NotStartedTriggerObservation(*common, TriggerNotStartedReason.COST_CEILING)
        original = InvocationOutcome.from_process(stdout="", stderr="", returncode=0, elapsed_ms=1)
        failed = FailedTriggerObservation(*common, TriggerFailureStage.DISCOVERY, original)
        self.assertEqual(matrix_observation_from_row(refused.as_row()), refused)
        self.assertEqual(matrix_observation_from_row(failed.as_row()), failed)
        summary = trigger_cohort_as_dict(summarize_trigger_cohort([refused, failed]))
        self.assertEqual(summary["incomplete_reasons"], {"cost_ceiling": 1, "discovery": 1})
        self.assertFalse(any(key.endswith("rate") for key in summary))
        row = refused.as_row()
        row["returncode"] = 0
        with self.assertRaises(ValueError):
            matrix_observation_from_row(row)


if __name__ == "__main__":
    unittest.main()
