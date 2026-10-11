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


if __name__ == "__main__":
    unittest.main()
