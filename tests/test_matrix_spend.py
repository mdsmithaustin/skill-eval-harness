import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
