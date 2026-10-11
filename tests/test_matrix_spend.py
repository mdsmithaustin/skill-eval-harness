import json
import os
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
        unknown_settled = threading.Event()
        barrier = threading.Barrier(2, timeout=5)
        def invoke(call):
            started.append(call.query_id)
            if call.query_id in {"0", "1"}:
                barrier.wait()
            if call.query_id == "1" and not unknown_settled.wait(5):
                raise TimeoutError("unknown settlement was not published")
            cost = (Measurement.unavailable("missing") if call.query_id == "0"
                    else Measurement.available(Money.from_raw("0.2"), provenance="provider_reported"))
            return Priced(call.query_id, cost)
        with tempfile.TemporaryDirectory() as td:
            with SpendAdmission.open(SpendPolicy.from_raw(1), population.paid_plan,
                                     root=Path(td)) as admission:
                persist = admission._persist
                def publish_settlement():
                    from spend_contracts import Settled, UnpricedCall
                    persist()
                    if any(isinstance(state, Settled) and isinstance(state.charge, UnpricedCall)
                           for state in admission.ledger.states.values()):
                        unknown_settled.set()
                with mock.patch.object(admission, "_persist", side_effect=publish_settlement):
                    results = admission.bounded_map(population, invoke, workers=2)
                self.assertEqual(sorted(started[:2]), ["0", "1"])
                self.assertEqual(started[2:], ["4"])
                self.assertEqual([value.call.query_id for value in results], ["0", "1", "2", "3", "4"])
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

    def test_interrupted_paid_result_drains_peer_and_retains_persistence_failure(self):
        from concurrent.futures import ThreadPoolExecutor

        from spend_contracts import Settled
        from spend_runtime import (
            BoundedRunError,
            MatrixNotStarted,
            MatrixStopReason,
            Priced,
            SpendAdmission,
            Started,
        )
        from telemetry import Measurement, Money
        population = self.population()
        interruption = KeyboardInterrupt("interrupted paid result")
        persistence_error = OSError("final persistence failed")
        prior_error = ValueError("prior persistence cause")
        persistence_error.__cause__ = prior_error
        peer_started = threading.Event()
        release_peer = threading.Event()
        peer_finished = threading.Event()
        started = []
        def invoke(call):
            started.append(call.query_id)
            if call.query_id == "0":
                if not peer_started.wait(5):
                    raise TimeoutError("paid peer did not start")
                return Priced("interrupted value", Measurement.available(
                    Money.from_raw("0.2"), provenance="provider_reported"), interruption=interruption)
            peer_started.set()
            if not release_peer.wait(5):
                raise TimeoutError("paid peer was not drained")
            peer_finished.set()
            return Priced("peer value", Measurement.available(
                Money.from_raw("0.3"), provenance="provider_reported"))
        import spend_runtime
        price = spend_runtime.price_measurement
        def charge(*args, **kwargs):
            result = price(*args, **kwargs)
            release_peer.set()
            return result
        executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(executor.shutdown, wait=True, cancel_futures=True)
        with tempfile.TemporaryDirectory() as td:
            with SpendAdmission.open(SpendPolicy.from_raw(1), population.paid_plan,
                                     root=Path(td)) as admission:
                persist = admission._persist
                def final_persistence():
                    if any(isinstance(state, Settled) for state in admission.ledger.states.values()):
                        raise persistence_error
                    persist()
                published = mock.Mock(side_effect=lambda value: value)
                with mock.patch.object(admission, "_persist", side_effect=final_persistence), \
                        mock.patch("spend_runtime.price_measurement", side_effect=charge), \
                        mock.patch("spend_runtime.ThreadPoolExecutor", return_value=executor), \
                        mock.patch.object(executor, "shutdown", wraps=executor.shutdown) as shutdown:
                    with self.assertRaises(BoundedRunError) as caught:
                        admission.bounded_map(population, invoke, workers=2, publish=published)
                error = caught.exception
                self.assertIs(error.cause, interruption)
                self.assertIs(error.__cause__, persistence_error)
                self.assertIs(interruption.__context__, persistence_error)
                self.assertIsNot(persistence_error.__context__, interruption)
                self.assertIs(persistence_error.__cause__, prior_error)
                self.assertEqual([item.call.query_id for item in error.executions], ["0", "1", "2", "3", "4"])
                self.assertEqual(error.executions[:2], (
                    Started(population.cells[0].call, "interrupted value"),
                    Started(population.cells[1].call, "peer value")))
                self.assertEqual(error.executions[2:], tuple(
                    MatrixNotStarted(cell.call, MatrixStopReason.INTERRUPTED) for cell in population.cells[2:]))
                self.assertEqual(str(admission.ledger.spent.amount), "0.5")
                self.assertEqual(sorted(started), ["0", "1"])
                self.assertTrue(peer_finished.is_set())
                published.assert_not_called()
                shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_drain_collects_all_paid_outcomes_before_charging_can_fail(self):
        from spend_contracts import InFlight
        from spend_runtime import (
            BoundedRunError,
            MatrixNotStarted,
            MatrixStopReason,
            Priced,
            SpendAdmission,
            Started,
        )
        from telemetry import Measurement, Money
        population = self.population()
        interruption = KeyboardInterrupt("controller stopped")
        charge_error = ValueError("charging failed")
        all_started = threading.Event()
        release = threading.Event()
        started = []
        finished = []
        def invoke(call):
            started.append(call.query_id)
            if len(started) == 3:
                all_started.set()
            if not release.wait(5):
                raise TimeoutError("admitted workers were not drained")
            finished.append(call.query_id)
            return Priced(f"paid value {call.query_id}", Measurement.available(
                Money.from_raw("0.2"), provenance="provider_reported"))
        def stopped():
            if all_started.is_set():
                release.set()
                raise interruption
            return False
        with tempfile.TemporaryDirectory() as td:
            with SpendAdmission.open(SpendPolicy.from_raw(1), population.paid_plan,
                                     root=Path(td)) as admission:
                published = mock.Mock(side_effect=lambda value: value)
                with mock.patch("spend_runtime.price_measurement", side_effect=charge_error):
                    with self.assertRaises(BoundedRunError) as caught:
                        admission.bounded_map(population, invoke, workers=3,
                                              publish=published, stopped=stopped)
                error = caught.exception
                self.assertIs(error.cause, interruption)
                self.assertEqual(error.executions[:3], tuple(
                    Started(cell.call, f"paid value {index}") for index, cell in enumerate(population.cells[:3])))
                self.assertIs(error.__cause__, charge_error)
                self.assertEqual(error.executions[3:], tuple(
                    MatrixNotStarted(cell.call, MatrixStopReason.INTERRUPTED) for cell in population.cells[3:]))
                self.assertEqual(sorted(started), ["0", "1", "2"])
                self.assertEqual(sorted(finished), ["0", "1", "2"])
                self.assertEqual(str(admission.ledger.spent.amount), "0")
                self.assertEqual([type(admission.ledger.states[cell.call.call_id]) for cell in population.cells[:3]],
                                 [InFlight, InFlight, InFlight])
                published.assert_not_called()

    def test_interrupted_future_inspection_keeps_the_paid_result_for_drain(self):
        from concurrent.futures import Future

        from spend_runtime import (
            BoundedRunError,
            MatrixNotStarted,
            MatrixStopReason,
            Priced,
            SpendAdmission,
            Started,
        )
        from telemetry import Measurement, Money
        population = self.population()
        interruption = KeyboardInterrupt("future inspection interrupted")
        future = Future()
        future.set_result(Priced("completed paid value", Measurement.available(
            Money.from_raw("0.2"), provenance="provider_reported")))
        with tempfile.TemporaryDirectory() as td:
            with SpendAdmission.open(SpendPolicy.from_raw(1), population.paid_plan,
                                     root=Path(td)) as admission:
                with mock.patch("spend_runtime.ThreadPoolExecutor") as executor, \
                        mock.patch.object(future, "exception", side_effect=[interruption, None]):
                    executor.return_value.submit.return_value = future
                    with self.assertRaises(BoundedRunError) as caught:
                        admission.bounded_map(population, mock.Mock(), workers=1)
                self.assertIs(caught.exception.cause, interruption)
                self.assertIs(caught.exception.__cause__, interruption)
                self.assertEqual(caught.exception.executions[0],
                                 Started(population.cells[0].call, "completed paid value"))
                self.assertEqual(caught.exception.executions[1:], tuple(
                    MatrixNotStarted(cell.call, MatrixStopReason.INTERRUPTED) for cell in population.cells[1:]))
                self.assertEqual(str(admission.ledger.spent.amount), "0.2")

    def test_worker_cancellation_retains_its_error_and_settles_assumed_price(self):
        from concurrent.futures import CancelledError

        from spend_contracts import MatrixPopulation
        from spend_runtime import Priced, SpendAdmission, WorkerRaised
        population = MatrixPopulation(self.population().cells[:1])
        worker_error = CancelledError("worker raised cancellation")
        def invoke(call) -> Priced[str]:
            raise worker_error
        with tempfile.TemporaryDirectory() as td:
            with SpendAdmission.open(SpendPolicy.from_raw(1, "0.2"), population.paid_plan,
                                     root=Path(td)) as admission:
                executions = admission.bounded_map(population, invoke, workers=1)
                self.assertEqual(executions, (WorkerRaised(population.cells[0].call, worker_error),))
                self.assertIs(executions[0].error, worker_error)
                self.assertEqual(str(admission.ledger.spent.amount), "0.2")

    @unittest.skipIf(os.name == "nt", "requires supported POSIX directory sync")
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
    def test_successful_final_pipe_drain_retains_complete_billing(self):
        import subprocess

        import skill_benchmark as benchmark
        from invocation_contracts import ProcessInvocationPlan, RawProcessCapture
        capture = RawProcessCapture()
        stream = b'{"type":"result","total_cost_usd":0.6}\n'
        process = mock.Mock(pid=12345, returncode=0)
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(["fixture"], 0.05, output=stream, stderr=b""),
            (stream, b""),
        ]
        with tempfile.TemporaryDirectory() as td:
            plan = ProcessInvocationPlan.from_values(
                ["fixture"], input_text=None, cwd=td, timeout_s=5, raw_capture=capture)
            with mock.patch.object(benchmark.subprocess, "Popen", return_value=process), \
                    mock.patch.object(benchmark.os, "killpg", create=True):
                benchmark.invoke_argv_with_timeout(plan)
        self.assertTrue(capture.facts.communication_complete)
        self.assertFalse(capture.facts.timed_out)
        cost, floor = matrix._matrix_dollars("claude", matrix._raw_invocation(matrix._CellCapture(capture)))
        self.assertEqual(str(cost.value.amount), "0.6")
        self.assertIsNone(floor)

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

    def test_timeout_keeps_a_safe_prefix_floor_before_a_truncated_record(self):
        from trigger_contracts import InvocationOutcome
        invocation = InvocationOutcome.from_timeout(
            stdout='{"type":"result","total_cost_usd":0.6}\n{"type":',
            stderr="", elapsed_ms=1)
        cost, floor = matrix._matrix_dollars("claude", invocation)
        self.assertIsNone(cost.value)
        self.assertEqual(str(floor.amount), "0.6")

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
