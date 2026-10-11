import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli


class BridgeCliBoundaryTests(unittest.TestCase):
    def test_bridge_requires_explicit_command_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root)
            tasks = root / "tasks.jsonl"
            self.assertEqual(run_cli("prepare", str(manifest), "--out", str(tasks))[0], 0)
            runs = root / "runs"
            code, _, stderr = run_cli("run-subagent", "--tasks", str(tasks), "--runs", str(runs),
                                      "--tool-bridge", "--tool-replay", "strict")
            self.assertEqual(code, 1)
            self.assertIn("tool bridge requires --agent-cmd", stderr)
            self.assertFalse(runs.exists())

    def test_unsupported_platform_refuses_before_child_and_runs(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root)
            tasks = root / "tasks.jsonl"
            self.assertEqual(run_cli("prepare", str(manifest), "--out", str(tasks))[0], 0)
            runs = root / "runs"
            with mock.patch("skill_benchmark._bridge_supported", return_value=False):
                code, _, stderr = run_cli("run-subagent", "--tasks", str(tasks), "--runs", str(runs),
                    "--tool-bridge", "--agent-cmd", "missing-child", "--tool-replay", "strict")
            self.assertEqual(code, 1)
            self.assertIn("nonreaping waitid", stderr)
            self.assertFalse(runs.exists())


class BridgeObservationTests(unittest.TestCase):
    def test_direct_acknowledged_search_cannot_establish_skill_open_success(self):
        import skill_benchmark as sb
        import tool_fault_contracts as domain
        identity = {"task_sha256": "sha256:" + "a" * 64, "case_input_sha256": "sha256:" + "b" * 64,
                    "run": {"case_id": "c", "variant": "with_skill", "run_number": 1,
                            "model": None, "population": "answer"},
                    "faults": [{"tool": "search", "output": "alpha", "is_error": False}],
                    "expected_turns": 1, "route": "offline_bridge"}
        rows = []
        def append(transition):
            rows.append(domain.journal_record(rows, transition))
        append({"type": "opened", "identity": identity})
        append({"type": "turn_opened", "turn": 1})
        resolution = domain._Resolution("fault", domain.fault_identity(identity["case_input_sha256"], 1),
            "alpha", {"availability": "known", "value": False})
        append(domain.bridge_offer(identity, 1, 1, "search", {"q": "alpha"}, resolution, 1))
        offer = rows[-1]
        append({"type": "acknowledged", "turn": 1, "exchange": 1,
                "offered_sha256": offer["record_sha256"], "ack": {"type": "ack", "protocol_version": 1,
                "id": 1, "turn_number": 1, "exchange_number": 1,
                "payload_sha256": offer["payload"]["response"]["payload_sha256"]}})
        append({"type": "turn_closed", "turn": 1})
        append({"type": "closed"})
        observation = domain.ToolObservation(identity, tuple(rows))
        self.assertEqual(observation.assess(domain.Count("errors")).value, 0)
        self.assertEqual(observation.assess(domain.Count("tool_calls")).value, 1)
        self.assertEqual(observation.assess(domain.SuccessfulOperations()).reason, "bridge_has_no_operation_semantics")
        self.assertEqual(sb.detect_trigger_records([], [Path("/mounted/SKILL.md")],
                         mediated_tools=observation).triggered, False)


@unittest.skipUnless(hasattr(os, "waitid") and hasattr(os, "WNOWAIT"),
                     "bridge requires nonreaping waitid")
class BridgeProcessTests(unittest.TestCase):
    def prepare(self, root, *, legacy=False, declared_error=True, output="alpha"):
        import skill_benchmark as sb
        case = {"id": "c", "split": "tune", "prompt": "Return alpha.",
                "assertions": [{"name": "answer", "type": "contains", "value": "alpha"}]}
        if not legacy:
            case["tool_faults"] = [{"tool": "search", "output": output, "is_error": declared_error}]
        manifest = make_eval_repo(root, cases=[case])
        tasks = root / "tasks.jsonl"
        self.assertEqual(run_cli("prepare", str(manifest), "--out", str(tasks))[0], 0)
        rows = [json.loads(line) for line in tasks.read_text().splitlines()]
        runs = root / "runs"
        if legacy:
            for row in rows:
                base = runs / row["run_dir"]
                base.mkdir(parents=True)
                (base / "tool-replay.json").write_text(json.dumps({"version": 1, "records": [{
                    "tool": "search", "key": sb.ToolReplayStore.call_key("search", {"q": "alpha"}), "output": "alpha"}]}))
        return tasks, rows, runs

    def command(self, tasks, runs, mode="success", *flags):
        import skill_benchmark as sb
        fixture = Path(__file__).parent / "fixtures" / "tool_bridge_child.py"
        child = shlex.join([sys.executable, str(fixture.resolve()), mode])
        return [sys.executable, sb.__file__, "run-subagent", "--tasks", str(tasks), "--runs", str(runs),
                "--tool-bridge", "--agent-cmd", child, "--tool-replay", "strict", "--timeout", "1", *flags]

    def execute(self, tasks, runs, mode="success", *flags):
        return subprocess.run(self.command(tasks, runs, mode, *flags), capture_output=True,
                              text=True, timeout=15, check=False)

    def test_known_error_crosses_real_cli_and_saved_reader(self):
        import skill_benchmark as sb
        import tool_fault_contracts as domain
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root, cases=[{
                "id": "c", "split": "tune", "prompt": "Return alpha.",
                "assertions": [{"name": "answer", "type": "contains", "value": "alpha"}],
                "tool_faults": [{"tool": "search", "output": "alpha", "is_error": True}],
            }])
            tasks = root / "tasks.jsonl"
            self.assertEqual(run_cli("prepare", str(manifest), "--out", str(tasks))[0], 0)
            runs = root / "runs"
            fixture = Path(__file__).parent / "fixtures" / "tool_bridge_child.py"
            child = shlex.join([sys.executable, str(fixture.resolve()), "success"])
            process = subprocess.run([sys.executable, sb.__file__, "run-subagent", "--tasks", str(tasks),
                                      "--runs", str(runs), "--tool-bridge", "--agent-cmd", child,
                                      "--tool-replay", "strict", "--timeout", "2"],
                                     capture_output=True, text=True, timeout=15, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            row = json.loads(tasks.read_text().splitlines()[0])
            base = runs / row["run_dir"]
            self.assertEqual((base / "output.md").read_text(), "alpha known True")
            observation = sb.read_tool_observation_base(base)
            self.assertEqual(observation.coverage, "closed")
            self.assertEqual(observation.assess(domain.Count("errors")).value, 1)
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_call", "tool": "search", "is_error": True}, base, {})[0], True)

    def test_legacy_only_bridge_closes_output_but_keeps_error_unavailable(self):
        import skill_benchmark as sb
        import tool_fault_contracts as domain
        with tempfile.TemporaryDirectory() as directory:
            tasks, rows, runs = self.prepare(Path(directory), legacy=True)
            absent_rows = tasks.read_bytes()
            result = self.execute(tasks, runs)
            self.assertEqual(result.returncode, 0, result.stderr)
            base = runs / rows[0]["run_dir"]
            self.assertEqual((base / "output.md").read_text(), "alpha unavailable None")
            observation = sb.read_tool_observation_base(base)
            self.assertEqual(observation.coverage, "closed")
            self.assertEqual(observation.assess(domain.Count("tool_calls")).value, 1)
            self.assertEqual(observation.assess(domain.Count("errors")).reason, "legacy_error_unavailable")
            self.assertEqual(tasks.read_bytes(), absent_rows)
            self.assertNotIn("tool_faults", rows[0])
            events = json.loads((base / "events.json").read_text())
            del events["mediated_tools"]
            (base / "events.json").write_text(json.dumps(events))
            self.assertEqual(sb.read_tool_observation_base(base).coverage, "missing")

    def test_same_output_known_false_is_delivered_without_operation_semantics(self):
        import skill_benchmark as sb
        import tool_fault_contracts as domain
        with tempfile.TemporaryDirectory() as directory:
            tasks, rows, runs = self.prepare(Path(directory), declared_error=False)
            result = self.execute(tasks, runs)
            self.assertEqual(result.returncode, 0, result.stderr)
            base = runs / rows[0]["run_dir"]
            self.assertEqual((base / "output.md").read_text(), "alpha known False")
            observation = sb.read_tool_observation_base(base)
            self.assertEqual(observation.assess(domain.Count("errors")).value, 0)
            self.assertEqual(observation.assess(domain.ExactOutput(1)).value, "alpha")
            self.assertEqual(sb.process_or_efficiency_assertion_result(
                {"type": "tool_call", "tool": "search", "is_error": False}, base, {})[0], True)
            self.assertIsInstance(observation.assess(domain.Count("file_reads")), domain.Unavailable)

    def test_blocked_response_write_consumes_deadline_without_ack(self):
        import skill_benchmark as sb
        with tempfile.TemporaryDirectory() as directory:
            tasks, rows, runs = self.prepare(Path(directory), output="x" * 60000)
            result = self.execute(tasks, runs, "blocked_response")
            self.assertEqual(result.returncode, 0, result.stderr)
            observation = sb.read_tool_observation_base(runs / rows[0]["run_dir"])
            self.assertEqual(observation.coverage, "open")
            self.assertEqual(observation.canonical_rows(), [])
            facts = json.loads(next(runs.glob(".tool-observations/*/turn-1/process.json")).read_text())
            self.assertEqual((facts["timed_out"], facts["group_stopped"], facts["reaped"]), (True, True, True))

    def test_offer_and_acked_prefix_timeout_remain_distinct(self):
        import skill_benchmark as sb
        import tool_fault_contracts as domain
        for mode, confirmed in (("offer_timeout", 0), ("prefix_timeout", 1)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                tasks, rows, runs = self.prepare(Path(directory))
                result = self.execute(tasks, runs, mode)
                self.assertEqual(result.returncode, 0, result.stderr)
                base = runs / rows[0]["run_dir"]
                observation = sb.read_tool_observation_base(base)
                self.assertEqual(observation.coverage, "open")
                self.assertEqual(len(observation.canonical_rows()), confirmed)
                self.assertIsInstance(observation.assess(domain.Count("tool_calls")), domain.Unavailable)
                self.assertEqual(sb.process_or_efficiency_assertion_result(
                    {"type": "tool_call", "tool": "search"}, base, {})[0], True if confirmed else None)
                facts = json.loads(next(runs.glob(".tool-observations/*/turn-1/process.json")).read_text())
                self.assertEqual((facts["timed_out"], facts["group_stopped"], facts["reaped"]), (True, True, True))

    def test_protocol_and_configured_bounds_reject_with_actual_cleanup(self):
        import skill_benchmark as sb
        cases = (("duplicate", (), "duplicate"), ("bad_ack", (), "ACK identity"),
                 ("invalid_utf8", (), "UnicodeDecodeError"),
                 ("partial", (), "deadline"),
                 ("oversized", ("--tool-bridge-max-frame-bytes", "1024"), "frame exceeds"),
                 ("stream", ("--tool-bridge-max-stream-bytes", "1024"), "stdout/stderr exceeds"),
                 ("success", ("--tool-bridge-max-outbound-bytes", "768"), "outbound stream exceeds"))
        for mode, flags, message in cases:
            with self.subTest(mode=mode, flags=flags), tempfile.TemporaryDirectory() as directory:
                tasks, rows, runs = self.prepare(Path(directory))
                result = self.execute(tasks, runs, mode, *flags)
                self.assertEqual(result.returncode, 0, result.stderr)
                base = runs / rows[0]["run_dir"]
                self.assertIn(message, (base / "output.md").read_text())
                self.assertEqual(sb.read_tool_observation_base(base).coverage, "open")
                facts = json.loads(next(runs.glob(".tool-observations/*/turn-1/process.json")).read_text())
                self.assertEqual((facts["group_stopped"], facts["reaped"]), (True, True))

    def test_reported_dollars_follow_actual_final_lifecycle(self):
        for mode, complete in (("final_nonzero", True), ("bad_sibling", True),
                               ("final_hang", False), ("held_pipes", False)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                tasks, _rows, runs = self.prepare(Path(directory))
                result = self.execute(tasks, runs, mode, "--max-cost-usd", "1")
                self.assertEqual(result.returncode, 0 if complete else 2, result.stderr)
                ledger = json.loads(next((runs / "spend").glob("*/spend-ceiling.json")).read_text())
                charge = ledger["calls"][0]["charge"]
                self.assertEqual(charge["basis"], "observed" if complete else "unpriced")
                self.assertEqual(charge.get("amount_usd", charge.get("observed_subtotal_usd")), "0.02")
                facts = json.loads(next(runs.glob(".tool-observations/*/turn-1/process.json")).read_text())
                self.assertEqual((facts["group_stopped"], facts["reaped"]), (True, True))
                self.assertEqual(facts["forced_termination"], not complete)
                self.assertEqual(len(ledger["calls"]), 2)

    def test_direct_journal_closure_failure_preserves_whole_reported_dollars(self):
        from unittest import mock

        import skill_benchmark as sb
        from spend_contracts import SpendPolicy
        with tempfile.TemporaryDirectory() as directory:
            tasks, rows, runs = self.prepare(Path(directory))
            command = self.command(tasks, runs)
            child = shlex.split(command[command.index("--agent-cmd") + 1])
            backend = sb._ToolBridgeBackend(tuple(child), 1, sb._BridgeLimits())
            original = sb._ToolReceiptJournal._commit
            def commit(journal, transition):
                if transition["type"] == "closed":
                    raise OSError("fixture journal closure failure")
                return original(journal, transition)
            with mock.patch.object(sb._ToolReceiptJournal, "_commit", commit):
                self.assertEqual(sb.run_subagent_tasks(rows, runs, backend, replay_mode="strict",
                    spend_policy=SpendPolicy.from_raw(1)), 0)
            ledger = json.loads(next((runs / "spend").glob("*/spend-ceiling.json")).read_text())
            self.assertEqual([row["charge"]["amount_usd"] for row in ledger["calls"]], ["0.02", "0.02"])
            self.assertEqual([row["charge"]["basis"] for row in ledger["calls"]], ["observed", "observed"])

    def test_direct_group_shutdown_is_observed_before_leader_reap(self):
        from unittest import mock

        import skill_benchmark as sb
        with tempfile.TemporaryDirectory() as directory:
            tasks, rows, runs = self.prepare(Path(directory))
            command = self.command(tasks, runs, "held_pipes")
            child = shlex.split(command[command.index("--agent-cmd") + 1])
            backend = sb._ToolBridgeBackend(tuple(child), 1, sb._BridgeLimits())
            observations = {}
            stopped_before_wait = []
            observe = sb.recovery_group_stopped
            wait = sb.subprocess.Popen.wait
            def group_stopped(pgid):
                result = observe(pgid)
                observations[pgid] = result[0]
                return result
            def reap(process, *args, **kwargs):
                if process.pid in observations:
                    stopped_before_wait.append(observations[process.pid])
                return wait(process, *args, **kwargs)
            with mock.patch.object(sb, "recovery_group_stopped", group_stopped), mock.patch.object(
                    sb.subprocess.Popen, "wait", reap):
                self.assertEqual(sb.run_subagent_tasks(rows, runs, backend, replay_mode="strict"), 0)
            self.assertEqual(stopped_before_wait, [True, True])

    def test_parent_interrupt_settles_retained_price_after_cleanup(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum), tempfile.TemporaryDirectory() as directory:
                tasks, _rows, runs = self.prepare(Path(directory))
                process = subprocess.Popen(self.command(tasks, runs, "interrupt", "--max-cost-usd", "1"),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        paths = list(runs.glob(".tool-observations/*/turn-1/stdout.bin"))
                        if paths and b'"type": "final"' in paths[0].read_bytes():
                            break
                        time.sleep(0.01)
                    else:
                        self.fail("child did not publish a final before the interrupt")
                    process.send_signal(signum)
                    _stdout, stderr = process.communicate(timeout=8)
                    self.assertEqual(process.returncode, 128 + signum, stderr)
                    ledger = json.loads(next((runs / "spend").glob("*/spend-ceiling.json")).read_text())
                    self.assertEqual(ledger["calls"][0]["charge"]["observed_subtotal_usd"], "0.02")
                    facts = json.loads(next(runs.glob(".tool-observations/*/turn-1/process.json")).read_text())
                    self.assertEqual((facts["group_stopped"], facts["reaped"], facts["forced_termination"]), (True, True, True))
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)
