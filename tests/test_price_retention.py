import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import run_cli, write_with_skill_task

import skill_benchmark as sb


class CapturedPriceTests(unittest.TestCase):
    def test_shared_claude_timeout_names_the_floor_and_judge_retains_it(self):
        process = sb.InvocationResult(
            stdout='{"type":"result","result":"candidate","total_cost_usd":0.06}',
            stderr="", returncode=124, elapsed_ms=1000,
            stdout_utf8_valid=True, stderr_utf8_valid=True,
            invocation_state=sb.InvocationState.TIMED_OUT, timed_out=True)
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
                sb, "run_argv_capture", return_value=process):
            fixture_bin = str(Path(td) / "missing-fixture")
            result = sb.claude_cli_invoke("prompt", isolation=sb.ContextIsolation.SEALED,
                                          timeout=1, claude_bin=fixture_bin)
        self.assertIsNone(result["cost_usd"])
        self.assertEqual(result["observed_subtotal_usd"], 0.06)
        self.assertEqual(result["returncode"], 124)
        self.assertTrue(result["timed_out"])
        with mock.patch.object(sb, "claude_cli_invoke", return_value=result):
            invocation = sb.claude_judge_invoke(
                "prompt", judge_model="fixture", claude_bin=fixture_bin,
                assertion_schema={"type": "object"}, extra_args=None, explore_hint=None)
        self.assertIsNone(invocation.cost_usd)
        self.assertEqual(invocation.observed_subtotal_usd, 0.06)
        self.assertEqual(invocation.invocation_state, sb.InvocationState.TIMED_OUT)
        sentinel = sb.claude_cli_invoke("prompt", isolation=sb.ContextIsolation.SEALED,
                                        timeout=1, claude_bin=fixture_bin)
        self.assertEqual(sentinel["invocation_state"], "spawn_failed")
        self.assertEqual(sentinel["returncode"], 127)
        self.assertIsNone(sentinel["cost_usd"])

    def shell_timeout(self, stdout, *, requires_delta=False):
        backend = sb.shell_agent_backend("agent", timeout=1)
        with mock.patch("skill_benchmark.subprocess.run", side_effect=
                        subprocess.TimeoutExpired("agent", 1, output=stdout)):
            return sb._invoke_paid_subagent(
                backend, requires_delta=requires_delta, prompt="task", workspace=Path("."),
                model=None, tool_executor=None)

    def test_shell_timeout_keeps_only_an_observed_subtotal(self):
        raw = b'{"answer":"candidate","usage":{"cost_usd":0.06},"telemetry_scope":"turn_delta","returncode":0,"timed_out":false}'
        priced = self.shell_timeout(raw)
        self.assertIsNone(priced.cost.value)
        self.assertEqual(str(priced.observed_subtotal.amount), "0.06")
        self.assertEqual(priced.value.evidence.process.returncode, 124)
        self.assertTrue(priced.value.evidence.process.timed_out)
        response, error = sb._subagent_artifact_response(priced.value)
        self.assertIn("timed out", error)
        self.assertNotIn("cost_usd", response.get("usage", {}))

    def test_shell_timeout_rejects_unsafe_original_bytes_and_json(self):
        cases = (
            b'{"answer":"\xff","usage":{"cost_usd":0.06}}',
            b'{"usage":{"cost_usd":0.01,"cost_usd":0.06}}',
            b'{"usage":{"cost_usd":0.06},"telemetry_scope":"turn_delta","telemetry_scope":"turn_delta"}',
            b'{"usage":{"cost_usd":0.06}',
            b'{"usage":{"cost_usd":0.06}} trailing',
            b'{"usage":{"cost_usd":0.06}}\n{}',
            b'{"usage":{"cost_usd":NaN}}',
            b'{"usage":{"cost_usd":Infinity}}',
            b'{"usage":{"cost_usd":1e999}}',
            b'{"usage":{"cost_usd":-0.06}}',
            b'{"usage":{"cost_usd":true}}',
            b'{"usage":{"cost_usd":"0.06"}}',
        )
        for raw in cases:
            with self.subTest(raw=raw):
                priced = self.shell_timeout(raw)
                self.assertIsNone(priced.cost.value)
                self.assertIsNone(priced.observed_subtotal)
                self.assertEqual(priced.cost.reason, "subagent_does_not_report_safe_dollars")
                self.assertEqual(priced.value.evidence.process.stdout_utf8_valid,
                                 b"\xff" not in raw)

    def test_multi_turn_timeout_requires_explicit_delta_scope(self):
        for scope in ("turn_delta", "conversation_cumulative", None, "bad", True):
            with self.subTest(scope=scope):
                response = {"answer": 3, "usage": {"cost_usd": 0.06, "input_tokens": "bad"}}
                if scope is not None:
                    response["telemetry_scope"] = scope
                priced = self.shell_timeout(json.dumps(response).encode(), requires_delta=True)
                self.assertIsNone(priced.cost.value)
                self.assertEqual(priced.observed_subtotal is not None, scope == "turn_delta")
                self.assertEqual(priced.cost.reason, "subagent_cost_incomplete_after_timeout" if scope == "turn_delta"
                                 else "subagent_turn_cost_lacks_delta_scope")
                self.assertEqual(str(priced.value.evidence.reported_cost.value.amount), "0.06")

    def test_opaque_callback_exception_cannot_supply_price_or_process(self):
        def raised(**kwargs):
            raise RuntimeError('{"usage":{"cost_usd":0.06},"returncode":124}')

        with self.assertRaisesRegex(RuntimeError, "cost_usd"):
            sb._invoke_paid_subagent(raised, requires_delta=False)

    def test_opaque_body_timeout_does_not_become_actual_process_timeout(self):
        priced = sb._invoke_paid_subagent(
            lambda: {"answer": "candidate", "usage": {"cost_usd": 0.06},
                     "returncode": 124, "timed_out": True}, requires_delta=False)
        self.assertEqual(str(priced.cost.value.amount), "0.06")
        self.assertIsNone(priced.observed_subtotal)
        self.assertIsNone(priced.value.evidence.process)

    def test_claude_invalid_sibling_schema_retains_isolated_dollars(self):
        for extra in ({"usage": {"input_tokens": "bad"}}, {"result": {}},
                      {"is_error": "bad"}, {"api_error_status": "bad"}):
            with self.subTest(extra=extra):
                parsed = sb.parse_claude_cli_json(json.dumps(
                    {"type": "result", "result": "candidate", "total_cost_usd": 0.06, **extra}))
                self.assertEqual(parsed["cost_usd"], 0.06)
                self.assertIsNotNone(parsed["parse_error"])

    def test_claude_unsafe_envelope_and_stream_never_supply_dollars(self):
        terminal = '{"type":"result","result":"candidate","total_cost_usd":0.06}'
        cases = (
            terminal[:-1], terminal + " trailing", terminal + "\n" + terminal,
            terminal + '\n{"type":"assistant","message":{"content":"later"}}',
            '{"type":"assistant","total_cost_usd":0.06}',
            '{"type":"result","result":"candidate","total_cost_usd":0.01,"total_cost_usd":0.06}',
            '{"type":"assistant","type":"result","result":"candidate","total_cost_usd":0.06}',
            '{"type":"result","result":"candidate","total_cost_usd":true}',
            '{"type":"result","result":"candidate","total_cost_usd":"0.06"}',
            '{"type":"result","result":"candidate","total_cost_usd":-0.06}',
            '{"type":"result","result":"candidate","total_cost_usd":NaN}',
            '{"type":"result","result":"candidate","total_cost_usd":1e999}',
            '{"type":"result","result":"candidate"}',
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertIsNone(sb.parse_claude_cli_json(raw)["cost_usd"])

    def test_claude_unrelated_duplicate_keys_keep_documented_stream_behavior(self):
        raw = '{"type":"result","id":"old","id":"new","result":"candidate","total_cost_usd":0.06}'
        parsed = sb.parse_claude_cli_json(raw)
        self.assertEqual(parsed["answer"], "candidate")
        self.assertEqual(parsed["cost_usd"], 0.06)
        self.assertIsNone(parsed["parse_error"])


class PriceRetentionCliTests(unittest.TestCase):
    def fixture(self, root, route, *, mode="timeout", exit_code=0, raw=None):
        _, tasks, run_dir = write_with_skill_task(root)
        first = json.loads(tasks.read_text())
        run_dir = str(Path(run_dir) / "run-1")
        first["run_dir"] = run_dir
        second = dict(first, run_number=2, run_dir=str(Path(run_dir).parent / "run-2"))
        tasks.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
        marker = root / "launches"
        response = ({"answer": "candidate", "usage": {"cost_usd": 0.06},
                     "telemetry_scope": "turn_delta", "returncode": 0, "timed_out": False}
                    if route == "shell" else
                    {"type": "result", "result": "candidate", "total_cost_usd": 0.06,
                     "usage": {"input_tokens": "bad" if mode == "invalid_usage" else 1,
                               "output_tokens": 1}})
        wire = raw if raw is not None else json.dumps(response).encode()
        stub = root / ("agent.py" if route == "shell" else "claude")
        stub.write_text(f'''#!{sys.executable}
import sys, time
from pathlib import Path
sys.stdin.read()
marker = Path({str(marker)!r})
n = len(marker.read_text().splitlines()) if marker.exists() else 0
with marker.open("a") as handle:
    handle.write("started\\n")
sys.stdout.buffer.write({wire!r})
sys.stdout.buffer.flush()
if {mode!r} == "timeout" and n == 0:
    time.sleep(3)
sys.exit({exit_code})
''')
        stub.chmod(0o755)
        flags = (("--agent-cmd", shlex.join([sys.executable, str(stub)]))
                 if route == "shell" else ("--claude-bin", str(stub)))
        command = "run-claude" if route == "native" else "run-subagent"
        return tasks, root / "runs", run_dir, marker, command, flags

    def invoke(self, fixture, *flags):
        tasks, runs, _, _, command, backend = fixture
        return run_cli(command, "--tasks", tasks, "--runs", runs, *backend,
                       "--timeout", "1", "--max-cost-usd", "0.1", *flags)

    def ledger(self, runs):
        paths = list((runs / "spend").glob("*/spend-ceiling.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def test_timeout_retains_floor_refuses_later_call_and_never_publishes_full_cost(self):
        for route in ("shell", "claude", "native"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                fixture = self.fixture(Path(td), route)
                _, runs, run_dir, marker, _, _ = fixture
                code, _, stderr = self.invoke(fixture)
                self.assertEqual(code, 2, stderr)
                self.assertEqual(marker.read_text(), "started\n")
                ledger = self.ledger(runs)
                self.assertEqual(ledger["spent_usd"], "0.06")
                self.assertEqual(ledger["spent_availability"], "partial")
                self.assertEqual([row["state"] for row in ledger["calls"]], ["settled", "not_started"])
                self.assertEqual(ledger["calls"][0]["charge"]["basis"], "unpriced")
                self.assertEqual(ledger["calls"][0]["charge"]["observed_subtotal_usd"], "0.06")
                for name in ("metadata.json", "metrics.json"):
                    artifact = json.loads((runs / run_dir / name).read_text())
                    self.assertIsNone(artifact["cost_normalized"].get("total_cost"))
                    self.assertNotIn("cost_usd", artifact)
                    self.assertFalse(artifact["provider_response_complete"])
                metadata = json.loads((runs / run_dir / "metadata.json").read_text())
                self.assertEqual(metadata["returncode"], 124)
                self.assertTrue(metadata["timed_out"])
                if route == "native":
                    self.assertEqual(metadata["cost_availability"], "partial")
                    self.assertEqual(metadata["observed_subtotal_usd"], 0.06)
                else:
                    diagnostic = metadata["subagent_rejected_calls"][0]
                    self.assertEqual(diagnostic["cost_availability"], "partial")
                    self.assertEqual(diagnostic["reported_cost_usd"], "0.06")
                    self.assertTrue(diagnostic["stdout_utf8_valid"])

    def test_assumption_cannot_reduce_timeout_floor_and_allows_later_admission(self):
        for route in ("shell", "claude", "native"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                fixture = self.fixture(Path(td), route)
                _, runs, run_dir, marker, _, _ = fixture
                code, _, stderr = self.invoke(fixture, "--assumed-cost-per-run-usd", "0.01")
                self.assertEqual(code, 0, stderr)
                self.assertEqual(marker.read_text(), "started\nstarted\n")
                charge = self.ledger(runs)["calls"][0]["charge"]
                self.assertEqual(charge["basis"], "assumed")
                self.assertEqual(charge["amount_usd"], "0.06")
                self.assertEqual(charge["observed_subtotal_usd"], "0.06")
                metadata = json.loads((runs / run_dir / "metadata.json").read_text())
                self.assertIsNone(metadata["cost_normalized"].get("total_cost"))

    def test_exited_claude_invalid_usage_keeps_observed_dollars_and_raw_rejection(self):
        for route in ("claude", "native"):
            for process_code in (0, 7):
                with self.subTest(route=route, process_code=process_code), tempfile.TemporaryDirectory() as td:
                    fixture = self.fixture(Path(td), route, mode="invalid_usage", exit_code=process_code)
                    _, runs, run_dir, _, _, _ = fixture
                    code, _, stderr = self.invoke(fixture, "--max-cost-usd", "0.05")
                    self.assertEqual(code, 2, stderr)
                    charge = self.ledger(runs)["calls"][0]["charge"]
                    self.assertEqual(charge, {"basis": "observed", "amount_usd": "0.06",
                                              "provenance": "provider_reported"})
                    metadata = json.loads((runs / run_dir / "metadata.json").read_text())
                    self.assertEqual(metadata["returncode"], process_code)
                    self.assertFalse(metadata["provider_response_complete"])
                    self.assertEqual(metadata["cost_normalized"]["total_cost"], 0.06)
                    self.assertNotIn("input_tokens", metadata["usage_normalized"])
                    self.assertIn("invalid Claude usage", (runs / run_dir / "output.md").read_text())
                    raw = ((runs / run_dir / "trace.jsonl").read_text() if route == "native" else
                           metadata["subagent_rejected_calls"][0]["raw_response"])
                    self.assertIn('"input_tokens": "bad"', raw)

    def test_invalid_claude_bytes_do_not_become_timeout_dollars(self):
        for route in ("claude", "native"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                fixture = self.fixture(Path(td), route, raw=
                    b'{"type":"result","result":"\xff","total_cost_usd":0.06}')
                code, _, stderr = self.invoke(fixture)
                self.assertEqual(code, 2, stderr)
                charge = self.ledger(fixture[1])["calls"][0]["charge"]
                self.assertIsNone(charge["observed_subtotal_usd"])

    def test_native_capture_failure_propagates_after_retained_price_settlement(self):
        for mode in ("timeout", "invalid_usage"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                fixture = self.fixture(Path(td), "native", mode=mode)
                error = OSError("deferred capture failure")
                with mock.patch("workspace_contracts.capture_workspace_changes", side_effect=error):
                    with self.assertRaises(OSError) as raised:
                        self.invoke(fixture)
                self.assertIs(raised.exception, error)
                charge = self.ledger(fixture[1])["calls"][0]["charge"]
                self.assertEqual(charge["basis"], "unpriced" if mode == "timeout" else "observed")
                self.assertEqual(charge.get("observed_subtotal_usd", charge.get("amount_usd")), "0.06")

    def test_multi_turn_shell_timeout_preserves_only_eligible_delta_floor(self):
        for scope in ("turn_delta", "conversation_cumulative", None, "bad"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as td:
                response = {"answer": "candidate", "usage": {"cost_usd": 0.06}}
                if scope is not None:
                    response["telemetry_scope"] = scope
                fixture = self.fixture(Path(td), "shell", raw=json.dumps(response).encode())
                tasks, runs, run_dir, marker, _, _ = fixture
                row = json.loads(tasks.read_text().splitlines()[0])
                row["turns"] = ["first", "second"]
                tasks.write_text(json.dumps(row) + "\n")
                code, _, stderr = self.invoke(fixture)
                self.assertEqual(code, 2, stderr)
                self.assertEqual(marker.read_text(), "started\n")
                ledger = self.ledger(runs)
                charge = ledger["calls"][0]["charge"]
                self.assertEqual(charge["basis"], "unpriced")
                self.assertEqual(charge["observed_subtotal_usd"], "0.06" if scope == "turn_delta" else None)
                self.assertEqual([row["state"] for row in ledger["calls"]], ["settled", "not_started"])
                turn_metadata = json.loads((runs / run_dir / "turn-1" / "metadata.json").read_text())
                self.assertEqual(turn_metadata["returncode"], 124)
                self.assertTrue(turn_metadata["timed_out"])
                self.assertIsNone(turn_metadata["cost_normalized"].get("total_cost"))
                self.assertEqual(turn_metadata["subagent_paid_evidence"]["reported_cost_usd"], "0.06")


if __name__ == "__main__":
    unittest.main()
