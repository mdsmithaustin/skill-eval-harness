import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from helpers import claude_stream_records, run_cli, write_with_skill_task

import skill_benchmark as sb
import workspace_contracts as wc


class NativeSpendTests(unittest.TestCase):
    def setup_batch(self, root, *, cost=0.6, returncode=0, hang=False, edit=False):
        manifest, tasks, run_dir = write_with_skill_task(root)
        row = json.loads(tasks.read_text())
        row["run_dir"] = str(Path(run_dir) / "run-1")
        second = dict(row, run_number=2, run_dir=str(Path(run_dir) / "run-2"))
        tasks.write_text("\n".join(json.dumps(item) for item in (row, second)) + "\n")
        marker = root / "launches"
        records = claude_stream_records(cost=0.6)
        if cost is None:
            records[-1].pop("total_cost_usd", None)
        else:
            records[-1]["total_cost_usd"] = cost
        stub = root / "claude"
        stub.write_text(f'''#!{sys.executable}
import os, sys, time
from pathlib import Path
sys.stdin.read()
Path({str(root / "child.pid")!r}).write_text(str(os.getpid()))
with Path({str(marker)!r}).open("a") as handle:
    handle.write("started\\n")
if {edit!r}:
    Path("candidate.txt").write_text("paid edit\\n")
if {hang!r}:
    time.sleep(60)
sys.stdout.write({''.join(json.dumps(item) + chr(10) for item in records)!r})
sys.exit({returncode})
''')
        stub.chmod(0o755)
        return manifest, tasks, root / "runs", marker, stub, row, second

    def invoke(self, tasks, runs, stub, *flags):
        return run_cli("run-claude", "--tasks", tasks, "--runs", runs,
                       "--claude-bin", stub, *flags)

    def ledgers(self, runs):
        return [json.loads(path.read_text()) for path in sorted(
            (runs / "spend").glob("*/spend-ceiling.json"))]

    def test_observed_crossing_stops_the_next_child(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, row, second = self.setup_batch(Path(td))
            code, stdout, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "0.5")
            self.assertEqual(code, 2, stderr)
            self.assertEqual(marker.read_text(), "started\n")
            self.assertTrue((runs / row["run_dir"] / "output.md").is_file())
            self.assertFalse((runs / second["run_dir"] / "output.md").exists())
            ledger = self.ledgers(runs)[0]
            self.assertEqual(ledger["spent_usd"], "0.6")
            self.assertEqual([item["state"] for item in ledger["calls"]],
                             ["settled", "not_started"])
            self.assertIn("spend-ceiling.json", stdout)

    def test_unavailable_directory_sync_does_not_block_admission(self):
        for unavailable in ("open", "fsync"):
            with self.subTest(unavailable=unavailable), tempfile.TemporaryDirectory() as td:
                _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
                original_open, original_fsync = os.open, os.fsync

                def open_file(path, flags, *args, mode=unavailable, real_open=original_open,
                              spend_root=runs / "spend", **kwargs):
                    if mode == "open" and isinstance(path, Path) and path.parent == spend_root:
                        raise PermissionError("directory descriptors unavailable")
                    return real_open(path, flags, *args, **kwargs)

                def sync_file(descriptor, mode=unavailable, real_fsync=original_fsync):
                    if mode == "fsync" and stat.S_ISDIR(os.fstat(descriptor).st_mode):
                        raise OSError("directory fsync unavailable")
                    return real_fsync(descriptor)

                with mock.patch("spend_runtime.os.open", side_effect=open_file), \
                        mock.patch("spend_runtime.os.fsync", side_effect=sync_file):
                    code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "0.5")
                self.assertEqual(code, 2, stderr)
                self.assertEqual(marker.read_text(), "started\n")
                self.assertEqual(self.ledgers(runs)[0]["spent_usd"], "0.6")

    def test_file_sync_and_replace_failure_never_start_a_child(self):
        for operation in ("fsync", "replace"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as td:
                _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
                code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "0")
                self.assertEqual(code, 2, stderr)
                with mock.patch(f"spend_runtime.os.{operation}", side_effect=OSError("file publication failed")):
                    with self.assertRaisesRegex(OSError, "file publication failed"):
                        self.invoke(tasks, runs, stub, "--max-cost-usd", "0.5")
                self.assertFalse(marker.exists())

    def test_zero_starts_no_child_and_preserves_the_plan(self):
        with tempfile.TemporaryDirectory() as td:
            manifest, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
            with mock.patch.dict(sb.WORKSPACE_BUILDERS, {"claude": mock.Mock(side_effect=AssertionError("workspace built"))}), \
                    mock.patch.object(wc, "snapshot_workspace", side_effect=AssertionError("baseline created")):
                code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "0")
            self.assertEqual(code, 2, stderr)
            self.assertFalse(marker.exists())
            ledger = self.ledgers(runs)[0]
            self.assertEqual([item["state"] for item in ledger["calls"]],
                             ["not_started", "not_started"])
            self.assertEqual(list(runs.rglob("output.md")), [])
            self.assertEqual(list(runs.rglob("metadata.json")), [])
            code, output, stderr = run_cli("benchmark", manifest, "--runs", runs)
            self.assertEqual(code, 0, stderr)
            report = json.loads(output)
            self.assertEqual(report["spend_invocations"][0]["invocation_id"], ledger["invocation_id"])
            self.assertNotEqual(report["availability"], "complete")

    def test_missing_cost_and_failure_close_admission(self):
        for returncode in (0, 7):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as td:
                _, tasks, runs, marker, stub, row, _ = self.setup_batch(
                    Path(td), cost=None, returncode=returncode)
                code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "10")
                self.assertEqual(code, 2, stderr)
                self.assertEqual(marker.read_text(), "started\n")
                ledger = self.ledgers(runs)[0]
                self.assertEqual(ledger["spent_availability"], "partial")
                self.assertEqual(ledger["calls"][0]["charge"]["basis"], "unpriced")
                self.assertTrue((runs / row["run_dir"] / "trace.jsonl").is_file())

    def test_known_missing_cost_requires_assumption_before_a_paid_start(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
            code, _, stderr = run_cli("run-codex", "--tasks", tasks, "--runs", runs,
                                      "--codex-cmd", stub, "--max-cost-usd", "1")
            self.assertEqual(code, 1)
            self.assertIn("requires --assumed-cost-per-run-usd", stderr)
            self.assertFalse(marker.exists())
            self.assertFalse(runs.exists())
            code, _, stderr = run_cli("run-codex", "--tasks", tasks, "--runs", runs,
                                      "--codex-cmd", stub, "--max-cost-usd", "0")
            self.assertEqual(code, 2, stderr)
            self.assertEqual([item["state"] for item in self.ledgers(runs)[0]["calls"]],
                             ["not_started", "not_started"])

    def test_assumption_prices_missing_cost_and_stops_the_next_child(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td), cost=None)
            code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "0.2",
                                          "--assumed-cost-per-run-usd", "0.3")
            self.assertEqual(code, 2, stderr)
            self.assertEqual(marker.read_text(), "started\n")
            self.assertEqual(self.ledgers(runs)[0]["calls"][0]["charge"]["basis"], "assumed")
            self.assertEqual(self.ledgers(runs)[0]["spent_usd"], "0.3")

    def test_observed_failure_and_artifact_failure_keep_the_charge(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td), returncode=7)
            code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "0.5")
            self.assertEqual(code, 2, stderr)
            self.assertEqual(marker.read_text(), "started\n")
            self.assertEqual(self.ledgers(runs)[0]["spent_usd"], "0.6")
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
            with mock.patch("skill_benchmark.write_runner_outcome", side_effect=OSError("artifact write failed")):
                with self.assertRaisesRegex(OSError, "artifact write failed"):
                    self.invoke(tasks, runs, stub, "--max-cost-usd", "1")
            self.assertEqual(marker.read_text(), "started\n")
            self.assertEqual(self.ledgers(runs)[0]["spent_usd"], "0.6")
            self.assertEqual(self.ledgers(runs)[0]["calls"][1]["state"], "planned")

    def test_capture_error_retains_observed_cost_before_capture(self):
        for assumption in (None, "0.1"):
            with self.subTest(assumption=assumption), tempfile.TemporaryDirectory() as td:
                _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
                error = OSError("workspace evidence failed")
                at_capture = []

                def capture(*args, at_capture=at_capture, runs=runs, error=error, **kwargs):
                    at_capture.append(self.ledgers(runs)[0])
                    raise error

                flags = ("--assumed-cost-per-run-usd", assumption) if assumption else ()
                with mock.patch("workspace_contracts.capture_workspace_changes", side_effect=capture), \
                        mock.patch("skill_benchmark.write_runner_outcome") as writer:
                    with self.assertRaises(OSError) as raised:
                        self.invoke(tasks, runs, stub, "--max-cost-usd", "1", *flags)
                self.assertIs(raised.exception, error)
                self.assertEqual(marker.read_text(), "started\n")
                self.assertEqual(writer.call_count, 0)
                expected_charge = {"basis": "observed", "amount_usd": "0.6",
                                   "provenance": "provider_reported"}
                self.assertEqual(len(at_capture), 1)
                self.assertEqual(at_capture[0]["calls"][0].get("charge"), expected_charge)
                self.assertEqual(at_capture[0]["spent_usd"], "0.6")
                ledger = self.ledgers(runs)[0]
                self.assertEqual(ledger["calls"][0]["charge"], expected_charge)
                self.assertEqual(ledger["spent_usd"], "0.6")
                self.assertEqual([item["state"] for item in ledger["calls"]],
                                 ["settled", "planned"])

    def test_cleanup_error_retains_observed_cost(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
            error = OSError("workspace cleanup failed")
            original_cleanup = tempfile.TemporaryDirectory.cleanup
            cleaned = []

            def cleanup(directory):
                original_cleanup(directory)
                path = Path(directory.name)
                cleaned.append(path)
                if path.name.startswith("claude-ws-"):
                    self.assertEqual(self.ledgers(runs)[0]["spent_usd"], "0.6")
                    raise error

            with mock.patch.object(tempfile.TemporaryDirectory, "cleanup", cleanup), \
                    mock.patch("skill_benchmark.write_runner_outcome") as writer:
                with self.assertRaises(OSError) as raised:
                    self.invoke(tasks, runs, stub, "--max-cost-usd", "1",
                                "--assumed-cost-per-run-usd", "0.1")
            self.assertIs(raised.exception, error)
            self.assertEqual(marker.read_text(), "started\n")
            self.assertEqual(writer.call_count, 0)
            self.assertEqual(len(cleaned), 3)
            self.assertEqual([path.exists() for path in cleaned], [False, False, False])
            ledger = self.ledgers(runs)[0]
            self.assertEqual(ledger["calls"][0]["charge"],
                             {"basis": "observed", "amount_usd": "0.6",
                              "provenance": "provider_reported"})
            self.assertEqual([item["state"] for item in ledger["calls"]],
                             ["settled", "planned"])

    def test_settlement_publication_error_skips_capture_and_cleans_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
            error = OSError("settlement publication failed")
            original_replace = os.replace
            original_snapshot = wc.snapshot_workspace
            directories = []
            settlements = []

            def snapshot(ws, baseline):
                directories.extend((ws, baseline.parent))
                return original_snapshot(ws, baseline)

            def replace(source, destination):
                if Path(destination).name == "spend-ceiling.json":
                    proposed = json.loads(Path(source).read_text())
                    if proposed["calls"][0]["state"] == "settled":
                        settlements.append(proposed["calls"][0]["charge"])
                        raise error
                return original_replace(source, destination)

            with mock.patch.object(wc, "snapshot_workspace", side_effect=snapshot), \
                    mock.patch("spend_runtime.os.replace", side_effect=replace), \
                    mock.patch.object(wc, "capture_workspace_changes") as capture, \
                    mock.patch("skill_benchmark.write_runner_outcome") as writer:
                with self.assertRaises(OSError) as raised:
                    self.invoke(tasks, runs, stub, "--max-cost-usd", "1")
            self.assertIs(raised.exception, error)
            self.assertEqual(marker.read_text(), "started\n")
            self.assertEqual(settlements, [{"basis": "observed", "amount_usd": "0.6",
                                            "provenance": "provider_reported"}])
            self.assertEqual((capture.call_count, writer.call_count), (0, 0))
            self.assertEqual(len(directories), 2)
            self.assertEqual([path.exists() for path in directories], [False, False])
            ledger = self.ledgers(runs)[0]
            self.assertEqual([item["state"] for item in ledger["calls"]],
                             ["in_flight", "planned"])
            self.assertEqual((ledger["spent_usd"], ledger["spent_availability"]), ("0", "partial"))

    def test_provider_interruption_skips_capture_and_cleans_group_and_workspace(self):
        for capped in (False, True):
            with self.subTest(capped=capped), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, tasks, runs, marker, stub, _, _ = self.setup_batch(root, hang=True)
                error = KeyboardInterrupt("provider interrupted")
                original_communicate = subprocess.Popen.communicate
                original_snapshot = wc.snapshot_workspace
                directories = []
                interrupted = []

                def snapshot(ws, baseline, directories=directories, original_snapshot=original_snapshot):
                    directories.extend((ws, baseline.parent))
                    return original_snapshot(ws, baseline)

                def communicate(process, *args, interrupted=interrupted, marker=marker,
                                error=error, original_communicate=original_communicate, **kwargs):
                    try:
                        return original_communicate(process, *args, **kwargs)
                    except subprocess.TimeoutExpired:
                        if marker.exists() and not interrupted:
                            interrupted.append(process.pid)
                            raise error
                        raise

                flags = ("--max-cost-usd", "1") if capped else ()
                with mock.patch.object(wc, "snapshot_workspace", side_effect=snapshot), \
                        mock.patch.object(subprocess.Popen, "communicate", communicate), \
                        mock.patch.object(wc, "capture_workspace_changes") as capture:
                    with self.assertRaises(KeyboardInterrupt) as raised:
                        self.invoke(tasks, runs, stub, *flags)
                self.assertIs(raised.exception, error)
                self.assertEqual(marker.read_text(), "started\n")
                self.assertEqual(interrupted, [int((root / "child.pid").read_text())])
                with self.assertRaises(ProcessLookupError):
                    os.kill(interrupted[0], 0)
                self.assertEqual(capture.call_count, 0)
                self.assertEqual(len(directories), 2)
                self.assertEqual([path.exists() for path in directories], [False, False])
                if capped:
                    ledger = self.ledgers(runs)[0]
                    self.assertEqual(ledger["calls"][0]["charge"],
                                     {"basis": "unpriced", "reason": "invocation_raised",
                                      "observed_subtotal_usd": None})
                    self.assertEqual(ledger["calls"][1]["state"], "planned")
                else:
                    self.assertFalse((runs / "spend").exists())

    def test_pricing_error_skips_capture_after_provider_return(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
            error = ValueError("pricing failed")
            original_snapshot = wc.snapshot_workspace
            directories = []

            def snapshot(ws, baseline):
                directories.extend((ws, baseline.parent))
                return original_snapshot(ws, baseline)

            with mock.patch.object(wc, "snapshot_workspace", side_effect=snapshot), \
                    mock.patch.object(sb, "Priced", side_effect=error), \
                    mock.patch.object(wc, "capture_workspace_changes") as capture:
                with self.assertRaises(ValueError) as raised:
                    self.invoke(tasks, runs, stub, "--max-cost-usd", "1")
            self.assertIs(raised.exception, error)
            self.assertEqual(marker.read_text(), "started\n")
            self.assertEqual(capture.call_count, 0)
            self.assertEqual(len(directories), 2)
            self.assertEqual([path.exists() for path in directories], [False, False])
            self.assertEqual(self.ledgers(runs)[0]["calls"][0]["charge"],
                             {"basis": "unpriced", "reason": "invocation_raised",
                              "observed_subtotal_usd": None})

    def test_returned_outcomes_capture_before_deletion_and_write_sidecars_afterward(self):
        original_build = sb.registered_workspace_builder("claude")
        for mode in ("completed", "failed", "timeout", "spawn_failed", "capture_failed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                _, tasks, runs, marker, stub, row, _ = self.setup_batch(
                    Path(td), returncode=7 if mode == "failed" else 0,
                    hang=mode == "timeout", edit=True)
                if mode == "spawn_failed":
                    stub.unlink()
                original_capture = wc.capture_workspace_changes
                original_writer = sb.write_runner_outcome
                captured = []
                sidecars = []
                built = []

                def build(pt, ws, built=built):
                    workspace = original_build(pt, ws)
                    built.append(workspace.attestation)
                    return workspace

                def capture(baseline, ws, changes, captured=captured,
                            original_capture=original_capture, **kwargs):
                    self.assertTrue(ws.is_dir())
                    captured.append(ws)
                    return original_capture(baseline, ws, changes, **kwargs)

                def write(base, outcome, *, sidecars, captured=captured, retained=sidecars,
                          original_writer=original_writer):
                    self.assertFalse(captured[-1].exists())
                    self.assertTrue((sidecars / "workspace-changes.json").is_file())
                    retained.append(sidecars)
                    return original_writer(base, outcome, sidecars=sidecars)

                capture_error = (mock.patch.object(wc, "_stage_changes", side_effect=OSError("evidence failed"))
                                 if mode == "capture_failed" else nullcontext())
                with mock.patch.object(wc, "capture_workspace_changes", side_effect=capture), \
                        mock.patch.object(sb, "write_runner_outcome", side_effect=write), \
                        mock.patch.dict(sb.WORKSPACE_BUILDERS, {"claude": build}), capture_error:
                    code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "0.5", "--timeout", "1")
                self.assertEqual(code, 0 if mode == "spawn_failed" else 2, stderr)
                self.assertEqual(marker.read_text() if marker.exists() else "", "" if mode == "spawn_failed" else "started\n")
                base = runs / row["run_dir"]
                self.assertTrue(sb.artifact_commit_valid(base))
                metadata = sb.read_metrics_base(base)
                expected_calls = 2 if mode == "spawn_failed" else 1
                self.assertEqual(len(built), expected_calls)
                self.assertEqual(metadata["fixture_tree_hash"], built[0].fixture_tree_hash)
                self.assertEqual(metadata["skill_tree_hash"], built[0].mounted_skill_tree_hash)
                manifest = json.loads((base / "workspace-changes.json").read_text())
                self.assertEqual((manifest["captured"], metadata["workspace_changes_captured"]),
                                 (mode != "capture_failed", mode != "capture_failed"))
                if mode not in {"spawn_failed", "capture_failed"}:
                    self.assertIn("+paid edit", (base / "candidate.patch").read_text())
                if mode == "timeout":
                    self.assertEqual((metadata["timed_out"], metadata["returncode"]), (True, 124))
                if mode == "capture_failed":
                    self.assertEqual(manifest["capture_error"],
                                     {"stage": "evidence", "reason": "OSError: evidence failed"})
                    self.assertEqual(self.ledgers(runs)[0]["spent_usd"], "0.6")
                if mode == "spawn_failed":
                    self.assertEqual(self.ledgers(runs)[0]["calls"][0]["charge"],
                                     {"basis": "no_model_spend", "reason": "spawn_failed_before_process"})
                self.assertEqual(len(sidecars), expected_calls)
                self.assertEqual([path.exists() for path in sidecars], [False] * len(sidecars))

    def test_invalid_policy_does_not_launch_or_write(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
            for value in ("-1", "nan", "inf"):
                with self.subTest(value=value):
                    code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", value)
                    self.assertEqual(code, 2)
                    self.assertIn("max-cost-usd", stderr)
            code, _, stderr = self.invoke(tasks, runs, stub, "--assumed-cost-per-run-usd", "0.1")
            self.assertEqual(code, 1)
            self.assertIn("requires --max-cost-usd", stderr)
            self.assertFalse(marker.exists())
            self.assertFalse(runs.exists())

    def test_two_invocations_preserve_ledgers_and_design(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td))
            self.assertEqual(self.invoke(tasks, runs, stub, "--max-cost-usd", "0.5")[0], 2)
            design = (runs / "answer-design.json").read_bytes()
            self.assertEqual(self.invoke(tasks, runs, stub, "--max-cost-usd", "0.5")[0], 2)
            ledgers = self.ledgers(runs)
            self.assertEqual(len(ledgers), 2)
            self.assertNotEqual(ledgers[0]["invocation_id"], ledgers[1]["invocation_id"])
            self.assertEqual((runs / "answer-design.json").read_bytes(), design)
            self.assertEqual(marker.read_text(), "started\nstarted\n")
            self.assertFalse((runs / "spend-ceiling.json").exists())
            manifest = Path(td) / "repo/evals/shared-benchmark.json"
            code, output, stderr = run_cli("cost-summary", "--manifest", manifest, "--runs", runs)
            self.assertEqual(code, 0, stderr)
            self.assertEqual(len(json.loads(output)["spend_invocations"]), 2)

    def test_recovery_anywhere_in_batch_rejects_before_writes(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, row, second = self.setup_batch(Path(td))
            second["recovery"] = {
                "checkpoint_path": "checkpoint.json", "expected_content": "{}",
                "recovery_prompt": "RECOVER", "refusal_prompt": "REFUSE",
                "forbidden_path": "forbidden.txt", "match": "json"}
            tasks.write_text("\n".join(json.dumps(item) for item in (row, second)) + "\n")
            existing = runs / second["run_dir"] / "recovery.json"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"prior evidence")
            before = {path.relative_to(runs): path.read_bytes() for path in runs.rglob("*") if path.is_file()}
            code, _, stderr = self.invoke(tasks, runs, stub, "--max-cost-usd", "1")
            self.assertEqual(code, 2)
            self.assertIn("capped recovery", stderr)
            self.assertFalse(marker.exists())
            self.assertEqual({path.relative_to(runs): path.read_bytes() for path in runs.rglob("*") if path.is_file()}, before)

    def test_uncapped_keeps_existing_artifacts_without_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, row, second = self.setup_batch(Path(td))
            self.assertEqual(self.invoke(tasks, runs, stub)[0], 0)
            self.assertEqual(marker.read_text(), "started\nstarted\n")
            self.assertTrue((runs / row["run_dir"] / "output.md").is_file())
            self.assertTrue((runs / second["run_dir"] / "output.md").is_file())
            self.assertFalse((runs / "spend").exists())

    def test_kill_after_admission_retains_unresolved_call(self):
        with tempfile.TemporaryDirectory() as td:
            _, tasks, runs, marker, stub, _, _ = self.setup_batch(Path(td), hang=True)
            process = subprocess.Popen([
                sys.executable, "skill_benchmark.py", "run-claude", "--tasks", str(tasks),
                "--runs", str(runs), "--claude-bin", str(stub), "--max-cost-usd", "1"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists(), process.communicate() if process.poll() is not None else "child did not start")
                ledger = self.ledgers(runs)[0]
                self.assertEqual(ledger["calls"][0]["state"], "in_flight")
                process.kill()
                process.communicate(timeout=5)
                self.assertEqual(self.ledgers(runs)[0]["calls"][0]["state"], "in_flight")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
                # The killed harness cannot clean the isolated child's group.
                for pidfile in Path(td).glob("child.pid"):
                    try:
                        os.killpg(int(pidfile.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
