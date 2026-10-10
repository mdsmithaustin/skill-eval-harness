import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helpers import claude_stream_records, run_cli, write_with_skill_task


class NativeSpendTests(unittest.TestCase):
    def setup_batch(self, root, *, cost=0.6, returncode=0, hang=False):
        manifest, tasks, run_dir = write_with_skill_task(root)
        row = json.loads(tasks.read_text())
        second = dict(row, run_number=2, run_dir=run_dir + "/run-2")
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
