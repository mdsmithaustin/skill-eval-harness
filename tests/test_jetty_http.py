import base64
import copy
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from helpers import (
    jetty_payload,
    jetty_task_upload,
    make_eval_repo,
)

import skill_benchmark as sb

CLI = Path(sb.__file__).resolve()


class JettyFixture:
    def __init__(self, *, submit=None, poll=None, detail=None, download_error=False, hold_submit=False):
        self.submit = submit
        self.poll = poll
        self.detail = detail
        self.download_error = download_error
        self.hold_submit = hold_submit
        self.requests = []
        self.submitted = threading.Event()
        self.release = threading.Event()
        self.serial = 0

    def response(self, method, path, body):
        self.requests.append({"method": method, "path": path, "body": body.decode(errors="replace"), "body_base64": base64.b64encode(body).decode()})
        if path == "/api/v1/sandbox/upload":
            archive = body.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]
            with zipfile.ZipFile(io.BytesIO(archive)) as uploaded:
                self.requests[-1]["zip_contents"] = {name: uploaded.read(name).decode() for name in uploaded.namelist()}
            return 200, {"file_paths": ["fixture-bundle"]}
        if path == "/v1/chat/completions":
            self.serial += 1
            self.submitted.set()
            if self.hold_submit and not self.release.wait(10):
                return 500, {"error": "fixture deadline"}
            if self.submit == "lost":
                return None, None
            if isinstance(self.submit, list):
                status = self.submit[min(self.serial - 1, len(self.submit) - 1)]
                return status, ({"trajectory_id": "fixture-trajectory", "storage_path": "c/t/0000"} if status == 200 else {"error": "scripted rejection"})
            if self.submit is not None:
                return 200, copy.deepcopy(self.submit)
            return 200, {"trajectory_id": "fixture-trajectory", "storage_path": "c/t/0000"}
        if path.startswith("/api/v1/db/trajectory/"):
            return 200, copy.deepcopy(self.poll) if self.poll is not None else {
                "trajectory_id": "fixture-trajectory", "status": "completed", "storage_path": "c/t/0000", "cost_usd": 0.6}
        if path.startswith("/api/v1/trajectory/"):
            return 200, copy.deepcopy(self.detail) if self.detail is not None else {
                "trajectory_id": "fixture-trajectory", "status": "completed", "storage_path": "c/t/0000",
                "steps": {"run": {"outputs": {"success": True, "usage": {"cost_usd": 0.6},
                            "results_files": [{"path": "c/t/0000/fixture-trajectory.run.0000.app--results--output.md", "content_type": "text/markdown"}]}}}}
        if path.startswith("/api/v1/file/"):
            return (404, {"error": "download missing"}) if self.download_error else (200, b"fixture answer\n")
        return 404, {"error": "unexpected fixture route"}

    def counts(self):
        return {name: sum(r["path"].startswith(prefix) for r in self.requests) for name, prefix in (
            ("upload", "/api/v1/sandbox/upload"), ("submit", "/v1/chat/completions"),
            ("poll", "/api/v1/db/trajectory/"), ("detail", "/api/v1/trajectory/"), ("download", "/api/v1/file/"))}


@contextmanager
def jetty_server(fixture):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.respond()

        def do_GET(self):
            self.respond()

        def respond(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            status, value = fixture.response(self.command, self.path, body)
            if status is None:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            data = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        fixture.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


class JettyHttpTests(unittest.TestCase):
    def child(self, root, rows, base_url, *flags, wait=True):
        payloads = root / "payloads.jsonl"
        payloads.write_text("".join(json.dumps(row) + "\n" for row in rows))
        out = root / "results.jsonl"
        command = [sys.executable, str(CLI), "run-jetty", "--payloads", str(payloads), "--out", str(out),
                   "--timeout", "1", "--poll-interval", "0.02", *flags]
        env = {k: v for k, v in os.environ.items() if not any(word in k.upper() for word in ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL"))}
        home = root / "home"
        home.mkdir(exist_ok=True)
        env.update(JETTY_API_TOKEN="loopback-fixture", JETTY_BASE_URL=base_url, HOME=str(home))
        process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        if not wait:
            return process, out
        try:
            stdout, stderr = process.communicate(timeout=15)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
            raise
        self.persist(root, {"command": command, "pid": process.pid, "pgid": process.pid,
                            "exit": process.returncode, "stdout": stdout, "stderr": stderr})
        return process.returncode, out, stdout, stderr

    def persist(self, root, data):
        receipt = os.environ.get("JETTY_PROOF_CHILDREN")
        if receipt:
            data["files"] = {str(p.relative_to(root)): p.read_text(errors="replace")
                             for p in root.rglob("*") if p.is_file() and p.name != "payloads.jsonl"}
            with Path(receipt).with_name("http-results.jsonl").open("a") as handle:
                handle.write(json.dumps(data) + "\n")

    def ledger_paths(self, out):
        return sorted(Path(str(out) + ".spend").glob("*/spend-ceiling.json"))

    def ledger(self, out):
        return json.loads(self.ledger_paths(out)[-1].read_text())

    def test_fresh_zero_has_zero_http_and_real_admitted_child_runs_all_routes(self):
        fixture = JettyFixture()
        with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
            root = Path(td)
            row = jetty_payload(files=[jetty_task_upload({"causal_marker": "zero-and-paid"})])
            code, out, _, stderr = self.child(root, [row], url, "--max-cost-usd", "0")
            self.assertEqual(code, 2, stderr)
            self.assertEqual(fixture.requests, [])
            self.assertFalse(Path(str(out) + ".attempts.json").exists())
            [refusal] = sb.load_jsonl(out)
            self.assertEqual(refusal["invocation_state"], "not_started")
            self.assertFalse(any(key in refusal for key in ("returncode", "status", "trajectory_id", "jetty")))
            code, out, _, stderr = self.child(root, [row], url, "--max-cost-usd", "1", "--assumed-cost-per-run-usd", "0.1")
            self.assertEqual(code, 0, stderr)
            self.assertEqual(fixture.counts(), {"upload": 1, "submit": 1, "poll": 1, "detail": 1, "download": 1})
            self.assertIn("zero-and-paid", fixture.requests[0]["zip_contents"]["tasks/task.json"])
            [record] = sb.load_jsonl(out)
            self.assertEqual(record["artifacts"][0]["content"], "fixture answer\n")
            charge = next(json.loads(p.read_text())["calls"][0]["charge"] for p in self.ledger_paths(out) if json.loads(p.read_text())["calls"][0]["state"] == "settled")
            self.assertEqual(charge, {"basis": "assumed", "amount_usd": "0.6", "reason": "jetty_billing_finality_unavailable", "observed_subtotal_usd": "0.6"})
            self.persist(root, {"fixture_requests": fixture.requests})

    def test_unknown_resume_refusal_and_admitted_replacement_preserve_exact_history(self):
        fixture = JettyFixture(submit="lost")
        with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
            root = Path(td)
            row = jetty_payload(files=[jetty_task_upload()])
            code, out, _, _ = self.child(root, [row], url, "--max-cost-usd", "1")
            self.assertEqual(code, 2)
            journal = Path(str(out) + ".attempts.json")
            old = journal.read_bytes()
            ledgers = {p: p.read_bytes() for p in self.ledger_paths(out)}
            code, _, _, _ = self.child(root, [row], url, "--max-cost-usd", "0")
            self.assertEqual(code, 1)
            self.assertEqual(journal.read_bytes(), old)
            code, _, _, _ = self.child(root, [row], url, "--max-cost-usd", "0", "--resubmit-unknown")
            self.assertEqual(code, 2)
            self.assertEqual(journal.read_bytes(), old)
            self.assertEqual(fixture.counts(), {"upload": 1, "submit": 1, "poll": 0, "detail": 0, "download": 0})
            fixture.submit = None
            code, _, _, stderr = self.child(root, [row], url, "--resubmit-unknown", "--max-cost-usd", "1", "--assumed-cost-per-run-usd", "0.1")
            self.assertEqual(code, 0, stderr)
            digest = row["harness"]["jetty_task_contract_sha256"]
            first, second = json.loads(journal.read_text())["attempts"][digest]["submission_attempts"]
            self.assertEqual(first, json.loads(old)["attempts"][digest]["submission_attempts"][0])
            self.assertNotEqual(first["ref"], second["ref"])
            self.assertEqual(second["replaces_unknown"], first["ref"])
            self.assertEqual({p: p.read_bytes() for p in ledgers}, ledgers)
            self.assertEqual(fixture.counts(), {"upload": 1, "submit": 2, "poll": 1, "detail": 1, "download": 1})
            self.persist(root, {"fixture_requests": fixture.requests})

    def test_pending_reconciles_at_zero_after_local_sources_are_deleted_then_replays_without_http(self):
        fixture = JettyFixture(poll={"trajectory_id": "fixture-trajectory", "status": "running", "cost_usd": 0.6})
        with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
            root = Path(td)
            local = root / "task.json"
            local.write_text('{"causal_marker":"uploaded-before-removal"}')
            upload = jetty_task_upload()
            upload.pop("content")
            upload["local_path"] = str(local)
            row = jetty_payload(files=[upload])
            code, out, _, stderr = self.child(root, [row], url, "--max-cost-usd", "1", "--poll-interval", "2")
            self.assertEqual(code, 2, stderr)
            self.assertEqual(fixture.counts(), {"upload": 1, "submit": 1, "poll": 1, "detail": 0, "download": 0})
            local.unlink()
            old_ledgers = {p: p.read_bytes() for p in self.ledger_paths(out)}
            fixture.poll = None
            fixture.requests.clear()
            code, _, _, stderr = self.child(root, [row], url, "--max-cost-usd", "0")
            self.assertEqual(code, 0, stderr)
            self.assertEqual(fixture.counts(), {"upload": 0, "submit": 0, "poll": 1, "detail": 1, "download": 1})
            self.assertTrue(any(json.loads(p.read_text())["calls"] == [] for p in self.ledger_paths(out)))
            self.assertEqual({p: p.read_bytes() for p in old_ledgers}, old_ledgers)
            journal = Path(str(out) + ".attempts.json")
            before_journal = journal.read_bytes()
            before_output = out.read_bytes()
            fixture.requests.clear()
            code, _, _, stderr = self.child(root, [row], url, "--max-cost-usd", "0")
            self.assertEqual(code, 0, stderr)
            self.assertEqual(fixture.requests, [])
            self.assertEqual(journal.read_bytes(), before_journal)
            self.assertEqual(out.read_bytes(), before_output)
            self.persist(root, {"fixture_requests": fixture.requests})

    def test_invalid_later_row_prevents_all_http_output_and_ledger_writes(self):
        fixture = JettyFixture()
        with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
            root = Path(td)
            valid = jetty_payload(files=[jetty_task_upload()])
            invalid = jetty_payload(harness={"case_id": "case-2", "run_dir": "case-2/with_skill"})
            invalid["jetty_request"]["model"] = "mutated"
            out = root / "results.jsonl"
            out.write_bytes(b"preserved prior output\n")
            code, _, _, stderr = self.child(root, [valid, invalid], url, "--max-cost-usd", "1")
            self.assertEqual(code, 1, stderr)
            self.assertIn("Jetty model-visible task/runbook/upload plan changed after attestation", stderr)
            self.assertEqual(out.read_bytes(), b"preserved prior output\n")
            self.assertEqual(fixture.requests, [])
            self.assertFalse(Path(str(out) + ".spend").exists())
            self.assertFalse(Path(str(out) + ".attempts.json").exists())

    def test_terminal_failure_and_download_failure_retain_only_safe_floor(self):
        for failure in ("terminal", "download", "tokens"):
            with self.subTest(failure=failure):
                poll = {"trajectory_id": "fixture-trajectory", "status": "failed" if failure == "terminal" else "completed", "storage_path": "c/t/0000", "cost_usd": 0.6}
                if failure == "tokens":
                    poll["input_tokens"] = "invalid"
                fixture = JettyFixture(poll=poll, download_error=failure == "download")
                with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
                    root = Path(td)
                    code, out, _, stderr = self.child(root, [jetty_payload()], url, "--max-cost-usd", "1")
                    self.assertEqual(code, 2, stderr)
                    self.assertEqual(self.ledger(out)["calls"][0]["charge"], {"basis": "unpriced", "reason": "jetty_billing_finality_unavailable", "observed_subtotal_usd": "0.6"})
                    expected = {"upload": 0, "submit": 1, "poll": 1, "detail": 0 if failure == "terminal" else 1, "download": 0 if failure == "terminal" else 1}
                    self.assertEqual(fixture.counts(), expected)
                    self.persist(root, {"fixture_requests": fixture.requests})

    def test_transport_retries_keep_one_identity_and_5xx_does_not_duplicate(self):
        for statuses in ([429, 429, 200], [429, 429, 429], [500]):
            with self.subTest(statuses=statuses):
                fixture = JettyFixture(submit=statuses)
                with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
                    root = Path(td)
                    code, out, _, stderr = self.child(root, [jetty_payload()], url, "--max-cost-usd", "1")
                    self.assertEqual(code, 2, stderr)
                    reconciled = int(statuses[-1] == 200)
                    self.assertEqual(fixture.counts(), {"upload": 0, "submit": len(statuses), "poll": reconciled, "detail": reconciled, "download": reconciled})
                    digest = jetty_payload()["harness"]["jetty_task_contract_sha256"]
                    attempts = json.loads(Path(str(out) + ".attempts.json").read_text())["attempts"][digest]["submission_attempts"]
                    self.assertEqual(len(attempts), 1)
                    self.assertEqual(self.ledger(out)["calls"][0]["charge"]["basis"], "unpriced")
                    self.persist(root, {"fixture_requests": fixture.requests})

    def test_killed_child_at_submitting_boundary_remains_unknown_without_resubmit(self):
        fixture = JettyFixture(hold_submit=True)
        with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
            root = Path(td)
            row = jetty_payload()
            process, out = self.child(root, [row], url, "--max-cost-usd", "1", wait=False)
            try:
                self.assertTrue(fixture.submitted.wait(5))
                journal = Path(str(out) + ".attempts.json")
                digest = row["harness"]["jetty_task_contract_sha256"]
                self.assertEqual(sb.JettyAttemptJournal(journal).entry(digest)["state"], "submitting")
                self.assertEqual(self.ledger(out)["calls"][0]["state"], "in_flight")
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, -signal.SIGKILL)
                before = journal.read_bytes()
                fixture.release.set()
                fixture.hold_submit = False
                code, _, _, _ = self.child(root, [row], url, "--max-cost-usd", "0")
                self.assertEqual(code, 1)
                self.assertEqual(journal.read_bytes(), before)
                self.assertEqual(fixture.counts(), {"upload": 0, "submit": 1, "poll": 0, "detail": 0, "download": 0})
                self.persist(root, {"killed_pid": process.pid, "pgid": process.pid, "exit": process.returncode, "stdout": stdout, "stderr": stderr, "fixture_requests": fixture.requests})
            finally:
                fixture.release.set()
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=5)

    def test_uncapped_import_and_benchmark_keep_answer_and_unavailable_complete_price(self):
        fixture = JettyFixture()
        with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
            root = Path(td)
            code, out, _, stderr = self.child(root, [jetty_payload()], url)
            self.assertEqual(code, 0, stderr)
            self.assertFalse(Path(str(out) + ".spend").exists())
            manifest = make_eval_repo(root / "manifest")
            runs = root / "runs"
            imported = subprocess.run([sys.executable, str(CLI), "import-jetty-results", "--manifest", str(manifest), "--jetty-runs", str(out), "--runs", str(runs)], cwd=root, capture_output=True, text=True, timeout=10, start_new_session=True, check=False)
            self.assertEqual(imported.returncode, 0, imported.stderr)
            metadata = json.loads((runs / "case-1/with_skill/metadata.json").read_text())
            self.assertEqual(metadata["observed_subtotal_usd"], "0.6")
            self.assertEqual(metadata["cost_reason"], "jetty_billing_finality_unavailable")
            self.assertEqual((runs / "case-1/with_skill/output.md").read_text(), "fixture answer\n")
            report = subprocess.run([sys.executable, str(CLI), "benchmark", str(manifest), "--runs", str(runs)], cwd=root, capture_output=True, text=True, timeout=15, start_new_session=True, check=False)
            self.assertEqual(report.returncode, 0, report.stderr)
            parsed_report = json.loads(report.stdout)
            observed = parsed_report["cost_summary"].get("observed", parsed_report["cost_summary"])
            self.assertEqual(observed["coverage"]["runs_with_dollar_cost"], 0)
            self.assertIsNone(observed["totals"]["total_cost_usd"])
            self.assertEqual(observed["totals"]["total_cost_usd_aggregate"]["reason_counts"]["jetty_billing_finality_unavailable"], 1)
            cost = subprocess.run([sys.executable, str(CLI), "cost-summary", "--manifest", str(manifest), "--runs", str(runs)], cwd=root, capture_output=True, text=True, timeout=15, start_new_session=True, check=False)
            self.assertEqual(cost.returncode, 0, cost.stderr)
            cost_report = json.loads(cost.stdout)
            self.assertEqual(cost_report["coverage"]["runs_with_dollar_cost"], 0)
            self.assertIsNone(cost_report["totals"]["total_cost_usd"])
            self.assertEqual(cost_report["totals"]["total_cost_usd_aggregate"]["reason_counts"]["jetty_billing_finality_unavailable"], 1)
            self.persist(root, {"fixture_requests": fixture.requests, "report": report.stdout, "cost_report": cost.stdout})

    def test_unknown_current_price_closes_fresh_admission_but_old_poll_continues(self):
        fixture = JettyFixture(poll={"trajectory_id": "fixture-trajectory", "status": "running"})
        with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
            root = Path(td)
            old = jetty_payload(harness={"case_id": "old", "run_dir": "old/with_skill"})
            code, out, _, stderr = self.child(root, [old], url, "--poll-interval", "2", "--max-cost-usd", "1")
            self.assertEqual(code, 2, stderr)
            historical = {p: p.read_bytes() for p in self.ledger_paths(out)}
            fixture.poll = None
            fixture.requests.clear()
            fresh = jetty_payload()
            second = jetty_payload(harness={"case_id": "case-2", "run_dir": "case-2/with_skill"})
            code, _, _, stderr = self.child(root, [fresh, second, old], url, "--max-cost-usd", "10")
            self.assertEqual(code, 2, stderr)
            records = sb.load_jsonl(out)
            self.assertEqual([r["attempt_state"] for r in records], ["artifacts_downloaded", "not_started", "artifacts_downloaded"])
            self.assertEqual(fixture.counts(), {"upload": 0, "submit": 1, "poll": 2, "detail": 2, "download": 2})
            current = next(json.loads(p.read_text()) for p in self.ledger_paths(out) if p not in historical)
            self.assertEqual([r["state"] for r in current["calls"]], ["settled", "not_started"])
            self.assertEqual(current["spent_usd"], "0.6")
            self.assertEqual({p: p.read_bytes() for p in historical}, historical)
            self.persist(root, {"fixture_requests": fixture.requests})

    def test_refusal_import_has_no_provider_facts_or_answer_and_report_remains_incomplete(self):
        fixture = JettyFixture()
        with tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
            root = Path(td)
            code, out, _, stderr = self.child(root, [jetty_payload()], url, "--max-cost-usd", "0")
            self.assertEqual(code, 2, stderr)
            manifest = make_eval_repo(root / "manifest")
            runs = root / "runs"
            imported = subprocess.run([sys.executable, str(CLI), "import-jetty-results", "--manifest", str(manifest), "--jetty-runs", str(out), "--runs", str(runs)], cwd=root, capture_output=True, text=True, timeout=10, start_new_session=True, check=False)
            self.assertEqual(imported.returncode, 0, imported.stderr)
            metadata = json.loads((runs / "case-1/with_skill/metadata.json").read_text())
            self.assertEqual(metadata["invocation_state"], "not_started")
            self.assertFalse(any(key in metadata for key in ("returncode", "served_model", "jetty_trajectory_id")))
            self.assertFalse((runs / "case-1/with_skill/output.md").exists())
            report = subprocess.run([sys.executable, str(CLI), "benchmark", str(manifest), "--runs", str(runs)], cwd=root, capture_output=True, text=True, timeout=15, start_new_session=True, check=False)
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertNotEqual(json.loads(report.stdout)["availability"], "complete")
            self.assertEqual(fixture.requests, [])
            self.persist(root, {"fixture_requests": fixture.requests, "report": report.stdout})

    def test_uncapped_statuses_preserve_lifecycle_exits_without_spend_directory(self):
        scenarios = (
            (JettyFixture(poll={"trajectory_id": "fixture-trajectory", "status": "failed", "cost_usd": 0.6}), 0, "artifacts_downloaded", {"upload": 0, "submit": 1, "poll": 1, "detail": 0, "download": 0}),
            (JettyFixture(submit="lost"), 1, "submission_unknown", {"upload": 0, "submit": 1, "poll": 0, "detail": 0, "download": 0}),
            (JettyFixture(submit=[429, 429, 429]), 1, "upload_completed", {"upload": 0, "submit": 3, "poll": 0, "detail": 0, "download": 0}),
            (JettyFixture(download_error=True), 1, "terminal_observed", {"upload": 0, "submit": 1, "poll": 1, "detail": 1, "download": 1}),
            (JettyFixture(poll={"trajectory_id": "fixture-trajectory", "status": "running"}), 1, "submission_acknowledged", {"upload": 0, "submit": 1, "poll": 1, "detail": 0, "download": 0}),
        )
        for fixture, expected_exit, state, counts in scenarios:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as td, jetty_server(fixture) as url:
                root = Path(td)
                code, out, _, stderr = self.child(root, [jetty_payload()], url, "--poll-interval", "2")
                self.assertEqual(code, expected_exit, stderr)
                self.assertEqual(sb.load_jsonl(out)[0]["attempt_state"], state)
                self.assertEqual(fixture.counts(), counts)
                self.assertFalse(Path(str(out) + ".spend").exists())
                self.persist(root, {"fixture_requests": fixture.requests})
