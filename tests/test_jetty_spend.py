import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import (
    FakeJettyClient,
    attest_jetty_payload,
    jetty_payload,
    jetty_task_upload,
    run_cli,
)

import skill_benchmark as sb


class JettySpendTests(unittest.TestCase):
    def test_invalid_later_payload_preserves_output_and_prevents_earlier_remote_work(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid = jetty_payload(files=[jetty_task_upload()])
            invalid = jetty_payload(harness={"case_id": "case-2", "run_dir": "case-2/with_skill"})
            invalid["jetty_request"]["model"] = "changed-after-attestation"
            payloads = root / "payloads.jsonl"
            payloads.write_text("".join(json.dumps(row) + "\n" for row in (valid, invalid)))
            out = root / "results.jsonl"
            out.write_bytes(b"previous output\n")
            client = FakeJettyClient()
            with mock.patch.dict(os.environ, {"JETTY_API_TOKEN": "fixture"}), \
                    mock.patch.object(sb, "JettyClient", return_value=client):
                try:
                    run_cli("run-jetty", "--payloads", payloads, "--out", out)
                except RuntimeError as exc:
                    self.assertEqual(str(exc), "Jetty model-visible task/runbook/upload plan changed after attestation")
            self.assertEqual(out.read_bytes(), b"previous output\n")
            self.assertEqual((client.upload_calls, client.submit_calls, client.poll_calls), (0, 0, 0))
            self.assertFalse(Path(str(out) + ".attempts.json").exists())
            self.assertFalse(Path(str(out) + ".spend").exists())

    def test_attested_invalid_later_coordinate_or_path_prevents_remote_work(self):
        cases = [
            {"case_id": None},
            {"case_id": ""},
            {"variant": "bogus"},
            {"run_number": 0},
            {"run_number": True},
            {"run_number": 1.0},
            {"run_dir": "case-2/\x00"},
            {"run_dir": "case-2/\nwith_skill"},
            {"run_dir": "case-2/\x7fwith_skill"},
        ]
        for fields in cases:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                valid = jetty_payload(files=[jetty_task_upload()])
                invalid = jetty_payload(harness={"case_id": "case-2", "run_dir": "case-2/with_skill", **fields})
                out = root / "results.jsonl"
                out.write_bytes(b"previous output\n")
                client = FakeJettyClient()
                with self.assertRaisesRegex(ValueError, "Jetty payload"):
                    self.invoke(root, [valid, invalid], client)
                self.assertEqual(out.read_bytes(), b"previous output\n")
                self.assertEqual((client.upload_calls, client.submit_calls, client.poll_calls), (0, 0, 0))
                self.assertFalse(Path(str(out) + ".attempts.json").exists())
                self.assertFalse(Path(str(out) + ".spend").exists())

    def test_null_model_and_normalized_output_path_remain_supported(self):
        with tempfile.TemporaryDirectory() as td:
            row = jetty_payload(harness={"run_dir": "case-1/./with_skill"})
            row["jetty_request"]["model"] = None
            attest_jetty_payload(row)
            (code, _, stderr), out = self.invoke(Path(td), [row], FakeJettyClient())
            self.assertEqual(code, 0, stderr)
            [record] = sb.load_jsonl(out)
            self.assertEqual(record["status"], "completed")
            self.assertIsNone(record["jetty"]["model"])
            self.assertEqual(record["harness"]["run_dir"], "case-1/./with_skill")

    def invoke(self, root, rows, client, *flags):
        payloads = root / "payloads.jsonl"
        payloads.write_text("".join(json.dumps(row) + "\n" for row in rows))
        out = root / "results.jsonl"
        with mock.patch.dict(os.environ, {"JETTY_API_TOKEN": "fixture"}), \
                mock.patch.object(sb, "JettyClient", return_value=client):
            result = run_cli("run-jetty", "--payloads", payloads, "--out", out,
                             "--timeout", "1", "--poll-interval", "0.01", *flags)
        return result, out

    def ledgers(self, out):
        return [json.loads(path.read_text()) for path in sorted(Path(str(out) + ".spend").glob("*/spend-ceiling.json"))]

    def test_zero_refuses_upload_and_admitted_attempt_has_a_real_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row = jetty_payload(files=[jetty_task_upload()])
            client = FakeJettyClient()
            (code, _, stderr), out = self.invoke(root, [row], client, "--max-cost-usd", "0")
            self.assertEqual(code, 2, stderr)
            [refusal] = sb.load_jsonl(out)
            self.assertEqual(refusal["invocation_state"], "not_started")
            self.assertEqual(self.ledgers(out)[0]["calls"][0]["state"], "not_started")
            self.assertEqual((client.upload_calls, client.submit_calls, client.poll_calls), (0, 0, 0))
            self.assertFalse(Path(str(out) + ".attempts.json").exists())
            (code, _, stderr), _ = self.invoke(root, [row], client, "--max-cost-usd", "1", "--assumed-cost-per-run-usd", "0.1")
            self.assertEqual(code, 0, stderr)
            self.assertEqual((client.upload_calls, client.submit_calls, client.poll_calls), (1, 1, 1))
            digest = row["harness"]["jetty_task_contract_sha256"]
            task = json.loads(Path(str(out) + ".attempts.json").read_text())["attempts"][digest]
            [attempt] = task["submission_attempts"]
            self.assertRegex(attempt["ref"], r"^[0-9a-f]{32}$")
            self.assertEqual(attempt["receipts"][0]["state"], "submitting")
            self.assertNotEqual(attempt["ref"], refusal["spend_call"]["journal_attempt_id"])

    def test_unknown_replacement_preserves_history_on_refusal_and_admission(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row = jetty_payload()
            (_, _, _), out = self.invoke(root, [row], FakeJettyClient(submit_error=ConnectionError("ack lost")), "--max-cost-usd", "1")
            path = Path(str(out) + ".attempts.json")
            before = path.read_bytes()
            old_ledgers = {path: path.read_bytes() for path in Path(str(out) + ".spend").glob("*/spend-ceiling.json")}
            client = FakeJettyClient()
            (code, _, _), _ = self.invoke(root, [row], client, "--max-cost-usd", "0", "--resubmit-unknown")
            self.assertEqual(code, 2)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sb.load_jsonl(out)[0]["prior_attempt"]["state"], "submission_unknown")
            (code, _, _), _ = self.invoke(root, [row], client, "--max-cost-usd", "1", "--assumed-cost-per-run-usd", "0.1", "--resubmit-unknown")
            self.assertEqual(code, 0)
            digest = row["harness"]["jetty_task_contract_sha256"]
            first, second = json.loads(path.read_text())["attempts"][digest]["submission_attempts"]
            self.assertEqual(first, json.loads(before)["attempts"][digest]["submission_attempts"][0])
            self.assertNotEqual(first["ref"], second["ref"])
            self.assertEqual(second["replaces_unknown"], first["ref"])
            self.assertEqual({p: p.read_bytes() for p in old_ledgers}, old_ledgers)

    def test_dollars_survive_invalid_tokens_conflicts_missing_values_and_local_failure(self):
        cases = [
            ({"cost_usd": 0.6, "usage": {"input_tokens": "malformed"}}, None, "0.6"),
            ({"cost": 0.6, "cost_usd": 0.6}, None, "0.6"),
            ({"cost": 0.6, "cost_usd": 0.7}, None, None),
            ({"cost_usd": 0.6}, {"cost_usd": 0.2}, "0.6"),
            ({"cost_usd": 0.6}, {"cost": 0.7, "cost_usd": 0.8}, "0.6"),
            ({"cost_usd": 0.6}, {}, "0.6"),
        ]
        for pending, later, expected in cases:
            with self.subTest(pending=pending, later=later), tempfile.TemporaryDirectory() as td:
                class PollingClient(FakeJettyClient):
                    def poll(self, collection, task, trajectory_id, *, pending=pending, later=later, **options):
                        options["on_observation"]({"status": "running", "trajectory_id": trajectory_id, **pending})
                        if later is None:
                            raise ConnectionError("poll response lost")
                        return {"status": "failed", "trajectory_id": trajectory_id, **later}
                (code, _, _), out = self.invoke(Path(td), [jetty_payload()], PollingClient(), "--max-cost-usd", "1")
                self.assertEqual(code, 2)
                ledger = self.ledgers(out)[0]
                self.assertEqual(ledger["calls"][0]["charge"], {"basis": "unpriced", "reason": "jetty_billing_finality_unavailable", "observed_subtotal_usd": expected})
                self.assertEqual(sb.load_jsonl(out)[0]["jetty_price_evidence"]["observed_subtotal_usd"], expected)

    def test_unrelated_later_response_cannot_increase_an_independent_floor(self):
        unrelated = [
            {"trajectory_id": "another-run"},
            {"collection": "another-collection"},
            {"storage_path": "another/task/0000"},
            {"jetty_metadata": {"task": "another-task"}},
            {"jetty": {"storage_path": "c/t/another-run"}},
        ]
        for evidence in unrelated:
            with self.subTest(evidence=evidence), tempfile.TemporaryDirectory() as td:
                class PollingClient(FakeJettyClient):
                    def poll(self, collection, task, trajectory_id, *, evidence=evidence, **options):
                        options["on_observation"]({"status": "running", "trajectory_id": trajectory_id, "cost_usd": 0.2})
                        return {"status": "failed", "trajectory_id": trajectory_id, "cost_usd": 0.6, **evidence}
                client = PollingClient(submission={"trajectory_id": "trajectory-1", "storage_path": "c/t/0000"})
                (code, _, _), out = self.invoke(Path(td), [jetty_payload()], client, "--max-cost-usd", "1")
                self.assertEqual(code, 2)
                self.assertEqual(self.ledgers(out)[0]["calls"][0]["charge"]["observed_subtotal_usd"], "0.2")

    def test_normalized_output_slot_collision_is_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = jetty_payload(harness={"run_dir": "case-1/with_skill"})
            later = jetty_payload(harness={"case_id": "case-2", "run_dir": "case-1/./with_skill"})
            out = root / "results.jsonl"
            out.write_bytes(b"previous output\n")
            client = FakeJettyClient()
            with self.assertRaisesRegex(ValueError, "duplicate or unsafe Jetty selected output slot"):
                self.invoke(root, [first, later], client, "--max-cost-usd", "1")
            self.assertEqual(out.read_bytes(), b"previous output\n")
            self.assertEqual((client.upload_calls, client.submit_calls), (0, 0))
            self.assertFalse(Path(str(out) + ".attempts.json").exists())
            self.assertFalse(Path(str(out) + ".spend").exists())

    def test_detail_dollars_survive_artifact_failure_and_floor_a_smaller_assumption(self):
        for assumption in (None, "0.1"):
            with self.subTest(assumption=assumption), tempfile.TemporaryDirectory() as td:
                detail = {"trajectory_id": "trajectory-1", "status": "completed", "storage_path": "c/t/0000",
                          "steps": {"run": {"outputs": {"success": True, "usage": {"cost_usd": 0.6}, "results_files": "invalid"}}}}
                flags = ("--assumed-cost-per-run-usd", assumption) if assumption else ()
                (code, _, _), out = self.invoke(Path(td), [jetty_payload()], FakeJettyClient(detail_response=detail), "--max-cost-usd", "1", *flags)
                self.assertEqual(code, 2 if assumption is None else 1)
                ledger = self.ledgers(out)[0]
                self.assertEqual(ledger["spent_usd"], "0.6")
                self.assertEqual(ledger["calls"][0]["charge"]["observed_subtotal_usd"], "0.6")
                self.assertEqual(ledger["calls"][0]["charge"]["basis"], "unpriced" if assumption is None else "assumed")
                self.assertEqual(sb.load_jsonl(out)[0]["attempt_state"], "terminal_observed")

    def test_interruption_settles_retained_floor_before_propagation(self):
        class InterruptedClient(FakeJettyClient):
            def poll(self, collection, task, trajectory_id, **options):
                options["on_observation"]({"status": "running", "trajectory_id": trajectory_id, "cost_usd": 0.6})
                raise KeyboardInterrupt("interrupted after paid evidence")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(KeyboardInterrupt, "interrupted after paid evidence"):
                self.invoke(root, [jetty_payload()], InterruptedClient(), "--max-cost-usd", "1")
            out = root / "results.jsonl"
            ledger = self.ledgers(out)[0]
            self.assertEqual(ledger["calls"][0]["state"], "settled")
            self.assertEqual(ledger["spent_usd"], "0.6")
            self.assertEqual(out.read_bytes(), b"")

    def test_journal_failure_before_submit_is_fatal_and_submits_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            client = FakeJettyClient()
            original = sb.JettyAttemptJournal._persist
            def persist(journal, data):
                if any(attempt["receipts"][-1]["state"] == "submitting" for task in data["attempts"].values() for attempt in task["submission_attempts"]):
                    raise OSError("submitting checkpoint failed")
                original(journal, data)
            with mock.patch.object(sb.JettyAttemptJournal, "_persist", persist), self.assertRaisesRegex(sb.JettyJournalPersistenceError, "submitting checkpoint failed"):
                self.invoke(Path(td), [jetty_payload()], client, "--max-cost-usd", "1")
            self.assertEqual(client.submit_calls, 0)
            ledger = self.ledgers(Path(td) / "results.jsonl")[0]
            self.assertEqual(ledger["calls"][0]["charge"]["basis"], "unpriced")
            self.assertEqual(ledger["calls"][0]["charge"]["observed_subtotal_usd"], None)

    def test_legacy_load_replay_and_refusal_do_not_rewrite_surviving_receipt(self):
        for durable in (False, True):
            with self.subTest(durable=durable), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                row = jetty_payload()
                first = FakeJettyClient() if durable else FakeJettyClient(submit_error=ConnectionError("old unknown"))
                (_, _, _), out = self.invoke(root, [row], first)
                journal_path = Path(str(out) + ".attempts.json")
                digest = row["harness"]["jetty_task_contract_sha256"]
                entry = sb.JettyAttemptJournal(journal_path).entry(digest)
                entry.pop("attempt_ref")
                entry["abandoned_unknown_submissions"] = 3
                entry["opaque_legacy_metadata"] = {"retained": ["original"]}
                legacy = {"schema_version": 1, "attempts": {digest: entry}, "legacy_root_metadata": "preserved"}
                journal_path.write_text(json.dumps(legacy) + "\n")
                before = journal_path.read_bytes()
                client = FakeJettyClient()
                flags = () if durable else ("--resubmit-unknown",)
                (code, _, stderr), _ = self.invoke(root, [row], client, "--max-cost-usd", "0", *flags)
                self.assertEqual(code, 0 if durable else 2, stderr)
                self.assertEqual(journal_path.read_bytes(), before)
                self.assertEqual((client.submit_calls, client.poll_calls, client.fetch_calls), (0, 0, 0))
                if durable:
                    self.assertEqual(sb.load_jsonl(out)[0], entry["record"])
                else:
                    (code, _, stderr), _ = self.invoke(root, [row], client, "--resubmit-unknown")
                    self.assertEqual(code, 0, stderr)
                    raw = json.loads(journal_path.read_text())
                    task = raw["attempts"][digest]
                    self.assertEqual(raw["legacy_root_metadata"], "preserved")
                    self.assertEqual(task["legacy_metadata"], entry)
                    self.assertEqual(task["legacy_abandoned_unknown_submissions"], 3)
                    self.assertEqual(task["submission_attempts"][0]["ref"], {"kind": "legacy_current", "task_contract_sha256": digest, "scope": "schema1-current"})
                    self.assertEqual(task["submission_attempts"][1]["replaces_unknown"], task["submission_attempts"][0]["ref"])

    def test_price_checkpoint_failure_retains_floor_and_stops_later_fresh_work(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = FakeJettyClient(poll_response={"trajectory_id": "trajectory-1", "status": "failed", "cost_usd": 0.6})
            original = sb.JettyAttemptJournal._persist
            def persist(journal, data):
                if any(attempt["receipts"][-1].get("dollar_floor_usd") == "0.6" for task in data["attempts"].values() for attempt in task["submission_attempts"]):
                    raise OSError("price checkpoint failed")
                original(journal, data)
            rows = [jetty_payload(), jetty_payload(harness={"case_id": "case-2", "run_dir": "case-2/with_skill"})]
            with mock.patch.object(sb.JettyAttemptJournal, "_persist", persist), self.assertRaisesRegex(sb.JettyJournalPersistenceError, "price checkpoint failed"):
                self.invoke(root, rows, client, "--max-cost-usd", "1")
            ledger = self.ledgers(root / "results.jsonl")[0]
            self.assertEqual([row["state"] for row in ledger["calls"]], ["settled", "planned"])
            self.assertEqual(ledger["spent_usd"], "0.6")
            self.assertEqual(client.submit_calls, 1)

    def test_settlement_failure_keeps_in_flight_ledger_and_journal_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            client = FakeJettyClient(poll_response={"trajectory_id": "trajectory-1", "status": "failed", "cost_usd": 0.6})
            original = sb.SpendAdmission._persist
            def persist(admission):
                if admission.ledger and any(row["state"] == "settled" for row in admission.ledger.as_dict()["calls"]):
                    raise OSError("settlement failed")
                original(admission)
            row = jetty_payload()
            with mock.patch.object(sb.SpendAdmission, "_persist", persist), self.assertRaisesRegex(OSError, "settlement failed"):
                self.invoke(root, [row], client, "--max-cost-usd", "1")
            out = root / "results.jsonl"
            self.assertEqual(self.ledgers(out)[0]["calls"][0]["state"], "in_flight")
            digest = row["harness"]["jetty_task_contract_sha256"]
            self.assertEqual(sb.JettyAttemptJournal(Path(str(out) + ".attempts.json")).entry(digest)["dollar_floor_usd"], "0.6")
            self.assertEqual(out.read_bytes(), b"")
