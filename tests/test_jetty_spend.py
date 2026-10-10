import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import FakeJettyClient, jetty_payload, jetty_task_upload, run_cli

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
