import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from helpers import attest_answer_design, make_eval_repo, run_cli, write_run

from langsmith_export import build_export, upload

PROJECT = "85c33318-924b-47d1-b11e-ad3ea7b39c0d"


class LangSmithTests(unittest.TestCase):
    def test_native_pair_through_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root)
            runs = root / "runs"
            for arm in ("with_skill", "without_skill"):
                events = [
                    {
                        "type": "command",
                        "name": "bash",
                        "input_summary": "secret-shell",
                        "status": "secret-shell",
                    }
                ]
                write_run(
                    runs / "case-1" / arm / "run-1",
                    "alpha",
                    events=events if arm == "without_skill" else {"schema_version": 2, "events": events},
                )
            attest_answer_design(manifest, runs)
            report = root / "benchmark.json"
            self.assertEqual(run_cli("benchmark", manifest, "--runs", runs, "--out", report)[0], 0)
            doc = json.loads(report.read_text())
            doc["availability"] = {"private": "secret-shell"}
            for row in doc["results"]:
                row["metadata"]["elapsed_ms"] = 173
                row["run_base"] = str(Path("archive/old-runs") / Path(row["run_base"]).relative_to(runs))
            relocated = root / "different" / "workspace"
            relocated.mkdir(parents=True)
            report = relocated / "benchmark.json"
            doc["results"].append({**doc["results"][0], "split": "holdout", "case_id": "private-case"})
            report.write_text(json.dumps(doc))
            with patch("http.client.HTTPSConnection", side_effect=AssertionError("network in preview")):
                code, output, error = run_cli(
                    "export-langsmith", relocated, "--runs", runs, "--project", PROJECT, "--study", "pilot"
                )
            self.assertEqual(code, 0, error)
            for secret in ("secret-shell", "private-case", "alpha", str(root)):
                self.assertNotIn(secret, output)
            payloads = [r["body"] for r in json.loads(output) if r["path"] == "runs"]
            parents = [r for r in payloads if "parent_run_id" not in r]
            self.assertTrue(all(r["extra"]["metadata"]["elapsed_ms"] == 173 for r in parents))
            self.assertEqual({r["inputs"]["variant"] for r in parents}, {"with_skill", "without_skill"})
            self.assertEqual(
                parents[0]["extra"]["metadata"]["comparison_group"],
                parents[1]["extra"]["metadata"]["comparison_group"],
            )
            self.assertEqual(
                {r["id"] for r in parents}, {r["parent_run_id"] for r in payloads if "parent_run_id" in r}
            )
            self.assertIn("benchmark.combined_pass_rate", output)

    def test_invalid_api_key_never_reaches_header_or_diagnostic(self):
        with patch("http.client.HTTPSConnection", side_effect=AssertionError("network before validation")):
            with self.assertRaisesRegex(ValueError, "^LANGSMITH_API_KEY contains invalid header characters$"):
                upload([], "https://example.invalid", "private-api-key\n")

    def test_scbench_outcome_origin_order_and_repeated_upload(self):
        document = {
            "source": "scbench",
            "schema_version": 1,
            "attempts": [
                {
                    "identity": {"run_id": "run", "arm": "stock", "problem": "p", "checkpoint": 1},
                    "metadata": {
                        "study": "pilot",
                        "started": "2026-11-18T10:00:00",
                        "cost_basis": "as_recorded_by_scbench",
                        "input": "private text instead of token count",
                        "skill_revision": "v12",
                        "skill_revision_source": "operator",
                        "effort": "medium",
                    },
                    "events": [
                        {
                            "kind": "tool_call",
                            "name": "exec",
                            "timestamp": "2026-11-18T17:05:00Z",
                            "source": "/private/log",
                        },
                        {"kind": "agent_message", "name": "message_to_lead", "text": "secret-shell"},
                    ],
                    "outcomes": [
                        {"key": "test_lost", "basis": "hidden_test_replay", "value": name}
                        for name in ("private-one", "private-two")
                    ],
                }
            ],
        }
        plan = build_export(document, project=PROJECT, study="pilot")
        self.assertEqual(plan, build_export(document, project=PROJECT, study="pilot"))
        self.assertEqual(len({r["body"]["id"] for r in plan}), 5)
        self.assertEqual([r["body"]["name"] for r in plan[:3]], ["scbench", "exec", "message_to_lead"])
        metadata = plan[0]["body"]["extra"]["metadata"]
        self.assertEqual(
            (metadata["skill_revision"], metadata["skill_revision_source"], metadata["effort"]),
            ("v12", "operator", "medium"),
        )
        self.assertNotIn("private", json.dumps(plan))
        self.assertNotIn("secret-shell", json.dumps(plan))
        network = MagicMock()
        persisted = {}

        def request(method, path, body, headers):
            record = json.loads(body)
            self.assertEqual(headers["X-Api-Key"], "fixture")
            uid = record["id"] if method == "POST" else path.rsplit("/", 1)[-1]
            if network.request.call_count == 5:
                network.getresponse.return_value.status = 503
            elif method == "POST" and uid in persisted:
                network.getresponse.return_value.status = 409
            else:
                persisted.setdefault(uid, {}).update(record)
                network.getresponse.return_value.status = 200

        network.request.side_effect = request
        with patch("http.client.HTTPSConnection", return_value=network):
            with self.assertRaisesRegex(ValueError, "HTTP 503"):
                upload(plan, "https://example.invalid/api/v1", "fixture")
            upload(plan, "https://example.invalid/api/v1", "fixture")
        self.assertEqual(len(persisted), 5)
        self.assertEqual(network.request.call_count, 14)

        document["attempts"][0]["outcomes"] = [
            {"key": "rate", "basis": "study_analysis", "value": 1 / 3},
            {"key": "total_tests", "basis": "checkpoint_results", "value": 100000},
        ]
        metrics = [
            r["body"]
            for r in build_export(document, project=PROJECT, study="pilot")
            if r["path"] == "feedback"
        ]
        self.assertEqual((metrics[0]["value"], metrics[0]["score"]), (1 / 3, 0.3333))
        self.assertEqual(metrics[1]["value"], 100000)
        self.assertNotIn("score", metrics[1])
