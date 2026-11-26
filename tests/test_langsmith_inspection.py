import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from helpers import run_cli

from langsmith_export import build_export


class LangSmithInspectionTests(unittest.TestCase):
    def test_provenance_privacy_and_parent_graph(self):
        project = "85c33318-924b-47d1-b11e-ad3ea7b39c0d"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            capture = root / "p/checkpoint_1/agent/workspace/raw-api/invocation"
            capture.mkdir(parents=True)
            request = capture / "one.request.json"
            request.write_text(
                '{"messages":[{"role":"user","content":"private-task"}]}'
            )
            llm = {
                "kind": "llm",
                "id": "r",
                "parent_id": "a",
                "model": "m",
                "provider": "openai",
                "response_texts": ["private-response"],
                "timestamp": "2026-11-26T18:00:00Z",
                "usage": {"input_tokens": 4, "output_tokens": 2},
                "request_body_source": request.relative_to(root).as_posix(),
                "request_body_digest": hashlib.sha256(request.read_bytes()).hexdigest(),
            }
            spans = [
                llm,
                {"kind": "agent", "id": "a", "name": "reviewer"},
                {
                    "kind": "prompt",
                    "id": "p",
                    "role": "user",
                    "prompt_text": "private-task",
                },
                {
                    "kind": "prompt",
                    "id": "q",
                    "role": "user",
                    "synthetic": True,
                    "prompt_text": "injection",
                },
            ]
            attempt = {
                "metadata": {},
                "identity": {
                    "arm": root.parent.name,
                    "run_id": root.name,
                    "problem": "p",
                    "checkpoint": 1,
                },
                "events": [],
                "spans": spans,
                "outcomes": [],
            }

            def export(**kwargs):
                return [
                    r["body"]
                    for r in build_export(
                        {
                            "source": "scbench",
                            "schema_version": 1,
                            "attempts": [attempt],
                        },
                        project=project,
                        study="test",
                        **kwargs,
                    )
                ]

            baseline = export()
            self.assertNotIn("private-task", json.dumps(baseline))
            self.assertNotIn("private-response", json.dumps(baseline))
            self.assertIn(
                "private-response", json.dumps(export(include_responses=True))
            )
            enriched = export(include_prompts=True, model_content_from=root)
            self.assertNotIn("injection", json.dumps(enriched))
            self.assertNotIn(str(root), json.dumps(enriched))
            agent = next(r for r in enriched if r["name"] == "reviewer")
            model = next(r for r in enriched if r["run_type"] == "llm")
            self.assertEqual(model["parent_run_id"], agent["id"])
            self.assertTrue(
                model["dotted_order"].startswith(agent["dotted_order"] + ".")
            )
            self.assertEqual(model["inputs"], json.loads(request.read_bytes()))
            self.assertEqual(model["outputs"]["usage_metadata"], llm["usage"])
            self.assertEqual([r["id"] for r in enriched], [r["id"] for r in baseline])
            index = root / "index.json"
            index.write_text(
                json.dumps(
                    {"source": "scbench", "schema_version": 1, "attempts": [attempt]}
                )
            )
            code, output, error = run_cli(
                "export-langsmith",
                index,
                "--project",
                project,
                "--study",
                "cli",
                "--include-prompts",
                "--tool-content-from",
                root,
                "--model-content-from",
                root,
            )
            self.assertEqual(code, 0, error)
            self.assertIn("private-task", output)
            request.write_text("{}")
            with self.assertRaisesRegex(ValueError, "digest"):
                export(model_content_from=root)
            spans[1]["parent_id"] = "r"
            with self.assertRaisesRegex(ValueError, "cyclic"):
                export()
