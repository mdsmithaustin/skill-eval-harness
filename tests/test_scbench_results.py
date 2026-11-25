import json
import tempfile
import unittest
from pathlib import Path

from scbench_results import import_run


class SCBenchImportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "solo" / "run-1"
        self.root.mkdir(parents=True)
        self.row = {"problem": "demo", "idx": 2, "checkpoint": "checkpoint_2", "passed_tests": 4}
        self.rows = self.root / "checkpoint_results.jsonl"
        self.rows.write_text(json.dumps(self.row), encoding="utf-8")

    def test_score_without_transcript_and_identity_bound_study(self):
        path = self.root / "checkpoints.jsonl"
        row = {"problem": "demo", "checkpoint": 2, "arm": "solo", "run_id": "run-1"}
        path.write_text("\n".join(json.dumps(record) for record in [
            {**row, "solved": False}, {**row, "run_id": "other", "solved": True}]), encoding="utf-8")
        attempt, = import_run(self.root, self.root)["attempts"]
        self.assertEqual(attempt["events"], [])
        self.assertEqual([(s["basis"], s["value"]) for s in attempt["outcomes"]],
                         [("checkpoint_results", 4), ("study_analysis", False)])

    def test_replay_confirmation_needs_matching_rollout(self):
        replay = {"problem": "demo", "checkpoint": 2, "lead_rollout": "rollout-abc.jsonl"}
        confirmation = {"problem": "demo", "checkpoint": 2, "ran": True, "lost": [{"test": "old_behavior"}]}
        for name, row in [("replay", replay), ("confirmations", confirmation)]:
            (self.root / f"review-value-{name}.json").write_text(json.dumps([row]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no replay rows match"):
            import_run(self.root, review=self.root)
        agent = self.root / "demo/checkpoint_2/agent"
        agent.mkdir(parents=True)
        (agent / "rollout-abc.jsonl").touch()
        attempt, = import_run(self.root, review=self.root)["attempts"]
        self.assertEqual(attempt["outcomes"][-1]["value"], "old_behavior")
        self.assertEqual(attempt["outcomes"][-1]["basis"], "hidden_test_replay")

    def test_tool_and_reviewer_events_preserve_sources_and_order(self):
        agent = self.root / "demo/checkpoint_2/agent"
        agent.mkdir(parents=True)
        records = [
            {"type": "session_meta", "payload": {"source": "exec"}},
            {"type": "response_item", "timestamp": "2026-01-01T00:01:00Z", "payload": {
                "type": "agent_message", "author": "/root/review", "recipient": "/root",
                "content": [{"text": "**Spec verdict: pass.** Reviewed the earlier snapshot."}]}},
            {"type": "response_item", "timestamp": "2026-01-01T00:02:00Z", "payload": {
                "type": "custom_tool_call", "name": "exec", "input": "await tools.apply_patch(patch)"}},
        ]
        (agent / "rollout-1.jsonl").write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
        attempt, = import_run(self.root)["attempts"]
        msg, tool = attempt["events"]
        self.assertEqual(msg["text"], "**Spec verdict: pass.** Reviewed the earlier snapshot.")
        self.assertEqual(msg["spec_verdict"], "pass")
        self.assertEqual(tool["name"], "exec_with_apply_patch")
        self.assertTrue(tool["source"].endswith("rollout-1.jsonl:3"))

    def test_confines_rollouts_and_symlinked_checkpoint_directories(self):
        checkpoint = self.root / "demo/checkpoint_2"
        checkpoint.mkdir(parents=True)
        outside = self.root.parent / "external"
        outside.mkdir()
        rollout = outside / "rollout-1.jsonl"
        rollout.touch()
        agent = checkpoint / "agent"
        agent.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes the evidence"):
            import_run(self.root)
        agent.unlink()
        agent.mkdir()
        (agent / rollout.name).symlink_to(rollout)
        with self.assertRaisesRegex(ValueError, "escapes the evidence"):
            import_run(self.root)

    def test_rejects_output_aliases_across_all_evidence_directories(self):
        study = self.root.parent / "study"
        study.mkdir()
        study_input = study / "checkpoints.jsonl"
        study_input.write_text(json.dumps({**self.row, "arm": "solo", "run_id": "run-1",
                                           "checkpoint": 2}), encoding="utf-8")
        for source in (self.rows, study_input):
            with self.subTest(source=source):
                alias = self.root.parent / "output"
                alias.symlink_to(source)
                with self.assertRaisesRegex(ValueError, "aliases an SCBench input"):
                    import_run(self.root, study, output=alias)
                alias.unlink()
        replay = study / "review-value-replay.json"
        replay.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "aliases an SCBench input"):
            import_run(self.root, review=study, output=replay)

    def test_duplicate_and_path_component_rejected(self):
        for rows in [[self.row, self.row], [{**self.row, "problem": "../other"}]]:
            self.rows.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                import_run(self.root)
