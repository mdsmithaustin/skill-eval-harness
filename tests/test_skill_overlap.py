import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli

from content_digests import directory_tree_sha256

DESCRIPTIONS = {
    "invoice-total": "Add the charges on all customer invoices to calculate turnover.",
    "shop-receipts": "Compute gross revenue by summing retail sales receipts.",
    "soil-moisture": "Check whether a houseplant needs watering.",
}
ENCODER = """\
import json
import sys

vectors = dict(zip(__DESCRIPTIONS__, [[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]], strict=True))
texts = json.load(sys.stdin)["texts"]
print(json.dumps({"embeddings": [vectors[text] for text in texts]}))
"""
CODEX = """\
import json
import os
import sys
from pathlib import Path

print(json.dumps({"type": "thread.started", "thread_id": "overlap-stub"}))
print(json.dumps({"type": "turn.started"}))
if "--output-last-message" in sys.argv:
    prompt = sys.stdin.read()
    assert "expected-result" not in prompt
    skills = {path.parent.name for path in Path("skills").glob("*/SKILL.md")}
    assert skills <= {"invoice-total", "shop-receipts"}
    answer = "expected-result" if skills == {"shop-receipts"} else "incorrect"
    Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text(answer, encoding="utf-8")
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": answer}}))
else:
    assert not sys.stdin.read()
    assert sys.argv[-1] in {"Make a revenue report.", "How often do I water a plant?", "Unfinished request."}
    paths = sorted((Path(os.environ["CODEX_HOME"]) / "skills").glob("*/SKILL.md"))
    assert [path.parent.name for path in paths] == ["invoice-total", "shop-receipts"]
    if sys.argv[-1] == "Unfinished request.":
        sys.exit(2)
    if sys.argv[-1] == "Make a revenue report.":
        for i, path in enumerate(paths):
            print(json.dumps({"type": "item.completed", "item": {
                "id": str(i), "type": "command_execution", "command": f"cat {path}",
                "status": "completed", "exit_code": 0, "aggregated_output": "loaded",
            }}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 0}}))
"""


class SkillOverlapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="overlap-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        invoice, receipts, soil = [f"skills/{name}/SKILL.md" for name in DESCRIPTIONS]
        self.manifest = make_eval_repo(self.root, skill_paths=[invoice, receipts, soil], cases=[
            {"id": "answer", "split": "tune", "kind": "behavior", "prompt": "Report the total.",
             "assertions": [{"type": "contains", "value": "expected-result"}]},
            {"id": "report", "split": "tune", "kind": "trigger", "should_trigger": True,
             "prompt": "Make a revenue report.", "expected_skills": [receipts]},
            {"id": "water", "split": "tune", "kind": "trigger", "should_trigger": False,
             "prompt": "How often do I water a plant?", "forbidden_skills": [invoice, receipts]},
            {"id": "plant", "split": "tune", "kind": "trigger", "should_trigger": True,
             "prompt": "Would more water help my fern?", "expected_skills": [soil]},
        ], references={"references/ledger.md": "Amounts are in dollars."})
        self.collection = self.manifest.parents[1] / "skills"
        for name, description in DESCRIPTIONS.items():
            (self.collection / name / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {description}\n---\n\nUse the reference.\n", encoding="utf-8")
        self.encoder = self.root / "embed.py"
        self.encoder.write_text(ENCODER.replace("__DESCRIPTIONS__", repr(list(DESCRIPTIONS.values()))), encoding="utf-8")
        self.codex = self.root / "codex.py"
        self.codex.write_text(CODEX, encoding="utf-8")
        self.scan_path = self.root / "scan.json"
        self.runs = self.root / "run"

    def scan(self, collection=None):
        return run_cli("overlap-scan", collection or self.collection, "--out", self.scan_path,
                       "--embedding-model", "fixture/description-encoder", "--embed-cmd",
                       shlex.join([sys.executable, str(self.encoder)]))

    def evaluate(self):
        return run_cli("overlap-evaluate", self.scan_path, self.manifest, "--pair", "1", "--split", "tune",
                       "--agent", "codex", "--model", "offline-model", "--runs", self.runs,
                       "--codex-cmd", shlex.join([sys.executable, str(self.codex)]))

    def test_overlap_ranks_descriptions_and_feeds_both_runners_once(self):
        code, _, error = self.scan()
        self.assertEqual(code, 0, error)
        scan = json.loads(self.scan_path.read_text(encoding="utf-8"))
        self.assertEqual(len(scan["pairs"]), 3)
        pair = scan["pairs"][0]
        self.assertGreater(pair["score"], max(row["score"] for row in scan["pairs"][1:]))
        self.assertEqual([scan["skills"][path]["description"] for path in pair["skill_paths"]],
                         [DESCRIPTIONS["invoice-total"], DESCRIPTIONS["shop-receipts"]])
        self.assertFalse(self.runs.exists())

        code, stdout, error = self.evaluate()
        self.assertEqual(code, 0, error)
        report = json.loads(stdout)
        outcome = json.loads((self.runs / "interaction.json").read_text(encoding="utf-8"))
        matrix = json.loads((self.runs / "trigger-matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(report["scan_sha256"], scan["scan_sha256"])
        self.assertEqual(report["candidate"], pair)
        self.assertEqual(report["trigger"]["skill_tree_sha256"], matrix["skill_tree_hash"])
        self.assertEqual(matrix["skill_tree_hash"], directory_tree_sha256(self.runs / "skills"))
        self.assertEqual(report["trigger"]["excluded_queries"], [
            {"query_id": "plant", "reason": "references_other_skills"},
        ])
        for label, path in zip(("a", "b"), pair["skill_paths"], strict=True):
            identity = outcome["skills"][label]
            self.assertEqual(identity["digest"], scan["skills"][path]["digest"])
            self.assertEqual(identity["digest"], directory_tree_sha256(self.runs / "skills" / identity["mount_name"]))
            self.assertEqual({key: report["skills"][label][key] for key in identity}, identity)
        activation = {row["query_id"]: row for row in report["trigger"]["observations"]}
        self.assertEqual(activation["report"]["activated"], {"a": True, "b": True})
        self.assertTrue(activation["report"]["pass"])
        routed = next(row for row in matrix["results"] if row["query_id"] == "report")
        self.assertEqual(routed["activated_skills"], [report["skills"]["b"]["trigger_id"]])
        self.assertIn(report["skills"]["a"]["trigger_id"],
                      routed["observation_metadata"]["unscoped_skill_detections"])
        self.assertEqual(activation["water"]["activated"], {"a": False, "b": False})
        self.assertEqual({arm: value["rate"] for arm, value in report["outcomes"]["arms"].items()},
                         {"neither": 0.0, "a": 0.0, "b": 1.0, "both": 0.0})
        self.assertEqual(report["outcomes"]["comparisons"]["both_vs_better_individual"]["estimate"], -1.0)
        self.assertEqual(report["outcomes"]["comparisons"]["interaction"]["conclusion"], "inconclusive")

    def test_rescanning_after_complete_and_partial_evaluations_preserves_candidates(self):
        repo = self.manifest.parents[1]
        authored = repo / "zz-authored" / "interaction" / "_inputs" / "skills" / "soil-moisture"
        authored.parent.mkdir(parents=True)
        (self.collection / "soil-moisture").rename(authored)
        original_manifest = self.manifest.read_text(encoding="utf-8")
        self.manifest.write_text(original_manifest.replace(
            "skills/soil-moisture/SKILL.md", (authored / "SKILL.md").relative_to(repo).as_posix()),
            encoding="utf-8")
        code, _, error = self.scan(repo)
        self.assertEqual(code, 0, error)
        scan = json.loads(self.scan_path.read_bytes())
        self.assertEqual(len(scan["skills"]), 3)
        self.assertIn(str((authored / "SKILL.md").resolve()), scan["skills"])

        self.runs = repo / "runs" / "complete"
        code, stdout, error = self.evaluate()
        self.assertEqual(code, 0, error)
        first = json.loads(stdout)
        code, _, error = self.scan(repo)
        self.assertEqual(code, 0, error)
        rescanned = json.loads(self.scan_path.read_bytes())
        self.assertEqual(len(rescanned["skills"]), 3)
        self.assertEqual(rescanned, scan)

        self.runs = repo / "runs" / "partial"
        reference = self.collection / "shop-receipts" / "references" / "ledger.md"
        before = reference.read_bytes()
        try:
            reference.write_text("Changed", encoding="utf-8")
            code, _, error = self.evaluate()
            self.assertNotEqual(code, 0)
            self.assertIn("skill changed since scan", error)
            self.assertTrue(any(self.runs.rglob("SKILL.md")))
        finally:
            reference.write_bytes(before)
        code, _, error = self.scan(repo)
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(self.scan_path.read_bytes()), scan)

        self.runs = repo / "runs" / "retry"
        code, stdout, error = self.evaluate()
        self.assertEqual(code, 0, error)
        retried = json.loads(stdout)
        self.assertEqual(retried["candidate"], first["candidate"])
        self.assertEqual(retried["skills"], first["skills"])
        self.assertEqual(retried["trigger"]["skill_tree_sha256"], first["trigger"]["skill_tree_sha256"])

    def test_changed_reference_is_visible_before_either_runner_starts(self):
        self.assertEqual(self.scan()[0], 0)
        (self.collection / "shop-receipts" / "references" / "ledger.md").write_text("Changed", encoding="utf-8")
        self.codex.unlink()
        code, _, error = self.evaluate()
        self.assertNotEqual(code, 0)
        self.assertIn("skill changed since scan", error)
        self.assertFalse((self.runs / "interaction").exists())
        self.assertFalse((self.runs / "trigger-matrix.json").exists())

    def test_trigger_failure_keeps_activation_unknown_and_outcomes_visible(self):
        self.assertEqual(self.scan()[0], 0)
        manifest = self.manifest.read_text(encoding="utf-8")
        self.manifest.write_text(manifest.replace("Make a revenue report.", "Unfinished request."), encoding="utf-8")
        code, stdout, error = self.evaluate()
        self.assertEqual(code, 2, error)
        report = json.loads(stdout)
        row = next(row for row in report["trigger"]["observations"] if row["query_id"] == "report")
        self.assertEqual(row["activated"], {"a": None, "b": None})
        self.assertIsNone(row["pass"])
        self.assertEqual(report["outcomes"]["complete_cases"], 1)

    def test_zero_vectors_produce_an_explained_error(self):
        self.encoder.write_text(ENCODER.replace("__DESCRIPTIONS__", repr(list(DESCRIPTIONS.values())))
                                .replace("[1, 0, 0]", "[0, 0, 0]"), encoding="utf-8")
        code, _, error = self.scan()
        self.assertNotEqual(code, 0)
        self.assertIn("finite and nonzero", error)
        self.assertFalse(self.scan_path.exists())
