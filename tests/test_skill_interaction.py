import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli

from content_digests import canonical_json_sha256, directory_tree_sha256
from skill_benchmark import write_artifact_commit

OFFLINE_CODEX = """\
import json
import sys
from pathlib import Path

prompt = sys.stdin.read()
assert "outcome-verified" not in prompt
assert "unrelated" not in prompt
inputs = list(Path("inputs").iterdir())
assert len(inputs) == 1
number = int(inputs[0].read_text(encoding="utf-8"))
paths = sorted(Path("skills").glob("*/SKILL.md"))
skills = tuple(path.parent.name for path in paths)
for path in paths:
    assert path.as_posix() in prompt
    assert (path.parent / "references" / "checklist.md").read_text(encoding="utf-8") == path.parent.name
threshold = {(): 10, ("first",): 15, ("second",): 14, ("first", "second"): 13}[skills]
answer = "outcome-verified" if number < threshold else "Task failed."
Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text(answer, encoding="utf-8")
for event in (
    {"type": "thread.started", "thread_id": "interaction-fixture"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"id": "answer", "type": "agent_message", "text": answer}},
    {"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 0}},
):
    print(json.dumps(event))
"""


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory(prefix="interaction-test-")
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name)
        cases = [
            {"id": f"case-{n:02}", "split": "tune", "prompt": "Complete the attached case.",
             "files": [f"fixture-{n}.txt"],
             "assertions": [{"type": "contains", "value": "outcome-verified"},
                            {"type": "regex", "value": "^outcome-verified$"}]}
            for n in range(20)
        ]
        cases.append({"id": "trigger-only", "split": "tune", "kind": "trigger",
                      "prompt": "An unrelated trigger case.", "should_trigger": False})
        cls.manifest = make_eval_repo(cls.root, cases=cases)
        for n in range(20):
            (cls.manifest.parent / f"fixture-{n}.txt").write_text(str(n), encoding="utf-8")
        for name in ("first", "second"):
            skill = cls.root / name
            reference = skill / "references" / "checklist.md"
            reference.parent.mkdir(parents=True)
            reference.write_text(name, encoding="utf-8")
            (skill / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: A distinct test skill.\n---\n", encoding="utf-8")
        stub = cls.root / "offline_codex.py"
        stub.write_text(OFFLINE_CODEX, encoding="utf-8")
        cls.runs = cls.root / "runs"
        code, stdout, stderr = run_cli(
            "interaction", cls.manifest, "--split", "tune",
            "--skill-a", cls.root / "first", "--skill-b", cls.root / "second" / "SKILL.md",
            "--agent", "codex", "--model", "offline-model", "--runs", cls.runs, "--k", "2",
            "--codex-cmd", shlex.join([sys.executable, str(stub)]))
        if code:
            raise AssertionError(f"interaction failed ({code})\n{stdout}\n{stderr}")
        cls.report = json.loads(stdout)

    def report_only(self, runs=None):
        code, stdout, stderr = run_cli(
            "interaction", self.manifest, "--runs", runs or self.runs, "--report-only")
        self.assertTrue(stdout, stderr)
        return code, json.loads(stdout)

    def test_controlled_four_arm_vector(self):
        self.assertEqual({arm: entry["rate"] for arm, entry in self.report["arms"].items()},
                         {"neither": 0.50, "a": 0.75, "b": 0.70, "both": 0.65})
        self.assertEqual((self.report["complete_cases"], self.report["planned_cases"]), (20, 20))
        self.assertEqual(self.report["repetitions"], 2)
        self.assertTrue(all(entry["scored_runs"] == 40 for entry in self.report["arms"].values()))
        self.assertEqual(self.report["better_individual"], "a")
        self.assertEqual(self.report["issues"], [])
        for comparison, estimate in [
            ("both_vs_neither", 0.15), ("both_vs_better_individual", -0.10), ("interaction", -0.30)
        ]:
            observed = self.report["comparisons"][comparison]
            self.assertEqual(observed["estimate"], estimate)
            self.assertEqual(observed["interval"]["confidence"], 0.95)
            self.assertEqual(observed["interval"]["n"], 20)
        for label, name in (("a", "first"), ("b", "second")):
            self.assertEqual(self.report["skills"][label], {
                "name": name, "mount_name": name, "digest_scheme": "skill-tree-sha256-v1",
                "digest": directory_tree_sha256(self.root / name),
            })
        self.assertEqual(self.report_only(), (0, self.report))

    def test_missing_repetition_withholds_only_dependent_comparisons(self):
        runs = self.root / "missing"
        shutil.copytree(self.runs, runs)
        shutil.rmtree(runs / "a" / "case-00" / "with_skill" / "run-2")
        code, report = self.report_only(runs)
        self.assertEqual(code, 2)
        self.assertIn({"code": "missing_run", "arm": "a", "case_id": "case-00", "run_number": 2},
                      report["issues"])
        self.assertEqual(report["complete_cases"], 19)
        self.assertIsNone(report["arms"]["a"]["rate"])
        self.assertEqual(report["comparisons"]["both_vs_neither"]["estimate"], 0.15)
        for comparison in ("both_vs_better_individual", "interaction"):
            self.assertEqual(report["comparisons"][comparison]["incomplete_cases"], ["case-00"])
            self.assertIsNone(report["comparisons"][comparison]["estimate"])

    def test_changed_fixture_withholds_all_comparisons(self):
        fixture = self.manifest.parent / "fixture-0.txt"
        previous = fixture.read_bytes()
        fixture.write_text("changed", encoding="utf-8")
        try:
            code, report = self.report_only()
        finally:
            fixture.write_bytes(previous)
        self.assertEqual(code, 2)
        self.assertEqual(report["issues"][0]["code"], "dataset_mismatch")
        self.assertTrue(all(row["estimate"] is None for row in report["comparisons"].values()))

    def test_mismatched_model_withholds_the_affected_comparisons(self):
        runs = self.root / "wrong-model"
        shutil.copytree(self.runs, runs)
        base = runs / "b" / "case-09" / "with_skill" / "run-1"
        path = base / "metadata.json"
        metadata = json.loads(path.read_bytes())
        metadata["requested_model"] = "another-model"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        write_artifact_commit(base)
        code, report = self.report_only(runs)
        self.assertEqual(code, 2)
        self.assertIn({"code": "model_or_agent_mismatch", "arm": "b", "case_id": "case-09", "run_number": 1},
                      report["issues"])
        self.assertIsNone(report["comparisons"]["interaction"]["estimate"])
        self.assertEqual(report["comparisons"]["both_vs_neither"]["estimate"], 0.15)

    def test_grader_identity_distinguishes_incompatible_evidence(self):
        runs = self.root / "wrong-grader"
        shutil.copytree(self.runs, runs)
        path = runs / "interaction-plan.json"
        plan = json.loads(path.read_bytes())
        plan["grader"]["sha256"] = "sha256:" + "0" * 64
        del plan["plan_sha256"]
        plan["plan_sha256"] = canonical_json_sha256(plan)
        path.write_text(json.dumps(plan), encoding="utf-8")
        code, report = self.report_only(runs)
        self.assertEqual(code, 2)
        self.assertEqual(report["issues"], [{"code": "grader_mismatch"}])
        self.assertTrue(all(row["estimate"] is None for row in report["comparisons"].values()))

    def test_replay_can_relocate_the_dataset_without_manifest_skill_roots(self):
        relocated = self.root / "relocated" / "evals"
        shutil.copytree(self.manifest.parent, relocated)
        manifest = relocated / self.manifest.name
        code, stdout, stderr = run_cli("interaction", manifest, "--runs", self.runs, "--report-only")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout), self.report)

    def test_report_only_rejects_new_skill_selection(self):
        code, stdout, stderr = run_cli(
            "interaction", self.manifest, "--runs", self.runs, "--report-only",
            "--skill-a", self.root / "second", "--skill-b", self.root / "first")
        self.assertNotEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertIn("omit --skill-a and --skill-b", stderr)

    def test_replay_detects_changes_to_grader_helpers_but_ignores_unrelated_code(self):
        copied = self.root / "grader-source"
        copied.mkdir()
        for path in Path(__file__).resolve().parents[1].glob("*.py"):
            shutil.copyfile(path, copied / path.name)
        module = copied / "skill_benchmark.py"
        original = module.read_text(encoding="utf-8")
        module.write_text(
            original + "\ndef unrelated_interaction_test_helper():\n    return 'unused'\n",
            encoding="utf-8",
        )
        argv = [sys.executable, "-m", "skill_benchmark", "interaction", str(self.manifest),
                "--runs", str(self.runs), "--report-only"]
        control = subprocess.run(argv, cwd=copied, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(control.returncode, 0, control.stderr)
        self.assertEqual(json.loads(control.stdout), self.report)

        start, marker = original.index("def assertion_severity("), '    severity = assertion.get("severity")'
        index = original.index(marker, start)
        module.write_text(
            original[:index] + '    if assertion.get("type") == "contains":\n        return "soft"\n' + original[index:],
            encoding="utf-8",
        )
        changed = subprocess.run(argv, cwd=copied, capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(changed.returncode, 2, changed.stderr)
        report = json.loads(changed.stdout)
        self.assertEqual(report["issues"], [{"code": "grader_mismatch"}])
        self.assertTrue(all(row["estimate"] is None for row in report["comparisons"].values()))
