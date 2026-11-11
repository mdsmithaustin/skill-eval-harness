"""Decoy-skill catalogs for the trigger matrix, and the library-size report.

A skill that triggers alone can lose its routing once other skills sit beside
it. trigger-distractors writes one catalog per library size for the existing
matrix; trigger-distractor-report reads one matrix report per size."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli

import skill_benchmark as sb

ROOT = Path(__file__).resolve().parent.parent
TARGET = ("---\nname: reviewer\n"
          "description: Review a proposed change and label how serious each finding is.\n---\nReview.\n")
DECOYS = {
    "change-auditor": "Audit a proposed change and label how serious each finding is.",
    "geo-facts": "Answer questions about the capital of a country such as France.",
    "poems": "Write sonnets and haiku on request.",
}
CASES = [
    {"id": "pos", "split": "tune", "kind": "trigger", "should_trigger": True,
     "prompt": "Review this proposed change and label how serious each finding is."},
    {"id": "neg", "split": "tune", "kind": "trigger", "should_trigger": False,
     "prompt": "Recommend a restaurant for dinner tonight."},
]


def write_decoys(root: Path, names=tuple(DECOYS)) -> Path:
    decoys = root / "decoys"
    for name in names:
        (decoys / name).mkdir(parents=True)
        (decoys / name / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {DECOYS.get(name, 'Unrelated.')}\n---\nBody.\n",
            encoding="utf-8")
    return decoys


def row(should_trigger, passed, *, missing=(), forbidden=(), triggered=None):
    return {"agent": "stub", "model": None, "should_trigger": should_trigger, "pass": passed,
            "triggered": passed if triggered is None else triggered,
            "expected_skills": ["skills/reviewer/SKILL.md"] if should_trigger else [],
            "missing_expected_skills": list(missing), "forbidden_activations": list(forbidden)}


class CatalogTests(unittest.TestCase):
    def build(self, root, *sizes, cases=CASES, decoys=tuple(DECOYS)):
        manifest = make_eval_repo(root, skill_name="reviewer", skill_text=TARGET, cases=cases)
        out = root / "catalogs"
        code, _, stderr = run_cli("trigger-distractors", manifest, "--distractors",
                                  write_decoys(root, decoys), "--sizes", ",".join(map(str, sizes)),
                                  "--out-dir", out)
        return code, stderr, out

    def test_each_catalog_scopes_the_trigger_cases_and_nests_in_the_next(self):
        with tempfile.TemporaryDirectory() as td:
            code, stderr, out = self.build(Path(td), 0, 1, 3)
            self.assertEqual((code, stderr), (0, ""))
            index = json.loads((out / "distractor-catalogs.json").read_text(encoding="utf-8"))
            catalogs = {entry["library_size"]: json.loads(Path(entry["manifest"]).read_text(encoding="utf-8"))
                        for entry in index["catalogs"]}
            chosen = {entry["library_size"]: entry["distractors"] for entry in index["catalogs"]}
            mounted = sorted(path.name for path in (out / "library-1" / "skills").iterdir())
        self.assertEqual(index["skills_under_test"], ["skills/reviewer/SKILL.md"])
        self.assertEqual((chosen[0], len(chosen[1]), sorted(chosen[3])), ([], 1, sorted(DECOYS)))
        self.assertLess(set(chosen[1]), set(chosen[3]))
        self.assertEqual(mounted, sorted(["reviewer", *chosen[1]]))
        positive, negative = catalogs[3]["cases"]
        self.assertEqual(positive["expected_skills"], ["skills/reviewer/SKILL.md"])
        self.assertEqual(positive["forbidden_skills"], [f"skills/{name}/SKILL.md" for name in sorted(DECOYS)])
        # A decoy loading on an unrelated request is not the skill's failure.
        self.assertEqual(negative["forbidden_skills"], ["skills/reviewer/SKILL.md"])
        self.assertNotIn("expected_skills", negative)
        self.assertNotIn("forbidden_skills", catalogs[0]["cases"][0])
        self.assertEqual(len(catalogs[3]["skill_paths"]), 4)

    def test_a_catalog_case_carries_only_what_the_matrix_reads(self):
        cases = [{**CASES[0], "files": ["fixtures/input.txt"], "tags": ["x"],
                  "expected_behavior": ["loads"]}, CASES[1]]
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "repo" / "evals" / "fixtures").mkdir(parents=True)
            (Path(td) / "repo" / "evals" / "fixtures" / "input.txt").write_text("x", encoding="utf-8")
            code, stderr, out = self.build(Path(td), 1, cases=cases)
            self.assertEqual((code, stderr), (0, ""))
            catalog = json.loads((out / "library-1" / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(catalog["cases"][0]),
                         ["expected_skills", "forbidden_skills", "id", "kind", "prompt",
                          "should_trigger", "split"])

    def test_an_output_directory_inside_a_source_is_refused_before_any_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = make_eval_repo(root, skill_name="reviewer", skill_text=TARGET, cases=CASES)
            decoys = write_decoys(root)
            for out in (root / "repo" / "skills" / "reviewer" / "catalogs", decoys / "catalogs",
                        root / "repo" / "skills"):
                with self.subTest(out=out.name):
                    code, _, stderr = run_cli("trigger-distractors", manifest, "--distractors", decoys,
                                              "--sizes", "0,1", "--out-dir", out)
                    self.assertNotEqual(code, 0)
                    self.assertIn("overlaps", stderr)
                    self.assertFalse((out / "library-0").exists())

    def test_a_symlink_out_of_a_skill_is_refused_and_nothing_is_left(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = make_eval_repo(root, skill_name="reviewer", skill_text=TARGET, cases=CASES)
            decoys = write_decoys(root, ("poems",))
            secret = root / "secret.txt"
            secret.write_text("private", encoding="utf-8")
            (decoys / "poems" / "notes.txt").symlink_to(secret)
            out = root / "catalogs"
            code, _, stderr = run_cli("trigger-distractors", manifest, "--distractors", decoys,
                                      "--sizes", "0,1", "--out-dir", out)
            self.assertNotEqual(code, 0)
            self.assertIn("symlink escaping the root", stderr)
            self.assertEqual(list(out.glob("library-*")), [])

    def test_a_decoy_directory_that_is_a_symlink_is_refused_before_any_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = make_eval_repo(root, skill_name="reviewer", skill_text=TARGET, cases=CASES)
            decoys = write_decoys(root, ("poems",))
            outside = root / "outside"
            outside.mkdir()
            (outside / "SKILL.md").write_text("---\nname: ext\ndescription: Elsewhere.\n---\n", encoding="utf-8")
            (decoys / "ext").symlink_to(outside, target_is_directory=True)
            for out in (outside, root / "catalogs"):
                with self.subTest(out=out.name):
                    code, _, stderr = run_cli("trigger-distractors", manifest, "--distractors", decoys,
                                              "--sizes", "2", "--out-dir", out)
                    self.assertNotEqual(code, 0)
                    self.assertIn("must not be symlinks: ['ext']", stderr)
                    self.assertEqual(list(out.glob("library-*")), [])

    def test_the_decoy_order_is_seeded(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = make_eval_repo(Path(td), skill_name="reviewer", skill_text=TARGET, cases=CASES)
            decoys = write_decoys(Path(td))
            picks = [sb.build_distractor_catalogs(manifest, decoys, [1], Path(td) / f"out-{run}", seed=seed)
                     ["catalogs"][0]["distractors"] for run, seed in enumerate((4, 4, 5, 6, 7, 8))]
        self.assertEqual(picks[0], picks[1])
        self.assertGreater(len({tuple(pick) for pick in picks}), 1)

    def test_unusable_inputs_are_usage_errors(self):
        scoped = [{**CASES[0], "expected_skills": ["skills/reviewer/SKILL.md"]}]
        private = [{"id": "p", "split": "tune", "kind": "trigger", "should_trigger": True,
                    "prompt_ref": "private/p.md"}]
        answer_only = [{"id": "a", "split": "tune", "prompt": "Do it.",
                        "assertions": [{"name": "x", "type": "contains", "value": "x"}]}]
        rows = [
            ({"sizes": (5,)}, "--sizes asks for 5 distractors"),
            ({"sizes": (1,), "decoys": ("poems", "reviewer")}, "share a name with the skill under test"),
            ({"sizes": (1,), "cases": scoped}, "already declares expected_skills or forbidden_skills"),
            ({"sizes": (1,), "cases": private}, "a catalog needs an inline prompt"),
            ({"sizes": (1,), "cases": answer_only}, "no kind:'trigger' cases"),
            ({"sizes": (1, 1)}, "--sizes must be distinct whole numbers of 0 or more"),
        ]
        for options, message in rows:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as td:
                (Path(td) / "repo" / "evals" / "private").mkdir(parents=True)
                (Path(td) / "repo" / "evals" / "private" / "p.md").write_text("Review it.", encoding="utf-8")
                sizes = options.pop("sizes")
                code, stderr, out = self.build(Path(td), *sizes, **options)
                self.assertNotEqual(code, 0)
                self.assertIn(message, stderr)
                self.assertFalse((out / "distractor-catalogs.json").exists())

    def test_an_existing_catalog_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(self.build(Path(td), 1)[0], 0)
            manifest = Path(td) / "repo" / "evals" / "shared-benchmark.json"
            code, _, stderr = run_cli("trigger-distractors", manifest, "--distractors", Path(td) / "decoys",
                                      "--sizes", "1", "--out-dir", Path(td) / "catalogs")
            self.assertNotEqual(code, 0)
            self.assertIn("already exists", stderr)
            code, _, stderr = run_cli("trigger-distractors", manifest, "--distractors", Path(td) / "decoys",
                                      "--sizes", "0,1", "--out-dir", Path(td) / "catalogs")
            self.assertIn("library-1 already exists", stderr)
            self.assertFalse((Path(td) / "catalogs" / "library-0").exists())


class LibraryReportTests(unittest.TestCase):
    def test_rates_decoy_counts_and_both_boundaries(self):
        decoy = "skills/change-auditor/SKILL.md"
        reports = {
            0: {"results": [row(True, True)] * 4 + [row(False, True, triggered=False)] * 2},
            # The decoy loads beside the skill: the skill still loads, the run fails.
            5: {"results": [row(True, True)] * 2 + [row(True, False, forbidden=[decoy], triggered=True)] * 2
                + [row(False, True, triggered=False), row(False, False, triggered=True)]},
            # The decoy wins outright on three of four runs.
            20: {"results": [row(True, True)] + [row(True, False, missing=["skills/reviewer/SKILL.md"],
                                                     forbidden=[decoy], triggered=False)] * 3
                 + [{**row(False, True), "pass": None}]},
        }
        cell = sb.distractor_report(reports)["cells"][0]
        by_size = {entry["library_size"]: entry for entry in cell["by_library_size"]}
        self.assertEqual(by_size[0]["should_trigger"], {"runs": 4, "expected_load_rate": 1.0, "pass_rate": 1.0})
        self.assertEqual(by_size[5]["should_trigger"], {"runs": 4, "expected_load_rate": 1.0, "pass_rate": 0.5})
        self.assertEqual(by_size[5]["should_not_trigger"], {"runs": 2, "fire_rate": 0.5})
        self.assertEqual(by_size[5]["distractor_activations"], {decoy: 2})
        self.assertEqual(by_size[20]["should_trigger"], {"runs": 4, "expected_load_rate": 0.25, "pass_rate": 0.25})
        self.assertEqual((by_size[20]["incomplete"], by_size[20]["should_not_trigger"]["fire_rate"]), (1, None))
        self.assertEqual(cell["degradation_boundary"], {"expected_load_rate": 20, "pass_rate": 5})

    def test_a_drop_of_exactly_the_threshold_counts(self):
        def at(passed):
            return {"results": [row(True, True)] * passed
                    + [row(True, False, missing=["skills/reviewer/SKILL.md"])] * (10 - passed)}
        for first, later in ((9, 8), (10, 9), (7, 6)):
            with self.subTest(first=first):
                cell = sb.distractor_report({0: at(first), 5: at(later)})["cells"][0]
                self.assertEqual(cell["degradation_boundary"]["expected_load_rate"], 5)

    def test_a_historical_incomplete_row_marked_failed_is_not_a_verdict(self):
        stale = {**row(True, False, missing=["skills/reviewer/SKILL.md"]), "observation_complete": False}
        cell = sb.distractor_report({0: {"results": [row(True, True), stale]},
                                     5: {"results": [row(True, True)]}})["cells"][0]
        first = cell["by_library_size"][0]
        self.assertEqual((first["incomplete"], first["should_trigger"]["runs"]), (1, 1))
        self.assertEqual(first["should_trigger"]["expected_load_rate"], 1.0)

    def test_a_cell_missing_at_one_size_gets_no_boundary(self):
        def at(passed, agent="stub"):
            return [{**row(True, True), "agent": agent}] * passed + [
                {**row(True, False, missing=["skills/reviewer/SKILL.md"]), "agent": agent}] * (4 - passed)
        report = sb.distractor_report({0: {"results": at(4) + at(4, "other")},
                                       5: {"results": at(2) + at(2, "other")},
                                       20: {"results": at(2)}})
        cells = {cell["agent"]: cell for cell in report["cells"]}
        self.assertEqual(cells["stub"]["degradation_boundary"]["expected_load_rate"], 5)
        self.assertEqual(cells["stub"]["missing_library_sizes"], [])
        self.assertEqual(cells["other"]["missing_library_sizes"], [20])
        self.assertEqual(cells["other"]["degradation_boundary"],
                         {"expected_load_rate": None, "pass_rate": None})

    def test_a_dip_that_recovers_is_not_a_boundary(self):
        def at(passed, total=4):
            return {"results": [row(True, True)] * passed
                    + [row(True, False, missing=["skills/reviewer/SKILL.md"])] * (total - passed)}
        cell = sb.distractor_report({0: at(4), 5: at(2), 20: at(4)})["cells"][0]
        self.assertEqual(cell["degradation_boundary"], {"expected_load_rate": None, "pass_rate": None})

    def test_an_unscoped_baseline_report_reads_triggered(self):
        legacy = {"agent": "stub", "model": None, "should_trigger": True, "pass": True, "triggered": True}
        cell = sb.distractor_report({0: {"results": [legacy]}, 5: {"results": [row(True, True)]}})["cells"][0]
        self.assertEqual(cell["by_library_size"][0]["should_trigger"]["expected_load_rate"], 1.0)

    def test_report_arguments_are_checked(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "r.json"
            path.write_text(json.dumps({"results": [row(True, True)]}), encoding="utf-8")
            for argv, message in (
                    (("--report", f"0={path}"), "needs at least two --report SIZE=PATH entries"),
                    (("--report", f"x={path}", "--report", f"5={path}"), "--report must be SIZE=PATH"),
                    (("--report", f"5={path}", "--report", f"5={path}"), "names library size 5 twice"),
                    (("--report", f"0={path}", "--report", f"5={path}", "--drop", "0"), "--drop must be")):
                with self.subTest(message=message):
                    code, _, stderr = run_cli("trigger-distractor-report", *argv)
                    self.assertNotEqual(code, 0)
                    self.assertIn(message, stderr)


class StubMatrixJourneyTests(unittest.TestCase):
    def test_the_matrix_runs_each_catalog_and_the_report_names_the_confusable_decoy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = make_eval_repo(root, skill_name="reviewer", skill_text=TARGET, cases=CASES)
            out = root / "catalogs"
            self.assertEqual(run_cli("trigger-distractors", manifest, "--distractors", write_decoys(root),
                                     "--sizes", "0,3", "--out-dir", out)[0], 0)
            reports = []
            for size in (0, 3):
                report = root / f"matrix-{size}.json"
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "run_trigger_matrix.py"),
                     str(out / f"library-{size}" / "catalog.json"), "--agent", "stub",
                     "--runs-per-query", "1", "--out", str(report)],
                    capture_output=True, text=True, timeout=120, check=False)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                reports += ["--report", f"{size}={report}"]
            summary = root / "library.json"
            self.assertEqual(run_cli("trigger-distractor-report", *reports, "--out", summary)[0], 0)
            cell = json.loads(summary.read_text(encoding="utf-8"))["cells"][0]
        alone, crowded = cell["by_library_size"]
        self.assertEqual(alone["should_trigger"]["pass_rate"], 1.0)
        # The stub loads every skill whose description shares two words with
        # the query, so the reviewer still loads and the look-alike loads too.
        self.assertEqual(crowded["should_trigger"], {"runs": 1, "expected_load_rate": 1.0, "pass_rate": 0.0})
        self.assertEqual(crowded["distractor_activations"], {"skills/change-auditor/SKILL.md": 1})
        self.assertEqual(cell["degradation_boundary"], {"expected_load_rate": None, "pass_rate": 3})


if __name__ == "__main__":
    unittest.main()
