import json
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OfflineJourneyCommandsTests(unittest.TestCase):
    def run_journey(self, filename: str) -> Path:
        tmp = tempfile.TemporaryDirectory(prefix="offline-journey-")
        self.addCleanup(tmp.cleanup)
        scratch = Path(tmp.name)
        text = (ROOT / "docs" / filename).read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
        self.assertGreater(len(blocks), 1)
        script = "\n".join(blocks)
        script = re.sub(r"^S=\$\(mktemp[^\n]+", f"S={shlex.quote(str(scratch))}", script)
        script = script.replace('PY="$(pwd)/.venv/bin/python3.12.14"',
                                f"PY={shlex.quote(sys.executable)}")
        result = subprocess.run(["bash", "-eu", "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=90, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return scratch

    def test_trajectory_commands_distinguish_answer_and_path(self):
        scratch = self.run_journey("did-my-skill-change-how-the-model-works.md")
        reports = [json.loads((scratch / name).read_text(encoding="utf-8"))
                   for name in ("bench.json", "loop-bench.json", "lenient-bench.json")]
        assertions = []
        for report in reports:
            row = next(r for r in report["results"]
                       if r["case_id"] == "c-review-path" and r["variant"] == "with_skill")
            assertions.append({a["name"]: a["passed"]
                               for a in row["assertions"] + row["qualitative_assertions"]})
        self.assertEqual(assertions, [
            {"severity-label": True, "skill-read": True, "no-reread-loop": True, "sound-steps": True},
            {"severity-label": True, "skill-read": True, "no-reread-loop": False, "sound-steps": False},
            {"severity-label": True, "skill-read": True, "no-reread-loop": False, "sound-steps": True},
        ])
        normal = scratch / "runs" / "c-review-path" / "with_skill" / "output.md"
        loop = scratch / "loop-runs" / "c-review-path" / "with_skill" / "output.md"
        self.assertEqual(normal.read_bytes(), loop.read_bytes())
        weak = next(c for c in reports[0]["trajectory_diff"]["cases"]
                    if c["case_id"] == "c-weak-outcome")
        self.assertEqual(weak["commands_only_with_skill"],
                         ["cat skills/demo/SKILL.md", "cat skills/demo/references/checklist.md"])
        self.assertEqual(weak["skill_invoked"], {"with_skill": 1.0, "without_skill": 0.0})
        documented = json.loads(re.findall(
            r"```json\n(.*?)\n```",
            (ROOT / "docs" / "did-my-skill-change-how-the-model-works.md").read_text(encoding="utf-8"),
            re.DOTALL)[0])
        self.assertEqual(weak, documented)
        flags = next(c["flags"] for c in reports[0]["case_flags"]
                     if c["case_id"] == "c-weak-outcome")
        self.assertEqual(flags, ["saturated/non-discriminating", "no objective lift"])

    def test_discovery_commands_produce_all_evidence_classes(self):
        scratch = self.run_journey("did-removing-this-break-discovery.md")
        reports = [json.loads((scratch / name).read_text(encoding="utf-8"))
                   for name in ("compare.json", "compare-three.json", "compare-full.json")]
        documented = [json.loads(block) for block in re.findall(
            r"```json\n(.*?)\n```",
            (ROOT / "docs" / "did-removing-this-break-discovery.md").read_text(encoding="utf-8"),
            re.DOTALL)]
        self.assertEqual(len(documented), 2)
        for actual, example in zip((reports[0], reports[2]), documented, strict=True):
            self.assertEqual({key: actual[key] for key in example}, example)
        self.assertEqual([r["evidence_class"] for r in reports],
                         ["refuted", "indeterminate", "confirmed_causal"])
        self.assertEqual([r["summary"]["regressed"] for r in reports], [0, 3, 6])
        self.assertEqual([r["summary"]["blocked"] for r in reports], [0, 0, 0])
        self.assertEqual([r["summary"]["mean_pass_delta"] for r in reports],
                         [0.0, -1.0, -0.5454545454545454])
        self.assertIn("p=0.25", reports[1]["note"])
        self.assertEqual({r["query_id"] for r in reports[2]["regressed_queries"]}, {
            "wtu-inspect-diff", "wtu-serious-code", "wtu-inspect-code",
            "wtu-code-diff-serious", "wtu-asked-check", "wtu-serious-problems",
        })


if __name__ == "__main__":
    unittest.main()
