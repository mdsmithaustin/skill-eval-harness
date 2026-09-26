import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import runner_contracts as rc
import skill_benchmark as sb
import workspace_contracts as wc
from artifact_contracts import observe_artifact_set


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_file(data: bytes, *, executable: bool = False) -> dict:
    return {"kind": "file", "sha256": sha(data), "size": len(data), "text": True,
            "executable": executable}


def binary_file(data: bytes) -> dict:
    return {"kind": "file", "sha256": sha(data), "size": len(data), "text": False,
            "executable": False}


def blob(data: bytes) -> dict:
    return {"kind": "blob", "blob": f"candidate-files/{sha(data)}"}


PATCH = {"kind": "patch"}
STATE = {"kind": "state"}


def write_tree(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def capture(root: Path, baseline: dict[str, bytes], edit: Callable[[Path], None], *,
            executable: frozenset[str] = frozenset(), **options) -> tuple[Path, Path]:
    def build(ws: Path) -> None:
        write_tree(ws, baseline)
        for rel in executable:
            (ws / rel).chmod(0o755)

    changes = root / "changes"
    changes.mkdir(parents=True)
    with wc.captured_workspace(prefix="test-ws-", changes_dir=changes, build=build, **options) as (ws, _):
        edit(ws)
    return changes, ws


def replay(root: Path, baseline: dict[str, bytes], executable: frozenset[str],
           patch: Path) -> dict[str, tuple[bytes, bool]]:
    git = shutil.which("git")
    if git is None:
        raise unittest.SkipTest("git is not installed")
    tree = root / "replay"
    tree.mkdir()
    write_tree(tree, baseline)
    for rel in executable:
        (tree / rel).chmod(0o755)
    subprocess.run([git, "apply", "--whitespace=nowarn", str(patch)],
                   cwd=tree, check=True, capture_output=True)
    return {path.relative_to(tree).as_posix(): (path.read_bytes(), bool(path.stat().st_mode & 0o111))
            for path in sorted(tree.rglob("*")) if path.is_file()}


def manifest(changes: Path) -> dict:
    return json.loads((changes / "workspace-changes.json").read_text(encoding="utf-8"))


def names_under(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def committed_run(root: Path, changes: Path) -> Path:
    run = root / "run"
    context = rc.OutcomeContext(provider="codex")
    sb.write_runner_outcome(run, rc.Completed(context, answer="done"), sidecars=changes)
    return run


def claim(run: Path) -> str | None:
    state = wc.workspace_changes_state(
        run, observe_artifact_set(run, declared_contract_version=1))
    return None if state is None else state.value


class CaptureTests(unittest.TestCase):
    def test_text_changes_become_one_patch_that_git_applies_onto_the_baseline(self):
        baseline = {
            "keep.txt": b"same\n",
            "edit.txt": b"a\nb\nc\n",
            "gone.txt": b"bye\n",
            "nested/deep/old.md": b"x",
        }

        def edit(ws: Path) -> None:
            (ws / "edit.txt").write_bytes(b"a\nB\nc\nd")
            (ws / "gone.txt").unlink()
            (ws / "nested/deep/old.md").write_bytes(b"y\n")
            (ws / "nested/deep/new.md").write_bytes(b"hello\n")
            (ws / "empty.txt").write_bytes(b"")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            changes, ws = capture(root, baseline, edit)
            data = manifest(changes)
            self.assertEqual(data["changes"], [
                {"path": "edit.txt", "change": "modified", "before": text_file(b"a\nb\nc\n"),
                 "after": text_file(b"a\nB\nc\nd"), "evidence": PATCH},
                {"path": "empty.txt", "change": "added", "after": text_file(b""), "evidence": PATCH},
                {"path": "gone.txt", "change": "deleted", "before": text_file(b"bye\n"), "evidence": PATCH},
                {"path": "nested/deep/new.md", "change": "added", "after": text_file(b"hello\n"),
                 "evidence": PATCH},
                {"path": "nested/deep/old.md", "change": "modified", "before": text_file(b"x"),
                 "after": text_file(b"y\n"), "evidence": PATCH},
            ])
            self.assertEqual(
                {key: data[key] for key in data if key != "changes"},
                {"schema_version": 1, "captured": True, "capture_error": None,
                 "workspace_root": str(ws), "workspace_root_realpath": os.path.realpath(ws),
                 "baseline_file_count": 4,
                 "limits": {"file_bytes": 1048576, "total_bytes": 33554432},
                 "patch": "candidate.patch"})
            self.assertEqual(names_under(changes), {"workspace-changes.json", "candidate.patch"})
            patch = (changes / "candidate.patch").read_text(encoding="utf-8")
            self.assertIn(
                "diff --git a/gone.txt b/gone.txt\ndeleted file mode 100644\n"
                "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n", patch)
            self.assertIn(
                "diff --git a/nested/deep/old.md b/nested/deep/old.md\n"
                "--- a/nested/deep/old.md\n+++ b/nested/deep/old.md\n@@ -1 +1 @@\n"
                "-x\n\\ No newline at end of file\n+y\n", patch)
            self.assertEqual(
                replay(root, baseline, frozenset(), changes / "candidate.patch"),
                {"edit.txt": (b"a\nB\nc\nd", False), "empty.txt": (b"", False),
                 "keep.txt": (b"same\n", False), "nested/deep/new.md": (b"hello\n", False),
                 "nested/deep/old.md": (b"y\n", False)})

    def test_executable_bit_is_recorded_and_git_apply_replays_it(self):
        baseline = {"run.sh": b"echo hi\n", "tool.sh": b"echo t\n", "gone.sh": b"echo g\n"}
        executable = frozenset({"tool.sh", "gone.sh"})

        def edit(ws: Path) -> None:
            (ws / "run.sh").chmod(0o755)
            (ws / "tool.sh").write_bytes(b"echo T\n")
            (ws / "tool.sh").chmod(0o644)
            (ws / "gone.sh").unlink()
            (ws / "new.sh").write_bytes(b"echo new\n")
            (ws / "new.sh").chmod(0o755)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            changes, _ = capture(root, baseline, edit, executable=executable)
            self.assertEqual(manifest(changes)["changes"], [
                {"path": "gone.sh", "change": "deleted",
                 "before": text_file(b"echo g\n", executable=True), "evidence": PATCH},
                {"path": "new.sh", "change": "added",
                 "after": text_file(b"echo new\n", executable=True), "evidence": PATCH},
                {"path": "run.sh", "change": "modified", "before": text_file(b"echo hi\n"),
                 "after": text_file(b"echo hi\n", executable=True), "evidence": PATCH},
                {"path": "tool.sh", "change": "modified",
                 "before": text_file(b"echo t\n", executable=True),
                 "after": text_file(b"echo T\n"), "evidence": PATCH},
            ])
            self.assertEqual((changes / "candidate.patch").read_text(encoding="utf-8"), (
                "diff --git a/gone.sh b/gone.sh\ndeleted file mode 100755\n"
                "--- a/gone.sh\n+++ /dev/null\n@@ -1 +0,0 @@\n-echo g\n"
                "diff --git a/new.sh b/new.sh\nnew file mode 100755\n"
                "--- /dev/null\n+++ b/new.sh\n@@ -0,0 +1 @@\n+echo new\n"
                "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"
                "diff --git a/tool.sh b/tool.sh\nold mode 100755\nnew mode 100644\n"
                "--- a/tool.sh\n+++ b/tool.sh\n@@ -1 +1 @@\n-echo t\n+echo T\n"))
            self.assertEqual(
                replay(root, baseline, executable, changes / "candidate.patch"),
                {"new.sh": (b"echo new\n", True), "run.sh": (b"echo hi\n", True),
                 "tool.sh": (b"echo T\n", False)})

    def test_binary_and_non_utf8_content_is_copied_by_digest_including_deleted_before_bytes(self):
        baseline = {"latin.txt": b"cafe\n", "old.bin": b"\x00zz"}

        def edit(ws: Path) -> None:
            (ws / "latin.txt").write_bytes(b"caf\xe9\n")
            (ws / "old.bin").unlink()
            (ws / "img.bin").write_bytes(b"\x00\x01\x02")

        with tempfile.TemporaryDirectory() as td:
            changes, _ = capture(Path(td), baseline, edit)
            data = manifest(changes)
            self.assertEqual(data["changes"], [
                {"path": "img.bin", "change": "added", "after": binary_file(b"\x00\x01\x02"),
                 "evidence": blob(b"\x00\x01\x02")},
                {"path": "latin.txt", "change": "modified", "before": text_file(b"cafe\n"),
                 "after": binary_file(b"caf\xe9\n"), "evidence": blob(b"caf\xe9\n")},
                {"path": "old.bin", "change": "deleted", "before": binary_file(b"\x00zz"),
                 "evidence": blob(b"\x00zz")},
            ])
            self.assertEqual((data["captured"], data["patch"]), (True, None))
            self.assertEqual(
                {path.name: path.read_bytes() for path in (changes / "candidate-files").iterdir()},
                {sha(b"\x00\x01\x02"): b"\x00\x01\x02", sha(b"caf\xe9\n"): b"caf\xe9\n",
                 sha(b"\x00zz"): b"\x00zz"})
            self.assertFalse((changes / "candidate.patch").exists())

    def test_oversize_and_total_cap_omit_content_and_withhold_the_claim(self):
        def edit(ws: Path) -> None:
            (ws / "a.bin").write_bytes(b"\x00" * 6)
            (ws / "b.bin").write_bytes(b"\x00\x01" * 3)
            (ws / "big.txt").write_bytes(b"0123456789\n")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            changes, _ = capture(root, {}, edit, limits=wc.EvidenceLimits(file_bytes=8, total_bytes=10))
            data = manifest(changes)
            self.assertEqual(data["changes"], [
                {"path": "a.bin", "change": "added", "after": binary_file(b"\x00" * 6),
                 "evidence": blob(b"\x00" * 6)},
                {"path": "b.bin", "change": "added", "after": binary_file(b"\x00\x01" * 3),
                 "evidence": {"kind": "omitted", "reason": "total_cap"}},
                {"path": "big.txt", "change": "added", "after": text_file(b"0123456789\n"),
                 "evidence": {"kind": "omitted", "reason": "oversize"}},
            ])
            self.assertEqual((data["captured"], data["limits"]), (False, {"file_bytes": 8, "total_bytes": 10}))
            self.assertEqual(names_under(changes),
                             {"workspace-changes.json", "candidate-files",
                              "candidate-files/" + sha(b"\x00" * 6)})
            self.assertEqual(claim(committed_run(root, changes)), "partial")

    def test_symlinks_are_recorded_by_target_and_never_followed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.txt").write_text("secret\n", encoding="utf-8")

            def edit(ws: Path) -> None:
                (ws / "link").symlink_to(outside / "secret.txt")
                (ws / "dirlink").symlink_to(outside, target_is_directory=True)
                (ws / "was-file").unlink()
                (ws / "was-file").symlink_to("keep.txt")

            changes, _ = capture(root, {"keep.txt": b"k\n", "was-file": b"w\n"}, edit)
            self.assertEqual(manifest(changes)["changes"], [
                {"path": "dirlink", "change": "added",
                 "after": {"kind": "symlink", "target": str(outside)}, "evidence": STATE},
                {"path": "link", "change": "added",
                 "after": {"kind": "symlink", "target": str(outside / "secret.txt")}, "evidence": STATE},
                {"path": "was-file", "change": "modified", "before": text_file(b"w\n"),
                 "after": {"kind": "symlink", "target": "keep.txt"}, "evidence": STATE},
            ])
            self.assertEqual(names_under(changes), {"workspace-changes.json"})
            self.assertEqual(claim(committed_run(root, changes)), "captured")

    def test_model_files_named_like_run_contract_files_never_land_under_their_own_basename(self):
        def edit(ws: Path) -> None:
            (ws / "metadata.json").write_bytes(b"\x00{}")
            (ws / "sub").mkdir()
            (ws / "sub" / "metrics.json").write_bytes(b"{}\n")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            changes, _ = capture(root, {}, edit)
            self.assertEqual([change["path"] for change in manifest(changes)["changes"]],
                             ["metadata.json", "sub/metrics.json"])
            run = committed_run(root, changes)
            self.assertEqual(
                sorted(path.relative_to(run).as_posix() for path in run.rglob("*")
                       if path.name in {"metadata.json", "metrics.json"}),
                ["metadata.json", "metrics.json"])
            self.assertEqual(claim(run), "captured")

    def test_an_untouched_workspace_is_a_captured_empty_change_set(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            changes, _ = capture(root, {"skills/demo/SKILL.md": b"# demo\n"}, lambda ws: None)
            data = manifest(changes)
            self.assertEqual((data["captured"], data["changes"], data["patch"], data["baseline_file_count"]),
                             (True, [], None, 1))
            self.assertEqual(names_under(changes), {"workspace-changes.json"})
            self.assertEqual(claim(committed_run(root, changes)), "captured")

    @unittest.skipIf(os.geteuid() == 0, "root reads mode-000 files")
    def test_an_unreadable_file_or_directory_fails_the_capture_without_losing_the_run(self):
        def lock_file(ws: Path) -> None:
            (ws / "locked.txt").write_bytes(b"x\n")
            (ws / "locked.txt").chmod(0)

        def lock_dir(ws: Path) -> None:
            (ws / "locked").mkdir()
            (ws / "locked" / "inner.txt").write_bytes(b"x\n")
            (ws / "locked").chmod(0)

        for name, edit in (("file", lock_file), ("dir", lock_dir)):
            with self.subTest(name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                changes, ws = capture(root, {"a.txt": b"a\n"}, edit)
                data = manifest(changes)
                self.assertEqual(
                    (data["captured"], data["capture_error"]["stage"], data["changes"], data["patch"],
                     data["workspace_root"], data["baseline_file_count"]),
                    (False, "diff", [], None, str(ws), None))
                self.assertIn("Permission denied", data["capture_error"]["reason"])
                self.assertEqual(names_under(changes), {"workspace-changes.json"})
                run = committed_run(root, changes)
                self.assertEqual((run / "output.md").read_text(encoding="utf-8"), "done")
                self.assertEqual(claim(run), "failed")

    @unittest.skipIf(os.geteuid() == 0, "root reads mode-000 files")
    def test_baseline_that_cannot_be_copied_records_a_baseline_failure(self):
        def build(ws: Path) -> None:
            (ws / "locked.txt").write_bytes(b"x\n")
            (ws / "locked.txt").chmod(0)

        with tempfile.TemporaryDirectory() as td:
            changes = Path(td) / "changes"
            changes.mkdir()
            with wc.captured_workspace(prefix="test-ws-", changes_dir=changes, build=build) as (ws, _):
                (ws / "new.txt").write_bytes(b"n\n")
            data = manifest(changes)
            self.assertEqual((data["captured"], data["capture_error"]["stage"], data["changes"]),
                             (False, "baseline", []))


class ClaimTests(unittest.TestCase):
    def changed_run(self, root: Path) -> tuple[Path, Path]:
        def edit(ws: Path) -> None:
            (ws / "a.txt").write_bytes(b"after\n")
            (ws / "b.bin").write_bytes(b"\x00")

        return capture(root, {"a.txt": b"before\n"}, edit)[0], root

    def test_captured_requires_every_referenced_file_in_the_committed_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            changes, root = self.changed_run(Path(td))
            run = committed_run(root, changes)
            self.assertEqual(claim(run), "captured")
            (run / "candidate.patch").unlink()
            self.assertEqual(claim(run), "invalid")

    def test_changed_files_without_their_patch_never_claim_capture(self):
        with tempfile.TemporaryDirectory() as td:
            changes, root = self.changed_run(Path(td))
            (changes / "candidate.patch").unlink()
            self.assertEqual(claim(committed_run(root, changes)), "invalid")

    def test_blob_named_for_other_content_is_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            changes, root = self.changed_run(Path(td))
            (changes / "candidate-files" / sha(b"\x00")).write_bytes(b"\x01")
            self.assertEqual(claim(committed_run(root, changes)), "invalid")

    def test_unparseable_or_inconsistent_manifest_is_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            changes, root = self.changed_run(Path(td))
            data = manifest(changes)
            data["captured"] = False
            (changes / "workspace-changes.json").write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(claim(committed_run(root, changes)), "invalid")
            (changes / "workspace-changes.json").write_text("{", encoding="utf-8")
            self.assertEqual(claim(committed_run(root, changes)), "invalid")

    def test_run_without_a_manifest_makes_no_claim(self):
        with tempfile.TemporaryDirectory() as td:
            empty = Path(td) / "empty"
            empty.mkdir()
            self.assertIsNone(claim(committed_run(Path(td), empty)))
            changes, root = self.changed_run(Path(td) / "with")
            self.assertEqual(claim(committed_run(root, changes)), "captured")


if __name__ == "__main__":
    unittest.main()
