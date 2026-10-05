"""Replay one committed regular text edit and run trusted fixture tests."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact_contracts import CompleteArtifactSet, observe_artifact_set
from workspace_contracts import (
    CapturedChanges,
    InPatch,
    Modified,
    RegularFile,
    WorkspaceChangesState,
    git_patch_entry,
    load_workspace_changes,
    workspace_changes_state,
)

ALLOWED_PATH = "inputs/name_tools.py"
TEST_ARGV = ("python3", "-B", "inputs/test_name_tools.py")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def verify_edit(run_dir: Path) -> str:
    artifact = observe_artifact_set(run_dir, declared_contract_version=1)
    if not isinstance(artifact, CompleteArtifactSet):
        raise TypeError("a complete committed artifact inventory is required")
    if workspace_changes_state(run_dir, artifact) is not WorkspaceChangesState.CAPTURED:
        raise ValueError("complete committed workspace capture is required")
    capture = load_workspace_changes(run_dir)
    if not isinstance(capture, CapturedChanges) or len(capture.changes) != 1:
        raise ValueError("exactly one captured modification is required")
    change = capture.changes[0]
    if (not isinstance(change, Modified) or change.path != ALLOWED_PATH
            or not isinstance(change.evidence, InPatch)
            or not isinstance(change.before, RegularFile)
            or not isinstance(change.after, RegularFile)
            or not change.before.text or not change.after.text
            or change.before.executable or change.after.executable):
        raise ValueError("only the allowed regular non-executable text module may change")
    original = (FIXTURES / "name_tools.py").read_bytes()
    if (hashlib.sha256(original).hexdigest() != change.before.sha256
            or len(original) != change.before.size):
        raise ValueError("before digest does not match the immutable fixture")
    patch = (run_dir / "candidate.patch").read_bytes()
    patch_text = patch.decode("utf-8")
    headers = [line for line in patch_text.splitlines()
               if line.startswith(("diff --git ", "--- ", "+++ "))]
    if headers != [f"diff --git a/{ALLOWED_PATH} b/{ALLOWED_PATH}",
                   f"--- a/{ALLOWED_PATH}", f"+++ b/{ALLOWED_PATH}"]:
        raise ValueError("patch contains unsafe or extra paths")
    with tempfile.TemporaryDirectory(prefix="edited-file-oracle-") as tmp:
        replay = Path(tmp)
        inputs = replay / "inputs"
        inputs.mkdir()
        for name in ("name_tools.py", "test_name_tools.py"):
            shutil.copyfile(FIXTURES / name, inputs / name)
        verified_patch = replay / "verified.patch"
        verified_patch.write_bytes(patch)
        stat = subprocess.run(
            ["git", "apply", "--numstat", str(verified_patch)], cwd=replay,
            text=True, capture_output=True, check=True, timeout=10,
        ).stdout.strip().split("\t")
        if len(stat) != 3 or stat[2] != ALLOWED_PATH or not all(v.isdigit() for v in stat[:2]):
            raise ValueError("patch must modify exactly the allowed text module")
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(verified_patch)], cwd=replay,
            capture_output=True, check=True, timeout=10,
        )
        module = replay / ALLOWED_PATH
        if module.is_symlink() or not module.is_file():
            raise ValueError("replayed module must be a regular file")
        edited = module.read_bytes()
        if (hashlib.sha256(edited).hexdigest() != change.after.sha256
                or len(edited) != change.after.size):
            raise ValueError("after digest does not match the reconstructed module")
        if patch_text != git_patch_entry(ALLOWED_PATH, change.before, change.after, original, edited):
            raise ValueError("patch must be the canonical text modification")
        tests = subprocess.run(
            list(TEST_ARGV), cwd=replay, text=True, capture_output=True, check=False,
            timeout=10, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if tests.returncode != 0:
            raise ValueError(f"trusted product tests failed\n{tests.stdout}{tests.stderr}")
    return "Verified committed edit passes trusted product tests."


def main() -> int:
    try:
        print(verify_edit(Path(sys.argv[1])))
        return 0
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
