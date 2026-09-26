"""What an answer run's model did to its temporary workspace, captured before deletion.

`captured_workspace` owns the workspace lifetime: build, baseline, yield to the
provider call, capture, delete. The capture stages `workspace-changes.json`,
`candidate.patch`, and `candidate-files/<sha256>` in a directory the caller
hands to the run writer as sidecars. Readers derive `workspace_changes_state`
from that manifest and the committed artifact inventory.
"""
from __future__ import annotations

import codecs
import difflib
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NoReturn, TypeAlias, TypeVar

from artifact_contracts import ArtifactSetObservation, CompleteArtifactSet
from json_contracts import strict_json_loads

WORKSPACE_CHANGES_NAME = "workspace-changes.json"
CANDIDATE_PATCH_NAME = "candidate.patch"
CANDIDATE_FILES_DIR = "candidate-files"
WORKSPACE_SIDECAR_NAMES = frozenset({WORKSPACE_CHANGES_NAME, CANDIDATE_PATCH_NAME, CANDIDATE_FILES_DIR})
WORKSPACE_CHANGES_VERSION = 1
FILE_EVIDENCE_CAP_BYTES = 1 << 20
TOTAL_EVIDENCE_CAP_BYTES = 32 << 20


@dataclass(frozen=True)
class RegularFile:
    sha256: str
    size: int
    text: bool
    executable: bool


@dataclass(frozen=True)
class Symlink:
    target: str


@dataclass(frozen=True)
class Special:
    mode: int


FileState: TypeAlias = RegularFile | Symlink | Special


@dataclass(frozen=True)
class InPatch:
    pass


@dataclass(frozen=True)
class InBlob:
    pass


@dataclass(frozen=True)
class InState:
    pass


OmitReason = Literal["oversize", "total_cap", "unreadable"]


@dataclass(frozen=True)
class Omitted:
    reason: OmitReason


Evidence: TypeAlias = InPatch | InBlob | InState | Omitted


@dataclass(frozen=True)
class Added:
    path: str
    after: FileState
    evidence: Evidence


@dataclass(frozen=True)
class Modified:
    path: str
    before: FileState
    after: FileState
    evidence: Evidence


@dataclass(frozen=True)
class Deleted:
    path: str
    before: FileState
    evidence: Evidence


FileChange: TypeAlias = Added | Modified | Deleted


@dataclass(frozen=True)
class EvidenceLimits:
    file_bytes: int = FILE_EVIDENCE_CAP_BYTES
    total_bytes: int = TOTAL_EVIDENCE_CAP_BYTES

    def __post_init__(self) -> None:
        for value in (self.file_bytes, self.total_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("evidence limits must be positive integers")


DEFAULT_LIMITS = EvidenceLimits()


CaptureStage = Literal["baseline", "diff", "evidence", "manifest"]


@dataclass(frozen=True)
class CapturedChanges:
    workspace_root: str
    workspace_root_realpath: str
    baseline_file_count: int
    limits: EvidenceLimits
    changes: tuple[FileChange, ...]

    def __post_init__(self) -> None:
        if type(self.baseline_file_count) is not int or self.baseline_file_count < 0:
            raise ValueError("baseline file count must be a non-negative integer")
        paths = [change.path for change in self.changes]
        if paths != sorted(set(paths)):
            raise ValueError("workspace changes must be sorted by unique path")
        for change in self.changes:
            _validate_change(change)

    @property
    def complete(self) -> bool:
        return not any(isinstance(change.evidence, Omitted) for change in self.changes)

    @property
    def has_patch(self) -> bool:
        return any(isinstance(change.evidence, InPatch) for change in self.changes)

    def blob_digests(self) -> frozenset[str]:
        return frozenset(
            side.sha256 for change in self.changes
            if isinstance(change.evidence, InBlob)
            and isinstance(side := _content_side(change), RegularFile))


@dataclass(frozen=True)
class CaptureFailed:
    stage: CaptureStage
    reason: str
    workspace_root: str
    workspace_root_realpath: str
    limits: EvidenceLimits

    def __post_init__(self) -> None:
        if self.stage not in ("baseline", "diff", "evidence", "manifest"):
            raise ValueError(f"unknown capture stage {self.stage!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("capture failure reason must be non-empty")


WorkspaceCapture: TypeAlias = CapturedChanges | CaptureFailed


@dataclass(frozen=True)
class WorkspaceBaseline:
    shadow: Path
    index: Mapping[str, FileState]


@dataclass(frozen=True)
class BaselineUnavailable:
    reason: str


Baseline: TypeAlias = WorkspaceBaseline | BaselineUnavailable


class WorkspaceChangesState(str, Enum):
    CAPTURED = "captured"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID = "invalid"


def _assert_never(value: NoReturn) -> NoReturn:
    raise TypeError(f"unhandled variant {type(value).__name__}")


def _valid_path(path: object) -> bool:
    return (isinstance(path, str) and not path.startswith("/")
            and all(part not in ("", ".", "..") for part in path.split("/")))


def _content_side(change: FileChange) -> FileState:
    if isinstance(change, (Added, Modified)):
        return change.after
    if isinstance(change, Deleted):
        return change.before
    _assert_never(change)


def _sides(change: FileChange) -> tuple[FileState, ...]:
    if isinstance(change, Added):
        return (change.after,)
    if isinstance(change, Modified):
        return (change.before, change.after)
    if isinstance(change, Deleted):
        return (change.before,)
    _assert_never(change)


def _validate_change(change: FileChange) -> None:
    if not _valid_path(change.path):
        raise ValueError(f"workspace change path must be relative and normalized: {change.path!r}")
    if isinstance(change, Modified) and change.before == change.after:
        raise ValueError(f"modified entry has identical sides: {change.path}")
    sides = _sides(change)
    evidence = change.evidence
    all_regular = all(isinstance(side, RegularFile) for side in sides)
    if isinstance(evidence, InPatch):
        consistent = all(isinstance(side, RegularFile) and side.text for side in sides)
    elif isinstance(evidence, InBlob):
        consistent = all_regular
    elif isinstance(evidence, InState):
        consistent = not all_regular
    elif isinstance(evidence, Omitted):
        consistent = evidence.reason in ("oversize", "total_cap", "unreadable")
    else:
        _assert_never(evidence)
    if not consistent:
        raise ValueError(f"evidence does not fit the recorded file states: {change.path}")


def blob_path(digest: str) -> str:
    return f"{CANDIDATE_FILES_DIR}/{digest}"


def _open_regular(path: Path) -> int:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise OSError(f"{path} stopped being a regular file during capture")
    return fd


def file_state(path: Path) -> FileState:
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode):
        return Symlink(os.readlink(path))
    if not stat.S_ISREG(mode):
        return Special(mode)
    digest, size, text = hashlib.sha256(), 0, True
    decoder = codecs.getincrementaldecoder("utf-8")()
    with os.fdopen(_open_regular(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
            if text:
                try:
                    decoder.decode(chunk)
                    text = b"\0" not in chunk
                except UnicodeDecodeError:
                    text = False
    if text:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            text = False
    return RegularFile(digest.hexdigest(), size, text, bool(mode & 0o111))


def index_tree(root: Path) -> dict[str, FileState]:
    index: dict[str, FileState] = {}
    pending = [root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    index[path.relative_to(root).as_posix()] = file_state(path)
    return index


def _failure_reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def snapshot_workspace(ws: Path, shadow: Path) -> Baseline:
    try:
        shutil.copytree(ws, shadow, symlinks=True)
        return WorkspaceBaseline(shadow, index_tree(shadow))
    except (OSError, UnicodeError, ValueError) as exc:
        return BaselineUnavailable(_failure_reason(exc))


def _read_verified(path: Path, expected: RegularFile) -> bytes | None:
    try:
        with os.fdopen(_open_regular(path), "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    return data if hashlib.sha256(data).hexdigest() == expected.sha256 else None


_GIT_QUOTED_PATH = re.compile(r'[\x00-\x1f\x7f"\\]')


def _patch_lines(data: bytes) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", data.decode("utf-8")) if data else []


def _git_mode(state: RegularFile) -> str:
    return "100755" if state.executable else "100644"


def git_patch_entry(path: str, before: RegularFile | None, after: RegularFile | None,
                    old: bytes, new: bytes) -> str:
    header = [f"diff --git a/{path} b/{path}\n"]
    if before is None and after is not None:
        header.append(f"new file mode {_git_mode(after)}\n")
    elif after is None and before is not None:
        header.append(f"deleted file mode {_git_mode(before)}\n")
    elif before is not None and after is not None and before.executable != after.executable:
        header += [f"old mode {_git_mode(before)}\n", f"new mode {_git_mode(after)}\n"]
    hunks = difflib.unified_diff(
        _patch_lines(old), _patch_lines(new),
        "/dev/null" if before is None else f"a/{path}",
        "/dev/null" if after is None else f"b/{path}")
    return "".join(header) + "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in hunks)


def _change(path: str, before: FileState | None, after: FileState | None,
            evidence: Evidence) -> FileChange:
    if before is None and after is not None:
        return Added(path, after, evidence)
    if after is None and before is not None:
        return Deleted(path, before, evidence)
    if before is not None and after is not None:
        return Modified(path, before, after, evidence)
    raise ValueError(f"change has no side: {path}")


class _EvidenceStager:

    def __init__(self, baseline: WorkspaceBaseline, ws: Path, changes_dir: Path,
                 limits: EvidenceLimits) -> None:
        self.baseline, self.ws, self.changes_dir, self.limits = baseline, ws, changes_dir, limits
        self.spent = 0
        self.patch: list[str] = []
        self.stored: set[str] = set()

    def _afford(self, cost: int) -> bool:
        if self.spent + cost > self.limits.total_bytes:
            return False
        self.spent += cost
        return True

    def evidence(self, path: str, before: FileState | None, after: FileState | None) -> Evidence:
        sides = [side for side in (before, after) if side is not None]
        regular = [side for side in sides if isinstance(side, RegularFile)]
        if len(regular) != len(sides):
            return InState()
        within = all(side.size <= self.limits.file_bytes for side in regular)
        if within and all(side.text for side in regular) and not _GIT_QUOTED_PATH.search(path):
            return self._patch(path, before, after)
        source = (self.ws if isinstance(after, RegularFile) else self.baseline.shadow) / path
        content = after if isinstance(after, RegularFile) else before
        if not isinstance(content, RegularFile) or content.size > self.limits.file_bytes:
            return Omitted("oversize")
        return self._blob(source, content)

    def _patch(self, path: str, before: FileState | None, after: FileState | None) -> Evidence:
        old_file = before if isinstance(before, RegularFile) else None
        new_file = after if isinstance(after, RegularFile) else None
        old = b"" if old_file is None else _read_verified(self.baseline.shadow / path, old_file)
        new = b"" if new_file is None else _read_verified(self.ws / path, new_file)
        if old is None or new is None:
            return Omitted("unreadable")
        entry = git_patch_entry(path, old_file, new_file, old, new)
        if not self._afford(len(entry.encode("utf-8"))):
            return Omitted("total_cap")
        self.patch.append(entry)
        return InPatch()

    def _blob(self, source: Path, content: RegularFile) -> Evidence:
        if content.sha256 in self.stored:
            return InBlob()
        if self.spent + content.size > self.limits.total_bytes:
            return Omitted("total_cap")
        data = _read_verified(source, content)
        if data is None:
            return Omitted("unreadable")
        target = self.changes_dir / blob_path(content.sha256)
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        self._afford(content.size)
        self.stored.add(content.sha256)
        return InBlob()


def _clear_outputs(changes_dir: Path) -> None:
    for name in WORKSPACE_SIDECAR_NAMES:
        path = changes_dir / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _state_json(state: FileState) -> dict[str, Any]:
    if isinstance(state, RegularFile):
        return {"kind": "file", "sha256": state.sha256, "size": state.size, "text": state.text,
                "executable": state.executable}
    if isinstance(state, Symlink):
        return {"kind": "symlink", "target": state.target}
    if isinstance(state, Special):
        return {"kind": "special", "mode": state.mode}
    _assert_never(state)


def _evidence_json(change: FileChange) -> dict[str, Any]:
    evidence = change.evidence
    if isinstance(evidence, InPatch):
        return {"kind": "patch"}
    if isinstance(evidence, InBlob):
        side = _content_side(change)
        if not isinstance(side, RegularFile):
            raise ValueError(f"blob evidence needs a regular file: {change.path}")
        return {"kind": "blob", "blob": blob_path(side.sha256)}
    if isinstance(evidence, InState):
        return {"kind": "state"}
    if isinstance(evidence, Omitted):
        return {"kind": "omitted", "reason": evidence.reason}
    _assert_never(evidence)


def _change_json(change: FileChange) -> dict[str, Any]:
    if isinstance(change, Added):
        sides = {"change": "added", "after": _state_json(change.after)}
    elif isinstance(change, Modified):
        sides = {"change": "modified", "before": _state_json(change.before),
                 "after": _state_json(change.after)}
    elif isinstance(change, Deleted):
        sides = {"change": "deleted", "before": _state_json(change.before)}
    else:
        _assert_never(change)
    return {"path": change.path, **sides, "evidence": _evidence_json(change)}


def manifest_json(capture: WorkspaceCapture) -> dict[str, Any]:
    captured = isinstance(capture, CapturedChanges)
    return {
        "schema_version": WORKSPACE_CHANGES_VERSION,
        "captured": captured and capture.complete,
        "capture_error": None if captured else {"stage": capture.stage, "reason": capture.reason},
        "workspace_root": capture.workspace_root,
        "workspace_root_realpath": capture.workspace_root_realpath,
        "baseline_file_count": capture.baseline_file_count if captured else None,
        "limits": {"file_bytes": capture.limits.file_bytes, "total_bytes": capture.limits.total_bytes},
        "patch": CANDIDATE_PATCH_NAME if captured and capture.has_patch else None,
        "changes": [_change_json(change) for change in capture.changes] if captured else [],
    }


def _write_manifest(capture: WorkspaceCapture, changes_dir: Path) -> None:
    text = json.dumps(manifest_json(capture), indent=2, ensure_ascii=False) + "\n"
    (changes_dir / WORKSPACE_CHANGES_NAME).write_bytes(text.encode("utf-8"))


def _stage_changes(baseline: WorkspaceBaseline, ws: Path, changes_dir: Path,
                   limits: EvidenceLimits, after: Mapping[str, FileState]) -> CapturedChanges:
    stager = _EvidenceStager(baseline, ws, changes_dir, limits)
    changes: list[FileChange] = []
    for path in sorted(baseline.index.keys() | after.keys()):
        old, new = baseline.index.get(path), after.get(path)
        if old != new:
            changes.append(_change(path, old, new, stager.evidence(path, old, new)))
    if stager.patch:
        (changes_dir / CANDIDATE_PATCH_NAME).write_text("".join(stager.patch), encoding="utf-8")
    return CapturedChanges(str(ws), os.path.realpath(ws), len(baseline.index), limits, tuple(changes))


def capture_workspace_changes(baseline: Baseline, ws: Path, changes_dir: Path, *,
                              limits: EvidenceLimits = DEFAULT_LIMITS) -> WorkspaceCapture:
    """Never raises for I/O, because a capture failure must not cost the run its receipt."""
    def failed(stage: CaptureStage, reason: str) -> CaptureFailed:
        return CaptureFailed(stage, reason, str(ws), os.path.realpath(ws), limits)

    capture: WorkspaceCapture
    if isinstance(baseline, BaselineUnavailable):
        capture = failed("baseline", baseline.reason)
    else:
        try:
            after = index_tree(ws)
        except (OSError, UnicodeError, ValueError) as exc:
            capture = failed("diff", _failure_reason(exc))
        else:
            try:
                capture = _stage_changes(baseline, ws, changes_dir, limits, after)
            except (OSError, UnicodeError, ValueError) as exc:
                capture = failed("evidence", _failure_reason(exc))
    if isinstance(capture, CaptureFailed):
        _clear_outputs(changes_dir)
    try:
        _write_manifest(capture, changes_dir)
        return capture
    except (OSError, UnicodeError, ValueError) as exc:
        _clear_outputs(changes_dir)
        capture = failed("manifest", _failure_reason(exc))
    try:
        _write_manifest(capture, changes_dir)
    except (OSError, UnicodeError, ValueError):
        _clear_outputs(changes_dir)
    return capture


B = TypeVar("B")


@contextmanager
def captured_workspace(*, prefix: str, changes_dir: Path, build: Callable[[Path], B],
                       limits: EvidenceLimits = DEFAULT_LIMITS) -> Iterator[tuple[Path, B]]:
    """Own the model workspace: create, build, baseline, yield, capture, delete.

    An exception skips capture."""
    with tempfile.TemporaryDirectory(prefix=prefix) as wd, \
            tempfile.TemporaryDirectory(prefix="workspace-baseline-") as bd:
        ws = Path(wd)
        built = build(ws)
        baseline = snapshot_workspace(ws, Path(bd) / "tree")
        yield ws, built
        capture_workspace_changes(baseline, ws, changes_dir, limits=limits)


def _field(obj: Mapping[str, Any], key: str) -> Any:
    if key not in obj:
        raise ValueError(f"workspace changes manifest is missing {key!r}")
    return obj[key]


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _typed(value: Any, kind: type, label: str) -> Any:
    if type(value) is not kind:
        raise ValueError(f"{label} must be {kind.__name__}")
    return value


def _parse_state(raw: Any) -> FileState:
    obj = _object(raw, "file state")
    kind = _field(obj, "kind")
    if kind == "file":
        return RegularFile(_typed(_field(obj, "sha256"), str, "sha256"),
                           _typed(_field(obj, "size"), int, "size"),
                           _typed(_field(obj, "text"), bool, "text"),
                           _typed(_field(obj, "executable"), bool, "executable"))
    if kind == "symlink":
        return Symlink(_typed(_field(obj, "target"), str, "symlink target"))
    if kind == "special":
        return Special(_typed(_field(obj, "mode"), int, "mode"))
    raise ValueError(f"unknown file state kind {kind!r}")


def _parse_evidence(raw: Any) -> Evidence:
    kind = _field(_object(raw, "evidence"), "kind")
    if kind == "patch":
        return InPatch()
    if kind == "blob":
        return InBlob()
    if kind == "state":
        return InState()
    if kind == "omitted":
        reason = _field(raw, "reason")
        if reason not in ("oversize", "total_cap", "unreadable"):
            raise ValueError(f"unknown omission reason {reason!r}")
        return Omitted(reason)
    raise ValueError(f"unknown evidence kind {kind!r}")


def _parse_change(raw: Any) -> FileChange:
    obj = _object(raw, "workspace change")
    path = _typed(_field(obj, "path"), str, "change path")
    evidence = _parse_evidence(_field(obj, "evidence"))
    change = _field(obj, "change")
    if change == "added":
        return Added(path, _parse_state(_field(obj, "after")), evidence)
    if change == "modified":
        return Modified(path, _parse_state(_field(obj, "before")),
                        _parse_state(_field(obj, "after")), evidence)
    if change == "deleted":
        return Deleted(path, _parse_state(_field(obj, "before")), evidence)
    raise ValueError(f"unknown change kind {change!r}")


def parse_workspace_changes(raw: Any) -> WorkspaceCapture:
    obj = _object(raw, "workspace changes manifest")
    if _typed(_field(obj, "schema_version"), int, "schema_version") != WORKSPACE_CHANGES_VERSION:
        raise ValueError("unsupported workspace changes schema version")
    _typed(_field(obj, "captured"), bool, "captured")
    limits_raw = _object(_field(obj, "limits"), "limits")
    limits = EvidenceLimits(_typed(_field(limits_raw, "file_bytes"), int, "file_bytes"),
                            _typed(_field(limits_raw, "total_bytes"), int, "total_bytes"))
    root = _typed(_field(obj, "workspace_root"), str, "workspace_root")
    realpath = _typed(_field(obj, "workspace_root_realpath"), str, "workspace_root_realpath")
    error = _field(obj, "capture_error")
    capture: WorkspaceCapture
    if error is None:
        changes = _field(obj, "changes")
        if not isinstance(changes, list):
            raise ValueError("changes must be a JSON array")
        capture = CapturedChanges(
            root, realpath, _typed(_field(obj, "baseline_file_count"), int, "baseline_file_count"),
            limits, tuple(_parse_change(item) for item in changes))
    else:
        error_obj = _object(error, "capture_error")
        stage = _field(error_obj, "stage")
        if stage not in ("baseline", "diff", "evidence", "manifest"):
            raise ValueError(f"unknown capture stage {stage!r}")
        capture = CaptureFailed(stage, _typed(_field(error_obj, "reason"), str, "reason"),
                                root, realpath, limits)
    if manifest_json(capture) != obj:
        raise ValueError("workspace changes manifest disagrees with its derived fields")
    return capture


def load_workspace_changes(run_dir: Path) -> WorkspaceCapture | None:
    path = run_dir / WORKSPACE_CHANGES_NAME
    if not path.exists() and not path.is_symlink():
        return None
    return parse_workspace_changes(strict_json_loads(path.read_bytes()))


def workspace_changes_state(run_dir: Path,
                            artifact: ArtifactSetObservation) -> WorkspaceChangesState | None:
    """The workspace-evidence claim for one run; None when no manifest exists.

    CAPTURED requires a complete artifact set, a canonical manifest with no
    omitted evidence, and every file it references committed in the inventory
    under its own digest. Zero changes is captured."""
    try:
        capture = load_workspace_changes(run_dir)
    except (OSError, ValueError):
        return WorkspaceChangesState.INVALID
    if capture is None:
        return None
    if not isinstance(artifact, CompleteArtifactSet):
        return WorkspaceChangesState.INVALID
    if isinstance(capture, CaptureFailed):
        return WorkspaceChangesState.FAILED
    inventory = artifact.inventory_sha256
    if capture.has_patch and CANDIDATE_PATCH_NAME not in inventory:
        return WorkspaceChangesState.INVALID
    if any(inventory.get(blob_path(digest)) != digest for digest in capture.blob_digests()):
        return WorkspaceChangesState.INVALID
    return WorkspaceChangesState.CAPTURED if capture.complete else WorkspaceChangesState.PARTIAL
