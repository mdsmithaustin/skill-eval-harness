from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cli_contracts import die
from content_digests import canonical_json_sha256, tree_sha256
from json_contracts import strict_json_loads
from trace_normalization import parse_trace_jsonl_text_with_lines

_DRAFT_NAME = "draft.json"
_DRAFT_SCHEMA_VERSION = 3
_ID = re.compile(r"session-([0-9a-f]{64})")
_REDACTED = "[REDACTED]"
_INJECTED_CONTEXT = ("<skill>", "<environment_context>", "<user_instructions>", "# AGENTS.md instructions")
_SECRET_VALUE = re.compile(
    r"""(?ix)
    (\b[\w-]*(?:api[_ -]?key|token|secret|password|credential|cookie|authorization)[\w-]*
        ["']?\s*[:=]\s*)
    (?:"[^"\r\n]*"|'[^'\r\n]*'|bearer\s+[^\s,;]+|[^\s,;}]+)
    """
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?-----END [^-\n]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"(?i)\bbearer[ \t]+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+)"),
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),
)


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    for value in secrets:
        text = text.replace(value, _REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_REDACTED, text)
    return _SECRET_VALUE.sub(_replace_secret, text)


def _replace_secret(match: re.Match[str]) -> str:
    value = match[0][len(match[1]):]
    quote = value[0] if value[0] in {'"', "'"} and value.endswith(value[0]) else ""
    return f"{match[1]}{quote}{_REDACTED}{quote}"


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _portable_relative_path(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name) and name != "." and path.as_posix() == name and not path.is_absolute()
        and ".." not in path.parts
        and not any(char in name for char in '\\:*?"<>|')
        and not any(ord(char) < 32 for char in name)
    )


def _fixture_names(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise TypeError("required fixtures must be a list of relative paths")
    names = []
    for value in raw:
        name = _text(value, "fixture path")
        if not _portable_relative_path(name):
            raise ValueError("fixture paths must be portable relative paths without traversal")
        names.append(name)
    return tuple(names)


def _fixture_path(root: Path, name: str) -> Path:
    path = root
    if path.is_symlink():
        raise ValueError(f"fixture paths must not contain symlinks: {name}")
    for component in PurePosixPath(name).parts:
        path /= component
        if path.is_symlink():
            raise ValueError(f"fixture paths must not contain symlinks: {name}")
    return path


_EXEC_WRAPPER = re.compile(
    r"\s*(?:(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*)?await\s+tools\.exec_command"
    r"\(\s*(\{.*\})\s*\)\s*;?(?:\s*text\(\s*\1\.output\s*\)\s*;?)?\s*",
    re.DOTALL,
)
_FILE_SUFFIXES = frozenset({
    ".csv", ".docx", ".gif", ".html", ".jpeg", ".jpg", ".json", ".jsonl", ".log",
    ".md", ".pdf", ".png", ".py", ".sql", ".svg", ".toml", ".tsv", ".txt",
    ".webp", ".xlsx", ".xml", ".yaml", ".yml", ".zip",
})
_REFERENCE = re.compile(r"`([^`\r\n]+)`|([^\s`,;!?()\[\]{}<>\"']+)")


def _prompt_files(prompt: str) -> Iterator[str]:
    for match in _REFERENCE.finditer(prompt):
        token = match[1] or match[2].rstrip(".:")
        path = PurePosixPath(token)
        if "@" not in token and _portable_relative_path(path.as_posix()) and path.suffix.lower() in _FILE_SUFFIXES:
            yield path.as_posix()


@dataclass(frozen=True)
class FileRead:
    path: str | None
    workdir: str | None = None

    def relative_to(self, base: str | None, cwd: str | None) -> str | None:
        if self.path is None:
            return None
        workspace = PurePosixPath(base) if base else None
        working = PurePosixPath(self.workdir or cwd or ".")
        if workspace is not None and not working.is_absolute():
            working = workspace / working
        path = working / self.path
        if workspace is not None:
            if not path.is_relative_to(workspace):
                return None
            path = path.relative_to(workspace)
        name = path.as_posix()
        return name if _portable_relative_path(name) else None


def _recorded_read(payload: dict[str, Any]) -> FileRead | None:
    name = payload.get("name")
    if payload.get("type") == "custom_tool_call" and name == "exec":
        code = payload.get("input")
        if not isinstance(code, str):
            return FileRead(None)
        wrapper = _EXEC_WRAPPER.fullmatch(code)
        if wrapper is None:
            return FileRead(None)
        encoded = wrapper[2]
    elif payload.get("type") == "function_call" and name in ("exec_command", "read_file"):
        encoded = payload.get("arguments")
    else:
        return None
    if not isinstance(encoded, str):
        return FileRead(None)
    try:
        args = strict_json_loads(encoded)
    except ValueError:
        return FileRead(None)
    if not isinstance(args, dict):
        return FileRead(None)
    workdir = args.get("workdir")
    if workdir is not None and not isinstance(workdir, str):
        return FileRead(None)
    if name == "read_file":
        path = args.get("path")
        return FileRead(path if isinstance(path, str) and path else None, workdir)
    command = args.get("cmd")
    if not isinstance(command, str):
        return FileRead(None)
    try:
        argv = shlex.split(command)
    except ValueError:
        return FileRead(None)
    if (
        len(argv) != 2 or argv[0] != "cat" or argv[1].startswith("-")
        or any(char in command for char in "$&#*?[];<>|\n`")
    ):
        return FileRead(None)
    return FileRead(argv[1], workdir)


@dataclass
class Turn:
    cwd: str | None
    request: str = ""
    request_line: int = 0
    output: str | None = None
    aborted: bool = False
    reads: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    def observe(self, payload: dict[str, Any], line: int, base: str | None) -> None:
        if payload.get("type") != "message":
            if self.request_line:
                read = _recorded_read(payload)
                if read is not None:
                    name = read.relative_to(base, self.cwd)
                    if name is None:
                        self.unresolved += (f"Unmapped tool input at trace line {line}",)
                    elif name not in self.reads:
                        self.reads += (name,)
            return
        content = payload.get("content")
        role = payload.get("role")
        if role == "user":
            parts = [part for part in content if isinstance(part, dict)] if isinstance(content, list) else []
            text = [
                part["text"] for part in parts
                if part.get("type") == "input_text" and isinstance(part.get("text"), str)
            ]
            request = "\n".join(value for value in text if not value.lstrip().startswith(_INJECTED_CONTEXT))
            attachments = any(part.get("type") != "input_text" for part in parts)
            if text and not request.strip() and not attachments:
                return
            self.request, self.request_line, self.output = request, line, None
            self.aborted = False
            self.reads = ()
            self.unresolved = (f"Unsupported attachment at trace line {line}",) if attachments else ()
        elif (
            role == "assistant" and self.request_line and not self.aborted
            and payload.get("phase") in (None, "final_answer")
        ):
            if isinstance(content, list):
                reply = "\n".join(
                    part["text"] for part in content
                    if isinstance(part, dict) and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                )
                if reply.strip():
                    self.output = reply


@dataclass(frozen=True)
class Session:
    identity: str
    trace_digest: str
    request_line: int
    prompt: str
    output: str | None
    reads: tuple[str, ...]
    unresolved: tuple[str, ...]

    @property
    def case_id(self) -> str:
        return f"session-{self.identity}"

    @classmethod
    def from_rollout(cls, raw: bytes) -> Session:
        records, errors, lines, _ = parse_trace_jsonl_text_with_lines(
            raw.decode("utf-8"), strict_json_errors=False,
        )
        ids: set[str] = set()
        base: str | None = None
        turn = Turn(base)
        for record, line in zip(records, lines, strict=True):
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            record_type = record.get("type")
            cwd = payload.get("cwd")
            if record_type == "session_meta":
                identity = payload.get("id") or payload.get("session_id")
                if isinstance(identity, str) and identity.strip():
                    ids.add(identity.strip())
                if isinstance(cwd, str) and PurePosixPath(cwd).is_absolute():
                    base = cwd
                    turn.cwd = base
            elif record_type == "turn_context" and isinstance(cwd, str):
                turn.cwd = cwd
            elif record_type == "event_msg":
                event = payload.get("type")
                if event in ("task_started", "turn_started"):
                    turn = Turn(base)
                elif event == "task_complete" and turn.request_line and not turn.aborted:
                    final = payload.get("last_agent_message")
                    if isinstance(final, str) and final.strip():
                        turn.output = final
                elif event in ("turn_aborted", "task_aborted"):
                    turn.output = None
                    turn.aborted = True
            elif record_type == "response_item":
                turn.observe(payload, line, base)
        if len(ids) != 1:
            raise ValueError("expected a Codex rollout for one session")
        if not turn.request.strip():
            raise ValueError("the latest Codex turn has no supported text request")
        if errors:
            turn.unresolved += ("Some trace records could not be read",)
        return cls(
            _digest(ids.pop().encode("utf-8")), _digest(raw), turn.request_line,
            turn.request, turn.output, turn.reads, turn.unresolved,
        )


@dataclass(frozen=True)
class Draft:
    path: Path
    case_id: str
    prompt: str
    required: tuple[str, ...]
    unresolved: tuple[str, ...]
    import_digest: str

    @classmethod
    def load(cls, path: Path) -> Draft:
        data = _json_object(path / _DRAFT_NAME, "draft")
        if data.get("schema_version") != _DRAFT_SCHEMA_VERSION:
            raise ValueError("import this session into a new private draft directory before review")
        case_id = _text(data.get("case_id"), "draft case ID")
        match = _ID.fullmatch(case_id)
        provenance = data.get("provenance")
        if match is None or not isinstance(provenance, dict) or provenance.get("session_sha256") != match[1]:
            raise ValueError("draft must contain a matching case ID and session digest")
        unresolved = data.get("unresolved_inputs")
        if not isinstance(unresolved, list):
            raise TypeError("draft unresolved_inputs must be a list")
        return cls(
            path, case_id, _text(data.get("prompt"), "draft prompt"),
            _fixture_names(data.get("required_inputs")),
            tuple(_text(item, "unresolved input") for item in unresolved),
            _text(provenance.get("import_sha256"), "draft import digest"),
        )

    def fixture_bytes(self) -> dict[str, bytes]:
        if len({PurePosixPath(name).name.casefold() for name in self.required}) != len(self.required):
            raise ValueError("duplicate fixture filenames: rename the replay copies and update required_inputs and the prompt")
        snapshots = {}
        missing = []
        for name in self.required:
            source = _fixture_path(self.path / "fixtures", name)
            if source.is_file():
                snapshots[name] = source.read_bytes()
            else:
                missing.append(name)
        if missing:
            raise ValueError("missing required inputs: " + ", ".join(missing))
        return snapshots


@dataclass(frozen=True)
class ApprovedCase:
    draft: Draft
    expected_behavior: list[str]
    assertions: list[Any]
    fixtures: dict[str, bytes]

    @classmethod
    def load(cls, draft: Draft, review_path: Path) -> ApprovedCase:
        fixtures = draft.fixture_bytes()
        review = _json_object(review_path, "review")
        if review.get("approved") is not True:
            raise ValueError("review must set approved to true after inspecting fixtures and assertions")
        if draft.unresolved and review.get("unresolved_inputs_reviewed") is not True:
            raise ValueError("resolve unmapped inputs, then set unresolved_inputs_reviewed to true in the review")
        recorded = _fixture_path(draft.path, "recorded-output.txt")
        if not recorded.is_file() or not recorded.read_text(encoding="utf-8").strip():
            raise ValueError("no recorded final answer: supply a redacted replay as recorded-output.txt")
        expected = review.get("expected_behavior")
        assertions = review.get("assertions")
        if not isinstance(expected, list) or not expected:
            raise ValueError("review must supply a non-empty expected_behavior list")
        if not isinstance(assertions, list) or not assertions:
            raise ValueError("review must supply a non-empty assertions list")
        hashes = {name: _digest(content) for name, content in fixtures.items()}
        if review.get("fixtures") != hashes:
            raise ValueError("review.fixtures must match the SHA-256 of every required fixture")
        return cls(draft, [_text(item, "expected behavior") for item in expected], assertions, fixtures)

    @property
    def fixture_directory(self) -> Path:
        return Path("fixtures") / self.draft.case_id / tree_sha256(self.fixtures.items())

    def manifest_row(self) -> dict[str, Any]:
        return {
            "id": self.draft.case_id,
            "kind": "behavior",
            "split": "tune",
            "source": "production",
            "eval_intent": "regression",
            "prompt": self.draft.prompt,
            "files": [(self.fixture_directory / name).as_posix() for name in self.fixtures],
            "expected_behavior": self.expected_behavior,
            "assertions": self.assertions,
        }

    def install_fixtures(self, manifest_dir: Path) -> Path | None:
        if not self.fixtures:
            return None
        relative = self.fixture_directory
        destination = _fixture_path(manifest_dir, relative.as_posix())
        if destination.is_dir():
            if any(
                not (path := _fixture_path(manifest_dir, (relative / name).as_posix())).is_file()
                or path.read_bytes() != content
                for name, content in self.fixtures.items()
            ):
                raise ValueError("existing regression fixture snapshot differs from the approved files")
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".regression-", dir=destination.parent) as temp:
            stage = Path(temp) / "fixtures"
            for name, content in self.fixtures.items():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            stage.rename(destination)
        return destination


@contextmanager
def lock_manifest(manifest: Path) -> Iterator[None]:
    lock = manifest.with_name(f".{manifest.name}.promotion.lock")
    with lock.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def import_session(
    trace: Path, out_dir: Path, fixture_root: Path | None,
    input_paths: list[str], redact_file: Path | None,
) -> tuple[Draft, bool]:
    session = Session.from_rollout(trace.read_bytes())
    destination = out_dir / session.case_id
    redaction = redact_file.read_bytes() if redact_file is not None else None
    import_digest = canonical_json_sha256({
        "trace": session.trace_digest,
        "fixture_root": str(fixture_root.resolve()) if fixture_root is not None else None,
        "required": sorted(set(input_paths)),
        "redaction": _digest(redaction) if redaction is not None else None,
    })

    def reuse() -> tuple[Draft, bool]:
        draft = Draft.load(destination)
        if draft.import_digest != import_digest:
            raise ValueError("trace or import options changed: use a new private --out-dir and review the new draft")
        return draft, False

    if destination.exists():
        return reuse()
    secrets = () if redaction is None else tuple(sorted(
        {line for line in redaction.decode("utf-8").splitlines() if line}, key=lambda value: (-len(value), value),
    ))
    prompt = _redact(session.prompt, secrets)
    observed_basenames = {PurePosixPath(name).name for name in session.reads}
    references = [
        name for name in _prompt_files(prompt) if "/" in name or name not in observed_basenames
    ]
    required = _fixture_names(list(dict.fromkeys([*input_paths, *session.reads, *references])))
    available: dict[str, bytes] = {}
    missing: dict[str, str] = {}
    for name in required:
        if _redact(name, secrets) != name:
            raise ValueError("a fixture path contains sensitive text; choose a sanitized filename")
        source = _fixture_path(fixture_root, name) if fixture_root is not None else None
        if source is None or not source.is_file():
            missing[name] = "file unavailable"
            continue
        raw = source.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            missing[name] = "review and supply non-UTF-8 file"
            continue
        if "\x00" in text:
            missing[name] = "review and supply binary file"
            continue
        available[name] = _redact(text, secrets).encode("utf-8")
    draft = {
        "schema_version": _DRAFT_SCHEMA_VERSION,
        "case_id": session.case_id,
        "prompt": prompt,
        "provenance": {
            "format": "codex-rollout",
            "session_sha256": session.identity,
            "trace_sha256": session.trace_digest,
            "import_sha256": import_digest,
            "request_line": session.request_line,
        },
        "required_inputs": list(required),
        "available_inputs": list(available),
        "missing_inputs": missing,
        "unresolved_inputs": list(session.unresolved),
        "has_recorded_output": session.output is not None,
    }
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".session-", dir=out_dir) as temp:
        stage = Path(temp) / "draft"
        stage.mkdir(mode=0o700)
        fixtures = stage / "fixtures"
        fixtures.mkdir()
        for name, content in available.items():
            target = fixtures / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (stage / _DRAFT_NAME).write_text(
            json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        if session.output is not None:
            (stage / "recorded-output.txt").write_text(_redact(session.output, secrets), encoding="utf-8")
        try:
            stage.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            return reuse()
    return Draft(destination, session.case_id, prompt, required, session.unresolved, import_digest), True


def import_case_command(args: argparse.Namespace) -> int:
    try:
        draft, created = import_session(
            Path(args.trace), Path(args.out_dir),
            Path(args.fixture_root) if args.fixture_root else None, args.require_file,
            Path(args.redact_file) if args.redact_file else None,
        )
        missing = [
            name for name in draft.required
            if not _fixture_path(draft.path / "fixtures", name).is_file()
        ]
    except (TypeError, ValueError, OSError) as exc:
        die(str(exc))
    print(f"{'Created' if created else 'Reused'} draft {draft.path}")
    if missing:
        print("Missing required inputs: " + ", ".join(missing))
    if draft.unresolved:
        print("Review unmapped inputs in draft.json before promotion.")
    if not (draft.path / "recorded-output.txt").is_file():
        print("No recorded final answer. Supply a redacted replay before promotion.")
    return 0
