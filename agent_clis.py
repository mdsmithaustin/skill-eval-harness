from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass as _dataclass
from pathlib import Path
from typing import Any

from agent_capabilities import VIBE_DEFAULT_CMD
from json_contracts import iter_json_objects
from trace_normalization import text_mentions_mounted_skill

CODEX_HOME_FILES = ("auth.json", "config.toml")


VIBE_READ_ONLY_TOOLS = ("skill", "read_file", "grep")


def seed_codex_home(codex_home: Path) -> dict[str, Any]:
    """Copy portable Codex auth/config into an isolated CODEX_HOME.

    `$CODEX_HOME/skills` is also Codex's skill-discovery surface, so seeding is
    file-allowlisted: copy auth/config files only, never user skills/plugins.
    `--ignore-user-config` keeps copied config from influencing native harness
    runs unless Codex needs it for auth compatibility."""
    codex_home.mkdir(parents=True, exist_ok=True)
    source = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    copied: list[str] = []
    for name in CODEX_HOME_FILES:
        src = source / name
        dst = codex_home / name
        if not src.is_file():
            continue
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)
        copied.append(name)
    return {"codex_home": str(codex_home), "codex_home_files_copied": copied, "config_isolated": True}


def codex_env_for_home(codex_home: Path) -> tuple[dict[str, str], dict[str, Any]]:
    env = os.environ.copy()
    seeded = seed_codex_home(codex_home)
    env["CODEX_HOME"] = seeded["codex_home"]
    meta = {**seeded, "codex_home": "<isolated CODEX_HOME outside workdir>"}
    return env, meta


CODEX_SESSIONS_DIRNAME = "sessions"


CODEX_ROLLOUT_SKILL_TAG = "<skill>"


_CODEX_SKILL_TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL) for tag in ("name", "path")
}


@_dataclass(frozen=True)
class CodexRollout:
    """Where the persisted transcript of one `codex exec --json` thread was
    looked for, and what it held. Codex writes it under
    `$CODEX_HOME/sessions/<y>/<m>/<d>/rollout-<stamp>-<thread_id>.jsonl`; the
    thread id comes from the stream's `thread.started` event. `status` is
    `found`, `not_found`, or `no_thread_id`; `home` names which CODEX_HOME held
    it (`isolated` for the run's scratch home, `ambient` for the caller's)."""

    status: str
    home: str | None = None
    file_name: str | None = None
    text: str | None = None

    def metadata(self) -> dict[str, Any]:
        """The JSON-safe record a trigger row keeps so a reader can tell whether
        the rollout detector had anything to read."""
        meta: dict[str, Any] = {"codex_rollout_status": self.status}
        if self.home is not None:
            meta["codex_rollout_home"] = self.home
        if self.file_name is not None:
            meta["codex_rollout_file"] = self.file_name
        return meta


def codex_thread_id(stdout: str) -> str | None:
    """The thread id Codex announces in its `thread.started` event."""
    for record in iter_json_objects(stdout, strict=False):
        if isinstance(record, dict) and record.get("type") == "thread.started":
            thread_id = record.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id.strip()
            return None
    return None


def codex_rollout_path(codex_home: Path, thread_id: str) -> Path | None:
    sessions = codex_home / CODEX_SESSIONS_DIRNAME
    if not sessions.is_dir():
        return None
    suffix = f"-{thread_id}.jsonl"
    matches = sorted(
        path for path in sessions.rglob("rollout-*.jsonl")
        if path.is_file() and path.name.endswith(suffix))
    return matches[0] if matches else None


def locate_codex_rollout(stdout: str, isolated_home: Path | None = None) -> CodexRollout:
    """Find the rollout for the thread `stdout` announces. The run's isolated
    CODEX_HOME is searched first, then the ambient `$CODEX_HOME` (default
    `~/.codex`) for a `--codex-cmd` wrapper that ignored the isolated home. A
    thread id is unique, so either hit is this run's own transcript."""
    thread_id = codex_thread_id(stdout)
    if thread_id is None:
        return CodexRollout(status="no_thread_id")
    homes: list[tuple[str, Path]] = []
    if isolated_home is not None:
        homes.append(("isolated", isolated_home))
    homes.append(("ambient", Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))))
    for label, home in homes:
        path = codex_rollout_path(home, thread_id)
        if path is not None:
            return CodexRollout(status="found", home=label, file_name=path.name,
                                text=path.read_text(encoding="utf-8", errors="replace"))
    return CodexRollout(status="not_found")


def _codex_skill_tag(text: str, tag: str) -> str | None:
    match = _CODEX_SKILL_TAG_PATTERNS[tag].search(text)
    return match.group(1).strip() if match else None


def codex_rollout_skill_loads(rollout_text: str, skill_names: list[str], copied_paths: list[Path]) -> list[str]:
    """Skill loads a Codex rollout proves through the CLI's own `<skill>`
    injection. The injection is a user-role `response_item` whose text opens
    with `<skill>`, names a mounted skill, and, when present, carries a `<path>`
    under the mount. Tool calls, listings, prose, and outputs can mention a skill without
    loading it, so they do not count here."""
    names = set(skill_names)
    evidence: list[str] = []
    for record in iter_json_objects(rollout_text):
        if not isinstance(record, dict) or record.get("type") != "response_item":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        item_type = str(payload.get("type") or "")
        if item_type == "message":
            if payload.get("role") != "user":
                continue
            for block in payload.get("content") or []:
                text = str(block.get("text") or "") if isinstance(block, dict) else ""
                if not text.startswith(CODEX_ROLLOUT_SKILL_TAG):
                    continue
                name = _codex_skill_tag(text, "name")
                path = _codex_skill_tag(text, "path")
                if name in names and (path is None or text_mentions_mounted_skill(path, copied_paths)):
                    evidence.append(f"rollout skill injection: {name}" + (f" ({path})" if path else ""))
    return evidence[:5]


def seed_vibe_home(vibe_home: Path) -> dict[str, Any]:
    """Create an isolated VIBE_HOME for native Vibe runs.

    Vibe discovers user skills/tools from VIBE_HOME, so every harness run gets a
    fresh home by default. Auth is carried by MISTRAL_API_KEY; if that env var is
    absent but the user has the documented ~/.vibe/.env file, copy only that env
    file and no skills/config/agents, preserving baseline isolation."""
    vibe_home.mkdir(parents=True, exist_ok=True)
    copied_env = False
    if not os.environ.get("MISTRAL_API_KEY"):
        source_home = Path(os.environ.get("VIBE_HOME") or (Path.home() / ".vibe"))
        src = source_home / ".env"
        if src.is_file() and src.resolve() != (vibe_home / ".env").resolve():
            shutil.copy2(src, vibe_home / ".env")
            copied_env = True
    return {"vibe_home": str(vibe_home), "vibe_env_file_copied": copied_env}


def vibe_env_for_home(vibe_home: Path, model: str | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    env = os.environ.copy()
    seeded = seed_vibe_home(vibe_home)
    env["VIBE_HOME"] = seeded["vibe_home"]
    if model:
        env["VIBE_ACTIVE_MODEL"] = model
    meta = {
        **seeded,
        "vibe_home": "<isolated VIBE_HOME outside workdir>",
        "vibe_home_outside_workdir": True,
        "config_isolated": True,
        **({"active_model_env": "VIBE_ACTIVE_MODEL"} if model else {}),
    }
    return env, meta


def build_vibe_cli_argv(vibe_cmd: str | None = None, *, prompt: str, cwd: Path | str | None = None,
                        output: str = "streaming", tools: Iterable[str] | None = VIBE_READ_ONLY_TOOLS,
                        auto_approve: bool = True, max_turns: int | None = None,
                        max_price: float | None = None, max_tokens: int | None = None) -> list[str]:
    try:
        argv = shlex.split(vibe_cmd or VIBE_DEFAULT_CMD)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid --vibe-cmd: {exc}") from exc
    if not argv:
        argv = [VIBE_DEFAULT_CMD]
    argv += ["--prompt", prompt, "--output", output]
    if cwd is not None:
        argv += ["--workdir", str(cwd)]
    argv.append("--trust")
    if auto_approve:
        argv.append("--auto-approve")
    for tool in (tools or ()):
        argv += ["--enabled-tools", str(tool)]
    if max_turns is not None:
        argv += ["--max-turns", str(max_turns)]
    if max_price is not None:
        argv += ["--max-price", str(max_price)]
    if max_tokens is not None:
        argv += ["--max-tokens", str(max_tokens)]
    return argv


def redact_vibe_prompt_arg(argv: list[str]) -> list[str]:
    redacted = list(argv)
    for idx in range(len(redacted) - 2, -1, -1):
        arg = redacted[idx]
        if arg == "--prompt":
            redacted[idx + 1] = "<prompt>"
            break
    return redacted


CLAUDE_TRIGGER_CONTEXT_ISOLATION_ARGS = (
    "--setting-sources", "project", "--strict-mcp-config",
    "--settings", '{"disableBundledSkills":true,"autoMemoryEnabled":false}',
)


def _walk_codex_host_skills() -> tuple[Path, list[str], list[str]]:
    root = Path.home() / ".agents" / "skills"
    host_skills: list[str] = []
    directories: list[str] = []
    walked: set[str] = set()
    for directory, subdirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(directory)
        if real in walked:
            subdirs.clear()
            continue
        walked.add(real)
        directories.append(directory)
        if "SKILL.md" in files:
            host_skills.append(str(Path(directory) / "SKILL.md"))
    host_skills.sort()
    return root, host_skills, directories


def codex_host_skill_paths() -> list[str]:
    """Every `SKILL.md` under the operator's `~/.agents/skills`, sorted."""
    return _walk_codex_host_skills()[1]


_CODEX_FILE_URL_SAFE = "/[]^|~';:!$&()*+,=@"


_CODEX_URL_DROPPED = {ord("\t"): None, ord("\n"): None, ord("\r"): None}


_CODEX_PATH_TAIL = r'[^\s"\],]*?(?=:?\\*(?:[\s"\],]|$))'


def _rust_debug_escape(text: str) -> str:
    escapes = {"\\": "\\\\", '"': '\\"', "\t": "\\t", "\n": "\\n", "\r": "\\r", "\0": "\\0"}
    return "".join(escapes.get(ch) or (ch if ch.isprintable() else f"\\u{{{ord(ch):x}}}") for ch in text)


def _codex_path_spellings(path: str) -> set[str]:
    encoded = {path, toml_basic_string(path)[1:-1], json.dumps(path)[1:-1],
               json.dumps(path, ensure_ascii=False)[1:-1]}
    url = "file://" + urllib.parse.quote(path.translate(_CODEX_URL_DROPPED), safe=_CODEX_FILE_URL_SAFE)
    return encoded | {_rust_debug_escape(text) for text in encoded} | {url}


@_dataclass(frozen=True)
class CodexHostPathRedactor:
    """Removes host skill paths from Codex output.

    `roots` are the spellings of `~/.agents/skills` and its realpath: any span
    starting with one is redacted up to the next path delimiter, which covers
    paths Codex prints in a form nobody derived. `paths` are the spellings of
    each walked directory, each `SKILL.md`, and their realpaths, which also
    covers a path with a space or quote in it and a symlink target outside the
    root. Both are sorted longest first so a whole path wins over its prefix."""

    roots: tuple[str, ...]
    paths: tuple[str, ...]

    def __call__(self, text: str) -> str:
        present = [s for s in self.paths if s in text]
        if not present and not any(s in text for s in self.roots):
            return text
        prefixes = sorted([*present, *self.roots], key=len, reverse=True)
        pattern = "(?:" + "|".join(map(re.escape, prefixes)) + ")" + _CODEX_PATH_TAIL
        return re.sub(pattern, "[REDACTED]", text)


def codex_host_path_redactor() -> CodexHostPathRedactor:
    root, host_skills, directories = _walk_codex_host_skills()
    roots = {str(root), os.path.realpath(root)}
    paths: set[str] = set()
    for path in {*host_skills, *directories}:
        for spelled in {path, os.path.realpath(path)} - roots:
            paths |= _codex_path_spellings(spelled)
    root_spellings: set[str] = set().union(*(_codex_path_spellings(r) for r in roots))
    return CodexHostPathRedactor(
        roots=tuple(sorted(root_spellings, key=len, reverse=True)),
        paths=tuple(sorted(paths - root_spellings, key=len, reverse=True)))


def toml_basic_string(value: str) -> str:
    """Encode `value` as a quoted TOML basic string.

    `json.dumps` is the wrong encoder here: JSON represents a `str` as
    UTF-16, so a character above U+FFFF (an emoji, for example) comes out as
    an escaped surrogate pair (`\\ud83d\\ude80`), which is not a Unicode
    scalar value and not legal TOML; Codex then reads the whole
    `skills.config` value as an opaque string and dies with `Error loading
    config.toml: invalid type: string ...`. TOML also never accepts `\\/`,
    which JSON allows (though does not by default emit). This encoder escapes
    only the quote, the backslash, and control characters (TOML requires
    every control character other than tab escaped, including U+007F, which
    JSON does not escape), and otherwise copies each character through
    literally, astral code points included."""
    out = ['"']
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == '\\':
            out.append('\\\\')
        elif ch == '\b':
            out.append('\\b')
        elif ch == '\t':
            out.append('\\t')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\f':
            out.append('\\f')
        elif ch == '\r':
            out.append('\\r')
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def codex_trigger_context_isolation_args() -> tuple[list[str], list[str]]:
    """Codex flags for a trigger, answer, or judge run, and the form a row records.

    An isolated CODEX_HOME still lists every skill under the operator's
    ~/.agents/skills, Codex's bundled skills, and the account's codex_apps
    connector. A run must keep the skills it mounts (`$CODEX_HOME/skills`, the
    workspace's `.agents/skills`) listed, so it cannot hide the skill catalog.
    Codex has no switch for the ~/.agents/skills root alone, so each host
    skill is disabled by path; the recorded form counts them instead of
    listing the operator's skill paths. Moving HOME would also move tool and
    version-manager lookups."""
    host_skills = codex_host_skill_paths()
    entries = ",".join(f"{{path={toml_basic_string(path)},enabled=false}}" for path in host_skills)

    def args(skills_config: str) -> list[str]:
        return ["-c", "skills.bundled.enabled=false", "-c", skills_config, "--disable", "apps"]

    return (args(f"skills.config=[{entries}]"),
            args(f"skills.config=<{len(host_skills)} host skill(s) disabled>"))
