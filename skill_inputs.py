from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass as _dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from ablation_model import (
    AblationMode,
    Arm,
    Component,
    ComponentClass,
    ExpectedProvenance,
    MaterializedArm,
    Mechanism,
    Provenance,
    TreeIdentity,
    ablation_id_of,
)
from cli_contracts import die
from content_digests import canonical_json_sha256, directory_tree_sha256
from json_contracts import strict_json_loads
from manifest_contracts import CaseKind, CasePopulation, Split
from trigger_contracts import SkillId, parse_skill_constraints

VALID_SPLITS = frozenset(Split.values())


def reject_nonfinite_numbers(value: Any, *, location: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite_numbers(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_nonfinite_numbers(child, location=f"{location}[{index}]")


class UniqueKeySafeLoader(yaml.SafeLoader):
    """PyYAML safe loader that rejects duplicate mapping keys."""


def _construct_unique_yaml_mapping(
    loader: UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"found duplicate key {key!r}", key_node.start_mark)
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = strict_json_loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        die(f"no such file: {path}")
    except json.JSONDecodeError as exc:
        die(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        die(f"{path} must contain a JSON object")
    return data


def apply_dataset_row(value: Any, row: dict[str, Any]) -> Any:
    """Fill {key} placeholders from a dataset row throughout a case template.
    Plain replace, not str.format — prompts and regex assertions legitimately
    contain braces ({output_dir}, quantifiers) that format() would explode on."""
    if isinstance(value, str):
        out = value
        for key, cell in row.items():
            out = out.replace("{" + str(key) + "}", str(cell))
        return out
    if isinstance(value, list):
        return [apply_dataset_row(item, row) for item in value]
    if isinstance(value, dict):
        return {key: apply_dataset_row(item, row) for key, item in value.items()}
    return value


def materialize_dataset_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The dataset abstraction (roadmap 2.5): a case with `template: <dataset>`
    fans one template over the dataset's rows into concrete cases — stable ids
    (`<case>-<row id or 1-based index>`), materialized EARLY so validation,
    leakage lint, prepare, grade, and report all see ordinary cases."""
    cases = manifest.get("cases", [])
    if not isinstance(cases, list):
        die("manifest.cases must be a list")
    datasets = manifest.get("datasets") or {}
    out: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, 1):
        if (not isinstance(case, dict)
                or not all(isinstance(key, str) for key in case)):
            die(f"manifest case #{case_index} must be an object with string keys")
        case_row = {
            key: value for key, value in case.items() if isinstance(key, str)
        }
        dataset_id = case_row.get("template")
        if not dataset_id:
            out.append(case_row)
            continue
        rows = datasets.get(str(dataset_id))
        if not isinstance(rows, list) or not rows:
            die(f"case {case.get('id')!r}: template references unknown or empty dataset {dataset_id!r}")
        for i, row in enumerate(rows, 1):
            if (not isinstance(row, dict)
                    or not all(isinstance(key, str) for key in row)):
                die(f"dataset {dataset_id!r}: row #{i} must be an object with string keys")
            dataset_row = {
                key: value for key, value in row.items() if isinstance(key, str)
            }
            materialized = {
                key: apply_dataset_row(value, dataset_row)
                for key, value in case_row.items() if key != "template"
            }
            materialized["id"] = f"{case_row.get('id')}-{dataset_row.get('id', i)}"
            materialized["dataset"] = str(dataset_id)
            out.append(materialized)
    return out


def iter_cases(manifest: dict[str, Any], split: str | None = None) -> list[dict[str, Any]]:
    cases = materialize_dataset_cases(manifest)
    if split:
        return [c for c in cases if c.get("split") == split]
    return cases


def is_trigger_case(case: dict[str, Any]) -> bool:
    """Trigger/discovery cases belong to the autonomous-trigger runners, whose
    output is a raw_autonomous_trigger_measurement — a different population from
    answer runs. Every grading path (benchmark, grade, judge) must exclude them
    through THIS predicate so the boundary cannot drift per-command."""
    try:
        kind = CaseKind.parse(case.get("kind", "behavior"))
    except ValueError:
        return False
    return kind.population is CasePopulation.TRIGGER


def repo_root_for_manifest(manifest_path: Path) -> Path:
    """A skill installer copies a skill directory verbatim, so a manifest under
    <base>/evals/<skill>/ (not just <base>/evals/) lets eval files sit outside
    the tree that reaches consumers."""
    if manifest_path.name == "shared-benchmark.json":
        evals_dir = manifest_path.parent
        if evals_dir.name != "evals":
            evals_dir = evals_dir.parent
        if evals_dir.name == "evals":
            return evals_dir.parent.resolve()
    return manifest_path.parent.resolve()


def load_manifest_source(path: Path) -> dict[str, Any]:
    """The no-code registry loader (roadmap 3.3): a manifest may be authored in
    YAML (compiled to the JSON manifest shape in memory), and `dataset_files`
    may point at JSONL row files loaded into `datasets`. Everything downstream
    of this loader — validation, leakage lint, prepare, grading — is unchanged."""
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            manifest = yaml.load(
                path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
        except FileNotFoundError:
            die(f"no such file: {path}")
        except yaml.YAMLError as exc:
            die(f"invalid YAML in {path}: {exc}")
        if not isinstance(manifest, dict):
            die(f"{path} must contain a YAML mapping")
    else:
        manifest = load_json(path)
    try:
        reject_nonfinite_numbers(manifest)
    except ValueError as exc:
        die(f"invalid manifest numeric value: {exc}")
    dataset_files = manifest.pop("dataset_files", None)
    if dataset_files is not None:
        if not isinstance(dataset_files, dict):
            die(f"{path}: dataset_files must map dataset ids to JSONL paths")
        datasets = dict(manifest.get("datasets") or {})
        for dataset_id, rel in dataset_files.items():
            rows_path = path.parent / str(rel)
            if not rows_path.is_file():
                die(f"{path}: dataset_files[{dataset_id!r}] does not exist: {rows_path}")
            rows = []
            for n, line in enumerate(rows_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = strict_json_loads(line)
                except json.JSONDecodeError as exc:
                    die(f"{rows_path}: line {n} is not valid JSON: {exc}")
                rows.append(row)
            datasets[str(dataset_id)] = rows
        manifest["datasets"] = datasets
    for case in iter_cases(manifest):
        if is_trigger_case(case) and ("expected_skills" in case or "forbidden_skills" in case):
            skill_paths = manifest.get("skill_paths")
            if not isinstance(skill_paths, list) or not all(isinstance(value, str) for value in skill_paths):
                die("manifest.skill_paths must be a list of strings for scoped trigger cases")
            try:
                parse_skill_constraints(
                    {**case, "should_trigger": expected_trigger_polarity(case) == "TRIGGER"},
                    frozenset(SkillId(value) for value in skill_paths),
                )
            except (TypeError, ValueError) as exc:
                die(f"case {case.get('id')!r}: {exc}")
    return manifest


COMPONENT_CLASSES = {item.value for item in ComponentClass}


SKILL_MECHANISMS = {item.value for item in Mechanism}


MECHANISM_CLASSES = {
    "frontmatter_field": {"discovery", "runtime"},
    "section": {"instructions"}, "list_item": {"instructions"},
    "patch": {"instructions", "discovery", "runtime"},
    "reference": {"resource"}, "script": {"resource"}, "asset": {"resource"},
    "preprocess": {"preprocess"},
}


DISCOVERY_FIELDS = {"name", "description", "when_to_use", "paths", "disable-model-invocation", "user-invocable"}


REQUIRED_FRONTMATTER_FIELDS = ("name", "description")


_COPY_EXCLUDE = {"evals", ".git"}


class AblationError(Exception):
    """Raised when an ablation cannot be validated or materialized."""


def ablation_by_id(manifest: dict[str, Any], aid: str) -> dict[str, Any] | None:
    """The one manifest→ablation lookup (previously five inline `next(...)` copies)."""
    return next((a for a in manifest.get("ablations", []) if a.get("id") == aid), None)


def ablation_components(ablation: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize an ablation entry to a list of components. Returns [] when the
    entry declares no removal — that absence IS the instruction-simulated mode."""
    if ablation.get("components"):
        return list(ablation["components"])
    if ablation.get("mechanism"):
        return [{"mechanism": ablation["mechanism"], "class": ablation.get("class"), "target": ablation.get("target", {})}]
    return []


def component_class(comp: dict[str, Any]) -> str | None:
    """The declared class, or one inferred from the mechanism/field."""
    if comp.get("class"):
        return comp["class"]
    mech, tgt = comp.get("mechanism"), comp.get("target", {})
    if mech == "frontmatter_field":
        return "discovery" if tgt.get("field") in DISCOVERY_FIELDS else "runtime"
    if mech in {"section", "list_item", "patch"}:
        return "instructions"
    if mech in {"reference", "script", "asset"}:
        return "resource"
    if mech == "preprocess":
        return "preprocess"
    return None


def resolve_skill_root(comp: dict[str, Any], skill_paths: list[str]) -> str | None:
    """The ONE default for a component's skill_root: the declared target.skill_root,
    else the first skill path. Used by both materialize() (which records the
    fingerprint) and _expected_component() (which rebuilds the expected fingerprint),
    so the recorded and expected skill_root cannot drift and silently downgrade every
    confirmation to INDETERMINATE."""
    r = (comp.get("target") or {}).get("skill_root")
    return r if r is not None else (skill_paths[0] if skill_paths else None)


def skill_root_key(repo_root: Path, rel: str) -> str:
    """Directory name a skill root is mounted under inside every built tree (the
    canonical with_skill tree, a materialized ablation's pre-edit and edited trees,
    an answer workspace's skills/, and the skills dir every trigger adapter mounts).

    The key is the skill's own directory name: the parent directory of a SKILL.md
    path, or the directory itself when the path names one. The Agent Skills
    specification discovers a skill through a directory named after it, so
    sanitizing the manifest string instead (`skills_good-pr_SKILL.md`, and
    `SKILL.md/SKILL.md` for a per-skill manifest's `SKILL.md`) broke discovery for
    every layout. A root that IS the repo root has no directory segment of its own;
    its key is the SKILL.md frontmatter `name`, never the checkout's basename, so
    the tree hash cannot depend on where the repo was cloned.

    This is the ONE owner of the key. _hash_tree includes this directory name, so
    any second derivation would make canonical_skill_tree_hash != the ablation's
    parent_skill_hash and break TreeIdentity.same_revision_as. skill_root_keys_for
    adds the collision gate on top of it."""
    src = Path(repo_root) / rel
    rel_dir = PurePosixPath(rel) if src.is_dir() else PurePosixPath(rel).parent
    name = rel_dir.name
    if name in {"", "."}:
        skill_md = src / "SKILL.md" if src.is_dir() else src
        if skill_md.name != "SKILL.md" or not skill_md.is_file():
            raise AblationError(
                f"skill root {rel!r} is the repo root itself, so its mount name comes from "
                f"SKILL.md frontmatter, but {skill_md} is not a SKILL.md file")
        value = frontmatter_value(skill_md.read_text(encoding="utf-8", errors="replace"), "name")
        if not isinstance(value, str) or not value.strip():
            raise AblationError(
                f"skill root {rel!r} is the repo root itself, so its mount name comes from "
                f"SKILL.md frontmatter, but {skill_md} has no non-empty name field")
        name = value.strip()
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    if not key or key in {".", ".."}:
        raise AblationError(f"skill root {rel!r} does not yield a usable mount directory name ({key!r})")
    return key


def skill_root_keys_for(repo_root: Path, skill_paths: Sequence[str]) -> list[str]:
    """Mount keys for skill roots, in declaration order, refusing two roots that
    would share a directory in the built tree (the copier would otherwise raise an
    unwrapped FileExistsError mid-materialization)."""
    keys: list[str] = []
    seen: dict[str, str] = {}
    for rel in skill_paths:
        key = skill_root_key(repo_root, rel)
        if key in seen:
            raise AblationError(
                f"skill roots {seen[key]!r} and {rel!r} both mount as {key!r}; each skill "
                "directory name must be unique across skill_paths, so rename one skill directory")
        seen[key] = rel
        keys.append(key)
    return keys


def mounted_skills_by_id(
    root_keys: Mapping[SkillId, str], copied: Sequence[Path],
) -> dict[SkillId, Path]:
    mounted = {(path.parent if path.name == "SKILL.md" else path).name: path for path in copied}
    if len(mounted) != len(copied) or set(mounted) != set(root_keys.values()):
        raise ValueError("mounted skill roots disagree with the declared catalog")
    return {identity: mounted[key] for identity, key in root_keys.items()}


def derived_population(components: list[dict[str, Any]]) -> str:
    """trigger if every component is discovery, else answer. Mixing the two is a
    cohesion error (different case populations, unattributable regression)."""
    classes = {component_class(c) for c in components}
    if "discovery" in classes and classes - {"discovery"}:
        raise AblationError("layer cohesion: discovery components cannot mix with answer-population components")
    return "trigger" if classes == {"discovery"} else "answer"


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_block_including_fences, body)."""
    if text.startswith("---\n"):
        i = text.find("\n---\n", 3)
        if i != -1:
            return text[: i + 5], text[i + 5:]
    return "", text


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the YAML frontmatter mapping with a real YAML parser, so block
    scalars, folded values, and empty values are handled correctly. {} if
    absent or not a mapping."""
    fm, _ = split_frontmatter(text)
    if not fm:
        return {}
    try:
        data = yaml.safe_load(fm[4:-5])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def frontmatter_value(text: str, field: str) -> Any:
    return parse_frontmatter(text).get(field)


def required_fields_present(text: str) -> bool:
    data = parse_frontmatter(text)
    for field in REQUIRED_FRONTMATTER_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _line_starts(text: str) -> tuple[list[str], list[int]]:
    lines = text.splitlines(keepends=True)
    starts, pos = [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln)
    starts.append(pos)
    return lines, starts


def _fenced_mask(lines: list[str]) -> list[bool]:
    """True for lines inside a fenced code block (``` or ~~~), delimiters
    included. Per CommonMark, a fence is closed only by a fence of the SAME
    character that is at least as long as the opener and has nothing but
    whitespace after it — so a ```` (4-tick) block is not closed by ``` (3)."""
    mask: list[bool] = []
    open_fence: tuple[str, int] | None = None
    for ln in lines:
        if open_fence is None:
            m = re.match(r"^\s*(`{3,}|~{3,})", ln)
            mask.append(bool(m))
            if m:
                open_fence = (m.group(1)[0], len(m.group(1)))
        else:
            mask.append(True)
            c = re.match(r"^\s*(`{3,}|~{3,})\s*$", ln)
            if c and c.group(1)[0] == open_fence[0] and len(c.group(1)) >= open_fence[1]:
                open_fence = None
    return mask


def _fenced_char_spans(text: str) -> list[tuple[int, int]]:
    lines, starts = _line_starts(text)
    return [(starts[i], starts[i + 1]) for i, masked in enumerate(_fenced_mask(lines)) if masked]


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def _inline_code_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of inline code: a run of N backticks closed by the next run of
    exactly N backticks (CommonMark). Lets link/reference parsing ignore code
    samples like `` `[x](path)` `` so they are never treated as real links."""
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "`":
            i += 1
            continue
        j = i
        while j < n and text[j] == "`":
            j += 1
        ticks = j - i
        k, closed = j, False
        while k < n:
            if text[k] == "`":
                m = k
                while m < n and text[m] == "`":
                    m += 1
                if m - k == ticks:
                    spans.append((i, m))
                    i, closed = m, True
                    break
                k = m
            else:
                k += 1
        if not closed:
            i = j
    return spans


def frontmatter_field_span(text: str, field: str) -> tuple[int, int] | None:
    """Char span of a top-level frontmatter field line plus any indented
    block-scalar continuation lines."""
    if not text.startswith("---\n"):
        return None
    lines, starts = _line_starts(text)
    field_re = re.compile(rf"^{re.escape(field)}:")
    start_i = None
    for idx in range(1, len(lines)):
        if lines[idx].rstrip("\n") == "---":
            break
        if field_re.match(lines[idx]):
            start_i = idx
            break
    if start_i is None:
        return None
    end_i = start_i + 1
    while end_i < len(lines):
        ln = lines[end_i]
        if ln.rstrip("\n") == "---":
            break
        if ln.strip() == "" or re.match(r"^[ \t]", ln):
            end_i += 1
        else:
            break
    return (starts[start_i], starts[end_i])


def _locate_section(lines: list[str], mask: list[bool], heading: str, *, err_context: str = "") -> tuple[int, int, int]:
    h = heading.strip()
    want_level = (len(h) - len(h.lstrip("#"))) if h.startswith("#") else None
    want = h.lstrip("#").strip().lower()
    start_i: int | None = None
    level: int | None = None
    for i, ln in enumerate(lines):
        if mask[i]:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m and m.group(2).strip().lower() == want and (want_level is None or len(m.group(1)) == want_level):
            start_i, level = i, len(m.group(1))
            break
    if start_i is None or level is None:
        raise AblationError(f"section not found{err_context}: {heading!r}")
    end_i = len(lines)
    for j in range(start_i + 1, len(lines)):
        if mask[j]:
            continue
        m = re.match(r"^(#{1,6})\s+", lines[j])
        if m and len(m.group(1)) <= level:
            end_i = j
            break
    return start_i, end_i, level


def section_span(text: str, heading: str) -> tuple[int, int]:
    """Char span of a markdown section: heading line through the next heading of
    equal-or-higher level. Fence-aware: a '##' inside a ``` block is code."""
    fm, body = split_frontmatter(text)
    base = len(fm)
    lines, starts = _line_starts(body)
    mask = _fenced_mask(lines)
    start_i, end_i, _ = _locate_section(lines, mask, heading)
    return (base + starts[start_i], base + starts[end_i])


def list_item_ops(text: str, section: str, contains: list[str]) -> list[tuple[int, int, str]]:
    fm, body = split_frontmatter(text)
    base = len(fm)
    lines, starts = _line_starts(body)
    mask = _fenced_mask(lines)
    start_i, section_end, _ = _locate_section(lines, mask, section, err_context=" for list_item")
    body_start = start_i + 1
    ops = []
    k = body_start
    while k < section_end:
        if mask[k]:
            k += 1
            continue
        stripped = lines[k].lstrip()
        indent = len(lines[k]) - len(stripped)
        is_bullet = stripped.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s", stripped)
        if is_bullet and any(c.lower() in lines[k].lower() for c in contains):
            j = k + 1
            while j < section_end:
                if mask[j]:
                    open_indent = len(lines[j]) - len(lines[j].lstrip())
                    if open_indent > indent:
                        j += 1
                        while j < section_end and mask[j]:
                            j += 1
                        continue
                    break
                nstripped = lines[j].lstrip()
                nindent = len(lines[j]) - len(nstripped)
                if nstripped == "" or nindent > indent:
                    j += 1
                else:
                    break
            ops.append((base + starts[k], base + starts[j], ""))
            k = j
        else:
            k += 1
    if not ops:
        raise AblationError(f"no matching list items in {section!r}")
    return ops


def preprocess_ops(text: str, contains: list[str]) -> list[tuple[int, int, str]]:
    """Remove inline `` !`command` `` spans and ```! fenced blocks whose command
    text matches any of `contains`. These preprocessing commands execute before
    the skill body reaches the model."""
    def matches(s: str) -> bool:
        return any(c.lower() in s.lower() for c in contains)
    ops: list[tuple[int, int, str]] = []
    for m in re.finditer(r"(?ms)^[ \t]*```!.*?\n[ \t]*`{3,}[ \t]*\n?", text):
        if matches(m.group(0)):
            ops.append((m.start(), m.end(), ""))
    covered = [(s, e) for s, e, _ in ops] + _fenced_char_spans(text) + _inline_code_spans(text)
    for m in re.finditer(r"!`[^`]*`", text):
        if _in_spans(m.start(), covered) or not matches(m.group(0)):
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line_end = line_end + 1 if line_end != -1 else len(text)
        if text[line_start:m.start()].strip() == "" and text[m.end():line_end].strip() == "":
            ops.append((line_start, line_end, ""))
        else:
            ops.append((m.start(), m.end(), ""))
    if not ops:
        raise AblationError(f"no preprocess command matched: {contains!r}")
    return ops


def reference_pointer_ops(text: str, relpath: str) -> list[tuple[int, int, str]]:
    """Unlink markdown links whose target is relpath: [text](relpath) -> text.
    Keeps the existing visible text, so no new prose is introduced (removal,
    not substitution)."""
    spans = _fenced_char_spans(text) + _inline_code_spans(text)
    ops = [(m.start(), m.end(), m.group(1)) for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text)
           if m.group(2).strip() == relpath and not _in_spans(m.start(), spans)]
    if not ops:
        raise AblationError(f"reference pointer not found outside code: {relpath!r}")
    return ops


def patch_delete_ops(text: str, patch: str) -> list[tuple[int, int, str]]:
    """Resolve a deletion-only unified diff to char-span deletions. A '+' line
    means the patch adds content — that is a swap, not an ablation."""
    lines, starts = _line_starts(text)
    ops, idx, saw = [], 0, False
    plines = patch.split("\n")
    k = 0
    while k < len(plines):
        h = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", plines[k])
        if not h:
            k += 1
            continue
        idx = int(h.group(1)) - 1
        k += 1
        while k < len(plines) and plines[k][:1] in (" ", "-", "+"):
            tag, content = plines[k][0], plines[k][1:]
            if tag == "+":
                raise AblationError("ablation patch is deletion-only; '+' lines indicate a swap (use swap:<id>)")
            cur = lines[idx].rstrip("\n") if idx < len(lines) else None
            if cur != content:
                raise AblationError(f"patch context mismatch at line {idx + 1}: {content!r}")
            if tag == "-":
                ops.append((starts[idx], starts[idx + 1], ""))
                saw = True
            idx += 1
            k += 1
    if not saw:
        raise AblationError("patch removed nothing")
    return ops


def _verify_hunks_match_class(text: str, spans: list[tuple[int, int, str]], declared_class: str) -> None:
    fm, _ = split_frontmatter(text)
    fm_end = len(fm)
    in_fm = any(e <= fm_end for s, e, _ in spans)
    in_body = any(s >= fm_end for s, e, _ in spans)
    if in_fm and in_body:
        raise AblationError("patch deletes from both the frontmatter and the body; split it into separate frontmatter and instructions patch components")
    if declared_class in ("discovery", "runtime"):
        if in_body:
            raise AblationError(f"patch declares class {declared_class!r} (a frontmatter edit) but a hunk deletes body content")
        field_spans = []
        for name in parse_frontmatter(text):
            fsp = frontmatter_field_span(text, str(name))
            if fsp is not None:
                field_spans.append((str(name), fsp[0], fsp[1]))
        for s, e, _ in spans:
            owner = next((nm for nm, fs, fe in field_spans if fs <= s and e <= fe), None)
            if owner is None:
                raise AblationError("patch deletes frontmatter content outside any field (a fence or blank line); patch a specific field instead")
            owner_is_discovery = owner in DISCOVERY_FIELDS
            if declared_class == "discovery" and not owner_is_discovery:
                raise AblationError(f"discovery patch deletes non-discovery field {owner!r}; use class 'runtime' for runtime fields or split the patch")
            if declared_class == "runtime" and owner_is_discovery:
                raise AblationError(f"runtime patch deletes discovery frontmatter field {owner!r}; use class 'discovery' for discovery fields")
    elif in_fm:
        raise AblationError(f"patch declares class {declared_class!r} but a hunk deletes frontmatter content")


def _check_disjoint(ops: list[tuple[int, int, str]]) -> None:
    spans = sorted((s, e) for s, e, _ in ops)
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            raise AblationError(f"components overlap near char {spans[i][0]}")


def _apply_edits(text: str, ops: list[tuple[int, int, str]]) -> str:
    for s, e, r in sorted(ops, key=lambda o: o[0], reverse=True):
        text = text[:s] + r + text[e:]
    return text


def _detect_newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def _write_text_preserving_newlines(path: Path, text_lf: str) -> None:
    nl = _detect_newline(path.read_bytes()) if path.exists() else "\n"
    out = text_lf.replace("\n", nl) if nl != "\n" else text_lf
    path.write_bytes(out.encode("utf-8"))


def skill_tree_hash(root: Path) -> str:
    """Hash an already-built skill tree at the attestation boundary.

    Producers must identify the exact immutable snapshot they mount. Rebuilding
    mutable sources to derive the hash would let the report name bytes no model
    actually observed.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"skill tree is not a directory: {root}")
    return directory_tree_sha256(root)


def _safe_under(base: Path, path: Path) -> Path:
    base_r = base.resolve()
    p = path.resolve()
    if p != base_r and base_r not in p.parents:
        raise AblationError(f"path escapes {base_r}: {path}")
    return p


def _reject_output_root_overlap(out_root: Path, repo_root: Path, manifest: dict[str, Any]) -> None:
    out = out_root.resolve()
    for r in manifest.get("skill_paths", []):
        src = (repo_root / r).resolve()
        src_dir = src if src.is_dir() else src.parent
        if out == src_dir:
            raise AblationError(f"output dir {out} is a source skill root; choose a directory outside the skill")
        if src_dir in out.parents:
            raise AblationError(f"output dir {out} is inside source skill root {src_dir}; choose a directory outside the skill")
        if out in src_dir.parents:
            raise AblationError(f"output dir {out} contains source skill root {src_dir}; choose a directory outside the skill tree")


def _reject_overlapping_skill_roots(repo_root: Path, manifest: dict[str, Any]) -> None:
    dirs: list[tuple[str, Path]] = []
    for r in manifest.get("skill_paths", []):
        src = (repo_root / r).resolve()
        dirs.append((r, src if src.is_dir() else src.parent))
    for i, (ri, di) in enumerate(dirs):
        for j, (rj, dj) in enumerate(dirs):
            if i == j:
                continue
            if di == dj:
                raise AblationError(f"skill roots {ri!r} and {rj!r} are copied from the same directory {di}; the ablated copy and an unablated copy would coexist — declare a single root")
            if di in dj.parents:
                raise AblationError(f"skill root {ri!r} (dir {di}) is an ancestor of skill root {rj!r}; copying it would include an unablated duplicate of {rj!r} — declare non-overlapping roots")
    skill_root_keys_for(repo_root, manifest.get("skill_paths", []))


def _copy_skill_root(src_dir: Path, dst_dir: Path) -> None:
    src_real = src_dir.resolve()
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in _COPY_EXCLUDE and not d.startswith(".")]
        for name in [*dirs, *files]:
            p = Path(root) / name
            if p.is_symlink():
                target = p.resolve()
                if target != src_real and src_real not in target.parents:
                    raise AblationError(f"skill root contains a symlink escaping the root: {p}")
    def ignore(_dir: str, names: list[str]) -> list[str]:
        return [n for n in names if n in _COPY_EXCLUDE or n.startswith(".")]
    shutil.copytree(src_dir, dst_dir, ignore=ignore)


def _required_target_keys(mech: str) -> list[str]:
    return {
        "frontmatter_field": ["field"], "section": ["heading"],
        "list_item": ["section"], "patch": ["patch"], "reference": ["path"],
        "script": ["path"], "asset": ["path"], "preprocess": ["contains"],
    }.get(mech, [])


def validate_ablation_removal(ablation: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Shape + safety validation for a declared removal. Apply-time gates
    (effect, disjointness, required-field preservation) run at materialize time."""
    if ablation.get("components") and ablation.get("mechanism"):
        raise AblationError("declare either mechanism+target or components, not both")
    comps = ablation_components(ablation)
    if not comps:
        return
    skill_paths = manifest.get("skill_paths", [])
    classes: set[str | None] = set()
    for comp in comps:
        mech = comp.get("mechanism")
        if mech not in SKILL_MECHANISMS:
            raise AblationError(f"unknown mechanism {mech!r}")
        cls = component_class(comp)
        if cls not in COMPONENT_CLASSES:
            raise AblationError(f"invalid component class {cls!r}")
        classes.add(cls)
        tgt = comp.get("target", {})
        if not isinstance(tgt, dict):
            raise AblationError("target must be an object")
        root = tgt.get("skill_root")
        if root is None and len(skill_paths) != 1:
            raise AblationError(f"component target missing skill_root (manifest has {len(skill_paths)} skill_paths)")
        if root is not None and root not in skill_paths:
            raise AblationError(f"skill_root {root!r} is not in manifest.skill_paths")
        for key in _required_target_keys(mech):
            if not tgt.get(key):
                raise AblationError(f"{mech} target missing {key!r}")
        for key in ("path", "patch"):
            v = tgt.get(key)
            if v and (Path(v).is_absolute() or ".." in Path(v).parts):
                raise AblationError(f"unsafe path (absolute or traversal): {v!r}")
        if comp.get("class") and comp["class"] not in MECHANISM_CLASSES.get(mech, COMPONENT_CLASSES):
            raise AblationError(f"mechanism {mech!r} is incompatible with declared class {comp['class']!r} (allowed: {sorted(MECHANISM_CLASSES.get(mech, []))})")
        if mech == "frontmatter_field":
            inferred = component_class({**comp, "class": None})
            if comp.get("class") and comp["class"] != inferred:
                raise AblationError(f"frontmatter_field {tgt.get('field')!r} is class {inferred}, not {comp['class']!r}")
        if mech in ("reference", "script", "asset") and Path(str(tgt.get("path", ""))).name == "SKILL.md":
            raise AblationError(f"{mech} may not target the skill's SKILL.md (use a frontmatter/section/patch mechanism)")
    if "discovery" in classes and classes - {"discovery"}:
        raise AblationError("layer cohesion: discovery cannot mix with answer-population components")


def _resolve_component_ops(comp: dict[str, Any], main_file: Path, root_dir: Path, repo_root: Path, aid: str) -> tuple[dict[Path, list[tuple[int, int, str]]], set[Path]]:
    mech = comp.get("mechanism")
    tgt = comp.get("target", {})
    text = main_file.read_text(encoding="utf-8-sig")
    ops: dict[Path, list[tuple[int, int, str]]] = {}
    deletes: set[Path] = set()
    if mech == "frontmatter_field":
        span = frontmatter_field_span(text, tgt["field"])
        if span is None:
            raise AblationError(f"frontmatter field not found: {tgt['field']!r}")
        ops[main_file] = [(span[0], span[1], "")]
    elif mech == "section":
        s, e = section_span(text, tgt["heading"])
        ops[main_file] = [(s, e, "")]
    elif mech == "list_item":
        ops[main_file] = list_item_ops(text, tgt["section"], tgt.get("contains", []))
    elif mech == "preprocess":
        ops[main_file] = preprocess_ops(text, tgt["contains"])
    elif mech == "patch":
        patch_file = _safe_under(repo_root, repo_root / tgt["patch"])
        spans = patch_delete_ops(text, patch_file.read_text(encoding="utf-8"))
        declared_class = component_class(comp)
        if declared_class is None:
            raise AblationError("patch component has no resolvable class")
        _verify_hunks_match_class(text, spans, declared_class)
        ops[main_file] = spans
    elif mech == "reference":
        mode = tgt.get("remove", "both")
        if mode in ("pointer", "both"):
            ops[main_file] = reference_pointer_ops(text, tgt["path"])
        if mode in ("content", "both"):
            deletes.add(_safe_under(root_dir, root_dir / tgt["path"]))
    elif mech in ("script", "asset"):
        deletes.add(_safe_under(root_dir, root_dir / tgt["path"]))
    else:
        raise AblationError(f"unknown mechanism {mech!r}")
    return ops, deletes


@_dataclass(frozen=True)
class ValidatedAblation:
    """The gate pile as a SMART CONSTRUCTOR. The only way to obtain one is
    ValidatedAblation.validate(), which runs every declaration-time gate
    (removal declared, mechanism/class/field consistency, path safety,
    non-overlapping skill roots, layer cohesion). Its existence is therefore proof
    the ablation is well-formed, so materialize() can take a ValidatedAblation
    instead of a raw dict — you cannot materialize an unvalidated ablation."""

    repo_root: Path
    manifest: dict[str, Any]
    ablation: dict[str, Any]
    components: tuple[dict[str, Any], ...]
    population: CasePopulation

    @classmethod
    def validate(cls, repo_root: Path, manifest: dict[str, Any], ablation: dict[str, Any]) -> ValidatedAblation:
        comps = ablation_components(ablation)
        if not comps:
            raise AblationError(f"ablation {ablation.get('id')!r} declares no removal (instruction-simulated)")
        validate_ablation_removal(ablation, manifest)
        _reject_overlapping_skill_roots(repo_root, manifest)
        population = CasePopulation(derived_population(comps))
        return cls(repo_root=repo_root, manifest=manifest, ablation=ablation, components=tuple(comps), population=population)


def materialize_ablation(repo_root: Path, manifest: dict[str, Any], ablation: dict[str, Any], out_root: Path) -> dict[str, Any]:
    """Backward-compatible dict facade over the typed core: validate, materialize,
    serialize. New code should use ValidatedAblation.validate() + materialize()."""
    return materialize(ValidatedAblation.validate(repo_root, manifest, ablation), out_root).as_legacy_dict()


def materialize_trigger_arm(repo_root: Path, manifest: dict[str, Any], ablation_id: str, out_root: Path) -> MaterializedArm:
    """Materialize a discovery/trigger-population ablation for autonomous trigger
    runners. This is the shared gate for Pi trigger and the trigger matrix, so
    they cannot diverge on which ablations are valid to mount."""
    ablation = ablation_by_id(manifest, ablation_id)
    if ablation is None:
        raise AblationError(f"unknown ablation: {ablation_id}")
    components = ablation_components(ablation)
    if not components:
        raise AblationError(f"ablation {ablation_id} is instruction-simulated; trigger ablations must declare a materialized removal")
    if derived_population(components) != "trigger":
        raise AblationError(f"ablation {ablation_id} is an answer-population ablation; trigger ablations must target discovery/trigger behavior")
    return materialize(ValidatedAblation.validate(repo_root, manifest, ablation), out_root)


def materialize_trigger_ablation(repo_root: Path, manifest: dict[str, Any], ablation_id: str, out_root: Path) -> dict[str, Any]:
    """Return the historical trigger materialization dictionary."""
    return materialize_trigger_arm(repo_root, manifest, ablation_id, out_root).as_legacy_dict()


def canonical_trigger_query(query: str) -> str:
    """Conservative inference identity for authored trigger prompts.

    Raw text is still executed and persisted, but Unicode compatibility forms,
    case-only variants, and whitespace-only edits are one experimental unit.
    Treating those cosmetic aliases as independent samples would manufacture
    replication without adding evidence.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("trigger query must be non-empty text")
    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = "".join(
        char for char in normalized
        if (unicodedata.category(char) != "Cf"
            and not 0x180B <= ord(char) <= 0x180F
            and not 0xFE00 <= ord(char) <= 0xFE0F
            and not 0xE0100 <= ord(char) <= 0xE01EF)
    )
    return " ".join(normalized.split())


def expected_provenance_for_ablation(
    manifest: dict[str, Any], ablation_id: str,
) -> ExpectedProvenance:
    """Authoritative treatment identity derived from one manifest declaration."""
    ablation = ablation_by_id(manifest, ablation_id)
    if ablation is None:
        raise ValueError(f"manifest does not declare ablation {ablation_id!r}")
    components = ablation_components(ablation)
    if not components:
        raise ValueError(f"ablation {ablation_id!r} has no materialized components")
    raw_skill_paths = manifest.get("skill_paths", [])
    if (not isinstance(raw_skill_paths, list)
            or not all(isinstance(path, str) for path in raw_skill_paths)):
        raise ValueError("manifest skill_paths must be a list of strings")
    skill_paths = [path for path in raw_skill_paths if isinstance(path, str)]
    return ExpectedProvenance(
        id=ablation_id,
        mode=(AblationMode.INVALID_SKILL if ablation.get("invalid_skill")
              else AblationMode.MATERIALIZED),
        population=CasePopulation(derived_population(components)),
        components=tuple(
            _expected_component(component, skill_paths)
            for component in components),
    )


def trigger_manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Self-verifying manifest treatment snapshot for trigger reports.

    The full loaded-manifest digest binds cases and dataset expansion; embedded
    expected-provenance records let an offline comparer verify that a runner's
    recorded edit is the edit the manifest declared.
    """
    trigger_ablations = []
    for ablation in manifest.get("ablations", []):
        if not isinstance(ablation, dict) or not isinstance(ablation.get("id"), str):
            continue
        components = ablation_components(ablation)
        if not components or derived_population(components) != "trigger":
            continue
        expected = expected_provenance_for_ablation(manifest, ablation["id"])
        trigger_ablations.append({
            "id": expected.id,
            "mode": expected.mode.value,
            "population": expected.population.value,
            "components": [component.fingerprint() for component in expected.components],
        })
    payload = {
        "schema_version": 1,
        "manifest_sha256": canonical_json_sha256(manifest),
        "skill_name": str(manifest.get("skill_name") or "skill-under-test"),
        "skill_paths": list(manifest.get("skill_paths", [])),
        "trigger_ablations": sorted(trigger_ablations, key=lambda item: item["id"]),
    }
    return {**payload, "identity_sha256": canonical_json_sha256(payload)}


def expected_provenance_from_trigger_identity(
    identity: dict[str, Any], ablation_id: str,
) -> ExpectedProvenance:
    """Parse one expected treatment from a validated trigger identity block."""
    entries = identity.get("trigger_ablations") if isinstance(identity, dict) else None
    if not isinstance(entries, list):
        raise TypeError("trigger manifest identity has no trigger_ablations list")
    matches = [entry for entry in entries
               if isinstance(entry, dict) and entry.get("id") == ablation_id]
    if len(matches) != 1:
        raise ValueError(
            f"trigger manifest identity must declare ablation {ablation_id!r} exactly once")
    entry = matches[0]
    components = entry.get("components")
    if not isinstance(components, list):
        raise TypeError("trigger manifest identity components must be a list")
    return ExpectedProvenance(
        id=entry.get("id"), mode=entry.get("mode"),
        population=entry.get("population"),
        components=tuple(Component.from_dict(component) for component in components),
    )


def materialize(validated: ValidatedAblation, out_root: Path) -> MaterializedArm:
    """Produce out_root/<id>/ holding the altered skill tree for a VALIDATED
    ablation, and return a MaterializedArm (which itself cannot exist without an
    edited tree + provenance). Runs the apply-time gates (output containment,
    net-deletion, disjointness, required-field). Raises AblationError on any gate."""
    repo_root, manifest, ablation = validated.repo_root, validated.manifest, validated.ablation
    comps = list(validated.components)
    population = validated.population
    _reject_output_root_overlap(out_root, repo_root, manifest)
    aid = ablation["id"]
    skill_paths = manifest.get("skill_paths", [])

    def root_for(comp: dict[str, Any]) -> str:
        root = resolve_skill_root(comp, skill_paths)
        if root is None:
            raise AblationError("ablation component has no resolvable skill_root")
        return root

    dest = out_root / aid
    if dest.exists():
        raise AblationError(f"output already exists: {dest}")
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".ablation-{aid}-", dir=out_root))
    try:
        roots: dict[str, tuple[Path, Path]] = {}
        rels = list(skill_paths or dict.fromkeys(root_for(c) for c in comps))
        for r, key in zip(rels, skill_root_keys_for(repo_root, rels), strict=True):
            src = _safe_under(repo_root, repo_root / r)
            src_dir = src if src.is_dir() else src.parent
            dst_dir = tmp / key
            _copy_skill_root(src_dir, dst_dir)
            main = dst_dir / "SKILL.md" if (src.is_dir() or src.name == "SKILL.md") else dst_dir / src.name
            roots[r] = (main, dst_dir)

        parent_skill_hash = skill_tree_hash(tmp)

        file_text: dict[Path, str] = {}
        file_ops: dict[Path, list[tuple[int, int, str]]] = {}
        delete_owner: dict[Path, int] = {}
        removed_by_component: list[int] = []
        isolation_warnings: list[str] = []
        for ci, comp in enumerate(comps):
            main, rdir = roots[root_for(comp)]
            ops, deletes = _resolve_component_ops(comp, main, rdir, repo_root, aid)
            removed = 0
            for f, edits in ops.items():
                file_text.setdefault(f, f.read_text(encoding="utf-8-sig"))
                file_ops.setdefault(f, []).extend(edits)
                edit_removed = sum((e - s) - len(rep) for s, e, rep in edits)
                removed += edit_removed
                orig_len = len(file_text[f])
                if orig_len and edit_removed / orig_len > 0.6:
                    isolation_warnings.append(f"component #{ci} ({comp.get('mechanism')}) removed {edit_removed}/{orig_len} bytes ({edit_removed / orig_len:.0%}) of {f.name} — large for one declared component")
            for d in deletes:
                if not d.exists():
                    raise AblationError(f"component #{ci}: file to remove not found: {d}")
                if d in delete_owner:
                    raise AblationError(f"components #{delete_owner[d]} and #{ci} both delete {d.name} (overlap)")
                delete_owner[d] = ci
                removed += d.stat().st_size
            if removed <= 0:
                raise AblationError(f"component #{ci} ({comp.get('mechanism')}) removed nothing (net-deletion gate)")
            removed_by_component.append(removed)

        for d in delete_owner:
            if d in file_ops:
                raise AblationError(f"{d.name} is both edited and deleted by the ablation (overlap)")
        for f, edits in file_ops.items():
            _check_disjoint(edits)
            _write_text_preserving_newlines(f, _apply_edits(file_text[f], edits))
        for d in delete_owner:
            d.unlink()

        if not ablation.get("invalid_skill"):
            for main, _ in roots.values():
                if not (main.exists() and main.is_file()):
                    raise AblationError(f'ablation removed the skill main file {main.name!r}; set "invalid_skill": true to run that as an invalid-skill experiment')
                if not required_fields_present(main.read_text(encoding="utf-8-sig")):
                    raise AblationError('required frontmatter field (name/description) became empty or missing; set "invalid_skill": true to run that as an invalid-skill experiment')

        skill_hash = skill_tree_hash(tmp)
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    for w in isolation_warnings:
        print(f"WARN ablation {aid}: {w}", file=sys.stderr)
    prov = Provenance(
        id=aid,
        mode=(AblationMode.INVALID_SKILL if ablation.get("invalid_skill")
              else AblationMode.MATERIALIZED),
        population=population,
        identity=TreeIdentity(canonical=parent_skill_hash, edited=skill_hash),
        components=tuple(
            Component(cls=ComponentClass(component_class(c)),
                      mechanism=Mechanism(c.get("mechanism")),
                      skill_root=root_for(c), target=c.get("target", {}),
                      removed_bytes=removed_by_component[i])
            for i, c in enumerate(comps)
        ),
    )
    arm = Arm(variant_truth=f"ablation:{aid}", blind=True, identity=prov.identity, provenance=prov)
    return MaterializedArm(
        arm=arm,
        dir=str(dest),
        skill_files={r: str(dest / main.relative_to(tmp)) for r, (main, _) in roots.items()},
        isolation_warnings=tuple(isolation_warnings),
    )


def materialized_tree_for_variant(repo_root: Path, manifest: dict[str, Any], variant: str, out_root: Path) -> dict[str, Any] | None:
    """For an ablation:<id> variant that declares a removal, materialize the tree
    and return its provenance (with skill_files). Returns None for non-ablation
    variants and for instruction-simulated ablations (no removal declared)."""
    aid = ablation_id_of(variant)
    if aid is None:
        return None
    ablation = ablation_by_id(manifest, aid)
    if ablation is None:
        raise AblationError(f"unknown ablation variant: {variant}")
    if not ablation_components(ablation):
        return None
    return materialize_ablation(repo_root, manifest, ablation, out_root)


def build_canonical_skill_tree(repo_root: Path, manifest: dict[str, Any], dest_dir: Path) -> Path:
    """Copy every manifest skill root into dest_dir/<key> with no edits — the
    canonical surface for with_skill, so it matches a materialized ablation arm
    file-for-file (the only difference being the ablation's declared edit)."""
    dest_dir = Path(dest_dir)
    _reject_overlapping_skill_roots(repo_root, manifest)
    _reject_output_root_overlap(dest_dir, repo_root, manifest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rels = list(manifest.get("skill_paths", []))
    for r, key in zip(rels, skill_root_keys_for(repo_root, rels), strict=True):
        src = _safe_under(repo_root, repo_root / r)
        src_dir = src if src.is_dir() else src.parent
        _copy_skill_root(src_dir, dest_dir / key)
    return dest_dir


def canonical_skill_tree_hash(repo_root: Path, manifest: dict[str, Any]) -> str:
    """Hash of the canonical (unedited) skill tree — the with_skill oracle. Built by
    the same copier and key-naming as materialize_ablation's pre-edit tree, so it
    equals a materialized ablation's parent_skill_hash. Both arms record it so the
    report can prove they were derived from the same skill revision."""
    tmp = Path(tempfile.mkdtemp(prefix=".canon-hash-"))
    try:
        build_canonical_skill_tree(repo_root, manifest, tmp / "tree")
        return skill_tree_hash(tmp / "tree")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mount_skill_tree(tree_dir: Path, skills_dir: Path) -> list[Path]:
    """Copy each per-root subdir of a canonical/materialized skill tree into an
    agent's skills dir. EVERY trigger arm (baseline and ablation, every adapter)
    mounts through here, so all arms expose an identical file surface under
    identical names — the only difference is the bytes a declared ablation edit
    removed. Returns the copied SKILL.md (or root dir) paths, which double as
    the skill-load detection needles for detect_trigger."""
    skills_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for root_dir in sorted(p for p in tree_dir.iterdir() if p.is_dir()):
        dest = skills_dir / root_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(root_dir, dest)
        copied.append(dest / "SKILL.md" if (dest / "SKILL.md").exists() else dest)
    return copied


def _expected_component(comp: dict[str, Any], skill_paths: list[str]) -> Component:
    root = resolve_skill_root(comp, skill_paths)
    if root is None:
        raise ValueError("ablation component has no resolvable skill_root")
    return Component(cls=ComponentClass(comp.get("class") or component_class(comp)),
                     mechanism=Mechanism(comp.get("mechanism")),
                     skill_root=root, target=comp.get("target", {}))


def expected_trigger_polarity(case: dict[str, Any]) -> str:
    """Resolve discovery polarity only from the validated explicit boolean."""
    value = case.get("should_trigger")
    if not isinstance(value, bool):
        raise TypeError("trigger case requires an explicit boolean should_trigger")
    return "TRIGGER" if value else "NO_TRIGGER"


UniqueKeySafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_yaml_mapping)
