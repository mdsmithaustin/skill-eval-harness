from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from content_digests import canonical_json_sha256, directory_tree_sha256
from json_contracts import strict_json_loads, write_json
from skill_inputs import (
    _copy_skill_root,
    parse_frontmatter,
    repo_root_for_manifest,
    skill_root_key,
)
from trigger_contracts import SkillId, parse_skill_constraints

DIGEST_SCHEME = "skill-tree-sha256-v1"
SCAN_VERSION = 1
_RUN_MARKER = ".skill-overlap-run"


@dataclass(frozen=True)
class SkillSnapshot:
    name: str
    mount_name: str
    digest_scheme: str
    digest: str
    description: str


def _snapshot_skill(source: Path, destination: Path) -> SkillSnapshot:
    _copy_skill_root(source.parent, destination)
    metadata = parse_frontmatter((destination / "SKILL.md").read_text(encoding="utf-8"))
    name, description = metadata.get("name"), metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{source}: SKILL.md requires a skill name")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{source}: SKILL.md requires a description to rank")
    return SkillSnapshot(name, skill_root_key(source.parent.parent, source.parent.name),
                         DIGEST_SCHEME, directory_tree_sha256(destination), description)


def _find_skills(collection: Path) -> list[Path]:
    if not collection.is_dir():
        raise ValueError(f"expected a skill collection directory: {collection}")
    roots: set[Path] = set()
    errors: list[OSError] = []
    for base, dirs, files in os.walk(collection, onerror=errors.append):
        root = Path(base)
        if (root / _RUN_MARKER).is_file():
            dirs.clear()
            continue
        if "SKILL.md" in files:
            roots.add(root / "SKILL.md")
            dirs.clear()
            continue
        dirs[:] = sorted(directory for directory in dirs if not directory.startswith("."))
        for directory in dirs:
            candidate = root / directory
            if candidate.is_symlink() and (candidate / "SKILL.md").is_file():
                roots.add(candidate.resolve() / "SKILL.md")
    if errors:
        raise errors[0]
    return sorted(roots)


def _description_vectors(descriptions: list[str], command: str, timeout: int) -> list[list[float]]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("--embed-cmd must name an embedding executable")
    try:
        process = subprocess.run(argv, input=json.dumps({"texts": descriptions}), text=True,
                                 encoding="utf-8", capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"embedding command timed out after {timeout}s") from exc
    if process.returncode:
        raise ValueError(f"embedding command exited {process.returncode}: {process.stderr[:500]}")
    response = strict_json_loads(process.stdout)
    rows = response.get("embeddings") if isinstance(response, dict) else None
    if not isinstance(rows, list) or len(rows) != len(descriptions):
        raise ValueError("embedding command must return one vector per description in 'embeddings'")
    vectors: list[list[float]] = []
    for row in rows:
        if (not isinstance(row, list) or not row
                or any(type(value) not in {float, int} for value in row)
                or (vectors and len(row) != len(vectors[0]))):
            raise ValueError("embedding vectors must have equal, positive dimensions and numeric values")
        try:
            values = [float(value) for value in row]
        except OverflowError as exc:
            raise ValueError("embedding vectors must be finite") from exc
        scale = max(abs(value) for value in values)
        if not scale or not all(math.isfinite(value) for value in values):
            raise ValueError("embedding vectors must be finite and nonzero")
        scaled = [value / scale for value in values]
        norm = math.hypot(*scaled)
        vectors.append([value / norm for value in scaled])
    return vectors


def scan_overlap(collection: Path, *, command: str, model: str, timeout: int,
                 out: Path | None) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("--embedding-model must name the model used by the encoder")
    sources = _find_skills(collection)
    if out is not None and any(source.parent in out.resolve().parents for source in sources):
        raise ValueError("--out must be outside the scanned skill roots")
    skills = {}
    for source in sources:
        with tempfile.TemporaryDirectory(prefix="overlap-scan-") as temporary:
            skills[str(source)] = asdict(_snapshot_skill(source, Path(temporary) / "skill"))
    vectors = (_description_vectors([skill["description"] for skill in skills.values()], command, timeout)
               if len(skills) >= 2 else [])
    scored = []
    for (a, va), (b, vb) in combinations(zip(skills, vectors), 2):
        cosine = math.fsum(x * y for x, y in zip(va, vb, strict=True))
        scored.append((max(-1.0, min(1.0, cosine)), a, b))
    scores = sorted(scored, key=lambda row: (-row[0], row[1], row[2]))
    scan = {
        "schema_version": SCAN_VERSION,
        "method": {"kind": "embedding_cosine", "source_field": "description", "model": model},
        "skills": skills,
        "pairs": [{"rank": rank, "score": score, "skill_paths": [a, b]}
                  for rank, (score, a, b) in enumerate(scores, 1)],
    }
    return {**scan, "scan_sha256": canonical_json_sha256(scan)}


def _selected_pair(scan_path: Path, rank: int) -> tuple[dict[str, Any], dict[str, Any]]:
    scan = strict_json_loads(scan_path.read_text(encoding="utf-8"))
    if not isinstance(scan, dict) or scan.get("schema_version") != SCAN_VERSION:
        raise ValueError("expected a version 1 overlap scan")
    digest = scan.pop("scan_sha256", None)
    if digest != canonical_json_sha256(scan):
        raise ValueError("overlap scan digest does not match the saved report")
    pairs, skills = scan.get("pairs"), scan.get("skills")
    if not isinstance(pairs, list) or not isinstance(skills, dict):
        raise TypeError("overlap scan must contain pairs and a skill inventory")
    if not 1 <= rank <= len(pairs):
        raise ValueError(f"pair rank {rank} is outside the scan's {len(pairs)} pairs")
    pair = pairs[rank - 1]
    paths = pair.get("skill_paths") if isinstance(pair, dict) else None
    if (not isinstance(paths, list) or len(paths) != 2 or paths[0] == paths[1]
            or any(not isinstance(path, str) or path not in skills for path in paths)
            or pair.get("rank") != rank):
        raise ValueError(f"overlap scan has invalid skill identities for rank {rank}")
    return {**scan, "scan_sha256": digest}, pair


def _reviewed_queries(manifest_path: Path, split: str | None,
                      sources: list[Path]) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, str]]]:
    from run_trigger_matrix import (
        cases_from_manifest,
        load_manifest,
        validate_trigger_rows,
    )

    manifest = load_manifest(manifest_path)
    root = repo_root_for_manifest(manifest_path)
    declared = frozenset(SkillId(path) for path in manifest.get("skill_paths", []))
    ids: dict[str, str] = {}
    for label, source in zip(("a", "b"), sources, strict=True):
        matches = [identity for identity in declared
                   if (root / identity).resolve() in {source, source.parent}]
        if len(matches) != 1:
            raise ValueError(f"reviewed manifest must name {source} once in skill_paths")
        ids[label] = matches[0]
    rows = validate_trigger_rows(cases_from_manifest(manifest, split), str(manifest_path), declared)
    selected_rows = []
    excluded = []
    for row in rows:
        constraints = parse_skill_constraints(row)
        if constraints is None or not constraints.selected <= set(ids.values()):
            excluded.append({"query_id": row["query_id"], "reason": (
                "unscoped" if constraints is None else "references_other_skills")})
        else:
            selected_rows.append(row)
    if not selected_rows:
        raise ValueError("no reviewed trigger queries scoped to the selected pair")
    return ids, selected_rows, excluded


def _activation_rows(report: dict[str, Any], ids: dict[str, str]) -> list[dict[str, Any]]:
    observations = []
    for row in report["results"]:
        measured = row["measurement_status"] == "complete"
        activated = dict.fromkeys(row.get("skill_detections", {}), False)
        activated.update(dict.fromkeys(row.get("activated_skills", []), True))
        unscoped = row.get("observation_metadata", {}).get("unscoped_skill_detections", {})
        activated.update({identity: bool(evidence) for identity, evidence in unscoped.items()})
        observations.append({
            "query_id": row["query_id"], "run_number": row["run_number"],
            "measurement_status": row["measurement_status"], "pass": row.get("pass"),
            "activated": {label: activated.get(identity) if measured else None
                          for label, identity in ids.items()},
        })
    return observations


def evaluate_overlap(scan_path: Path, *, rank: int, manifest: Path, runs: Path,
                     agent: str, model: str, split: str | None, repetitions: int,
                     timeout: int, options: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    from run_trigger_matrix import MatrixRunStopped, matrix_exit_code, run_matrix
    from skill_interaction import run_interaction

    scan, pair = _selected_pair(scan_path, rank)
    sources = [Path(path) for path in pair["skill_paths"]]
    candidates = [scan["skills"][str(source)] for source in sources]
    if any(runs == source.parent or source.parent in runs.parents for source in sources):
        raise ValueError("--runs must be outside the selected skill roots")
    keys = [skill_root_key(source.parent.parent, source.parent.name) for source in sources]
    if len(set(keys)) != 2 or candidates[0]["name"].strip() == candidates[1]["name"].strip():
        raise ValueError("selected skills need distinct mount names and skill names")
    reviewed_ids, queries, excluded = _reviewed_queries(manifest, split, sources)
    runs.mkdir(parents=True, exist_ok=False)
    (runs / _RUN_MARKER).touch(exist_ok=False)
    stage = runs / "skills"
    stage.mkdir()
    identities = {}
    for label, source, key, candidate in zip(("a", "b"), sources, keys, candidates, strict=True):
        staged = asdict(_snapshot_skill(source, stage / key))
        if staged != candidate:
            raise ValueError(f"{source}: skill changed since scan (expected {candidate['digest']}, "
                             f"staged {staged['digest']}); rescan the collection")
        identities[label] = {field: staged[field] for field in ("name", "mount_name", "digest_scheme", "digest")}
    tree_hash = directory_tree_sha256(stage)
    staged_ids = {label: f"skills/{key}/SKILL.md" for label, key in zip(("a", "b"), keys, strict=True)}
    reviewed_to_staged = {reviewed_ids[label]: staged_id for label, staged_id in staged_ids.items()}
    remapped_queries = [{**query, **{
        field: [reviewed_to_staged[identity] for identity in query[field]]
        for field in ("expected_skills", "forbidden_skills") if field in query
    }} for query in queries]
    catalog = runs / "trigger-catalog.json"
    write_json(catalog, {"version": 1, "skill_name": "overlap", "skill_paths": list(staged_ids.values()),
                         "variants": ["with_skill", "without_skill"], "cases": []})
    write_json(runs / "trigger-queries.json", remapped_queries)

    interaction_path = runs / "interaction.json"
    outcome_code = run_interaction(
        manifest=manifest, runs=runs / "interaction",
        sources=(str(stage / keys[0]), str(stage / keys[1])), model=model, agent=agent,
        split=split, repetitions=repetitions, timeout=timeout, out=str(interaction_path),
        report_only=False, options=options,
    )
    outcome = strict_json_loads(interaction_path.read_text(encoding="utf-8"))
    if outcome["skills"] != identities:
        raise ValueError("interaction staged different skill revisions; see interaction.json")
    trigger_path = runs / "trigger-matrix.json"
    try:
        trigger = run_matrix(catalog, remapped_queries, agents=[agent], models=[model],
                             runs_per_query=repetitions, timeout=timeout, workers=1,
                             backend_options=options, collect_unscoped=True,
                             expected_skill_tree_hash=tree_hash)
    except MatrixRunStopped as stopped:
        write_json(trigger_path, stopped.report)
        raise stopped.cause
    write_json(trigger_path, trigger)
    report = {
        "schema_version": 1, "scan_sha256": scan["scan_sha256"], "candidate": pair,
        "method": scan["method"],
        "agent": agent, "model": model, "split": split,
        "skills": {label: {**identity, "reviewed_id": reviewed_ids[label], "trigger_id": staged_ids[label]}
                   for label, identity in identities.items()},
        "trigger": {
            "report": trigger_path.name, "data": "trigger-queries.json", "skill_tree_sha256": tree_hash,
            "summary": trigger["summary"], "observations": _activation_rows(trigger, staged_ids),
            "excluded_queries": excluded,
        },
        "outcomes": {"report": interaction_path.name, **{
            key: outcome[key] for key in ("plan_sha256", "dataset_sha256", "planned_cases", "complete_cases",
                                         "better_individual", "arms", "comparisons", "issues")
        }},
    }
    write_json(runs / "overlap.json", report)
    return 2 if outcome_code or matrix_exit_code(trigger) else 0, report
