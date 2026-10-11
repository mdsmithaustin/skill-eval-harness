#!/usr/bin/env python3
"""Run autonomous Pi skill-trigger evals from shared-benchmark manifests.

This is `skill-trigger-matrix --agent pi` under its own name and defaults. It
does not force `--skill`: the matrix mounts the skill in an isolated Pi config
dir, runs Pi with normal skill discovery, and detects whether the model loaded
the skill from Read/Skill tool calls against the mounted path. Models can
under-trigger, and Pi also loads a skill when the user names `/skill:name`.

The Pi runner used to carry its own copy of the matrix loop. The copies
drifted: this one rebuilt the skill tree for every repetition, stopped the
whole run on one crashed query, and ran Pi from a directory holding the skill
files it was measuring discovery of. The report is now the matrix report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cli_contracts import native_spend_policy
from json_contracts import write_json
from process_runtime import stops_on_signal
from run_trigger_matrix import (
    DEFAULT_TIMEOUT_S,
    MatrixRunStopped,
    eval_rows_from_args,
    matrix_exit_code,
    run_matrix,
)
from skill_inputs import VALID_SPLITS


def build_arg_parser() -> argparse.ArgumentParser:
    """The runner's CLI surface, buildable without parsing (shared-constant
    guards in the tests introspect it, e.g. --split choices == VALID_SPLITS)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("manifest")
    ap.add_argument("--eval-set", help="JSON file with {query, should_trigger} rows; defaults to manifest trigger cases")
    ap.add_argument("--split", choices=sorted(VALID_SPLITS))
    ap.add_argument("--runs-per-query", type=int, default=3, help="repetitions per query; a trigger RATE needs repetition (default 3, the floor docs/tuning-skill-activation.md recommends)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                    help=f"seconds per Pi run (default {DEFAULT_TIMEOUT_S}, the trigger matrix's default)")
    ap.add_argument("--model")
    ap.add_argument("--max-cost-usd", type=float, help="per-invocation paid admission ceiling; active calls can overshoot")
    ap.add_argument("--assumed-cost-per-run-usd", type=float, help="labeled charge for an unpriced repetition; requires --max-cost-usd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-runs", help="optional directory for per-query trace.jsonl/events.json/metrics.json artifacts")
    ap.add_argument("--ablation", help="materialize this (discovery-population) ablation id and trigger-test the altered skill")
    return ap


@stops_on_signal
def main() -> int:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest)
    rows = eval_rows_from_args(args, manifest_path)
    if not rows:
        raise SystemExit("no trigger queries: add kind:'trigger' cases to the manifest or pass --eval-set")
    try:
        report = run_matrix(
            manifest_path, rows, agents=["pi"],
            models=[args.model] if args.model is not None else None,
            runs_per_query=args.runs_per_query, timeout=args.timeout, workers=args.workers,
            trace_runs=Path(args.trace_runs) if args.trace_runs else None,
            ablation=args.ablation, spend_policy=native_spend_policy(args),
            spend_root=Path(str(args.out) + ".spend"))
    except MatrixRunStopped as stopped:
        try:
            write_json(Path(args.out), stopped.report)
        finally:
            raise stopped.cause
    write_json(Path(args.out), report)
    print(json.dumps(report["summary"], indent=2))
    return matrix_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
