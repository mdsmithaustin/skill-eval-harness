# Do two skills work better together?

Use `skill-benchmark interaction` to compare a task with neither skill, A alone, B alone, and both. Each arm uses the same answer cases, fixtures, model, grader, and number of repetitions. The native answer runner asks the model to read the mounted skills. Use the trigger matrix to test automatic activation.

```bash
skill-benchmark interaction evals/shared-benchmark.json \
  --skill-a ~/.agents/skills/pr --skill-b ~/.agents/skills/unslop \
  --agent codex --model MODEL_ID --split tune --k 2 \
  --runs /tmp/pr-interaction --out /tmp/pr-interaction.json
```

Use a standard benchmark manifest with single-turn answer cases. Include at least one gate or critical outcome assertion per case. The experiment grades with built-in deterministic assertions that apply in every arm. It skips trigger cases. The paths in `--skill-a` and `--skill-b` select the skills mounted in the experiment. Each path can point to a skill directory or its `SKILL.md`. Choose a new directory for `--runs`. The example above makes eight model calls per selected case.

The runner freezes each prompt, fixture, skill, and golden answer it uses before making model calls. `interaction-plan.json` records the case contract, grader identity, model, repetition count, and expected designs for all four arms. Each skill's `digest_scheme` is `skill-tree-sha256-v1`. This hashes the exact staged skill directory. For each regular file, ordered by relative POSIX path components, SHA-256 receives the path in UTF-8, one NUL byte, and the file bytes. The skill-copy rules exclude hidden entries and standard generated directories. The digest includes bundled references.

## Read the result

`arms` reports a rate for each of `neither`, `a`, `b`, and `both`. A run's rate is the share of outcome assertions it passed, with the standard critical-assertion veto. The report averages repetitions within each case and gives each case equal weight. A case is complete for a comparison when it has all planned repetitions in every arm used by that comparison.

| Report comparison | Calculation |
| --- | --- |
| `both_vs_neither` | Both minus neither |
| `both_vs_better_individual` | Both minus the higher of the A and B rates |
| `interaction` | Both minus A minus B plus neither |

Estimates are fractions. An estimate of `0.15` means a gain of 15 percentage points. A negative interaction means the joint gain over baseline was smaller than the sum of the two individual gains. The better individual result can be `tie`.

Intervals use paired sign flips over cases, at 95% confidence. The comparison with the better individual skill takes the minimum of two intervals computed with a Bonferroni correction. Small samples, constant differences, and intervals that include zero can lead to an inconclusive result. The top-level `complete_cases` counts cases with all four arms available. Each comparison also reports its own complete case count.

## Inspect retained evidence

```bash
skill-benchmark interaction evals/shared-benchmark.json \
  --runs /tmp/pr-interaction --report-only
```

The report command checks the current selected cases and their fixture bytes against the saved plan. It checks the built-in grader, each arm's design, and the committed artifacts for every planned run. It regrades those artifacts without making model calls. `issues` names missing or incompatible evidence. A comparison with incomplete evidence for any of its required arms reports `withheld` and a null estimate. The command exits with status 2 if any of the planned four-arm cases is incomplete. When a backend cannot verify the served model, the report records the requested model and exposes the unavailable verification in `served_model_checks`.
