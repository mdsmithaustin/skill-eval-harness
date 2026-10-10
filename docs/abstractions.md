# Key abstractions

The harness is a pipeline. Each stage hands one well-defined object to the next: a manifest
becomes task rows, task rows become files on disk, files on disk become graded result rows,
and result rows become a report. Because the boundaries are explicit, you can swap the
runner without touching grading, and the default grading path never has to call a model.

This is the **engineering lens** on the terms in [`vocabulary.md`](vocabulary.md): what each object *is* in the code and what it hands downstream. The glossary defines the words; this page shows their shape. Symbols below point at `skill_benchmark.py` at the line where each abstraction is defined.

## The objects, in pipeline order

| Abstraction | Defined by | What it hands downstream |
|---|---|---|
| CLI invocation | `cli_contracts.ValidatedLegacyCLIInvocation` | One validated command, typed projections, and the explicit legacy-handler adapter. |
| Manifest | `validate_manifest` | The full test definition for one skill. |
| Case | `iter_cases` | One scenario with a prompt and graders. |
| Variant | `manifest_contracts.ExecutionVariant` / `task_variants` | The arm a case runs under; the lift axis. |
| Split | `manifest_contracts.Split` | Which cases are visible during iteration. |
| Assertion | `assertion_result` | A single pass/fail check over one run. |
| Prepared task row | `prepared_task_rows` | A runner-neutral unit of work. |
| Run-output contract | `discover_run_bases_under` | The files a runner leaves on disk. |
| Runner / adapter | `run_codex` | Turns a task row into contract files. |
| Trace normalization | `normalize_trace_records` | Runner-specific events, made uniform. |
| Judge plumbing | `collect_judge_tasks` | Qualitative checks, deferred to a model you supply. |
| Grade result row | `grade_case_variant` | One scored row per case/variant/run. |
| Benchmark report | `build_benchmark_report` | Aggregates, lift, and flags. |

## CLI invocation

`argparse` owns syntax and help text, but its namespace is untrusted wire data. Immediately after
parsing, `CLIInvocation.from_namespace` validates the closed `CLICommand` vocabulary, converts path
arguments to `Path`, parses split, execution-variant, and model identities, and rejects non-finite
or command-invalid numeric limits. Its raw argument bag is recursively frozen. Established handlers
do not yet consume the typed projections: they cross the single named `to_legacy_namespace`
adapter, while new or migrated handlers can accept the typed value. This is migration scaffolding,
not a claim that `ty` already checks every handler's options.

The parser choices and `CLICommand` must be the same set. At dispatch, built-in handlers and
backend-projected answer entrypoints must form a disjoint, complete partition of that set. Adding a
command therefore requires an explicit parser spelling, enum member, handler owner, documentation,
and tests; an unknown or multiply owned command cannot fall through a string-based `if` chain.

## Manifest

The `shared-benchmark.json` manifest, under `evals/` or under `evals/<skill>/`, is the source of truth. It names the skill, the files under
test (`skill_paths`), the comparison arms (`variants`), the split policy, the `cases`, and
the `ablations`. `validate_manifest` rejects a manifest whose fixtures are missing, whose
regex assertions do not compile, or whose hidden cases lack a private prompt reference.
Optional blocks (`harness`, `jetty`, `run_protocol`) carry interop and runner hints without
changing the core shape.

## Case

A case is one scenario: an `id`, a `split`, a descriptive `kind` (for example `behavior`,
`negative`, `adversarial`, or `trigger`), the `prompt` or a private `prompt_ref`, optional fixture `files`, `expected_behavior` notes, the
`assertions`, and a taxonomy the report slices on (`domain`, `difficulty`, `trigger_type`,
`success_goals`). `iter_cases` filters by the explicitly requested split. The CLI keeps split
outputs distinct; authors/CI still control when hidden splits are run and whether prompts live in
private `prompt_ref` files.

## Variant

A variant is the arm a case runs under: `with_skill`, `without_skill`, an optional pinned
`old_skill`, or `ablation:<id>`. Variants are orthogonal to cases, which is the whole point
of the tool. Lift is the difference between `with_skill` and `without_skill` on the same
case, so the variant axis is where skill effect becomes measurable. `variant_instruction`
writes the per-arm instruction; the baseline must run from a workspace that cannot read the
skill files, or the comparison means nothing.

In process, every arm is an `ExecutionVariant`: a validated `str` subtype that closes the base
vocabulary and owns the `ablation:<id>` encoding. It retains the existing JSON/path spelling, so
the type strengthens the boundary without changing persisted artifacts. `task_variants` returns
these typed values instead of unchecked strings.

## Split

`tune`, `holdout`, and `holdback` label the intended evaluation phase. Tune cases are for
iteration; holdout is intended for end-of-round scoring; holdback is intended to remain outside the
skill/docs/eval descriptions until after scoring. The harness filters and reports these labels but
does not provide access control—repositories and CI must keep private material private. `prepare` refuses to emit a hidden case with no
`prompt_ref` unless you pass `--allow-missing-prompts` for dry-run planning.

`Split` is the corresponding closed `str` subtype. `CaseKind` remains intentionally extensible
for descriptive labels, but it owns the only structural classification the runners need:
`answer` versus `trigger`. `PreparedTask.from_row` parses all three identity values at the wire
boundary, and the executable task carries their precise types thereafter.

## Assertion

An assertion is one check. The code-side registries are `TEXT_ASSERTIONS`,
`PROCESS_ASSERTIONS`, `EFFICIENCY_ASSERTIONS`, and `QUALITATIVE_ASSERTIONS`
(`skill_benchmark.py:66-95`):

- **Text** (`contains`, `contains_any`, `contains_all`, `excludes_any`, `regex`,
  `not_regex`, `file_exists`, `json_field_equals`, `golden_output`, `similarity`,
  `structured_output`, `script`): runs against `output.md` and sibling files.
- **Process** (`skill_invoked`, `command_ran`, `command_not_ran`, `command_order`,
  `tool_call`, `tool_count_le`, `no_repeated_command_loop`): runs against normalized trace
  events, and fails closed when the evidence is absent.
- **Efficiency** (`total_tokens_le`, `elapsed_seconds_le`, `command_count_le`): runs against
  metrics.
- **Qualitative** (`judge`, `rubric`, and the `factuality` preset): deferred to a model you
  supply.

`assertion_result` returns `{name, type, passed, evidence, score}`. Every result carries a
`score` (binary detectors mirror `passed` as `1.0`/`0.0`; `similarity`, `golden_output`, and a
graded `script` oracle set a real value), and `grade_case_variant` stamps a `severity`
(`critical`/`gate`/`soft`) and an `oracle` tier (`strong`/`demo`/`live`) on each.

Severity decides how a result counts; the three tiers are defined under **Severity** in
[`vocabulary.md`](vocabulary.md#things-you-assert). The
graded shape (roadmap 2.2, ported from `adewale/anti-slop-writing`) adds two `judge` assertion
forms:

```jsonc
// anchored dimension: the judge scores against named, observable anchors
{ "type": "judge", "graded_dimensions": [
  { "name": "specificity", "scale": "1-5",
    "rubric": "5 = names the failure mode and the mechanism; 1 = generic restatement" }
] }

// dynamic rubric: the judge drafts case-specific criteria, then grades against them
{ "type": "judge", "dynamic_rubric": { "instruction": "draft 3-5 criteria from the brief",
                                       "minimum_criteria": 3 } }
```

A graded assertion answers "how much better," where the binary `passed` answers only "right or
wrong." That distinction is the reason a saturated binary case can still show graded lift. An
optional `reference_score`/`reference_graded_score` on a case sets a no-regression floor, and
`build_paired_summary` reports a paired `graded` channel and a sign-flip significance test
beside the raw lift.

## Prepared task row

`prepared_task_rows` fans `cases × variants × models × runs_per_variant` into runner-neutral
rows (the `model` axis, from `prepare --models`, adds a run-dir segment only when two or more
models run, so single-model layouts are unchanged). Each row carries the `prompt`, the absolute
`input_files`, the `skill_paths`, the `instruction` for its arm, and the `run_dir` it must
write to. Generation rows omit `expected_behavior` and rubrics unless you pass
`--include-answer-key`, so a runner cannot accidentally feed the answer key to the model under
test.

There are two in-process types. `PreparedTaskDraft` may hold partial planning data but is never
executable. `PreparedTaskDraft.validate()` / `PreparedTask.from_row()` constructs a strict
`PreparedTask`: identifiers and paths are validated, split and execution variant are closed,
repetition is positive, `run_dir` is safe and relative, and `without_skill` cannot carry skill
paths. Every native runner and Jetty exporter requires the executable type.

`CaseId`, `ModelId`, and `RunNumber` keep the run identity dimensions distinct from ordinary
strings and integers. They serialize as the existing scalar values, while construction rejects an
empty case/model or a boolean, zero, or negative repetition before that identity reaches pairing.

The prepared rows also form a persisted `answer-design.json`: the exact expected
case/model/repetition identities plus a digest of manifest inputs, referenced oracles, and
variant instructions. Each produced run repeats the design, task, and instruction digests.
Report builders re-derive the current eval contract and require exact design coverage before
publishing paired or aggregate headlines; surviving rows from an incomplete design are retained
only as explicitly observed diagnostics.

## Run-output contract

`PreparedTask.recovery` optionally holds a typed `RecoveryCase` from `invocation_contracts.py`.
`PreparedTask.from_row()` parses its external fields. Recovery content contributes to case and task fingerprints.
The fixed `run-agent` lifecycle, also used by `run-codex` and `run-claude`, retains one workspace across fresh processes.
Recovery rows write `recovery.json` and raw snapshots rather than ordinary answer artifacts, workspace diffs, grades, or normalized paired telemetry.
`RecoveryCapture` passes capture destinations and the initial checkpoint condition directly through existing adapters to `invoke_argv_with_timeout`.
That function remains the sole subprocess owner. Ordinary outcome factories and `captured_workspace` callers do not change.
See the [recovery reference](recovery.md) for its output contract and limitations.

The ordinary answer contract is the file boundary between an answer runner and the harness:

```
runs/<case_id>/<variant>/[run-<n>/]output.md
runs/<case_id>/<variant>/[run-<n>/]metadata.json     # optional
runs/<case_id>/<variant>/[run-<n>/]trace.jsonl        # optional, raw
runs/<case_id>/<variant>/[run-<n>/]events.json        # optional, normalized
runs/<case_id>/<variant>/[run-<n>/]metrics.json       # optional, normalized
runs/<case_id>/<variant>/[run-<n>/]artifact-commit.json # harness-written commit marker
runs/<case_id>/<variant>/[run-<n>/]workspace-changes.json # answer runners and run-subagent: candidate workspace edits
runs/<case_id>/<variant>/[run-<n>/]candidate.patch        # text edits as a git patch, only when non-empty
runs/<case_id>/<variant>/[run-<n>/]candidate-files/<sha256> # content copies a patch cannot carry
```

`discover_case_model_roots` finds each case's variant directories (with or without a model
segment), and `discover_run_bases_under` and `read_output_base` read the runs under them. A runner that writes these
files is a valid runner, whether it is Pi, Codex, Jetty, a subagent, or a person with a text
editor. Harness-owned schema-v1 writers commit their required files and SHA-256 inventory by
writing `artifact-commit.json` last; a missing or stale marker makes such a declared artifact
set incomplete. Legacy or externally written runs that do not declare that contract version
remain readable through the compatibility boundary. This boundary is the main extension seam
in the codebase.

`artifact_contracts.observe_artifact_set` reads that seam into exactly one frozen value:
`LegacyArtifactSet | MissingArtifactCommit | InvalidArtifactCommit | IncompleteArtifactSet |
CompleteArtifactSet`. A malformed marker and a valid marker whose files were interrupted or
tampered with are therefore different states. Existing dictionary readers retain
`artifact_set_complete` as a compatibility projection and expose the reasoned state alongside it.
Likewise, `read_event_log_base` produces `MissingEventLog | InvalidEventLog | LoadedEventLog`;
`read_events_base` is only the legacy tuple adapter. JSON parsing lives in
`json_contracts.py` as two rules that differ only on a repeated object key: `strict_json_loads`
rejects it and is the rule for every artifact the harness authors or validates, while
`parse_stream_json`/`stream_json_loads` resolves it last-value-wins (stdlib semantics) and is the
rule for bytes an external agent CLI wrote (`codex exec --json` repeats `id`), reporting the
repeated names so a row can record them under `stream_duplicate_keys`. Both reject non-finite
numbers. `iter_json_objects` and `parse_trace_jsonl_text` select the rule with `strict`.

For ordinary rows, answer runners and `run-subagent` also record what the model did to its temporary workspace.
`workspace_contracts.captured_workspace` owns that directory's lifetime: build, copy a baseline,
run the provider, diff, delete. The diff is a sorted tuple of `Added | Modified | Deleted`
changes; each side is a `RegularFile | Symlink | Special | Unreadable` read by `lstat`, so a
model-created link is recorded by target and never followed. `RegularFile` records the executable
bit, so a chmod-only change is `Modified`. Only files are indexed, as in git, so an empty directory
is never a change. A file that cannot be read, or a directory that cannot be listed, becomes one
`Unreadable(mode)` entry with `Omitted("unreadable")` evidence; the capture does not descend into
that directory and does not report the baseline files under it as deleted. Each change carries
one `InPatch | InBlob | InState | Omitted` evidence value. Text within the per-file cap goes into
`candidate.patch`, which `git apply` replays onto the baseline, mode lines included; the manifest
records the patch's SHA-256. Binary or non-UTF-8 content, paths git would have to quote, and a
modified text file whose before side exceeds 1 MiB are copied to `candidate-files/<sha256>`, and
those content-addressed names keep model basenames such as `metadata.json` out of the run
directory. Content past the 1 MiB per-file or 32 MiB per-run cap is `Omitted`. The capture runs
for every returned outcome, including `TimedOut`. Handled I/O failures write a `captured: false`
manifest with `capture_error` and the receipt still commits. Readers derive
`workspace_changes_captured` and `workspace_changes_state` (`captured | partial | failed |
invalid`) next to `artifact_set_complete`. The claim holds only for a complete artifact set whose
manifest parses and omits nothing, whose `candidate.patch` inventory digest equals the manifest's
recorded digest, and whose blobs the commit inventory holds under their own digests. A run without
a manifest gets neither key. The claim does not feed `execution_valid`.

`content_digests.py` owns how bytes are hashed. `file_sha256` is the digest an artifact commit
records and verifies; `tree_sha256` hashes a file tree as (relative path, bytes) entries sorted by
path component, each framed as path, NUL, then content. The canonical `skill_tree_hash`, the
workspace fixture hash, the script-oracle trees in the eval contract, and the Jetty upload plan
all go through it, so a digest computed from the upload plan equals the canonical one it is
checked against. Each skill root sits in a skill tree under its own directory name, so the
hashed paths are the paths an agent lists. The judge's explore-surface digest frames directories too and stays separate.

## Runner / adapter

An **answer runner** consumes prepared task rows and produces the run-output contract for ordinary rows. The repo
ships Pi answer smoke (`examples/adewale-workspace/run_pi_smoke.py`), Codex (`run_codex:11806`), Claude (`run_claude:12095`, capturing real
per-run cost), Gemini CLI and Mistral Vibe (`run-agent --agent gemini|vibe`, using isolated provider homes outside the workdir), the in-process
subagent runner (`run_subagent:15391`, which hosts record/replay tool I/O via `ToolReplayStore`),
Jetty (`JettyClient:4278` and the export/run/import commands), and any runner that writes the
contract directly. Each answer runner registers a workspace builder so one cross-runner invariant
proves its `without_skill` arm is skill-free (CF.2). Autonomous trigger runners are separate: they
read trigger cases from the manifest directly, never consume answer task rows, and emit trigger
observations plus optional traces rather than answer grades.

Native answer backends return the frozen `Completed | TimedOut | SpawnFailed | ProviderFailed`
union from `runner_contracts.py`. `OutcomeContext` validates provider, telemetry, and elapsed-time
fields; `write_runner_outcome` exhaustively writes the ordinary disk contract.
`OutcomeContext` extras cannot set the derived `cost_availability`, `observed_subtotal_usd`, or
`cost_reason` fields. The publisher derives those fields from typed timeout cost
evidence. Nested subagent rejection diagnostics retain their reported cost labels.
Recovery rows use the separate lifecycle described above. A backend therefore cannot
independently set timeout, return code, answer, and failure into a contradictory bag. The harness
calls no model during default grading; it reads what the runner left behind. The explicit
`--allow-scripts` and `--embed-cmd` modes may invoke caller-supplied external oracle subprocesses.

`spend_contracts.py` owns an immutable `AnswerCall | SubagentTurnCall` plan, closed call states, and observed, assumed, unpriced, or proven nonbillable charges. It reuses `RunCoordinate`, `Money`, and `Measurement`. Derived totals retain unknown costs as partial evidence. `spend_runtime.py` owns serial `SpendAdmission.run`, which publishes admission before its callback and settlement afterward in one exclusive invocation directory. Native runners retain workspace, subprocess, artifact, and recovery policy. Reports read each ledger without changing the immutable answer design. The [native spend walkthrough](limit-native-spend.md) describes the operator contract.

Subagent admission binds each required external callback turn to its prepared task digest, `RunCoordinate`, and positive turn number. The immutable plan uses the scripted `turns` list, or turn 1 for a single call. Provider-internal turn limits do not add calls. Built-in backends capture immutable reported dollars, scope, and actual process evidence before response validation. Accepted responses freeze recursively and thaw at the existing dictionary validators. Rejected responses retain safe dollars and raw envelopes in diagnostics. Multi-turn prices require explicit `turn_delta`. Cumulative or unspecified counters remain provider diagnostics. Assumptions remain in the ledger.

Actual process timeouts make trustworthy captured dollars a subtotal rather than a whole-call price. Native Claude and subagent admission pass that floor through `Priced.observed_subtotal` to the existing settlement owner. Partial artifact labels preserve the observation without publishing a complete cost. Strict original UTF-8 and complete shell JSON protect shell capture. Claude price capture separately rejects ambiguous dollars or terminal stream structure while retaining its documented last-value-wins handling for unrelated stream keys. Invalid Claude token records become protocol-error records for runner publication. Their original provider bytes remain in the raw trace.

`parse_claude_cli_json` isolates reported dollars without process facts. The shared
`claude_cli_invoke` owner classifies them after observing the process. An actual timeout
returns `cost_usd=None` and a nullable `observed_subtotal_usd`. Natural exits retain safe
full dollars even when provider, answer, or token validation fails. Native answer and default
subagent callers explicitly transfer the subtotal into their existing closed timeout evidence.

`run_subagent_tasks` retains conversation, workspace, replay, and turn-artifact ownership. Settlement precedes artifact publication and final workspace capture. Each remaining refusal passes through `SpendAdmission.run` without invoking its callback, adding history, or writing a provider turn. The private subagent terminal types live beside the common transactional `write_runner_outcome` publisher. They bind refused or rejected call identities to the root and derive null return codes, absent process lifecycle, and false provider completeness. The process outcome union stays closed. Committed inventory completeness remains a separate reader observation. An invocation that starts no call preserves all prior destination content. A new started conversation replaces the root with its own incomplete result and current sidecars when required turns cannot run.

The native answer loop lazily enters `captured_workspace` through an `ExitStack` inside the admitted callback. It returns the priced outcome and actual `WorkspaceAttestation` before workspace exit. `SpendAdmission.run` persists settlement before capture and cleanup, so later local errors preserve the charge. The writer consumes captured sidecars after workspace exit and before the changes directory closes. Exceptions from setup, invocation, pricing, or settlement skip capture while the workspace context cleans up. Returned failure, timeout, and spawn-failure outcomes still capture evidence.

Before a native provider subprocess starts, `invocation_contracts.py` constructs one
`ProcessInvocationPlan`: immutable argv, stdin, working directory, environment, a positive
`TimeoutSeconds`, and an optional `redact_output` function that the subprocess owner applies to the
whole captured stdout and stderr before it caps stderr (the Codex trigger adapter uses it to strip
host skill paths). `redact_stdout=False` limits it to stderr, which Codex answer runs use because
their stdout is the trace a `tool_sequence` assertion reads. `run_argv_capture` accepts only that plan and returns the closed
`InvocationResult` lifecycle. The answer backend receives the smaller `InvocationRequest`, whose
model and timeout are also precise values. Provider adapters can choose wire formats, but they
cannot omit or disagree about process inputs after the plan boundary. The same module owns the
shared `InvocationState` vocabulary: `InvocationResult` admits only process-boundary states, while
provider or harness failures remain semantic classifications and never rewrite the observed return
code. `InvocationRequest.effort` carries a requested effort level; `run_agent_tasks` refuses it
before any spend when the backend declares no `effort_control` or the level is not among its
`effort_levels`.

`completion_contracts.py` records how each ordinary answer run ended. `StopObservation` normalizes a
provider's stop reason into the closed `StopClass` (`completed`, `truncated`, `turn_limit`,
`refused`, `other`, `unavailable`) and keeps the raw value beside it. `ServedModel` applies one rule
to every backend: one reported model is credited and compared with the request (a dated snapshot
suffix or a family alias still matches); several reported models credit none and read `mixed` when
the requested model is among them, `mismatch` when it is not. Claude subagent turns are not
counted, because a subagent may use another model by design. The check reads `match`, `mismatch`,
`mixed`, `unverifiable`, `unavailable` or `not_requested`. `EffortSetting` records the requested
level and how the backend applied it, or `backend_default`. The shared writer fills `unavailable`
and `backend_default` for any runner that reports nothing, so an old run and a run with no evidence
are distinguishable. `execution_valid` ([execution validity](vocabulary.md#run-artifacts)) treats a
truncated, turn-limited or wrong-model run as unscorable, because grading it would blame the
requested model for the eval's limits or for another model's answer; a refusal and a mixed run stay graded and are counted in the report's `run_endings`
block.

`observation_contracts.py` owns how the harness says whether it observed something.
`Availability` (`complete`, `partial`, `unavailable`, `not_applicable`) is the canonical vocabulary;
`Availability.parse` also reads the older spellings still persisted in run artifacts (`incomplete`,
`unknown`, `unobserved`, `missing`, `not-applicable`), so code compares members instead of
re-spelling strings, and a test fails if a production module writes a retired spelling again.
`TelemetrySource` is the one list of where a usage or cost number came from, and the usage and cost
source sets that the answer path and the trigger path both validate against are derived from it.

## Trace normalization

Runners disagree on event shape. Codex emits `command_execution` and `turn.completed`; Pi
emits `message_end` usage aliases; Jetty emits trajectory records. `normalize_trace_record`
and `normalize_trace_records` collapse these into one schema-versioned `events.json` plus
`metrics.json`, tagged with the source. Loading `events.json` first constructs the closed event-log
observation above. `trace_contracts.EventState` then classifies each event as
completed, in-progress, failed, or unknown; only completed operations contribute command/tool/file
counts. Process and efficiency assertions read the normalized form, never the raw prose, because
inferring tool use from answer text is how false evidence gets in.

Autonomous trigger execution crosses the parallel `trigger_contracts.py` boundary: process state,
completion evidence, detection evidence, and the final observation are closed values before
`trigger_reporting.py` constructs a complete, incomplete, or empty cohort.

## Judge plumbing

Qualitative assertions defer. `collect_judge_tasks` gathers every `judge`/`rubric` assertion
across runs and keys each by `judge_task_id` (`case::variant::run-n::assertion`, with a `model`
segment on a multi-model run). `grade --judge-tasks` can serialize that queue, while the
`judge` command reconstructs the same tasks from the manifest and run directory rather than
reading the optional queue file. Before queuing, `grading_contracts.JudgeTask` validates and freezes
the case/model/variant/run identity, paths, assertion, conversation, and prompt/evidence
fingerprints. `judge_prompt` renders the case, expected behavior, rubric,
and candidate output into a prompt — including the anchored dimensions or dynamic-rubric
instruction for a graded assertion; `run_one_judge_task` pipes it to the `--judge-cmd` you
supply or to a native `--judge-backend` (`claude`, `codex`, `gemini`, or `vibe`) plus
`--judge-model`;
both routes construct the frozen `judge_contracts.JudgeInvocation` boundary before any verdict
parsing. It closes return code, output, immutable usage/cost telemetry, provenance, and model
identity, so a newly registered backend cannot feed a dictionary-shaped partial contract into row
assembly. Its keyword-only `observed_subtotal_usd` requires actual `TIMED_OUT` state and code 124.
It cannot coexist with full `cost_usd`, including zero. Timeout and spawn failure cannot carry full
cost. Top-level metadata reserves `cost_usd`, `cost_normalized`, `cost_availability`,
`observed_subtotal_usd`, `cost_reason`, `cost_aggregate`, `telemetry`, and process lifecycle fields.
Construction and dataclass reconstruction enforce the same checks. Nested diagnostics remain
legal and cannot supply canonical process or price facts.

One recursive billed-leaf policy serves repeat and panel consensus, benchmark judge spend, and
saved reporting. Parents carry derived `cost_aggregate` buckets and never add another charge.
Two $0.06 timeout floors retain a partial USD subtotal of $0.12, zero whole-price observations,
and two unavailable calls. Whole-price totals, ratios, and `verdicts_with_cost` exclude those floors.
The saved reader validates scalar, normalized, and v3 prices before choosing a channel. It rejects
contradictory timeout or parent price projections and competing membership paths. Completed legacy
rows retain their supported price semantics. Incompatible bases and currencies retain their
reasons without reviving rejected sums or converting money.

`merge_repeated_judge_rows` majority-votes pass/fail and medians scores across repeats. The
harness picks no model. At the result boundary, `judge_verdict.py` parses one strict variant:
boolean, scored, dimension-scored, dynamic-rubric, or consensus. Pass is derived from the typed
payload; duplicate IDs and contradictory score/threshold/pass rows are rejected. The serialized
row still carries `{judge_task_id, verdict_kind, passed, score, evidence}` — plus
`dimension_scores`/`criteria` for a graded verdict and normalized `usage_normalized`/
`cost_normalized` for judge spend — merged back whenever `grade` or `benchmark` receives
`--judge-results`. Both task and result carry `judge_input_sha256`, binding the verdict to the
exact rendered prompt, candidate output, and evidence. A stale or mismatched result is rejected
or re-queued even when its `judge_task_id` still matches.

Every per-run record shares one key, `manifest_contracts.RunCoordinate` (case, execution variant,
run number, and a model on a model-fanned run). `judge_task_id` is that coordinate's rendering
with an assertion label, so a judge task, a human judgement and a result row cannot name the same
run differently. Repeated runs of one judge and a panel of judge models fold their verdicts with one
rule, `judge_verdict.resolve_consensus`: a strict majority passes, an explicit `--quorum` overrides
it, and an exact tie is decided by the median score only against an explicit threshold, else it is
`unresolved` and does not pass. Both merges report the same `agreement` block, so a judge that
disagrees with itself is visible rather than averaged away.

Human verdicts have one shape, `human_judgements.HumanJudgement`: a run coordinate, an optional
judge assertion, a `pass | fail | unsure` verdict and a note. `render-viewer --serve` writes them
to `feedback.json`; `judge-alignment --labels feedback.json` turns each pass/fail verdict on a named
assertion into a label for that assertion's `judge_task_id`; `error-analysis --feedback` puts the
run-level notes into its review queue. A reviewer writes a verdict once, and the calibration label
and the review note cannot disagree.

## Findings, gate policy and eval health

`findings.py` is the one vocabulary for "this eval has a problem". `CaseFlag` is the closed set of
per-case benchmark flags; each value is the exact wire text (three carry a `": detail"` suffix), and
consumers compare members instead of matching substrings. `FindingKind` registers every finding
kind the harness emits, from `audit-manifest`, readiness, `profile-skill`, `cost-summary`,
`contamination` and `judge-robustness`; each kind declares its `Subject` (`skill`, `eval`,
`grader`, `run`), its default `Severity` and the `EvalMark` it is evidence against, if any.
`Finding.as_dict` keeps the historical `{kind, severity, message, evidence}` shape.

`eval_health` is a view over findings, not a second copy of them. It takes the findings and a
per-mark `observed` flag, and rates each of the five marks
([defined in the glossary](vocabulary.md#eval-health)): a mark with a finding of its kinds is
`concern`, an observed mark with none is `ok`, and the rest are `unavailable`, which is not `ok`.
`audit_manifest_report` supplies the observations: a recorded case `source` for mark 1, a graded
reference or null answer from `known_answer_check` for mark 2, the run conditions for mark 5
(`run_condition_findings`, on any benchmark), and a complete benchmark for marks 3 and 4, whose
run-measured findings (`run_measured_findings`) are computed only then. It feeds the
audit findings and the readiness blocker findings through the same view, so a blocker such as
`floor-eval` counts against mark 3 like any other finding of that kind.

`gate_policy.py` decides what fails a command. A `GatePolicy` names finding kinds and severities,
and `decide` fails on any matching finding and, when told the evidence is incomplete, fails closed.
Four presets carry the older flags' meaning: `READINESS` (`blockers`), `SELF_JUDGING`
(`strict-judge`), `CONTAMINATION` and `JUDGE_ROBUSTNESS`. `audit-manifest --fail-on-blockers` and
`--strict-judge` evaluate through the first two, `contamination --fail-on-contamination` and
`judge-robustness --fail-on-findings` through the last two. Each command says when its evidence is
incomplete: the benchmark's availability, contamination's `coverage` of answer runs, robustness's
`summary.availability`. `gate_exit` prints each reason and returns the exit code.
`parse_fail_on` reads the kinds, severities and preset names a user passes to
`audit-manifest --fail-on` and rejects an unknown token. Grading options such as
`--strict` are not gates: they change how verdicts are scored, not whether a command fails.

## Grade result row

`grade_case_variant` produces one row per case/variant/run. It separates objective, process,
efficiency, and qualitative counts, computes each pass rate, marks `missing_output` when a run
never produced text, and carries the run `metadata`. Deferred judge assertions leave a task
behind rather than a verdict. Grading reads from disk and calls no model, which is what makes
a default re-grade cheap and deterministic; opt-in script and embedding oracles are the explicit
external-process exceptions.

Before aggregation, every objective and qualitative row becomes one
`SatisfiedAssertion | FailedAssertion | UnavailableAssertion | SkippedAssertion`. Severity and
oracle tier are closed enums, scores must be finite, and unavailable/skipped rows cannot enter a
pass-rate denominator. The legacy dictionary rows are serialized only after typed aggregation, so
`passed: null`, dependency skips, and observed failures no longer share an implicit falsy branch.

## Benchmark report

`build_benchmark_report` invokes the shared grader for each discovered run, then
turns those in-memory result rows into the artifact you read. It does not consume the output
of the `grade` command. Before arithmetic,
`experimental_pairs.py` constructs exact `(case, model, repetition, population)` identities and
requires one eligible treatment and control arm from an explicit `ContrastSpec`. The default
skill-presence contrast maps to the existing `with_skill`/`without_skill` wire rows; ablation
confirmation pairs `with_skill` with `ablation:<id>` under the contrast `ablation:<id>` (a missing
arm blocks as `missing_ablation:<id>`), and `EDIT_CONTRAST` pairs `with_skill` against
`old_skill` for `paired_edit_summary`, the same-run edit comparison. `contrast_for` returns the declared contrast for a pair
of arms and refuses any other pairing, so an arm is never relabelled into another arm's slot. Each
contrast names its `held_fixed` factors (today `HeldFixedFactor.EFFORT`), and
`ContrastSpec.comparability` blocks a pair whose arms differ on one.
The stable identity of a comparison result is `(contrast_id, pair_key)`. Blocked rows and pairing
diagnostics serialize that contrast ID, so two different comparisons over the same execution rows
cannot collide or lose their causal question at a persistence boundary.
`build_paired_summary` computes per-case lift
(`with_skill` minus `without_skill`, normalized gain, and a flag when the skill hurts) only from
those pairs; missing/ineligible arms remain in `pairing` diagnostics and duplicate arms fail.
Each paired block also carries `effect_estimates.sign_flip_interval`, the sign-flip test inverted
into a confidence interval. The test (`sign_flip_significance`) and the interval share one
sign-flip core, so the interval excludes zero exactly when the test rejects "no lift", sampled
path included, and `noise_check`, which reports the cases that moved, the smallest p-value those cases can
reach, the interval half-width and the headroom left in `without_skill`. `effect_estimates.Estimate`
builds all three blocks from one set of deltas and stamps each with its `InferenceUnit`, defined
under **Inference unit** in [`vocabulary.md`](vocabulary.md#report-signals). Through that
comparability check, `construct_pairs` blocks a pair whose arms ran at different effort
(`effort_mismatch`) or where only one arm recorded effort, in the benchmark, the ablation
confirmation, and `token-overhead` alike.
`build_slice_summary` breaks results down
by domain, difficulty, trigger type, and success goal. Case flags mark saturated, no-lift,
flaky, and with-skill-failed cases, and `effect_estimates.ceiling_or_floor` separates the two
ways a case stops discriminating: both arms always pass (ceiling) or both always fail (floor, which
`suggest-cases` never offers for hardening). These flags, the leakage lint
(`prompt_assertion_leakage_findings:972`), and the split discipline are the part of the tool
no surveyed eval framework copies.

`report_contracts.report_cohort` classifies each attempted reporting population as
`EmptyReportCohort | CompleteReportCohort | PartialReportCohort |
NotApplicableReportCohort` from one stable-identity attempt sequence before computing variant and
slice headlines. `metric_cohort` then refines that same population per rate: row completeness can
never authorize a survivor-only metric mean. Partial cohorts expose observed diagnostics but
withhold headline means; empty and non-applicable populations remain distinct. `UnitRate` rejects
booleans, non-finite numbers, and values
outside `[0, 1]` before `statistics` can aggregate them. Attempted, observed, and blocked counts are
derived from the same deeply frozen attempt dispositions, preventing consumers from assembling
inconsistent coverage or relying on Python object identity.

The pairing key carries `CaseId`, `ModelId | None`, `RunNumber`, and the closed
`ExperimentalPopulation` enum. The pair also carries a contrast whose canonical factor coordinates
separate activation, skill set, and content revision; those differing treatment coordinates do not
pollute the shared repetition identity. `ExperimentalPair` is generic in its payload, so pairing run paths
retains `Path` while pairing result rows retains their mapping interface; the shared constructor no
longer erases every downstream payload to `Any`.

When introducing or extending an abstraction, review the whole semantic path rather than only its
constructor:

1. name the untrusted boundary and the one owner that validates it;
2. define stable value identity independently of Python object identity and treatment differences;
3. keep process, provider, trace, judge, artifact, coverage, and telemetry facts on separate axes;
4. make every refinement monotone, especially per-metric report coverage;
5. serialize only at the artifact/UI edge, then re-validate on read; and
6. prove the model twice: malformed/contradictory runtime cases plus `ty` narrowing and exhaustive
   consumption.

The compact owner inventory and extension rules are in [`typed-python.md`](typed-python.md). The
failure models and adversarial proof styles are in
[`correctness-by-construction-audit.md`](correctness-by-construction-audit.md).

## What changes when you extend the tool

Two abstractions absorbed most of the roadmap, and they remain the seams to reach for next.
The numeric `score` and the `severity` tier live in the **assertion result shape**
(`assertion_result`) and the totals in `grade_case_variant`; the `model` sweep is a third axis
in the fan-out (`prepared_task_rows`) with a grouping in `build_benchmark_report`
(`by_model`/`model_analysis`). Both are cross-cutting: a change to either touches every place
that *aggregates* a run (each pass-rate/report view) or *identifies* one (`run_dir`, run
discovery, `judge_task_id`), so extend by auditing those consumers, not just the definition
site. Touch these carefully and most other features fall into place around them.
