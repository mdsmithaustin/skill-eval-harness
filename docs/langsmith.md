# Review skill experiments in LangSmith

Use the UUID of an existing LangSmith tracing project and a study name that identifies the collection of attempts. Reuse the same study name when you retry an export. Choose a new name for a fresh native execution, even when its case IDs and repetitions match the previous execution.

Preview a native benchmark. Supply the original runs root to include provider and tool evidence. Without it, the report supplies the results and metadata.

```sh
skill-benchmark export-langsmith benchmark.json --runs eval-runs/latest \
  --project "$PROJECT_UUID" --study routing-v2 > preview.json
```

For an imported SCBench run, pass the result index in place of the benchmark. The local index can contain reviewer messages, hidden test names and source file references. The projection includes event names, actor labels, checkpoint outcomes and ordinal event numbers. Free-form messages, commands, output bodies and source paths stay local. Hidden-test replay labels become opaque identifiers. Review identifiers and metadata in the preview before uploading.

```sh
skill-benchmark export-langsmith scbench-index.json \
  --project "$PROJECT_UUID" --study reviewer-v12 > preview.json
```

Native export selects tune rows. An SCBench index explicitly selects its archived run. Choose a tuning run when creating the index. Each checkpoint has its own trace. Filter by study and thread ID to follow a problem across checkpoints, or by event number to reconstruct observed order. When the source records both boundaries, the child spans the observed interval. A usage record with only a completion timestamp appears as a point observation. For native result-only rows, the report timestamp locates the trace. Use recorded cost and duration fields for comparisons. Feedback retains the original numeric value. Its optional LangSmith score uses the service's four-decimal precision and supported range.

## Inspect model calls and skill use

Re-run `import-scbench` against the original archive to enrich an older index. Use a new study name for the first enriched upload so any previously uploaded flat events remain in their original trace. A retry of the enriched study uses source-based child IDs. Open a checkpoint, expand its lead and reviewer sessions, then select an LLM or tool run. The Models panel uses these individual LLM runs. It can take time for the project dashboard to reflect ingested traces.

- **Codex.** Rollouts supply session ancestry, turn model, observed request usage, tool call/result pairs, command exit codes and messages. Native answer runs can retain their job’s rollouts with `SKILL_HARNESS_CODEX_ROLLOUTS=1`. A native stdout-only run may have turn totals without individual requests. The root records its evidence source in `model_evidence`.
- **Claude Code.** The importer correlates streamed fragments and project transcripts using request and message identifiers. An Agent tool can link a child session to its caller. Other archived sessions may have unknown ancestry. Cached input is included in input totals and also shown separately in the usage breakdown.
- **Skill evaluation.** Compare the prepared task, an explicit skill request, an injected skill, a successful target skill operation, and the checkpoint result as separate observations. A skill appearing in a prompt establishes exposure. Use the subsequent tools and scored outcome to evaluate whether the skill changed behavior.
- **Source inspection.** SCBench runs carry an `index_ref` into the local index. Follow the indexed source and result references to the archive. Provider recording varies by CLI version, so an observed model call may have usage and a model name without the complete outbound conversation.

### Select content for inspection

The default preview includes names, timing, result status, evidence labels, available token counts, and content digests. Choose the fields you want to make visible to LangSmith evaluators. Run a preview before adding `--upload`.

| Option | Content |
| --- | --- |
| `--include-prompts` | Recorded user tasks and delegation briefs that the importer identified. Injected instruction bodies stay in the local archive. |
| `--include-responses` | Visible assistant text blocks, reviewer reports and the saved native answer. Transcript fragments can be partial. |
| `--tool-content-from RUN_ROOT` | Recorded tool arguments and results that match the original checkpoint, source record and digest. Commands and file contents may appear here. |
| `--model-content-from RUN_ROOT` | Recorded Claude Messages bodies or Codex inference payloads with verified digests. These can contain instructions, conversation history, tool data, and account or session identifiers. |
| `--model-content-side` | Select `request`, `response`, or `both`. The default with `--model-content-from` is `both`. |

Use the original SCBench run directory for `RUN_ROOT`. For a native answer benchmark, supply its original runs directory to both `--runs` and the content option.

### Capture during a new run

For a native `run-claude` or `run-codex` answer run, set `SKILL_HARNESS_MODEL_BODIES=1` in the runner environment. On the SCBench fork, set `capture_api_bodies: true` on the Claude Code or Codex agent configuration. Capture increases the amount of private data stored in the local run archive. Upload is a separate step.

Claude Code writes Messages request and response objects in file mode. The harness matches successful responses to recorded requests using Claude’s index. A request file without a linked response remains an attempt with unknown outcome. [Claude Code capture format](https://code.claude.com/docs/en/monitoring-usage#api-request-body-event)

Codex writes an inference trace when its CLI supports `CODEX_ROLLOUT_TRACE_ROOT` (verified with 0.170.0). The importer uses the thread, inference and response IDs to match the recorded request parameters, returned output items, and token usage to the model run. Codex serializes the response as an envelope containing its observed output items. HTTP headers, SSE framing, and transport bytes are outside that format. Existing rollouts still supply whatever evidence they recorded. The root’s `captured_request_bodies` and `captured_response_bodies` show actual coverage for each attempt.

To preview recorded responses from a native run:

```sh
skill-benchmark export-langsmith benchmark.json --runs "$RUN_ROOT" \
  --project "$PROJECT_UUID" --study response-review \
  --model-content-from "$RUN_ROOT" --model-content-side response > preview.json
```

Use a separate study name when changing which content you publish. [LangSmith LLM trace format](https://docs.langchain.com/langsmith/log-llm-trace)

## Upload and retry

Set `LANGSMITH_API_KEY` and repeat the preview command with `--upload` to send it. `LANGSMITH_ENDPOINT` selects an HTTPS API base for a regional or self-hosted deployment. The default is the US LangSmith endpoint. Stable IDs allow you to rerun an interrupted upload with the same project, study and artifacts.

Hosted LangSmith accepts run start times within 24 hours of ingestion. Before uploading to the standard US or regional LangSmith API, the exporter checks every root and child run against the current UTC time. A timestamp outside that window stops the entire upload before the first request. Preview works for archives of any age and preserves the recorded dates. Custom endpoints enforce their own ingestion policies.

For long studies, export newly completed checkpoints within the window. For an older study, inspect the local preview or run a new skill comparison from the original task and starting snapshot.

In a native trace, `comparison_group` identifies the case, model, repetition and population shared by counterpart arms. Consult the benchmark report for pairing eligibility and lift. On SCBench, compare outcomes by problem and checkpoint, then inspect the ordered events around a regression. A hidden-test replay outcome and a checkpoint score have distinct feedback keys so you can filter by evidence source.
