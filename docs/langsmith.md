# Review skill experiments in LangSmith

Use the UUID of an existing LangSmith tracing project and a study name that identifies the collection of attempts. Reuse the same study name when you retry an export. Choose a new name for a fresh native execution, even when its case IDs and repetitions match the previous execution.

Preview a native benchmark. Supply the original runs root to include tool events. Without it, the report supplies the results and metadata.

```sh
skill-benchmark export-langsmith benchmark.json --runs eval-runs/latest \
  --project "$PROJECT_UUID" --study routing-v2 > preview.json
```

For an imported SCBench run, pass the result index in place of the benchmark. The local index can contain reviewer messages, hidden test names and source file references. The projection includes event names, actor labels, checkpoint outcomes and ordinal event numbers. Free-form messages, commands, output bodies and source paths stay local. Hidden-test replay labels become opaque identifiers. Review identifiers and metadata in the preview before uploading.

```sh
skill-benchmark export-langsmith scbench-index.json \
  --project "$PROJECT_UUID" --study reviewer-v12 > preview.json
```

Native export selects tune rows. An SCBench index explicitly selects its archived run. Choose a tuning run when creating the index. Each checkpoint has its own trace. Filter by study and thread ID to follow a problem across checkpoints, or by event number to reconstruct observed order. Child runs mark observations at a point in time. For native result-only rows, the report timestamp locates the trace. Use recorded cost and duration fields for comparisons. Feedback retains the original numeric value. Its optional LangSmith score uses the service's four-decimal precision and supported range.

Set `LANGSMITH_API_KEY` and repeat the preview command with `--upload` to send it. `LANGSMITH_ENDPOINT` selects an HTTPS API base for a regional or self-hosted deployment. The default is the US LangSmith endpoint. Stable IDs allow you to rerun an interrupted upload with the same project, study and artifacts.

Hosted LangSmith accepts run start times within 24 hours of ingestion. Before uploading to the standard US or regional LangSmith API, the exporter checks every root and child run against the current UTC time. A timestamp outside that window stops the entire upload before the first request. Preview works for archives of any age and preserves the recorded dates. Custom endpoints enforce their own ingestion policies.

For long studies, export newly completed checkpoints within the window. For an older study, inspect the local preview or run a new skill comparison from the original task and starting snapshot.

In a native trace, `comparison_group` identifies the case, model, repetition and population shared by counterpart arms. Consult the benchmark report for pairing eligibility and lift. On SCBench, compare outcomes by problem and checkpoint, then inspect the ordered events around a regression. A hidden-test replay outcome and a checkpoint score have distinct feedback keys so you can filter by evidence source.
