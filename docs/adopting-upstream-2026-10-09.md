# Upstream adoption after the 2026-10-09 sync

The fork accepted fourteen implementation PRs, PR26 through PR39. Skill authors gain measured Pi retries, offline walkthroughs, per-invocation spend admission, simulated tool outputs, a Linux offline bridge, and optional verifier diagnostics. Maintainers gain existing-provider conformance tests and canonical trigger owners with dependency-derived source identity.

The accepted implementation predecessor is `8a85843da0d0914a8bc19044d5976eb71774ff7d`, with tree `aa58bb0dd9b1c7c1da67df8da0bb82f49c9ab153`. Its [main CI](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38143022884) passed Linux Python 3.10, 3.11, and 3.12 and the Windows Python 3.12 text-contract job. The [acceptance comment](https://github.com/mdsmithaustin/skill-eval-harness/pull/39#issuecomment-6109560422) binds that merge to the reviewed tree. This documentation update follows that predecessor. Its own review, PR CI, merge, and main CI remain separate acceptance work.

The user narrowed the work to ordinary fork maintenance. Acceptance used regression tests, compatibility checks, documentation, independent Code and Docs review, and repository CI. Each later pass changes the corresponding current status. Earlier failures remain part of the record.

## Pinned inputs and sync checkpoint

The initial fork was `c23ed79554995c60cbd63cc9da9f47837cfa6f9a`. Upstream `main` was `0cc7392473c3d8e2e7de7412239ba39229101604`, including the merged PR97 work. The [dated sync record](upstream-syncs/2026-10-09.md) retains the normal sync merge `d0f02b64e7bea2ba95cdf3731b85aa27f25f4ddc` and its original checks. PR26 landed through the normal merge `95054f1166d84cb9434ae34376bd617546ce9078`. The annotated `sync-upstream-2026-10-09` tag points to that accepted PR26 merge. Its tag object is `b8a8e7bc04b4e48341432d000f8c11fc07e6787a`.

These candidate heads identify the original assessment. Their contents are historical inputs, not claims about the current upstream branches.

| Upstream candidate | Pinned head |
| --- | --- |
| PR33 | `288fbf670c561b6fa8a0803ea9452eaf1faff055` |
| PR62 | `9c68fe17cad6305eacfbf4559722756e34d98e63` |
| PR82 | `c62339a4dfdb466b533252605a5150bed4e39140` |
| PR85 | `75b22e32a1a63f80e3096b61e93b340cd2d108c3` |
| PR86 | `991bb54c97fe85bba31b3209006e14b5bf94a740` |
| PR87 | `4f7f8c8b507e25d1065c97aa4d9b4f601bf38d0a` |
| PR88 | `1e165393b013eee5b7dac92f64362368080141c6` |
| PR89 | `26b63e7b8e8c482ad5ecce89bbd91d7f1f4d831b` |
| PR90 | `1768d57b4d5489ae7e3b6b165d1053a1a4eb1efc` |
| PR91 | `3eabc48c13b6b3e0e03af5cecc0ff9c36f563e4b` |
| PR92 | `1fa749a17ebbaaa016c45acde335baf642245357` |
| PR93 | `3ecd844db80af7ca400acf76d8b56677271e4e51` |
| PR94 | `32ed363220d92a258d99463590d0fbb80cdd94ab` |
| PR95 | `f1cb61f0bd004c21bfa999d16b6b6714e5dfabc7` |
| PR96 | `779b9a39707072032daa963b9687611825a2642d` |
| PR102 | `04763168bc073791f31bc11be88c4aee822b23aa` |

## Retained value and rejected scope

| Upstream work | Fork decision and reason |
| --- | --- |
| Main, including PR97, and PR102 | Merge main first for shared path validation and grading tests. PR29 separately restores the PR102 telemetry-state test absent from the pinned tip. |
| PR87 | Measure retries only from complete Pi streams. Unsupported or incomplete evidence remains unavailable. |
| PR88 | Adapt executable trace and discovery journeys to current report behavior and the Pi facade. |
| PR82 | Retain conformance and trace-conservation checks for existing providers. Defer Antigravity because this adoption has no fork consumer for it. Provider declarations alone do not establish enforced containment. |
| PR85 spend | Admit and settle actual answer, subagent, judge, Jetty, and matrix calls. Keep unique invocation ledgers and retain unknown prices and observed floors. |
| PR85 diagnostics | Retain optional verifier authoring leads. Defer executable model ordering and the plugin-eval importer. No consumer justifies importer translation or replacement of valid manifests. |
| PR86 | Retain paired declarations and conservative observations, then add a bounded offline bridge under the existing supervisor. Exclude the live shell example because its working directory does not enforce its claimed confinement. |
| PR91 through PR96 | Use applicable ownership and identity ideas. Move the fork's current definitions into five canonical owners. Reject wholesale PR96 restoration and the unrelated broad grading and report split. |
| PR33, PR62, PR89, and PR90 | Skip superseded implementations, the superseded aggregate split, and upgrade guidance already present in the fork. |

The premise reviews changed concrete choices. The upstream spend ledger overwrote earlier invocations, and judge admission covered an entire panel batch. The fork instead admits each actual paid call and writes an exclusive invocation ledger. One call can overshoot a ceiling. An explicit assumption cannot erase an observed floor. A command is not free merely because its tools resolve offline. See the [spend guide](limit-native-spend.md) and [price regression tests](../tests/test_judge_spend.py).

The upstream fault check matched totals by tool name and could accept an unrelated failed call. Current observations bind the prepared task, run, turn, exchange, input, output, source, and fault identity. Direct callable delivery establishes output only. Its delivered errors and operation success remain unavailable. Positive confirmed prefixes can support witnessed claims, while exact counts and absence require closure. The bridge carries known or unavailable error data in the exact ACKed payload. ACK proves child protocol acceptance. See the [offline dialogue reference](offline-tool-dialogue.md) and [fault tests](../tests/test_tool_faults.py).

An unequal-cohort counterexample produced a model-order inversion despite both models behaving identically on each case. Repetition weights of 100/1 versus 1/100 yielded 99.01% versus 0.99%. Complete case names alone were insufficient. Eligible producer provenance was also missing. The fork defers model ordering entirely. Optional verifier diagnostics require their own complete cohorts and remain authoring aids that do not change grades, gates, or ordinary queues. See the [diagnostic contract](commands.md#optional-verifier-diagnostics).

The upstream identity inventory omitted fork dependencies that determine completion, observations, and content digests. PR39 derives the packaged executable dependency closure and registry bindings from current fork source. It moves current code into `skill_inputs`, `process_runtime`, `trace_normalization`, `agent_clis`, and `trigger_identity`. The reviewers differed on the value of broader extraction. The accepted unit preserves current policy owners and adopts the narrower trigger goal. Saved identity schema 4 requires regeneration of both schema 3 comparison arms. Matching saved reports need not match today's source bytes. See [upgrade guidance](upgrading.md) and [identity tests](../tests/test_trigger_identity.py).

## Accepted implementation sequence

All twenty-eight recorded PR and main CI runs completed successfully at the corresponding head or merge below. Review links identify the accepted verdict, including PR32's `PASS+NOTES`. The Windows job is a bounded text-contract lane. It does not execute the whole product. PR26 retains merge ancestry. PR27 through PR39 landed as separate squash units.

| Fork PR and accepted behavior | Reviewed head | Merge | Independent review | PR CI | Main CI |
| --- | --- | --- | --- | --- | --- |
| [PR26](https://github.com/mdsmithaustin/skill-eval-harness/pull/26) Sync, recovery, and path repairs | `a80331c7bdf007b46e65742c4c105a42fd433de3` | `95054f1166d84cb9434ae34376bd617546ce9078` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/26#issuecomment-6091396842) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38007255008) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38007612717) |
| [PR27](https://github.com/mdsmithaustin/skill-eval-harness/pull/27) Measured Pi retries | `4152988e98ee060230faa8c9f6a0d4103c5d4ba8` | `1b2ad0dab70bd31fb9c82ffb6802db8a9aa8eafe` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/27#issuecomment-6091444773) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38007728872) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38007995913) |
| [PR28](https://github.com/mdsmithaustin/skill-eval-harness/pull/28) Offline trace and discovery journeys | `aee4b3d4bec72306614aa64096ea32a208b06633` | `60f711fadc10f3bb86537d284df92013a29d8a26` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/28#issuecomment-6091527788) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38008362698) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38008651995) |
| [PR29](https://github.com/mdsmithaustin/skill-eval-harness/pull/29) Provider and telemetry conformance | `83da648e3b83091fa6569eb25e91cffef0858791` | `3afd3c136117d6da6c94959cae01e4925dc43d07` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/29#issuecomment-6091624990) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38009032081) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38009541827) |
| [PR30](https://github.com/mdsmithaustin/skill-eval-harness/pull/30) Native answer spend admission | `d974199589f8186761b4ab47fd4fc8920aa182c5` | `cb1d75e3dc9dab6e87cee9be85158782c8e0a05d` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/30#issuecomment-6091891163) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38010943761) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38011286739) |
| [PR31](https://github.com/mdsmithaustin/skill-eval-harness/pull/31) Subagent spend and Claude prices | `b99e39ee80cef08d5f5db9ba5b9432ef8d224b9f` | `b265928bffb342c11f2655afc96564cec2c196bd` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/31#issuecomment-6093589874) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38022325188) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38022896161) |
| [PR32](https://github.com/mdsmithaustin/skill-eval-harness/pull/32) Natural-exit empty stderr capture | `504dc8533f179ac4baa74aa6fed93838efa42498` | `5c0aca0d440b3c0f64c230d8ea12d1f6d2c2e5f1` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/32#issuecomment-6094374117) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38028902728) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38029122110) |
| [PR33](https://github.com/mdsmithaustin/skill-eval-harness/pull/33) Judge observations and spend | `5c56498786eaa39fcc8385439c0a5cc3275afc01` | `22d98c99b973c1afbed4c0c2b557c865eac186ee` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/33#issuecomment-6097863286) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38054823059) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38055101721) |
| [PR34](https://github.com/mdsmithaustin/skill-eval-harness/pull/34) Jetty attempt admission | `a9809865984317075300c137a390339844c545d7` | `1ba23e4459fa9c9067431895884230fb15965c2e` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/34#issuecomment-6105958255) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38115857675) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38116178050) |
| [PR35](https://github.com/mdsmithaustin/skill-eval-harness/pull/35) Matrix admission and retained reports | `1d25beb7c1a1a9df13774989078b67d8c4680fe5` | `9dbf6ee7ac8cc4a9fccce68712674a64908a96c2` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/35#issuecomment-6106611539) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38120963622) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38121430061) |
| [PR36](https://github.com/mdsmithaustin/skill-eval-harness/pull/36) Simulated callable outputs | `97975c7cfd86a7bf050a9ec62e73e79b296736c1` | `82233a85158b900bd76a5f1486d2c33554f5f2cb` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/36#issuecomment-6107099762) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38124611942) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38124957024) |
| [PR37](https://github.com/mdsmithaustin/skill-eval-harness/pull/37) Bounded offline dialogue | `bace6dca5f5b0717d1befdaa45e4a4edd5e3c48d` | `a97091845362d66a44e4be4f44f05a8185afa330` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/37#issuecomment-6107947014) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38131102381) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38131478083) |
| [PR38](https://github.com/mdsmithaustin/skill-eval-harness/pull/38) Optional verifier diagnostics | `f37545e28440898b29b2aa825461250f0e4d63ca` | `9f9c7cf7e12ac46be4402d8b8ee985defe316d3b` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/38#issuecomment-6108393719) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38134515229) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38134914531) |
| [PR39](https://github.com/mdsmithaustin/skill-eval-harness/pull/39) Canonical trigger owners and identity | `e646c2a84bcbd91e85ac30407dec9ed509b0f20f` | `8a85843da0d0914a8bc19044d5976eb71774ff7d` | [Review](https://github.com/mdsmithaustin/skill-eval-harness/pull/39#issuecomment-6109421198) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38142482835) | [Run](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38143022884) |

The current trigger [AUTHOR](https://github.com/mdsmithaustin/skill-eval-harness/pull/39#issuecomment-6109421025) and [independent PASS](https://github.com/mdsmithaustin/skill-eval-harness/pull/39#issuecomment-6109421198) describe the accepted implementation. Local full-suite evidence at that reviewed head reports 2,045 tests with 18 documented skips. The installed-wheel checker imports 39 packaged modules, runs three entrypoints, generates four paired Stub reports, compares both pairs, and runs recorded Pi, Claude, Codex, and Vibe children. These offline fixtures establish contracts, not live provider behavior.

## Historical failures and unresolved observations

The sync's unresolved-index inventory failure, recovery race, path diagnostic regression, interpreter selection, shifted doc references, and missing-pip attempt remain in the [dated record](upstream-syncs/2026-10-09.md). Its [failed path review](https://github.com/mdsmithaustin/skill-eval-harness/pull/26#issuecomment-6091233855) precedes the [repaired pass](https://github.com/mdsmithaustin/skill-eval-harness/pull/26#issuecomment-6091396842).

The shared Claude price change first promoted timeout floors to whole prices. A second review found stale legacy price aliases in generated consensus parents. Both [the timeout failure](https://github.com/mdsmithaustin/skill-eval-harness/pull/31#issuecomment-6092831760) and [the legacy-price failure](https://github.com/mdsmithaustin/skill-eval-harness/pull/31#issuecomment-6093417943) remain public alongside the [accepted repair](https://github.com/mdsmithaustin/skill-eval-harness/pull/31#issuecomment-6093589874). Three unintended one-second host Claude launches occurred when a test patched the wrong seam. Their network activity, billing, independent process cleanup, and provider-artifact effects remain unknown. Later offline passes do not resolve them.

Judge, Jetty, matrix, callable, bridge, and trigger units also had failed checkpoints before acceptance. Jetty review reproduced unusable selected rows submitting before import rejection and invalid successor histories permitting submission. Matrix verification retained earlier failures and a temporary-path prompt failure. Callable review found a dropped count alias and a command-population error filter, followed by three failing regressions and a repair. Bridge review requested full-frame bounds before output receipt serialization and input selection before error availability. Its [first Windows CI](https://github.com/mdsmithaustin/skill-eval-harness/actions/runs/38129221939) failed type checking. The [repaired bridge AUTHOR](https://github.com/mdsmithaustin/skill-eval-harness/pull/37#issuecomment-6107946446), [fresh pass](https://github.com/mdsmithaustin/skill-eval-harness/pull/37#issuecomment-6107947014), and [acceptance](https://github.com/mdsmithaustin/skill-eval-harness/pull/37#issuecomment-6108040380) describe the later result. Trigger review found that a syntax error during source-change detection discarded retained observations. The [final pass](https://github.com/mdsmithaustin/skill-eval-harness/pull/39#issuecomment-6109421198) records its repair.

Overwritten initial command records and missing historical exit statuses were not recovered. The final Jetty verifier cannot restore the first 22 commands' lost identities or scratch inventory. Legacy short rows in the canonical decision TSV lack evidence or result cells. Append-only clarifications preserve those absent fields as unknown. The maintenance trail review passes through row 196 after those clarifications. That verdict covers the examined history and does not approve this later documentation update.

Direct callback tests do not prove consumption by built-in backends. The offline bridge remains Linux-only. Recorded prices do not establish provider eligibility or live billing finality. Ledger persistence failures remain unresolved accounting outcomes. Source and saved-record hashes check consistency and reproducibility, not authentication. Connector HTTP 403 failures were integration permission failures, not observed automatic approval-review rejections.

## Reproduce the current contracts

Use the [contributor setup](../CONTRIBUTING.md#local-setup) and its pinned test dependencies. These commands run from the checkout root.

```sh
python3 -m unittest discover -s tests -p 'test_doc*.py' -v
python3 skill_benchmark.py --help
python3 run_pi_trigger_eval.py --help
python3 run_trigger_matrix.py --help
python3 scripts/check_installed_wheel.py
```

The documentation command fails on a nonzero exit, a zero-test summary, stale source references, broken relative links, or contradicted facts. Each help command fails on a nonzero exit or missing usage output. The wheel checker fails on a nonzero exit or a missing module, entrypoint, report comparison, or required-byte rejection check. Its installed commands run outside the checkout. The complete authoritative gate list remains in [CONTRIBUTING](../CONTRIBUTING.md#validation) and [CI](../.github/workflows/ci.yml).

The [offline journeys](../tests/test_offline_journeys.py), [backend conformance](../tests/test_backend_conformance.py), [telemetry conservation](../tests/test_telemetry_domain.py), [spend tests](../tests/test_spend_contracts.py), [bridge tests](../tests/test_tool_bridge.py), and [identity tests](../tests/test_trigger_identity.py) are clone-accessible proof paths. Public PR review comments and Actions runs supply acceptance evidence. Private transcripts and raw provider artifacts are operator records and are not required to read this summary.

## Preservation and remaining cleanup

The operator verified a local 63-head Git bundle, an 81,589-file evidence archive with sixteen FIFO entries, a same-session transcript prefix, and a pre-cleanup supplement. These are local preservation records rather than clone-accessible proof. Actual older cleanup removed 114 task worktrees, 26 local task refs, 13 remote task refs, and the exact stopped task sandbox after ownership and content checks. Ten finished dirty proposals had all 32 working files preserved. The sync tag and unrelated resources remain. The user's main `uv.lock` is unchanged at SHA-256 `fd02fb6c2f5d12889e274a2271558ea09cf07997552e01b52b23e65b9400b011`. The final writer, fresh independent docs reviewer, and trail-review resources remain pending until documentation acceptance. Final archival supplements and removal of those resources require a later actual receipt.
