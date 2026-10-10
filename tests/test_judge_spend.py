import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import make_eval_repo, run_cli, trace_event, write_run

import skill_benchmark as sb
from manifest_contracts import RunCoordinate
from runner_contracts import OutcomeContext, Provider
from spend_contracts import (
    AnswerCall,
    JudgeCall,
    Refused,
    SpendLedger,
    SpendPlan,
    SpendPolicy,
    SpendStopReason,
    SubagentTurnCall,
)


class JudgeSpendTests(unittest.TestCase):
    def batch(self, root, *, verdict=None, cost=0.6, assertion=None, events=None, mutation=""):
        assertion = {"name": "quality", "type": "judge", "rubric": ["Be correct."], "gate": True, **(assertion or {})}
        manifest = make_eval_repo(root, cases=[{
            'id': 'case-1', 'split': 'tune', 'prompt': 'Do the task.',
            'assertions': [assertion],
        }])
        runs = root / 'runs'
        write_run(runs / 'case-1' / 'with_skill', 'candidate', events=events)
        marker = root / 'launches'
        stub = root / 'judge-child'
        answer = verdict if isinstance(verdict, str) else json.dumps(verdict if verdict is not None else {'passed': True})
        payload = {'type': 'result', 'result': answer,
                   'usage': {'input_tokens': 3, 'output_tokens': 4}}
        if cost is not None:
            payload['total_cost_usd'] = cost
        stub.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
prompt = sys.stdin.read()
with Path({str(marker)!r}).open('a') as handle:
    handle.write('started\\n')
with Path({str(root / 'prompts.jsonl')!r}).open('a') as handle:
    handle.write(json.dumps(prompt) + '\\n')
{mutation}
print('deterministic judge stderr', file=sys.stderr)
print({json.dumps(payload)!r} if '--output-format' in sys.argv else {answer!r})
''')
        stub.chmod(0o755)
        return manifest, runs, stub, marker

    def invoke(self, root, route='native', *flags):
        manifest, runs, stub, marker = self.batch(root)
        out = root / 'results.jsonl'
        route_flags = (('--judge-cmd', str(stub)) if route == 'shell' else
                       ('--judge-backend', 'codex', '--codex-cmd', str(stub)) if route == 'codex' else
                       ('--judge-model', 'judge-a', '--claude-bin', str(stub)))
        code, stdout, stderr = run_cli('judge', manifest, '--runs', runs,
                                     '--out', out, '--variant', 'with_skill', *route_flags, *flags)
        return code, stdout, stderr, out, runs, marker

    def test_zero_cap_preserves_ready_results_without_starting(self):
        for route in ('native', 'shell'):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                code, _, stderr, out, runs, marker = self.invoke(
                    Path(td), route, '--max-cost-usd', '0', '--judge-runs', '2')
                self.assertEqual(code, 2, stderr)
                self.assertTrue(out.is_file(), stderr)
                row = json.loads(out.read_text())
                self.assertEqual(row['availability'], 'partial')
                self.assertEqual(len(row['judge_runs']), 2)
                self.assertEqual([item['spend_refusal_reason'] for item in row['judge_runs']],
                                 ['cost_ceiling', 'cost_ceiling'])
                self.assertFalse(marker.exists())
                ledgers = list((runs / 'spend').glob('*/spend-ceiling.json'))
                self.assertEqual(len(ledgers), 1)
                self.assertEqual([item['state'] for item in json.loads(ledgers[0].read_text())['calls']],
                                 ['not_started', 'not_started'])

    def execute(self, root, setup, *, route='native', flags=()):
        manifest, runs, stub, _marker = setup
        out = root / 'results.jsonl'
        route_flags = (('--judge-cmd', str(stub)) if route == 'shell' else
                       ('--judge-backend', 'codex', '--codex-cmd', str(stub)) if route == 'codex' else
                       ('--judge-model', 'judge-a', '--claude-bin', str(stub)))
        code, stdout, stderr = run_cli('judge', manifest, '--runs', runs,
                                     '--out', out, '--variant', 'with_skill', *route_flags, *flags)
        rows = [json.loads(line) for line in out.read_text().splitlines()] if code in (0, 2) and out.exists() else []
        return code, rows, stdout, stderr

    def ledger(self, runs):
        paths = list((runs / 'spend').glob('*/spend-ceiling.json'))
        self.assertEqual(len(paths), 1)
        raw = json.loads(paths[0].read_text())
        self.assertEqual(SpendLedger.from_dict(raw).as_dict(), raw)
        return raw

    def test_paid_crossing_retains_every_panel_repeat(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root)
            code, rows, _, stderr = self.execute(root, setup, flags=(
                '--max-cost-usd', '0.5', '--judge-runs', '2',
                '--judge-panel', 'judge-a', '--judge-panel', 'judge-b', '--quorum', '2'))
            self.assertEqual(code, 2, stderr)
            self.assertEqual(setup[3].read_text(), 'started\n')
            self.assertEqual(rows[0]['judge_models'], ['judge-a', 'judge-b'])
            self.assertEqual([len(member['judge_runs']) for member in rows[0]['judge_panel']], [2, 2])
            self.assertFalse(rows[0]['passed'])
            self.assertIsNone(rows[0]['cost_usd'])
            self.assertEqual(rows[0]['cost_aggregate']['USD']['known_subtotal'], '0.6')
            self.assertEqual(self.ledger(setup[1])['spent_usd'], '0.6')
            leaves = [leaf for member in rows[0]['judge_panel'] for leaf in member['judge_runs']]
            self.assertEqual([leaf.get('returncode') for leaf in leaves], [0, None, None, None])
            self.assertEqual([item['state'] for item in self.ledger(setup[1])['calls']],
                             ['settled', 'not_started', 'not_started', 'not_started'])

    def test_missing_cost_zero_and_assumptions_have_distinct_results(self):
        for cost, assumption, expected_code, expected_launches, basis, spent in (
                (None, None, 2, 1, 'unpriced', '0'),
                (0, None, 0, 2, 'observed', '0.0'),
                (None, '0.3', 2, 1, 'assumed', '0.3')):
            with self.subTest(cost=cost, assumption=assumption), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                setup = self.batch(root, cost=cost)
                flags = ('--max-cost-usd', '0.2', '--judge-runs', '2')
                if assumption:
                    flags += ('--assumed-cost-per-run-usd', assumption)
                code, rows, _, stderr = self.execute(root, setup, flags=flags)
                self.assertEqual(code, expected_code, stderr)
                self.assertEqual(setup[3].read_text(), 'started\n' * expected_launches)
                ledger = self.ledger(setup[1])
                self.assertEqual(ledger['calls'][0]['charge']['basis'], basis)
                self.assertEqual(ledger['spent_usd'], spent)
                first = rows[0]['judge_runs'][0]
                self.assertEqual(first['cost_usd'], cost)
                self.assertNotIn('assumed_cost_per_run_usd', first)

    def test_saved_per_step_consensus_checks_each_leaf_and_preserves_parent_decision(self):
        events = {'events': [trace_event('command', name='Bash', input_summary='echo one'),
                             trace_event('command', name='Bash', input_summary='echo two')]}
        for panel in (False, True):
            with self.subTest(panel=panel), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                setup = self.batch(root, assertion={'per_step': True}, events=events,
                    verdict={'criteria': [{'name': 'step-1', 'met': True}, {'name': 'step-2', 'met': True}]})
                flags = ('--judge-runs', '2')
                if panel:
                    flags += ('--judge-panel', 'judge-a', '--judge-panel', 'judge-b', '--quorum', '1')
                    negative = {'type': 'result', 'total_cost_usd': 0.6,
                                'result': json.dumps({'criteria': [{'name': 'step-1', 'met': False},
                                                                  {'name': 'step-2', 'met': False}]})}
                    setup[2].write_text(setup[2].read_text().replace("print('deterministic judge stderr', file=sys.stderr)",
                        f"if '--model' in sys.argv and sys.argv[sys.argv.index('--model') + 1] == 'judge-b':\n"
                        f"    print({json.dumps(negative)!r})\n    raise SystemExit(0)\n"
                        "print('deterministic judge stderr', file=sys.stderr)"))
                code, rows, _, stderr = self.execute(root, setup, flags=flags)
                self.assertEqual(code, 0, stderr)
                self.assertNotIn('criteria', rows[0])
                report = root / 'benchmark.json'

                def benchmark(setup=setup, root=root, report=report):
                    code, _, stderr = run_cli('benchmark', setup[0], '--runs', setup[1],
                        '--variant', 'with_skill', '--judge-results', root / 'results.jsonl', '--out', report)
                    self.assertEqual(code, 0, stderr)
                    return json.loads(report.read_text())['results'][0]

                graded = benchmark()
                self.assertEqual((graded['deferred_judge_tasks'], graded['qualitative_passed']), (0, 1))
                self.assertEqual(graded['qualitative_assertions'][0]['score'], 0.5 if panel else 1.0)
                leaf = rows[0].get('judge_panel', [rows[0]])[0]['judge_runs'][0]
                leaf['minimum_criteria'] = 1
                (root / 'results.jsonl').write_text(json.dumps(rows[0]) + '\n')
                self.assertEqual(benchmark()['deferred_judge_tasks'], 1)

    def test_known_missing_shell_and_native_routes_reject_before_writes(self):
        for route in ('shell', 'codex', 'gemini', 'vibe'):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                setup = self.batch(root)
                out = root / 'results.jsonl'
                out.write_text('preserved')
                if route == 'shell':
                    code, _, _, stderr = self.execute(root, setup, route=route,
                        flags=('--max-cost-usd', '1'))
                else:
                    code, _, stderr = run_cli('judge', setup[0], '--runs', setup[1],
                        '--out', out, '--variant', 'with_skill', '--judge-backend', route,
                        '--max-cost-usd', '1')
                self.assertEqual(code, 1, stderr)
                self.assertIn('requires --assumed-cost-per-run-usd', stderr)
                self.assertEqual(out.read_text(), 'preserved')
                self.assertFalse(setup[3].exists())
                self.assertFalse((setup[1] / 'spend').exists())

    def test_paid_malformed_and_transcript_failure_keep_settled_price(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root, verdict='not a verdict')
            code, rows, _, stderr = self.execute(root, setup, flags=(
                '--max-cost-usd', '0.5', '--judge-runs', '2'))
            self.assertEqual(code, 2, stderr)
            self.assertIsNone(rows[0]['cost_usd'])
            self.assertEqual(rows[0]['cost_aggregate']['USD']['known_subtotal'], '0.6')
            self.assertFalse(rows[0]['judge_runs'][0]['judge_observation_complete'])
            self.assertEqual(self.ledger(setup[1])['calls'][0]['charge']['amount_usd'], '0.6')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root)
            transcripts = root / 'transcripts'
            transcripts.write_text('not a directory')
            with self.assertRaises(OSError):
                self.execute(root, setup, flags=('--max-cost-usd', '1', '--transcripts', str(transcripts)))
            self.assertEqual(setup[3].read_text(), 'started\n')
            self.assertEqual(self.ledger(setup[1])['spent_usd'], '0.6')

    def test_refusals_validate_each_verdict_kind_in_both_merges(self):
        events = {'events': [trace_event('command', name='Bash', input_summary='echo ok')]}
        shapes = [
            ({'atLeast': 0.7}, {'score': 1}, 'scored'),
            ({'score_scale': [1, 5], 'threshold': 4}, {'score': 5}, 'scored'),
            ({'dynamic_rubric': {'minimum_criteria': 3, 'instruction': 'Find concrete criteria.'}},
             {'criteria': [{'name': str(i), 'met': True} for i in range(3)]}, 'dynamic'),
            ({'graded_dimensions': [{'name': 'clarity', 'rubric': 'Be clear.'}]},
             {'dimension_scores': {'clarity': 5}}, 'dimensions'),
            ({'per_step': True}, {'criteria': [{'name': 'step-1', 'met': True}]}, 'dynamic'),
        ]
        for assertion, verdict, kind in shapes:
            for repeats in (1, 2):
                with self.subTest(kind=kind, assertion=assertion, repeats=repeats), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    setup = self.batch(root, assertion=assertion, verdict=verdict, events=events)
                    code, rows, _, stderr = self.execute(root, setup, flags=(
                        '--max-cost-usd', '0.5', '--judge-runs', str(repeats),
                        '--judge-panel', 'judge-a', '--judge-panel', 'judge-b'))
                    self.assertEqual(code, 2, stderr)
                    self.assertEqual(rows[0]['availability'], 'partial')
                    self.assertFalse(rows[0]['passed'])
                    panel = rows[0]['judge_panel']
                    leaves = [leaf for member in panel for leaf in member.get('judge_runs', [member])]
                    self.assertEqual(len(leaves), 2 * repeats)
                    self.assertEqual(leaves[0]['verdict_kind'], kind)
                    self.assertEqual([leaf['judge_observation_kind'] for leaf in leaves[1:]],
                                     ['missing'] * (2 * repeats - 1))
                    for leaf in leaves[1:]:
                        self.assertEqual(leaf['invocation_state'], 'not_started')
                        self.assertNotIn('returncode', leaf)
                        self.assertIsNotNone(sb.judge_observation_incomplete_reason(leaf))

    def test_all_five_guards_run_before_cost_support_and_do_not_spend(self):
        events = {'events': [trace_event('command', name='Bash', input_summary='echo ok')]}
        for guard, expected, complete in (
                ('missing', 'per-step judge requires trajectory evidence', False),
                ('empty', 'no completed trajectory steps', True),
                ('steps', 'trajectory changed after task creation', False),
                ('input', 'input changed after task creation', False),
                ('trajectory', 'requested judge trajectory evidence is incomplete', False)):
            for route in ('shell', 'native', 'codex'):
                for cap in ('0', '1'):
                    with self.subTest(guard=guard, route=route, cap=cap), tempfile.TemporaryDirectory() as td:
                        root = Path(td)
                        setup = self.batch(root, events=events)
                        task = sb.collect_judge_tasks(setup[0], setup[1], variants=['with_skill'])[0]
                        if guard in ('missing', 'empty', 'steps'):
                            task['assertion']['per_step'] = True
                            task.pop('judge_input_sha256', None)
                            if guard == 'steps':
                                task['trajectory_steps_sha256'] = 'bad-fingerprint'
                            else:
                                path = Path(task['run_base']) / 'events.json'
                                if guard == 'missing':
                                    path.unlink()
                                else:
                                    path.write_text(json.dumps({'events': []}))
                        elif guard == 'input':
                            task['judge_input_sha256'] = 'sha256:' + '0' * 64
                        else:
                            (Path(task['run_base']) / 'events.json').unlink()
                        flags = ('--max-cost-usd', cap, '--judge-runs', '2')
                        if guard == 'trajectory':
                            flags += ('--judge-trajectory',)
                        with mock.patch.object(sb, 'collect_judge_tasks', return_value=[task]):
                            code, rows, _, stderr = self.execute(root, setup, route=route, flags=flags)
                        self.assertEqual(code, 0 if complete else 2, stderr)
                        self.assertIn(expected, rows[0]['judge_runs'][0]['evidence'])
                        self.assertEqual(rows[0]['judge_observation_complete'], complete)
                        self.assertEqual(len(rows[0]['judge_runs']), 2)
                        self.assertEqual(self.ledger(setup[1])['calls'], [])
                        self.assertFalse(setup[3].exists())
                        for leaf in rows[0]['judge_runs']:
                            self.assertEqual(leaf['cost_normalized']['source'], 'not_applicable')
                            self.assertNotIn('invocation_state', leaf)
                            self.assertNotIn('observed_subtotal_usd', leaf)

    def test_mixed_guard_and_ready_zero_cap_preserves_both_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root)
            ready = sb.collect_judge_tasks(setup[0], setup[1], variants=['with_skill'])[0]
            guard = dict(ready, judge_task_id='guard', judge_input_sha256='sha256:' + '0' * 64)
            with mock.patch.object(sb, 'collect_judge_tasks', return_value=[guard, ready]):
                code, rows, _, stderr = self.execute(root, setup, flags=('--max-cost-usd', '0'))
            self.assertEqual(code, 2, stderr)
            self.assertEqual([row['judge_task_id'] for row in rows], ['guard', ready['judge_task_id']])
            self.assertNotIn('spend_refusal_reason', rows[0])
            self.assertEqual(rows[1]['spend_refusal_reason'], 'cost_ceiling')
            self.assertEqual(len(self.ledger(setup[1])['calls']), 1)

    def test_retained_input_survives_a_child_mutating_live_files(self):
        for per_step in (False, True):
            with self.subTest(per_step=per_step), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                run = root / 'runs' / 'case-1' / 'with_skill'
                mutation = f"Path({str(run / 'output.md')!r}).write_text('changed candidate')\nPath({str(run / 'events.json')!r}).write_text('{{}}')"
                setup = self.batch(root, assertion={'per_step': True} if per_step else {},
                    verdict={'criteria': [{'name': 'step-1', 'met': True}]} if per_step else {'passed': True},
                    events={'events': [trace_event('command', name='Bash', input_summary='echo ok')]}, mutation=mutation)
                code, rows, _, stderr = self.execute(root, setup, flags=(
                    '--max-cost-usd', '10', '--judge-runs', '2', '--judge-trajectory'))
                self.assertEqual(code, 0, stderr)
                prompts = [json.loads(line) for line in (root / 'prompts.jsonl').read_text().splitlines()]
                self.assertEqual(prompts[0], prompts[1])
                self.assertIn('candidate', prompts[0])
                self.assertNotIn('changed candidate', prompts[1])
                self.assertEqual(len(rows[0]['judge_runs']), 2)
                self.assertTrue(rows[0]['judge_observation_complete'])
                self.assertEqual(rows[0]['judge_runs'][0]['judge_input_sha256'], rows[0]['judge_runs'][1]['judge_input_sha256'])

    def test_exploration_cleanup_failure_follows_paid_settlement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root)
            original = tempfile.TemporaryDirectory
            class FailedCallCleanup(original):
                def cleanup(self):
                    super().cleanup()
                    if Path(self.name).name.startswith('judge-call-'):
                        raise OSError('invocation scratch cleanup failed')
            with mock.patch('skill_benchmark.tempfile.TemporaryDirectory', FailedCallCleanup):
                with self.assertRaisesRegex(OSError, 'invocation scratch cleanup failed'):
                    self.execute(root, setup, flags=('--judge-explore', '--max-cost-usd', '10'))
            self.assertEqual(setup[3].read_text(), 'started\n')
            ledger = self.ledger(setup[1])
            self.assertEqual(ledger['spent_usd'], '0.6')
            self.assertEqual(ledger['calls'][0]['state'], 'settled')
            self.assertEqual(ledger['calls'][0]['charge']['basis'], 'observed')

    def test_text_only_does_not_read_unconsumed_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root)
            (root / 'runs' / 'case-1' / 'with_skill' / 'unrelated').symlink_to(root / 'missing')
            with mock.patch('shutil.copytree', side_effect=AssertionError('unrelated files copied')):
                code, rows, _, stderr = self.execute(root, setup, flags=('--max-cost-usd', '10'))
            self.assertEqual(code, 0, stderr)
            self.assertTrue(rows[0]['passed'])

    def test_exploration_retains_source_binding_and_stricter_provider_copy(self):
        for mode, cap in (('explore', None), ('explore', '10'), ('explore', '0'), ('trajectory+explore', '10')):
            with self.subTest(mode=mode, cap=cap), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                setup = self.batch(root, events={'events': [trace_event('command', name='Bash', input_summary='echo ok')]})
                run = setup[1] / 'case-1' / 'with_skill'
                (run / 'notes.txt').write_text('retained notes')
                (run / 'CLAUDE.md').write_text('private agent context')
                (run / 'gold.json').write_text('oracle')
                (run / 'innocent.txt').symlink_to(run / 'gold.json')
                (run / '.claude').mkdir()
                (run / '.claude' / 'settings.json').write_text('{}')
                (run / 'grading-files').mkdir()
                (run / 'grading-files' / 'innocent.txt').write_text('excluded directory')
                task = sb.collect_judge_tasks(setup[0], setup[1], variants=['with_skill'])[0]
                expected_binding = sb.judge_input_material(task, 'candidate', evidence_mode=mode, run_base=run)
                expected_source_hash = expected_binding[3]
                setup[2].write_text(setup[2].read_text().replace('prompt = sys.stdin.read()',
                    f"""prompt = sys.stdin.read()
import os
with Path({str(root / 'explore-probes.jsonl')!r}).open('a') as handle:
    handle.write(json.dumps({{'cwd': os.getcwd(), 'listing': sorted(str(p.relative_to(Path.cwd())) for p in Path.cwd().rglob('*')), 'notes': Path('notes.txt').read_text()}}) + '\\n')
Path({str(run / 'notes.txt')!r}).write_text('live mutation')
"""))
                flags = ('--judge-runs', '2', '--judge-explore')
                if mode == 'trajectory+explore':
                    flags += ('--judge-trajectory',)
                if cap is not None:
                    flags += ('--max-cost-usd', cap)
                scratch_paths = []
                real_temporary = tempfile.TemporaryDirectory
                def owned_temporary(*args, real_temporary=real_temporary, scratch_paths=scratch_paths, **kwargs):
                    directory = real_temporary(*args, **kwargs)
                    scratch_paths.append(Path(directory.name))
                    return directory
                with mock.patch('skill_benchmark.tempfile.TemporaryDirectory', side_effect=owned_temporary):
                    code, rows, _, stderr = self.execute(root, setup, flags=flags)
                self.assertTrue(scratch_paths)
                self.assertEqual([path.exists() for path in scratch_paths], [False] * len(scratch_paths))
                self.assertEqual(code, 2 if cap == '0' else 0, stderr)
                leaves = rows[0]['judge_runs']
                self.assertEqual([leaf['judge_context_sha256'] for leaf in leaves], [expected_source_hash] * 2)
                self.assertEqual([leaf['judge_input_sha256'] for leaf in leaves], [expected_binding[0]] * 2)
                if cap == '0':
                    self.assertFalse(setup[3].exists())
                else:
                    probes = [json.loads(line) for line in (root / 'explore-probes.jsonl').read_text().splitlines()]
                    self.assertEqual([probe['notes'] for probe in probes], ['retained notes'] * 2)
                    self.assertNotEqual(probes[0]['cwd'], probes[1]['cwd'])
                    for probe in probes:
                        self.assertEqual(probe['listing'], ['events.json', 'notes.txt', 'output.md'])
                        self.assertFalse(Path(probe['cwd']).exists())
                    prompts = [json.loads(line) for line in (root / 'prompts.jsonl').read_text().splitlines()]
                    self.assertEqual(prompts, [expected_binding[1]] * 2)
                    self.assertNotIn('judge-explore-', prompts[0])
                    self.assertNotIn('--max-cost-usd', prompts[0])

    def test_each_exploring_repeat_and_panel_member_gets_retained_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root)
            run = setup[1] / 'case-1' / 'with_skill'
            (run / 'notes.txt').write_text('retained notes')
            task = sb.collect_judge_tasks(setup[0], setup[1], variants=['with_skill'])[0]
            expected = sb.judge_input_material(task, 'candidate', evidence_mode='explore', run_base=run)
            setup[2].write_text(setup[2].read_text().replace('prompt = sys.stdin.read()',
                f"""prompt = sys.stdin.read()
import os
with Path({str(root / 'mutation-probes.jsonl')!r}).open('a') as handle:
    handle.write(json.dumps({{'cwd': os.getcwd(), 'notes': Path('notes.txt').read_text()}}) + '\\n')
Path('notes.txt').write_text('provider mutation')
"""))
            code, rows, _, stderr = self.execute(root, setup, flags=(
                '--judge-explore', '--judge-runs', '2', '--max-cost-usd', '10',
                '--judge-panel', 'judge-a', '--judge-panel', 'judge-b'))
            self.assertEqual(code, 0, stderr)
            self.assertEqual(setup[3].read_text(), 'started\n' * 4)
            probes = [json.loads(line) for line in (root / 'mutation-probes.jsonl').read_text().splitlines()]
            self.assertEqual([probe['notes'] for probe in probes], ['retained notes'] * 4)
            self.assertEqual(len({probe['cwd'] for probe in probes}), 4)
            self.assertFalse(any(Path(probe['cwd']).exists() for probe in probes))
            self.assertEqual((run / 'notes.txt').read_text(), 'retained notes')
            leaves = [leaf for member in rows[0]['judge_panel'] for leaf in member['judge_runs']]
            self.assertEqual([leaf['judge_input_sha256'] for leaf in leaves], [expected[0]] * 4)
            self.assertEqual([leaf['judge_context_sha256'] for leaf in leaves], [expected[3]] * 4)
            prompts = [json.loads(line) for line in (root / 'prompts.jsonl').read_text().splitlines()]
            self.assertEqual(prompts, [expected[1]] * 4)

    def test_capped_and_uncapped_effective_material_and_result_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root, assertion={'atLeast': 0.7}, verdict={'score': 0.8},
                               events={'events': [trace_event('command', name='Bash', input_summary='echo ok')]})
            for mode_flags in ((), ('--judge-trajectory',), ('--judge-explore',),
                               ('--judge-trajectory', '--judge-explore')):
                with self.subTest(mode=mode_flags):
                    code, uncapped, _, stderr = self.execute(root, setup, flags=mode_flags)
                    self.assertEqual(code, 0, stderr)
                    code, capped, _, stderr = self.execute(root, setup, flags=(*mode_flags, '--max-cost-usd', '10'))
                    self.assertEqual(code, 0, stderr)
                    self.assertEqual(capped, uncapped)
                    prompts = [json.loads(line) for line in (root / 'prompts.jsonl').read_text().splitlines()]
                    self.assertEqual(prompts[-1], prompts[-2])
                    self.assertTrue(capped[0]['passed'])
                    self.assertEqual(capped[0]['threshold'], 0.7)

    def test_exploration_scratch_cleans_on_publication_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root)
            setup[2].write_text(setup[2].read_text().replace('prompt = sys.stdin.read()',
                f"import os\nPath({str(root / 'cwd')!r}).write_text(os.getcwd())\nprompt = sys.stdin.read()"))
            transcripts = root / 'transcripts'
            transcripts.write_text('not a directory')
            with self.assertRaises(OSError):
                self.execute(root, setup, flags=('--judge-explore', '--max-cost-usd', '10', '--transcripts', str(transcripts)))
            self.assertFalse(Path((root / 'cwd').read_text()).exists())
            self.assertEqual(self.ledger(setup[1])['spent_usd'], '0.6')

    def test_duplicate_ready_identities_fail_before_ledger_and_output_write(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            setup = self.batch(root)
            task = sb.collect_judge_tasks(setup[0], setup[1], variants=['with_skill'])[0]
            out = root / 'results.jsonl'
            out.write_text('preserved')
            with mock.patch.object(sb, 'collect_judge_tasks', return_value=[task, task]):
                with self.assertRaisesRegex(ValueError, 'duplicate call identities'):
                    self.execute(root, setup, flags=('--max-cost-usd', '1'))
            self.assertEqual(out.read_text(), 'preserved')
            self.assertFalse((setup[1] / 'spend').exists())
            self.assertFalse(setup[3].exists())

    def test_actual_timeout_floor_settles_before_later_refusal(self):
        for assumption, basis in ((None, 'unpriced'), ('0.1', 'assumed')):
            with self.subTest(assumption=assumption), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                setup = self.batch(root)
                setup[2].write_text(setup[2].read_text() + '\nimport time\nsys.stdout.flush()\ntime.sleep(30)\n')
                real_invoke = sb.claude_cli_invoke
                def timed_invoke(prompt, real_invoke=real_invoke, **options):
                    return real_invoke(prompt, timeout=1, **options)
                flags = ('--max-cost-usd', '0.5', '--judge-runs', '2')
                if assumption:
                    flags += ('--assumed-cost-per-run-usd', assumption)
                with mock.patch.object(sb, 'claude_cli_invoke', side_effect=timed_invoke):
                    code, rows, _, stderr = self.execute(root, setup, flags=flags)
                self.assertEqual(code, 2, stderr)
                self.assertEqual(setup[3].read_text(), 'started\n')
                leaf = rows[0]['judge_runs'][0]
                self.assertEqual(leaf['returncode'], 124)
                self.assertEqual(leaf['observed_subtotal_usd'], 0.6)
                self.assertIsNone(leaf['cost_usd'])
                ledger = self.ledger(setup[1])
                self.assertEqual(ledger['spent_usd'], '0.6')
                self.assertEqual(ledger['calls'][0]['charge']['basis'], basis)
                self.assertEqual(ledger['calls'][0]['charge']['observed_subtotal_usd'], '0.6')
                self.assertEqual(ledger['calls'][1]['state'], 'not_started')


class JudgeSpendIdentityTests(unittest.TestCase):
    def test_closed_judge_identity_roundtrip_and_invalid_fields(self):
        valid = {'judge_task_id': 'case::arm::run-1::quality',
                 'judge_input_sha256': 'sha256:' + 'a' * 64, 'backend': 'claude',
                 'requested_model': 'judge-a', 'repeat': 1}
        call = JudgeCall(**valid)
        self.assertEqual(call.as_dict(), {'kind': 'judge', **valid})
        ledger = SpendLedger.planned('a' * 32, SpendPolicy.from_raw('0'), SpendPlan((call,)))
        raw = ledger.as_dict()
        self.assertEqual(SpendLedger.from_dict(raw), ledger)
        for field, values in (
                ('judge_task_id', ['', ' ', None, 3, '\ud800']),
                ('judge_input_sha256', ['', 'a' * 64, 'sha256:' + 'A' * 64, None]),
                ('backend', ['answer', '', None, 1]),
                ('requested_model', ['', ' ', 1, '\ud800']),
                ('repeat', [0, -1, True, 1.5, '1'])):
            for value in values:
                with self.subTest(field=field, value=repr(value)), self.assertRaises((TypeError, ValueError)):
                    JudgeCall(**{**valid, field: value})
        for change in ('identity', 'duplicate', 'extra'):
            bad = json.loads(json.dumps(raw))
            if change == 'identity':
                bad['calls'][0]['call']['repeat'] = 2
            elif change == 'duplicate':
                bad['calls'].append(bad['calls'][0])
            else:
                bad['calls'][0]['call']['guard'] = 'not-a-paid-call'
            with self.subTest(change=change), self.assertRaises(ValueError):
                SpendLedger.from_dict(bad)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            SpendPlan((call, call))

    def test_subagent_artifact_constructor_keeps_turn_identity_closed(self):
        coordinate = RunCoordinate.of('case-1', 'with_skill', 1)
        turn = SubagentTurnCall('sha256:' + 'a' * 64, coordinate, 1)
        context = OutcomeContext(provider=Provider.SUBAGENT, metadata_extra={
            'answer_task_sha256': 'sha256:' + 'a' * 64, 'case_id': 'case-1',
            'variant': 'with_skill', 'run_number': 1})
        terminal = sb._BudgetRefusal((Refused(turn, SpendStopReason.COST_CEILING),))
        artifact = sb._NoProcessArtifact(context, terminal)
        self.assertEqual(artifact.terminal.refused[0].call.as_dict()['kind'], 'subagent_turn')
        judge = JudgeCall('quality', 'sha256:' + 'b' * 64, 'claude', 'judge-a', 1)
        for call in (judge, AnswerCall('sha256:' + 'a' * 64, coordinate)):
            with self.subTest(kind=call.as_dict()['kind']):
                with self.assertRaisesRegex(ValueError, 'requires refused turn calls'):
                    sb._BudgetRefusal((Refused(call, SpendStopReason.COST_CEILING),))
                with self.assertRaisesRegex(TypeError, 'requires a turn call'):
                    sb._OpaqueResponseRejected(call, 'bad call identity')

    def test_documented_optional_score_keeps_paid_and_refused_repeats(self):
        import os
        import subprocess

        source = Path(__file__).resolve().parents[1]
        demo = source / 'examples' / 'demo-skill'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sentinels = root / 'bin'
            sentinels.mkdir()
            env = {**os.environ, 'PATH': f'{sentinels}:/usr/bin:/bin'}
            for name in ('claude', 'codex', 'gemini', 'vibe', 'pi'):
                child = sentinels / name
                child.write_text('#!/bin/sh\nexit 197\n')
                child.chmod(0o755)
                blocked = subprocess.run([name, '--version'], env=env, timeout=5,
                                         capture_output=True, check=False)
                self.assertEqual(blocked.returncode, 197, name)
            tasks, runs, out = root / 'tasks.jsonl', root / 'runs', root / 'judge.jsonl'

            def cli(*argv):
                return subprocess.run([sys.executable, str(source / 'skill_benchmark.py'),
                                       *map(str, argv)], env=env, timeout=30,
                                      text=True, capture_output=True, check=False)

            prepared = cli('prepare', demo / 'evals' / 'shared-benchmark.json',
                           '--split', 'tune', '--out', tasks)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            answered = cli('run-codex', '--tasks', tasks, '--runs', runs,
                           '--codex-cmd', f'{sys.executable} {demo / "stub_runner.py"}',
                           '--max-cost-usd', '.02', '--assumed-cost-per-run-usd', '.01')
            self.assertEqual(answered.returncode, 2, answered.stderr)
            judged = cli('judge', demo / 'evals' / 'shared-benchmark.json', '--runs', runs,
                         '--judge-cmd', f'{sys.executable} {demo / "stub_judge.py"}',
                         '--judge-runs', '2', '--max-cost-usd', '.02',
                         '--assumed-cost-per-run-usd', '.03', '--out', out)
            self.assertEqual(judged.returncode, 2, judged.stderr)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            members = rows[0]['judge_runs']
            self.assertEqual(len(members), 2)
            self.assertEqual(members[0]['verdict_kind'], 'scored')
            self.assertEqual(members[0]['score'], 1.0)
            self.assertEqual(members[1]['judge_observation_kind'], 'missing')
            self.assertEqual(members[1]['spend_refusal_reason'], 'cost_ceiling')
            self.assertNotIn('returncode', members[1])
            self.assertFalse(rows[0]['judge_observation_complete'])
            ledgers = [json.loads(path.read_text()) for path in
                       (runs / 'spend').glob('*/spend-ceiling.json')]
            judge_ledger = next(ledger for ledger in ledgers
                                if ledger['calls'][0]['call']['kind'] == 'judge')
            self.assertEqual([call['state'] for call in judge_ledger['calls']],
                             ['settled', 'not_started', 'not_started', 'not_started'])
            self.assertIsNone(members[0]['cost_usd'])
            self.assertEqual(judge_ledger['spent_usd'], '0.03')
            benchmark = cli('benchmark', demo / 'evals' / 'shared-benchmark.json',
                            '--runs', runs, '--judge-results', out, '--out', root / 'benchmark.json')
            self.assertEqual(benchmark.returncode, 0, benchmark.stderr)
            summary = cli('cost-summary', '--manifest', demo / 'evals' / 'shared-benchmark.json',
                          '--runs', runs, '--judge-results', out, '--out', root / 'cost-summary.json')
            self.assertEqual(summary.returncode, 0, summary.stderr)
            counts = json.loads((root / 'cost-summary.json').read_text())['judge']
            self.assertEqual([counts[name] for name in ('requested_calls', 'billed_calls', 'not_started_calls',
                                                       'nonbillable_calls', 'unverified_calls')], [4, 1, 3, 0, 0])
