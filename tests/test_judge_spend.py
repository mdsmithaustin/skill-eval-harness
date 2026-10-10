import json
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli, write_run


class JudgeSpendTests(unittest.TestCase):
    def batch(self, root, *, verdict=None, cost=0.6):
        manifest = make_eval_repo(root, cases=[{
            'id': 'case-1', 'split': 'tune', 'prompt': 'Do the task.',
            'assertions': [{'name': 'quality', 'type': 'judge', 'rubric': ['Be correct.'], 'gate': True}],
        }])
        runs = root / 'runs'
        write_run(runs / 'case-1' / 'with_skill', 'candidate')
        marker = root / 'launches'
        stub = root / 'judge-child'
        payload = {'type': 'result', 'result': json.dumps(verdict or {'passed': True}),
                   'usage': {'input_tokens': 3, 'output_tokens': 4}}
        if cost is not None:
            payload['total_cost_usd'] = cost
        stub.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
prompt = sys.stdin.read()
with Path({str(marker)!r}).open('a') as handle:
    handle.write('started\\n')
print({json.dumps(payload)!r} if '--output-format' in sys.argv else {json.dumps(verdict or {'passed': True})!r})
''')
        stub.chmod(0o755)
        return manifest, runs, stub, marker

    def invoke(self, root, route='native', *flags):
        manifest, runs, stub, marker = self.batch(root)
        out = root / 'results.jsonl'
        route_flags = ('--judge-cmd', str(stub)) if route == 'shell' else (
            '--judge-model', 'judge-a', '--claude-bin', str(stub))
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
