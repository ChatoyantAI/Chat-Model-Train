"""Behavioral checks for the directory/dependency cleanup; no provider calls."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_benchmarks import benchmark_api as api, characterbench, frozen_backend
from src.evaluation.roleplay_benchmarks.endpoint_transport import JournalClient, JournalSender, identity
from src.evaluation.roleplay_benchmarks.execution import EndpointLimiter
from src.evaluation.roleplay_benchmarks.sources import ROOT, read_json, write_json
from tests.evaluation import test_unified_runner as fixtures


class StructureTests(unittest.TestCase):
    def test_character_port_matches_frozen_requests_results_and_retries(self):
        config = fixtures.UnifiedRunnerTests().config()
        protocol = frozen_backend.load(config)
        tasks = [dict(id=f'case-{i}', language=language, dimension='fixture', index=i,
                      scale_max=scale, actor_prompt=prompt,
                      judge_template='Judge exactly: ' + protocol.SENTINEL)
                 for i, (language, scale, prompt) in enumerate([
                     ('en', 5, 'English fixture'), ('zh', 3, '中文样本'), ('en', 5, 'invalid fixture')])]
        requests = [[], []]

        class Client:
            def __init__(self, captured):
                self.captured = captured

            def call(self, work, role, messages, parser):
                self.captured.append(json.dumps([role, messages], ensure_ascii=False))
                if messages[0]['content'] == 'invalid fixture':
                    raise ValueError('invalid actor response')
                text = '{"response":"Hello."}' if role == 'actor' else '3'
                return text, parser(text)

        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / name for name in ('frozen', 'current')]
            for root in roots:
                out = root / 'outputs/full-public-v1'
                write_json(out / 'character-inputs.json', tasks)
                write_json(out / 'character-input-audit.json', {'invalid': []})
            with patch.object(api, 'Client', return_value=Client(requests[0])):
                old = protocol.run_character(roots[0], roots[0] / 'outputs/full-public-v1',
                                             'candidate', SimpleNamespace(halt=None))
            with patch.object(characterbench, 'JournalClient', return_value=Client(requests[1])):
                new = characterbench.run(config, roots[1], SimpleNamespace(halt=None, directory=roots[1]),
                                         SimpleNamespace(read=read_json, write=write_json))
            self.assertEqual(old, new)
            self.assertEqual(Counter(requests[0]), Counter(requests[1]))
            for task in tasks:
                suffix = f"outputs/full-public-v1/candidate/characterbench/{task['id']}/result.json"
                self.assertEqual(read_json(roots[0] / suffix), read_json(roots[1] / suffix))
            self.assertEqual(new['completed'], 2)
            failures = list((roots[1] / 'outputs/full-public-v1/candidate/characterbench/case-2').glob('failure-*.json'))
            self.assertEqual(len(failures), 2)

    def test_explicit_client_replays_legacy_scope_without_mutating_global_client(self):
        config = fixtures.UnifiedRunnerTests().config()
        role = config['actor']
        messages = [{'role': 'user', 'content': 'same saved sample'}]
        payload = {'model': role['model'], 'messages': messages, 'stream': False, **role['sampling']}
        original_methods = api.Client.__init__, api.Client.call
        ledger = SimpleNamespace(check=lambda: None, append=lambda value: None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_URL': 'http://fixture', 'TEST_KEY': 'fixture'}):
            root = Path(tmp)
            work = root / 'minimax/cases/case-1/requests'
            # This is the pre-refactor scoped_clients wire identity, not a new format.
            scope = identity({'directory': str(work.relative_to(root)), 'role': 'actor',
                              'revision': role['revision'], 'payload': payload})
            key = identity({'scope': scope, 'payload': {**payload, **config['thinking']['local']},
                            'revision': role['revision'], 'url': api.endpoint(role)[0]})
            journal = root / 'transport-journal' / key[:2] / key
            write_json(journal / 'state.json', {'operation_id': 'legacy-success'})
            write_json(journal / 'legacy-success.outcome.json', {
                'status': 'response', 'response': fixtures.response('already paid')})
            sender = JournalSender(config, EndpointLimiter({'local': 2, 'remote': 2}),
                                   SimpleNamespace(ledger=ledger), root)
            client_config = {**config, 'roles': {'actor': role}}
            client = JournalClient(client_config, sender, root)
            with patch.object(api, 'transport', side_effect=AssertionError('Successful request billed twice')):
                self.assertEqual(client.call(work, 'actor', messages)[0], 'already paid')
            self.assertEqual(sender.calls, 0)
            self.assertEqual(sender.replays, 1)
            self.assertEqual(original_methods, (api.Client.__init__, api.Client.call))

    def test_launcher_and_spawned_writer_work_outside_repo_without_script_import_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            help_result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluation/run_public_evaluation.py'), '--help'],
                                         cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn('audit-results', help_result.stdout)
            missing = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluation/run_public_evaluation.py'),
                                      'run', '--execute'], cwd=tmp, text=True, capture_output=True, timeout=30)
            self.assertEqual(missing.returncode, 2)
            self.assertIn('--run-dir', missing.stderr)
            runner = Path(tmp) / 'cold_writer.py'
            runner.write_text(
                f'import sys\nsys.path.insert(0, {str(ROOT)!r})\n'
                'from src.evaluation.roleplay_benchmarks.storage import ProcessLedger\n'
                "if __name__ == '__main__':\n"
                f'    ledger = ProcessLedger({str(Path(tmp) / "usage")!r}, batch_size=1)\n'
                "    ledger.append({'status': 'completed', 'cost': .01})\n"
                '    ledger.close()\n', encoding='utf-8')
            result = subprocess.run([sys.executable, str(runner)], cwd=tmp, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            entries = [json.loads(line) for path in (Path(tmp) / 'usage').glob('*.jsonl')
                       for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(entries, [{'status': 'completed', 'cost': .01}])


if __name__ == '__main__':
    unittest.main()
