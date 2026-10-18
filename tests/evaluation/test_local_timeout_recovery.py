import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation.recover_local_coser_timeouts import recover
from src.evaluation.roleplay_benchmarks.unified import _hash


class LocalRecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        env = patch.dict(os.environ, {'FIXTURE_LOCAL': 'http://127.0.0.1:1/v1',
                         'FIXTURE_REMOTE': 'https://example.test/v1', 'FIXTURE_KEY': 'unused'})
        env.start()
        self.addCleanup(env.stop)
        actor = dict(model='actor', revision='r1', base_url_env='FIXTURE_LOCAL',
                     api_key_env='FIXTURE_KEY', kind='local', sampling={'temperature': .7, 'max_tokens': 8192})
        judge = dict(actor, model='judge', kind='remote', base_url_env='FIXTURE_REMOTE',
                     sampling={'temperature': 0, 'max_tokens': 8192})
        self.config = dict(schema_version=1, benchmarks=['coser'], actor=actor, judge=judge,
                           max_attempts=3, timeout_seconds=600, max_requests_per_run=100,
                           concurrency={'endpoints': {'local': 128, 'remote': 96}, 'benchmarks': {'coser': 192}})
        self.write('manifest.json', dict(runner='unified-roleplay-v2', status='partial',
                                        config=self.config, config_sha256=_hash(self.config)))
        self.write('coser/manifest.json', {'config': self.config})
        self.complete = 'coser/outputs/full-public-v1/candidate/coser/status-0001.json'
        self.write(self.complete, {'status': 'completed', 'scores': {'avg': 50}})

    def write(self, name, value):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return p

    def request(self, key, model='actor', success=False):
        folder = 'transport-journal/aa/' + key
        p = self.write(folder + '/state.json', {'operation_id': 'timeout'})
        self.write(folder + '/timeout.outcome.json', dict(operation_id='timeout', model=model,
                   revision='r1', status='uncertain', error_type='TimeoutError'))
        if success:
            self.write(folder + '/ok.outcome.json', dict(operation_id='ok', status='response',
                response={'choices': [{'message': {'content': 'cached'}, 'finish_reason': 'stop'}]}))
        return p

    def test_only_local_timeouts_released_and_evidence_preserved(self):
        local = self.request('local')
        paid = self.request('paid', model='judge')
        before = {str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*.outcome.json')}
        completed = (self.root / self.complete).read_bytes()
        result = recover(self.root, execute=True)
        self.assertEqual(result['local_timeouts_released'], 1)
        self.assertFalse(local.exists())
        self.assertTrue(paid.exists())
        self.assertTrue((Path(result['audit'])/'states/local.json').exists())
        for name, data in before.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        self.assertEqual((self.root / self.complete).read_bytes(), completed)
        after = json.loads((self.root/'manifest.json').read_text())['config']
        self.assertEqual(after, {**self.config, 'timeout_seconds': 1200})

    def test_existing_success_is_restored_instead_of_reissued(self):
        state = self.request('replay', success=True)
        result = recover(self.root, execute=True)
        self.assertEqual(result['local_timeouts_released'], 0)
        self.assertEqual(json.loads(state.read_text())['operation_id'], 'ok')

    def test_dry_run_and_running_lock_do_not_mutate(self):
        state = self.request('dry')
        recover(self.root)
        self.assertTrue(state.exists())
        (self.root/'.run.lock').touch()
        with self.assertRaises(ValueError):
            recover(self.root, execute=True)
        self.assertTrue(state.exists())


if __name__ == '__main__':
    unittest.main()
