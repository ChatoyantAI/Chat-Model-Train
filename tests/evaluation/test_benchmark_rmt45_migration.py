"""Offline checks for the separately frozen full-run migration extension."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.roleplay_benchmarks.community import generate_case

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/evaluation/frozen_public_v1/run_rmt45_long_migration.py'
spec = importlib.util.spec_from_file_location('rmt45_migration_tested', SCRIPT)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class LongDialogueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.config = {'benchmark': 'rmtbench', 'languages': ['en', 'zh'], 'seed': 42, 'repeats': 2}
        self.row = {'idx': 'test-role', 'name': 'Role', 'description': 'Profile'}
        for field in ('character_understanding', 'character_maintenance', 'implicit_motivations'):
            self.row[field] = [[f'{field}-{block}-{turn}' for turn in range(7)] for block in range(2)]
        self.row['user_preference'] = [
            {'chat': [f'preference-{block}-{turn}' for turn in range(7)], 'question': 'Recall?', 'answer': 'PRIVATE_REFERENCE'}
            for block in range(2)]
        for lang in ('en', 'zh'):
            (self.base/f'rmtbench_claude45_{lang}.json').write_text(json.dumps([self.row]), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_repeat_count_lengths_order_and_unique_blocks(self):
        rows = extension.long_cases(self.config, self.base)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(len(row['layout']), 4)
            self.assertEqual(len(set(row['layout'])), 4)
            self.assertEqual(len(row['turns']), 28)
            for offset in range(0, 28, 7):
                self.assertEqual([t['block_turn'] for t in row['turns'][offset:offset+7]], list(range(7)))
        self.assertNotEqual(rows[0]['layout'], rows[1]['layout'])

    def test_reproducible_and_seed_changes_selection(self):
        rows = extension.long_cases(self.config, self.base)
        self.assertEqual(rows, extension.long_cases(self.config, self.base))
        self.assertNotEqual(rows, extension.long_cases({**self.config, 'seed': 123}, self.base))

    def test_claude45_single_reference_stays_on_last_block_turn(self):
        rows = extension.long_cases(self.config, self.base)
        refs = [t for row in rows for t in row['turns'] if 'reference_answer' in t]
        self.assertTrue(refs)
        self.assertTrue(all(t['block_turn'] == 6 and t['reference_answer'] == 'PRIVATE_REFERENCE' for t in refs))

    def test_generation_preserves_history_across_blocks_without_reference_leak(self):
        row = extension.long_cases(self.config, self.base)[0]
        calls = []
        class Client:
            def call(self, directory, role, messages):
                calls.append(json.loads(json.dumps(messages)))
                return f'own-reply-{len(calls)}', None
        work = self.base/'output'
        work.mkdir()
        generate_case(row, self.config, {'actor': 'Role-play {name}. {profile} {language}'}, work, Client())
        self.assertEqual(len(calls), 28)
        self.assertEqual(len(calls[7]), 16)  # system + seven prior pairs + next user
        self.assertIn('own-reply-1', json.dumps(calls[7]))
        self.assertNotIn('PRIVATE_REFERENCE', json.dumps(calls))
        self.assertTrue(json.loads((work/'trajectory.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main()
