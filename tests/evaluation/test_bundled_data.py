"""A new checkout restores exact evaluation inputs without Hugging Face calls."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_benchmarks import preflight, sources
from src.evaluation.roleplay_benchmarks.community_protocols import build_cases


ROOT = sources.ROOT
BUNDLE = Path('data/evaluation/roleplay_benchmarks')


class BundledDataTests(unittest.TestCase):
    def copy_hf_bundle(self, root):
        destination = root / BUNDLE
        destination.mkdir(parents=True)
        shutil.copyfile(ROOT / BUNDLE / 'manifest.json', destination / 'manifest.json')
        for name in ('minimax', 'rmtbench'):
            shutil.copytree(ROOT / BUNDLE / name, destination / name)

    def test_all_inputs_are_pinned_bytes_and_have_expected_coverage(self):
        catalog = sources.read_json(ROOT / BUNDLE / 'manifest.json')['benchmarks']
        self.assertEqual(set(catalog), set(sources.registry()))
        for name, entry in catalog.items():
            self.assertEqual(entry['revision'], sources.registry()[name]['revision'])
            for relative, record in entry['files'].items():
                path = ROOT / BUNDLE / name / relative
                self.assertEqual(path.stat().st_size, record['bytes'])
                self.assertEqual(sources.digest(path), record['sha256'])
        self.assertEqual(len(sources.read_json(ROOT / BUNDLE / 'coser/data/test/test_set.json')), 200)
        character = ROOT / BUNDLE / 'characterbench/eval_data/raw_data'
        self.assertEqual(len(list(character.glob('*.json'))), 13)
        self.assertEqual(sum(len(sources.read_json(p)) for p in character.glob('*.json')), 3250)
        for lang, count in [('en', 45), ('zh', 50)]:
            path = ROOT / BUNDLE / f'minimax/data/{lang}/seeds.jsonl'
            self.assertEqual(sum(bool(line.strip()) for line in path.read_text(encoding='utf-8').splitlines()), count)

    def test_fresh_restore_is_offline_and_preserves_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_hf_bundle(root)
            for name in ('minimax', 'rmtbench'):
                spec = sources.registry()[name]
                config = sources.read_json(ROOT / f'configs/evaluation/roleplay_benchmarks/{name}.example.json')
                config['limit_cases'] = None
                versions = ('claude35', 'claude45') if name == 'rmtbench' else (None,)
                expected = [build_cases({**config, 'version': version}, ROOT / BUNDLE / name) for version in versions]
                with patch.object(sources, 'ROOT', root), patch.object(sources, 'download', side_effect=AssertionError('Unexpected source download')):
                    self.assertEqual(sources.sync_one(name, spec)['status'], 'complete')
                    self.assertEqual(sources.verify_one(name, spec)['status'], 'complete')
                    self.assertEqual(sources.restore_bundled(name, spec), 0)
                    self.assertEqual(sources.sync_one(name, spec)['status'], 'complete')
                    actual = [build_cases({**config, 'version': version}, root / spec['path']) for version in versions]
                    self.assertEqual(actual, expected)
            with patch.object(sources, 'ROOT', root), patch.object(preflight, 'ROOT', root):
                report = preflight.inspect_source('minimax')
                self.assertEqual(report['inventory'], {'en/seeds': 45, 'zh/seeds': 50})
                for lang in ('en', 'zh'):
                    self.assertFalse(report['published_trajectory_audit'][lang]['available'])
                    self.assertIn('missing_files', report['published_trajectory_audit'][lang])

    def test_modified_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_hf_bundle(root)
            (root / BUNDLE / 'minimax/data/en/seeds.jsonl').write_text('{}\n', encoding='utf-8')
            with patch.object(sources, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Bundled minimax file'):
                sources.restore_bundled('minimax', sources.registry()['minimax'])

    def test_existing_modified_input_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_hf_bundle(root)
            spec = sources.registry()['minimax']
            existing = root / spec['path'] / 'data/en/seeds.jsonl'
            existing.parent.mkdir(parents=True)
            existing.write_text('existing evidence\n', encoding='utf-8')
            with patch.object(sources, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Existing minimax input differs'):
                sources.restore_bundled('minimax', spec)
            self.assertEqual(existing.read_text(encoding='utf-8'), 'existing evidence\n')


if __name__ == '__main__':
    unittest.main()
