"""Scoring regressions exercised against the actual pinned CoSER worker."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_benchmarks import assessment as a, her_official, her_coser, unified
from src.evaluation.roleplay_benchmarks.community_protocols import MINI_DIMS

ROOT = Path(__file__).resolve().parents[2]


class AssessmentTests(unittest.TestCase):
    def test_offline_audit_detects_bad_score_and_keeps_sources_unchanged(self):
        from src.evaluation.roleplay_benchmarks.assessment_audit import save_audit
        critique = {d: {'flaws': [], 'score': 100} for d in a.DIMENSIONS}
        critique['Storyline Quality']['score'] = 50
        case = {'simulation': [{'role': 'A', 'content': 'A:'}], 'critique': critique, 'score': 87.5}
        data = {'cases': {'book': case}, 'scores': {'avg': 87.5}}
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'result.json', Path(tmp)/'audit.json'
            source.write_text(json.dumps(data))
            before = source.read_bytes()
            result = save_audit([source], output)
            self.assertEqual(result['model_calls'], 0)
            self.assertEqual(result['counts']['scoring_errors_cases'], 1)
            self.assertEqual(result['counts']['empty_or_label_only_turns'], 1)
            with self.assertRaises(FileExistsError):
                save_audit([source], output)
            self.assertEqual(source.read_bytes(), before)

    def test_bad_new_judgment_is_journaled_but_never_success_cached(self):
        from src.evaluation.roleplay_benchmarks import benchmark_api as api
        from src.evaluation.roleplay_benchmarks.endpoint_transport import JournalSender, coser_server
        from tests.evaluation.test_her_coser import Ledger
        actor = {'model': 'actor', 'revision': 'a', 'kind': 'local',
                 'base_url_env': 'AUDIT_URL', 'api_key_env': 'AUDIT_KEY'}
        judge = {**actor, 'model': 'judge', 'kind': 'remote'}
        config = {'assessment': {'version': a.VERSION, 'coser_role_thoughts': 'include'},
                  'actor': actor, 'judge': judge, 'max_requests_per_run': 10, 'timeout_seconds': 5}
        payload = {'model': 'judge', 'messages': [{'role': 'system', 'content':
                   'Evaluate in terms of Storyline Quality. Return "flaws".'},
                   {'role': 'user', 'content': 'A: Hello'}]}
        body = lambda text: {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}], 'usage': {}}
        bad = '{"Storyline Quality":{"flaws":[{"type":"wrong"}]}}'
        good = '{"Storyline Quality":{"flaws":[{"severity":5}]}}'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'AUDIT_URL': 'http://fixture', 'AUDIT_KEY': 'test'}):
            sender = JournalSender(config, unified.EndpointLimiter({'local': 1, 'remote': 2}), Ledger(), tmp)
            with patch.object(api, 'transport', side_effect=[body(bad), body(good)]) as transport, coser_server(sender) as port:
                req = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(payload).encode(),
                              headers={'X-Eval-Recovery-Scope': 'judge-case-1'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(req, timeout=5)
                error.exception.close()
                for _ in range(2):
                    with urlopen(req, timeout=5) as response:
                        self.assertEqual(json.load(response)['choices'][0]['message']['content'], good)
                self.assertEqual(transport.call_count, 2)
            evidence = [json.loads(p.read_text())['response']['choices'][0]['message']['content']
                        for p in Path(tmp).rglob('*.outcome.json')]
            self.assertCountEqual(evidence, [bad, good])

    def test_missing_severity_is_not_silently_replaced_with_one(self):
        for flaw in ({}, {'severity': True}, {'severity': '5'}, {'severity': 5.0}, {'severity': 0}, {'severity': 11}):
            with self.subTest(flaw=flaw), self.assertRaises(ValueError):
                a.validate_critique({'D': {'flaws': [flaw]}}, 'D')
        with self.assertRaises(ValueError):
            a.validate_critique({'wrong': {'flaws': []}}, 'D')
        self.assertEqual(a.deduction_score({'flaws': [{'severity': 10}, {'severity': 5}]}, 20), 55)
        with self.assertRaises(ValueError):
            a.deduction_score({'flaws': []}, 0)

    def test_judge_json_does_not_default_or_accept_duplicate_keys(self):
        self.assertEqual(a.extract_critique('```json\n{"D":{"flaws":[]}}\n```'), {'D': {'flaws': []}})
        for text in ('bad', '{"D":{"flaws":[],"flaws":[{}]}}', '{"D":NaN}', '{} {}'):
            with self.assertRaises(ValueError):
                a.extract_critique(text)
        parser = a.judge_parser({'messages': [{'content': 'in terms of Storyline Quality. "flaws"'}]})
        with self.assertRaises(ValueError):
            parser('{"Storyline Quality":{"flaws":[{}]}}')

    def test_labels_thought_visibility_and_empty_turns(self):
        native = [{'role': 'A', 'content': '[private\nplan](nods)Hello'}, {'role': 'Environment', 'content': 'Night falls.'}]
        tagged = [{'role': 'A', 'content': '<system_thinking>secret</system_thinking><role_thinking>private\nplan</role_thinking><role_action>nods</role_action>Hello'}, native[1]]
        for thoughts in ('include', 'exclude'):
            self.assertEqual(a.judge_messages(native, thoughts), a.judge_messages(tagged, thoughts))
        self.assertEqual(a.judge_messages(native, 'exclude')[0]['content'], 'A: (nods)Hello')
        self.assertEqual(a.judge_messages(native, 'include')[1]['content'], 'Environment: Night falls.')
        self.assertEqual(a.add_speaker('Quotation:\nA: hello', 'A'), 'A: Quotation:\nA: hello')
        self.assertEqual(a.add_speaker('**A:** hello', 'A'), '**A:** hello')
        self.assertEqual(a.role_content('...'), '...')  # Silence can be a real model choice, not a transport failure.
        self.assertEqual(a.recipient_view('[private\nplan]<role_action>nods</role_action>Hi'), '<role_action>nods</role_action>Hi')
        for text in ('', 'A:', '[]()', '<system_thinking>plan</system_thinking>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                a.judge_messages([{'role': 'A', 'content': text}], 'include')
        with self.assertRaises(ValueError):
            a.judge_messages([{'role': 'A', 'content': '[only private thoughts]'}], 'exclude')

    def test_worker_preparation_records_patches_without_changing_vendor(self):
        from src.evaluation.roleplay_benchmarks import frozen_backend
        config = json.loads((ROOT/'configs/evaluation/roleplay_benchmarks/historical/gemma4-e2b-it-coser-20260926.json').read_text())
        config['assessment'] = {'version': a.VERSION, 'coser_role_thoughts': 'include'}
        config['benchmark_options'] = {'coser': {'limit_cases': 1}}
        source = ROOT/'vendor/benchmarks/CoSER/gca_evaluation/main.py'
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = unified.prepare(config, Path(tmp)/'run')
            child = Path(tmp)/'run/coser'
            metadata = manifest['children']['coser']
            frozen_backend.verify(child, metadata)
            main = (child/'coser-worker-source/gca_evaluation/main.py').read_text()
            compile(main, 'main.py', 'exec')
            self.assertIn('validate_critique(response, dimension)', main)
            self.assertIn('candidates = sorted(candidates)', main)
            self.assertIn('assessment_sha256', manifest)
            self.assertIn('if False:', (child/'coser-worker-source/gca_evaluation/agent.py').read_text())
        self.assertEqual(before, source.read_bytes())

    def test_real_native_agent_preserves_repetition_in_reviewed_mode(self):
        from unittest.mock import Mock
        source = (ROOT/'vendor/benchmarks/CoSER/gca_evaluation/agent.py').read_text(encoding='utf-8')
        raw = 'A: I will stay. I will stay. I will stay.'
        utils = ModuleType('utils'); utils.get_response_with_retry = lambda **kwargs: raw
        for reviewed in (False, True):
            tree = ast.parse(a.patch_agent(source) if reviewed else source)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Agent')
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'chat')
            cleanup = Mock(return_value='A: I will stay.')
            ns = {'fix_repeation': cleanup, 'logger': SimpleNamespace(info=lambda *args: None)}
            exec(compile(ast.Module(body=[method], type_ignores=[]), 'agent', 'exec'), ns)
            actor = SimpleNamespace(messages=[], retrievers={}, model='candidate-candidate', name='A')
            with patch.dict('sys.modules', {'utils': utils}):
                result = ns['chat'](actor)
            self.assertEqual(result, raw if reviewed else 'A: I will stay.')
            self.assertEqual(cleanup.call_count, 0 if reviewed else 1)

    def test_actual_judging_path_matches_native_and_her_views(self):
        source = (ROOT/'vendor/benchmarks/CoSER/gca_evaluation/main.py').read_text(encoding='utf-8')
        template = {'self-play-deduct-template': 'Judge in terms of {dimension_name}: {original_conversation}',
                    'dimension_details': {d: {'dimension_brief': '', 'dimension_criteria': ''} for d in a.DIMENSIONS}}
        utils = ModuleType('utils'); utils.set_cache_path = lambda *args: None
        prompts = ModuleType('prompts'); prompts.critic_prompts = template
        original_sample = {'book_title': 'Book', 'i_p': 0, 'i_c': 0,
            'circumstance': {'plot': {'i_p': 0, 'summary': ''}, 'i_c': 0, 'scenario': '',
                            'major_characters': ['A'], 'dialogues': [{'character': 'A', 'message': '[reference]Hi'}]},
            'simulation': [{'role': 'A', 'content': '[private](nods)Hello'}, {'role': 'Environment', 'content': 'Night falls.'}],
            'involved_character_profiles': {'A': 'profile'}}
        for thoughts in ('include', 'exclude'):
            observed = []
            for protocol in ('native', 'format', 'her'):
                with self.subTest(protocol=protocol, thoughts=thoughts), tempfile.TemporaryDirectory() as tmp:
                    base = source if protocol == 'native' else her_coser.patch_judge_source(source) if protocol == 'format' else her_official.patch_main(source, 'without-system')
                    tree = ast.parse(a.patch_main(base, thoughts))
                    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gca_judging')
                    calls = []
                    def judge(parsers, **kw):
                        calls.append(kw['messages'])
                        dimension = next(d for d in a.DIMENSIONS if 'in terms of ' + d in kw['messages'][0]['content'])
                        value = json.dumps({dimension: {'flaws': [{'severity': 5}]}})
                        for parser in parsers:
                            value = parser(value)
                        return value
                    ns = dict(os=os, json=json, NSP='NSP', ENVIRONMENT='Environment',
                        args=SimpleNamespace(regenerate=False, reevaluate=False, continue_from=0),
                        logger=SimpleNamespace(info=lambda *args: None), get_response_json=judge,
                        judge_messages=a.judge_messages, validate_critique=a.validate_critique,
                        build_critic_prompt=a.build_critic_prompt,
                        extract_critique=a.extract_critique, deduction_score=a.deduction_score,
                        remove_inner_thoughts=her_coser.hide_inner_thoughts, calculate_bleu_rouge=lambda *args: (0, 0))
                    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'worker', 'exec'), ns)
                    sample = deepcopy(original_sample)
                    if protocol == 'her':
                        sample['simulation'][0]['content'] = '<system_thinking>PLAN</system_thinking><role_thinking>private</role_thinking><role_action>nods</role_action>Hello'
                    previous = Path.cwd()
                    try:
                        os.chdir(tmp)
                        path = Path('exp/simulation/case_actor.json'); path.parent.mkdir(parents=True)
                        path.write_text(json.dumps([sample]))
                        with patch.dict('sys.modules', {'utils': utils, 'prompts': prompts}):
                            scores, cases = ns['gca_judging']('case.json', 'actor', None, 'judge')
                    finally:
                        os.chdir(previous)
                    self.assertEqual(scores['Storyline Quality'], 76.5)
                    self.assertEqual(len(calls), 4)
                    observed.append(calls)
                    self.assertNotIn('PLAN', json.dumps(calls))
                    self.assertEqual('[private]' in calls[0][1]['content'], thoughts == 'include')
                    self.assertEqual('[reference]' in calls[0][0]['content'], thoughts == 'include')
                    self.assertNotIn('{major_characters}', json.dumps(calls))
                    self.assertIn('(Only apply to the main characters: A)', calls[2][0]['content'])
            self.assertEqual(observed[0], observed[1])
            self.assertEqual(observed[0], observed[2])

    def test_minimax_partial_repeat_never_enters_headline(self):
        manifest = {'assessment': {}, 'config': {'benchmark': 'minimax', 'languages': ['en'], 'repeats': 3}}
        manifest['assessment'] = {'version': a.VERSION}
        cases = [{'id': str(i), 'source_id': 'seed', 'language': 'en'} for i in range(3)]
        rows = [{'id': str(i), 'complete': i < 2, 'scores': {d: 90 for d in MINI_DIMS}} for i in range(3)]
        original = {'scores': {'en': {'overall': 90}}, 'cases': rows}
        result = a.review_public_summary(original, manifest, cases)
        self.assertEqual(result['scores']['en']['dimensions'], {})
        self.assertEqual(result['partial_scores']['en']['overall'], 90)
        self.assertEqual(result['missing_cases'], ['2'])
        self.assertEqual(original['scores']['en']['overall'], 90)
        rows[-1]['complete'] = True
        result = a.review_public_summary(original, manifest, cases)
        self.assertEqual(result['scores']['en']['overall'], 90)
        self.assertEqual(result['scores']['en']['included_sessions'], 3)

    def test_rmt_excludes_incomplete_scenes_and_preserves_weighting(self):
        manifest = {'assessment': {'version': a.VERSION}, 'config': {'benchmark': 'rmtbench', 'languages': ['en']}}
        cases = [{'id': str(i), 'language': 'en', 'turns': [{'field': 'character_understanding'}]*n}
                 for i, n in enumerate((2, 6, 1))]
        summary = {'scores': {'en': {}}, 'cases': [
            {'id': '0', 'complete': True, 'scores': {'EE': 1}},
            {'id': '1', 'complete': True, 'scores': {'EE': 5}},
            {'id': '2', 'complete': False, 'scores': {'EE': 5}}]}
        result = a.review_public_summary(summary, manifest, cases)
        self.assertEqual(result['scores']['en']['EE']['mean'], 4)
        self.assertEqual(result['scores']['en']['EE']['count'], 8)
        self.assertEqual(result['missing_cases'], ['2'])


if __name__ == '__main__':
    unittest.main()
