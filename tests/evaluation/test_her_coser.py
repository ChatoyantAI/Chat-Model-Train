import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
from urllib.request import Request, urlopen

from src.evaluation.roleplay_benchmarks import benchmark_api as api
from src.evaluation.roleplay_benchmarks.her_coser import valid_actor_content, hide_inner_thoughts, patch_judge_source, PROTOCOL, actor_prompt_payload, COSER_OUTPUT_INSTRUCTION
from scripts.evaluation.convert_her_to_coser import HER_OUTPUT_INSTRUCTION
from src.evaluation.roleplay_benchmarks.endpoint_transport import JournalSender, coser_server
from src.evaluation.roleplay_benchmarks.unified import EndpointLimiter


class Ledger:
    def __init__(self):
        self.ledger = self
    def check(self):
        pass
    def append(self, row):
        pass


class HerCoSERTests(unittest.TestCase):
    def test_actual_judging_function_keeps_thoughts_but_metrics_do_not(self):
        root = Path(__file__).resolve().parents[2]
        source = root / 'vendor/benchmarks/CoSER/gca_evaluation/main.py'
        if not source.exists():
            self.skipTest('CoSER vendor not installed')
        parsed = ast.parse(patch_judge_source(source.read_text(encoding='utf-8')))
        function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'gca_judging')
        utils = ModuleType('utils'); utils.set_cache_path = lambda p: None
        prompts = ModuleType('prompts')
        dimensions = ['Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality']
        prompts.critic_prompts = {'self-play-deduct-template': '{dimension_name} {original_conversation}',
                                 'dimension_details': {d: {'dimension_brief': '', 'dimension_criteria': ''} for d in dimensions}}
        calls = []
        def judge(*a, **kw):
            calls.append(kw['messages'])
            return {d: {'flaws': []} for d in dimensions}
        def metrics(reference, simulation):
            self.assertNotIn('[', reference[0]['message'])
            self.assertNotIn('[', simulation[0]['content'])
            return 0, 0
        ns = {'os': os, 'json': json, 'args': SimpleNamespace(regenerate=False, reevaluate=False, continue_from=0),
              'NSP': 'NSP', 'ENVIRONMENT': 'Environment', 'logger': SimpleNamespace(info=lambda *a: None),
              'remove_inner_thoughts': hide_inner_thoughts, 'get_response_json': judge,
              'extract_json': lambda x: x, 'calculate_bleu_rouge': metrics}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
        sample = [{'book_title': 'Fixture', 'i_p': 0, 'i_c': 0,
                   'circumstance': {'plot': {'i_p': 0, 'summary': ''}, 'i_c': 0, 'scenario': '',
                     'major_characters': ['A'], 'dialogues': [{'character': 'A', 'message': '[reference thought]Hi'}]},
                   'simulation': [{'role': 'A', 'content': 'A: [private thought](nods)Hi'}],
                   'involved_character_profiles': {'A': 'A profile'}}]
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'utils': utils, 'prompts': prompts}):
            old = os.getcwd()
            try:
                os.chdir(tmp)
                path = Path('exp/simulation/case_actor.json'); path.parent.mkdir(parents=True)
                path.write_text(json.dumps(sample))
                ns['gca_judging']('case.json', 'actor', None, 'judge')
            finally:
                os.chdir(old)
        self.assertEqual(len(calls), 4)
        for messages in calls:
            self.assertIn('[reference thought]', messages[0]['content'])
            self.assertIn('[private thought]', messages[1]['content'])

    def payload(self):
        return {'model': 'actor', 'messages': [
            {'role': 'system', 'content': 'Persona and scene.\n===Requirements===\n'
             'Use [your thought] for thoughts. Use (your action) for actions.\n\nLimit your response to 60 words.'},
            {'role': 'assistant', 'content': '[My private\nthought](nods)Hello'},
            {'role': 'user', 'content': 'Other: [Their private\nthought](waves)Hi'},
        ], 'temperature': .7, 'max_tokens': 8192}

    def test_recipient_cannot_see_multiline_thoughts(self):
        self.assertEqual(hide_inner_thoughts('[secret\nplan](nods)Yes'), '(nods)Yes')
        self.assertEqual(valid_actor_content('<system_thinking>plan</system_thinking>'
            '<role_thinking>fear</role_thinking><role_action>nods</role_action>Yes'), '[fear](nods)Yes')
        with self.assertRaises(ValueError):
            valid_actor_content('<system_thinking>only a plan</system_thinking>')

    def test_actor_format_preserves_profiles_history_and_sampling(self):
        payload = self.payload()
        prefix = 'Character profile\nScenario\nOther characters\n\n'
        suffix = '\n\nSpeak concisely as humans, instead of being verbose. Limit your response to 60 words.\n\n'
        payload['messages'][0]['content'] = prefix + '===Requirements===\n' + COSER_OUTPUT_INSTRUCTION + suffix
        before = json.dumps(payload, sort_keys=True)
        changed = actor_prompt_payload(payload)
        self.assertEqual(changed['messages'][0]['content'], prefix + HER_OUTPUT_INSTRUCTION.strip('\n') + suffix)
        self.assertEqual(changed['messages'][1:], payload['messages'][1:])
        self.assertEqual(changed['temperature'], payload['temperature'])
        self.assertEqual(changed['max_tokens'], payload['max_tokens'])
        self.assertEqual(json.dumps(payload, sort_keys=True), before)
        with self.assertRaises(ValueError):
            actor_prompt_payload(self.payload())

    def test_no_system_prompt_keeps_scene_and_native_mode_preserves_reply(self):
        payload = self.payload()
        prefix, suffix = 'Profile and scene\n', '\nLimit your response to 60 words.'
        payload['messages'][0]['content'] = prefix + '===Requirements===\n' + COSER_OUTPUT_INSTRUCTION + suffix
        converted = actor_prompt_payload(payload, 'her-no-system-v1')
        self.assertNotIn('system_thinking', converted['messages'][0]['content'])
        self.assertIn('<role_thinking>', converted['messages'][0]['content'])
        self.assertTrue(converted['messages'][0]['content'].startswith(prefix))
        self.assertTrue(converted['messages'][0]['content'].endswith(suffix))
        self.assertEqual(converted['messages'][1:], payload['messages'][1:])
        self.assertEqual(valid_actor_content('Howl: [fear](nods)Yes', strict=True, native=True), 'Howl: [fear](nods)Yes')

    def test_strict_response_rejects_empty_speaker_and_mismatched_tags(self):
        for native in (False, True):
            with self.assertRaises(ValueError):
                valid_actor_content('Howl Jenkins:', strict=True, native=native)
        for text in ('Eddie: <system_thinking>plan</role_thinking><role_action>smiles</role_action>Hello',
                     'Eddie: <system_thinking>unfinished',
                     'Eddie: <role_thinking>fear</role_action>Hello',
                     'Eddie: <system_thinking>plan</system_thinking>'):
            with self.assertRaises(ValueError):
                valid_actor_content(text, strict=True)
        self.assertEqual(valid_actor_content('Eddie: <system_thinking>plan</system_thinking><role_thinking>fear</role_thinking><role_action>smiles</role_action>Hello', strict=True), 'Eddie: [fear](smiles)Hello')

    def test_strict_proxy_keeps_invalid_evidence_and_retries_without_replaying(self):
        actor = {'model': 'actor', 'revision': 'r1', 'kind': 'local',
                 'base_url_env': 'TEST_HER_URL', 'api_key_env': 'TEST_HER_KEY'}
        config = {'protocol': PROTOCOL, 'actor': actor, 'judge': {**actor, 'model': 'judge', 'kind': 'remote'},
                  'actor_output_validation': 'role-content-v2', 'max_requests_per_run': 10, 'timeout_seconds': 5}
        response = lambda text: {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}], 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TEST_HER_URL': 'http://127.0.0.1:1/v1', 'TEST_HER_KEY': 'unused'}):
            sender = JournalSender(config, EndpointLimiter({'local': 2, 'remote': 2}), Ledger(), directory)
            with patch.object(api, 'transport', side_effect=[response('Howl:'), response('Howl: Hello')]) as transport, coser_server(sender) as port:
                from urllib.error import HTTPError
                req = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(self.payload()).encode(), headers={'X-Eval-Recovery-Scope': 'case-1'})
                with self.assertRaises(HTTPError):
                    urlopen(req, timeout=5)
                for _ in range(2):
                    with urlopen(req, timeout=5) as reply:
                        self.assertEqual(json.load(reply)['choices'][0]['message']['content'], 'Howl: Hello')
                self.assertEqual(transport.call_count, 2)
            outcomes = [json.loads(p.read_text()) for p in Path(directory).rglob('*.outcome.json')]
            self.assertEqual(len(outcomes), 2)
            self.assertIn('Howl:', [row['response']['choices'][0]['message']['content'] for row in outcomes])

    def test_prompt_mode_journals_actual_prompt_replays_and_leaves_judge_unchanged(self):
        actor = {'model': 'actor', 'revision': 'r1', 'kind': 'local',
                 'base_url_env': 'TEST_HER_URL', 'api_key_env': 'TEST_HER_KEY'}
        judge = {**actor, 'model': 'judge', 'kind': 'remote'}
        config = {'protocol': PROTOCOL, 'actor_prompt_format': 'her-official-v1',
                  'actor': actor, 'judge': judge, 'max_requests_per_run': 10, 'timeout_seconds': 5}
        payload = self.payload()
        payload['messages'][0]['content'] = 'Profile.\n===Requirements===\n' + COSER_OUTPUT_INSTRUCTION + '\n\nLimit your response to 60 words.'
        response = {'choices': [{'message': {'content': 'Hi'}, 'finish_reason': 'stop'}], 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'TEST_HER_URL': 'http://127.0.0.1:1/v1', 'TEST_HER_KEY': 'unused'}):
            sender = JournalSender(config, EndpointLimiter({'local': 2, 'remote': 2}), Ledger(), directory)
            with patch.object(api, 'transport', return_value=response) as transport:
                sender.call(actor, payload, 5, 'case-1')
                sender.call(actor, payload, 5, 'case-1')
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(transport.call_args.args[1], actor_prompt_payload(payload))
                helper = {'model': 'judge', 'messages': [{'role': 'user', 'content': 'Judge this'}]}
                sender.call(judge, helper, 5, 'judge-1')
                self.assertEqual(transport.call_args.args[1], helper)
            outcomes = [json.loads(p.read_text()) for p in Path(directory).rglob('*.outcome.json')]
            saved = next(r for r in outcomes if r['model'] == 'actor')
            self.assertEqual(saved['request'], actor_prompt_payload(payload))

    def test_raw_journal_converted_delivery_and_zero_request_replay(self):
        actor = {'model': 'actor', 'revision': 'r1', 'kind': 'local',
                 'base_url_env': 'TEST_HER_URL', 'api_key_env': 'TEST_HER_KEY'}
        judge = {**actor, 'model': 'judge', 'kind': 'remote'}
        config = {'protocol': PROTOCOL, 'actor': actor, 'judge': judge,
                  'max_requests_per_run': 10, 'timeout_seconds': 5}
        raw = '<system_thinking>private plan</system_thinking><role_thinking>fear</role_thinking><role_action>nods</role_action>Yes'
        response = {'choices': [{'message': {'content': raw}, 'finish_reason': 'stop'}], 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'TEST_HER_URL': 'http://127.0.0.1:1/v1', 'TEST_HER_KEY': 'unused'}):
            sender = JournalSender(config, EndpointLimiter({'local': 2, 'remote': 2}), Ledger(), directory)
            with patch.object(api, 'transport', return_value=response) as transport, coser_server(sender) as port:
                for _ in range(2):
                    req = Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                  data=json.dumps(self.payload()).encode(),
                                  headers={'X-Eval-Recovery-Scope': 'case-1'})
                    with urlopen(req, timeout=5) as reply:
                        self.assertEqual(json.load(reply)['choices'][0]['message']['content'], '[fear](nods)Yes')
                self.assertEqual(transport.call_count, 1)
                sent = transport.call_args.args[1]
                self.assertEqual(sent, self.payload())
                # Helpers retain the original request and response.
                helper = {'model': 'judge', 'messages': [{'role': 'user', 'content': 'Judge this'}]}
                req = Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                              data=json.dumps(helper).encode(), headers={'X-Eval-Recovery-Scope': 'judge-1'})
                with urlopen(req, timeout=5) as reply:
                    self.assertEqual(json.load(reply)['choices'][0]['message']['content'], raw)
                self.assertEqual(transport.call_args.args[1], helper)
            outcomes = [json.loads(p.read_text()) for p in Path(directory).rglob('*.outcome.json')]
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(all(r['response']['choices'][0]['message']['content'] == raw for r in outcomes))


if __name__ == '__main__':
    unittest.main()
