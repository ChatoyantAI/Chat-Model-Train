import ast
from functools import partial
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from src.evaluation.roleplay_benchmarks import benchmark_api as api
from src.evaluation.roleplay_benchmarks import her_official as her
from src.evaluation.roleplay_benchmarks.endpoint_transport import JournalSender, coser_server
from src.evaluation.roleplay_benchmarks.unified import EndpointLimiter, validate

ROOT = Path(__file__).resolve().parents[2]


class Ledger:
    def __init__(self):
        self.ledger = self
    def check(self):
        pass
    def append(self, row):
        pass


class HEROfficialTests(unittest.TestCase):
    def config(self, mode='with-system'):
        return json.loads((ROOT / 'configs/evaluation/roleplay_benchmarks/historical' /
            f'qwen3-0.6b-her-27000-coser-official-{mode}.json').read_text())

    def test_source_pin_and_both_explicit_modes(self):
        root = Path(her.__file__).parent / 'her_upstream'
        pin = json.loads((root / 'provenance.json').read_text())
        for name, sha in pin['sha256'].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), sha)
        for mode in her.MODES:
            config = self.config(mode)
            validate(config)
            config['thinking']['local']['chat_template_kwargs']['enable_thinking'] = True
            with self.assertRaises(ValueError):
                validate(config)
        config = self.config()
        config.pop('her_system_thinking')
        with self.assertRaises(ValueError):
            validate(config)

    def test_real_worker_multiturn_history_and_official_prompts(self):
        source = ROOT / 'vendor/benchmarks/CoSER/gca_evaluation/main.py'
        dataset = json.loads((ROOT / 'vendor/benchmarks/CoSER/data/test/test_set.json').read_text(encoding='utf-8'))
        for mode in her.MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                tree = ast.parse(her.patch_main(source.read_text(encoding='utf-8'), mode))
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gca_simulation')
                sample = deepcopy(next(s for s in dataset if len([c for c in s['speaking_characters_w_env'] if c != 'Environment']) >= 2))
                chars = [c for c in sample['speaking_characters_w_env'] if c != 'Environment']
                a, b = chars[:2]
                sample['speaking_characters_w_env'] = [a, b]
                turns, calls = [], []
                def respond(model, messages, **kwargs):
                    calls.append(deepcopy(messages))
                    if model == 'helper':
                        # Alternate two roles until the minimum end-chat round.
                        target = 'random' if len(turns) == 1 else ('<END CHAT>' if len(turns) >= 6 else (b if len(turns) % 2 else a))
                        return 'Reasoning: Continue.\nNext Speaker: ' + target
                    name = a if messages[0]['content'].startswith('You are ' + a + ' from ') else b
                    text = (f'<system_thinking>strategy-{len(turns)}</system_thinking>' if mode == 'with-system' else '')
                    text += f'<role_thinking>secret-{len(turns)}</role_thinking><role_action>nods</role_action>Hello {len(turns)}'
                    turns.append((name, text))
                    return text
                utils = ModuleType('utils')
                utils.set_cache_path = lambda *a: None
                utils.get_response_with_retry = respond
                ns = dict(os=os, json=json, random=SimpleNamespace(choice=lambda choices: b), Agent=her.Agent,
                    open=partial(open, encoding='utf-8'),
                    NSP='NSP', ENVIRONMENT='Environment', special_characters=['NSP', 'Environment'],
                    args=SimpleNamespace(regenerate=False, continue_from=0, wo_thought=False),
                    logger=SimpleNamespace(info=lambda *a: None),
                    get_character_prompt=partial(her.prompts.get_character_prompt, model_type='her' if mode=='with-system' else 'her_nosys'),
                    get_environment_prompt=her.prompts.get_environment_prompt, get_nsp_prompt=her.prompts.get_nsp_prompt,
                    parse_nsp_response=her.utils.parse_nsp_response, recipient_view=her.recipient_view,
                    remove_system_thinking=her.utils.remove_system_thinking, add_speaker_name=her.utils.add_speaker_name)
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
                before = Path.cwd()
                try:
                    os.chdir(tmp)
                    Path('case.json').write_text(json.dumps([sample]))
                    with patch.dict('sys.modules', {'utils': utils}):
                        result = ns['gca_simulation']('case.json', 'actor', 'helper', 'helper', None)
                finally:
                    os.chdir(before)
                actor_calls = [m for m in calls if m[0]['content'].startswith('You are ')]
                self.assertEqual(len(actor_calls), 6)
                first, own_next = actor_calls[0], actor_calls[2]
                prompt = first[0]['content']
                self.assertIn('ONLY playing ' + a, prompt)
                self.assertIn('approximately 200 words', prompt)
                self.assertNotIn('60 words', prompt)
                self.assertEqual('System Thinking:' in prompt, mode == 'with-system')
                own = [m for m in own_next if m['role'] == 'assistant']
                self.assertEqual(own[0], {'role': 'assistant', 'content': turns[0][1], 'name': a})
                other = next(m for m in own_next if m.get('name') == b)
                self.assertIn('<role_action>nods</role_action>', other['content'])
                self.assertNotIn('secret-', other['content'])
                self.assertNotIn('strategy-', other['content'])
                self.assertEqual(result[0]['simulation'][0]['content'], turns[0][1])
                nsp_calls = [m for m in calls if m[0]['content'].startswith('Your task is to predict')]
                self.assertIn({'role': 'assistant', 'content': 'random'}, nsp_calls[1])

    def test_judge_executes_official_rubric_and_keeps_role_thoughts(self):
        source = ROOT / 'vendor/benchmarks/CoSER/gca_evaluation/main.py'
        tree = ast.parse(her.patch_main(source.read_text(encoding='utf-8'), 'with-system'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gca_judging')
        calls = []
        dims = list(her.prompts.CRITIC_PROMPTS['dimension_details'])
        def judge(*args, **kwargs):
            calls.append(kwargs['messages'])
            return {d: {'flaws': []} for d in dims}
        utils = ModuleType('utils'); utils.set_cache_path = lambda *a: None
        ns = dict(os=os, json=json, NSP='NSP', ENVIRONMENT='Environment', judge_view=her.judge_view,
            args=SimpleNamespace(regenerate=False, reevaluate=False, continue_from=0),
            logger=SimpleNamespace(info=lambda *a: None), get_response_json=judge, extract_json=lambda x: x,
            calculate_bleu_rouge=lambda *a: (0, 0))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), ns)
        sample = [{'book_title':'Book','i_p':0,'i_c':0,'involved_character_profiles':{'A':'profile'},
            'circumstance':{'plot':{'i_p':0,'summary':'plot'},'i_c':0,'scenario':'scene','major_characters':['A'],
                'dialogues':[{'character':'A','message':'<role_thinking>reference</role_thinking>Hi'}]},
            'simulation':[{'role':'A','content':'<system_thinking>PLAN</system_thinking><role_thinking>FEAR</role_thinking><role_action>nods</role_action>Hi'}]}]
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'utils':utils}):
            before = Path.cwd()
            try:
                os.chdir(tmp)
                path = Path('exp/simulation/case_actor.json'); path.parent.mkdir(parents=True)
                path.write_text(json.dumps(sample))
                ns['gca_judging']('case.json','actor',None,'judge')
            finally:
                os.chdir(before)
        self.assertEqual(len(calls), 4)
        for i, messages in enumerate(calls):
            self.assertEqual(messages[1]['content'], 'A: [FEAR](nods)Hi')
            self.assertIn('[reference]', messages[0]['content'])
            self.assertNotIn('PLAN', json.dumps(messages))
            expected = her.prompts.CRITIC_PROMPTS['self-play-deduct-template'].format(
                book='Book',plot_summary='plot',scenario='scene',character_profiles='### A\n\nprofile',
                original_conversation='A: [reference]Hi',major_characters='A',additional_instructions='',
                dimension_name=dims[i],**her.prompts.CRITIC_PROMPTS['dimension_details'][dims[i]])
            self.assertEqual(messages[0]['content'],expected)

    def test_completion_and_chat_raw_journal_replay(self):
        for mode in her.MODES:
            config = self.config(mode)
            config['actor']['sampling']['max_tokens'] = 8192  # no real tokenize endpoint in this test
            actor = config['actor']
            raw = '<system_thinking>plan</system_thinking><role_thinking>fear</role_thinking><role_action>nods</role_action>Yes'
            choice = {'finish_reason': 'stop'}
            choice.update({'text': raw} if mode == 'with-system' else {'message': {'content': raw}})
            body = {'choices':[choice], 'usage':{'total_tokens':100}}
            payload = {'model':actor['model'], 'messages':[{'role':'system','content':'official fixture'},
                {'role':'user','content':'start'}, {'role':'user','name':'B','content':'B: Hello'}],
                'max_tokens':8192,'temperature':.7}
            env = {actor['base_url_env']:'http://127.0.0.1:1/v1',actor['api_key_env']:'unused'}
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env):
                sender = JournalSender(config,EndpointLimiter({'local':2,'remote':2}),Ledger(),tmp)
                with patch.object(api, 'transport', return_value=body) as transport, coser_server(sender) as port:
                    for _ in range(2):
                        req = Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(payload).encode(),
                                      headers={'X-Eval-Recovery-Scope':'case'})
                        with urlopen(req,timeout=5) as reply:
                            content = json.load(reply)['choices'][0]['message']['content']
                            self.assertIn('<role_thinking>fear</role_thinking>',content)
                            self.assertEqual('system_thinking' in content,mode=='with-system')
                    self.assertEqual(transport.call_count,1)
                    wire = transport.call_args.args[1]
                    if mode == 'with-system':
                        self.assertEqual(wire['prompt'],her.utils.format_qwen_prompt(payload['messages']))
                        self.assertNotIn('chat_template_kwargs',wire)
                        self.assertNotIn('messages',wire)
                    else:
                        self.assertEqual(wire['messages'],payload['messages'])
                        self.assertNotIn('repetition_penalty',wire)
                        self.assertFalse(wire['chat_template_kwargs']['enable_thinking'])
                outcomes = list(Path(tmp).rglob('*.outcome.json'))
                self.assertEqual(len(outcomes),1)
                self.assertEqual(json.loads(outcomes[0].read_text())['response'],body)


if __name__ == '__main__':
    unittest.main()
