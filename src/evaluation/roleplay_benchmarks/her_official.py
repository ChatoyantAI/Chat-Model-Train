"""HER CoSER prompt/history/judge semantics on the shared evaluation transport.

Upstream is pinned in her_upstream/provenance.json. The frozen CoSER source is
never edited: only a new run's worker copy is patched, with fail-closed anchors.
"""
from copy import deepcopy

from .her_upstream import prompts, utils
from .her_coser import valid_actor_content

PROTOCOL = 'her-coser-official-v2'
MODES = ('with-system', 'without-system')


def actor_payload(payload, mode):
    """Official HER uses raw Qwen completions; her_nosys uses server chat."""
    result = deepcopy(payload)
    if mode == 'with-system':
        result['prompt'] = utils.apply_qwen_chat_template(result.pop('messages'))
        # Raw ChatML has no native Qwen <think> prefix and bypasses the server
        # chat template. HER system_thinking is ordinary generated role text.
        for key in ('chat_template_kwargs', 'thinking', 'reasoning_effort'):
            result.pop(key, None)
        result['stop'] = utils.get_stop_tokens('her')
    elif mode == 'without-system':
        # Do not inject upstream's 1.15 sampling override. Keep the endpoint
        # default unless the caller explicitly supplies a repetition penalty.
        result['stop'] = ['<|im_end|>', '<|endoftext|>']
    else:
        raise ValueError('Unknown HER system-thinking mode')
    return result


def response_content(body):
    choice = body['choices'][0]
    return choice['message']['content'] if 'message' in choice else choice['text']


def deliver_actor_response(body, mode='with-system'):
    """Validate the role reply, but preserve ALL raw tags in self history."""
    content = response_content(body)
    valid_actor_content(content, strict=True)
    if mode == 'without-system':
        # Upstream VLLMModel.chat strips system/native analysis before returning
        # to CharacterAgent. Role-thinking/action markup must remain intact.
        content = utils.remove_system_thinking(content)
    result = deepcopy(body)
    result['choices'][0]['message'] = {'role': 'assistant', 'content': content}
    return result


def recipient_view(content):
    cleaned = utils.remove_system_thinking(utils.remove_long_role_thinking(content))
    return utils.remove_inner_thoughts(cleaned)


def judge_view(content, speaker=None):
    cleaned = utils.convert_her_format(utils.remove_system_thinking(content))
    return utils.add_speaker_name(cleaned, speaker) if speaker else cleaned


class Agent:
    """Synchronous equivalent of HER CharacterAgent using our cached transport."""
    def __init__(self, model, name, database=None, system_prompt='', retrieval_target=None):
        if database or retrieval_target:
            raise ValueError('HER official adapter supports no-RAG evaluation only')
        self.model, self.name = model, name
        self.messages = [{'role': 'system', 'content': system_prompt}]

    def update(self, role, content, name=None):
        self.messages.append({'role': role, 'content': content, **({'name': name} if name else {})})

    def chat(self):
        from utils import get_response_with_retry  # isolated worker's cached API
        response = get_response_with_retry(model=self.model, messages=self.messages, max_tokens=8192)
        if not isinstance(response, str) or not response.strip():
            raise ValueError('Empty HER agent reply')
        return response


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Pinned CoSER/HER integration anchor changed: ' + old[:100])
    return text.replace(old, new, 1)


def patch_main(text, mode):
    if mode not in MODES:
        raise ValueError('Unknown HER mode')
    model_type = 'her' if mode == 'with-system' else 'her_nosys'
    imports = (
        'from functools import partial\n'
        'from src.evaluation.roleplay_benchmarks.her_official import Agent, recipient_view, judge_view\n'
        'from src.evaluation.roleplay_benchmarks.her_upstream.prompts import '
        'get_character_prompt, get_environment_prompt, get_nsp_prompt\n'
        'from src.evaluation.roleplay_benchmarks.her_upstream.utils import '
        'parse_nsp_response, remove_system_thinking, add_speaker_name\n'
        f'get_character_prompt = partial(get_character_prompt, model_type={model_type!r})\n'
    )
    text = _replace(text, 'random.seed(42)', imports + 'random.seed(42)')
    text = _replace(text, '                from utils import add_speaker_name\n', '')
    text = _replace(text, 'next_actor = extract_nsp(response)',
                    'next_actor = parse_nsp_response(response, speaking_characters_w_env)')
    old = """                    if not invalid_nsp:
                        current_agent.update('assistant', response)
                    else:
                        current_agent.update('assistant', next_actor)
                        # reinput system prompt 
                        current_agent.update('user', current_agent.messages[0]['content'])"""
    text = _replace(text, old, "                    current_agent.update('assistant', next_actor)")
    text = _replace(text, '                    # response = add_speaker_name(response, actor)',
                    "                    if actor == ENVIRONMENT:\n                        response = remove_system_thinking(response)")
    text = _replace(text, "other_agent.update('assistant', response)",
                    "other_agent.update('assistant', response, name=actor)")
    text = _replace(text, "other_agent.update('user', add_speaker_name(remove_inner_thoughts(response), actor))",
                    "other_agent.update('user', add_speaker_name(recipient_view(response), actor), name=actor)")
    text = _replace(text, "remove_inner_thoughts(m['content'])", "judge_view(m['content'], m['role'])")
    text = _replace(text, "remove_inner_thoughts(m['message'])", "judge_view(m['message'])")
    text = _replace(text, 'from prompts import critic_prompts',
                    'from src.evaluation.roleplay_benchmarks.her_upstream.prompts import CRITIC_PROMPTS as critic_prompts')
    # HER uses str.format rather than a sequence of placeholder substitutions.
    start = '            critic_prompt = '
    line = next(line for line in text.splitlines() if line.startswith(start))
    text = _replace(text, line, start + "critic_prompts['self-play-deduct-template'].format("
        "book=book_title, plot_summary=circumstance['plot']['summary'], scenario=scenario_str, "
        "character_profiles=character_profile_str, original_conversation=reference_str, "
        "major_characters=', '.join(major_characters), additional_instructions=additional_instructions, "
        "dimension_name=dimension, dimension_brief=critic_prompts['dimension_details'][dimension]['dimension_brief'], "
        "dimension_criteria=critic_prompts['dimension_details'][dimension]['dimension_criteria'])")
    return text


def patch_adapter(text):
    # HER appends each message separately, including consecutive user turns.
    start = '        merged = []\n'
    end = '        payload, remote, is_helper = request_settings'
    a, b = text.index(start), text.index(end)
    if b <= a:
        raise ValueError('Unexpected frozen message adapter')
    return text[:a] + '        merged = copy.deepcopy(messages)\n' + text[b:]
