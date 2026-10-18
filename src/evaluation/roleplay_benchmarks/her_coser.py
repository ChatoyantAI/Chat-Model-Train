"""Output-only HER conversion and recipient visibility for CoSER."""
from copy import deepcopy
import re

from .her_format import convert_her_to_coser, remove_system_thinking, HER_OUTPUT_INSTRUCTION

PROTOCOL = 'her-coser-format-v1'

COSER_OUTPUT_INSTRUCTION = (
    "Your output should include **thought**, **speech**, and **action**. "
    "Use [your thought] for thoughts, which others can't see, e.g. [I'm terrified, but I must appear strong.]. "
    "Use (your action) for actions, which others can see, such as (watches silently, trying to control her fear and anger)."
)


HER_NO_SYSTEM_INSTRUCTION = """=Requirements=
Your output should contain only the role-play response. The response include the role's thought, speech and action. Use <role_thinking> your thought </role_thinking> and <role_action> your action</role_action> as needed. The role's thought should be invisible to others."""


def actor_prompt_payload(payload, mode='her-official-v1'):
    """Replace only the pinned CoSER actor format block; retain all other bytes."""
    result = deepcopy(payload)
    messages = result.get('messages', [])
    if not messages or messages[0].get('role') != 'system':
        raise ValueError('HER actor format requires the CoSER system message')
    text = messages[0]['content']
    old = '===Requirements===\n' + COSER_OUTPUT_INSTRUCTION
    if text.count(old) != 1:
        raise ValueError('Expected exactly one pinned CoSER output-format block')
    if mode == 'her-official-v1':
        instruction = HER_OUTPUT_INSTRUCTION.strip('\n')
    elif mode == 'her-no-system-v1':
        instruction = HER_NO_SYSTEM_INSTRUCTION
    else:
        raise ValueError('Unknown HER prompt format')
    messages[0]['content'] = text.replace(old, instruction, 1)
    return result


def patch_judge_source(text):
    """Keep role thoughts for LLM judging; retain old BLEU/ROUGE inputs."""
    replacements = [
        ("remove_inner_thoughts(m['content'])", "m['content']"),
        ("remove_inner_thoughts(m['message'])", "m['message']"),
        ('calculate_bleu_rouge(reference[args.continue_from:], simulation[args.continue_from:])',
         "calculate_bleu_rouge([m if m['character'] == ENVIRONMENT else "
         "{**m, 'message': remove_inner_thoughts(m['message'])} for m in reference[args.continue_from:]], "
         "[m if m['role'] == ENVIRONMENT else {**m, 'content': remove_inner_thoughts(m['content'])} "
         "for m in simulation[args.continue_from:]])"),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('Unexpected CoSER judging source for HER output adapter')
        text = text.replace(old, new)
    return text


def valid_actor_content(content, strict=False, native=False):
    if strict and not native:
        # Never silently discard speech following an incorrectly closed analysis
        # block. Preserve the raw outcome in the journal and let the worker retry.
        stack = []
        for match in re.finditer(r'<\s*(/?)\s*(system[_\s]*think(?:ing)?|role_thinking|role_action|role_speech|think|thought|role_think)\s*>', content):
            closing, name = match.groups()
            name = re.sub(r'[_\s]', '', name)
            if not closing:
                stack.append(name)
            elif not stack or stack.pop() != name:
                raise ValueError('Malformed HER tags; raw response retained for retry')
        if stack:
            raise ValueError('Unclosed HER tags; raw response retained for retry')
    converted = content.strip() if native else convert_her_to_coser(content)
    if not converted:
        raise ValueError('HER response contains no role reply after removing system thinking')
    if strict and re.fullmatch(r'[^:\n<>]{1,160}:\s*', converted):
        raise ValueError('Actor response contains only a speaker name')
    return converted


def actor_response(body, strict=False, native=False):
    result = deepcopy(body)
    result['choices'][0]['message']['content'] = valid_actor_content(
        result['choices'][0]['message']['content'], strict=strict, native=native)
    return result


def hide_inner_thoughts(content):
    """Recipient visibility, including multiline thoughts missed by old CoSER."""
    content = remove_system_thinking(content)
    content = re.sub(r'<role_thinking>.*?</role_thinking>', '', content, flags=re.DOTALL)
    return re.sub(r'\[.*?\]', '', content, flags=re.DOTALL).strip()
