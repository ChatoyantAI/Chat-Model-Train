"""Convert one HER response for CoSER judging, preserving the original response.

Rules follow cydu24/HER at 1e5fd856b9e23d3af41445e3c4ca734e24e012ef,
eval_code/benchmarks/multi_turn/coser/utils.py: remove_system_thinking and
convert_her_format. This standalone adapter does not change the frozen evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

UPSTREAM_COMMIT = "1e5fd856b9e23d3af41445e3c4ca734e24e012ef"
UPSTREAM_URL = (
    f"https://github.com/cydu24/HER/blob/{UPSTREAM_COMMIT}/"
    "eval_code/benchmarks/multi_turn/coser/utils.py"
)

# Exact HER branch of upstream get_output_format_instruction.
HER_OUTPUT_INSTRUCTION = """
=Requirements=
Your output should follow this two-part structure in strict order:
1. System Thinking: A single block at the very beginning, wrapped in <system_thinking> and </system_thinking>. This is the third-person analysis of how to portray the role.
2. Role-play Response: The response include the role's thought, speech and action. Use <role_thinking> your thought </role_thinking> and <role_action> your action</role_action> as needed. The role's thought should be invisible to others.
"""


def remove_system_thinking(text: str) -> str:
    """Match upstream removal, including its handling of truncated blocks."""
    text = re.sub(
        r'<\s*system[_\s]*think(?:ing)?\s*>.*?</\s*system[_\s]*think(?:ing)?\s*>',
        '', text, flags=re.DOTALL,
    )
    text = re.sub(r'<\s*system[_\s]*think(?:ing)?\s*>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'</\s*system[_\s]*think(?:ing)?\s*>', '', text)
    for tag in ('think', 'thought', 'role_think'):
        text = re.sub(rf'<{tag}>.*?</{tag}>', '', text, flags=re.DOTALL)
        text = re.sub(rf'<{tag}>.*', '', text, flags=re.DOTALL)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return re.sub(r'\n+', '\n', text).strip()


def convert_her_to_coser(text: str) -> str:
    """Remove system analysis; retain role thoughts, actions and speech."""
    text = remove_system_thinking(text)
    for tag, opening, closing in (
        ('role_thinking', '[', ']'),
        ('role_action', '(', ')'),
        ('role_speech', '', ''),
    ):
        text = text.replace(f'<{tag}>', opening).replace(f'</{tag}>', closing)
    return text.strip()


def conversion_record(text: str) -> dict:
    converted = convert_her_to_coser(text)
    return {
        'protocol': 'her-official-format-to-coser-v1',
        'upstream': UPSTREAM_URL,
        'raw_content': text,
        'coser_content': converted,
        'empty_after_conversion': not bool(converted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-format', choices=('text', 'openai-json'), default='text')
    parser.add_argument('--output', type=Path, required=True, help='New JSON artifact; never overwrites a file')
    args = parser.parse_args()
    text = args.input.read_text(encoding='utf-8')
    if args.input_format == 'openai-json':
        text = json.loads(text)['choices'][0]['message']['content']
    if not isinstance(text, str):
        raise ValueError('Response content must be a string')
    record = conversion_record(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    # An empty converted response is evidence, not a valid scored answer.
    return 2 if record['empty_after_conversion'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
