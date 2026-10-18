"""Reviewed scoring semantics, separate from the frozen historical protocols.

No API calls and no changes to benchmark questions, sampling, or score formulae.
Only explicitly versioned new runs use these patches; old results stay intact.
"""
from copy import deepcopy
import json
import re
import statistics

from .her_format import convert_her_to_coser
from .her_coser import hide_inner_thoughts

VERSION = '20261018'
DIMENSIONS = ('Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality')


def validate(config):
    value = config.get('assessment')
    if value is None:
        return
    if (not isinstance(value, dict) or set(value) != {'version', 'coser_role_thoughts'}
            or value['version'] != VERSION or value['coser_role_thoughts'] not in ('include', 'exclude')):
        raise ValueError('assessment requires version=20261018 and coser_role_thoughts=include/exclude')


def contract(config):
    """Make the limits of comparison visible alongside scores, not just in docs."""
    reviewed = bool(config.get('assessment'))
    return {
        'version': VERSION if reviewed else 'historical',
        'coser_role_thoughts': config.get('assessment', {}).get('coser_role_thoughts', 'protocol-dependent'),
        'coser_speaker_labels': 'all messages from structured role' if reviewed else 'protocol-dependent',
        'coser_repetition_cleanup': 'disabled; preserve generated evidence' if reviewed else 'protocol-dependent',
        'coser_rubric': 'pinned HER version of CoSER rubric, valid JSON example, resolved major_characters' if reviewed else 'protocol-dependent',
        'coser_score_formula': 'clip(100 - 5 * (sum(severity) - 0.3 * actor_turns), 0, 100)',
        'actor_protocol': config.get('protocol', 'frozen-gemma-glm-v1'),
        'actor_prompt_format': config.get('actor_prompt_format'),
        'her_system_thinking': config.get('her_system_thinking'),
        'published_leaderboard_comparable': False,
        'limitations': [
            'GLM and other replacement judges are not the original leaderboard judges.',
            'CoSER native and HER actor prompts/history differ; score changes alone do not isolate training effects.',
            'MiniMax simulator, judge wording and score anchors are local and not human-calibrated.',
            'RMT public-only coverage excludes SEC/Poison; Claude35 and Claude45 are distinct datasets.',
        ],
    }


def add_speaker(text, speaker):
    # A mention later in the text is not a leading speaker label. Preserve it.
    prefix = r'^\s*(?:\*\*)?' + re.escape(speaker) + r'(?:\*\*)?\s*[:：](?:\*\*)?'
    return text if re.match(prefix, text) else f'{speaker}: {text}'


def role_content(text, speaker=None):
    if not isinstance(text, str):
        raise ValueError('Non-text CoSER response')
    text = convert_her_to_coser(text).strip()
    body = text
    if speaker:
        body = re.sub(r'^\s*(?:\*\*)?' + re.escape(speaker) + r'(?:\*\*)?\s*[:：](?:\*\*)?', '', body).strip()
    # Empty tags, a bare speaker label, or analysis alone cannot earn a score.
    if (not body or re.fullmatch(r'[^:\n<>]{1,160}:\s*', body)
            or re.fullmatch(r'(?:(?:\[\s*\]|\(\s*\))\s*)+', body)):
        raise ValueError('CoSER response contains no role content')
    return text


def recipient_view(text):
    text = re.sub(r'<long_role_thinking>(.*?)</long_role_thinking>',
                  r'<role_thinking>\1</role_thinking>', text, flags=re.S)
    return hide_inner_thoughts(text)


def judge_messages(messages, thoughts, reference=False):
    role_key, text_key = ('character', 'message') if reference else ('role', 'content')
    result = []
    for message in messages:
        speaker = message[role_key]
        if speaker == 'NSP':
            continue
        text = convert_her_to_coser(message[text_key])
        if not reference:
            role_content(text, speaker)
        if thoughts == 'exclude':
            text = hide_inner_thoughts(text)
            # A thought-only response may become empty in this view. Mark that
            # scene invalid instead of treating a blank as a successful reply.
            if not reference:
                role_content(text, speaker)
        result.append({**message, text_key: add_speaker(text, speaker)})
    if not reference and not any(m[role_key] != 'Environment' for m in result):
        raise ValueError('No actor turns to judge')
    return result


def validate_critique(obj, dimension):
    if not isinstance(obj, dict) or set(obj) != {dimension} or not isinstance(obj[dimension], dict):
        raise ValueError('Judge must return exactly the requested dimension')
    value = obj[dimension]
    flaws = value.get('flaws')
    if not isinstance(flaws, list):
        raise ValueError('Missing flaws list')
    for flaw in flaws:
        if not isinstance(flaw, dict) or type(flaw.get('severity')) is not int or not 1 <= flaw['severity'] <= 10:
            raise ValueError('Missing or invalid severity; no default deduction')
    return obj


def extract_critique(text, **kwargs):
    """One JSON object only (optional fences); duplicate keys are invalid."""
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate judge key')
            value[key] = item
        return value
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.S)
        if not match:
            raise ValueError('Malformed judge JSON fence')
        text = match.group(1).strip()
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite judge value')))


def deduction_score(critique, actor_turns):
    if type(actor_turns) is not int or actor_turns < 1:
        raise ValueError('No actor turns to score')
    validate_critique({'dimension': critique}, 'dimension')
    return max(0, min(100 - (sum(f['severity'] for f in critique['flaws']) - .3 * actor_turns) * 5, 100))


def judge_parser(payload):
    messages = payload.get('messages', [])
    prompt = messages[0].get('content', '') if messages else ''
    dimensions = [d for d in DIMENSIONS if f'in terms of {d}' in prompt]
    if len(dimensions) == 1 and '"flaws"' in prompt:
        return lambda text: validate_critique(extract_critique(text), dimensions[0])
    return lambda text: text


def build_critic_prompt(dimension, **fields):
    # HER fixes the malformed JSON example in the original CoSER template;
    # the four rubric meanings are unchanged. Use it for BOTH actor formats.
    from .her_upstream.prompts import CRITIC_PROMPTS
    spec = CRITIC_PROMPTS['dimension_details'][dimension]
    criteria = spec['dimension_criteria'].replace('{major_characters}', fields['major_characters'])
    return CRITIC_PROMPTS['self-play-deduct-template'].format(
        **fields, dimension_name=dimension, dimension_brief=spec['dimension_brief'],
        dimension_criteria=criteria)


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Reviewed CoSER source anchor changed: ' + old[:80])
    return text.replace(old, new, 1)


def patch_main(text, thoughts):
    """Applied after the chosen native/HER prompt adapter in a new worker copy."""
    if thoughts not in ('include', 'exclude'):
        raise ValueError('Choose role thought visibility')
    imports = ('from src.evaluation.roleplay_benchmarks.assessment import '
               'judge_messages, validate_critique, extract_critique, deduction_score, recipient_view, build_critic_prompt\n')
    text = _replace(text, 'random.seed(42)', imports + 'random.seed(42)')
    start = text.index('        # Remove inner thoughts for fair comparison')
    end = text.index('        logger.info(f\'===Simulation', start)
    text = text[:start] + (
        f'        simulation = judge_messages(simulation, {thoughts!r})\n'
        f'        reference = judge_messages(reference, {thoughts!r}, reference=True)\n'
        "        simulation_str = '\\n\\n'.join(m['content'] for m in simulation)\n"
        "        reference_str = '\\n\\n'.join(m['message'] for m in reference)\n\n"
    ) + text[end:]
    start = text.index('        def parse_response(response, **kwargs):')
    end = text.index("        logger.info(f'{book_title}", start)
    text = text[:start] + (
        '        def parse_response(response, **kwargs):\n'
        '            return validate_critique(response, dimension)\n\n'
    ) + text[end:]
    text = _replace(text, '[extract_json, parse_response]', '[extract_critique, parse_response]')
    line = next(line for line in text.splitlines() if line.startswith('            critic_prompt = '))
    text = _replace(text, line, '            critic_prompt = build_critic_prompt(dimension, '
        "book=book_title, plot_summary=circumstance['plot']['summary'], scenario=scenario_str, "
        'character_profiles=character_profile_str, original_conversation=reference_str, '
        "major_characters=', '.join(major_characters), additional_instructions=additional_instructions)")
    line = next(line for line in text.splitlines() if "res[dimension]['score'] = max(" in line)
    text = _replace(text, line, "            res[dimension]['score'] = deduction_score(res[dimension], actor_rounds)")
    # Reproducible NSP fallback across fresh subprocesses/PYTHONHASHSEEDs.
    text = _replace(text, 'candidates = list(candidates)', 'candidates = sorted(candidates)')
    return text


def patch_agent(text):
    # Native CoSER cleans repetition conditionally by model name; HER does not.
    # Keep raw repetitions for all candidates so the judge sees the same evidence.
    old = "if not any(self.model.lower().startswith(model_type) for model_type in ['gpt', 'claude']) and self.name != 'NSP':"
    return _replace(text, old, 'if False:  # Reviewed protocol: preserve repetition evidence for the judge.')


def review_public_summary(summary, manifest, cases):
    """Complete-session headlines; retain all partial information separately."""
    if not manifest.get('assessment'):
        return summary
    result = deepcopy(summary)
    result['partial_scores'] = result['scores']
    result['assessment'] = manifest['assessment']
    result['score_basis'] = 'complete scenes; MiniMax includes a seed only after every configured repeat'
    rows = {r['id']: r for r in result['cases']}
    config = manifest['config']
    scores = {}
    expected_units = complete_units = 0
    for language in config['languages']:
        selected = [c for c in cases if c['language'] == language]
        if config['benchmark'] == 'minimax':
            from .community_protocols import MINI_DIMS
            seeds = {}
            for case in selected:
                seeds.setdefault(case['source_id'], []).append(case)
            complete_seeds = [group for group in seeds.values() if len(group) == config['repeats']
                              and all(rows[c['id']]['complete'] for c in group)]
            expected_units += len(seeds)
            complete_units += len(complete_seeds)
            seed_means = [{d: statistics.mean(rows[c['id']]['scores'][d] for c in group)
                           for d in MINI_DIMS} for group in complete_seeds]
            dims = {d: statistics.mean(s[d] for s in seed_means) for d in MINI_DIMS} if seed_means else {}
            scores[language] = {'dimensions': dims, 'complete_seeds': len(seed_means),
                                'included_sessions': sum(map(len, complete_seeds)),
                                'expected_repeats_per_seed': config['repeats'],
                                'calibration': 'local scoring anchors; not calibrated to published MiniMax scores'}
            if dims:
                worlds = statistics.mean(dims[d] for d in ('basics', 'logic', 'knowledge'))
                stories = statistics.mean(dims[d] for d in ('diversity', 'content_logic'))
                scores[language].update(Worlds=worlds, Stories=stories, Preferences=dims['interaction'],
                                        overall=.5*worlds + .25*stories + .25*dims['interaction'])
        else:
            expected_units += len(selected)
            complete_units += sum(rows[c['id']]['complete'] for c in selected)
            # Weighting remains the historical per-judgment mean, but every
            # included scene must have all of its required judgments.
            values = {}
            for case in selected:
                row = rows[case['id']]
                if row['complete']:
                    for dim, value in row.get('scores', {}).items():
                        count = len(case['turns']) if dim not in ('CM', 'UPA') else sum(
                            t['field'] == 'character_maintenance' if dim == 'CM' else 'reference_answer' in t
                            for t in case['turns'])
                        if count:
                            values.setdefault(dim, []).append((value, count))
            scores[language] = {}
            for dim, units in values.items():
                count = sum(n for _, n in units)
                mean = sum(v*n for v, n in units) / count
                scores[language][dim] = {'mean': mean, 'count': count,
                    'scale': '0/1' if dim in ('CM', 'UPA') else '1-5',
                    'display_0_100': mean * (100 if dim in ('CM', 'UPA') else 20)}
    result['scores'] = scores
    result['score_coverage'] = {'complete': complete_units == expected_units and expected_units > 0,
                                'included_units': complete_units, 'expected_units': expected_units,
                                'unit': 'seed with every configured repeat' if config['benchmark'] == 'minimax' else 'scene'}
    result['partial_results'] = result.get('partial_results', False) or not result['score_coverage']['complete']
    result['missing_cases'] = [r['id'] for r in result['cases'] if not r['complete']]
    return result
