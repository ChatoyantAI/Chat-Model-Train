"""Read-only audit of saved CoSER results, including nested local evidence files."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .assessment import DIMENSIONS, deduction_score, add_speaker, role_content
from .her_coser import hide_inner_thoughts


def _results(value, location='$'):
    if isinstance(value, dict):
        if isinstance(value.get('cases'), dict) and isinstance(value.get('scores'), dict):
            yield location, value
        else:
            for key, child in value.items():
                yield from _results(child, location + '/' + str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _results(child, location + '/' + str(i))


def audit_files(paths):
    sources, rows, seen = [], [], set()
    for filename in paths:
        path = Path(filename).resolve()
        raw = path.read_bytes()
        sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        for location, result in _results(json.loads(raw)):
            fingerprint = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            for case_id, case in result['cases'].items():
                messages = [m for m in case.get('simulation', []) if m.get('role') != 'NSP']
                actor_messages = [m for m in messages if m.get('role') != 'Environment']
                empty, missing_labels, repetitions = [], [], []
                previous = {}
                for i, message in enumerate(messages):
                    text, speaker = message.get('content'), message.get('role')
                    try:
                        role_content(text, speaker)
                    except (ValueError, TypeError):
                        empty.append(i)
                    if not isinstance(text, str) or not isinstance(speaker, str):
                        continue
                    if add_speaker(text, speaker) != text:
                        missing_labels.append(i)
                    # Diagnostic only: same speaker, identical spoken text after
                    # removing thoughts/actions. Never changes the official score.
                    speech = hide_inner_thoughts(text)
                    speech = re.sub(r'<role_action>.*?</role_action>|\(.*?\)', '', speech, flags=re.S)
                    speech = re.sub(r'^\s*' + re.escape(speaker) + r'\s*[:：]', '', speech)
                    speech = ' '.join(speech.split()).casefold()
                    if speaker != 'Environment' and len(speech) >= 20:
                        key = (speaker, speech)
                        if key in previous:
                            repetitions.append({'turn': i, 'previous_turn': previous[key]})
                        previous[key] = i
                errors, recomputed = [], {}
                for dimension in DIMENSIONS:
                    try:
                        value = deduction_score(case['critique'][dimension], len(actor_messages))
                        stored = case['critique'][dimension]['score']
                        if type(stored) not in (int, float) or not math.isfinite(stored) or abs(stored - value) > 1e-6:
                            raise ValueError('stored score differs from formula')
                        recomputed[dimension] = value
                    except (KeyError, TypeError, ValueError) as exc:
                        errors.append({'dimension': dimension, 'error': str(exc)})
                if len(recomputed) == 4:
                    mean = sum(recomputed.values()) / 4
                    value = case.get('score')
                    if type(value) not in (int, float) or not math.isfinite(value) or abs(value - mean) > 1e-6:
                        errors.append({'dimension': 'avg', 'error': 'stored average differs from dimensions'})
                rows.append({'source': str(path), 'location': location, 'case_id': case_id,
                    'actor_turns': len(actor_messages), 'empty_or_label_only_turns': empty,
                    'missing_leading_speaker_labels': missing_labels, 'exact_spoken_repeats': repetitions,
                    'scoring_errors': errors, 'recomputed_scores': recomputed})
    counts = Counter()
    for row in rows:
        for key in ('empty_or_label_only_turns', 'missing_leading_speaker_labels', 'exact_spoken_repeats', 'scoring_errors'):
            counts[key + '_cases'] += bool(row[key])
            counts[key] += len(row[key])
    return {'audit': 'saved-coser-assessment-20261018', 'model_calls': 0,
            'sources': sources, 'unique_results': len(seen), 'cases': len(rows), 'counts': dict(counts),
            'scope': 'Selected saved artifacts only; not a representative sample of all runs.',
            'limitations': 'Checks saved judge views, not missing raw responses. Repetition is an exact-text diagnostic, not a new scoring rule.',
            'rows': rows}


def save_audit(paths, output):
    result = audit_files(paths)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return {k: v for k, v in result.items() if k != 'rows'}
