#!/home/gyr/lusychat-training/.venv/bin/python
from __future__ import annotations

import bisect
import collections
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path('/home/gyr/lusychat-training')
UNION = BASE / 'judge-dialogue-union-50pct-v1'
FINAL = BASE / 'judge-dialogue-selection-v1'
SCHEMA_VERSION = 'lusy-dialogue-selection-v1'
SEED = 'lusy-dialogue-selection-v1-20260727'
BUDGET = 20_000
CREATED_AT = datetime.now(timezone.utc).isoformat()

FEATURES = UNION / '03_screening/dialogue_screening_features.parquet'
MATERIAL = UNION / '01_input/dialogue_material.parquet'
TURN_MANIFEST = UNION / '01_input/turn_manifest.parquet'
REPETITION_ROOT = BASE / 'lusy-chat-4m-v2/02_features/assistant_repetition_v1/01_features'
REPETITION_FEATURES = REPETITION_ROOT / 'assistant_repetition_features.parquet'
REPETITION_AUTO_NATIVE = REPETITION_ROOT / 'excluded_native_targets.parquet'
REPETITION_REVIEW_NATIVE = REPETITION_ROOT / 'review_native_targets.parquet'
VALID_JSONL_SOURCES = [
    BASE / 'judge-qwen36-35b-dialogue-evidence-v2/01_results_25pct/valid.jsonl',
    BASE / 'judge-glm47-flash-dialogue-evidence-v2/01_results_second_25pct/valid.jsonl',
]

CORE_DIMS = [
    'character_fidelity',
    'context_coherence',
    'user_response_relevance',
    'story_progression',
    'writing_quality',
    'trajectory_health',
]
ALL_DIMS = [
    'character_fidelity',
    'context_coherence',
    'greeting_style_adherence',
    'user_response_relevance',
    'story_progression',
    'writing_quality',
    'trajectory_health',
]
CORE_WEIGHTS = {
    'character_fidelity': 0.28,
    'context_coherence': 0.18,
    'user_response_relevance': 0.12,
    'story_progression': 0.18,
    'writing_quality': 0.12,
    'trajectory_health': 0.12,
}
SERIOUS_ANOMALIES = {
    'missing_core_audited_score',
    'quality_floor_3_with_critical_flag',
    'health_3_with_health_failure_flag',
    'quality_floor_3_with_major_weakest_turn',
}
EVIDENCE_FLAG_MARKERS = (
    'quote_mismatch',
    'sustained_evidence_gate_failed',
    'major_evidence_gate_failed',
    'score_arithmetic_mismatch',
)
IDENTITY_META_TYPES = {
    'ai_identity_break',
    'meta_refusal',
    'out_of_character',
}
CORE_POOL_NAMES = {
    'core_claude',
    'core_flash_liked',
    'liked_strong_known',
}


def compact(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':')
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(dialogue_id: str) -> str:
    return hashlib.sha256(
        f'{SEED}|{dialogue_id}'.encode('utf-8')
    ).hexdigest()


def safe_json_list(value: Any) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def safe_json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def percentile_average(values: list[float]) -> list[float]:
    ordered = sorted(values)
    total = len(ordered)
    if not total:
        return []
    return [
        (
            bisect.bisect_left(ordered, value)
            + bisect.bisect_right(ordered, value)
        )
        / (2.0 * total)
        for value in values
    ]


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = collections.Counter(str(row.get(field)) for row in rows)
    return dict(sorted(counter.items()))


def quantiles(values: list[float]) -> dict[str, float | None]:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return {'min': None, 'p10': None, 'median': None, 'p90': None, 'max': None, 'mean': None}
    def q(frac: float) -> float:
        return clean[min(len(clean) - 1, max(0, int(round(frac * (len(clean) - 1)))))]
    return {
        'min': clean[0],
        'p10': q(0.10),
        'median': q(0.50),
        'p90': q(0.90),
        'max': clean[-1],
        'mean': sum(clean) / len(clean),
    }


def selection_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    quality = row.get('final_quality_score')
    raw = row.get('quality_composite_raw')
    return (
        -(float(quality) if quality is not None else -1.0),
        -(float(raw) if raw is not None else -1.0),
        row['selection_stable_hash'],
        row['dialogue_id'],
    )


def prepare(staging: Path) -> None:
    if FINAL.exists():
        raise FileExistsError(f'final output already exists: {FINAL}')
    if not UNION.exists():
        raise FileNotFoundError(UNION)
    if staging.parent.resolve() != BASE.resolve():
        raise RuntimeError(f'unsafe staging parent: {staging}')
    if not staging.name.startswith('.judge-dialogue-selection-v1.staging-'):
        raise RuntimeError(f'unsafe staging name: {staging}')
    staging.mkdir(parents=True, exist_ok=True)
    for name in ['00_manifest', '01_ranking', '02_tiers', '03_reports', 'scripts']:
        (staging / name).mkdir(exist_ok=True)


def load_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = pq.read_table(FEATURES).to_pylist()
    material_columns = [
        'dialogue_id',
        'candidate_target_ids',
        'candidate_message_ids',
        'candidate_pool_counts_json',
        'candidate_language_counts_json',
        'candidate_topic_counts_json',
        'candidate_turn_bucket_counts_json',
        'candidate_length_bucket_counts_json',
        'candidate_guideline_counts_json',
    ]
    material_rows = pq.read_table(MATERIAL, columns=material_columns).to_pylist()
    material = {row['dialogue_id']: row for row in material_rows}
    if len(rows) != 57_874 or len(material) != 57_874:
        raise RuntimeError(f'unexpected input size: features={len(rows)} material={len(material)}')
    ids = [row['dialogue_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError('duplicate dialogue_id in screening features')
    if set(ids) != set(material):
        raise RuntimeError('feature/material dialogue_id closure mismatch')
    return rows, material



def attach_reason_audit(rows: list[dict[str, Any]]) -> None:
    by_id = {row['dialogue_id']: row for row in rows}
    seen: set[str] = set()
    for path in VALID_JSONL_SOURCES:
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                dialogue_id = str(obj['dialogue_id'])
                if dialogue_id in seen:
                    raise RuntimeError(f'duplicate valid dialogue audit: {dialogue_id}')
                row = by_id.get(dialogue_id)
                if row is None:
                    raise RuntimeError(f'valid JSONL dialogue missing from union: {dialogue_id}')
                if row.get('judge_status') != 'valid':
                    raise RuntimeError(f'valid JSONL conflicts with terminal status: {dialogue_id}')
                reason_audit = obj.get('reason_audit') or {}
                summaries: dict[str, Any] = {}
                for dim in ALL_DIMS:
                    audit = reason_audit.get(dim) or {}
                    baseline = audit.get('baseline_score')
                    positive = audit.get('audited_positive_adjustment')
                    negative = audit.get('audited_negative_adjustment')
                    audited_score = audit.get('audited_score')
                    if audited_score is None:
                        audited_score = row.get(f'audited_{dim}')
                    computed = None
                    if all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in [baseline, positive, negative]
                    ):
                        computed = int(baseline) + int(positive) + int(negative)
                    score_aligned = bool(
                        computed is not None
                        and isinstance(audited_score, int)
                        and computed == int(audited_score)
                        and bool(audit.get('score_consistent', True))
                    )
                    positive_turns = [int(x) for x in (audit.get('positive_verified_turns') or [])]
                    negative_turns = [int(x) for x in (audit.get('negative_verified_turns') or [])]
                    cited_turns = [int(x) for x in (audit.get('cited_turns') or [])]
                    model_score = row.get(f'model_{dim}')
                    changed = bool(
                        model_score is not None
                        and audited_score is not None
                        and model_score != audited_score
                    )
                    summary = {
                        'baseline_score': baseline,
                        'audited_positive_adjustment': positive,
                        'positive_verified_turns': positive_turns,
                        'audited_negative_adjustment': negative,
                        'negative_verified_turns': negative_turns,
                        'computed_score': computed,
                        'audited_score': audited_score,
                        'reason_score_aligned': score_aligned,
                    }
                    summaries[dim] = summary
                    row[f'audit_{dim}_baseline_score'] = baseline
                    row[f'audit_{dim}_positive_verified_turns'] = positive_turns
                    row[f'audit_{dim}_negative_verified_turns'] = negative_turns
                    row[f'audit_{dim}_cited_turns'] = cited_turns
                    row[f'audit_{dim}_audited_positive_adjustment'] = positive
                    row[f'audit_{dim}_audited_negative_adjustment'] = negative
                    row[f'audit_{dim}_computed_score'] = computed
                    row[f'audit_{dim}_reason_characters'] = audit.get('reason_characters')
                    row[f'audit_{dim}_evidence_gate_passed'] = audit.get('evidence_gate_passed')
                    row[f'audit_{dim}_quote_match_passed'] = audit.get('quote_match_passed')
                    row[f'audit_{dim}_reason_score_aligned'] = score_aligned
                    row[f'audit_{dim}_model_vs_audited_changed'] = changed
                    row[f'audit_{dim}_audited_reason_summary_json'] = compact(summary)
                row['audited_reason_summary_json'] = compact(summaries)
                row['all_applicable_reasons_score_aligned'] = all(
                    summaries[dim]['reason_score_aligned']
                    for dim in ALL_DIMS
                    if row.get(f'audited_{dim}') is not None
                )
                seen.add(dialogue_id)
    expected = {
        row['dialogue_id'] for row in rows if row.get('judge_status') == 'valid'
    }
    if seen != expected:
        raise RuntimeError(
            f'valid reason audit closure mismatch: seen={len(seen)} expected={len(expected)} '
            f'missing={len(expected-seen)} extra={len(seen-expected)}'
        )
    for row in rows:
        if row['dialogue_id'] in seen:
            continue
        row['audited_reason_summary_json'] = '{}'
        row['all_applicable_reasons_score_aligned'] = None
        for dim in ALL_DIMS:
            for suffix, value in [
                ('baseline_score', None),
                ('positive_verified_turns', []),
                ('negative_verified_turns', []),
                ('cited_turns', []),
                ('audited_positive_adjustment', None),
                ('audited_negative_adjustment', None),
                ('computed_score', None),
                ('reason_characters', None),
                ('evidence_gate_passed', None),
                ('quote_match_passed', None),
                ('reason_score_aligned', None),
                ('model_vs_audited_changed', None),
                ('audited_reason_summary_json', '{}'),
            ]:
                row[f'audit_{dim}_{suffix}'] = value


def attach_repetition_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    union_ids = {row['dialogue_id'] for row in rows}
    by_id = {row['dialogue_id']: row for row in rows}
    direct: dict[str, dict[str, Any]] = {
        dialogue_id: {
            'max_raw_edit': None,
            'max_body_edit': None,
            'max_raw_jaccard': None,
            'max_body_jaccard': None,
            'max_copy_fraction': None,
        }
        for dialogue_id in union_ids
    }
    direct_columns = [
        'dialogue_id',
        'raw_edit_similarity',
        'body_edit_similarity',
        'raw_char5_jaccard',
        'body_char5_jaccard',
        'raw_current_copy_fraction',
        'body_current_copy_fraction',
    ]
    for batch in pq.ParquetFile(REPETITION_FEATURES).iter_batches(
        batch_size=131_072, columns=direct_columns
    ):
        for item in batch.to_pylist():
            dialogue_id = item['dialogue_id']
            if dialogue_id not in direct:
                continue
            target = direct[dialogue_id]
            for source, name in [
                ('raw_edit_similarity', 'max_raw_edit'),
                ('body_edit_similarity', 'max_body_edit'),
                ('raw_char5_jaccard', 'max_raw_jaccard'),
                ('body_char5_jaccard', 'max_body_jaccard'),
            ]:
                value = item.get(source)
                if value is not None and math.isfinite(float(value)):
                    target[name] = max(target[name] if target[name] is not None else -1.0, float(value))
            copy_values = [
                float(item[name])
                for name in ['raw_current_copy_fraction', 'body_current_copy_fraction']
                if item.get(name) is not None and math.isfinite(float(item[name]))
            ]
            if copy_values:
                target['max_copy_fraction'] = max(
                    target['max_copy_fraction'] if target['max_copy_fraction'] is not None else -1.0,
                    *copy_values,
                )

    def native_map(path: Path) -> dict[str, dict[str, Any]]:
        return {
            str(item['native_message_id']): item
            for item in pq.read_table(path).to_pylist()
        }

    auto_native = native_map(REPETITION_AUTO_NATIVE)
    review_native = native_map(REPETITION_REVIEW_NATIVE)
    agg: dict[str, dict[str, Any]] = {
        dialogue_id: {
            'auto_ids': set(),
            'review_ids': set(),
            'auto_rules': set(),
            'review_rules': set(),
            'native_max_edit': None,
            'native_max_jaccard': None,
            'native_max_copy': None,
            'turn_manifest_count': 0,
        }
        for dialogue_id in union_ids
    }
    turn_columns = ['dialogue_id', 'native_message_id']
    for batch in pq.ParquetFile(TURN_MANIFEST).iter_batches(
        batch_size=131_072, columns=turn_columns
    ):
        for item in batch.to_pylist():
            dialogue_id = str(item['dialogue_id'])
            native_id = str(item['native_message_id'])
            target = agg[dialogue_id]
            target['turn_manifest_count'] += 1
            for source, kind in [(auto_native, 'auto'), (review_native, 'review')]:
                meta = source.get(native_id)
                if meta is None:
                    continue
                target[f'{kind}_ids'].add(native_id)
                target[f'{kind}_rules'].update(str(x) for x in (meta.get('decision_rules') or []))
                for field, output in [
                    ('max_edit_similarity', 'native_max_edit'),
                    ('max_char5_jaccard', 'native_max_jaccard'),
                    ('max_current_copy_fraction', 'native_max_copy'),
                ]:
                    value = meta.get(field)
                    if value is not None and math.isfinite(float(value)):
                        target[output] = max(
                            target[output] if target[output] is not None else -1.0,
                            float(value),
                        )

    for dialogue_id, row in by_id.items():
        target = agg[dialogue_id]
        metrics = direct[dialogue_id]
        auto_ids = sorted(target['auto_ids'])
        review_ids = sorted(target['review_ids'])
        row['turn_manifest_assistant_turn_count'] = int(target['turn_manifest_count'])
        row['turn_manifest_count_matches_scoreable_turns'] = (
            int(target['turn_manifest_count']) == int(row.get('scoreable_turns') or -1)
        )
        row['repetition_auto_turn_count'] = len(auto_ids)
        row['repetition_review_turn_count'] = len(review_ids)
        row['repetition_auto_native_message_ids_json'] = compact(auto_ids)
        row['repetition_review_native_message_ids_json'] = compact(review_ids)
        row['repetition_auto_rules_json'] = compact(sorted(target['auto_rules']))
        row['repetition_review_rules_json'] = compact(sorted(target['review_rules']))
        row['repetition_auto_duplicate_signal'] = bool(auto_ids)
        row['repetition_review_candidate_signal'] = bool(review_ids)
        edits = [
            value for value in [
                metrics['max_raw_edit'], metrics['max_body_edit'], target['native_max_edit']
            ] if value is not None
        ]
        jaccards = [
            value for value in [
                metrics['max_raw_jaccard'], metrics['max_body_jaccard'], target['native_max_jaccard']
            ] if value is not None
        ]
        copies = [
            value for value in [metrics['max_copy_fraction'], target['native_max_copy']]
            if value is not None
        ]
        row['repetition_max_similarity'] = max(edits) if edits else None
        row['repetition_max_char5_jaccard'] = max(jaccards) if jaccards else None
        row['repetition_max_copy_fraction'] = max(copies) if copies else None

    summary = {
        'auto_dialogues': sum(row['repetition_auto_duplicate_signal'] for row in rows),
        'auto_turns': sum(row['repetition_auto_turn_count'] for row in rows),
        'review_dialogues': sum(row['repetition_review_candidate_signal'] for row in rows),
        'review_turns': sum(row['repetition_review_turn_count'] for row in rows),
        'turn_manifest_rows': sum(row['turn_manifest_assistant_turn_count'] for row in rows),
        'turn_count_mismatch_dialogues': sum(not row['turn_manifest_count_matches_scoreable_turns'] for row in rows),
        'auto_review_dialogue_overlap': sum(
            row['repetition_auto_duplicate_signal'] and row['repetition_review_candidate_signal']
            for row in rows
        ),
    }
    expected = {
        'auto_dialogues': 383,
        'auto_turns': 459,
        'review_dialogues': 165,
        'review_turns': 194,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise RuntimeError(f'repetition union count mismatch {key}: {summary[key]} != {value}')
    summary['join_key'] = 'union turn_manifest.native_message_id -> native target sets -> dialogue_id'
    summary['detector_type'] = 'deterministic assistant repetition detector; independent of Judge flags'
    return summary


def reason_frequencies(rows: list[dict[str, Any]]) -> dict[tuple[str, str], collections.Counter[str]]:
    result: dict[tuple[str, str], collections.Counter[str]] = {}
    for judge in ['qwen', 'glm']:
        for dim in ALL_DIMS:
            result[(judge, dim)] = collections.Counter(
                (row.get(f'reason_{dim}') or '').strip()
                for row in rows
                if row.get('judge_status') == 'valid'
                and row.get('judge_family') == judge
                and row.get(f'audited_{dim}') is not None
            )
    return result


def enrich_rows(
    rows: list[dict[str, Any]],
    material: dict[str, dict[str, Any]],
) -> None:
    frequencies = reason_frequencies(rows)
    for row in rows:
        dialogue_id = row['dialogue_id']
        mat = material[dialogue_id]
        pools = safe_json_dict(mat.get('candidate_pool_counts_json'))
        row['candidate_target_ids'] = mat.get('candidate_target_ids') or []
        row['candidate_message_ids'] = mat.get('candidate_message_ids') or []
        row['candidate_pool_counts_json'] = compact(pools)
        row['candidate_language_counts_json'] = mat.get('candidate_language_counts_json') or '{}'
        row['candidate_topic_counts_json'] = mat.get('candidate_topic_counts_json') or '{}'
        row['candidate_turn_bucket_counts_json'] = mat.get('candidate_turn_bucket_counts_json') or '{}'
        row['candidate_length_bucket_counts_json'] = mat.get('candidate_length_bucket_counts_json') or '{}'
        row['candidate_guideline_counts_json'] = mat.get('candidate_guideline_counts_json') or '{}'
        for pool_name in sorted(CORE_POOL_NAMES):
            row[f'has_{pool_name}'] = int(pools.get(pool_name, 0) or 0) > 0
        row['has_any_core_provenance'] = bool(
            row.get('contains_core')
            or any(int(pools.get(name, 0) or 0) > 0 for name in CORE_POOL_NAMES)
        )
        row['selection_stable_hash'] = stable_hash(dialogue_id)

        status_valid = row.get('judge_status') == 'valid'
        core_values = [row.get(f'audited_{dim}') for dim in CORE_DIMS]
        complete_core = all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in core_values
        )
        quality_fields_complete = bool(
            status_valid
            and complete_core
            and isinstance(row.get('quality_floor'), int)
            and isinstance(row.get('usable_turn_bucket'), int)
        )
        row['quality_fields_complete'] = quality_fields_complete
        if quality_fields_complete:
            weighted = sum(
                CORE_WEIGHTS[dim] * float(row[f'audited_{dim}'])
                for dim in CORE_DIMS
            )
            row['weighted_core_score_0_4'] = weighted
            row['quality_composite_raw'] = (
                0.80 * (weighted / 4.0)
                + 0.12 * (float(row['quality_floor']) / 4.0)
                + 0.08 * (float(row['usable_turn_bucket']) / 4.0)
            )
        else:
            row['weighted_core_score_0_4'] = None
            row['quality_composite_raw'] = None

        anomalies = set(safe_json_list(row.get('screening_anomalies_json')))
        calibration = [str(x) for x in safe_json_list(row.get('calibration_flags_json'))]
        flag_types = set(safe_json_list(row.get('critical_flag_types_json')))
        severities = set(safe_json_list(row.get('critical_flag_severities_json')))
        serious = sorted(anomalies & SERIOUS_ANOMALIES)
        terminal_rejudge = not status_valid
        missing_rejudge = status_valid and not quality_fields_complete
        internal_rejudge = status_valid and bool(serious)
        rejudge_codes: list[str] = []
        if terminal_rejudge:
            rejudge_codes.append(f'terminal:{row.get("judge_status")}')
        if missing_rejudge:
            rejudge_codes.append('missing_required_quality_fields')
        rejudge_codes.extend(f'internal_contradiction:{name}' for name in serious)
        row['terminal_rejudge_signal'] = terminal_rejudge
        row['missing_fields_rejudge_signal'] = missing_rejudge
        row['internal_contradiction_rejudge_signal'] = internal_rejudge
        row['rejudge_required'] = bool(rejudge_codes)
        row['rejudge_reason_codes_json'] = compact(sorted(set(rejudge_codes)))

        critical_review = int(row.get('critical_flag_unique_count') or 0) > 0
        nsfw_review = any(
            item.startswith('assistant_nsfw:')
            and ('mismatch' in item or 'gate_failed' in item)
            for item in calibration
        )
        confidence = row.get('confidence')
        low_confidence = isinstance(confidence, int) and confidence < 3
        greeting = row.get('audited_greeting_style_adherence')
        greeting_low = bool(
            row.get('greeting_applicable')
            and isinstance(greeting, int)
            and greeting <= 1
        )
        identity_meta = bool(flag_types & IDENTITY_META_TYPES)
        row['critical_review_signal'] = critical_review
        row['critical_severity_present'] = 'critical' in severities
        row['major_severity_present'] = 'major' in severities
        row['identity_meta_review_signal'] = identity_meta
        row['nsfw_metadata_review_signal'] = nsfw_review
        row['low_confidence_review_signal'] = low_confidence
        row['greeting_low_review_signal'] = greeting_low
        row['diagnosis_review_only'] = bool((row.get('overall_diagnosis') or '').strip())

        if status_valid:
            applicable = [
                dim for dim in ALL_DIMS
                if row.get(f'audited_{dim}') is not None
            ]
            reasons = [(row.get(f'reason_{dim}') or '').strip() for dim in applicable]
            coverage = (
                sum(bool(reason) for reason in reasons) / len(reasons)
                if reasons else 0.0
            )
            specificity_values: list[float] = []
            uniqueness_values: list[float] = []
            truncated = 0
            for dim, reason in zip(applicable, reasons):
                specificity_values.append(max(0.0, min(1.0, (len(reason) - 20.0) / 160.0)))
                frequency = frequencies[(row['judge_family'], dim)][reason]
                uniqueness_values.append(
                    1.0 - min(1.0, math.log10(max(1, frequency)) / 4.0)
                )
                truncated += int(len(reason) >= 240)
            specificity = (
                sum(specificity_values) / len(specificity_values)
                if specificity_values else 0.0
            )
            template_uniqueness = (
                sum(uniqueness_values) / len(uniqueness_values)
                if uniqueness_values else 0.0
            )
            evidence_issue_count = sum(
                any(marker in item for marker in EVIDENCE_FLAG_MARKERS)
                for item in calibration
            )
            evidence_alignment = 1.0 - min(evidence_issue_count, 14) / 14.0
            model_audited_diffs = [
                abs(float(row[f'model_{dim}']) - float(row[f'audited_{dim}']))
                for dim in applicable
                if row.get(f'model_{dim}') is not None
                and row.get(f'audited_{dim}') is not None
            ]
            score_audit_alignment = (
                1.0 - (sum(model_audited_diffs) / len(model_audited_diffs)) / 4.0
                if model_audited_diffs else None
            )
            reliability = (
                0.30 * coverage
                + 0.20 * specificity
                + 0.20 * template_uniqueness
                + 0.30 * evidence_alignment
            )
            row['reason_applicable_dimension_count'] = len(applicable)
            row['reason_coverage'] = coverage
            row['reason_specificity'] = specificity
            row['reason_template_uniqueness'] = template_uniqueness
            row['reason_evidence_issue_count'] = evidence_issue_count
            row['reason_evidence_alignment'] = evidence_alignment
            row['reason_score_audit_alignment'] = score_audit_alignment
            row['reason_truncation_fraction'] = truncated / len(applicable) if applicable else None
            row['reason_reliability'] = reliability
            row['reason_review_signal'] = bool(
                reliability < 0.55
                or evidence_issue_count >= 7
                or (truncated / len(applicable) if applicable else 0.0) >= 0.75
            )
        else:
            for field in [
                'reason_applicable_dimension_count',
                'reason_coverage',
                'reason_specificity',
                'reason_template_uniqueness',
                'reason_evidence_issue_count',
                'reason_evidence_alignment',
                'reason_score_audit_alignment',
                'reason_truncation_fraction',
                'reason_reliability',
            ]:
                row[field] = None
            row['reason_review_signal'] = True

        review_codes: list[str] = []
        for enabled, name in [
            (critical_review, 'critical_flag'),
            (identity_meta, 'identity_or_meta_flag'),
            (nsfw_review, 'nsfw_metadata_conflict'),
            (low_confidence, 'low_confidence'),
            (greeting_low, 'greeting_applicable_score_le_1'),
            (row['reason_review_signal'], 'reason_reliability_review'),
            (bool(row.get('repetition_auto_duplicate_signal')), 'deterministic_repetition_auto'),
            (bool(row.get('repetition_review_candidate_signal')), 'deterministic_repetition_review'),
        ]:
            if enabled:
                review_codes.append(name)
        row['review_signal_codes_json'] = compact(review_codes)

    for judge in ['qwen', 'glm']:
        complete = [
            row for row in rows
            if row.get('judge_family') == judge
            and row.get('quality_composite_raw') is not None
        ]
        values = [float(row['quality_composite_raw']) for row in complete]
        percentiles = percentile_average(values)
        for row, percentile in zip(complete, percentiles):
            row['quality_percentile_within_judge'] = percentile
            row['final_quality_score'] = 100.0 * percentile
        ordered = sorted(complete, key=selection_sort_key)
        for rank, row in enumerate(ordered, 1):
            row['quality_rank_within_judge'] = rank
    for row in rows:
        row.setdefault('quality_percentile_within_judge', None)
        row.setdefault('final_quality_score', None)
        row.setdefault('quality_rank_within_judge', None)

    ranked = sorted(
        [row for row in rows if row.get('final_quality_score') is not None],
        key=selection_sort_key,
    )
    for rank, row in enumerate(ranked, 1):
        row['global_calibrated_quality_rank'] = rank
    for row in rows:
        row.setdefault('global_calibrated_quality_rank', None)
        quality_base = bool(
            row.get('judge_status') == 'valid'
            and not row['rejudge_required']
            and row['quality_fields_complete']
            and int(row['quality_floor']) >= 2
            and int(row['usable_turn_bucket']) >= 3
        )
        clean_quality = bool(
            quality_base
            and int(row['core_score_min']) >= 2
        )
        clean_eligible = bool(
            clean_quality
            and not row['critical_review_signal']
            and not row.get('repetition_auto_duplicate_signal', False)
            and not row.get('repetition_review_candidate_signal', False)
            and not row.get('greeting_low_review_signal', False)
        )
        row['quality_base_wide'] = quality_base
        row['clean_quality_no_low_core'] = clean_quality
        row['clean_eligible_for_best'] = clean_eligible


def category_tags(row: dict[str, Any]) -> set[str]:
    language = str(row['language_group'])
    tags = {
        f'judge:{row["judge_family"]}',
        f'lang:{language}',
        'lang:en' if language == 'en' else 'lang:non_en',
        f'topic:{row["topic"]}',
        f'turn:{row["turn_bucket"]}',
    }
    return tags


def build_constraints() -> dict[str, dict[str, int | None]]:
    return {
        'judge:qwen': {'min': 9_000, 'max': 11_000},
        'judge:glm': {'min': 9_000, 'max': 11_000},
        'lang:en': {'min': 11_600, 'max': 12_400},
        'lang:non_en': {'min': 7_600, 'max': 8_400},
        'lang:es': {'min': 3_600, 'max': None},
        'lang:pt': {'min': 1_400, 'max': None},
        'lang:fr': {'min': 1_000, 'max': None},
        'lang:zh': {'min': 500, 'max': None},
        'lang:de': {'min': 240, 'max': None},
        'lang:ja': {'min': 160, 'max': None},
        'lang:other': {'min': 40, 'max': None},
        'topic:explicit': {'min': 7_000, 'max': 8_000},
        'topic:romance_suggestive': {'min': 5_000, 'max': 6_400},
        'topic:other_rp': {'min': 5_800, 'max': 7_200},
        'topic:unknown': {'min': 100, 'max': 400},
        'turn:05_08': {'min': 4_000, 'max': 5_000},
        'turn:09_16': {'min': 6_200, 'max': 7_200},
        'turn:17_32': {'min': 6_200, 'max': 7_000},
        'turn:33_64': {'min': 1_600, 'max': 2_200},
        'turn:65_plus': {'min': 200, 'max': 400},
    }


def select_best(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [row for row in rows if row['clean_eligible_for_best']]
    if len(candidates) < BUDGET:
        raise RuntimeError(f'clean eligible candidates below budget: {len(candidates)}')
    ordered = sorted(candidates, key=selection_sort_key)
    by_id = {row['dialogue_id']: row for row in candidates}
    constraints = build_constraints()
    constraint_lists = {
        name: [row for row in ordered if name in category_tags(row)]
        for name in constraints
    }
    clean_neutral = [row for row in candidates if row['source_partition'] == 'neutral_mmr']
    cluster_members: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in clean_neutral:
        if row.get('cluster_uid'):
            cluster_members[str(row['cluster_uid'])].append(row)
    for members in cluster_members.values():
        members.sort(key=selection_sort_key)
    cluster_caps = {
        cluster: min(
            150,
            max(1, math.ceil(1.5 * BUDGET * len(members) / len(clean_neutral))),
        )
        for cluster, members in cluster_members.items()
    }

    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = collections.defaultdict(set)
    selection_order: list[str] = []
    counts: collections.Counter[str] = collections.Counter()
    cluster_counts: collections.Counter[str] = collections.Counter()

    def add(row: dict[str, Any], reason: str) -> None:
        dialogue_id = row['dialogue_id']
        reasons[dialogue_id].add(reason)
        if dialogue_id in selected:
            return
        selected[dialogue_id] = row
        selection_order.append(dialogue_id)
        counts.update(category_tags(row))
        cluster = row.get('cluster_uid')
        if cluster:
            cluster_counts[str(cluster)] += 1

    def can_add(row: dict[str, Any]) -> bool:
        if row['dialogue_id'] in selected:
            return False
        for tag in category_tags(row):
            maximum = constraints.get(tag, {}).get('max')
            if maximum is not None and counts[tag] + 1 > int(maximum):
                return False
        cluster = row.get('cluster_uid')
        if cluster and cluster_counts[str(cluster)] + 1 > cluster_caps[str(cluster)]:
            return False
        return True

    for row in ordered:
        if row['has_any_core_provenance']:
            add(row, 'protected_clean_core')
    for row in ordered:
        if row['has_effective_summary']:
            add(row, 'protected_clean_memory')
    for cluster in sorted(cluster_members):
        representative = cluster_members[cluster][0]
        add(representative, 'required_mmr_cluster_coverage')

    required_count = len(selected)
    required_max_violations = []
    for name, spec in constraints.items():
        maximum = spec['max']
        if maximum is not None and counts[name] > int(maximum):
            required_max_violations.append(
                {'constraint': name, 'actual': counts[name], 'max': maximum}
            )

    scarcity_order = sorted(
        constraints,
        key=lambda name: (
            -(int(constraints[name]['min'] or 0) / max(1, len(constraint_lists[name]))),
            name,
        ),
    )
    unsatisfied_during_fill: list[dict[str, Any]] = []
    for name in scarcity_order:
        minimum = int(constraints[name]['min'] or 0)
        pointer = 0
        pool = constraint_lists[name]
        while counts[name] < minimum and len(selected) < BUDGET:
            chosen = None
            while pointer < len(pool):
                candidate = pool[pointer]
                pointer += 1
                if can_add(candidate):
                    chosen = candidate
                    break
            if chosen is None:
                unsatisfied_during_fill.append(
                    {
                        'constraint': name,
                        'actual': counts[name],
                        'min': minimum,
                        'reason': 'no_addable_clean_candidate',
                    }
                )
                break
            add(chosen, f'minimum_quota:{name}')

    for row in ordered:
        if len(selected) >= BUDGET:
            break
        if can_add(row):
            add(row, 'quality_fill')

    if len(selected) != BUDGET:
        unsatisfied_during_fill.append(
            {
                'constraint': 'best_budget',
                'actual': len(selected),
                'min': BUDGET,
                'max': BUDGET,
                'reason': 'maxima_or_cluster_caps_prevented_fill',
            }
        )

    selected_rows = [selected[dialogue_id] for dialogue_id in selection_order]
    for index, dialogue_id in enumerate(selection_order, 1):
        selected[dialogue_id]['best_selection_order'] = index
        selected[dialogue_id]['best_selection_reasons_json'] = compact(sorted(reasons[dialogue_id]))
    quality_sorted = sorted(selected_rows, key=selection_sort_key)
    for rank, row in enumerate(quality_sorted, 1):
        row['provisional_best_rank'] = rank
    for row in rows:
        row.setdefault('best_selection_order', None)
        row.setdefault('best_selection_reasons_json', '[]')
        row.setdefault('provisional_best_rank', None)
        row['provisional_best20k'] = row['dialogue_id'] in selected

    validation: dict[str, Any] = {}
    for name, spec in constraints.items():
        actual = counts[name]
        minimum = spec['min']
        maximum = spec['max']
        satisfied = (
            (minimum is None or actual >= int(minimum))
            and (maximum is None or actual <= int(maximum))
        )
        validation[name] = {
            'min': minimum,
            'max': maximum,
            'actual': actual,
            'satisfied': satisfied,
        }
    eligible_core = sum(row['has_any_core_provenance'] for row in candidates)
    selected_core = sum(row['has_any_core_provenance'] for row in selected_rows)
    eligible_memory = sum(row['has_effective_summary'] for row in candidates)
    selected_memory = sum(row['has_effective_summary'] for row in selected_rows)
    selected_clusters = {
        str(row['cluster_uid'])
        for row in selected_rows
        if row.get('cluster_uid') and row['source_partition'] == 'neutral_mmr'
    }
    mandatory = {
        'clean_core': {
            'eligible': eligible_core,
            'selected': selected_core,
            'satisfied': eligible_core == selected_core,
        },
        'clean_memory': {
            'eligible': eligible_memory,
            'selected': selected_memory,
            'satisfied': eligible_memory == selected_memory,
        },
        'clean_mmr_clusters': {
            'eligible': len(cluster_members),
            'covered': len(selected_clusters),
            'satisfied': len(cluster_members) == len(selected_clusters),
        },
        'budget': {
            'expected': BUDGET,
            'actual': len(selected_rows),
            'satisfied': len(selected_rows) == BUDGET,
        },
    }
    unsatisfied = [name for name, item in validation.items() if not item['satisfied']]
    unsatisfied += [name for name, item in mandatory.items() if not item['satisfied']]
    meta = {
        'candidate_count': len(candidates),
        'required_initial_count': required_count,
        'constraints': validation,
        'mandatory': mandatory,
        'required_max_violations': required_max_violations,
        'unsatisfied_during_fill': unsatisfied_during_fill,
        'unsatisfied_constraints': sorted(set(unsatisfied)),
        'cluster_cap': {
            'formula': 'min(150,max(1,ceil(1.5*budget*eligible_cluster_size/eligible_neutral_count)))',
            'eligible_neutral': len(clean_neutral),
            'clusters': len(cluster_members),
            'min': min(cluster_caps.values()) if cluster_caps else None,
            'max': max(cluster_caps.values()) if cluster_caps else None,
        },
    }
    return quality_sorted, meta


def assign_tiers(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tiers: dict[str, list[dict[str, Any]]] = {
        'best': [],
        'borderline': [],
        'rejudge': [],
        'hold': [],
        'core_rescue': [],
        'repetition_reject': [],
        'elite_A': [],
        'strong_B': [],
    }
    for row in rows:
        if row['provisional_best20k']:
            tier = 'best'
        elif row['rejudge_required']:
            tier = 'rejudge'
        elif row.get('repetition_auto_duplicate_signal'):
            tier = 'hold'
        elif row['quality_base_wide'] or row.get('repetition_review_candidate_signal'):
            tier = 'borderline'
        else:
            tier = 'hold'
        row['selection_tier'] = tier
        row['core_rescue'] = bool(row['has_any_core_provenance'] and tier != 'best')
        row['repetition_reject'] = bool(row.get('repetition_auto_duplicate_signal'))
        row['elite_A'] = bool(
            tier == 'best'
            and int(row.get('core_score_min') or -1) >= 2
            and int(row.get('core_dimensions_ge_3') or 0) >= 4
            and not row['rejudge_required']
            and not row['critical_review_signal']
            and not row['greeting_low_review_signal']
            and not row.get('repetition_auto_duplicate_signal')
            and not row.get('repetition_review_candidate_signal')
        )
        row['strong_B'] = bool(tier == 'best' and not row['elite_A'])
        row['best_quality_band'] = 'elite_A' if row['elite_A'] else ('strong_B' if row['strong_B'] else '')
        tiers[tier].append(row)
        if row['core_rescue']:
            tiers['core_rescue'].append(row)
        if row['repetition_reject']:
            tiers['repetition_reject'].append(row)
        if row['elite_A']:
            tiers['elite_A'].append(row)
        if row['strong_B']:
            tiers['strong_B'].append(row)
    for name, subset in tiers.items():
        subset.sort(key=selection_sort_key)
        for rank, row in enumerate(subset, 1):
            if name not in {'core_rescue', 'repetition_reject', 'elite_A', 'strong_B'}:
                row['tier_rank'] = rank
            elif name == 'core_rescue':
                row['core_rescue_rank'] = rank
            elif name == 'repetition_reject':
                row['repetition_reject_rank'] = rank
            elif name == 'elite_A':
                row['elite_A_rank'] = rank
            else:
                row['strong_B_rank'] = rank
    for row in rows:
        row.setdefault('tier_rank', None)
        row.setdefault('core_rescue_rank', None)
        row.setdefault('repetition_reject_rank', None)
        row.setdefault('elite_A_rank', None)
        row.setdefault('strong_B_rank', None)
    return tiers


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        path,
        compression='zstd',
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=4096,
    )


def build_report(
    rows: list[dict[str, Any]],
    tiers: dict[str, list[dict[str, Any]]],
    selection_meta: dict[str, Any],
    source_hashes: dict[str, Any],
    repetition_summary: dict[str, Any],
) -> dict[str, Any]:
    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            'count': len(subset),
            'judge': distribution(subset, 'judge_family'),
            'language': distribution(subset, 'language_group'),
            'topic': distribution(subset, 'topic'),
            'replay_mode': distribution(subset, 'replay_mode'),
            'turn_bucket': distribution(subset, 'turn_bucket'),
            'source_partition': distribution(subset, 'source_partition'),
            'core_provenance': sum(row['has_any_core_provenance'] for row in subset),
            'effective_memory': sum(row['has_effective_summary'] for row in subset),
            'greeting_low_review': sum(row['greeting_low_review_signal'] for row in subset),
            'critical_review': sum(row['critical_review_signal'] for row in subset),
            'nsfw_metadata_review': sum(row['nsfw_metadata_review_signal'] for row in subset),
            'low_confidence_review': sum(row['low_confidence_review_signal'] for row in subset),
            'reason_review': sum(row['reason_review_signal'] for row in subset),
            'quality_score': quantiles([
                row['final_quality_score'] for row in subset
                if row.get('final_quality_score') is not None
            ]),
            'reason_reliability': quantiles([
                row['reason_reliability'] for row in subset
                if row.get('reason_reliability') is not None
            ]),
        }

    clean_candidates = [row for row in rows if row['clean_eligible_for_best']]
    naive = sorted(clean_candidates, key=selection_sort_key)[:BUDGET]
    best = tiers['best']
    best_ids = {row['dialogue_id'] for row in best}
    naive_ids = {row['dialogue_id'] for row in naive}
    serious_counts = collections.Counter()
    for row in rows:
        for item in safe_json_list(row.get('screening_anomalies_json')):
            if item in SERIOUS_ANOMALIES:
                serious_counts[str(item)] += 1
    pool_counts: dict[str, dict[str, int]] = {}
    for pool_name in sorted(CORE_POOL_NAMES):
        field = f'has_{pool_name}'
        pool_counts[pool_name] = {
            'all': sum(row[field] for row in rows),
            'clean_eligible': sum(row[field] and row['clean_eligible_for_best'] for row in rows),
            'best': sum(row[field] and row['provisional_best20k'] for row in rows),
            'core_rescue': sum(row[field] and row['core_rescue'] for row in rows),
        }
    reason_trace: dict[str, Any] = {}
    for judge in ['qwen', 'glm']:
        judge_rows = [
            row for row in rows
            if row.get('judge_family') == judge
            and row.get('judge_status') == 'valid'
        ]
        per_dimension: dict[str, Any] = {}
        for dim in ALL_DIMS:
            applicable = [
                row for row in judge_rows
                if row.get(f'audited_{dim}') is not None
            ]
            per_dimension[dim] = {
                'applicable': len(applicable),
                'reason_score_aligned': sum(
                    bool(row.get(f'audit_{dim}_reason_score_aligned'))
                    for row in applicable
                ),
                'model_vs_audited_changed': sum(
                    bool(row.get(f'audit_{dim}_model_vs_audited_changed'))
                    for row in applicable
                ),
                'positive_verified_turn_references': sum(
                    len(row.get(f'audit_{dim}_positive_verified_turns') or [])
                    for row in applicable
                ),
                'negative_verified_turn_references': sum(
                    len(row.get(f'audit_{dim}_negative_verified_turns') or [])
                    for row in applicable
                ),
                'reason_characters': quantiles([
                    float(row.get(f'audit_{dim}_reason_characters'))
                    for row in applicable
                    if row.get(f'audit_{dim}_reason_characters') is not None
                ]),
            }
        reason_trace[judge] = {
            'valid_dialogues': len(judge_rows),
            'all_applicable_reasons_score_aligned': sum(
                bool(row.get('all_applicable_reasons_score_aligned'))
                for row in judge_rows
            ),
            'dimensions': per_dimension,
        }

    report = {
        'schema_version': SCHEMA_VERSION,
        'created_at': CREATED_AT,
        'status': (
            'complete'
            if not selection_meta['unsatisfied_constraints']
            and not selection_meta['unsatisfied_during_fill']
            and not selection_meta['required_max_violations']
            else 'complete_with_explicit_constraint_violations'
        ),
        'source': source_hashes,
        'input_count': len(rows),
        'quality_formula': {
            'core_dimensions': CORE_DIMS,
            'weights': CORE_WEIGHTS,
            'raw': '0.80*(weighted_core/4)+0.12*(quality_floor/4)+0.08*(usable_turn_bucket/4)',
            'cross_judge_calibration': 'average-rank ECDF independently within each judge_family',
            'final_quality_score': '100 * quality_percentile_within_judge',
            'excluded_from_quality_composite': [
                'greeting_style_adherence',
                'reason text/reliability',
                'NSFW metadata',
                'critical flags',
                'confidence',
                'overall diagnosis',
                'teacher/provenance',
                'MDS novelty',
                'deterministic repetition similarity',
            ],
        },
        'reason_fields': {
            'affects_quality_or_tier': False,
            'reliability_formula': '0.30*coverage+0.20*specificity+0.20*template_uniqueness+0.30*evidence_alignment',
            'alignment': 'evidence calibration flags plus separate model-vs-audited score alignment',
            'truncation': 'fraction of applicable dimension reasons with length >=240 characters',
        },
        'reason_audit_trace': reason_trace,
        'gate_counts': {
            'valid': sum(row['judge_status'] == 'valid' for row in rows),
            'quality_fields_complete': sum(row['quality_fields_complete'] for row in rows),
            'quality_base_wide': sum(row['quality_base_wide'] for row in rows),
            'clean_quality_no_low_core': sum(row['clean_quality_no_low_core'] for row in rows),
            'clean_eligible_for_best': sum(row['clean_eligible_for_best'] for row in rows),
            'rejudge_required': sum(row['rejudge_required'] for row in rows),
            'terminal_rejudge': sum(row['terminal_rejudge_signal'] for row in rows),
            'missing_fields_rejudge': sum(row['missing_fields_rejudge_signal'] for row in rows),
            'internal_contradiction_rejudge': sum(row['internal_contradiction_rejudge_signal'] for row in rows),
            'critical_review_only': sum(row['critical_review_signal'] and not row['rejudge_required'] for row in rows),
            'greeting_low_review_only': sum(row['greeting_low_review_signal'] and not row['rejudge_required'] for row in rows),
            'nsfw_metadata_review_only': sum(row['nsfw_metadata_review_signal'] and not row['rejudge_required'] for row in rows),
            'low_confidence_review_only': sum(row['low_confidence_review_signal'] and not row['rejudge_required'] for row in rows),
        },
        'serious_internal_anomalies': dict(sorted(serious_counts.items())),
        'deterministic_repetition': {
            **repetition_summary,
            'policy': 'auto duplicate blocks best and is preserved in repetition_reject; review candidate is borderline/human-review; no history is deleted',
            'independent_of_judge_flags': True,
        },
        'downstream_turn_judge_workload': {
            'best_dialogues': len(best),
            'candidate_target_count_total': sum(int(row.get('candidate_target_count') or 0) for row in best),
            'candidate_target_count': quantiles([float(row.get('candidate_target_count') or 0) for row in best]),
            'turn_manifest_assistant_turn_total': sum(int(row.get('turn_manifest_assistant_turn_count') or 0) for row in best),
            'turn_manifest_assistant_turns_per_dialogue': quantiles([float(row.get('turn_manifest_assistant_turn_count') or 0) for row in best]),
            'turn_bucket_distribution': distribution(best, 'turn_bucket'),
            'turn_count_mismatch_dialogues': sum(not row.get('turn_manifest_count_matches_scoreable_turns', False) for row in best),
        },
        'outstanding_quality_checks': {
            'relationship_transition_quality': 'not yet available as a reliable deterministic dialogue label; evaluate in downstream per-turn Judge/second audit',
            'user_takeover_or_user_action_control': 'not yet available as a reliable deterministic dialogue label; evaluate in downstream per-turn Judge/second audit',
        },
        'selection': selection_meta,
        'tiers': {name: summarize(subset) for name, subset in tiers.items()},
        'core_provenance': pool_counts,
        'quality_cost_vs_unconstrained_top20k': {
            'overlap': len(best_ids & naive_ids),
            'swaps_for_constraints': BUDGET - len(best_ids & naive_ids),
            'naive_quality': quantiles([row['final_quality_score'] for row in naive]),
            'selected_quality': quantiles([row['final_quality_score'] for row in best]),
            'naive_core': sum(row['has_any_core_provenance'] for row in naive),
            'selected_core': sum(row['has_any_core_provenance'] for row in best),
            'naive_memory': sum(row['has_effective_summary'] for row in naive),
            'selected_memory': sum(row['has_effective_summary'] for row in best),
            'naive_cluster_coverage': len({row['cluster_uid'] for row in naive if row.get('cluster_uid')}),
            'selected_cluster_coverage': len({row['cluster_uid'] for row in best if row.get('cluster_uid')}),
        },
    }
    return report


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: build_selection_v1.py STAGING_DIR')
    staging = Path(sys.argv[1]).resolve()
    prepare(staging)
    status_path = staging / 'BUILD_STATUS.json'
    write_json(status_path, {'schema_version': SCHEMA_VERSION, 'stage': 'loading', 'updated_at': CREATED_AT})

    source_paths = [
        FEATURES,
        MATERIAL,
        UNION / '00_manifest/MERGE_CONTRACT.json',
        UNION / '00_manifest/UNION_REPORT.json',
        UNION / '00_manifest/SHA256SUMS',
        TURN_MANIFEST,
        REPETITION_FEATURES,
        REPETITION_AUTO_NATIVE,
        REPETITION_REVIEW_NATIVE,
        *VALID_JSONL_SOURCES,
    ]
    source_hashes = {
        str(path): {'size': path.stat().st_size, 'sha256': sha256_file(path)}
        for path in source_paths
    }
    write_json(staging / '00_manifest/SOURCE_SHA256SUMS.json', source_hashes)

    rows, material = load_rows()
    attach_reason_audit(rows)
    repetition_summary = attach_repetition_features(rows)
    enrich_rows(rows, material)
    write_json(status_path, {'schema_version': SCHEMA_VERSION, 'stage': 'selecting', 'updated_at': datetime.now(timezone.utc).isoformat()})
    best, selection_meta = select_best(rows)
    tiers = assign_tiers(rows)

    report = build_report(rows, tiers, selection_meta, source_hashes, repetition_summary)
    contract = {
        'schema_version': SCHEMA_VERSION,
        'created_at': CREATED_AT,
        'selection_unit': 'replay_dialogue',
        'primary_key': 'dialogue_id',
        'result_trace_key': 'judge_key',
        'budget': BUDGET,
        'seed': SEED,
        'quality': report['quality_formula'],
        'reason_policy': report['reason_fields'],
        'gates': {
            'rejudge': 'terminal nonvalid, missing required quality fields, or serious internal contradiction',
            'clean_eligible_for_best': 'valid, not rejudge, quality_floor>=2, usable_turn_bucket>=3, all six core scores>=2, and no critical flag',
            'critical_flags': 'review-only and block direct best; never silently dropped',
            'greeting_low': 'excluded from cross-judge composite; if applicable and audited score<=1, block direct best and route to borderline/review; N/A is not penalized',
            'nsfw_metadata': 'review/stratification only; never a quality gate',
            'confidence': 'review-only; never a quality gate',
            'overall_diagnosis': 'audit text only; never a quality gate',
            'deterministic_repetition_auto': 'blocks direct best; preserved in repetition_reject overlay; not a Judge flag',
            'deterministic_repetition_review': 'blocks direct best and enters borderline/human review; not a Judge flag',
        },
        'constraints': build_constraints(),
        'mandatory': [
            'all clean eligible core provenance dialogues',
            'all clean eligible effective-memory dialogues',
            'at least one clean eligible dialogue from every clean eligible neutral MMR cluster',
        ],
        'stable_tie_break': 'sha256(seed|dialogue_id), then dialogue_id',
        'source_files': source_hashes,
        'frozen_union_modified': False,
    }
    write_json(staging / '00_manifest/SELECTION_CONTRACT.json', contract)
    write_json(staging / '03_reports/AUDIT_REPORT.json', report)

    master = sorted(
        rows,
        key=lambda row: (
            {'best': 0, 'borderline': 1, 'rejudge': 2, 'hold': 3}[row['selection_tier']],
            *selection_sort_key(row),
        ),
    )
    write_json(status_path, {'schema_version': SCHEMA_VERSION, 'stage': 'writing_parquet', 'updated_at': datetime.now(timezone.utc).isoformat()})
    write_parquet(staging / '01_ranking/dialogue_ranking_master.parquet', master)
    write_parquet(staging / '02_tiers/provisional_best20k.parquet', tiers['best'])
    write_parquet(staging / '02_tiers/borderline.parquet', tiers['borderline'])
    write_parquet(staging / '02_tiers/rejudge.parquet', tiers['rejudge'])
    write_parquet(staging / '02_tiers/hold.parquet', tiers['hold'])
    write_parquet(staging / '02_tiers/core_rescue.parquet', tiers['core_rescue'])
    write_parquet(staging / '02_tiers/repetition_reject.parquet', tiers['repetition_reject'])
    write_parquet(staging / '02_tiers/elite_A.parquet', tiers['elite_A'])
    write_parquet(staging / '02_tiers/strong_B.parquet', tiers['strong_B'])

    tier_counts = {name: len(subset) for name, subset in tiers.items()}
    best_summary = report['tiers']['best']
    readme = f'''# LusyChat Dialogue Selection v1

This is a derived, reproducible selection over the frozen union 50% dialogue Judge features. No frozen union file or raw Judge output was modified.

## Result

- Ranking master: {len(rows):,} replay dialogues.
- Provisional best: {tier_counts['best']:,}.
- Borderline: {tier_counts['borderline']:,}.
- Rejudge: {tier_counts['rejudge']:,}.
- Hold: {tier_counts['hold']:,}.
- Core rescue overlay: {tier_counts['core_rescue']:,}.
- Deterministic repetition reject/review overlay: {tier_counts['repetition_reject']:,}.
- Elite A overlay inside best: {tier_counts['elite_A']:,}.
- Strong B overlay inside best: {tier_counts['strong_B']:,}.

## Quality score

The score uses only six audited core dimensions plus quality_floor and usable_turn_bucket. Greeting, reason text, NSFW metadata, critical flags, confidence, diagnosis, provenance and MDS novelty are excluded from the quality composite. The raw composite is converted to an average-rank percentile independently inside each Judge family.

## Best-20k constraints

All clean eligible core and effective-memory dialogues are mandatory. Every clean eligible neutral MMR cluster must be represented. Judge, language, topic and visible-turn ranges are defined in 00_manifest/SELECTION_CONTRACT.json.

Constraint status: {report['status']}.
Unsatisfied constraints: {compact(selection_meta['unsatisfied_constraints'])}.

Best Judge distribution: {compact(best_summary['judge'])}.
Best language distribution: {compact(best_summary['language'])}.
Best topic distribution: {compact(best_summary['topic'])}.
Best turn distribution: {compact(best_summary['turn_bucket'])}.

All free-text diagnoses and reasons remain audit evidence only and do not change dialogue quality.
'''
    (staging / 'README.md').write_text(readme, encoding='utf-8')

    write_json(
        status_path,
        {
            'schema_version': SCHEMA_VERSION,
            'stage': 'complete',
            'status': report['status'],
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'rows': len(rows),
            'tier_counts': tier_counts,
            'unsatisfied_constraints': selection_meta['unsatisfied_constraints'],
        },
    )

    checksum_files = [
        staging / '00_manifest/SELECTION_CONTRACT.json',
        staging / '00_manifest/SOURCE_SHA256SUMS.json',
        staging / '01_ranking/dialogue_ranking_master.parquet',
        staging / '02_tiers/provisional_best20k.parquet',
        staging / '02_tiers/borderline.parquet',
        staging / '02_tiers/rejudge.parquet',
        staging / '02_tiers/hold.parquet',
        staging / '02_tiers/core_rescue.parquet',
        staging / '02_tiers/repetition_reject.parquet',
        staging / '02_tiers/elite_A.parquet',
        staging / '02_tiers/strong_B.parquet',
        staging / '03_reports/AUDIT_REPORT.json',
        staging / 'README.md',
        staging / 'scripts/build_selection_v1.py',
        status_path,
    ]
    checksum_lines = [
        f'{sha256_file(path)}  {path.relative_to(staging).as_posix()}'
        for path in checksum_files
    ]
    (staging / '00_manifest/SHA256SUMS').write_text(
        '\n'.join(checksum_lines) + '\n', encoding='utf-8'
    )

    if selection_meta['unsatisfied_constraints'] or selection_meta['required_max_violations']:
        # Publish with explicit violations in report; never hide them.
        pass
    os.replace(staging, FINAL)
    print(compact({
        'status': report['status'],
        'output': str(FINAL),
        'tier_counts': tier_counts,
        'selection': selection_meta,
    }))


if __name__ == '__main__':
    main()
