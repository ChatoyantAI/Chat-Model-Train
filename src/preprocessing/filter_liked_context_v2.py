#!/usr/bin/env python3
"""Build context-independent, technically eligible liked-model targets.

Memory/context eligibility is intentionally deferred to pre-mds-context-v2.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCRIPT_ROOT = Path(__file__).resolve().parent
PIPELINE_SRC = Path(os.environ.get("LUSY_PIPELINE_SRC", str(SCRIPT_ROOT)))
sys.path.insert(0, str(PIPELINE_SRC))
from lusy_extract.screening_text import screen_turn  # noqa: E402
from format_repair import repair_balanced_nesting  # noqa: E402


INVENTORY = Path(
    "/home/gyr/lusychat-training/liked-augmentation-v1/01_inventory/liked_assistant_inventory.parquet"
)
BUNDLE_ROOT = Path("/home/gyr/lusychat-training/lusy-chat-4m-v1/bundles")
DEFAULT_OUTPUT = Path(
    "/home/gyr/lusychat-training/pre-mds-context-v2/"
    "01_target_index/liked_technical_shards"
)

BUNDLE_COLUMNS = [
    "conversation_id",
    "turns",
    "logical_turn_count",
    "trainable_turn_count",
    "character_id",
    "user_id",
    "conversation_status",
    "conversation_create_time",
    "conversation_update_time",
    "persona_id",
    "conversation_station",
    "is_root_conversation",
    "conversation_bucket",
    "initial_greeting",
    "initial_greeting_language",
    "character_name",
    "character_visible",
    "character_status",
    "character_station",
    "character_intro",
    "character_description",
    "character_gender",
    "character_age",
    "character_tags",
    "character_greeting",
    "character_nsfw",
    "character_scenario",
    "character_personality",
    "character_qa_examples",
    "character_prompt",
    "character_style",
    "localized_name",
    "localized_headline",
    "localized_intro",
    "localized_description",
    "localized_gender",
    "localized_greeting",
    "localized_greetings",
    "localized_scenario",
    "localized_personality",
    "localized_qa_examples",
    "localized_prompt",
    "localized_example_reply",
    "has_exact_localized_character_description",
    "source_user_language",
    "source_user_country_code",
    "source_user_register_source",
    "source_user_station",
    "source_user_is_deleted_user",
    "persona_record_id",
    "persona_owner_user_id",
    "persona_is_deleted",
    "persona_has_nickname",
    "persona_has_age",
    "persona_has_personality",
    "persona_has_pinned_memories",
    "memory_summaries",
    "memory_summary_count",
    "requires_privacy_and_minor_scrub",
    "requires_content_safety_and_pii_scrub",
    "minor_nsfw_hard_exclusion",
    "is_fork_conversation",
    "has_assistant_version_anomaly",
    "has_user_version_anomaly",
]

OPEN_TO_CLOSE = {
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "«": "»",
    "‹": "›",
    "„": "“",
    "‚": "‘",
    "»": "«",
}
CLOSERS = set(OPEN_TO_CLOSE.values())
WS_RE = re.compile(r"\s+", re.UNICODE)
THINK_HEAD_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
PLATFORM_META_RE = re.compile(
    r"(language model|training data|system prompt|developer message|platform policy|provider policy|"
    r"openai policy|anthropic policy|语言模型|训练数据|系统提示|平台政策|開発者メッセージ|"
    r"언어 모델|훈련 데이터|моделью языка|данн(?:ые|ых) обучения)",
    re.IGNORECASE,
)
AI_SELF_RE = re.compile(
    r"(^|[.!?。！？]\s*)(as an ai|i(?:'m| am) (?:just )?(?:an? )?(?:ai|chatbot)|"
    r"作为(?:一个)?(?:ai|人工智能|语言模型)|我是(?:一个)?(?:ai|人工智能|语言模型|聊天机器人)|"
    r"私は(?:ai|人工知能|言語モデル)|저는 (?:ai|인공지능|언어 모델)|"
    r"como (?:una? )?(?:ia|inteligencia artificial)|sou (?:uma? )?(?:ia|inteligência artificial)|"
    r"je suis (?:une? )?(?:ia|intelligence artificielle)|ich bin (?:eine? )?(?:ki|künstliche intelligenz)|"
    r"я (?:являюсь )?(?:ии|искусственн(?:ый|ым) интеллектом))",
    re.IGNORECASE,
)
CHARACTER_AI_RE = re.compile(
    r"\b(ai|artificial intelligence|robot|android|chatbot|virtual assistant|cyborg)\b|"
    r"人工智能|机器人|機器人|アンドロイド|ロボット|인공지능|로봇|андроид|робот",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", required=True, help="inclusive range, e.g. 0-15")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shard-name", required=True)
    return parser.parse_args()


def parse_range(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(spec)]


def strip_cot(text: str | None) -> tuple[str, bool]:
    value = text or ""
    stripped, count = THINK_HEAD_RE.subn("", value, count=1)
    return stripped.strip(), bool(count)


def validate_format(text: str) -> dict[str, Any]:
    quote_stack: list[str] = []
    quote_style: str | None = None
    asterisk_open = False
    errors: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "*":
            end = index + 1
            while end < len(text) and text[end] == "*":
                end += 1
            run = end - index
            if run >= 2:
                index = end
                continue
            if quote_stack:
                errors.append({"offset": index, "token": "*", "error": "asterisk_inside_quote"})
            asterisk_open = not asterisk_open
            index += 1
            continue
        if char == '"':
            if asterisk_open:
                errors.append({"offset": index, "token": char, "error": "quote_inside_asterisk"})
            if quote_stack and quote_stack[-1] == '"':
                quote_stack.pop()
            else:
                quote_stack.append('"')
                quote_style = quote_style or "ascii_double"
            index += 1
            continue
        if char in CLOSERS and quote_stack and quote_stack[-1] == char:
            quote_stack.pop()
            index += 1
            continue
        if char in OPEN_TO_CLOSE:
            if asterisk_open:
                errors.append({"offset": index, "token": char, "error": "quote_inside_asterisk"})
            quote_stack.append(OPEN_TO_CLOSE[char])
            quote_style = quote_style or f"{char}{OPEN_TO_CLOSE[char]}"
            index += 1
            continue
        # Curly apostrophe is common in English contractions (I’m, don’t).
        # Treat it as a closer only when a matching curly-single quote is open.
        if char in CLOSERS and char != "’":
            errors.append({"offset": index, "token": char, "error": "unexpected_quote_closer"})
        index += 1
    if asterisk_open:
        errors.append({"offset": len(text), "token": "*", "error": "unclosed_asterisk"})
    if quote_stack:
        errors.append(
            {"offset": len(text), "token": quote_stack[-1], "error": "unclosed_quote"}
        )
    error_types = sorted({item["error"] for item in errors})
    return {
        "format_valid": not errors,
        "quote_style": quote_style,
        "quote_balanced": not any("quote" in value for value in error_types),
        "asterisk_balanced": "unclosed_asterisk" not in error_types,
        "cross_nesting_valid": not any(
            value in {"asterisk_inside_quote", "quote_inside_asterisk"} for value in error_types
        ),
        "format_error_type": error_types,
        "format_error_offsets": errors[:50],
    }


def character_is_ai(row: dict[str, Any]) -> bool:
    fields = [
        "character_name",
        "character_intro",
        "character_description",
        "character_scenario",
        "character_personality",
        "character_prompt",
        "localized_name",
        "localized_intro",
        "localized_description",
        "localized_scenario",
        "localized_personality",
        "localized_prompt",
    ]
    return bool(CHARACTER_AI_RE.search("\n".join(str(row.get(key) or "") for key in fields)))


def identity_break(text: str, is_ai_character: bool) -> tuple[bool, float, str | None]:
    platform_match = PLATFORM_META_RE.search(text)
    if platform_match:
        return True, 0.98, platform_match.group(0)
    self_match = AI_SELF_RE.search(text)
    if self_match and not is_ai_character:
        return True, 0.96, self_match.group(0)
    return False, 0.05, None


def normalized_language(value: str | None) -> str:
    language = (value or "unknown").strip()
    aliases = {
        "zh": "zh-TW",
        "zh-cn": "zh-TW",
        "zh-tw": "zh-TW",
        "cn": "zh-TW",
    }
    return aliases.get(language.casefold(), language)


def languages_compatible(user_language: str | None, assistant_language: str | None) -> bool:
    user = normalized_language(user_language)
    assistant = normalized_language(assistant_language)
    if "unknown" in {user, assistant} or not user or not assistant:
        return True
    return user == assistant


def lexical_units(text: str) -> int:
    # Latin/Cyrillic words and each CJK/Hangul character count as one unit.
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žА-Яа-яЁё0-9_]+", text)
    cjk = re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
    return len(words) + len(cjk)


def guideline_compatibility(text: str) -> tuple[str | None, int]:
    count = lexical_units(text)
    has_narration = "*" in text
    if 40 <= count <= 50 and not has_narration:
        return "chat", count
    if 200 <= count <= 250:
        return "scene", count
    if 520 <= count <= 650:
        return "longText", count
    return None, count


def topic_for(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in ["character_tags", "character_name", "character_intro", "character_scenario"]
    ).casefold()
    explicit_words = ["nsfw", "smut", "explicit", "sex", "hentai", "18+"]
    romance_words = ["romance", "romantic", "boyfriend", "girlfriend", "husband", "wife", "lover"]
    if row.get("character_nsfw") or any(word in text for word in explicit_words):
        return "explicit"
    if any(word in text for word in romance_words):
        return "romance_suggestive"
    return "other_rp" if text.strip() else "unknown"


def choose_memory(row: dict[str, Any], target: dict[str, Any]) -> dict[str, Any] | None:
    target_inc = int(target["assistant_increase_id"])
    target_time = target.get("assistant_create_time")
    eligible: list[dict[str, Any]] = []
    for memory in row.get("memory_summaries") or []:
        boundary = memory.get("compressed_until_increase_id")
        created = memory.get("created_at")
        if boundary is None or int(boundary) >= target_inc:
            continue
        if target_time is not None and created is not None and created >= target_time:
            continue
        eligible.append(memory)
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("id") or ""),
        )
    )
    return eligible[-1]


def character_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "character_name",
        "character_intro",
        "character_description",
        "character_gender",
        "character_age",
        "character_tags",
        "character_greeting",
        "character_nsfw",
        "character_scenario",
        "character_personality",
        "character_qa_examples",
        "character_prompt",
        "character_style",
        "localized_name",
        "localized_headline",
        "localized_intro",
        "localized_description",
        "localized_gender",
        "localized_greeting",
        "localized_greetings",
        "localized_scenario",
        "localized_personality",
        "localized_qa_examples",
        "localized_prompt",
        "localized_example_reply",
        "has_exact_localized_character_description",
    ]
    return {key: row.get(key) for key in keys}


def inspect_conversation(
    row: dict[str, Any], candidates: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    scope_reason: str | None = None
    if (
        int(row.get("conversation_station") or 0) != 100
        or int(row.get("character_station") or 0) != 100
        or int(row.get("source_user_station") or 0) != 100
    ):
        scope_reason = "station_not_100"
    elif str(row.get("character_visible") or "").casefold() != "public":
        scope_reason = "character_not_public"
    elif int(row.get("conversation_status") or 0) != 1:
        scope_reason = "conversation_inactive"
    elif int(row.get("character_status") or 0) != 1:
        scope_reason = "character_inactive"
    elif row.get("source_user_is_deleted_user"):
        scope_reason = "source_user_deleted"
    elif row.get("is_fork_conversation") or not row.get("is_root_conversation"):
        scope_reason = "fork_conversation"
    elif row.get("minor_nsfw_hard_exclusion"):
        scope_reason = "minor_nsfw_hard_exclusion"
    ai_character = character_is_ai(row)
    complete_turns: list[dict[str, Any]] = []
    first_bad_turn_index: int | None = None
    first_bad_reason: str | None = None
    seen_pairs: set[tuple[str, str]] = set()
    previous_assistant_hash: str | None = None
    raw_turns = row.get("turns") or []
    for logical_position, raw_turn in enumerate(raw_turns, 1):
        user_content = (raw_turn.get("user_content") or "").strip()
        assistant_content_original, cot_removed = strip_cot(raw_turn.get("assistant_content"))
        if not user_content or not assistant_content_original:
            if first_bad_turn_index is None:
                first_bad_turn_index = logical_position
                first_bad_reason = "empty_turn"
            continue
        initial_format = validate_format(assistant_content_original)
        repair = repair_balanced_nesting(
            assistant_content_original, initial_format, validate_format
        )
        assistant_content = repair.content if repair.repaired else assistant_content_original
        turn = dict(raw_turn)
        turn["assistant_content_original"] = assistant_content_original
        turn["assistant_content_clean"] = assistant_content
        turn["cot_removed"] = cot_removed
        turn["format_repaired"] = repair.repaired
        turn["format_repairable"] = repair.repairable
        turn["format_repair_reason"] = repair.reason
        turn["format_repair_operations"] = list(repair.operations)
        turn["format_original_sha256"] = hashlib.sha256(
            assistant_content_original.encode("utf-8")
        ).hexdigest()
        turn["format_repaired_sha256"] = hashlib.sha256(
            assistant_content.encode("utf-8")
        ).hexdigest()
        turn["turn_index"] = logical_position
        hard_screen = screen_turn(user_content, assistant_content)
        pair_hash = (hard_screen.user_hash, hard_screen.assistant_hash)
        hard_reason = hard_screen.reasons[0] if hard_screen.reasons else None
        if hard_reason is None and pair_hash in seen_pairs:
            hard_reason = "duplicate_pair_in_conversation"
        if hard_reason is None and previous_assistant_hash == hard_screen.assistant_hash:
            hard_reason = "consecutive_assistant_loop"
        turn["hard_screen_flags"] = list(hard_screen.flags)
        format_result = validate_format(assistant_content)
        turn.update(format_result)
        broke_identity, identity_confidence, identity_evidence = identity_break(
            assistant_content, ai_character
        )
        turn["identity_break"] = broke_identity
        turn["identity_break_confidence"] = identity_confidence
        turn["identity_break_evidence"] = identity_evidence
        turn_bad_reason: str | None = None
        if not raw_turn.get("turn_is_trainable"):
            turn_bad_reason = "turn_not_trainable"
        elif int(raw_turn.get("current_user_count") or 0) != 1 or int(
            raw_turn.get("current_assistant_count") or 0
        ) != 1:
            turn_bad_reason = "non_unique_current_version"
        elif int(raw_turn.get("user_version_count") or 0) != 1 or int(
            raw_turn.get("assistant_candidate_count") or 0
        ) != 1:
            turn_bad_reason = "mutation_or_regenerate"
        elif raw_turn.get("mutation_events"):
            turn_bad_reason = "mutation_or_regenerate"
        # Production encoding: 1=liked, 2=disliked, 0/NULL=neutral.
        elif int(raw_turn.get("assistant_evaluation") or 0) == 2:
            turn_bad_reason = "disliked"
        elif hard_reason:
            turn_bad_reason = "hard_screen:" + hard_reason
        elif not format_result["format_valid"]:
            turn_bad_reason = "format_invalid"
        elif broke_identity:
            turn_bad_reason = "identity_break"
        turn["technical_bad_reason"] = turn_bad_reason
        complete_turns.append(turn)
        seen_pairs.add(pair_hash)
        previous_assistant_hash = hard_screen.assistant_hash
        if first_bad_turn_index is None and turn_bad_reason:
            first_bad_turn_index = turn["turn_index"]
            first_bad_reason = turn_bad_reason

    turn_by_assistant = {
        int(turn["assistant_increase_id"]): turn
        for turn in complete_turns
        if turn.get("assistant_increase_id") is not None
    }
    snapshot = character_snapshot(row)
    for increase_id, inventory in candidates.items():
        base = {
            "target_id": str(increase_id),
            "target_increase_id": increase_id,
            "target_message_id": inventory.get("target_message_id"),
            "conversation_id": row.get("conversation_id"),
            "conversation_bucket": int(row.get("conversation_bucket") or 0),
            "user_id": row.get("user_id"),
            "character_id": row.get("character_id"),
            "model": inventory.get("model"),
            "model_id": inventory.get("model_id"),
            "teacher_family": inventory.get("teacher_family"),
            "model_tier": inventory.get("model_tier"),
            "evaluation": inventory.get("evaluation"),
        }
        reasons: list[str] = []
        turn = turn_by_assistant.get(increase_id)
        if scope_reason:
            reasons.append(scope_reason)
        if turn is None:
            reasons.append("target_turn_not_found")
        else:
            position = int(turn["turn_index"])
            if position < 5:
                reasons.append("fewer_than_5_complete_turns")
            if first_bad_turn_index is not None and position >= first_bad_turn_index:
                reasons.append("after_clean_prefix_break:" + str(first_bad_reason))
            if int(inventory.get("regenerated_times") or 0) > 0:
                reasons.append("target_regenerated")
            if not languages_compatible(turn.get("user_language"), turn.get("assistant_language")):
                reasons.append("raw_language_mismatch")
            if not reasons:
                guideline_mode, unit_count = guideline_compatibility(
                    turn["assistant_content_clean"]
                )
                base.update(
                    {
                        "target_position": position,
                        "conversation_length": len(raw_turns),
                        "target_time": turn.get("assistant_create_time"),
                        "user_increase_id": turn.get("user_increase_id"),
                        "user_content": turn.get("user_content"),
                        "target_content": turn["assistant_content_clean"],
                        "target_content_original": turn["assistant_content_original"]
                        if turn["format_repaired"]
                        else None,
                        "user_language_existing": normalized_language(turn.get("user_language")),
                        "assistant_language_existing": normalized_language(
                            turn.get("assistant_language")
                        ),
                        "continued": position < len(raw_turns),
                        "context_mode": None,
                        "memory_id": None,
                        "memory_boundary_increase_id": None,
                        "memory_summary_text": None,
                        "memory_state_json": None,
                        "format_valid": turn["format_valid"],
                        "quote_style": turn["quote_style"],
                        "quote_balanced": turn["quote_balanced"],
                        "asterisk_balanced": turn["asterisk_balanced"],
                        "cross_nesting_valid": turn["cross_nesting_valid"],
                        "format_error_type": json.dumps(
                            turn["format_error_type"], ensure_ascii=False
                        ),
                        "format_error_offsets": json.dumps(
                            turn["format_error_offsets"], ensure_ascii=False
                        ),
                        "cot_removed": turn["cot_removed"],
                        "format_repaired": turn["format_repaired"],
                        "format_repair_reason": turn["format_repair_reason"],
                        "format_repair_operations": json.dumps(
                            turn["format_repair_operations"], ensure_ascii=False
                        ),
                        "format_original_sha256": turn["format_original_sha256"],
                        "format_repaired_sha256": turn["format_repaired_sha256"],
                        "history_format_repair_count": sum(
                            bool(item.get("format_repaired"))
                            for item in complete_turns
                            if int(item["turn_index"]) <= position
                        ),
                        "history_format_repair_ids": json.dumps(
                            [
                                int(item["assistant_increase_id"])
                                for item in complete_turns
                                if int(item["turn_index"]) <= position
                                and item.get("format_repaired")
                            ]
                        ),
                        "history_format_repairs_json": json.dumps(
                            [
                                {
                                    "turn_index": int(item["turn_index"]),
                                    "assistant_increase_id": int(
                                        item["assistant_increase_id"]
                                    ),
                                    "assistant_message_id": item.get(
                                        "assistant_message_id"
                                    ),
                                    "original_content": item[
                                        "assistant_content_original"
                                    ],
                                    "repaired_content": item[
                                        "assistant_content_clean"
                                    ],
                                    "operations": item[
                                        "format_repair_operations"
                                    ],
                                    "original_sha256": item[
                                        "format_original_sha256"
                                    ],
                                    "repaired_sha256": item[
                                        "format_repaired_sha256"
                                    ],
                                }
                                for item in complete_turns
                                if int(item["turn_index"]) <= position
                                and item.get("format_repaired")
                            ],
                            ensure_ascii=False,
                        ),
                        "hard_screen_flags": json.dumps(
                            turn["hard_screen_flags"], ensure_ascii=False
                        ),
                        "identity_break_confidence": turn[
                            "identity_break_confidence"
                        ],
                        "character_is_ai": ai_character,
                        "guideline_mode": guideline_mode,
                        "guideline_attached": guideline_mode is not None,
                        "lexical_units": unit_count,
                        "topic": topic_for(row),
                        "character_snapshot_json": json.dumps(
                            snapshot, ensure_ascii=False, default=str
                        ),
                        "initial_greeting": row.get("initial_greeting"),
                        "initial_greeting_language": row.get(
                            "initial_greeting_language"
                        ),
                        "persona_id": row.get("persona_id"),
                        "persona_record_id": row.get("persona_record_id"),
                        "persona_is_deleted": row.get("persona_is_deleted"),
                        "persona_has_nickname": row.get("persona_has_nickname"),
                        "persona_has_age": row.get("persona_has_age"),
                        "persona_has_personality": row.get(
                            "persona_has_personality"
                        ),
                        "persona_has_pinned_memories": row.get(
                            "persona_has_pinned_memories"
                        ),
                        "source_user_language": row.get("source_user_language"),
                        "source_user_country_code": row.get(
                            "source_user_country_code"
                        ),
                    }
                )
                selected.append(base)
        if reasons:
            base["rejection_reasons"] = reasons
            base["target_position"] = turn.get("turn_index") if turn else None
            base["first_bad_turn_index"] = first_bad_turn_index
            base["first_bad_reason"] = first_bad_reason
            excluded.append(base)
    return selected, excluded


def main() -> int:
    args = parse_args()
    started = time.time()
    buckets = parse_range(args.buckets)
    output_root = Path(args.output_root) / args.shard_name
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = pq.ParquetFile(INVENTORY).read()
    mask = pc.and_(
        pc.and_(
            pc.and_(inventory["base_valid"], pc.invert(inventory["already_in_current_sft"])),
            pc.equal(inventory["evaluation"], 1),
        ),
        pc.is_in(
            inventory["model_tier"],
            value_set=pa.array(["strong_known", "rp_specialized", "experimental"]),
        ),
    )
    inventory = inventory.filter(mask)
    bucket_values = pa.array(buckets, type=inventory["conversation_bucket"].type)
    inventory = inventory.filter(
        pc.is_in(inventory["conversation_bucket"], value_set=bucket_values)
    )
    by_conversation: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
    for row in inventory.to_pylist():
        by_conversation[row["conversation_id"]][int(row["increase_id"])] = row

    selected_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for bucket in buckets:
        path = BUNDLE_ROOT / f"conversation_bucket={bucket:03d}" / "part-000000.parquet"
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=128, columns=BUNDLE_COLUMNS):
            for row in batch.to_pylist():
                candidates = by_conversation.get(row["conversation_id"])
                if not candidates:
                    continue
                selected, excluded = inspect_conversation(row, candidates)
                selected_rows.extend(selected)
                excluded_rows.extend(excluded)

    selected_path = output_root / "eligible.parquet"
    excluded_path = output_root / "excluded.parquet"
    if selected_rows:
        pq.write_table(
            pa.Table.from_pylist(selected_rows),
            selected_path,
            compression="zstd",
            compression_level=6,
        )
    if excluded_rows:
        pq.write_table(
            pa.Table.from_pylist(excluded_rows),
            excluded_path,
            compression="zstd",
            compression_level=6,
        )
    reason_counts: collections.Counter[str] = collections.Counter()
    for row in excluded_rows:
        reason_counts.update(row["rejection_reasons"])
    report = {
        "shard": args.shard_name,
        "buckets": args.buckets,
        "input_candidates": inventory.num_rows,
        "eligible": len(selected_rows),
        "excluded": len(excluded_rows),
        "eligible_tiers": dict(
            collections.Counter(row["model_tier"] for row in selected_rows)
        ),
        "eligible_models": collections.Counter(
            row["model"] for row in selected_rows
        ).most_common(),
        "rejection_reasons": reason_counts.most_common(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(selected_rows) + len(excluded_rows) == inventory.num_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
