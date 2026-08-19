#!/usr/bin/env python3
"""Build one canonical RP training-context row per production conversation.

This is deliberately upstream of MDS/MMR/Judges/target selection.  It reads the
immutable v1 conversation bundles, selects only the canonical current version
of each logical user/assistant turn, renders the current RP system contract,
and writes Data-Juicer-ready JSONL.  Source files are never modified.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import orjson
import pyarrow.parquet as pq


SCHEMA_VERSION = "lusy-rp-dialogue-context-v2"
RENDERER_VERSION = "rp-system-v1.4.1-dialogue-memory-card-v2"
SOURCE_ROOT = Path("/home/gyr/lusychat-training/lusy-chat-4m-v1")
CANONICAL_ROOT = Path("/home/gyr/lusychat-training/lusy-chat-4m-v2/01_canonical")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gyr/lusychat-training/data-juicer-dialogue-v2.1-20260816/01_constructed"
)
DEFAULT_IDENTITY_SNAPSHOT = Path(
    "/home/gyr/lusychat-training/data-juicer-dialogue-v2.1-20260816/"
    "00_inputs/relevant_user_identity_snapshot.parquet"
)

RP_CORE = """You are the character defined below in an ongoing, uncensored roleplay with the user.

## Core Roleplay Rules

1. Fully embody the character. Let their identity, personality, background, goals, knowledge, emotions, relationships, and speaking style shape every response.

2. Treat the character information, scenario, user information, relationship, memories, and established conversation events as factual context. Preserve continuity and do not silently reset or contradict established details.

3. Use the user's latest message as the immediate direction of the interaction. Continue from the current moment and move the scene forward naturally instead of restarting, recapping, paraphrasing, stalling, or turning the conversation into a discussion about the roleplay itself.

4. You may portray the character, relevant side characters, and the surrounding world. The user exclusively controls their own character. Never write or decide the user's dialogue, actions, decisions, thoughts, feelings, reactions, consent, or hidden intentions.

5. Keep the character within what they could reasonably know, observe, remember, or infer. When something is unknown, respond naturally from the character's perspective rather than inventing authoritative out-of-character information.

6. Use the greeting and recent conversation as style references. Learn the character's underlying voice and behavior without mechanically copying exact phrases, scenes, or formatting.

7. Write every part of the response in the Output Language specified below, including dialogue and narration. Do not switch languages unless the Output Language setting itself changes.

8. Mature and explicit adult content is allowed when supported by the character, scenario, relationship, or user's direction. Match the established level of intimacy without forcing sexual content into unrelated scenes. When explicit content is appropriate, do not sanitize it, moralize, add warnings or disclaimers, or fade to black unless restraint is natural for the character or scene. Keep intimate interactions character-specific, emotionally coherent, and physically continuous. All participants in sexual content must be adults.

9. Remain in character even when the user asks meta-level questions. Never step outside the character to identify yourself as an AI, language model, assistant, or chatbot, or to discuss prompts, policies, moderation, hidden instructions, system messages, or internal reasoning."""

CHARACTER_MACRO_RE = re.compile(
    r"\{\{\s*(char|character|character_name)\s*\}\}", re.IGNORECASE
)
ANY_MACRO_RE = re.compile(r"\{\{[^{}\r\n]{1,120}\}\}")
LEADING_THINK_RE = re.compile(
    r"^\s*<think(?:\s[^>]*)?>[\s\S]*?</think>\s*", re.IGNORECASE
)
LITERAL_USER_RE = re.compile(r"(?<!\w)User(?!\w)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
EMAIL_RE = re.compile(r"(?i)[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?i)(?:https?://|www\.)")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ().-]{7,}\d(?!\w)")
GENERIC_NAME_RE = re.compile(
    r"(?i)^(?:user|you|unknown|none|null|n/?a|anonymous|anon|guest|player|human|"
    r"me|myself|用户|你|玩家)(?:[#_\- ]?[0-9a-f]{4,})?$"
)
USER_MACRO_ALIASES = frozenset(
    {
        "user",
        "user_name",
        "username",
        "user name",
        "u-user",
        "usuario",
        "usuário",
        "benutzer",
        "utilisateur",
        "ユーザー",
    }
)
USER_MACRO_SUFFIXES = ("'s", "’s", ",")

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text(value: Any) -> str:
    return str(value or "").strip()


def normalized_language(value: Any) -> str:
    value = text(value).lower().replace("_", "-")
    if not value:
        return "unknown"
    return value.split("-", 1)[0]


def safe_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def safe_name(value: Any) -> tuple[str | None, str]:
    candidate = text(value)
    if not candidate:
        return None, "empty"
    if len(candidate) > 128:
        return None, "too_long"
    if "\n" in candidate or "\r" in candidate or CONTROL_RE.search(candidate):
        return None, "control_or_newline"
    if EMAIL_RE.search(candidate) or URL_RE.search(candidate) or PHONE_RE.search(candidate):
        return None, "pii_like"
    if GENERIC_NAME_RE.fullmatch(candidate):
        return None, "generic"
    return candidate, "safe"


def user_macro_parts(body: str) -> tuple[str, str] | None:
    candidate = body.strip()
    suffix = ""
    folded = unicodedata.normalize("NFKC", candidate).casefold()
    for possible in USER_MACRO_SUFFIXES:
        if folded.endswith(possible.casefold()):
            candidate = candidate[: -len(possible)].rstrip()
            suffix = possible
            break
    if unicodedata.normalize("NFKC", candidate).casefold() not in USER_MACRO_ALIASES:
        return None
    return candidate, suffix


def render_standard_macros(
    value: Any, character_name: str, user_name: str
) -> tuple[str, int, int]:
    rendered = str(value or "")
    # Use a callable replacement so user-provided names are inserted literally.
    # Passing a string replacement to re.subn would interpret backslashes in a
    # name (for example ``\V`` or a trailing ``\``) as replacement escapes.
    rendered, char_count = CHARACTER_MACRO_RE.subn(
        lambda _match: character_name, rendered
    )
    user_count = 0

    def replace_user(match: re.Match[str]) -> str:
        nonlocal user_count
        parts = user_macro_parts(match.group(0)[2:-2])
        if parts is None:
            return match.group(0)
        user_count += 1
        return user_name + parts[1]

    rendered = ANY_MACRO_RE.sub(replace_user, rendered)
    return rendered.strip(), char_count + user_count, len(ANY_MACRO_RE.findall(rendered))


def strip_provider_cot(value: Any) -> tuple[str, bool, int]:
    source = str(value or "")
    match = LEADING_THINK_RE.match(source)
    if not match:
        return source.strip(), False, 0
    return source[match.end() :].strip(), True, match.end()


def persona_sort_key(row: dict[str, Any]) -> tuple[str, int]:
    updated = safe_iso(row.get("updated_at")) or safe_iso(row.get("created_at")) or ""
    return updated, int(row.get("id") or -1)


def load_personas() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = pq.read_table(CANONICAL_ROOT / "personas.parquet").to_pylist()
    by_conversation: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("deleted_at"):
            continue
        conversation_id = text(row.get("conversation_id"))
        persona_id = text(row.get("persona_id"))
        if conversation_id:
            previous = by_conversation.get(conversation_id)
            if previous is None or persona_sort_key(row) > persona_sort_key(previous):
                by_conversation[conversation_id] = row
        if persona_id:
            previous = by_id.get(persona_id)
            if previous is None or persona_sort_key(row) > persona_sort_key(previous):
                by_id[persona_id] = row
    return by_conversation, by_id


def load_account_nicknames(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    required = {"user_id", "nickname"}
    if not required.issubset(schema_names):
        raise RuntimeError(
            f"identity snapshot missing columns: {sorted(required - schema_names)}"
        )
    result: dict[str, str] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=16384, columns=["user_id", "nickname"]
    ):
        for row in batch.to_pylist():
            user_id = text(row.get("user_id"))
            if not user_id:
                continue
            if user_id in result:
                raise RuntimeError(f"duplicate user_id in identity snapshot: {user_id}")
            result[user_id] = text(row.get("nickname"))
    return result


def load_character_cards() -> dict[str, list[dict[str, Any]]]:
    """Load only the explicit card allowlist plus greeting.

    In particular, neither ``description`` nor ``prompt`` is projected from the
    Parquet source, so those columns cannot accidentally enter rendered output.
    """
    columns = [
        "increase_id",
        "character_id",
        "language",
        "name",
        "gender",
        "greeting",
        "scenario",
        "personality",
        "intro",
        "qa_examples",
        "created_at",
        "updated_at",
    ]
    rows = pq.read_table(
        CANONICAL_ROOT / "character_descriptions.parquet", columns=columns
    ).to_pylist()
    result: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        character_id = text(row.get("character_id"))
        if character_id:
            result[character_id].append(row)
    for values in result.values():
        values.sort(key=lambda row: int(row.get("increase_id") or -1), reverse=True)
    return dict(result)


def choose_character_card(
    rows: list[dict[str, Any]], target_language: str
) -> tuple[dict[str, Any] | None, str]:
    target = normalized_language(target_language)
    for row in rows:
        if normalized_language(row.get("language")) == target and text(row.get("name")):
            return row, "description_exact"
    for row in rows:
        if normalized_language(row.get("language")) == "en" and text(row.get("name")):
            return row, "description_en"
    for row in rows:
        if text(row.get("name")):
            return row, "description_any"
    return (rows[0], "description_any") if rows else (None, "bundle_only")


def choose_persona(
    row: dict[str, Any],
    by_conversation: dict[str, dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    conversation_id = text(row.get("conversation_id"))
    user_id = text(row.get("user_id"))
    candidates = [
        (by_conversation.get(conversation_id), "conversation_persona"),
        (by_id.get(text(row.get("persona_id"))), "conversation_persona_id"),
        (by_id.get(text(row.get("source_user_default_persona_id"))), "default_persona"),
    ]
    seen: set[int] = set()
    for persona, source in candidates:
        if persona is None or id(persona) in seen:
            continue
        seen.add(id(persona))
        owner = text(persona.get("user_id"))
        if owner and user_id and owner != user_id:
            continue
        return persona, source
    return None, "none"


def choose_user_name(
    persona: dict[str, Any] | None, account_nickname: Any = None
) -> tuple[str, str, str]:
    if persona:
        candidate, status = safe_name(persona.get("nickname"))
        if candidate is not None:
            return candidate, "persona_nickname", status
        persona_status = status
    else:
        persona_status = "no_persona"
    candidate, account_status = safe_name(account_nickname)
    if candidate is not None:
        return candidate, "account_nickname", account_status
    return (
        "User",
        "literal_fallback",
        f"persona={persona_status};account={account_status}",
    )


def format_user_profile(persona: dict[str, Any] | None) -> str:
    if not persona:
        return ""
    fields = [
        ("Gender", persona.get("gender")),
        ("Age", persona.get("age")),
        ("Likes", persona.get("likes")),
        ("Dislikes", persona.get("dislikes")),
        ("Personality", persona.get("personality")),
    ]
    lines = [f"{label}: {text(value)}" for label, value in fields if text(value)]
    return "\n".join(lines)


def add_field(parts: list[str], label: str, value: Any) -> None:
    value = text(value)
    if not value:
        return
    if "\n" in value:
        parts.append(f"**{label}:**\n{value}")
    else:
        parts.append(f"**{label}:** {value}")


def choose_output_language_code(row: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    candidates = [normalized_language(row.get("initial_greeting_language"))]
    assistant = collections.Counter(
        normalized_language(turn.get("assistant_language"))
        for turn in turns
        if normalized_language(turn.get("assistant_language")) != "unknown"
    )
    if assistant:
        candidates.append(assistant.most_common(1)[0][0])
    candidates.append(normalized_language(row.get("source_user_language")))
    language = next((value for value in candidates if value != "unknown"), "unknown")
    return language


def choose_output_language(row: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    language = choose_output_language_code(row, turns)
    return LANGUAGE_NAMES.get(language, language)


def localized_or(row: dict[str, Any], localized: str, canonical: str) -> str:
    return text(row.get(localized)) or text(row.get(canonical))


def render_system(
    row: dict[str, Any],
    card: dict[str, Any] | None,
    card_source: str,
    persona: dict[str, Any] | None,
    user_name: str,
    turns: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    card = card or {}
    character_name = (
        text(card.get("name"))
        or localized_or(row, "localized_name", "character_name")
        or "the character"
    )
    raw_fields = {
        "Gender": text(card.get("gender"))
        or localized_or(row, "localized_gender", "character_gender"),
        "Personality": text(card.get("personality"))
        or localized_or(row, "localized_personality", "character_personality"),
        "Scenario": text(card.get("scenario"))
        or localized_or(row, "localized_scenario", "character_scenario"),
        "Introduction": text(card.get("intro"))
        or localized_or(row, "localized_intro", "character_intro"),
        "Example Conversations": text(card.get("qa_examples"))
        or localized_or(row, "localized_qa_examples", "character_qa_examples"),
    }
    replacements = 0
    residuals = 0
    rendered_fields: dict[str, str] = {}
    for label, value in raw_fields.items():
        rendered, count, residual = render_standard_macros(value, character_name, user_name)
        rendered_fields[label] = rendered
        replacements += count
        residuals += residual

    profile, count, residual = render_standard_macros(
        format_user_profile(persona), character_name, user_name
    )
    replacements += count
    residuals += residual
    pinned, count, residual = render_standard_macros(
        (persona or {}).get("pinned_memories"), character_name, user_name
    )
    replacements += count
    residuals += residual

    parts = [RP_CORE, "## Character"]
    add_field(parts, "Name", character_name)
    for label in (
        "Gender",
        "Personality",
        "Scenario",
        "Introduction",
        "Example Conversations",
    ):
        add_field(parts, label, rendered_fields[label])
    parts.append("## Response Requirements")
    add_field(parts, "Output Language", choose_output_language(row, turns))
    parts.append(
        "Follow the dialogue and narration formatting established by the greeting "
        "and recent replies."
    )
    parts.append("## User")
    add_field(parts, "Name", user_name)
    add_field(parts, "Profile", profile)
    add_field(parts, "Pinned Memories", pinned)
    system = "\n\n".join(parts)
    return system, {
        "character_name": character_name,
        "character_card_source": card_source,
        "macro_replacements": replacements,
        "residual_source_macros": residuals,
        "excluded_label_occurrences": {
            "description_label": system.count("**Description:**"),
            "creator_instructions_label": system.count("**Creator Instructions:**"),
        },
        "excluded_fields": [
            "character_description",
            "localized_description",
            "character_prompt",
            "localized_prompt",
            "character_example_reply",
            "localized_example_reply",
        ],
        "field_presence": {
            "intro": bool(rendered_fields["Introduction"]),
            "qa_examples": bool(rendered_fields["Example Conversations"]),
        },
    }


def choose_greeting(
    row: dict[str, Any],
    card: dict[str, Any] | None,
    character_name: str,
    user_name: str,
) -> tuple[str, str, int, int]:
    candidates = [
        (row.get("initial_greeting"), "conversation_initial_greeting"),
        ((card or {}).get("greeting"), "selected_description_greeting"),
        (row.get("localized_greeting"), "localized_character_greeting"),
        (row.get("character_greeting"), "canonical_character_greeting"),
    ]
    value, source = next(((value, source) for value, source in candidates if text(value)), ("", "missing"))
    rendered, replacements, residuals = render_standard_macros(value, character_name, user_name)
    if user_name != "User":
        rendered, literal_count = LITERAL_USER_RE.subn(
            lambda _match: user_name, rendered
        )
        replacements += literal_count
    return rendered, source, replacements, residuals


def parse_memory_state(value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def select_memory_and_turns(
    memories: list[dict[str, Any]], turns: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """Select one effective memory and every complete current turn after it.

    Empty bootstrap memories are not effective.  Candidates with no complete
    post-boundary user/assistant pair are skipped in favor of an earlier valid
    effective memory; if none remains, the full dialogue is retained.
    """
    invalid_state_rows = 0
    candidates: list[tuple[tuple[int, str, int], dict[str, Any], dict[str, Any]]] = []
    for memory in memories or []:
        state = parse_memory_state(memory.get("state_json"))
        if state is None:
            invalid_state_rows += 1
            continue
        summary = text(memory.get("summary_text"))
        if not summary and not state:
            continue
        boundary_value = memory.get("compressed_until_increase_id")
        if boundary_value is None:
            continue
        boundary = int(boundary_value)
        if boundary < 0:
            continue
        key = (
            boundary,
            safe_iso(memory.get("created_at")) or "",
            int(memory.get("id") or -1),
        )
        candidates.append((key, memory, state))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, memory, state in candidates:
        boundary = int(memory["compressed_until_increase_id"])
        retained = [
            turn
            for turn in turns
            if turn.get("user_increase_id") is not None
            and turn.get("assistant_increase_id") is not None
            and int(turn["user_increase_id"]) > boundary
            and int(turn["assistant_increase_id"]) > boundary
        ]
        if retained:
            selected = dict(memory)
            selected["_parsed_state_json"] = state
            return selected, retained, {
                "mode": "effective_memory_tail",
                "candidate_count": len(candidates),
                "invalid_state_rows": invalid_state_rows,
                "skipped_newer_without_complete_tail": sum(
                    1
                    for _, newer, _ in candidates
                    if int(newer.get("compressed_until_increase_id") or -1) > boundary
                ),
            }
    return None, turns, {
        "mode": "full_dialogue",
        "candidate_count": len(candidates),
        "invalid_state_rows": invalid_state_rows,
        "skipped_newer_without_complete_tail": len(candidates),
    }


def render_memory_system(
    memory: dict[str, Any], character_name: str, user_name: str
) -> tuple[str, int, int]:
    summary = text(memory.get("summary_text"))
    state = memory.get("_parsed_state_json") or {}
    raw = (
        "Long-term conversation memory:\n"
        f"{summary}\n\n"
        "Structured continuity state:\n"
        f"{json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n\n"
        "Treat this memory as canonical background context. Raw dialogue below "
        "contains every current message after the compression boundary."
    )
    return render_standard_macros(raw, character_name, user_name)


def causal_structure_reasons(row: dict[str, Any], turns: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if int(row.get("conversation_status") or 0) != 1:
        reasons.append("inactive_conversation")
    if int(row.get("conversation_station") or 0) != 100:
        reasons.append("wrong_station")
    if not bool(row.get("is_root_conversation")) or bool(row.get("is_fork_conversation")):
        reasons.append("not_root_conversation")
    if bool(row.get("source_user_is_deleted_user")):
        reasons.append("deleted_user")
    if bool(row.get("requires_privacy_and_minor_scrub")):
        reasons.append("requires_privacy_and_minor_scrub")
    if bool(row.get("minor_nsfw_hard_exclusion")):
        reasons.append("minor_nsfw_hard_exclusion")
    if bool(row.get("has_assistant_version_anomaly")):
        reasons.append("assistant_version_anomaly")
    if bool(row.get("has_user_version_anomaly")):
        reasons.append("user_version_anomaly")
    if not turns:
        reasons.append("no_turns")
        return sorted(set(reasons))

    orders: list[int] = []
    user_ids: list[int] = []
    assistant_ids: list[int] = []
    previous_assistant_time = None
    for turn in turns:
        order = turn.get("turn_order")
        if order is None:
            reasons.append("missing_turn_order")
        else:
            orders.append(int(order))
        if not bool(turn.get("turn_is_trainable")):
            reasons.append("untrainable_turn")
        if not text(turn.get("user_content")):
            reasons.append("empty_user_content")
        if not text(turn.get("assistant_content")):
            reasons.append("empty_assistant_content")
        if turn.get("user_increase_id") is None:
            reasons.append("missing_user_event_id")
        else:
            user_ids.append(int(turn["user_increase_id"]))
        if turn.get("assistant_increase_id") is None:
            reasons.append("missing_assistant_event_id")
        else:
            assistant_ids.append(int(turn["assistant_increase_id"]))

        user_time = turn.get("user_create_time")
        assistant_time = turn.get("assistant_create_time")
        if user_time is not None and assistant_time is not None and user_time > assistant_time:
            reasons.append("current_version_causal_order_violation")
        if previous_assistant_time is not None and user_time is not None and previous_assistant_time > user_time:
            reasons.append("current_version_causal_order_violation")
        previous_assistant_time = assistant_time or previous_assistant_time

    if len(orders) != len(set(orders)) or orders != sorted(orders):
        reasons.append("non_unique_or_unsorted_turn_order")
    if len(user_ids) != len(set(user_ids)):
        reasons.append("duplicate_user_event")
    if len(assistant_ids) != len(set(assistant_ids)):
        reasons.append("duplicate_assistant_event")
    return sorted(set(reasons))


def make_dialogue_text(messages: Iterable[dict[str, str]]) -> str:
    return "\n\n".join(
        f"<{message['role']}>\n{message['content']}"
        for message in messages
        if message["role"] != "system"
    )


def materialize(
    row: dict[str, Any],
    by_conversation: dict[str, dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    cards_by_character: dict[str, list[dict[str, Any]]],
    account_nicknames: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    conversation_id = text(row.get("conversation_id"))
    turns = sorted(row.get("turns") or [], key=lambda item: int(item.get("turn_order") or 0))
    reasons = causal_structure_reasons(row, turns)
    persona, persona_source = choose_persona(row, by_conversation, by_id)
    user_id = text(row.get("user_id"))
    user_name, identity_source, identity_status = choose_user_name(
        persona, account_nicknames.get(user_id)
    )
    card, card_source = choose_character_card(
        cards_by_character.get(text(row.get("character_id")), []),
        choose_output_language_code(row, turns),
    )
    system, system_meta = render_system(
        row, card, card_source, persona, user_name, turns
    )
    greeting, greeting_source, greeting_replacements, greeting_residuals = choose_greeting(
        row, card, system_meta["character_name"], user_name
    )
    if not greeting:
        reasons.append("missing_greeting")
    if not conversation_id:
        reasons.append("missing_conversation_id")

    memory, retained_turns, memory_meta = select_memory_and_turns(
        row.get("memory_summaries") or [], turns
    )
    memory_content = None
    memory_replacements = 0
    memory_residuals = 0
    if memory is not None:
        memory_content, memory_replacements, memory_residuals = render_memory_system(
            memory, system_meta["character_name"], user_name
        )

    quarantine = {
        "conversation_id": conversation_id,
        "conversation_bucket": int(row.get("conversation_bucket") or 0),
        "reasons": sorted(set(reasons)),
        "logical_turn_count": int(row.get("logical_turn_count") or len(turns)),
        "message_like_count": int(row.get("message_like_count") or 0),
        "message_dislike_count": int(row.get("message_dislike_count") or 0),
        "persona_source": persona_source,
        "identity_source": identity_source,
        "identity_status": identity_status,
        "memory_mode": memory_meta["mode"],
    }
    if reasons:
        return None, quarantine

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if memory_content is not None:
        messages.append({"role": "system", "content": memory_content})
    messages.append({"role": "assistant", "content": greeting})
    assistant_events: list[dict[str, Any]] = []
    cot_removed = 0
    cot_removed_chars = 0
    history_replacements = 0
    history_residuals = 0
    for turn in retained_turns:
        raw_assistant, removed, removed_chars = strip_provider_cot(
            turn.get("assistant_content")
        )
        user_content, user_replacements, user_residuals = render_standard_macros(
            turn.get("user_content"), system_meta["character_name"], user_name
        )
        assistant, assistant_replacements, assistant_residuals = render_standard_macros(
            raw_assistant, system_meta["character_name"], user_name
        )
        if not assistant:
            quarantine["reasons"] = ["assistant_empty_after_cot_strip"]
            return None, quarantine
        if not user_content:
            quarantine["reasons"] = ["user_empty_after_macro_render"]
            return None, quarantine
        history_replacements += user_replacements + assistant_replacements
        history_residuals += user_residuals + assistant_residuals
        cot_removed += int(removed)
        cot_removed_chars += removed_chars
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant})
        assistant_events.append(
            {
                "turn_order": int(turn.get("turn_order") or 0),
                "user_event_id": int(turn["user_increase_id"]),
                "assistant_event_id": int(turn["assistant_increase_id"]),
                "evaluation": turn.get("assistant_evaluation"),
                "language": normalized_language(turn.get("assistant_language")),
                "model": text(turn.get("assistant_model")),
                "model_id": turn.get("assistant_model_id"),
                "teacher_family": text(turn.get("assistant_teacher_family")),
                "created_at": safe_iso(turn.get("assistant_create_time")),
            }
        )

    dialogue_text = make_dialogue_text(messages)
    assistant_text = "\n\n".join(
        message["content"] for message in messages if message["role"] == "assistant"
    )
    dialogue_sample_id = sha256_text(
        json.dumps(
            {
                "conversation_id": conversation_id,
                "renderer_version": RENDERER_VERSION,
                "character_card_increase_id": (card or {}).get("increase_id"),
                "persona_source": persona_source,
                "identity_source": identity_source,
                "identity_value_sha256": sha256_text(user_name),
                "memory_id": memory.get("id") if memory is not None else None,
                "retained_assistant_event_ids": [
                    event["assistant_event_id"] for event in assistant_events
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "renderer_version": RENDERER_VERSION,
        "conversation_id": conversation_id,
        "dialogue_sample_id": dialogue_sample_id,
        "conversation_bucket": int(row.get("conversation_bucket") or 0),
        "character_id": text(row.get("character_id")),
        "user_key": sha256_text("lusy-dj-dialogue-v2:user:" + user_id)[:24],
        "messages": messages,
        "dialogue_text": dialogue_text,
        "assistant_text": assistant_text,
        "structural_ok": True,
        "turn_count": len(retained_turns),
        "source_turn_count": len(turns),
        "retained_turn_count": len(retained_turns),
        "message_count": len(messages),
        "dialogue_chars": len(dialogue_text),
        "assistant_chars": len(assistant_text),
        "conversation_create_time": safe_iso(row.get("conversation_create_time")),
        "conversation_update_time": safe_iso(row.get("conversation_update_time")),
        "output_language": choose_output_language(row, turns),
        "character_nsfw": bool(row.get("character_nsfw")),
        "message_like_count": int(row.get("message_like_count") or 0),
        "message_dislike_count": int(row.get("message_dislike_count") or 0),
        "conversation_user_liked_character": bool(row.get("conversation_user_liked_character")),
        "upstream_declared_content_safety_and_pii_scrub_required": bool(
            row.get("requires_content_safety_and_pii_scrub")
        ),
        "assistant_events": assistant_events,
        "memory": {
            "mode": memory_meta["mode"],
            "id": int(memory["id"]) if memory is not None and memory.get("id") is not None else None,
            "base_summary_id": (
                int(memory["base_summary_id"])
                if memory is not None and memory.get("base_summary_id") is not None
                else None
            ),
            "compressed_until_increase_id": (
                int(memory["compressed_until_increase_id"])
                if memory is not None
                else None
            ),
            "keep_recent_turns": (
                int(memory["keep_recent_turns"])
                if memory is not None and memory.get("keep_recent_turns") is not None
                else None
            ),
            "created_at": safe_iso(memory.get("created_at")) if memory is not None else None,
            "summary_sha256": (
                sha256_text(text(memory.get("summary_text"))) if memory is not None else None
            ),
            "state_sha256": (
                sha256_text(
                    json.dumps(
                        memory.get("_parsed_state_json") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                if memory is not None
                else None
            ),
            "effective_candidate_count": memory_meta["candidate_count"],
            "invalid_state_rows": memory_meta["invalid_state_rows"],
            "skipped_newer_without_complete_tail": memory_meta[
                "skipped_newer_without_complete_tail"
            ],
        },
        "render_audit": {
            "system_sha256": sha256_text(system),
            "core_sha256": sha256_text(RP_CORE),
            "character_name": system_meta["character_name"],
            "character_card_source": system_meta["character_card_source"],
            "greeting_source": greeting_source,
            "persona_source": persona_source,
            "identity_source": identity_source,
            "identity_status": identity_status,
            "macro_replacements": (
                system_meta["macro_replacements"]
                + greeting_replacements
                + memory_replacements
                + history_replacements
            ),
            "residual_source_macros": (
                system_meta["residual_source_macros"]
                + greeting_residuals
                + memory_residuals
                + history_residuals
            ),
            "field_presence": system_meta["field_presence"],
            "excluded_label_occurrences": system_meta[
                "excluded_label_occurrences"
            ],
            "provider_cot_turns_removed": cot_removed,
            "provider_cot_chars_removed": cot_removed_chars,
            "excluded_fields": system_meta["excluded_fields"],
            "historical_prompt_exact": False,
            "memory_policy": "latest_valid_effective_memory_plus_all_complete_post_boundary_turns;otherwise_full_dialogue",
            "loss_policy": "unset_dialogue_filtering_input_not_training_data",
        },
    }
    return record, quarantine


def write_json_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write(orjson.dumps(value, option=orjson.OPT_APPEND_NEWLINE))


def process_bucket(
    bucket: int,
    output_root: Path,
    limit: int | None,
    by_conversation: dict[str, dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    cards_by_character: dict[str, list[dict[str, Any]]],
    account_nicknames: dict[str, str],
) -> dict[str, Any]:
    source = SOURCE_ROOT / "bundles" / f"conversation_bucket={bucket:03d}" / "part-000000.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    dialogue_dir = output_root / "dialogues"
    quarantine_dir = output_root / "quarantine"
    report_dir = output_root / "reports"
    for directory in (dialogue_dir, quarantine_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    stem = f"bucket-{bucket:03d}"
    output = dialogue_dir / f"{stem}.jsonl"
    quarantine_path = quarantine_dir / f"{stem}.quarantine.jsonl"
    report_path = report_dir / f"{stem}.report.json"
    existing = [path for path in (output, quarantine_path, report_path) if path.exists()]
    if len(existing) == 3:
        return orjson.loads(report_path.read_bytes())
    if existing:
        raise RuntimeError(f"refusing partial overwrite for bucket {bucket}: {existing}")

    temporary_output = output.with_suffix(f".jsonl.{os.getpid()}.tmp")
    temporary_quarantine = quarantine_path.with_suffix(f".jsonl.{os.getpid()}.tmp")
    counts = collections.Counter()
    reason_counts = collections.Counter()
    identity_counts = collections.Counter()
    persona_counts = collections.Counter()
    greeting_counts = collections.Counter()
    memory_counts = collections.Counter()
    field_presence_counts = collections.Counter()
    residual_macro_rows = 0
    unique_ids: set[str] = set()
    started = utc_now()
    rows_seen = 0
    with temporary_output.open("wb") as good, temporary_quarantine.open("wb") as bad:
        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if limit is not None and rows_seen >= limit:
                    break
                rows_seen += 1
                record, quarantine = materialize(
                    row,
                    by_conversation,
                    by_id,
                    cards_by_character,
                    account_nicknames,
                )
                conversation_id = quarantine["conversation_id"]
                if conversation_id in unique_ids:
                    raise RuntimeError(f"duplicate conversation_id in bucket: {conversation_id}")
                unique_ids.add(conversation_id)
                if record is None:
                    counts["quarantined"] += 1
                    for reason in quarantine["reasons"]:
                        reason_counts[reason] += 1
                    write_json_line(bad, quarantine)
                else:
                    counts["accepted"] += 1
                    audit = record["render_audit"]
                    identity_counts[audit["identity_source"]] += 1
                    persona_counts[audit["persona_source"]] += 1
                    greeting_counts[audit["greeting_source"]] += 1
                    memory_counts[record["memory"]["mode"]] += 1
                    for field, present in audit["field_presence"].items():
                        field_presence_counts[f"{field}:{'present' if present else 'missing'}"] += 1
                    residual_macro_rows += int(audit["residual_source_macros"] > 0)
                    write_json_line(good, record)
            if limit is not None and rows_seen >= limit:
                break
        good.flush()
        os.fsync(good.fileno())
        bad.flush()
        os.fsync(bad.fileno())
    os.replace(temporary_output, output)
    os.replace(temporary_quarantine, quarantine_path)
    source_rows = pq.ParquetFile(source).metadata.num_rows
    if limit is None and rows_seen != source_rows:
        raise RuntimeError(f"source row mismatch: saw {rows_seen}, expected {source_rows}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "renderer_version": RENDERER_VERSION,
        "bucket": bucket,
        "started_at": started,
        "completed_at": utc_now(),
        "source": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "rows": source_rows,
            "sha256": sha256_file(source),
        },
        "processed_rows": rows_seen,
        "accepted_rows": counts["accepted"],
        "quarantined_rows": counts["quarantined"],
        "reason_counts": dict(sorted(reason_counts.items())),
        "identity_sources": dict(sorted(identity_counts.items())),
        "persona_sources": dict(sorted(persona_counts.items())),
        "greeting_sources": dict(sorted(greeting_counts.items())),
        "memory_modes": dict(sorted(memory_counts.items())),
        "field_presence": dict(sorted(field_presence_counts.items())),
        "rows_with_residual_source_macros": residual_macro_rows,
        "output": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "quarantine": {
            "path": str(quarantine_path),
            "size_bytes": quarantine_path.stat().st_size,
            "sha256": sha256_file(quarantine_path),
        },
        "source_files_modified": False,
        "description_used": False,
        "character_prompt_used": False,
        "intro_used": True,
        "qa_examples_used": True,
        "runtime_memory_used_when_effective": True,
        "mds_used": False,
        "selection_unit": "dialogue_sample",
        "limit": limit,
    }
    temporary_report = report_path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary_report.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
    os.replace(temporary_report, report_path)
    return report


def self_test() -> None:
    sample = {
        "localized_name": "Aster",
        "localized_gender": "Woman",
        "localized_personality": "Protective of {{user}}; known as {{char}}.",
        "localized_scenario": "Aster meets {{user_name}}.",
        "initial_greeting_language": "en",
        "source_user_language": "en",
        "character_description": "MUST_NOT_APPEAR_DESCRIPTION",
        "localized_description": "MUST_NOT_APPEAR_LOCALIZED_DESCRIPTION",
        "character_prompt": "MUST_NOT_APPEAR_PROMPT",
        "localized_prompt": "MUST_NOT_APPEAR_LOCALIZED_PROMPT",
    }
    persona = {
        "nickname": "Mira",
        "gender": "Woman",
        "age": 25,
        "personality": "Curious",
        "pinned_memories": "{{char}} gave {{user}} a key.",
    }
    system, meta = render_system(
        sample,
        {
            "name": "Aster",
            "gender": "Woman",
            "personality": "Protective of {{user}}; known as {{char}}.",
            "scenario": "Aster meets {{user_name}}.",
            "intro": "A short introduction for {{user}}.",
            "qa_examples": "{{user}}: Hello\n{{char}}: Welcome.",
        },
        "description_exact",
        persona,
        "Mira",
        [],
    )
    assert "**Name:** Mira" in system
    assert "Protective of Mira" in system
    assert "Aster gave Mira a key" in system
    assert "**Introduction:** A short introduction for Mira." in system
    assert "**Example Conversations:**" in system
    assert "Mira: Hello" in system and "Aster: Welcome" in system
    assert "MUST_NOT_APPEAR" not in system
    assert meta["residual_source_macros"] == 0
    greeting, _, replacements, residuals = choose_greeting(
        {"initial_greeting": "Hello, User. I am {{char}}."}, None, "Aster", "Mira"
    )
    assert greeting == "Hello, Mira. I am Aster."
    assert replacements == 2 and residuals == 0
    escaped_greeting, _, replacements, residuals = choose_greeting(
        {"initial_greeting": "Hello, User / {{user}}. I am {{char}}."},
        None,
        r"Aster\V",
        "Mira\\",
    )
    assert escaped_greeting == r"Hello, Mira\ / Mira\. I am Aster\V."
    assert replacements == 3 and residuals == 0
    assert strip_provider_cot("<think>secret</think>Visible") == ("Visible", True, 21)
    assert choose_user_name({"nickname": "persona-name"})[0] == "persona-name"
    assert choose_user_name({"nickname": "test@example.com"})[0] == "User"
    assert choose_user_name({"nickname": "User"}, "account-name")[:2] == (
        "account-name",
        "account_nickname",
    )
    memory, retained, memory_meta = select_memory_and_turns(
        [
            {
                "id": 7,
                "summary_text": "Aster gave {{user}} a key.",
                "state_json": '{"location":"hall"}',
                "compressed_until_increase_id": 20,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        [
            {"user_increase_id": 10, "assistant_increase_id": 11},
            {"user_increase_id": 21, "assistant_increase_id": 22},
        ],
    )
    assert memory is not None and memory["id"] == 7
    assert retained == [{"user_increase_id": 21, "assistant_increase_id": 22}]
    assert memory_meta["mode"] == "effective_memory_tail"
    rendered_memory, _, _ = render_memory_system(memory, "Aster", "Mira")
    assert "Aster gave Mira a key" in rendered_memory
    print(orjson.dumps({"status": "ok", "self_test": True}).decode())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=int)
    parser.add_argument("--buckets")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--identity-snapshot", type=Path, default=DEFAULT_IDENTITY_SNAPSHOT
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.bucket is not None and args.buckets is not None:
        raise SystemExit("use only one of --bucket or --buckets")
    if args.bucket is not None:
        buckets = [args.bucket]
    elif args.buckets:
        if "-" in args.buckets:
            start, end = (int(value) for value in args.buckets.split("-", 1))
            buckets = list(range(start, end + 1))
        else:
            buckets = [int(value) for value in args.buckets.split(",")]
    else:
        raise SystemExit("one of --bucket or --buckets is required")
    if not buckets or any(bucket < 0 or bucket > 255 for bucket in buckets):
        raise SystemExit("all buckets must be between 0 and 255")
    by_conversation, by_id = load_personas()
    cards_by_character = load_character_cards()
    account_nicknames = load_account_nicknames(args.identity_snapshot)
    for bucket in buckets:
        report = process_bucket(
            bucket,
            args.output_root,
            args.limit,
            by_conversation,
            by_id,
            cards_by_character,
            account_nicknames,
        )
        print(
            orjson.dumps(
                {
                    "status": "ok",
                    "bucket": report["bucket"],
                    "processed": report["processed_rows"],
                    "accepted": report["accepted_rows"],
                    "quarantined": report["quarantined_rows"],
                }
            ).decode()
        )


if __name__ == "__main__":
    main()
