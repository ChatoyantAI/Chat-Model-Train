"""High-fidelity reconstruction of production roleplay model input messages.

The implementation in this module is a standalone port of the prompt-construction
path at production commit ``f8ab72c314fafa843677a94269e29dbdfb94f344``.  It
deliberately has no dependency on the production repository or a live database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


PRODUCTION_CODE_COMMIT = "f8ab72c314fafa843677a94269e29dbdfb94f344"
SMALL_CONTEXT_WINDOW_K = 16
KEEP_RECENT_TURNS = 25
TEXT_MEDIA_TYPES = frozenset((0, 4, 5))
TEMPLATE_DIR = Path(__file__).with_name("templates")

CONTINUE_USER_QUERY = (
    "Continue seamlessly from the exact point where the previous response ended.\n"
    "Advance the plot with meaningful new developments. Each continuation should "
    "change the situation, tension, information, relationship, or emotional state "
    "while remaining consistent with the established context and causality.\n"
    "Do not restart the scene, recap, summarize, paraphrase, or repeat recent "
    "dialogue, narration, descriptions, actions, conclusions, or prompts to the user.\n"
    "Preserve the established continuity, chronology, tone, point of view, pacing, "
    "formatting, and character behavior.\n"
    "When the established scene includes flirtation, intimate physical contact, "
    "sexual tension, or sexual activity, actively heighten the chemistry and move the "
    "intimacy forward. Let the character show bolder initiative, clearer desire, "
    "stronger tension, and sustained physical and emotional momentum in direct response "
    "to what the user has already established. When the context supports it, gradually "
    "increase the intensity instead of remaining coy, vague, repetitive, or stalled. "
    "Do not abruptly sanitize, fade out, skip over, rush, or force escalation.\n"
    "Do not invent or decide the user's dialogue, thoughts, actions, choices, or "
    "decisions.\n"
    "Do not mention this request, these instructions, or that the response is a "
    "continuation."
)

CHAT_IMAGE_SYSTEM_PROMPT = (
    "The current user message may include a structured [User uploaded image context] block "
    "before [User message]. This block contains information about images the user uploaded "
    "for the current turn. Pay attention to the uploaded image information and consider it "
    "together with the user's text when forming the reply. Keep the response natural and "
    "aligned with the character and conversation."
)

CHAT_IMAGE_USER_GUIDELINE = (
    "- If image context is present, pay attention to the user's uploaded image information "
    "and respond naturally with both the image context and the user's text in mind."
)

OUTPUT_LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "de": "German",
    "zh-TW": "Traditional Chinese",
}

_ZH_LANGUAGE_ALIASES = {
    "zh",
    "zh-cn",
    "zh-hans",
    "zh-hant",
    "zh-hk",
    "zh-mo",
    "zh-tw",
}

_DYNAMIC_PLACEHOLDER_PATTERN = re.compile(
    r"\{\{(user|char|subject|object|possessive_adjective|possessive_pronoun|reflexive|sub|obj|poss|ref)\}\}",
    re.IGNORECASE,
)
_PLACEHOLDER_ALIASES = {
    "sub": "subject",
    "obj": "object",
    "poss": "possessive_adjective",
    "ref": "reflexive",
}
_PRONOUNS_BY_GENDER = {
    0: {
        "subject": "he",
        "object": "him",
        "possessive_adjective": "his",
        "possessive_pronoun": "his",
        "reflexive": "himself",
    },
    1: {
        "subject": "she",
        "object": "her",
        "possessive_adjective": "her",
        "possessive_pronoun": "hers",
        "reflexive": "herself",
    },
    2: {
        "subject": "they",
        "object": "them",
        "possessive_adjective": "their",
        "possessive_pronoun": "theirs",
        "reflexive": "themselves",
    },
}


class ContextBuildError(ValueError):
    """Raised when frozen inputs cannot be reconstructed safely."""


@dataclass(frozen=True)
class RestoredHistory:
    """Physical message rows restored to their state at candidate generation time."""

    messages: tuple[dict[str, Any], ...]
    image_contexts_by_message_key: Mapping[str, tuple[dict[str, Any], ...]]
    trigger_message_key: str | None
    audit: Mapping[str, int]


@dataclass(frozen=True)
class ContextBuildResult:
    """Reconstructed model input and non-content audit metadata."""

    input_messages: tuple[dict[str, str], ...]
    audit: Mapping[str, Any]


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized)


def restore_history(
    comparison: Mapping[str, Any],
    context_rows: Sequence[Mapping[str, Any]],
    image_rows: Sequence[Mapping[str, Any]],
    mutation_rows: Sequence[Mapping[str, Any]],
) -> RestoredHistory:
    """Reverse later edits/version changes and recover generation-time text rows."""

    started_at = _parse_time(comparison["timestamps"].get("started_at"))
    messages_by_increase_id: dict[int, dict[str, Any]] = {}
    active_by_increase_id: dict[int, bool] = {}
    trigger_message_key: str | None = None

    for row in context_rows:
        message = dict(row["message"])
        increase_id = int(message["increase_id"])
        if row.get("is_trigger_message"):
            trigger_message_key = str(message["message_key"])
            continue
        messages_by_increase_id[increase_id] = message
        active_by_increase_id[increase_id] = bool(
            row.get("active_at_generation_heuristic")
        )

    reversed_regenerations = 0
    reversed_version_selections = 0
    reversed_edits = 0
    for event in reversed(mutation_rows):
        event_time = _parse_time(event.get("created_at"))
        if (
            event_time is None
            or started_at is None
            or event_time <= started_at
        ):
            continue

        event_type = event.get("event_type")
        if event_type == "regenerate":
            replacement = event.get("replacement_increase_id")
            source = event.get("source_increase_id")
            applied = replacement in active_by_increase_id or source in active_by_increase_id
            if replacement in active_by_increase_id:
                active_by_increase_id[int(replacement)] = False
            if source in active_by_increase_id:
                active_by_increase_id[int(source)] = True
            reversed_regenerations += int(applied)
        elif event_type == "select_version":
            target = event.get("target_increase_id")
            source = event.get("source_increase_id")
            applied = target in active_by_increase_id or source in active_by_increase_id
            if target in active_by_increase_id:
                active_by_increase_id[int(target)] = False
            if source in active_by_increase_id:
                active_by_increase_id[int(source)] = True
            reversed_version_selections += int(applied)
        elif event_type == "edit":
            target = event.get("target_increase_id")
            if target in messages_by_increase_id and event.get("before_content") is not None:
                messages_by_increase_id[int(target)]["content"] = event["before_content"]
                reversed_edits += 1

    images_by_message: dict[str, list[dict[str, Any]]] = {}
    for image in image_rows:
        if image.get("analysis_status") != "completed":
            continue
        analysis_text = str(image.get("analysis_text") or "").strip()
        if not analysis_text:
            continue
        message_key = str(image["message_key"])
        images_by_message.setdefault(message_key, []).append(
            {
                "sort_order": int(image.get("sort_order") or 0),
                "analysis_text": analysis_text,
            }
        )
    for contexts in images_by_message.values():
        contexts.sort(key=lambda item: item["sort_order"])

    restored: list[dict[str, Any]] = []
    for row in context_rows:
        if row.get("is_trigger_message"):
            continue
        increase_id = int(row["message"]["increase_id"])
        message = messages_by_increase_id.get(increase_id)
        if message is None or not active_by_increase_id.get(increase_id, False):
            continue
        if message.get("role") not in ("user", "assistant"):
            continue
        if int(message.get("media_type") or 0) not in TEXT_MEDIA_TYPES:
            continue
        restored.append(dict(message))

    frozen_images = {
        key: tuple(contexts) for key, contexts in images_by_message.items()
    }
    return RestoredHistory(
        messages=tuple(restored),
        image_contexts_by_message_key=frozen_images,
        trigger_message_key=trigger_message_key,
        audit={
            "physical_rows": len(context_rows),
            "active_text_rows": len(restored),
            "reversed_regenerations": reversed_regenerations,
            "reversed_version_selections": reversed_version_selections,
            "reversed_edits": reversed_edits,
        },
    )


def select_mode_by_tokens(max_tokens: int) -> str:
    if max_tokens <= 50:
        return "chat"
    if max_tokens >= 450:
        return "longText"
    return "scene"


def _normalize_language_code(language: str | None) -> str | None:
    if not language:
        return None
    normalized = language.strip().replace("_", "-")
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in _ZH_LANGUAGE_ALIASES:
        return "zh-TW"
    return lowered


def _supported_language(language: str | None) -> str:
    normalized = _normalize_language_code(language)
    if not normalized:
        return "en"
    canonical = {key.lower(): key for key in OUTPUT_LANGUAGE_NAMES}
    if normalized.lower() in canonical:
        return canonical[normalized.lower()]
    return canonical.get(normalized.lower().split("-", 1)[0], "en")


def _output_language_name(language: str | None) -> str:
    return OUTPUT_LANGUAGE_NAMES[_supported_language(language)]


def _load_templates() -> tuple[dict[str, str], dict[str, Any]]:
    names = (
        "prompt_chat",
        "prompt_scene",
        "prompt_long_text",
        "prompt_chat_user_priority",
        "prompt_scene_user_priority",
        "prompt_long_text_user_priority",
        "prompt_collaborative",
        "prompt_green",
        "prompt_green_collaborative",
        "prompt_restricted_topics",
    )
    try:
        templates = {
            name: (TEMPLATE_DIR / f"{name}.txt").read_text(encoding="utf-8")
            for name in names
        }
        basic = json.loads(
            (TEMPLATE_DIR / "basic_chat_prompts.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextBuildError(f"cannot load frozen production templates: {exc}") from exc
    return templates, basic


def _remove_markdown_links(text: Any) -> str:
    if not text:
        return ""
    result = re.sub(r"!\[.*?\]\(.*?\)", "", str(text))
    result = re.sub(r"\[.*?\]\(.*?\)", "", result)
    result = re.sub(r"http[s]?://\S+", "", result)
    return result.strip()


def _pronouns_for_gender(user_gender: int | None) -> Mapping[str, str]:
    if user_gender in (1, 4):
        return _PRONOUNS_BY_GENDER[1]
    if user_gender == 2:
        return _PRONOUNS_BY_GENDER[2]
    return _PRONOUNS_BY_GENDER[0]


def _match_placeholder_case(value: str, placeholder: str) -> str:
    if placeholder.isupper():
        return value.upper()
    if placeholder[:1].isupper() and placeholder[1:].islower():
        return value[:1].upper() + value[1:]
    return value


def replace_dynamic_placeholders(
    text: str | None,
    *,
    user_name: str = "user",
    character_name: str = "",
    user_gender: int | None = None,
) -> str | None:
    if text is None:
        return None
    pronouns = _pronouns_for_gender(user_gender)

    def replacer(match: re.Match[str]) -> str:
        placeholder = match.group(1)
        key = _PLACEHOLDER_ALIASES.get(placeholder.lower(), placeholder.lower())
        if key == "user":
            return user_name or "user"
        if key == "char":
            return character_name or ""
        return _match_placeholder_case(pronouns[key], placeholder)

    return _DYNAMIC_PLACEHOLDER_PATTERN.sub(replacer, text)


def _substitute_card_macros(
    value: str, *, character_name: str, user_name: str
) -> str:
    replacements = {
        "{{char}}": character_name,
        "{{Char}}": character_name,
        "{{CHAR}}": character_name,
        "{{user}}": user_name,
        "{{User}}": user_name,
        "{{USER}}": user_name,
    }
    for macro, replacement in replacements.items():
        value = value.replace(macro, replacement)
    return value


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _matches_lorebook_key(
    text: str,
    key: str,
    *,
    case_sensitive: bool,
    match_whole_words: bool,
) -> bool:
    key = key.strip()
    if not key:
        return False
    flags = 0 if case_sensitive else re.IGNORECASE
    if match_whole_words:
        return re.search(rf"(?<!\w){re.escape(key)}(?!\w)", text, flags) is not None
    if case_sensitive:
        return key in text
    return key.casefold() in text.casefold()


def _is_lorebook_entry_active(
    entry: Mapping[str, Any],
    *,
    scan_text: str,
    character_name: str,
    user_name: str,
) -> bool:
    if entry.get("constant") is True:
        return True
    keys = entry.get("keys")
    if not isinstance(keys, list) or not keys:
        return False
    primary_matches = [
        _matches_lorebook_key(
            scan_text,
            _substitute_card_macros(
                key, character_name=character_name, user_name=user_name
            ),
            case_sensitive=entry.get("case_sensitive") is True,
            match_whole_words=entry.get("match_whole_words") is True,
        )
        for key in keys
        if isinstance(key, str)
    ]
    if not any(primary_matches):
        return False
    secondary_keys = entry.get("secondary_keys")
    if (
        entry.get("selective") is not True
        or not isinstance(secondary_keys, list)
        or not secondary_keys
    ):
        return True
    secondary_matches = [
        _matches_lorebook_key(
            scan_text,
            _substitute_card_macros(
                key, character_name=character_name, user_name=user_name
            ),
            case_sensitive=entry.get("case_sensitive") is True,
            match_whole_words=entry.get("match_whole_words") is True,
        )
        for key in secondary_keys
        if isinstance(key, str)
    ]
    logic = _as_int(entry.get("selective_logic"), 0)
    if logic == 1:
        return not all(secondary_matches)
    if logic == 2:
        return not any(secondary_matches)
    if logic == 3:
        return all(secondary_matches)
    return any(secondary_matches)


def _build_card_lorebook_prompt(
    card_lorebook: Any,
    *,
    query: str,
    message_list: Sequence[Mapping[str, Any]],
    character_name: str,
    user_name: str,
) -> str:
    if not isinstance(card_lorebook, dict):
        return ""
    entries = card_lorebook.get("entries")
    if not isinstance(entries, list):
        return ""
    scan_parts = [str(query or "")]
    for message in list(message_list)[-8:]:
        if message.get("content"):
            scan_parts.append(str(message["content"]))
    scan_text = "\n".join(scan_parts)

    activated: list[tuple[int, int, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if not _is_lorebook_entry_active(
            entry,
            scan_text=scan_text,
            character_name=character_name,
            user_name=user_name,
        ):
            continue
        content = _substitute_card_macros(
            str(entry.get("content") or "").strip(),
            character_name=character_name,
            user_name=user_name,
        )
        if content:
            activated.append((_as_int(entry.get("insertion_order"), 100), index, content))
    activated.sort(key=lambda item: (-item[0], item[1]))

    selected: list[str] = []
    used_characters = 0
    for _, _, content in activated:
        added = len(content) + (2 if selected else 0)
        if used_characters + added > 24_000:
            continue
        selected.append(content)
        used_characters += added
    if not selected:
        return ""
    return "**Character Card Lore:**\n" + "\n\n".join(selected)


def _inject_prompt_block_before_last_anchor(
    template: str, prompt_block: str, anchor: str
) -> str:
    if not template or not prompt_block:
        return template
    block = prompt_block.strip()
    if not block or block in template:
        return template
    anchor_index = template.rfind(anchor)
    if anchor_index < 0:
        return f"{template.rstrip()}\n\n{block}\n"
    return (
        f"{template[:anchor_index].rstrip()}\n\n"
        f"{block}\n\n"
        f"{template[anchor_index:].lstrip()}"
    )


def _collaborative_mode_instruction(mode: str) -> str:
    instructions = {
        "chat": (
            "**Dialogue Mode:**\n"
            "- Respond only with the active character's spoken dialogue.\n"
            "- Do not output narration, action prose, quotation marks, asterisks, "
            "brackets, stage directions, or name prefixes.\n"
            "- Express reaction, emotion, and initiative through wording, rhythm, "
            "pauses, and subtext.\n"
            "- Interpret narration and multi-character material in the user's story "
            "contribution before choosing what the active character says next.\n"
            "- End on a live conversational beat that leaves room for the story to continue."
        ),
        "scene": (
            "**Scene Mode:**\n"
            "- Write a compact live scene in close third-person, anchored in the "
            "active character's perception unless the established focus clearly shifts.\n"
            "- Interweave spoken dialogue with selective action, sensory detail, and "
            "environmental response.\n"
            "- Keep the scene moving; each beat should change action, tension, "
            "information, distance, or emotional position.\n"
            "- Write dialogue in quotation marks and narration in asterisks, with no "
            "name prefixes and no nested formatting.\n"
            "- End on an immediate, unfinished beat that invites the next story contribution."
        ),
        "longText": (
            "**Long-Form Mode:**\n"
            "- Sustain a coherent extended scene in close third-person, anchored in "
            "the active character's perception unless the established focus clearly shifts.\n"
            "- Build a connected multi-paragraph sequence of action, reaction, dialogue, "
            "observation, and emotional change.\n"
            "- Give each paragraph a distinct purpose while preserving spatial logic, "
            "timeline, causality, and physical continuity.\n"
            "- Use richer sensory and psychological detail without padding, repetition, "
            "recap, or unrelated subplots.\n"
            "- Write dialogue in quotation marks and narration in asterisks, with no "
            "name prefixes and no nested formatting.\n"
            "- End on a live beat rather than a summary or closed conclusion."
        ),
    }
    return instructions.get(mode, instructions["scene"])


def _build_persona_prompt_block(persona: Mapping[str, Any] | None) -> str:
    if not persona:
        return ""
    lines = ["\n<user_profile>"]
    for key in ("nickname", "gender", "age", "likes", "dislikes"):
        value = persona.get(key)
        if value not in (None, "", []):
            label = "name" if key == "nickname" else key
            lines.append(f"{label}: {value}")
    personality = persona.get("personality") or persona.get("about_you")
    if personality not in (None, ""):
        lines.append(f"personality: {personality}")
    pinned_memories = persona.get("pinned_memories") or []
    if pinned_memories:
        lines.append("[Pinned Memories]")
        lines.extend(f"- {memory}" for memory in pinned_memories if memory)
    lines.append("</user_profile>")
    return "\n".join(lines)


def _length_guideline(mode: str, max_tokens: int, language: str) -> str:
    guidelines = {
        "chat": (
            f"\n\n<guideline>\n"
            f"- Target length: {int(max_tokens * 0.8)}-{max_tokens} words total\n"
            f"- Pure dialogue only, no actions or descriptions\n"
            f"- Prioritize direct speech, immediate responses, emotional tone\n"
            f"- Never include word count, token count, or meta commentary\n"
            f"- All output must be in {language} language\n"
            f"</guideline>"
        ),
        "scene": (
            f"\n\n<guideline>\n"
            f"- Target length: {int(max_tokens * 0.8)}-{max_tokens} words total\n"
            f"- Interweave dialogue and narration naturally for engaging rhythm\n"
            f"- Alternate between speech and action for flow\n"
            f"- Reply in this format: Dialogue in \"quotes\", narration in *asterisks*. Don't nest symbols (no quotes inside asterisks, no asterisks inside quotes).\n"
            f"- Never include word count, token count, or meta commentary\n"
            f"- All output must be in {language} language\n"
            f"</guideline>"
        ),
        "longText": (
            f"\n\n<guideline>\n"
            f"- Target length: {int(max_tokens * 0.8)}-{max_tokens} words total\n"
            f"- Blend dialogue and rich narration seamlessly for immersive experience\n"
            f"- Layer sensory details, emotional depth, and character voice\n"
            f"- Let moments breathe, build atmosphere organically\n"
            f"- Reply in this format: Dialogue in \"quotes\", narration in *asterisks*. Don't nest symbols (no quotes inside asterisks, no asterisks inside quotes).\n"
            f"- Never include word count, token count, or meta commentary\n"
            f"- All output must be in {language} language\n"
            f"</guideline>"
        ),
    }
    return guidelines.get(mode, guidelines["scene"])


def _append_extra_user_guideline(length_guideline: str, extra: str | None) -> str:
    extra = (extra or "").strip()
    if not extra:
        return length_guideline
    if not length_guideline:
        return f"\n\n<guideline>\n{extra}\n</guideline>"
    close_tag = "</guideline>"
    if close_tag not in length_guideline:
        return f"{length_guideline}\n{extra}"
    prefix, suffix = length_guideline.rsplit(close_tag, 1)
    return f"{prefix}{extra}\n{close_tag}{suffix}"


def _build_augmented_chat_query(
    query: str, image_contexts: Sequence[Mapping[str, Any]]
) -> str:
    image_blocks = [
        f"Image {index}:\n{context['analysis_text']}"
        for index, context in enumerate(image_contexts, start=1)
    ]
    return (
        "[User uploaded image context]\n"
        "The user uploaded the following images for this turn. "
        "These descriptions summarize the visible image content.\n\n"
        + "\n\n".join(image_blocks)
        + "\n\n[User message]\n"
        + (query or "").strip()
    )


def _group_messages_to_turns(
    messages: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    turns: list[list[Mapping[str, Any]]] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    current_turn: list[Mapping[str, Any]] | None = None
    for position, message in enumerate(messages):
        if int(message.get("media_type") or 0) not in TEXT_MEDIA_TYPES:
            continue
        message_key = str(message.get("message_key") or position)
        turn = grouped.get(message_key)
        if turn is None:
            if message.get("role") == "assistant" and current_turn is not None:
                turn = current_turn
            else:
                turn = []
                turns.append(turn)
            grouped[message_key] = turn
        turn.append(message)
        current_turn = turn
    return turns


def _max_turn_increase_id(turn: Iterable[Mapping[str, Any]]) -> int:
    return max(int(message.get("increase_id") or 0) for message in turn)


def _select_candidate_history(
    messages: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any] | None,
    context_window_k: int,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any] | None, dict[str, Any]]:
    is_small = 0 < context_window_k < SMALL_CONTEXT_WINDOW_K
    boundary: int | None = None
    boundary_reason = "none"
    effective_summary: Mapping[str, Any] | None = None

    if is_small:
        turns = _group_messages_to_turns(messages)
        if len(turns) > KEEP_RECENT_TURNS:
            removed_turn = turns[-KEEP_RECENT_TURNS - 1]
            boundary = _max_turn_increase_id(removed_turn)
            messages = [message for turn in turns[-KEEP_RECENT_TURNS:] for message in turn]
            boundary_reason = "small_context_recent_turns"
    elif summary is not None:
        boundary = int(summary.get("compressed_until_increase_id") or 0)
        messages = [
            message
            for message in messages
            if int(message.get("increase_id") or 0) > boundary
        ]
        effective_summary = summary
        boundary_reason = "memory_summary"
    else:
        turns = _group_messages_to_turns(messages)
        if len(turns) >= KEEP_RECENT_TURNS:
            if len(turns) == KEEP_RECENT_TURNS:
                boundary = 0
            else:
                removed_turn = turns[-KEEP_RECENT_TURNS - 1]
                boundary = _max_turn_increase_id(removed_turn)
                messages = [
                    message
                    for message in messages
                    if int(message.get("increase_id") or 0) > boundary
                ]
            boundary_reason = "memory_bootstrap_recent_turns"

    return list(messages), effective_summary, {
        "is_small_context_model": is_small,
        "history_boundary": boundary,
        "history_boundary_reason": boundary_reason,
        "summary_record_present": summary is not None,
        "summary_ignored_for_small_context": bool(is_small and summary is not None),
    }


def _coalesce_history_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    image_contexts_by_message_key: Mapping[str, Sequence[Mapping[str, Any]]],
    user_name: str,
    character_name: str,
    user_gender: int | None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    result: list[dict[str, str]] = []
    historical_image_contexts_used = 0
    for item in messages:
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _remove_markdown_links(item.get("content"))
        if role == "assistant":
            content = replace_dynamic_placeholders(
                content,
                user_name=user_name,
                character_name=character_name,
                user_gender=user_gender,
            ) or ""
        else:
            image_contexts = image_contexts_by_message_key.get(
                str(item.get("message_key")), ()
            )
            if image_contexts:
                historical_image_contexts_used += len(image_contexts)
                content = _build_augmented_chat_query(content, image_contexts)

        if result and result[-1]["role"] == role:
            if content:
                result[-1]["content"] = (
                    f"{result[-1]['content']}\n\n{content}".strip()
                )
        else:
            result.append({"role": str(role), "content": content})

    dropped_unanswered_user_block = 0
    if result and result[-1]["role"] == "user":
        result.pop()
        dropped_unanswered_user_block = 1
    return result, {
        "coalesced_history_messages": len(result),
        "dropped_unanswered_user_block": dropped_unanswered_user_block,
        "historical_image_contexts_used": historical_image_contexts_used,
    }


def _format_character_prompt(
    template: str,
    *,
    character: Mapping[str, Any],
    persona: Mapping[str, Any] | None,
    query: str,
    message_list: Sequence[Mapping[str, Any]],
    language: str,
    max_tokens: int,
    mode: str,
    user_gender: int | None,
) -> str:
    character_name = str(character.get("name") or "your dear")
    user_name = str(persona.get("nickname")) if persona and persona.get("nickname") else "user"
    first_message = _remove_markdown_links(character.get("greeting"))
    card_description = str(character.get("card_prompt_description") or "").strip()
    card_lorebook_block = _build_card_lorebook_prompt(
        character.get("card_lorebook"),
        query=query,
        message_list=message_list,
        character_name=character_name,
        user_name=user_name,
    )
    replacements = {
        "char": character_name,
        "Char": character_name,
        "CHAR": character_name,
        "User": user_name,
        "user": user_name,
        "USER": user_name,
        "gender": str(character.get("gender") or ""),
        "Gender": str(character.get("gender") or ""),
        "GENDER": str(character.get("gender") or ""),
        "age": str(character.get("age") or ""),
        "description": str(character.get("description") or ""),
        "card_prompt_description_block": (
            f"- Description: {card_description}" if card_description else ""
        ),
        "card_lorebook_block": card_lorebook_block,
        "personality": str(character.get("personality")),
        "scenario": str(character.get("scenario")),
        "qa_examples": str(character.get("qa_examples")),
        "first_message": first_message,
        "language": language,
        "max_tokens": str(max_tokens),
        "mode_instruction": _collaborative_mode_instruction(mode),
    }

    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        return replacements.get(key, match.group(0))

    result = re.sub(r"\{\{(.*?)\}\}", replacer, template)
    return replace_dynamic_placeholders(
        result,
        user_name=user_name,
        character_name=character_name,
        user_gender=user_gender,
    ) or ""


def _runtime_value(runtime: Mapping[str, Any], key: str) -> Any:
    if key not in runtime:
        raise ContextBuildError(f"runtime snapshot is missing {key!r}")
    return runtime[key]


def build_candidate_context(
    comparison: Mapping[str, Any],
    candidate_side: str,
    restored_history: RestoredHistory,
    runtime: Mapping[str, Any],
) -> ContextBuildResult:
    """Build the candidate-specific ``input_messages`` used for single-response judging."""

    if candidate_side not in ("left", "right"):
        raise ContextBuildError("candidate_side must be 'left' or 'right'")
    runtime_candidates = _runtime_value(runtime, "candidates")
    if candidate_side not in runtime_candidates:
        raise ContextBuildError(f"runtime snapshot has no {candidate_side!r} candidate")

    max_tokens = int(_runtime_value(runtime, "chat_mod"))
    user_gender = runtime.get("user_gender")
    context_window_k = int(runtime_candidates[candidate_side]["context_window_k"])
    is_test_account = bool(_runtime_value(runtime, "is_test_account"))
    prompt_experiment = runtime.get("prompt_experiment") or {}
    prompt_variant = prompt_experiment.get("variant_value")
    mode = select_mode_by_tokens(max_tokens)

    trigger = comparison["trigger"]
    trigger_type = trigger.get("type")
    query = CONTINUE_USER_QUERY if trigger_type == "continue" else str(trigger.get("content") or "")
    language_code = str(trigger.get("language") or "en")
    output_language = _output_language_name(language_code)

    summary = comparison.get("latest_memory_summary_before_generation")
    selected_history, effective_summary, trim_audit = _select_candidate_history(
        restored_history.messages,
        summary,
        context_window_k,
    )

    character_snapshot = comparison["character_snapshot_at_export"]
    description = character_snapshot["description"]
    greeting_source = "conversation_initial_greeting"
    greeting = comparison.get("conversation", {}).get("initial_greeting")
    if not greeting:
        greeting_source = "character_description_greeting"
        greeting = description.get("greeting")

    character = {
        "name": description.get("name"),
        "gender": description.get("gender"),
        "age": character_snapshot.get("age"),
        "description": description.get("description"),
        "personality": description.get("personality"),
        "scenario": description.get("scenario"),
        "qa_examples": description.get("qa_examples"),
        "greeting": greeting,
        "card_prompt_description": character_snapshot.get("card_prompt_description"),
        "card_lorebook": character_snapshot.get("card_lorebook"),
    }
    persona = comparison.get("persona_snapshot_at_export") or None
    character_name = str(character.get("name") or "your dear")
    user_name = str(persona.get("nickname")) if persona and persona.get("nickname") else "user"

    templates, basic_chat_prompts = _load_templates()
    if is_test_account:
        template_name = (
            "prompt_green_collaborative"
            if bool(character_snapshot.get("third_person_perspective"))
            else "prompt_green"
        )
    elif bool(character_snapshot.get("third_person_perspective")):
        template_name = "prompt_collaborative"
    else:
        template_name = {
            "chat": "prompt_chat",
            "scene": "prompt_scene",
            "longText": "prompt_long_text",
        }[mode]
        if prompt_variant == "user_priority":
            template_name += "_user_priority"

    template = _inject_prompt_block_before_last_anchor(
        templates[template_name],
        templates["prompt_restricted_topics"],
        "**Output Format:**",
    )
    system_content = _format_character_prompt(
        template,
        character=character,
        persona=persona,
        query=query,
        message_list=selected_history,
        language=output_language,
        max_tokens=max_tokens,
        mode=mode,
        user_gender=user_gender,
    )
    system_content += _build_persona_prompt_block(persona)
    system_content = replace_dynamic_placeholders(
        system_content,
        user_name=user_name,
        character_name=character_name,
        user_gender=user_gender,
    ) or ""

    input_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content}
    ]

    current_image_contexts = restored_history.image_contexts_by_message_key.get(
        restored_history.trigger_message_key or "", ()
    )
    if current_image_contexts:
        input_messages.append(
            {"role": "system", "content": CHAT_IMAGE_SYSTEM_PROMPT}
        )

    summary_used = False
    if effective_summary is not None:
        summary_state = effective_summary.get("state_json") or {}
        summary_text = str(effective_summary.get("summary_text") or "").strip()
        if summary_text or summary_state:
            summary_used = True
            memory_content = (
                "Long-term conversation memory:\n"
                f"{summary_text}\n\n"
                "Structured continuity state:\n"
                f"{json.dumps(summary_state, ensure_ascii=False)}\n\n"
                "Treat this memory as canonical background context. Raw dialogue below is the newest post-summary tail."
            )
            input_messages.append({"role": "system", "content": memory_content})

    greeting_included = not is_test_account
    if greeting_included:
        if not greeting:
            greeting_source = "basic_chat_prompt"
            language_prompts = basic_chat_prompts.get(
                language_code, basic_chat_prompts.get("en", {})
            )
            greeting = language_prompts.get("basic_greeting", "")
        formatted_greeting = _format_character_prompt(
            str(greeting or ""),
            character=character,
            persona=persona,
            query=query,
            message_list=selected_history,
            language=output_language,
            max_tokens=max_tokens,
            mode=mode,
            user_gender=user_gender,
        )
        input_messages.append(
            {"role": "assistant", "content": _remove_markdown_links(formatted_greeting)}
        )
    else:
        greeting_source = "embedded_or_omitted_by_green_template"

    history_messages, history_audit = _coalesce_history_messages(
        selected_history,
        image_contexts_by_message_key=restored_history.image_contexts_by_message_key,
        user_name=user_name,
        character_name=character_name,
        user_gender=user_gender,
    )
    input_messages.extend(history_messages)

    current_query = query
    if current_image_contexts:
        current_query = _build_augmented_chat_query(current_query, current_image_contexts)
    guideline = _length_guideline(mode, max_tokens, output_language)
    guideline = _append_extra_user_guideline(
        guideline,
        CHAT_IMAGE_USER_GUIDELINE if current_image_contexts else None,
    )
    input_messages.append({"role": "user", "content": current_query + guideline})

    retained_ids = [int(message.get("increase_id") or 0) for message in selected_history]
    audit = {
        "production_code_commit": PRODUCTION_CODE_COMMIT,
        "template": template_name,
        "mode": mode,
        "chat_mod": max_tokens,
        "context_window_k": context_window_k,
        "summary_used": summary_used,
        **trim_audit,
        "history_first_retained_increase_id": min(retained_ids) if retained_ids else None,
        "history_last_retained_increase_id": max(retained_ids) if retained_ids else None,
        "history_retained_physical_messages": len(selected_history),
        "greeting_source": greeting_source,
        "greeting_included": greeting_included,
        "current_image_contexts_used": len(current_image_contexts),
        **dict(restored_history.audit),
        **history_audit,
        "field_sources": {
            "character": "frozen export-time snapshot",
            "persona": "frozen export-time snapshot",
            "conversation_greeting": "frozen conversation snapshot",
            "summary": "frozen latest-before-generation record",
            "history": "frozen rows plus mutation-event reconstruction",
            "query": "frozen trigger record or production Continue constant",
            "chat_mod": "current production snapshot",
            "user_gender": "current production snapshot",
            "is_test_account": "current production/config snapshot",
            "prompt_experiment": "current production snapshot",
            "context_window_k": "current production model snapshot",
        },
        "current_snapshot_fields_present": True,
    }
    return ContextBuildResult(tuple(input_messages), audit)
