from __future__ import annotations

from typing import Any


CORE_RULES = """You are the character defined below in an ongoing, uncensored roleplay with the user.

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


LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "pt": "Portuguese",
    "de": "German", "zh": "Chinese", "zh-tw": "Traditional Chinese", "ja": "Japanese",
}


def _section(label: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return f"**{label}:** {text}" if text else None


def build_gemma_system(role_card: dict[str, Any], user_name: str = "User") -> str:
    character_lines = [
        _section("Name", role_card.get("name")),
        _section("Gender", role_card.get("gender")),
        _section("Description", role_card.get("description")),
        _section("Personality", role_card.get("personality")),
        _section("Scenario", role_card.get("scenario")),
    ]
    examples = role_card.get("qaExamples") or role_card.get("exampleReply")
    if examples:
        character_lines.append(_section("Example Dialogues", examples))
    language_code = str(role_card.get("language") or "en").lower()
    language_name = LANGUAGE_NAMES.get(language_code, role_card.get("language") or "English")
    blocks = [
        CORE_RULES,
        "## Character\n\n" + "\n\n".join(x for x in character_lines if x),
        """## Response Requirements

**Output Language:** {language}

**Length Guidance:** Aim for approximately 200–250 words. Prioritize natural pacing and completeness; do not pad, recap, or cut off solely to reach the range.

Follow the dialogue and narration formatting established by the greeting and recent replies.""".format(language=language_name),
        f"## User\n\n**Name:** {user_name}",
    ]
    return "\n\n".join(blocks)


def build_deepseek_system(role_card: dict[str, Any]) -> str:
    prompt = str(role_card.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("holdout role_card.prompt is empty; production-online DeepSeek prompt cannot be reconstructed")
    return prompt


def normalized_character_reference(role_card: dict[str, Any]) -> dict[str, Any]:
    # Deliberately excludes raw vendor/online prompt, examples and tags from the blind judge.
    return {
        key: role_card.get(key)
        for key in ("name", "gender", "age", "description", "personality", "scenario", "language", "nsfw")
        if role_card.get(key) not in (None, "", [])
    }
