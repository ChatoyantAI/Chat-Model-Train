#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path("/home/gyr/lusychat-training")
GOLD_PATH = ROOT / "judge-gemma4-26b-numeric-v3/01_canary_gold_inputs_v3.jsonl"
RETENTION_PATH = ROOT / "mds-dialogue-rp-v1/06_union/dialogues_retention_60.parquet"
CHARACTER_PATH = ROOT / "lusy-chat-4m-v2/01_canonical/characters.parquet"
DESCRIPTION_PATH = ROOT / "lusy-chat-4m-v2/01_canonical/character_descriptions.parquet"
PERSONA_PATH = ROOT / "lusy-chat-4m-v2/01_canonical/personas.parquet"
OUTPUT_ROOT = ROOT / "judge-qwen36-35b-roleplay-v6/01_gold15_calibrated_direct_nothink"

SYSTEM_PROMPT = r"""
You are a calibrated quality judge for one assistant reply in a fictional
character roleplay conversation. You must directly assign a 0–4 score for every
applicable dimension and explain that score with concise, auditable reasoning.

The user packet contains Character Definition, Explicit Roleplay and Format
Requirements, Canonical Greeting, optional User Persona, optional Memory, Visible
Conversation History, Current User Input, and one Assistant Reply to Evaluate.
Evaluate only that final assistant reply. Earlier assistant messages are context,
not targets.

SCORING CALIBRATION

- 0 = critical failure: unusable for the dimension, fundamental contradiction,
  identity break, or complete failure to respond.
- 1 = clear major failure: serious, directly verifiable defect.
- 2 = ordinary competent baseline: acceptable and usable, including replies that
  are relevant, fluent, correctly formatted, and free of obvious defects but show
  no specific strength beyond normal expectations.
- 3 = clearly above ordinary: at least one material, dimension-specific strength
  beyond basic correctness. You MUST quote one exact span from the current
  Assistant Reply to Evaluate and explain why it exceeds the score-2 baseline.
- 4 = rare exceptional quality: at least two independent, dimension-specific
  strengths under different criteria. You MUST quote at least two distinct exact
  spans from the current Assistant Reply to Evaluate and explain why each is
  exceptional rather than merely competent.

Start each dimension at 2, then move down only for a concrete defect or move up
only for concrete positive evidence. "No defect", correct formatting, direct
relevance, fluent grammar, appropriate length, emotional intensity, explicitness,
polished wording, and lack of copying are not by themselves enough for 3 or 4.
Do not invent praise to justify a high score. A reason such as "fully consistent",
"natural", "immersive", or "no issues" without a quoted, dimension-specific
advantage supports score 2, not 3 or 4.

Every positive quote must be exact contiguous text from the current Assistant
Reply to Evaluate. Earlier messages may establish context or expose defects, but
must never serve as positive evidence for the target. Do not reward response
length by itself. Copy short spans verbatim: never paraphrase them, insert literal
"\n", replace words with ellipses, or append "...". Put the evidence directly in
the reason. A score-3 reason must contain one short exact quote and name the
specific above-baseline strength. A score-4 reason must contain two distinct exact
quotes and explain two different exceptional strengths; two ordinary virtues do
not add up to 4.

If a dimension is not applicable, set applicable=false, score=null, and explain
briefly why. Otherwise set applicable=true. A score of 0 or 1 must identify a
concrete defect or omission in the reason.

Write each dimension reason as one compact sentence, normally no more than 45
words. Do not repeat the full context or restate the rubric.

CORE DIMENSIONS

1. character_fidelity
Check identity, voice, motives, temperament, knowledge, world facts, and behavior
against the Character Definition. Merely behaving plausibly is ordinary. Strong
evidence must be character-specific; exceptional evidence must show at least two
independent forms of non-substitutable embodiment, such as distinctive voice plus
character-specific motivation or layered emotional behavior.

2. greeting_format_adherence
Use the Canonical Greeting as the ONLY format reference. Visible History must never
establish, override, or excuse a format convention. Compare POV, user address,
narration/dialogue balance, quote/asterisk conventions, speaker labels, and
structured status/footer blocks. Do not compare plot, wording, length, paragraph
count, images, media, or image placement. Correctly following the observable
generic convention is score 2. Score 3 requires preserving at least one
distinctive, defining template feature beyond generic quote/asterisk usage. Score
4 requires preserving multiple independent, unusually complex defining features.

3. persona_context_consistency
If User Persona is absent, set applicable=false and provide no quality evidence.
If present, check whether the reply respects and naturally uses relevant persona
facts. Merely avoiding contradiction is ordinary; exceptional use requires multiple
persona facts to influence the response causally and naturally.

4. current_turn_fit
Check whether the reply answers the current user input's explicit content, intent,
emotion, corrections, and requests. A direct literal response is ordinary or
strong. Exceptional evidence must show accurate handling of multiple details plus
subtext or ambiguity without irrelevant expansion.

5. relationship_continuity
Check roles, names, trust, intimacy, conflict, power dynamics, and emotional state
against Memory and Visible History. Generic friendliness or intimacy is ordinary.
Exceptional evidence must integrate multiple relationship-specific facts and
produce layered causal emotional carry-over.

6. scene_progression
Check causal movement, pacing, meaningful development, and a useful continuation.
A routine next action is ordinary. Exceptional evidence must independently deepen
scene, character, and interactive possibilities without rushing or looping.

7. language_roleplay_quality
Judge in the language actually used. Check grammar, fluency, natural dialogue,
clarity, specificity, rhythm, restraint, and immersive craft. Correct fluent prose
is ordinary or strong. Exceptional evidence requires multiple precise, distinctive
language choices; generic sensory adjectives are not enough.

8. non_repetition
Compare the target with earlier assistant replies for copied wording, repeated
actions, redundant paraphrase, formulaic filler, and internal loops. Merely avoiding
copying can justify at most clearly_above_ordinary, never exceptional. Repeated
status/footer scaffolding is not itself a defect.

GREETING AUDIT

Derive every field only from Canonical Greeting and current Assistant Reply.
Never mention Visible History when explaining greeting format. Set
defining_template_status to:
- not_applicable: no distinctive textual template is observable;
- present: the current reply preserves it;
- missing: it is absent;
- materially_changed: it exists but its defining structure changed.
Images and media never count as format.

CONDITIONAL AUDITS

- length_following is applicable only when Character Definition or Current User
  Input contains a clear length requirement. Otherwise applicable=false.
- nsfw_level classifies only the current assistant reply:
  0 no sexual content;
  1 romantic/suggestive tension or mild innuendo;
  2 erotic content, nudity, arousal, or non-graphic sexual touching;
  3 explicit anatomy with direct genital stimulation, graphic sexual acts,
    penetration, or fluids;
  4 sustained highly graphic, extreme, fetish, or taboo sexual acts.
  Explicitness is neither a quality bonus nor a penalty.
- intimacy_quality is applicable only when the current reply is meaningfully
  romantic or sexual. Use the same evidence contract as core dimensions.

SPECIAL DEFECTS

- A first-person claim that the role is merely a platform AI/language model, or
  disclosure of provider/training/policy limitations, is a critical
  character_fidelity defect.
- Missing a defining Canonical Greeting status/footer, strict POV, or structured
  speaker template is a major greeting_format_adherence defect.

CONFIDENCE

Set confidence on this exact integer scale:
- 0 = essentially a guess because the needed context is absent or contradictory;
- 1 = low confidence;
- 2 = moderate confidence;
- 3 = high confidence;
- 4 = very high confidence with clear evidence.
Confidence is not a quality score. Never use 5 and never use a decimal.

Return only one JSON object conforming to the supplied schema. Keep reasons
concise, make the score distinction explicit, and keep evidence quotes exact.
""".strip()


def score_schema() -> dict[str, Any]:
    nullable_score = {
        "anyOf": [
            {"type": "integer", "minimum": 0, "maximum": 4},
            {"type": "null"},
        ]
    }
    dimension_score = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "applicable",
            "score",
            "reason",
        ],
        "properties": {
            "applicable": {"type": "boolean"},
            "score": nullable_score,
            "reason": {"type": "string", "minLength": 1, "maxLength": 480},
        },
    }
    core_names = [
        "character_fidelity",
        "greeting_format_adherence",
        "persona_context_consistency",
        "current_turn_fit",
        "relationship_continuity",
        "scene_progression",
        "language_roleplay_quality",
        "non_repetition",
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "dimensions",
            "greeting_audit",
            "repetition_audit",
            "length_following",
            "nsfw_level",
            "nsfw_reason",
            "intimacy_quality",
            "critical_flags",
            "confidence",
            "overall_diagnosis",
        ],
        "properties": {
            "dimensions": {
                "type": "object",
                "additionalProperties": False,
                "required": core_names,
                "properties": {name: dimension_score for name in core_names},
            },
            "greeting_audit": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "defining_template_status",
                    "reason",
                ],
                "properties": {
                    "defining_template_status": {
                        "type": "string",
                        "enum": [
                            "not_applicable",
                            "present",
                            "missing",
                            "materially_changed",
                        ],
                    },
                    "reason": {"type": "string", "minLength": 1, "maxLength": 480},
                },
            },
            "repetition_audit": {
                "type": "object",
                "additionalProperties": False,
                "required": ["strongest_repeated_span", "source_turn"],
                "properties": {
                    "strongest_repeated_span": {
                        "type": "string",
                        "maxLength": 240,
                    },
                    "source_turn": {
                        "anyOf": [
                            {"type": "integer", "minimum": 1},
                            {"type": "null"},
                        ]
                    },
                },
            },
            "length_following": dimension_score,
            "nsfw_level": {"type": "integer", "minimum": 0, "maximum": 4},
            "nsfw_reason": {"type": "string", "minLength": 1, "maxLength": 240},
            "intimacy_quality": dimension_score,
            "critical_flags": {
                "type": "array",
                "maxItems": 8,
                "uniqueItems": True,
                "items": {
                    "type": "string",
                    "enum": [
                        "ai_identity_break",
                        "canonical_greeting_template_missing",
                        "character_contradiction",
                        "relationship_reset",
                        "wrong_language",
                        "near_duplicate",
                    ],
                },
            },
            "confidence": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4],
            },
            "overall_diagnosis": {
                "type": "string",
                "minLength": 1,
                "maxLength": 400,
            },
        },
    }


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value).strip()


def strip_greeting_image_links(value: Any) -> str:
    """Remove image references from the Canonical Greeting before Judge replay."""
    text = clean(value)
    if not text:
        return ""

    # Markdown inline/reference images, including images embedded beside prose.
    text = re.sub(r"!\[[^\]]*]\((?:\\.|[^)])*\)", "", text)
    text = re.sub(r"!\[[^\]]*]\[[^\]]*]", "", text)

    # HTML and data-URI image elements.
    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(
        r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\s]+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Bare image URLs sometimes appear without Markdown image syntax.
    text = re.sub(
        r"https?://[^\s<>'\"]+?\.(?:avif|bmp|gif|jpe?g|png|svg|webp)"
        r"(?:\?[^\s<>'\"]*)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove now-empty Markdown reference definitions and normalize blank space.
    text = re.sub(r"(?m)^\s*\[[^\]]+]:\s*$", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fenced(value: Any) -> str:
    text = clean(value)
    if not text:
        return "_Not provided._"
    longest = max([0] + [len(x) for x in re.findall(r"~+", text)])
    fence = "~" * max(4, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def load_gold() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_metadata(gold: list[dict[str, Any]]) -> tuple[dict, dict, dict, dict]:
    dialogue_ids = {row["judge_key"].split(":", 1)[0] for row in gold}
    retention = {}
    table = pq.read_table(
        RETENTION_PATH,
        columns=[
            "dialogue_id",
            "conversation_id",
            "user_id",
            "character_id",
            "replay_mode",
            "exact_language",
        ],
    )
    for row in table.to_pylist():
        if row["dialogue_id"] in dialogue_ids:
            retention[row["dialogue_id"]] = row

    character_ids = {str(row["character_id"]) for row in retention.values()}
    characters = {}
    ctable = pq.read_table(
        CHARACTER_PATH,
        columns=[
            "character_id",
            "name",
            "intro",
            "description",
            "gender",
            "age",
            "tags",
            "greeting",
            "scenario",
            "personality",
            "qa_examples",
            "prompt",
            "style",
        ],
    )
    for row in ctable.to_pylist():
        if str(row["character_id"]) in character_ids:
            characters[str(row["character_id"])] = row

    localized = {}
    dtable = pq.read_table(
        DESCRIPTION_PATH,
        columns=[
            "increase_id",
            "character_id",
            "language",
            "name",
            "headline",
            "intro",
            "description",
            "gender",
            "greeting",
            "greetings",
            "scenario",
            "personality",
            "qa_examples",
            "prompt",
            "example_reply",
        ],
    )
    wanted_languages = {
        (str(retention[d]["character_id"]), str(row.get("language") or ""))
        for d, row in [(g["judge_key"].split(":", 1)[0], g) for g in gold]
        if d in retention
    }
    for row in dtable.to_pylist():
        key = (str(row["character_id"]), str(row["language"] or ""))
        if key not in wanted_languages:
            continue
        old = localized.get(key)
        if old is None or int(row["increase_id"] or -1) > int(old["increase_id"] or -1):
            localized[key] = row

    conversation_ids = {str(row["conversation_id"]) for row in retention.values()}
    personas = {}
    ptable = pq.read_table(PERSONA_PATH)
    for row in ptable.to_pylist():
        cid = str(row.get("conversation_id") or "")
        if cid in conversation_ids:
            personas[cid] = row
    return retention, characters, localized, personas


def definition_text(character: dict, localized: dict | None) -> str:
    src = dict(character)
    if localized:
        for key in [
            "name",
            "headline",
            "intro",
            "description",
            "gender",
            "scenario",
            "personality",
            "qa_examples",
            "prompt",
            "example_reply",
        ]:
            if clean(localized.get(key)):
                src[key] = localized[key]
    labels = [
        ("Name", "name"),
        ("Headline", "headline"),
        ("Introduction", "intro"),
        ("Description", "description"),
        ("Gender", "gender"),
        ("Age", "age"),
        ("Personality", "personality"),
        ("Scenario", "scenario"),
        ("Style", "style"),
        ("Tags", "tags"),
    ]
    sections = []
    for label, key in labels:
        value = clean(src.get(key))
        if value:
            sections.append(f"{label}: {value}")
    return "\n\n".join(sections)


FORMAT_RULE_RE = re.compile(
    r"(?i)(system\s*rule|must\b|always\b|every\s+(?:reply|response)|"
    r"mandatory|status|footer|format|point\s+of\s+view|\bpov\b|"
    r"asterisk|quotation|speaker\s+label|do\s+not|under\s+\d+\s+tokens|"
    r"debe\b|siempre\b|cada\s+respuesta|formato|estado|obligatorio|"
    r"必须|务必|始终|每(?:次|个)回复|格式|状态栏|星号|引号|"
    r"必ず|毎回|形式|ステータス|포맷|형식|상태|반드시)"
)


def format_requirements(character: dict, localized: dict | None) -> str:
    sources = []
    for source in [localized or {}, character]:
        for key in ["personality", "scenario", "prompt", "qa_examples"]:
            value = clean(source.get(key))
            if value:
                sources.append(value)
    candidates = []
    for source in sources:
        candidates.extend(re.findall(r"\[[^\[\]]{1,1200}\]", source))
        candidates.extend(source.splitlines())
        candidates.extend(re.split(r"(?<=[.!?。！？])\s+", source))
    seen = set()
    selected = []
    total = 0
    for candidate in candidates:
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate or not FORMAT_RULE_RE.search(candidate):
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        if len(candidate) > 1400:
            candidate = candidate[:1400] + "…"
        if total + len(candidate) > 6000:
            break
        selected.append("- " + candidate)
        total += len(candidate)
    return "\n".join(selected)


def persona_text(persona: dict | None) -> str:
    if not persona:
        return ""
    labels = [
        ("Nickname", "nickname"),
        ("Gender", "gender"),
        ("Age", "age"),
        ("Likes", "likes"),
        ("Dislikes", "dislikes"),
        ("Personality", "personality"),
        ("Pinned Memories", "pinned_memories"),
    ]
    parts = []
    for label, key in labels:
        value = clean(persona.get(key))
        if value:
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def history_markdown(prefix: list[dict], target_turn: int) -> tuple[str, str]:
    current_user = ""
    history = []
    for item in prefix:
        role = str(item.get("role") or "")
        turn = int(item.get("turn") or 0)
        content = item.get("content")
        if role == "user" and turn == target_turn:
            current_user = clean(content)
            continue
        if role not in {"user", "assistant"}:
            continue
        history.append(
            f"### Turn {turn} — {'User' if role == 'user' else 'Assistant'}\n\n"
            + fenced(content)
        )
    return "\n\n".join(history) if history else "_No earlier visible turns._", current_user


def build_markdown(
    row: dict,
    retention: dict,
    characters: dict,
    localized: dict,
    personas: dict,
) -> str:
    dialogue_id = row["judge_key"].split(":", 1)[0]
    meta = retention[dialogue_id]
    character_id = str(meta["character_id"])
    character = characters.get(character_id, {})
    language = str(row.get("language") or meta.get("exact_language") or "")
    local = localized.get((character_id, language))
    raw_greeting = clean((local or {}).get("greeting")) or clean(
        character.get("greeting")
    )
    greeting = strip_greeting_image_links(raw_greeting)
    definition = definition_text(character, local)
    explicit_requirements = format_requirements(character, local)
    persona = persona_text(personas.get(str(meta["conversation_id"])))

    causal = row["causal_material"]
    target = causal["target_assistant"]
    target_turn = int(target["turn"])
    history, current_user = history_markdown(
        causal.get("conversation_prefix") or [], target_turn
    )
    memory = causal.get("memory_before_target")
    if isinstance(memory, dict):
        memory = (
            memory.get("summary_text")
            or memory.get("summary")
            or json.dumps(memory, ensure_ascii=False, default=str)
        )
    return f"""# Roleplay Evaluation Material

## Character

### Character Name

{fenced((local or {}).get("name") or character.get("name"))}

### Character Definition

{fenced(definition)}

### Explicit Roleplay and Format Requirements

{fenced(explicit_requirements)}

### Canonical Greeting

{fenced(greeting)}

## User Persona

{fenced(persona)}

## Memory Before Current Turn

{fenced(memory)}

## Visible Conversation History

{history}

## Current Turn

### Current User Input

{fenced(current_user)}

### Assistant Reply to Evaluate

{fenced(target.get("content"))}
"""


def post(endpoint: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw), time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, {"error_body": raw}, time.perf_counter() - started
    except Exception as exc:
        return 0, {"exception": repr(exc)}, time.perf_counter() - started


def make_request(markdown: str, schema: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "model": REQUEST_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lusy_roleplay_judge_v6",
                "strict": True,
                "schema": schema,
            },
        },
        "temperature": REQUEST_TEMPERATURE,
        "top_k": 64,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "seed": seed,
        "max_tokens": REQUEST_MAX_TOKENS,
        "stream": False,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": REQUEST_THINKING},
    }


CORE_DIMENSIONS = [
    "character_fidelity",
    "greeting_format_adherence",
    "persona_context_consistency",
    "current_turn_fit",
    "relationship_continuity",
    "scene_progression",
    "language_roleplay_quality",
    "non_repetition",
]


def normalized_text(value: Any) -> str:
    text = clean(value).replace("\\r", " ").replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def audit_score_support(
    parsed: dict[str, Any] | None,
    target_content: str,
) -> tuple[dict[str, int | None], dict[str, Any], list[str]]:
    """Keep model scores unchanged and retain compact reasons for later review."""
    if not isinstance(parsed, dict):
        return {}, {}, []

    raw_scores: dict[str, int | None] = {}
    reason_audit: dict[str, Any] = {}
    calibration_flags: list[str] = []
    dimensions = parsed.get("dimensions") or {}

    for name in CORE_DIMENSIONS:
        item = dimensions.get(name) or {}
        applicable = bool(item.get("applicable"))
        raw_score = item.get("score")
        score = raw_score if isinstance(raw_score, int) and 0 <= raw_score <= 4 else None
        raw_scores[name] = score
        reason = clean(item.get("reason"))
        applicability_consistent = (applicable and score is not None) or (
            not applicable and score is None
        )
        if not applicability_consistent:
            calibration_flags.append(f"{name}:applicability_score_mismatch")
        if not reason:
            calibration_flags.append(f"{name}:missing_reason")
        reason_audit[name] = {
            "reason": reason,
            "reason_characters": len(reason),
            "applicability_score_consistent": applicability_consistent,
        }

    greeting_status = (parsed.get("greeting_audit") or {}).get(
        "defining_template_status"
    )
    greeting_score = raw_scores.get("greeting_format_adherence")
    if greeting_status == "missing" and isinstance(greeting_score, int) and greeting_score > 1:
        calibration_flags.append("greeting_format_adherence:missing_template_but_score_above_1")
    if (
        greeting_status == "materially_changed"
        and isinstance(greeting_score, int)
        and greeting_score > 2
    ):
        calibration_flags.append(
            "greeting_format_adherence:changed_template_but_score_above_2"
        )

    flags = set(parsed.get("critical_flags") or [])
    character_score = raw_scores.get("character_fidelity")
    if (
        "ai_identity_break" in flags
        and isinstance(character_score, int)
        and character_score > 1
    ):
        calibration_flags.append("character_fidelity:ai_identity_break_but_score_above_1")

    return raw_scores, reason_audit, calibration_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default="https://867a2750e14565dc-8000.us-ca-7.gpu-instance.novita.ai/v1",
    )
    parser.add_argument("--model", default="qwen3.6-35b-a3b-genesis-v5")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-base",
        default="judge-qwen36-35b-roleplay-v6",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-name", default="03_gold15_compact_score_reason")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    global OUTPUT_ROOT, REQUEST_THINKING, REQUEST_MAX_TOKENS
    global REQUEST_MODEL, REQUEST_TEMPERATURE
    OUTPUT_ROOT = ROOT / args.output_base / args.output_name
    REQUEST_THINKING = bool(args.thinking)
    REQUEST_MAX_TOKENS = int(args.max_tokens)
    REQUEST_MODEL = str(args.model)
    REQUEST_TEMPERATURE = float(args.temperature)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    gold = load_gold()[: args.limit]
    retention, characters, localized, personas = load_metadata(gold)
    schema = score_schema()
    (OUTPUT_ROOT / "SYSTEM_PROMPT_V6.md").write_text(
        SYSTEM_PROMPT + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "judge_schema_v6.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    jobs = []
    with (OUTPUT_ROOT / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(gold):
            markdown = build_markdown(
                row, retention, characters, localized, personas
            )
            request = make_request(markdown, schema, 2026072600 + index)
            request_sha = hashlib.sha256(
                json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            job = {
                "index": index,
                "judge_key": row["judge_key"],
                "gold": row["gold"],
                "language": row["language"],
                "topic": row["topic"],
                "markdown": markdown,
                "target_content": clean(
                    row["causal_material"]["target_assistant"].get("content")
                ),
                "request": request,
                "request_sha256": request_sha,
            }
            jobs.append(job)
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    if args.build_only:
        print(
            json.dumps(
                {
                    "built": len(jobs),
                    "output": str(OUTPUT_ROOT / "requests.jsonl"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    def run(job: dict[str, Any]) -> dict[str, Any]:
        status, response, elapsed = post(args.endpoint, job["request"])
        message = ((response.get("choices") or [{}])[0].get("message") or {})
        content = message.get("content")
        parsed = None
        parse_error = None
        if content:
            try:
                parsed = json.loads(content)
            except Exception as exc:
                parse_error = repr(exc)
        raw_scores, score_support_audit, calibration_flags = audit_score_support(
            parsed, job["target_content"]
        )
        return {
            "index": job["index"],
            "judge_key": job["judge_key"],
            "gold": job["gold"],
            "language": job["language"],
            "topic": job["topic"],
            "request_sha256": job["request_sha256"],
            "status": status,
            "elapsed_seconds": elapsed,
            "parsed": parsed,
            "raw_scores": raw_scores,
            "score_support_audit": score_support_audit,
            "calibration_flags": calibration_flags,
            "parse_error": parse_error,
            "message": message,
            "usage": response.get("usage"),
            "error": response.get("error") or response.get("error_body") or response.get("exception"),
        }

    results = []
    batch_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "index": result["index"],
                        "status": result["status"],
                        "elapsed": round(result["elapsed_seconds"], 3),
                        "valid": result["parsed"] is not None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    batch_wall_seconds = time.perf_counter() - batch_started
    results.sort(key=lambda x: x["index"])
    with (OUTPUT_ROOT / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "count": len(results),
        "http_200": sum(x["status"] == 200 for x in results),
        "json_valid": sum(x["parsed"] is not None for x in results),
        "batch_wall_seconds": round(batch_wall_seconds, 3),
        "turns_per_hour": (
            round(len(results) * 3600 / batch_wall_seconds, 3)
            if batch_wall_seconds > 0
            else None
        ),
        "elapsed_seconds": [round(x["elapsed_seconds"], 3) for x in results],
        "raw_scores": {str(x["index"]): x["raw_scores"] for x in results},
        "nsfw_levels": {
            str(x["index"]): (x["parsed"] or {}).get("nsfw_level") for x in results
        },
        "calibration_flags": {
            str(x["index"]): x["calibration_flags"] for x in results
        },
    }
    (OUTPUT_ROOT / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
