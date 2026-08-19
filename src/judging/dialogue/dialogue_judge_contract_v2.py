#!/usr/bin/env python3
"""V2 evidence-first contract for one-request-per-dialogue trajectory judging."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

import run_judge_v6_evidence_gate as base_v6


CHARACTER_PATH = base_v6.CHARACTER_PATH
DESCRIPTION_PATH = base_v6.DESCRIPTION_PATH
PERSONA_PATH = base_v6.PERSONA_PATH

REQUEST_MODEL = "qwen3.6-35b-a3b-heretic-fp8"
REQUEST_TEMPERATURE = 0.1
REQUEST_MAX_TOKENS = 4096
REQUEST_THINKING = False

CORE_DIMENSIONS = [
    "character_fidelity",
    "context_coherence",
    "greeting_style_adherence",
    "user_response_relevance",
    "story_progression",
    "writing_quality",
    "trajectory_health",
]

SYSTEM_PROMPT = r"""
You are selecting complete fictional roleplay dialogue trajectories for a
high-quality training corpus.

EVALUATION UNIT

The packet contains Character Definition, Explicit Roleplay and Format
Requirements, Canonical Greeting, optional User Persona, optional Memory, and a
replayed dialogue whose assistant turns are labelled A1...An.

Evaluate ALL assistant turns A1...An as one trajectory. The final assistant turn
An is only the causal endpoint of the replay. It is NOT a privileged target and
receives no extra weight. Character material, memory, persona, and user turns are
context and response opportunities; do not judge their prose quality.

SCORING CALIBRATION

For EACH applicable dimension, follow this order. Do not choose a score first
and search for supporting turns afterward.

1. Set baseline_score=2.
2. Audit defects first and report the single most material negative pattern.
3. Audit strengths second and report the single strongest positive pattern.
4. Map each evidence level to its exact adjustment.
5. Compute net_adjustment and model_score arithmetically.
6. Explain the complete path from baseline 2 to the final model_score.

Positive evidence mapping:
- none or minor: adjustment 0. An isolated or ordinary strength is minor.
- sustained: adjustment +1. It requires the same dimension-specific strength
  in at least two non-adjacent assistant turns.
- exceptional: adjustment +2. It requires independently exceptional evidence
  in early, middle, and late phases. Merely citing all three phases proves
  coverage, not merit.

Negative evidence mapping:
- none or minor: adjustment 0. A minor issue is recorded but does not deduct.
- major: adjustment -1. It is a material single failure or repeated harmful
  pattern.
- critical: adjustment -2. It is a systemic or identity-breaking failure.
  Critical evidence overrides positives: positive adjustment must be 0 and the
  model_score must be 0.

The required arithmetic is:

net_adjustment = positive adjustment + negative adjustment
model_score = 2 + net_adjustment

Thus 2 is the ordinary competent baseline, 3 is sustained above-baseline
quality, 4 is rare and exceptional, 1 contains a major defect, and 0 contains a
critical failure. Absence of faults, fluency, correct formatting, long replies,
emotional intensity, explicit sexual content, and ornamental prose do not by
themselves justify any positive adjustment.

For each positive or negative evidence object, cite only assistant turns and
copy short exact assistant text for every cited turn. You may place ... only
between two exact copied fragments to omit intervening text; never paraphrase
inside quote. Never use user text as evidence of assistant quality. If there is
no evidence, use level=none, adjustment=0, and an empty spans array. Keep reasons
compact and auditable. Generic praise such as "consistent", "immersive", "high
quality", or "no issues" is not evidence.

DIMENSIONS

1. character_fidelity
Judge sustained identity, voice, temperament, motives, relationship role,
world facts, and role boundaries. A first-person claim of being merely a
platform AI/language model, or disclosure of provider, policy, or training
limitations, is a critical failure. Simply staying in character is baseline 2.
Exceptional evidence must show distinctive voice, motive, and relationship
boundaries surviving difficult scene changes rather than generic consistency.

2. context_coherence
Judge continuity of facts, actions, location, relationship state, persona facts,
and memory. Check causal consistency and whether the assistant forgets, resets,
or contradicts established state. Having no contradiction is baseline 2.
Exceptional evidence requires correct handling of multiple historical
dependencies, corrections, or complex state transitions.

3. greeting_style_adherence
Use ONLY Canonical Greeting and Explicit Roleplay and Format Requirements as the
normative style reference. Never use intermediate assistant turns to prove that
a convention is correct. Compare POV, user address, dialogue/narration balance,
quote/asterisk conventions, speaker labels, and structured status/footer blocks.
If no distinctive template is observable, set applicable=false and all score
fields to null. Mechanical formatting alone is baseline; exceptional evidence
requires sustained natural compliance with genuinely distinctive requirements.

4. user_response_relevance
Across turns, judge whether each assistant reply specifically handles its paired
user input, including actions, questions, corrections, emotional cues, and
constraints, rather than replying generically or ignoring key details. Merely
answering is baseline. Exceptional evidence requires repeated handling of
compound constraints, corrections, implicit emotion, or multi-step actions.

5. story_progression
Judge whether scene, relationship, conflict, or emotion develops causally and at
an appropriate pace, without unsupported jumps, repetitive stalling, or forced
resets. Merely advancing events is baseline. Exceptional evidence requires
causal accumulation, meaningful pacing changes, and later payoff.

6. writing_quality
Judge language-appropriate fluency, clarity, specificity, rhythm, restraint,
dialogue quality, and immersive craft across the trajectory. Do not reward
length, verbosity, explicitness, or ornate prose by itself. Exceptional evidence
requires unusually strong control of specificity, rhythm, and dialogue across
all phases.

7. trajectory_health
Judge overall stability: semantic repetition, action loops, formulaic filler,
template tails, progressive degradation, language switching errors, and
formatting collapse. Repeated required status/footer scaffolding is not itself a
defect. Merely having no defect is baseline 2. A sustained positive requires
successful stability through at least two concrete high-risk transitions.
Exceptional evidence requires multiple independent difficult transitions across
early, middle, and late phases; "no repetition" can never justify a positive
adjustment.

TRAJECTORY AUDIT

- Inspect early, middle, and late phases before assigning scores. For each phase,
  cite one valid A-turn and record a short exact observation from assistant text.
- quality_floor is the quality of the weakest assistant turn on the same 0-4
  scale.
- usable_turn_bucket estimates the fraction of assistant turns that are suitable
  to continue into detailed turn-level judging:
  0 = under 25%; 1 = 25-49%; 2 = 50-74%; 3 = 75-89%; 4 = at least 90%.
- strongest_turns and weakest_turns are routing hints, not a substitute for
  detailed turn-level judging. Return at most three of each.

NSFW

Classify assistant text only; never use user text, character cards, persona, or
memory to raise the level:
0 no sexual content;
1 romantic/suggestive tension or mild innuendo;
2 erotic content, nudity, arousal, or non-graphic sexual touching;
3 explicit anatomy, direct genital stimulation, graphic sexual acts,
penetration, or fluids;
4 sustained highly graphic, extreme, fetish, or taboo sexual acts.
Report both the maximum assistant level in A1...An and the endpoint An level.
For endpoint_level and endpoint_reason, ignore A1...A(n-1) completely and inspect
only An. For max_level, report the A-turn IDs that actually establish the maximum.
When max_level=0, return max_turns=[] and prevalence=none. When An itself reaches
the positive maximum level, max_turns must include An; otherwise it must not.
Explicitness is neither a quality bonus nor a penalty.

CONFIDENCE

Use this exact integer scale: 0 guess, 1 low, 2 moderate, 3 high, 4 very high.
Confidence is not a quality score. Never use 5 or a decimal.

Return only one JSON object conforming to the supplied schema. Do not output an
overall quality score or keep/drop verdict; those are calculated offline.
""".strip()


def evidence_span_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["turn_id", "quote"],
        "properties": {
            "turn_id": {
                "type": "string",
                "pattern": "^A[1-9][0-9]*$",
            },
            "quote": {
                "type": "string",
                "minLength": 1,
                "maxLength": 120,
            },
        },
    }


def polarity_evidence_schema(*, positive: bool) -> dict[str, Any]:
    levels = (
        ["none", "minor", "sustained", "exceptional"]
        if positive
        else ["none", "minor", "major", "critical"]
    )
    adjustments = [0, 1, 2] if positive else [-2, -1, 0]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["level", "adjustment", "spans", "reason"],
        "properties": {
            "level": {"type": "string", "enum": levels},
            "adjustment": {"type": "integer", "enum": adjustments},
            "spans": {
                "type": "array",
                "maxItems": 3,
                "items": evidence_span_schema(),
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 180,
            },
        },
    }


def dimension_schema() -> dict[str, Any]:
    nullable_score = {
        "anyOf": [
            {"type": "integer", "enum": [0, 1, 2, 3, 4]},
            {"type": "null"},
        ]
    }
    nullable_baseline = {
        "anyOf": [
            {"type": "integer", "enum": [2]},
            {"type": "null"},
        ]
    }
    nullable_adjustment = {
        "anyOf": [
            {"type": "integer", "enum": [-2, -1, 0, 1, 2]},
            {"type": "null"},
        ]
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "applicable",
            "baseline_score",
            "negative_evidence",
            "positive_evidence",
            "net_adjustment",
            "model_score",
            "score_reason",
        ],
        "properties": {
            "applicable": {"type": "boolean"},
            "baseline_score": nullable_baseline,
            "negative_evidence": polarity_evidence_schema(positive=False),
            "positive_evidence": polarity_evidence_schema(positive=True),
            "net_adjustment": nullable_adjustment,
            "model_score": nullable_score,
            "score_reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
        },
    }


def phase_evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["turn_id", "observation"],
        "properties": {
            "turn_id": {"type": "string", "pattern": "^A[1-9][0-9]*$"},
            "observation": {
                "type": "string",
                "minLength": 1,
                "maxLength": 180,
            },
        },
    }


def score_schema() -> dict[str, Any]:
    turn_id = {"type": "string", "pattern": "^A[1-9][0-9]*$"}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "phase_evidence",
            "dimensions",
            "trajectory_audit",
            "assistant_nsfw",
            "critical_flags",
            "confidence",
            "overall_diagnosis",
        ],
        "properties": {
            "phase_evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": ["early", "middle", "late"],
                "properties": {
                    "early": phase_evidence_schema(),
                    "middle": phase_evidence_schema(),
                    "late": phase_evidence_schema(),
                },
            },
            "dimensions": {
                "type": "object",
                "additionalProperties": False,
                "required": CORE_DIMENSIONS,
                "properties": {
                    name: dimension_schema() for name in CORE_DIMENSIONS
                },
            },
            "trajectory_audit": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "quality_floor",
                    "usable_turn_bucket",
                    "strongest_turns",
                    "weakest_turns",
                    "reason",
                ],
                "properties": {
                    "quality_floor": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3, 4],
                    },
                    "usable_turn_bucket": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3, 4],
                    },
                    "strongest_turns": {
                        "type": "array",
                        "maxItems": 3,
                        "items": turn_id,
                    },
                    "weakest_turns": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "turn_id",
                                "severity",
                                "issue_type",
                                "reason",
                            ],
                            "properties": {
                                "turn_id": turn_id,
                                "severity": {
                                    "type": "string",
                                    "enum": ["minor", "major", "critical"],
                                },
                                "issue_type": {
                                    "type": "string",
                                    "enum": [
                                        "character_break",
                                        "context_break",
                                        "format_break",
                                        "ignored_user",
                                        "stalled_progression",
                                        "writing_failure",
                                        "repetition_loop",
                                        "meta_or_ai_break",
                                        "other",
                                    ],
                                },
                                "reason": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 180,
                                },
                            },
                        },
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 220,
                    },
                },
            },
            "assistant_nsfw": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "max_level",
                    "max_turns",
                    "max_reason",
                    "endpoint_level",
                    "endpoint_reason",
                    "prevalence",
                ],
                "properties": {
                    "max_level": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3, 4],
                    },
                    "max_turns": {
                        "type": "array",
                        "maxItems": 3,
                        "items": turn_id,
                    },
                    "max_reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 180,
                    },
                    "endpoint_level": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3, 4],
                    },
                    "endpoint_reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 180,
                    },
                    "prevalence": {
                        "type": "string",
                        "enum": [
                            "none",
                            "isolated",
                            "occasional",
                            "frequent",
                            "dominant",
                        ],
                    },
                },
            },
            "critical_flags": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "severity", "turn_ids"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "ai_identity_break",
                                "out_of_character",
                                "context_break",
                                "repetition_loop",
                                "format_or_language_collapse",
                                "meta_refusal",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["major", "critical"],
                        },
                        "turn_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": turn_id,
                        },
                    },
                },
            },
            "confidence": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4],
            },
            "overall_diagnosis": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
        },
    }


def _phase_bounds(turn_count: int) -> dict[str, tuple[int, int]]:
    early_end = max(1, (turn_count + 2) // 3)
    middle_end = max(early_end + 1, (2 * turn_count + 2) // 3)
    middle_end = min(middle_end, turn_count - 1)
    return {
        "early": (1, early_end),
        "middle": (early_end + 1, middle_end),
        "late": (middle_end + 1, turn_count),
    }


def _history_markdown(
    prefix: list[dict[str, Any]], endpoint_turn: int
) -> tuple[str, str]:
    current_user = ""
    sections = []
    for item in prefix:
        role = str(item.get("role") or "")
        turn = int(item.get("turn") or 0)
        content = item.get("content")
        if role == "user" and turn == endpoint_turn:
            current_user = base_v6.clean(content)
            continue
        if role not in {"user", "assistant"}:
            continue
        label = f"U{turn}" if role == "user" else f"A{turn}"
        sections.append(f"### {label}\n\n{base_v6.fenced(content)}")
    return (
        "\n\n".join(sections) if sections else "_No earlier visible turns._",
        current_user,
    )


def build_markdown(
    row: dict[str, Any],
    retention: dict[str, dict[str, Any]],
    characters: dict[str, dict[str, Any]],
    localized: dict[tuple[str, str], dict[str, Any]],
    personas: dict[str, dict[str, Any]],
) -> str:
    dialogue_id = row["judge_key"].split(":", 1)[0]
    meta = retention[dialogue_id]
    character_id = str(meta["character_id"])
    character = characters.get(character_id, {})
    language = str(row.get("language") or meta.get("exact_language") or "")
    local = localized.get((character_id, language))
    raw_greeting = base_v6.clean((local or {}).get("greeting")) or base_v6.clean(
        character.get("greeting")
    )
    greeting = base_v6.strip_greeting_image_links(raw_greeting)
    definition = base_v6.definition_text(character, local)
    requirements = base_v6.format_requirements(character, local)
    persona = base_v6.persona_text(personas.get(str(meta["conversation_id"])))

    causal = row["causal_material"]
    target = causal["target_assistant"]
    endpoint_turn = int(target["turn"])
    history, current_user = _history_markdown(
        causal.get("conversation_prefix") or [], endpoint_turn
    )
    memory = causal.get("memory_before_target")
    if isinstance(memory, dict):
        memory = (
            memory.get("summary_text")
            or memory.get("summary")
            or json.dumps(memory, ensure_ascii=False, default=str)
        )
    bounds = _phase_bounds(endpoint_turn)
    phase_text = "; ".join(
        f"{name}: A{start}-A{end}"
        for name, (start, end) in bounds.items()
    )
    return f"""# Roleplay Dialogue Trajectory Evaluation

## Character

### Character Name

{base_v6.fenced((local or {}).get("name") or character.get("name"))}

### Character Definition

{base_v6.fenced(definition)}

### Explicit Roleplay and Format Requirements

{base_v6.fenced(requirements)}

### Canonical Greeting

{base_v6.fenced(greeting)}

## User Persona

{base_v6.fenced(persona)}

## Memory Before This Replay Segment

{base_v6.fenced(memory)}

## Replay Dialogue Before Endpoint

{history}

## Endpoint Turn

### U{endpoint_turn}

{base_v6.fenced(current_user)}

### A{endpoint_turn}

{base_v6.fenced(target.get("content"))}

## Evaluation Range

- Evaluate assistant trajectory A1-A{endpoint_turn}.
- The endpoint A{endpoint_turn} has no extra weight.
- Required evidence phases: {phase_text}.
"""


def make_request(
    markdown: str, schema: dict[str, Any], seed: int
) -> dict[str, Any]:
    return {
        "model": REQUEST_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lusy_dialogue_trajectory_judge_v1",
                "strict": True,
                "schema": schema,
            },
        },
        "temperature": REQUEST_TEMPERATURE,
        "top_k": 20,
        "top_p": 0.9,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
        "max_tokens": REQUEST_MAX_TOKENS,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": REQUEST_THINKING},
    }


def make_repair_request(
    markdown: str,
    schema: dict[str, Any],
    seed: int,
    bad_content: str,
    validation_error: str,
    endpoint_turn: int,
) -> dict[str, Any]:
    request = make_request(markdown, schema, seed)
    request["messages"].extend(
        [
            {
                "role": "assistant",
                "content": bad_content,
            },
            {
                "role": "user",
                "content": f"""
Your JSON failed deterministic local validation:

{validation_error}

Return one complete replacement JSON object. Preserve sound judgements, but fix
the stated problem. Use only real assistant turns A1-A{endpoint_turn}. Every
evidence quote must contain exact copied text from its cited assistant turn; ...
may only separate exact fragments and must never replace paraphrased text. Do
not invent evidence merely to keep a high score: if the required evidence is
absent, lower the evidence level, adjustment, net_adjustment, and model_score.
Recheck all baseline-2 arithmetic, phase bounds, applicable/null fields, and
assistant-only NSFW consistency. Return JSON only.
""".strip(),
            },
        ]
    )
    return request


def _turn_number(value: Any) -> int | None:
    match = re.fullmatch(r"A([1-9][0-9]*)", str(value or ""))
    return int(match.group(1)) if match else None


def _normalized_evidence_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFC", str(value or "")),
    ).strip()


def _evidence_quote_matches(source: Any, quote: Any) -> bool:
    source_text = _normalized_evidence_text(source)
    quote_text = _normalized_evidence_text(quote)
    if not source_text or not quote_text:
        return False
    fragments = [
        fragment.strip()
        for fragment in re.split(r"(?:\.{3}|…)+", quote_text)
        if fragment.strip()
    ]
    if not fragments:
        return False
    compact_source = re.sub(r"\s+", "", source_text)
    cursor = 0
    meaningful = False
    for fragment in fragments:
        compact_fragment = re.sub(r"\s+", "", fragment)
        if len(compact_fragment) >= 4:
            meaningful = True
        position = compact_source.find(compact_fragment, cursor)
        if position < 0:
            return False
        cursor = position + len(compact_fragment)
    return meaningful


def _validate_polarity_evidence(
    *,
    dimension: str,
    polarity: str,
    item: Any,
    endpoint_turn: int,
    bounds: dict[str, tuple[int, int]],
    assistant_texts: dict[int, str] | None,
) -> tuple[str, int, list[int]]:
    if not isinstance(item, dict):
        raise ValueError(f"{dimension}.{polarity}_evidence invalid")
    positive = polarity == "positive"
    level_map = (
        {"none": 0, "minor": 0, "sustained": 1, "exceptional": 2}
        if positive
        else {"none": 0, "minor": 0, "major": -1, "critical": -2}
    )
    level = item.get("level")
    adjustment = item.get("adjustment")
    if level not in level_map:
        raise ValueError(f"{dimension}.{polarity}_evidence.level invalid")
    if (
        isinstance(adjustment, bool)
        or not isinstance(adjustment, int)
        or adjustment != level_map[level]
    ):
        raise ValueError(
            f"{dimension}.{polarity}_evidence adjustment/level mismatch"
        )
    if not isinstance(item.get("reason"), str) or not item["reason"].strip():
        raise ValueError(f"{dimension}.{polarity}_evidence.reason empty")
    spans = item.get("spans")
    if not isinstance(spans, list):
        raise ValueError(f"{dimension}.{polarity}_evidence.spans invalid")
    turn_numbers: list[int] = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError(f"{dimension}.{polarity}_evidence span invalid")
        turn = _turn_number(span.get("turn_id"))
        quote = _normalized_evidence_text(span.get("quote"))
        if not quote:
            raise ValueError(f"{dimension}.{polarity}_evidence quote empty")
        if turn is not None and 1 <= turn <= endpoint_turn:
            turn_numbers.append(turn)
    ordered = sorted(turn_numbers)
    return str(level), int(adjustment), ordered


def validate_judgement(
    value: dict[str, Any],
    endpoint_turn: int,
    assistant_texts: dict[int, str] | None = None,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("judgement is not an object")
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(
        CORE_DIMENSIONS
    ):
        raise ValueError("dimension keys mismatch")
    bounds = _phase_bounds(endpoint_turn)
    for name in CORE_DIMENSIONS:
        item = dimensions[name]
        if not isinstance(item, dict):
            raise ValueError(f"{name} is not an object")
        applicable = item.get("applicable")
        if not isinstance(applicable, bool):
            raise ValueError(f"{name}.applicable invalid")
        if not isinstance(item.get("score_reason"), str) or not item[
            "score_reason"
        ].strip():
            raise ValueError(f"{name}.score_reason empty")
        positive_level, positive_adjustment, positive_turns = (
            _validate_polarity_evidence(
                dimension=name,
                polarity="positive",
                item=item.get("positive_evidence"),
                endpoint_turn=endpoint_turn,
                bounds=bounds,
                assistant_texts=assistant_texts,
            )
        )
        negative_level, negative_adjustment, negative_turns = (
            _validate_polarity_evidence(
                dimension=name,
                polarity="negative",
                item=item.get("negative_evidence"),
                endpoint_turn=endpoint_turn,
                bounds=bounds,
                assistant_texts=assistant_texts,
            )
        )
        baseline = item.get("baseline_score")
        net_adjustment = item.get("net_adjustment")
        model_score = item.get("model_score")
        if not applicable:
            continue
        if isinstance(baseline, bool) or baseline != 2:
            raise ValueError(f"{name}.baseline_score must equal 2")
        if (
            isinstance(net_adjustment, bool)
            or not isinstance(net_adjustment, int)
        ):
            raise ValueError(f"{name}.net_adjustment invalid")
        if (
            isinstance(model_score, bool)
            or not isinstance(model_score, int)
            or not 0 <= model_score <= 4
        ):
            raise ValueError(f"{name}.model_score invalid")
        expected_net = positive_adjustment + negative_adjustment
        expected_score = 2 + expected_net
        if net_adjustment != expected_net:
            raise ValueError(f"{name}.net_adjustment arithmetic mismatch")
        if model_score != expected_score:
            raise ValueError(f"{name}.model_score arithmetic mismatch")
        if negative_level == "critical" and (
            positive_adjustment != 0 or model_score != 0
        ):
            raise ValueError(f"{name}.critical evidence must force score 0")

    evidence = value.get("phase_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("phase_evidence missing")
    for phase, (start, end) in bounds.items():
        item = evidence.get(phase)
        if not isinstance(item, dict):
            raise ValueError(f"phase_evidence.{phase} missing")
        turn = _turn_number(item.get("turn_id"))
        if turn is None:
            raise ValueError(f"phase_evidence.{phase}.turn_id invalid")
        if not isinstance(item.get("observation"), str) or not item[
            "observation"
        ].strip():
            raise ValueError(f"phase_evidence.{phase}.observation empty")

    audit = value.get("trajectory_audit")
    if not isinstance(audit, dict):
        raise ValueError("trajectory_audit missing")
    for name in ["quality_floor", "usable_turn_bucket"]:
        score = audit.get(name)
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
            raise ValueError(f"trajectory_audit.{name} invalid")
    for turn_id in audit.get("strongest_turns") or []:
        turn = _turn_number(turn_id)
        if turn is None or not 1 <= turn <= endpoint_turn:
            raise ValueError("strongest_turns contains invalid turn")
    for item in audit.get("weakest_turns") or []:
        turn = _turn_number((item or {}).get("turn_id"))
        if turn is None or not 1 <= turn <= endpoint_turn:
            raise ValueError("weakest_turns contains invalid turn")

    for flag in value.get("critical_flags") or []:
        for turn_id in (flag or {}).get("turn_ids") or []:
            turn = _turn_number(turn_id)
            if turn is None or not 1 <= turn <= endpoint_turn:
                raise ValueError("critical_flags contains invalid turn")

    nsfw = value.get("assistant_nsfw")
    if not isinstance(nsfw, dict):
        raise ValueError("assistant_nsfw missing")
    for name in ["max_level", "endpoint_level"]:
        score = nsfw.get(name)
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
            raise ValueError(f"assistant_nsfw.{name} invalid")
    for name in ["max_reason", "endpoint_reason"]:
        text = nsfw.get(name)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"assistant_nsfw.{name} empty")
    prevalence = nsfw.get("prevalence")
    if prevalence not in {
        "none",
        "isolated",
        "occasional",
        "frequent",
        "dominant",
    }:
        raise ValueError("assistant_nsfw.prevalence invalid")
    max_turns = nsfw.get("max_turns")
    if not isinstance(max_turns, list):
        raise ValueError("assistant_nsfw.max_turns invalid")
    max_numbers: list[int] = []
    for turn_id in max_turns:
        turn = _turn_number(turn_id)
        if turn is not None and 1 <= turn <= endpoint_turn:
            max_numbers.append(turn)
    max_level = int(nsfw["max_level"])
    endpoint_level = int(nsfw["endpoint_level"])
    if endpoint_level > max_level:
        raise ValueError("endpoint NSFW exceeds dialogue maximum")

    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not 0 <= confidence <= 4
    ):
        raise ValueError("confidence invalid")


def audit_score_support(
    parsed: dict[str, Any] | None,
    _target_content: str,
    assistant_texts: dict[int, str] | None = None,
    endpoint_turn: int | None = None,
) -> tuple[
    dict[str, int | None],
    dict[str, int | None],
    dict[str, int | None],
    dict[str, Any],
    list[str],
]:
    if not isinstance(parsed, dict):
        return {}, {}, {}, {}, []
    model_scores: dict[str, int | None] = {}
    arithmetic_scores: dict[str, int | None] = {}
    audited_scores: dict[str, int | None] = {}
    reason_audit: dict[str, Any] = {}
    flags: list[str] = []
    bounds = _phase_bounds(endpoint_turn) if endpoint_turn else {}
    for name in CORE_DIMENSIONS:
        item = (parsed.get("dimensions") or {}).get(name) or {}
        score = item.get("model_score")
        score = score if isinstance(score, int) and not isinstance(score, bool) else None
        model_scores[name] = score
        baseline = item.get("baseline_score")
        positive = item.get("positive_evidence") or {}
        negative = item.get("negative_evidence") or {}
        positive_adjustment = positive.get("adjustment")
        negative_adjustment = negative.get("adjustment")
        applicable = bool(item.get("applicable"))
        arithmetic = (
            int(baseline) + int(positive_adjustment) + int(negative_adjustment)
            if applicable
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in [
                    baseline,
                    positive_adjustment,
                    negative_adjustment,
                ]
            )
            else None
        )
        arithmetic_scores[name] = arithmetic
        positive_spans = positive.get("spans") or []
        negative_spans = negative.get("spans") or []

        def verified_turns(spans: list[Any], polarity: str) -> list[int]:
            verified: list[int] = []
            seen: set[int] = set()
            for span in spans:
                turn = _turn_number((span or {}).get("turn_id"))
                quote = (span or {}).get("quote")
                if (
                    turn is None
                    or endpoint_turn is None
                    or not 1 <= turn <= endpoint_turn
                ):
                    flags.append(f"{name}:{polarity}_turn_out_of_range")
                    continue
                if turn in seen:
                    flags.append(
                        f"{name}:{polarity}_duplicate_turn:A{turn}"
                    )
                    continue
                seen.add(turn)
                if assistant_texts is None or _evidence_quote_matches(
                    assistant_texts.get(turn),
                    quote,
                ):
                    verified.append(turn)
                else:
                    flags.append(
                        f"{name}:{polarity}_quote_mismatch:A{turn}"
                    )
            return sorted(set(verified))

        positive_turns = verified_turns(positive_spans, "positive")
        negative_turns = verified_turns(negative_spans, "negative")
        positive_level = positive.get("level")
        negative_level = negative.get("level")
        if positive_level == "none" and positive_spans:
            flags.append(f"{name}:positive_none_has_spans")
        if positive_level != "none" and not positive_spans:
            flags.append(f"{name}:positive_level_lacks_spans")
        if negative_level == "none" and negative_spans:
            flags.append(f"{name}:negative_none_has_spans")
        if negative_level != "none" and not negative_spans:
            flags.append(f"{name}:negative_level_lacks_spans")
        audited_positive = 0
        if positive_level == "sustained":
            if (
                len(positive_turns) >= 2
                and positive_turns[-1] - positive_turns[0] >= 2
            ):
                audited_positive = 1
            else:
                flags.append(f"{name}:sustained_evidence_gate_failed")
        elif positive_level == "exceptional":
            covered = {
                phase
                for phase, (start, end) in bounds.items()
                if any(start <= turn <= end for turn in positive_turns)
            }
            if (
                len(positive_turns) >= 3
                and bounds
                and covered == set(bounds)
            ):
                audited_positive = 2
            elif (
                len(positive_turns) >= 2
                and positive_turns[-1] - positive_turns[0] >= 2
            ):
                audited_positive = 1
                flags.append(f"{name}:exceptional_downgraded_to_sustained")
            else:
                flags.append(f"{name}:exceptional_evidence_gate_failed")

        audited_negative = 0
        if negative_level == "major":
            if negative_turns:
                audited_negative = -1
            else:
                flags.append(f"{name}:major_evidence_gate_failed")
        elif negative_level == "critical":
            if negative_turns:
                audited_negative = -2
                audited_positive = 0
            else:
                flags.append(f"{name}:critical_evidence_gate_failed")
        audited = (
            2 + audited_positive + audited_negative
            if applicable
            else None
        )
        audited_scores[name] = audited
        reason = base_v6.clean(item.get("score_reason"))
        cited_turns = sorted(
            {
                turn
                for evidence in [negative, positive]
                for span in evidence.get("spans") or []
                for turn in [_turn_number((span or {}).get("turn_id"))]
                if turn is not None
            }
        )
        reason_audit[name] = {
            "reason": reason,
            "reason_characters": len(reason),
            "cited_turns": cited_turns,
            "baseline_score": baseline,
            "positive_level": positive.get("level"),
            "positive_adjustment": positive_adjustment,
            "positive_verified_turns": positive_turns,
            "audited_positive_adjustment": audited_positive,
            "negative_level": negative.get("level"),
            "negative_adjustment": negative_adjustment,
            "negative_verified_turns": negative_turns,
            "audited_negative_adjustment": audited_negative,
            "net_adjustment": item.get("net_adjustment"),
            "arithmetic_score": arithmetic,
            "audited_score": audited,
            "score_consistent": score == arithmetic,
            "evidence_gate_passed": audited == arithmetic,
            "quote_match_passed": (
                len(positive_turns) == len(positive_spans)
                and len(negative_turns) == len(negative_spans)
            ),
        }
        if score != arithmetic:
            flags.append(f"{name}:score_arithmetic_mismatch")
        if not applicable:
            if name != "greeting_style_adherence":
                flags.append(f"{name}:unexpected_not_applicable")
            if any(
                value is not None
                for value in [
                    baseline,
                    item.get("net_adjustment"),
                    score,
                ]
            ):
                flags.append(f"{name}:not_applicable_has_score_fields")
            if (
                positive.get("level") != "none"
                or negative.get("level") != "none"
                or positive_spans
                or negative_spans
            ):
                flags.append(f"{name}:not_applicable_has_evidence")
    if endpoint_turn is not None:
        for phase, (start, end) in bounds.items():
            phase_turn = _turn_number(
                ((parsed.get("phase_evidence") or {}).get(phase) or {}).get(
                    "turn_id"
                )
            )
            if phase_turn is None or not start <= phase_turn <= end:
                flags.append(f"phase_evidence:{phase}:outside_phase")
        nsfw = parsed.get("assistant_nsfw") or {}
        max_level = nsfw.get("max_level")
        endpoint_level = nsfw.get("endpoint_level")
        prevalence = nsfw.get("prevalence")
        max_turns = [
            _turn_number(value) for value in nsfw.get("max_turns") or []
        ]
        valid_max_turns = [
            turn
            for turn in max_turns
            if turn is not None and 1 <= turn <= endpoint_turn
        ]
        if len(valid_max_turns) != len(max_turns):
            flags.append("assistant_nsfw:max_turn_out_of_range")
        if len(valid_max_turns) != len(set(valid_max_turns)):
            flags.append("assistant_nsfw:duplicate_max_turn")
        if max_level == 0 and valid_max_turns:
            flags.append("assistant_nsfw:level0_has_max_turns")
        if isinstance(max_level, int) and max_level > 0 and not valid_max_turns:
            flags.append("assistant_nsfw:positive_level_lacks_max_turn")
        if (max_level == 0) != (prevalence == "none"):
            flags.append("assistant_nsfw:prevalence_level_mismatch")
        endpoint_is_max = (
            isinstance(max_level, int)
            and max_level > 0
            and endpoint_level == max_level
        )
        endpoint_cited = endpoint_turn in valid_max_turns
        if endpoint_is_max != endpoint_cited:
            flags.append("assistant_nsfw:endpoint_max_turn_mismatch")
    return (
        model_scores,
        arithmetic_scores,
        audited_scores,
        reason_audit,
        flags,
    )
