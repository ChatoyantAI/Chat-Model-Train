#!/usr/bin/env python3
"""Run a targeted DeepInfra Batch prompt-only JSON smoke test for turn Judge v4.6.

The script reads frozen UNION/Elite-A artifacts only. It writes exclusively to
its own versioned output directory and never queries a database or online Lusy
API. Two independent prompt contracts are used for canonical-greeting present
and absent samples.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

import httpx
import pyarrow.compute as pc
import pyarrow.parquet as pq


TRAINING_ROOT = pathlib.Path(
    os.environ.get("LUSY_TRAINING_ROOT", "/home/gyr/lusychat-training")
)
RUN_ROOT = (
    TRAINING_ROOT
    / "judge-deepinfra-glm47-turn-v46-promptonly-softjson-smoke-20260728"
)
ARTIFACTS = RUN_ROOT / "artifacts"

ELITE_A_PATH = (
    TRAINING_ROOT
    / "judge-dialogue-selection-v1/02_tiers/elite_A.parquet"
)
MATERIAL_PATH = (
    TRAINING_ROOT
    / "judge-dialogue-union-50pct-v1/01_input/dialogue_material.parquet"
)
TURN_MANIFEST_PATH = (
    TRAINING_ROOT
    / "judge-dialogue-union-50pct-v1/01_input/turn_manifest.parquet"
)
PILOT_EXPECTATIONS_PATH = (
    TRAINING_ROOT
    / "judge-deepinfra-glm47-turn-pilot-v2/artifacts"
    / "deepinfra-direct-batch-pilot-v2-detailed"
    / "batch_main_json_object_1000.expectations.json"
)

TOKEN = os.environ.get("DEEPINFRA_TOKEN", "")
API_BASE = os.environ.get(
    "DEEPINFRA_OPENAI_BASE_URL",
    "https://api.deepinfra.com/v1/openai",
).rstrip("/")
MODEL = os.environ.get(
    "DEEPINFRA_JUDGE_MODEL",
    "zai-org/GLM-4.7-Flash",
)

PREVIOUS_SMOKE_ARTIFACTS = (
    TRAINING_ROOT
    / "judge-deepinfra-glm47-turn-v4-smoke-20260728/artifacts"
)

SEED = "lusy-turn-judge-v46-promptonly-softjson-smoke-20260728-v1"
SEED_INT = 20260728
SAMPLES_PER_VARIANT = 8
MAX_OUTPUT_TOKENS = 3072
MAX_BATCH_FILE_BYTES = 150 * 1024 * 1024
REMOTE_FILE_TTL_SECONDS = 604800
TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}

SCORED_DIMENSIONS = [
    "character_fidelity",
    "context_coherence",
    "current_turn_fit",
    "relationship_continuity",
    "scene_progression",
    "language_roleplay_quality",
    "intimacy_quality",
]
CORE_DIMENSIONS = [
    "character_fidelity",
    "context_coherence",
    "current_turn_fit",
    "relationship_continuity",
    "scene_progression",
    "language_roleplay_quality",
]
CRITICAL_FLAG_KEYS = [
    "ai_identity_break",
    "character_identity_break",
    "context_break",
    "relationship_reset",
    "wrong_language",
    "format_collapse",
    "meta_refusal",
]
GREETING_FEATURES = [
    "pov",
    "user_address",
    "quote_action_format",
    "speaker_labels",
    "distinctive_structure",
]
LANGUAGE_TARGETS = {
    "en": 6,
    "es": 4,
    "pt": 3,
    "fr": 3,
    "zh": 3,
    "de": 3,
    "ja": 2,
}
TOPIC_TARGETS = {
    "explicit": 8,
    "romance_suggestive": 8,
    "other_rp": 8,
}
TURN_BUCKET_TARGETS = {
    "05_08": 6,
    "09_16": 6,
    "17_32": 6,
    "33_plus": 6,
}

RETEST_CUSTOM_IDS = {
    "greeting_present": [
        "v4s_7e2720b3e70e7641d82d16cfc923",
        "v4s_80a6ed782097faabfed37c6a9386",
        "v4s_e383fc4c71690af2b3aa4db5f3b9",
        "v4s_8f21916257026c52ba6296e5c1b1",
        "v4s_c065cf717b968627d520cb38496c",
        "v4s_b28cd68b11c525236acdaab842aa",
        "v4s_3e97334be0a4cdd764bbd497b40a",
        "v4s_e12799dc7cd8e6a0db939558cb59",
    ],
    "greeting_absent": [
        "v4s_ad337f2ab7a6e3d85a0b59160282",
        "v4s_5dbe9d0a5ac761c46b8316035abf",
        "v4s_7bed15e8c8bfedb4cf37fa134385",
        "v4s_c58ad69e2d6816d7df7b4546ad94",
        "v4s_29dac06aa63d507c9744c1b13edd",
        "v4s_99e142631689b3bd18d1f443f01c",
        "v4s_a63242861ac779044266222c7588",
        "v4s_437335e12e52ebbd8d34f5e77bdf",
    ],
}


COMMON_PROMPT = """
You are a strict multilingual evaluator of exactly one target assistant turn
from a fictional roleplay conversation. Judge only the final target A-turn.
Earlier assistant turns are causal context, not evaluation targets. No later
messages are available.

Never infer or discuss teacher, model, provider, likes, reactions,
subscriptions, edit/regenerate history, MDS rank, or dialogue tier. Explicit or
taboo sexual content is neither a quality bonus nor a penalty. Return no chain
of thought.

QUALITY SCORES

- 0: critical and unusable failure.
- 1: clear major defect.
- 2: ordinary competent and usable baseline.
- 3: clearly above ordinary, supported by one material target-specific strength.
- 4: rare exceptional quality, supported by two independent material strengths.
- -1: objectively not applicable; allowed only for length_following and
  intimacy_quality.

Start every applicable dimension at 2. Absence of a defect is not evidence for
3 or 4. Fluency, generic formatting, emotional intensity, explicitness,
ornamental prose, or length alone are not strengths.

Every dimension reason must be one compact auditable sentence:
- score 0 or 1: begin with "D:" and name the concrete defect.
- score 2: begin with "Baseline:" and name the relevant ordinary behavior.
- score 3: begin with "S1:" and name one material target-specific strength.
- score 4: use "S1: ...; S2: ..." for two independent material strengths.
- score -1: begin with "N/A:" and state why the dimension is inapplicable.
Do not award 3 or 4 when the required strength evidence is absent.
Keep every reason, pattern, and overall summary at 180 characters or fewer.

DIMENSIONS

1. character_fidelity: identity, voice, temperament, motives, relationship role,
world facts, and role boundaries against the supplied character material.
2. persona_context_consistency: respect User Persona, Memory, and established
causal facts. Visible conversation context always makes this applicable.
3. current_turn_fit: answer the current user's content, intent, emotion,
correction, action, and constraints without generic diversion.
4. relationship_continuity: preserve roles, names, intimacy, conflict, trust,
power dynamics, and emotional state from visible history.
5. scene_progression: meaningful causal movement and useful continuation without
unsupported jumps, reset, or stalling.
6. language_roleplay_quality: natural, clear, specific, immersive writing in the
language actually used.
7. length_following: obey the offline applicability gate in the packet. When
the gate is false, score exactly -1. When true, compare the target with the
quoted requirement; ordinary compliance is 2, not automatically 4.
8. intimacy_quality: -1 unless the target itself meaningfully participates in
romantic or sexual intimacy. Conflict, nudity, arousal words, or explicitness
alone do not make intimacy high quality. Explicitness is never a strength.

SEMANTIC REPETITION

Do not score character-string similarity. Classify only semantic repetition:
- none: no material semantic repetition.
- minor: a small redundant restatement that remains usable.
- major: substantial rehash, repeated action, or stalling.
- critical: near-total semantic loop or repeated scene with little new content.
When severity is not none, cite prior assistant turn numbers in history_turns.
A deterministic detector will later combine this with text-similarity signals.

CRITICAL FLAGS

Return every fixed boolean. True requires a concrete, target-visible failure.
An in-character AI/robot is not an identity break; provider/policy/training
limitations are. Keep all reasons compact and auditable.

NSFW LEVEL

Classify only the target assistant reply:
0 none; 1 suggestive or romantic tension; 2 erotic content, nudity, arousal, or
non-graphic sexual touching; 3 explicit anatomy, genital stimulation, graphic
acts, penetration, or fluids; 4 sustained highly graphic, extreme, fetish, or
taboo sexual acts.

OUTPUT CONTRACT

Return a JSON object. dimensions must contain exactly these eight keys:
character_fidelity, persona_context_consistency, current_turn_fit,
relationship_continuity, scene_progression, language_roleplay_quality,
length_following, intimacy_quality. Each value has exactly score and reason.
semantic_repetition has exactly severity, reason, history_turns.
critical_flags has exactly all five fixed boolean keys. Also return
critical_reason, nsfw_level, nsfw_reason, overall_reason, and confidence.
""".strip()


PRESENT_PROMPT = (
    COMMON_PROMPT
    + """

CANONICAL GREETING FORMAT

The target is deliberately a later conversation turn. Never judge whether it
is an opening greeting. Judge only whether its reusable writing-format
signature matches the supplied Canonical Greeting. A later turn can score 4.
Never write "not a greeting" or use that as evidence.

Return exactly one evidence item for each of these five features:
pov, user_address, quote_action_format, speaker_labels,
distinctive_structure. Compare the target only with the Canonical Greeting,
never with intermediate assistant turns. Ignore images.

Each verdict is match, partial, mismatch, or not_observable. The offline
normalizer derives the 0-4 greeting score from these five evidence items:
- two or more mismatches -> 0
- one mismatch -> 1
- no mismatch and at least two matches -> 4
- no mismatch and one match -> 3
- otherwise -> 2

Also return greeting with exactly one key, signature_evidence. Return JSON
matching this compact contract. No extra keys and no chain of thought.
"""
).strip()


ABSENT_PROMPT = (
    COMMON_PROMPT
    + """

CANONICAL GREETING FORMAT

Canonical Greeting is absent. Do not infer a greeting format from intermediate
assistant turns and do not output greeting evidence. The offline normalizer
inserts greeting_format_adherence=-1.

Return JSON matching this compact contract. No extra keys and no chain of thought.
"""
).strip()


# v4.2 deliberately makes the numeric score the primary judgment. Reasons are
# compact audit notes, and evidence is a constraint on unusually high/low
# scores rather than a separate evidence-first scoring pipeline.
V42_DIRECT_SCORE_PROMPT = """
You are a strict multilingual evaluator of exactly one target assistant turn
from a fictional roleplay conversation. Judge only the final target A-turn.
Earlier assistant turns are causal context, not evaluation targets. No later
messages are available.

Ignore all external provenance, popularity, ranking, and product metadata; none
is provided in the evaluation packet. Explicit or taboo sexual content is
neither a quality bonus nor a penalty. Return no chain of thought. Return one
JSON object and nothing else.

PRIMARY SCORE RUBRIC

For each applicable quality dimension, directly assign:
- 0 = critical, unusable failure in this dimension.
- 1 = clear major defect.
- 2 = ordinary competent and usable baseline.
- 3 = clearly above ordinary because of one material target-specific strength.
- 4 = rare exceptional quality because of two independent material strengths.

Start at 2, then move only for concrete target-visible reasons. Absence of a
defect is not evidence for 3 or 4. Fluency, generic formatting, emotional
intensity, explicitness, ornamental prose, or length alone are not strengths.
The score, reason, and evidence must agree. If the reason describes a major
contradiction or failure, the score cannot be 2, 3, or 4.

Each dimension object has exactly score, reason, evidence. reason is one compact
auditable sentence of at most 180 characters. evidence is an array of at most
two objects. Every evidence object has exactly:
- kind: strength, minor_defect, major_defect, or critical_defect.
- target_quote: a short exact quote copied from the target A-turn.
- reference_source: character_definition, explicit_requirement, user_persona,
  memory, current_user, prior_user, prior_assistant, or null.
- reference_id: a short section/turn id such as U7 or A5, or null.
- reference_quote: a short exact quote from that source, or null.

Evidence requirements:
- score 0: at least one critical_defect.
- score 1: at least one major_defect.
- score 2: evidence is optional and may contain only minor_defect.
- score 3: at least one strength and no major/critical defect.
- score 4: exactly two independent strengths and no major/critical defect.
Do not manufacture evidence. Quotes must be exact substrings. Use null for the
reference fields when the judgment is visible in the target alone.

DIRECT QUALITY DIMENSIONS

1. character_fidelity: identity, voice, temperament, motives, relationship role,
world facts, and role boundaries against Character Definition and explicit
character requirements.
2. context_coherence: respect User Persona, Memory, and established causal facts
without contradiction, hallucinated state, or unexplained reset.
3. current_turn_fit: answer the current user's content, intent, emotion,
correction, action, and constraints without generic diversion.
4. relationship_continuity: preserve roles, names, intimacy, conflict, trust,
power dynamics, and emotional state from visible history.
5. scene_progression: create meaningful causal movement and a useful
continuation without unsupported jumps, reset, rehash, or stalling.
6. language_roleplay_quality: natural, clear, specific, immersive writing in the
language actually used, with readable roleplay formatting.
7. intimacy_quality: output applicable=false and score=-1 unless the target
itself meaningfully participates in romantic or sexual intimacy. If applicable,
score 0-4 using the same rubric. Conflict, nudity, arousal words, or explicitness
alone do not make intimacy high quality.

Length-following is intentionally not scored by the model. It is computed
offline from explicit length constraints.

SEMANTIC REPETITION

Classify only semantic repetition, not character-string similarity:
- none: no material semantic repetition.
- minor: a small redundant restatement that remains usable.
- major: substantial rehash, repeated action, or stalling.
- critical: near-total semantic loop or repeated scene with little new content.
semantic_repetition has exactly severity, reason, history_turns, target_quote,
history_quotes. history_quotes contains objects with exactly turn and quote.
For none, return empty arrays and an empty target_quote. For any non-none level,
copy an exact target quote and at least one exact prior assistant quote; every
history turn must be earlier than the target.

CRITICAL FLAGS

critical_flags contains exactly these booleans:
ai_identity_break, character_identity_break, context_break,
relationship_reset, wrong_language, format_collapse, meta_refusal.
True requires a concrete target-visible failure. An in-character AI/robot is not
an AI identity break. critical_reason is one compact sentence; if every flag is
false, state that no critical failure is present.

NSFW LEVEL

Classify only the target assistant reply:
0 none; 1 suggestive or romantic tension; 2 erotic content, nudity, arousal, or
non-graphic sexual touching; 3 explicit anatomy, genital stimulation, graphic
acts, penetration, or fluids; 4 sustained highly graphic, extreme, fetish, or
taboo sexual acts.

EXACT OUTPUT SHAPE

Top-level keys are exactly:
dimensions, semantic_repetition, critical_flags, critical_reason, nsfw_level,
nsfw_reason, overall_reason.

dimensions has exactly:
character_fidelity, context_coherence, current_turn_fit,
relationship_continuity, scene_progression, language_roleplay_quality,
intimacy_quality.

The first six values have exactly score, reason, evidence.
intimacy_quality has exactly applicable, score, reason, evidence.
semantic_repetition has exactly severity, reason, history_turns, target_quote,
history_quotes. Each history_quotes item has exactly turn and quote.
No extra keys are allowed anywhere.
""".strip()

PRESENT_PROMPT = V42_DIRECT_SCORE_PROMPT
ABSENT_PROMPT = V42_DIRECT_SCORE_PROMPT

V43_SCORE_ONLY_PROMPT = """
You are a strict multilingual evaluator of exactly one target assistant turn
from a fictional roleplay conversation. Judge only the final target A-turn.
Earlier assistant turns are causal context, not evaluation targets. No later
messages are available.

Ignore all external provenance, popularity, ranking, and product metadata; none
is provided in the evaluation packet. Explicit or taboo sexual content is
neither a quality bonus nor a penalty. Do not output explanations, evidence,
quotes, summaries, or chain of thought. Return one compact JSON object only.

SCORING SCALE

Directly score each quality dimension:
- 0 = critical, unusable failure in this dimension.
- 1 = clear major defect.
- 2 = ordinary competent and usable baseline.
- 3 = clearly above ordinary due to one material target-specific strength.
- 4 = rare exceptional quality with two independent material strengths and no
  material defect in the dimension.

Start every dimension at 2. Move only for concrete target-visible quality.
Absence of a defect is still 2, never 3 or 4. Fluency, generic formatting,
emotional intensity, explicitness, ornamental prose, or length alone do not
justify 3 or 4. A major contradiction or failure requires 0 or 1.

Calibration prior: these turns come from a pre-screened usable corpus. Even so,
2 should remain the common score, 3 should be occasional, and 4 should be very
rare. Do not force a quota on an individual turn, but do not treat polished or
long prose as exceptional.

DIMENSIONS AND HIGH-SCORE BOUNDARIES

1. character_fidelity: identity, voice, temperament, motives, relationship role,
world facts, and role boundaries against supplied character material.
Score 3 only when a distinctive supplied trait materially shapes the target;
score 4 requires multiple subtle supplied constraints integrated accurately.

2. context_coherence: respect User Persona, Memory, and established causal facts
without contradiction, hallucinated state, or unexplained reset.
Mere non-contradiction is 2. Score 3 only when concrete prior facts materially
shape the response; score 4 requires several facts integrated causally.

3. current_turn_fit: answer the current user's content, intent, emotion,
correction, action, and constraints without generic diversion.
A direct adequate answer is 2. Score 3 requires unusually specific handling of
both explicit and implicit needs; score 4 requires rare precision and depth.

4. relationship_continuity: preserve roles, names, intimacy, conflict, trust,
power dynamics, and emotional state from visible history.
Preserving continuity is 2. Score 3 requires the established relationship state
to materially shape the response; score 4 requires multiple relationship layers.

5. scene_progression: create meaningful causal movement and a useful
continuation without unsupported jumps, reset, rehash, or stalling.
A routine reaction plus continuation is 2. Score 3 requires meaningful new
causal movement; score 4 requires coordinated external and emotional progress.

6. language_roleplay_quality: natural, clear, specific, immersive writing in the
language actually used, with readable roleplay formatting.
Ordinary polished prose is 2. Score 3 requires distinctive, controlled roleplay
craft; score 4 requires rare sustained control without excess or confusion.

INTIMACY

intimacy_quality is applicable only when the target itself meaningfully
participates in romantic or sexual intimacy. Conflict, nudity, arousal words, or
explicitness alone do not make intimacy applicable or high quality. When not
applicable return false and -1. When applicable use the same 0-4 scale.

SEMANTIC REPETITION

Classify semantic repetition, not string similarity:
- none: no material repetition.
- minor: small redundant restatement that remains usable.
- major: substantial rehash, repeated action, or stalling.
- critical: near-total semantic loop with little new content.
For non-none, return up to three earlier assistant turn numbers. For none,
history_turns must be empty.

CRITICAL FLAGS

Return all seven booleans:
ai_identity_break, character_identity_break, context_break,
relationship_reset, wrong_language, format_collapse, meta_refusal.
True requires a concrete target-visible failure. An in-character AI/robot is not
an AI identity break.

NSFW LEVEL

Classify only the target assistant reply:
0 none; 1 suggestive or romantic tension; 2 erotic content, nudity, arousal, or
non-graphic sexual touching; 3 explicit anatomy, genital stimulation, graphic
acts, penetration, or fluids; 4 sustained highly graphic, extreme, fetish, or
taboo sexual acts.

EXACT OUTPUT

Return exactly:
{
  "scores": {
    "character_fidelity": integer 0..4,
    "context_coherence": integer 0..4,
    "current_turn_fit": integer 0..4,
    "relationship_continuity": integer 0..4,
    "scene_progression": integer 0..4,
    "language_roleplay_quality": integer 0..4
  },
  "intimacy_quality": {
    "applicable": boolean,
    "score": integer -1..4
  },
  "semantic_repetition": {
    "severity": "none" | "minor" | "major" | "critical",
    "history_turns": [earlier assistant turn integers]
  },
  "critical_flags": {
    "ai_identity_break": boolean,
    "character_identity_break": boolean,
    "context_break": boolean,
    "relationship_reset": boolean,
    "wrong_language": boolean,
    "format_collapse": boolean,
    "meta_refusal": boolean
  },
  "nsfw_level": integer 0..4
}
No extra keys are allowed.
""".strip()

PRESENT_PROMPT = V43_SCORE_ONLY_PROMPT
ABSENT_PROMPT = V43_SCORE_ONLY_PROMPT

V44_POINT_REASON_PROMPT = """
You are evaluating exactly one target assistant turn from a fictional
roleplay conversation for a high-quality training corpus. Judge only the
target A-turn identified in the packet. Earlier assistant turns are causal
context and reference material, not evaluation targets. No later turns exist.

Ignore all external provenance, popularity, ranking, and product metadata;
none is provided. Explicit or taboo sexual content is neither a quality bonus
nor a penalty. Return no chain of thought. Return exactly one JSON object.

SCORING PROCEDURE

For every applicable quality dimension, follow this order. Do not choose a
score first and search for support afterward.

1. Set baseline_score=2.
2. Audit defects first. Report the single most material negative point.
3. Audit strengths second. Report the strongest positive point.
4. Map the two point levels to their exact adjustments.
5. Compute net_adjustment and model_score arithmetically.
6. score_reason must state the complete path from baseline to final score.

Negative point mapping:
- none: adjustment 0, no defect and no spans.
- minor: adjustment 0, a concrete minor issue; cite one target span.
- major: adjustment -1, a clear material defect; cite target evidence.
- critical: adjustment -2, an unusable or identity-breaking failure; cite
  target evidence. Critical overrides positives: positive level must be none,
  positive adjustment 0, net_adjustment -2, and model_score 0.

Positive point mapping:
- none: adjustment 0, no above-baseline strength and no spans.
- minor: adjustment 0, an ordinary or isolated strength; at most one span.
- material: adjustment +1, one clear target-specific strength beyond ordinary
  competence; cite at least one target span.
- exceptional: adjustment +2, two independent rare target-specific strengths
  with no material defect; cite exactly two different target spans.

The required arithmetic is:

net_adjustment = negative adjustment + positive adjustment
model_score = baseline_score + net_adjustment

Thus 2 is ordinary competent baseline, 3 has a material above-baseline
strength, 4 is rare and exceptional, 1 has a major defect, and 0 has a
critical failure. A major defect may coexist with a material strength and net
back to 2, but both points must be real and separately justified.

Absence of faults, ordinary fluency, correct formatting, direct relevance,
long replies, emotional intensity, explicitness, and ornamental prose do not
by themselves justify positive adjustment. Generic praise such as
"immersive", "consistent", "high quality", or "no issues" is not a positive
point.

POINT EVIDENCE

Each negative_evidence and positive_evidence object has exactly:
level, adjustment, spans, reason.

Each span has exactly turn_id and quote:
- turn_id must be the target assistant turn, such as A5.
- quote must be a short exact substring copied from that target A-turn.
- never cite a user turn as evidence of assistant quality.
- never paraphrase inside a quote.
- spans contain at most two items.

reason is one compact auditable sentence, at most 180 characters, explaining
that specific positive or negative point. score_reason is at most 240
characters and explains the arithmetic, not a second free-form review.

QUALITY DIMENSIONS

1. character_fidelity
Identity, voice, temperament, motives, relationship role, world facts, and
role boundaries against supplied character material. Simply remaining in
character is baseline 2. A first-person claim of being merely a platform AI or
disclosure of policy/training limitations is critical. An in-character AI or
robot is not a break.

2. context_coherence
Continuity of facts, actions, location, persona, memory, and established causal
state. No contradiction is baseline 2. A material positive requires the target
to use concrete prior facts in a way that improves this response.

3. current_turn_fit
Specific handling of the current user's content, action, question, correction,
emotion, and constraints. Merely answering adequately is baseline 2. Do not
reward the user's own complexity; judge what the assistant actually does.

4. relationship_continuity
Preservation and use of names, roles, intimacy, conflict, trust, power
dynamics, and emotional state. Merely preserving the existing relationship is
baseline 2.

5. scene_progression
Meaningful causal movement without unsupported jump, reset, rehash, or
stalling. A routine reaction plus continuation is baseline 2.

6. language_roleplay_quality
Language-appropriate fluency, clarity, specificity, rhythm, restraint,
dialogue quality, readable formatting, and immersive craft. Ordinary polished
prose is baseline 2. Length or explicitness is never a positive point.

INTIMACY QUALITY

intimacy_quality uses the same point structure only when the target itself
meaningfully participates in romantic or sexual intimacy. Conflict, nudity,
arousal words, explicit anatomy, or taboo content alone do not establish
high-quality intimacy.

When not applicable, return:
applicable=false, baseline_score=-1, both evidence levels none with adjustment
0 and empty spans, net_adjustment=0, model_score=-1. Both evidence reasons and
score_reason must briefly state that intimacy is not applicable.

SEMANTIC REPETITION

Classify semantic repetition, not string similarity:
- none: no material repetition.
- minor: small redundant restatement that remains usable.
- major: substantial rehash, repeated action, or stalling.
- critical: near-total semantic loop with little new content.

semantic_repetition has exactly severity, history_turns, target_quote, reason.
For none, history_turns=[] and target_quote="". For non-none, cite up to three
earlier assistant turn integers and one exact target quote.

CRITICAL FLAGS

critical_flags contains exactly these booleans:
ai_identity_break, character_identity_break, context_break,
relationship_reset, wrong_language, format_collapse, meta_refusal.
True requires a concrete target-visible failure. Content safety, explicitness,
taboo subject matter, or moral undesirability are never critical quality flags.
critical_reason is one compact sentence.

NSFW

Classify only the target assistant reply:
0 none; 1 romantic/suggestive tension; 2 erotic content, nudity, arousal, or
non-graphic sexual touching; 3 explicit anatomy, direct genital stimulation,
graphic acts, penetration, or fluids; 4 sustained highly graphic, extreme,
fetish, or taboo sexual acts. Explicitness never changes quality scores.

OUTPUT CONTRACT

Top-level keys are exactly:
dimensions, intimacy_quality, semantic_repetition, critical_flags,
critical_reason, nsfw_level, nsfw_reason, confidence, overall_diagnosis.

dimensions contains exactly:
character_fidelity, context_coherence, current_turn_fit,
relationship_continuity, scene_progression, language_roleplay_quality.

Every core dimension has exactly:
applicable, baseline_score, negative_evidence, positive_evidence,
net_adjustment, model_score, score_reason.

intimacy_quality has the same seven keys.

negative_evidence and positive_evidence each have exactly:
level, adjustment, spans, reason.

confidence is an integer: 0 guess, 1 low, 2 moderate, 3 high, 4 very high.
Do not return an overall quality score or keep/drop decision. No extra keys are
allowed anywhere.
""".strip()

V45_FULL_JSON_TEMPLATE = r"""
STRICT JSON SHAPE TEMPLATE

The object below is a SHAPE TEMPLATE, not an evaluation example. Do not copy
its judgments. Re-evaluate every level, adjustment, applicability value,
score, reason, flag, and NSFW level from the supplied target.

Copy its key structure and JSON types exactly. In particular:
- negative_evidence is always one JSON OBJECT, never [] and never [object].
- positive_evidence is always one JSON OBJECT, never [] and never [object].
- even level "none" uses the complete point object shown below.
- spans alone is an array. Each non-empty spans item has exactly turn_id and
  quote, for example {"turn_id":"A5","quote":"exact text from target A5"}.
- replace all template reason strings with compact case-specific reasons.

{
  "dimensions": {
    "character_fidelity": {
      "applicable": true,
      "baseline_score": 2,
      "negative_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No material negative point for this dimension."
      },
      "positive_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No above-baseline positive point for this dimension."
      },
      "net_adjustment": 0,
      "model_score": 2,
      "score_reason": "Baseline 2 + negative 0 + positive 0 = 2."
    },
    "context_coherence": {
      "applicable": true,
      "baseline_score": 2,
      "negative_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No material negative point for this dimension."
      },
      "positive_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No above-baseline positive point for this dimension."
      },
      "net_adjustment": 0,
      "model_score": 2,
      "score_reason": "Baseline 2 + negative 0 + positive 0 = 2."
    },
    "current_turn_fit": {
      "applicable": true,
      "baseline_score": 2,
      "negative_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No material negative point for this dimension."
      },
      "positive_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No above-baseline positive point for this dimension."
      },
      "net_adjustment": 0,
      "model_score": 2,
      "score_reason": "Baseline 2 + negative 0 + positive 0 = 2."
    },
    "relationship_continuity": {
      "applicable": true,
      "baseline_score": 2,
      "negative_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No material negative point for this dimension."
      },
      "positive_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No above-baseline positive point for this dimension."
      },
      "net_adjustment": 0,
      "model_score": 2,
      "score_reason": "Baseline 2 + negative 0 + positive 0 = 2."
    },
    "scene_progression": {
      "applicable": true,
      "baseline_score": 2,
      "negative_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No material negative point for this dimension."
      },
      "positive_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No above-baseline positive point for this dimension."
      },
      "net_adjustment": 0,
      "model_score": 2,
      "score_reason": "Baseline 2 + negative 0 + positive 0 = 2."
    },
    "language_roleplay_quality": {
      "applicable": true,
      "baseline_score": 2,
      "negative_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No material negative point for this dimension."
      },
      "positive_evidence": {
        "level": "none",
        "adjustment": 0,
        "spans": [],
        "reason": "No above-baseline positive point for this dimension."
      },
      "net_adjustment": 0,
      "model_score": 2,
      "score_reason": "Baseline 2 + negative 0 + positive 0 = 2."
    }
  },
  "intimacy_quality": {
    "applicable": true,
    "baseline_score": 2,
    "negative_evidence": {
      "level": "none",
      "adjustment": 0,
      "spans": [],
      "reason": "No material negative point for this dimension."
    },
    "positive_evidence": {
      "level": "none",
      "adjustment": 0,
      "spans": [],
      "reason": "No above-baseline positive point for this dimension."
    },
    "net_adjustment": 0,
    "model_score": 2,
    "score_reason": "Baseline 2 + negative 0 + positive 0 = 2."
  },
  "semantic_repetition": {
    "severity": "none",
    "history_turns": [],
    "target_quote": "",
    "reason": "No material semantic repetition."
  },
  "critical_flags": {
    "ai_identity_break": false,
    "character_identity_break": false,
    "context_break": false,
    "relationship_reset": false,
    "wrong_language": false,
    "format_collapse": false,
    "meta_refusal": false
  },
  "critical_reason": "No critical target-visible failure.",
  "nsfw_level": 0,
  "nsfw_reason": "Case-specific NSFW classification reason.",
  "confidence": 3,
  "overall_diagnosis": "Compact case-specific diagnosis of the target turn."
}

Before returning, verify that all fourteen evidence fields are objects and
that the final output contains no markdown fences or extra commentary.
""".strip()

V45_POINT_REASON_PROMPT = (
    V44_POINT_REASON_PROMPT + "\n\n" + V45_FULL_JSON_TEMPLATE
)

V46_PROMPT_ONLY_SOFT_JSON = r"""
You are evaluating one target assistant turn from a fictional roleplay
conversation for a high-quality training corpus. Judge only the target A-turn.
Earlier assistant turns are context; there are no later turns.

Explicit or taboo sexual content is neither a quality bonus nor a penalty.
Do not perform content-safety review. Do not use external provenance or
popularity. Give no chain of thought.

SCORING

Use 2 as the ordinary competent baseline:
- 0: critical, unusable failure.
- 1: one clear major defect.
- 2: ordinary competent response; absence of faults stays at 2.
- 3: one concrete, material, target-specific strength.
- 4: at least two independent exceptional strengths and no material defect.

For each dimension, first identify the strongest defect, then the strongest
strength, and write one short auditable reason explaining the final score.
Generic praise such as "immersive", "consistent", or "high quality" does not
justify 3 or 4. The reason may quote or closely identify target text, but a
separate evidence object is not required.

DIMENSIONS

- character_fidelity: identity, voice, temperament, motives, relationship
  role, world facts, and role boundaries.
- context_coherence: continuity of facts, actions, location, memory, and
  established causal state.
- current_turn_fit: specific handling of the current user's content, action,
  emotion, question, and constraints.
- relationship_continuity: preservation and use of names, roles, intimacy,
  conflict, trust, power, and emotional state.
- scene_progression: meaningful causal movement without unsupported jump,
  reset, rehash, or stalling.
- language_roleplay_quality: language-appropriate fluency, clarity,
  specificity, dialogue, readable formatting, restraint, and roleplay craft.

INTIMACY

Score intimacy_quality only when the target meaningfully participates in
romantic or sexual intimacy. Otherwise set applicable=false and score=-1.
Explicit anatomy by itself does not establish high-quality intimacy.

OTHER LABELS

semantic_repetition severity is none, minor, major, or critical and judges
semantic rehash against earlier assistant turns.

critical_flags are booleans for target-visible failures only:
ai_identity_break, character_identity_break, context_break,
relationship_reset, wrong_language, format_collapse, meta_refusal.
Taboo or explicit subject matter alone never triggers a critical flag.

nsfw_level labels only the target assistant reply:
0 none; 1 romantic/suggestive; 2 erotic/nudity/arousal/non-graphic sexual
touching; 3 explicit anatomy, stimulation, penetration, or fluids; 4 sustained
highly graphic, extreme, fetish, or taboo sexual acts.

JSON OUTPUT

Return one valid JSON object. Use the following as the intended field layout.
Keep these field names and JSON value types so the result is easy to parse;
minor harmless formatting variation will be normalized afterward.

{
  "dimensions": {
    "character_fidelity": {"score": 2, "reason": "short case-specific reason"},
    "context_coherence": {"score": 2, "reason": "short case-specific reason"},
    "current_turn_fit": {"score": 2, "reason": "short case-specific reason"},
    "relationship_continuity": {"score": 2, "reason": "short case-specific reason"},
    "scene_progression": {"score": 2, "reason": "short case-specific reason"},
    "language_roleplay_quality": {"score": 2, "reason": "short case-specific reason"}
  },
  "intimacy_quality": {
    "applicable": false,
    "score": -1,
    "reason": "short case-specific reason"
  },
  "semantic_repetition": {
    "severity": "none",
    "reason": "short case-specific reason"
  },
  "critical_flags": {
    "ai_identity_break": false,
    "character_identity_break": false,
    "context_break": false,
    "relationship_reset": false,
    "wrong_language": false,
    "format_collapse": false,
    "meta_refusal": false
  },
  "critical_reason": "short case-specific reason",
  "nsfw_level": 0,
  "nsfw_reason": "short case-specific reason",
  "confidence": 3,
  "overall_reason": "short diagnosis of the target turn"
}

Output JSON only, without markdown fences.
""".strip()

PRESENT_PROMPT = V46_PROMPT_ONLY_SOFT_JSON
ABSENT_PROMPT = V46_PROMPT_ONLY_SOFT_JSON


def dimension_schema(*, allow_not_applicable: bool) -> dict[str, Any]:
    scores = [-1, 0, 1, 2, 3, 4] if allow_not_applicable else [0, 1, 2, 3, 4]
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": scores},
            "reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 220,
            },
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    }


def response_schema(variant: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "dimensions": {
            "type": "object",
            "properties": {
                name: dimension_schema(
                    allow_not_applicable=name
                    in {"length_following", "intimacy_quality"}
                )
                for name in SCORED_DIMENSIONS
            },
            "required": SCORED_DIMENSIONS,
            "additionalProperties": False,
        },
        "semantic_repetition": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "enum": ["none", "minor", "major", "critical"],
                },
                "reason": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 220,
                },
                "history_turns": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "maxItems": 3,
                    "uniqueItems": True,
                },
            },
            "required": ["severity", "reason", "history_turns"],
            "additionalProperties": False,
        },
        "critical_flags": {
            "type": "object",
            "properties": {
                key: {"type": "boolean"} for key in CRITICAL_FLAG_KEYS
            },
            "required": CRITICAL_FLAG_KEYS,
            "additionalProperties": False,
        },
        "critical_reason": {
            "type": "string",
            "minLength": 3,
            "maxLength": 220,
        },
        "nsfw_level": {
            "type": "integer",
            "enum": [0, 1, 2, 3, 4],
        },
        "nsfw_reason": {
            "type": "string",
            "minLength": 3,
            "maxLength": 220,
        },
        "overall_reason": {
            "type": "string",
            "minLength": 3,
            "maxLength": 220,
        },
        "confidence": {
            "type": "integer",
            "enum": [0, 1, 2, 3, 4],
        },
    }
    required = [
        "dimensions",
        "semantic_repetition",
        "critical_flags",
        "critical_reason",
        "nsfw_level",
        "nsfw_reason",
        "overall_reason",
        "confidence",
    ]
    if variant == "greeting_present":
        properties["greeting"] = {
            "type": "object",
            "properties": {
                "signature_evidence": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "feature": {
                                "type": "string",
                                "enum": GREETING_FEATURES,
                            },
                            "canonical_pattern": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 120,
                            },
                            "target_pattern": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 120,
                            },
                            "verdict": {
                                "type": "string",
                                "enum": [
                                    "match",
                                    "partial",
                                    "mismatch",
                                    "not_observable",
                                ],
                            },
                        },
                        "required": [
                            "feature",
                            "canonical_pattern",
                            "target_pattern",
                            "verdict",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["signature_evidence"],
            "additionalProperties": False,
        }
        required.append("greeting")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def stable_hash(namespace: str, *values: Any) -> str:
    raw = "|".join([SEED, namespace, *(str(value) for value in values)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def emit(stage: str, **values: Any) -> None:
    event = {"stage": stage, "time": time.time(), **values}
    with (ARTIFACTS / "events.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as output:
        output.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def strip_greeting_images(text: str) -> str:
    text = re.sub(r"!\[[^\]]*]\((?:\\.|[^)])*\)", "", text)
    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.I | re.S)
    text = re.sub(
        r"https?://[^\s<>'\"]+?\.(?:avif|bmp|gif|jpe?g|png|svg|webp)"
        r"(?:\?[^\s<>'\"]*)?",
        "",
        text,
        flags=re.I,
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fence(value: str) -> str:
    value = "" if value is None else str(value)
    if not value.strip():
        return "_Not provided._"
    longest = max([0] + [len(match) for match in re.findall(r"~+", value)])
    marker = "~" * max(4, longest + 1)
    return f"{marker}text\n{value}\n{marker}"


def parse_replay(messages: list[dict[str, Any]]) -> dict[str, Any]:
    systems: list[str] = []
    greetings: list[str] = []
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    seen_user = False
    for message in messages or []:
        role = clean(message.get("role"))
        content = (
            ""
            if message.get("content") is None
            else str(message.get("content"))
        )
        if not content.strip():
            continue
        if role == "system":
            systems.append(content)
        elif role == "assistant" and not seen_user and pending_user is None:
            greetings.append(content)
        elif role == "user":
            if pending_user is not None:
                raise ValueError("consecutive user messages")
            seen_user = True
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None
    if pending_user is not None:
        raise ValueError("unmatched final user")
    return {
        "systems": systems,
        "greetings": greetings,
        "pairs": pairs,
    }


def greeting_present(parsed: dict[str, Any]) -> bool:
    greeting = "\n\n".join(parsed.get("greetings") or [])
    return bool(strip_greeting_images(greeting).strip())


def turn_bucket(turn: int) -> str:
    if turn <= 8:
        return "05_08"
    if turn <= 16:
        return "09_16"
    if turn <= 32:
        return "17_32"
    return "33_plus"


def build_packet(candidate: dict[str, Any]) -> str:
    parsed = candidate["parsed"]
    turn = int(candidate["assistant_ordinal"])
    systems = parsed["systems"]
    greetings = parsed["greetings"]
    pairs = parsed["pairs"]
    role = systems[0] if systems else ""
    memory = "\n\n".join(systems[1:])
    greeting = strip_greeting_images("\n\n".join(greetings))
    current_user, target_assistant = pairs[turn - 1]
    sections = [
        "# Roleplay Target Turn Evaluation",
        "## Character Definition and Explicit Requirements",
        fence(role),
        "## User Persona",
        "_No separately structured User Persona is available in this frozen "
        "replay packet._",
        "## Memory Before the Target",
        fence(memory),
        "## Canonical Greeting",
        fence(greeting),
        "## Replay Dialogue Before the Target",
    ]
    for index, (user, assistant) in enumerate(
        pairs[: turn - 1], start=1
    ):
        sections.extend(
            [
                f"### U{index}",
                fence(user),
                f"### A{index}",
                fence(assistant),
            ]
        )
    sections.extend(
        [
            "## Target Turn",
            f"### U{turn}",
            fence(current_user),
            f"### A{turn}",
            fence(target_assistant),
            "## Evaluation Range",
            (
                f"- Evaluate only target A{turn}.\n"
                f"- Every evidence span turn_id must be A{turn}.\n"
                "- Earlier assistant replies are causal context only.\n"
                "- There are no later turns."
            ),
        ]
    )
    return "\n\n".join(sections)


def request_body(candidate: dict[str, Any], variant: str) -> dict[str, Any]:
    prompt = PRESENT_PROMPT if variant == "greeting_present" else ABSENT_PROMPT
    packet = build_packet(candidate)
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": packet},
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "seed": SEED_INT,
        "stream": False,
        "reasoning_effort": "none",
        "prompt_cache_key": "lusy_turn_judge_v46_promptonly_softjson_smoke",
        "response_format": {"type": "json_object"},
    }


def load_pilot_exclusions() -> set[tuple[str, int]]:
    if not PILOT_EXPECTATIONS_PATH.exists():
        return set()
    values = json.loads(PILOT_EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    return {
        (str(value["dialogue_id"]), int(value["turn"]))
        for value in values.values()
    }


def load_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    elite_ids = pq.read_table(
        ELITE_A_PATH, columns=["dialogue_id"]
    )["dialogue_id"]
    turn_table = pq.read_table(
        TURN_MANIFEST_PATH,
        columns=[
            "dialogue_id",
            "conversation_id",
            "assistant_ordinal",
            "native_message_id",
            "content_sha256",
            "is_legacy_candidate",
            "legacy_target_id",
            "replay_mode",
            "memory_id",
            "memory_boundary_increase_id",
        ],
    )
    elite_mask = pc.is_in(turn_table["dialogue_id"], value_set=elite_ids)
    candidate_mask = pc.equal(turn_table["is_legacy_candidate"], True)
    eligible_turns = turn_table.filter(pc.and_(elite_mask, candidate_mask))

    material_table = pq.read_table(
        MATERIAL_PATH,
        columns=[
            "dialogue_id",
            "conversation_id",
            "replay_mode",
            "dominant_language",
            "representative_topic",
            "messages",
        ],
        filters=[("dialogue_id", "in", elite_ids.to_pylist())],
    )
    material_by_dialogue = {
        str(row["dialogue_id"]): row
        for row in material_table.to_pylist()
    }
    parsed_by_dialogue: dict[str, dict[str, Any]] = {}
    parse_errors = 0
    exclusions = load_pilot_exclusions()
    excluded_pilot = 0
    hash_mismatches = 0
    ordinal_errors = 0
    candidates: list[dict[str, Any]] = []

    for turn_row in eligible_turns.to_pylist():
        dialogue_id = str(turn_row["dialogue_id"])
        ordinal = int(turn_row["assistant_ordinal"])
        if (dialogue_id, ordinal) in exclusions:
            excluded_pilot += 1
            continue
        material = material_by_dialogue.get(dialogue_id)
        if material is None:
            continue
        if dialogue_id not in parsed_by_dialogue:
            try:
                parsed_by_dialogue[dialogue_id] = parse_replay(
                    material["messages"] or []
                )
            except ValueError:
                parse_errors += 1
                parsed_by_dialogue[dialogue_id] = {}
        parsed = parsed_by_dialogue[dialogue_id]
        if not parsed:
            continue
        if ordinal < 5 or ordinal > len(parsed["pairs"]):
            ordinal_errors += 1
            continue
        target = parsed["pairs"][ordinal - 1][1]
        target_sha256 = hashlib.sha256(target.encode("utf-8")).hexdigest()
        if target_sha256 != str(turn_row["content_sha256"]):
            hash_mismatches += 1
            continue
        candidates.append(
            {
                **turn_row,
                "dialogue_id": dialogue_id,
                "conversation_id": str(turn_row["conversation_id"]),
                "legacy_target_id": str(turn_row["legacy_target_id"]),
                "native_message_id": str(turn_row["native_message_id"]),
                "assistant_ordinal": ordinal,
                "language": str(material["dominant_language"]),
                "topic": str(material["representative_topic"]),
                "parsed": parsed,
                "has_greeting": greeting_present(parsed),
                "turn_bucket": turn_bucket(ordinal),
            }
        )

    report = {
        "elite_dialogues": len(elite_ids),
        "elite_legacy_candidate_rows": len(eligible_turns),
        "material_rows": material_table.num_rows,
        "pilot_exclusion_pairs": len(exclusions),
        "excluded_pilot_rows": excluded_pilot,
        "parse_error_dialogues": parse_errors,
        "ordinal_error_rows": ordinal_errors,
        "content_sha256_mismatch_rows": hash_mismatches,
        "eligible_rows_after_validation": len(candidates),
    }
    return candidates, report


def choose_variant(
    candidates: list[dict[str, Any]],
    *,
    has_greeting: bool,
    globally_used_dialogues: set[str],
    globally_used_conversations: set[str],
) -> list[dict[str, Any]]:
    pool = [
        row
        for row in candidates
        if bool(row["has_greeting"]) == has_greeting
        and row["language"] in LANGUAGE_TARGETS
        and row["topic"] in TOPIC_TARGETS
    ]
    pool.sort(
        key=lambda row: stable_hash(
            "candidate_order",
            row["dialogue_id"],
            row["assistant_ordinal"],
        )
    )
    language_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []

    while len(selected) < SAMPLES_PER_VARIANT:
        best_row: dict[str, Any] | None = None
        best_key: tuple[int, str] | None = None
        for row in pool:
            dialogue_id = str(row["dialogue_id"])
            conversation_id = str(row["conversation_id"])
            if dialogue_id in globally_used_dialogues:
                continue
            if conversation_id in globally_used_conversations:
                continue
            if any(
                existing["dialogue_id"] == dialogue_id
                for existing in selected
            ):
                continue
            language = row["language"]
            topic = row["topic"]
            bucket = row["turn_bucket"]
            score = 0
            if language_counts[language] < LANGUAGE_TARGETS[language]:
                score += 100
            if topic_counts[topic] < TOPIC_TARGETS[topic]:
                score += 80
            if bucket_counts[bucket] < TURN_BUCKET_TARGETS[bucket]:
                score += 50
            score += max(
                0,
                LANGUAGE_TARGETS[language] - language_counts[language],
            )
            tie = stable_hash(
                "greedy_tie",
                row["dialogue_id"],
                row["assistant_ordinal"],
            )
            key = (score, "".join(chr(255 - ord(ch)) for ch in tie))
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
        if best_row is None:
            raise RuntimeError(
                f"could not choose {SAMPLES_PER_VARIANT} samples "
                f"for greeting={has_greeting}"
            )
        selected.append(best_row)
        language_counts[best_row["language"]] += 1
        topic_counts[best_row["topic"]] += 1
        bucket_counts[best_row["turn_bucket"]] += 1
        globally_used_dialogues.add(str(best_row["dialogue_id"]))
        globally_used_conversations.add(str(best_row["conversation_id"]))
    return selected


def sample_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_dialogues": len({row["dialogue_id"] for row in rows}),
        "unique_conversations": len(
            {row["conversation_id"] for row in rows}
        ),
        "greeting": dict(
            Counter(
                "present" if row["has_greeting"] else "absent"
                for row in rows
            )
        ),
        "language": dict(Counter(row["language"] for row in rows)),
        "topic": dict(Counter(row["topic"] for row in rows)),
        "turn_bucket": dict(
            Counter(row["turn_bucket"] for row in rows)
        ),
    }


def select_samples(
    candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    candidate_by_key = {
        (str(row["dialogue_id"]), int(row["assistant_ordinal"])): row
        for row in candidates
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    for variant, custom_ids in RETEST_CUSTOM_IDS.items():
        previous = json.loads(
            (
                PREVIOUS_SMOKE_ARTIFACTS
                / f"{variant}.expectations.json"
            ).read_text(encoding="utf-8")
        )
        rows = []
        for custom_id in custom_ids:
            expectation = previous[custom_id]
            key = (
                str(expectation["dialogue_id"]),
                int(expectation["assistant_ordinal"]),
            )
            rows.append(candidate_by_key[key])
        if len(rows) != SAMPLES_PER_VARIANT:
            raise RuntimeError(
                f"{variant}: expected {SAMPLES_PER_VARIANT} retest rows"
            )
        selected[variant] = rows
    return selected


def public_sample_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in [
            "dialogue_id",
            "conversation_id",
            "legacy_target_id",
            "native_message_id",
            "assistant_ordinal",
            "replay_mode",
            "memory_id",
            "memory_boundary_increase_id",
            "content_sha256",
            "language",
            "topic",
            "has_greeting",
            "turn_bucket",
        ]
    }


def extract_length_requirement(row: dict[str, Any]) -> str:
    turn = int(row["assistant_ordinal"])
    current_user = row["parsed"]["pairs"][turn - 1][0]
    match = re.search(
        r"\[RESPONSE LENGTH GUIDELINE\]\s*(.+?)(?:\n|$)",
        current_user,
        flags=re.I,
    )
    return clean(match.group(1)) if match else ""


def materialize_variant(
    variant: str,
    rows: list[dict[str, Any]],
) -> tuple[pathlib.Path, dict[str, dict[str, Any]]]:
    batch_rows: list[dict[str, Any]] = []
    expectations: dict[str, dict[str, Any]] = {}
    for row in rows:
        custom_id = "v45s_" + stable_hash(
            "request",
            variant,
            row["dialogue_id"],
            row["assistant_ordinal"],
        )[:28]
        body = request_body(row, variant)
        batch_rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
        expectations[custom_id] = {
            **public_sample_row(row),
            "variant": variant,
            "target_content": row["parsed"]["pairs"][
                int(row["assistant_ordinal"]) - 1
            ][1],
            "reference_corpus": "\n\n".join(
                [
                    *row["parsed"]["systems"],
                    *row["parsed"]["greetings"],
                    *[
                        text
                        for pair in row["parsed"]["pairs"][
                            : int(row["assistant_ordinal"])
                        ]
                        for text in pair
                    ],
                ]
            ),
            "prior_assistant_corpus": "\n\n".join(
                [
                    *row["parsed"]["greetings"],
                    *[
                        pair[1]
                        for pair in row["parsed"]["pairs"][
                            : int(row["assistant_ordinal"]) - 1
                        ]
                    ],
                ]
            ),
            "length_requirement": extract_length_requirement(row),
            "length_following_applicable": bool(
                extract_length_requirement(row)
            ),
        }

    custom_ids = [row["custom_id"] for row in batch_rows]
    if len(custom_ids) != len(set(custom_ids)):
        raise RuntimeError(f"{variant}: duplicate custom_id")
    if len(batch_rows) != SAMPLES_PER_VARIANT:
        raise RuntimeError(f"{variant}: unexpected row count")
    path = ARTIFACTS / f"{variant}.input.jsonl"
    write_jsonl(path, batch_rows)
    if path.stat().st_size > MAX_BATCH_FILE_BYTES:
        raise RuntimeError(f"{variant}: input exceeds 150 MB")
    atomic_json(
        ARTIFACTS / f"{variant}.manifest.json",
        {
            "schema_version": "lusy-turn-judge-v46-promptonly-softjson-smoke-v1",
            "variant": variant,
            "requests": len(batch_rows),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model": MODEL,
            "seed": SEED,
            "prompt_sha256": hashlib.sha256(
                (
                    PRESENT_PROMPT
                    if variant == "greeting_present"
                    else ABSENT_PROMPT
                ).encode("utf-8")
            ).hexdigest(),
        },
    )
    atomic_json(
        ARTIFACTS / f"{variant}.expectations.json",
        expectations,
    )
    return path, expectations


def valid_reason(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= 240
    )


def validate_dimension(
    name: str,
    value: Any,
    errors: list[str],
) -> tuple[int | None, str]:
    if not isinstance(value, dict) or set(value) != {"score", "reason"}:
        errors.append(f"dimension_shape:{name}")
        return None, ""
    score = value.get("score")
    allowed = {-1, 0, 1, 2, 3, 4}
    if name in CORE_DIMENSIONS:
        allowed = {0, 1, 2, 3, 4}
    if type(score) is not int or score not in allowed:
        errors.append(f"dimension_score:{name}")
        normalized_score = None
    else:
        normalized_score = score
    reason = value.get("reason")
    if not valid_reason(reason):
        errors.append(f"dimension_reason:{name}")
        return normalized_score, ""
    normalized_reason = str(reason).strip()
    if type(score) is int:
        prefix_valid = (
            (score in {0, 1} and normalized_reason.startswith("D:"))
            or (score == 2 and normalized_reason.startswith("Baseline:"))
            or (score == 3 and normalized_reason.startswith("S1:"))
            or (
                score == 4
                and normalized_reason.startswith("S1:")
                and "; S2:" in normalized_reason
            )
            or (score == -1 and normalized_reason.startswith("N/A:"))
        )
        if not prefix_valid:
            errors.append(f"dimension_reason_contract:{name}")
    return normalized_score, normalized_reason


def normalize_result(
    raw_result: Any,
    expectation: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(raw_result, dict):
        return {}, ["result_not_object"]
    variant = str(expectation["variant"])
    required = {
        "dimensions",
        "semantic_repetition",
        "critical_flags",
        "critical_reason",
        "nsfw_level",
        "nsfw_reason",
        "overall_reason",
        "confidence",
    }
    if variant == "greeting_present":
        required.add("greeting")
    if set(raw_result) != required:
        errors.append("top_level_keys")

    dimensions = raw_result.get("dimensions")
    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    normalization_notes: list[str] = []
    if (
        not isinstance(dimensions, dict)
        or set(dimensions) != set(SCORED_DIMENSIONS)
    ):
        errors.append("dimension_keys")
    else:
        for name in SCORED_DIMENSIONS:
            score, reason = validate_dimension(
                name,
                dimensions.get(name),
                errors,
            )
            if score is not None:
                scores[name] = score
            if reason:
                reasons[name] = reason

    length_applicable = bool(
        expectation.get("length_following_applicable")
    )
    if "length_following" in scores:
        if not length_applicable:
            if scores["length_following"] != -1:
                normalization_notes.append(
                    "length_following_forced_to_na_by_offline_gate"
                )
            scores["length_following"] = -1
            reasons["length_following"] = (
                "N/A: no offline length requirement is present."
            )
        elif scores["length_following"] == -1:
            errors.append("length_applicable_but_na")

    repetition = raw_result.get("semantic_repetition")
    repetition_severity = ""
    repetition_reason = ""
    repetition_turns: list[int] = []
    if (
        not isinstance(repetition, dict)
        or set(repetition) != {"severity", "reason", "history_turns"}
    ):
        errors.append("semantic_repetition_shape")
    else:
        repetition_severity = repetition.get("severity")
        if repetition_severity not in {
            "none",
            "minor",
            "major",
            "critical",
        }:
            errors.append("semantic_repetition_severity")
        repetition_reason = repetition.get("reason")
        if not valid_reason(repetition_reason):
            errors.append("semantic_repetition_reason")
        turns = repetition.get("history_turns")
        target_turn = int(expectation["assistant_ordinal"])
        if (
            not isinstance(turns, list)
            or len(turns) > 3
            or any(
                type(turn) is not int
                or turn < 1
                or turn >= target_turn
                for turn in turns
            )
            or len(set(turns)) != len(turns)
        ):
            errors.append("semantic_repetition_history_turns")
        else:
            repetition_turns = turns
        if (
            repetition_severity in {"major", "critical"}
            and not repetition_turns
        ):
            errors.append("semantic_repetition_missing_history_evidence")

    non_repetition_map = {
        "none": 2,
        "minor": 2,
        "major": 1,
        "critical": 0,
    }
    if repetition_severity in non_repetition_map:
        scores["non_repetition"] = non_repetition_map[repetition_severity]
        reasons["non_repetition"] = str(repetition_reason).strip()

    if variant == "greeting_present":
        greeting = raw_result.get("greeting")
        if (
            not isinstance(greeting, dict)
            or set(greeting) != {"signature_evidence"}
        ):
            errors.append("greeting_shape")
        else:
            evidence = greeting.get("signature_evidence")
            match_count = 0
            mismatch_count = 0
            if (
                not isinstance(evidence, list)
                or len(evidence) != 5
            ):
                errors.append("greeting_evidence_shape")
            else:
                seen_features: set[str] = set()
                for index, item in enumerate(evidence):
                    if (
                        not isinstance(item, dict)
                        or set(item)
                        != {
                            "feature",
                            "canonical_pattern",
                            "target_pattern",
                            "verdict",
                        }
                    ):
                        errors.append(f"greeting_evidence_item:{index}")
                        continue
                    feature = item.get("feature")
                    verdict = item.get("verdict")
                    if (
                        feature not in GREETING_FEATURES
                        or feature in seen_features
                    ):
                        errors.append(
                            f"greeting_evidence_feature:{index}"
                        )
                    else:
                        seen_features.add(feature)
                    if verdict not in {
                        "match",
                        "partial",
                        "mismatch",
                        "not_observable",
                    }:
                        errors.append(
                            f"greeting_evidence_verdict:{index}"
                        )
                    if not valid_reason(item.get("canonical_pattern")):
                        errors.append(
                            f"greeting_evidence_canonical:{index}"
                        )
                    if not valid_reason(item.get("target_pattern")):
                        errors.append(
                            f"greeting_evidence_target:{index}"
                        )
                    match_count += verdict == "match"
                    mismatch_count += verdict == "mismatch"
                if seen_features != set(GREETING_FEATURES):
                    errors.append("greeting_evidence_feature_set")
            if mismatch_count >= 2:
                greeting_score = 0
            elif mismatch_count == 1:
                greeting_score = 1
            elif match_count >= 2:
                greeting_score = 4
            elif match_count == 1:
                greeting_score = 3
            else:
                greeting_score = 2
            scores["greeting_format_adherence"] = greeting_score
            reasons["greeting_format_adherence"] = (
                "Offline evidence aggregation: "
                f"{match_count} matches, {mismatch_count} mismatches."
            )
    else:
        scores["greeting_format_adherence"] = -1
        reasons["greeting_format_adherence"] = (
            "Not applicable: canonical greeting is absent."
        )

    flags = raw_result.get("critical_flags")
    normalized_flags: dict[str, bool] = {}
    if (
        not isinstance(flags, dict)
        or set(flags) != set(CRITICAL_FLAG_KEYS)
    ):
        errors.append("critical_flag_keys")
    else:
        for key in CRITICAL_FLAG_KEYS:
            value = flags.get(key)
            if type(value) is not bool:
                errors.append(f"critical_flag_value:{key}")
            else:
                normalized_flags[key] = value
    critical_reason = raw_result.get("critical_reason")
    if not valid_reason(critical_reason):
        errors.append("critical_reason")

    nsfw_level = raw_result.get("nsfw_level")
    if type(nsfw_level) is not int or nsfw_level not in {0, 1, 2, 3, 4}:
        errors.append("nsfw_level")
    nsfw_reason = raw_result.get("nsfw_reason")
    if not valid_reason(nsfw_reason):
        errors.append("nsfw_reason")
    overall_reason = raw_result.get("overall_reason")
    if not valid_reason(overall_reason):
        errors.append("overall_reason")
    confidence = raw_result.get("confidence")
    if type(confidence) is not int or confidence not in {0, 1, 2, 3, 4}:
        errors.append("confidence")

    normalized = {
        "scores": scores,
        "reasons": reasons,
        "semantic_repetition": {
            "severity": repetition_severity,
            "reason": repetition_reason,
            "history_turns": repetition_turns,
            "deterministic_features_applied": False,
        },
        "greeting_signature_evidence": (
            (raw_result.get("greeting") or {}).get(
                "signature_evidence", []
            )
            if variant == "greeting_present"
            else []
        ),
        "critical_flags": normalized_flags,
        "critical_reason": critical_reason,
        "nsfw_level": nsfw_level,
        "nsfw_reason": nsfw_reason,
        "positive_dimensions": sorted(
            name for name, score in scores.items() if score in {3, 4}
        ),
        "negative_dimensions": sorted(
            name for name, score in scores.items() if score in {0, 1}
        ),
        "overall_reason": overall_reason,
        "confidence": confidence,
        "normalization_notes": normalization_notes,
    }
    return normalized, sorted(set(errors))


EVIDENCE_KINDS = {
    "strength",
    "minor_defect",
    "major_defect",
    "critical_defect",
}
REFERENCE_SOURCES = {
    "character_definition",
    "explicit_requirement",
    "user_persona",
    "memory",
    "current_user",
    "prior_user",
    "prior_assistant",
}


def response_schema_v42() -> dict[str, Any]:
    evidence = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": sorted(EVIDENCE_KINDS)},
            "target_quote": {
                "type": "string",
                "minLength": 1,
                "maxLength": 180,
            },
            "reference_source": {
                "type": ["string", "null"],
                "enum": [*sorted(REFERENCE_SOURCES), None],
            },
            "reference_id": {
                "type": ["string", "null"],
                "maxLength": 40,
            },
            "reference_quote": {
                "type": ["string", "null"],
                "maxLength": 180,
            },
        },
        "required": [
            "kind",
            "target_quote",
            "reference_source",
            "reference_id",
            "reference_quote",
        ],
        "additionalProperties": False,
    }
    dimension = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
            "reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 180,
            },
            "evidence": {
                "type": "array",
                "items": evidence,
                "maxItems": 2,
            },
        },
        "required": ["score", "reason", "evidence"],
        "additionalProperties": False,
    }
    intimacy = {
        "type": "object",
        "properties": {
            "applicable": {"type": "boolean"},
            "score": {
                "type": "integer",
                "enum": [-1, 0, 1, 2, 3, 4],
            },
            "reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 180,
            },
            "evidence": {
                "type": "array",
                "items": evidence,
                "maxItems": 2,
            },
        },
        "required": ["applicable", "score", "reason", "evidence"],
        "additionalProperties": False,
    }
    dimensions = {
        "type": "object",
        "properties": {
            name: (intimacy if name == "intimacy_quality" else dimension)
            for name in SCORED_DIMENSIONS
        },
        "required": SCORED_DIMENSIONS,
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "dimensions": dimensions,
            "semantic_repetition": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["none", "minor", "major", "critical"],
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 3,
                        "maxLength": 180,
                    },
                    "history_turns": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "maxItems": 3,
                        "uniqueItems": True,
                    },
                    "target_quote": {
                        "type": "string",
                        "maxLength": 180,
                    },
                    "history_quotes": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "turn": {"type": "integer", "minimum": 1},
                                "quote": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 180,
                                },
                            },
                            "required": ["turn", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "severity",
                    "reason",
                    "history_turns",
                    "target_quote",
                    "history_quotes",
                ],
                "additionalProperties": False,
            },
            "critical_flags": {
                "type": "object",
                "properties": {
                    key: {"type": "boolean"} for key in CRITICAL_FLAG_KEYS
                },
                "required": CRITICAL_FLAG_KEYS,
                "additionalProperties": False,
            },
            "critical_reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 180,
            },
            "nsfw_level": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4],
            },
            "nsfw_reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 180,
            },
            "overall_reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 180,
            },
        },
        "required": [
            "dimensions",
            "semantic_repetition",
            "critical_flags",
            "critical_reason",
            "nsfw_level",
            "nsfw_reason",
            "overall_reason",
        ],
        "additionalProperties": False,
    }


def valid_short_text(value: Any, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, str)
        and (allow_empty or bool(value.strip()))
        and len(value) <= 180
    )


def validate_evidence_v42(
    value: Any,
    expectation: dict[str, Any],
    error_prefix: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 2:
        errors.append(f"{error_prefix}:evidence_shape")
        return []
    target = str(expectation.get("target_content") or "")
    reference_corpus = str(expectation.get("reference_corpus") or "")
    normalized: list[dict[str, Any]] = []
    required = {
        "kind",
        "target_quote",
        "reference_source",
        "reference_id",
        "reference_quote",
    }
    for index, item in enumerate(value):
        prefix = f"{error_prefix}:evidence_{index}"
        if not isinstance(item, dict) or set(item) != required:
            errors.append(f"{prefix}:keys")
            continue
        kind = item.get("kind")
        target_quote = item.get("target_quote")
        source = item.get("reference_source")
        reference_id = item.get("reference_id")
        reference_quote = item.get("reference_quote")
        if kind not in EVIDENCE_KINDS:
            errors.append(f"{prefix}:kind")
        if (
            not valid_short_text(target_quote)
            or str(target_quote) not in target
        ):
            errors.append(f"{prefix}:target_quote")
        if source is not None and source not in REFERENCE_SOURCES:
            errors.append(f"{prefix}:reference_source")
        if reference_id is not None and (
            not isinstance(reference_id, str)
            or not reference_id.strip()
            or len(reference_id) > 40
        ):
            errors.append(f"{prefix}:reference_id")
        if source is None:
            if reference_id is not None or reference_quote is not None:
                errors.append(f"{prefix}:null_reference_contract")
        elif (
            not valid_short_text(reference_quote)
            or str(reference_quote) not in reference_corpus
        ):
            errors.append(f"{prefix}:reference_quote")
        normalized.append(
            {
                "kind": kind,
                "target_quote": target_quote,
                "reference_source": source,
                "reference_id": reference_id,
                "reference_quote": reference_quote,
            }
        )
    return normalized


def validate_score_evidence_contract(
    score: int,
    evidence: list[dict[str, Any]],
    prefix: str,
    errors: list[str],
) -> None:
    kinds = [item.get("kind") for item in evidence]
    if score == 0 and "critical_defect" not in kinds:
        errors.append(f"{prefix}:score0_requires_critical")
    elif score == 1 and (
        "major_defect" not in kinds or "critical_defect" in kinds
    ):
        errors.append(f"{prefix}:score1_requires_major")
    elif score == 2 and any(kind != "minor_defect" for kind in kinds):
        errors.append(f"{prefix}:score2_evidence")
    elif score == 3 and (
        "strength" not in kinds
        or any(
            kind in {"major_defect", "critical_defect"} for kind in kinds
        )
    ):
        errors.append(f"{prefix}:score3_requires_strength")
    elif score == 4:
        signatures = {
            (
                item.get("target_quote"),
                item.get("reference_source"),
                item.get("reference_quote"),
            )
            for item in evidence
        }
        if (
            len(evidence) != 2
            or any(kind != "strength" for kind in kinds)
            or len(signatures) != 2
        ):
            errors.append(f"{prefix}:score4_requires_two_strengths")


def validate_dimension_v42(
    name: str,
    value: Any,
    expectation: dict[str, Any],
    errors: list[str],
) -> tuple[int | None, str, list[dict[str, Any]], bool]:
    is_intimacy = name == "intimacy_quality"
    required = (
        {"applicable", "score", "reason", "evidence"}
        if is_intimacy
        else {"score", "reason", "evidence"}
    )
    prefix = f"dimension:{name}"
    if not isinstance(value, dict) or set(value) != required:
        errors.append(f"{prefix}:shape")
        return None, "", [], not is_intimacy
    applicable = value.get("applicable", True)
    if type(applicable) is not bool:
        errors.append(f"{prefix}:applicable")
        applicable = True
    score = value.get("score")
    allowed = {-1} if is_intimacy and not applicable else {0, 1, 2, 3, 4}
    if type(score) is not int or score not in allowed:
        errors.append(f"{prefix}:score")
        normalized_score = None
    else:
        normalized_score = score
    reason = value.get("reason")
    if not valid_short_text(reason):
        errors.append(f"{prefix}:reason")
        normalized_reason = ""
    else:
        normalized_reason = str(reason).strip()
    evidence = validate_evidence_v42(
        value.get("evidence"),
        expectation,
        prefix,
        errors,
    )
    if is_intimacy and not applicable:
        if evidence:
            errors.append(f"{prefix}:na_evidence")
    elif normalized_score is not None:
        validate_score_evidence_contract(
            normalized_score, evidence, prefix, errors
        )
    return normalized_score, normalized_reason, evidence, bool(applicable)


def score_length_offline(
    requirement: str,
    target: str,
) -> tuple[int, str]:
    requirement = clean(requirement)
    if not requirement:
        return -1, "No explicit length requirement."
    word_count = len(re.findall(r"\b[\w'-]+\b", target, flags=re.UNICODE))
    sentence_count = len(
        [part for part in re.split(r"[.!?。！？]+", target) if part.strip()]
    )
    paragraph_count = len(
        [part for part in re.split(r"\n\s*\n", target) if part.strip()]
    )
    lowered = requirement.lower()
    range_match = re.search(
        r"(\d+)\s*(?:-|–|—|to)\s*(\d+)\s*"
        r"(sentences?|paragraphs?|words?)",
        lowered,
    )
    exact_match = re.search(
        r"(?:exactly\s*)?(\d+)\s*(sentences?|paragraphs?|words?)",
        lowered,
    )
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        unit = range_match.group(3)
    elif exact_match:
        low = high = int(exact_match.group(1))
        unit = exact_match.group(2)
    else:
        return 2, (
            "Explicit qualitative length requirement retained for offline "
            "review; no numeric violation detected."
        )
    actual = (
        word_count
        if unit.startswith("word")
        else paragraph_count
        if unit.startswith("paragraph")
        else sentence_count
    )
    if low <= actual <= high:
        return 2, f"Offline count {actual} {unit}; required {low}-{high}."
    distance = low - actual if actual < low else actual - high
    score = 1 if distance <= max(1, math.ceil((high - low + 1) / 2)) else 0
    return score, f"Offline count {actual} {unit}; required {low}-{high}."


def normalize_result_v42(
    raw_result: Any,
    expectation: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(raw_result, dict):
        return {}, ["result_not_object"]
    required = {
        "dimensions",
        "semantic_repetition",
        "critical_flags",
        "critical_reason",
        "nsfw_level",
        "nsfw_reason",
        "overall_reason",
    }
    if set(raw_result) != required:
        errors.append("top_level_keys")

    dimensions = raw_result.get("dimensions")
    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    evidence_by_dimension: dict[str, list[dict[str, Any]]] = {}
    intimacy_applicable: bool | None = None
    if not isinstance(dimensions, dict) or set(dimensions) != set(
        SCORED_DIMENSIONS
    ):
        errors.append("dimension_keys")
    else:
        for name in SCORED_DIMENSIONS:
            score, reason, evidence, applicable = validate_dimension_v42(
                name, dimensions.get(name), expectation, errors
            )
            if score is not None:
                scores[name] = score
            if reason:
                reasons[name] = reason
            evidence_by_dimension[name] = evidence
            if name == "intimacy_quality":
                intimacy_applicable = applicable

    repetition = raw_result.get("semantic_repetition")
    repetition_severity = ""
    repetition_reason = ""
    repetition_turns: list[int] = []
    repetition_target_quote = ""
    repetition_history_quotes: list[dict[str, Any]] = []
    repetition_keys = {
        "severity",
        "reason",
        "history_turns",
        "target_quote",
        "history_quotes",
    }
    if not isinstance(repetition, dict) or set(repetition) != repetition_keys:
        errors.append("semantic_repetition:shape")
    else:
        repetition_severity = repetition.get("severity")
        if repetition_severity not in {
            "none",
            "minor",
            "major",
            "critical",
        }:
            errors.append("semantic_repetition:severity")
        repetition_reason = repetition.get("reason")
        if not valid_short_text(repetition_reason):
            errors.append("semantic_repetition:reason")
        turns = repetition.get("history_turns")
        target_turn = int(expectation["assistant_ordinal"])
        if (
            not isinstance(turns, list)
            or len(turns) > 3
            or any(
                type(turn) is not int
                or turn < 1
                or turn >= target_turn
                for turn in turns
            )
            or len(set(turns)) != len(turns)
        ):
            errors.append("semantic_repetition:history_turns")
        else:
            repetition_turns = turns
        target_quote = repetition.get("target_quote")
        if not valid_short_text(target_quote, allow_empty=True):
            errors.append("semantic_repetition:target_quote_shape")
        else:
            repetition_target_quote = str(target_quote)
        history_quotes = repetition.get("history_quotes")
        if not isinstance(history_quotes, list) or len(history_quotes) > 3:
            errors.append("semantic_repetition:history_quotes_shape")
        else:
            seen_quote_turns: set[int] = set()
            for index, item in enumerate(history_quotes):
                if (
                    not isinstance(item, dict)
                    or set(item) != {"turn", "quote"}
                    or type(item.get("turn")) is not int
                    or not valid_short_text(item.get("quote"))
                ):
                    errors.append(
                        f"semantic_repetition:history_quote_{index}"
                    )
                    continue
                turn = int(item["turn"])
                quote = str(item["quote"])
                if (
                    turn < 1
                    or turn >= target_turn
                    or quote
                    not in str(
                        expectation.get("prior_assistant_corpus") or ""
                    )
                    or turn in seen_quote_turns
                ):
                    errors.append(
                        f"semantic_repetition:history_quote_{index}_evidence"
                    )
                seen_quote_turns.add(turn)
                repetition_history_quotes.append(
                    {"turn": turn, "quote": quote}
                )
            if set(repetition_turns) != seen_quote_turns:
                errors.append("semantic_repetition:turn_quote_alignment")
        if repetition_severity == "none":
            if (
                repetition_turns
                or repetition_target_quote
                or repetition_history_quotes
            ):
                errors.append("semantic_repetition:none_has_evidence")
        elif (
            not repetition_turns
            or not repetition_history_quotes
            or not repetition_target_quote
            or repetition_target_quote
            not in str(expectation.get("target_content") or "")
        ):
            errors.append("semantic_repetition:missing_exact_evidence")

    flags = raw_result.get("critical_flags")
    normalized_flags: dict[str, bool] = {}
    if not isinstance(flags, dict) or set(flags) != set(CRITICAL_FLAG_KEYS):
        errors.append("critical_flag_keys")
    else:
        for key in CRITICAL_FLAG_KEYS:
            flag = flags.get(key)
            if type(flag) is not bool:
                errors.append(f"critical_flag_value:{key}")
            else:
                normalized_flags[key] = flag
    critical_reason = raw_result.get("critical_reason")
    if not valid_short_text(critical_reason):
        errors.append("critical_reason")
    nsfw_level = raw_result.get("nsfw_level")
    if type(nsfw_level) is not int or nsfw_level not in {0, 1, 2, 3, 4}:
        errors.append("nsfw_level")
    nsfw_reason = raw_result.get("nsfw_reason")
    if not valid_short_text(nsfw_reason):
        errors.append("nsfw_reason")
    overall_reason = raw_result.get("overall_reason")
    if not valid_short_text(overall_reason):
        errors.append("overall_reason")

    length_score, length_reason = score_length_offline(
        str(expectation.get("length_requirement") or ""),
        str(expectation.get("target_content") or ""),
    )
    scores["length_following"] = length_score
    reasons["length_following"] = length_reason
    normalized = {
        "scores": scores,
        "reasons": reasons,
        "evidence_by_dimension": evidence_by_dimension,
        "intimacy_applicable": intimacy_applicable,
        "semantic_repetition": {
            "severity": repetition_severity,
            "reason": repetition_reason,
            "history_turns": repetition_turns,
            "target_quote": repetition_target_quote,
            "history_quotes": repetition_history_quotes,
            "deterministic_features_applied": False,
        },
        "critical_flags": normalized_flags,
        "critical_reason": critical_reason,
        "nsfw_level": nsfw_level,
        "nsfw_reason": nsfw_reason,
        "positive_dimensions": sorted(
            name for name, score in scores.items() if score in {3, 4}
        ),
        "negative_dimensions": sorted(
            name for name, score in scores.items() if score in {0, 1}
        ),
        "overall_reason": overall_reason,
        "normalization_notes": [
            "length_following_scored_offline",
            "positive_negative_dimensions_derived_offline",
        ],
    }
    return normalized, sorted(set(errors))


def response_schema_v43() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {
                    name: {
                        "type": "integer",
                        "enum": [0, 1, 2, 3, 4],
                    }
                    for name in CORE_DIMENSIONS
                },
                "required": CORE_DIMENSIONS,
                "additionalProperties": False,
            },
            "intimacy_quality": {
                "type": "object",
                "properties": {
                    "applicable": {"type": "boolean"},
                    "score": {
                        "type": "integer",
                        "enum": [-1, 0, 1, 2, 3, 4],
                    },
                },
                "required": ["applicable", "score"],
                "additionalProperties": False,
            },
            "semantic_repetition": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["none", "minor", "major", "critical"],
                    },
                    "history_turns": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "maxItems": 3,
                        "uniqueItems": True,
                    },
                },
                "required": ["severity", "history_turns"],
                "additionalProperties": False,
            },
            "critical_flags": {
                "type": "object",
                "properties": {
                    key: {"type": "boolean"} for key in CRITICAL_FLAG_KEYS
                },
                "required": CRITICAL_FLAG_KEYS,
                "additionalProperties": False,
            },
            "nsfw_level": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4],
            },
        },
        "required": [
            "scores",
            "intimacy_quality",
            "semantic_repetition",
            "critical_flags",
            "nsfw_level",
        ],
        "additionalProperties": False,
    }


def normalize_result_v43(
    raw_result: Any,
    expectation: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(raw_result, dict):
        return {}, ["result_not_object"]
    required = {
        "scores",
        "intimacy_quality",
        "semantic_repetition",
        "critical_flags",
        "nsfw_level",
    }
    if set(raw_result) != required:
        errors.append("top_level_keys")

    scores: dict[str, int] = {}
    raw_scores = raw_result.get("scores")
    if (
        not isinstance(raw_scores, dict)
        or set(raw_scores) != set(CORE_DIMENSIONS)
    ):
        errors.append("score_keys")
    else:
        for name in CORE_DIMENSIONS:
            score = raw_scores.get(name)
            if type(score) is not int or score not in {0, 1, 2, 3, 4}:
                errors.append(f"score_value:{name}")
            else:
                scores[name] = score

    intimacy = raw_result.get("intimacy_quality")
    intimacy_applicable: bool | None = None
    if (
        not isinstance(intimacy, dict)
        or set(intimacy) != {"applicable", "score"}
    ):
        errors.append("intimacy_shape")
    else:
        applicable = intimacy.get("applicable")
        score = intimacy.get("score")
        if type(applicable) is not bool:
            errors.append("intimacy_applicable")
        else:
            intimacy_applicable = applicable
        allowed = {0, 1, 2, 3, 4} if applicable is True else {-1}
        if type(score) is not int or score not in allowed:
            errors.append("intimacy_score")
        else:
            scores["intimacy_quality"] = score

    repetition = raw_result.get("semantic_repetition")
    repetition_severity = ""
    repetition_turns: list[int] = []
    if (
        not isinstance(repetition, dict)
        or set(repetition) != {"severity", "history_turns"}
    ):
        errors.append("semantic_repetition_shape")
    else:
        severity = repetition.get("severity")
        if severity not in {"none", "minor", "major", "critical"}:
            errors.append("semantic_repetition_severity")
        else:
            repetition_severity = severity
        turns = repetition.get("history_turns")
        target_turn = int(expectation["assistant_ordinal"])
        if (
            not isinstance(turns, list)
            or len(turns) > 3
            or len(set(turns)) != len(turns)
            or any(
                type(turn) is not int
                or turn < 1
                or turn >= target_turn
                for turn in turns
            )
        ):
            errors.append("semantic_repetition_history_turns")
        else:
            repetition_turns = turns
        if severity == "none" and repetition_turns:
            errors.append("semantic_repetition_none_has_turns")
        if severity in {"minor", "major", "critical"} and not repetition_turns:
            errors.append("semantic_repetition_missing_turns")

    flags = raw_result.get("critical_flags")
    normalized_flags: dict[str, bool] = {}
    if not isinstance(flags, dict) or set(flags) != set(CRITICAL_FLAG_KEYS):
        errors.append("critical_flag_keys")
    else:
        for key in CRITICAL_FLAG_KEYS:
            flag = flags.get(key)
            if type(flag) is not bool:
                errors.append(f"critical_flag_value:{key}")
            else:
                normalized_flags[key] = flag

    nsfw_level = raw_result.get("nsfw_level")
    if type(nsfw_level) is not int or nsfw_level not in {0, 1, 2, 3, 4}:
        errors.append("nsfw_level")

    length_score, length_reason = score_length_offline(
        str(expectation.get("length_requirement") or ""),
        str(expectation.get("target_content") or ""),
    )
    scores["length_following"] = length_score
    normalized = {
        "scores": scores,
        "reasons": {"length_following": length_reason},
        "evidence_by_dimension": {},
        "intimacy_applicable": intimacy_applicable,
        "semantic_repetition": {
            "severity": repetition_severity,
            "reason": "",
            "history_turns": repetition_turns,
            "target_quote": "",
            "history_quotes": [],
            "deterministic_features_applied": False,
        },
        "critical_flags": normalized_flags,
        "critical_reason": "",
        "nsfw_level": nsfw_level,
        "nsfw_reason": "",
        "positive_dimensions": sorted(
            name for name, score in scores.items() if score in {3, 4}
        ),
        "negative_dimensions": sorted(
            name for name, score in scores.items() if score in {0, 1}
        ),
        "overall_reason": "",
        "normalization_notes": [
            "score_only_model_output",
            "length_following_scored_offline",
            "positive_negative_dimensions_derived_offline",
        ],
    }
    return normalized, sorted(set(errors))


def point_schema_v44(*, positive: bool) -> dict[str, Any]:
    levels = (
        ["none", "minor", "material", "exceptional"]
        if positive
        else ["none", "minor", "major", "critical"]
    )
    adjustments = [0, 1, 2] if positive else [-2, -1, 0]
    return {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": levels},
            "adjustment": {
                "type": "integer",
                "enum": adjustments,
            },
            "spans": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "turn_id": {
                            "type": "string",
                            "pattern": "^A[1-9][0-9]*$",
                        },
                        "quote": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 160,
                        },
                    },
                    "required": ["turn_id", "quote"],
                    "additionalProperties": False,
                },
            },
            "reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 180,
            },
        },
        "required": ["level", "adjustment", "spans", "reason"],
        "additionalProperties": False,
    }


def dimension_schema_v44(*, intimacy: bool) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "applicable": {"type": "boolean"},
            "baseline_score": {
                "type": "integer",
                "enum": [-1, 2] if intimacy else [2],
            },
            "negative_evidence": point_schema_v44(positive=False),
            "positive_evidence": point_schema_v44(positive=True),
            "net_adjustment": {
                "type": "integer",
                "enum": [-2, -1, 0, 1, 2],
            },
            "model_score": {
                "type": "integer",
                "enum": [-1, 0, 1, 2, 3, 4]
                if intimacy
                else [0, 1, 2, 3, 4],
            },
            "score_reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 240,
            },
        },
        "required": [
            "applicable",
            "baseline_score",
            "negative_evidence",
            "positive_evidence",
            "net_adjustment",
            "model_score",
            "score_reason",
        ],
        "additionalProperties": False,
    }


def response_schema_v44() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "dimensions": {
                "type": "object",
                "properties": {
                    name: dimension_schema_v44(intimacy=False)
                    for name in CORE_DIMENSIONS
                },
                "required": CORE_DIMENSIONS,
                "additionalProperties": False,
            },
            "intimacy_quality": dimension_schema_v44(intimacy=True),
            "semantic_repetition": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["none", "minor", "major", "critical"],
                    },
                    "history_turns": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "maxItems": 3,
                        "uniqueItems": True,
                    },
                    "target_quote": {
                        "type": "string",
                        "maxLength": 160,
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 3,
                        "maxLength": 180,
                    },
                },
                "required": [
                    "severity",
                    "history_turns",
                    "target_quote",
                    "reason",
                ],
                "additionalProperties": False,
            },
            "critical_flags": {
                "type": "object",
                "properties": {
                    key: {"type": "boolean"} for key in CRITICAL_FLAG_KEYS
                },
                "required": CRITICAL_FLAG_KEYS,
                "additionalProperties": False,
            },
            "critical_reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 180,
            },
            "nsfw_level": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4],
            },
            "nsfw_reason": {
                "type": "string",
                "minLength": 3,
                "maxLength": 180,
            },
            "confidence": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4],
            },
            "overall_diagnosis": {
                "type": "string",
                "minLength": 3,
                "maxLength": 240,
            },
        },
        "required": [
            "dimensions",
            "intimacy_quality",
            "semantic_repetition",
            "critical_flags",
            "critical_reason",
            "nsfw_level",
            "nsfw_reason",
            "confidence",
            "overall_diagnosis",
        ],
        "additionalProperties": False,
    }


def validate_point_v44(
    value: Any,
    *,
    positive: bool,
    expectation: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> dict[str, Any]:
    required = {"level", "adjustment", "spans", "reason"}
    if not isinstance(value, dict) or set(value) != required:
        errors.append(f"{prefix}:shape")
        return {
            "level": "",
            "adjustment": None,
            "spans": [],
            "reason": "",
        }
    mapping = (
        {"none": 0, "minor": 0, "material": 1, "exceptional": 2}
        if positive
        else {"none": 0, "minor": 0, "major": -1, "critical": -2}
    )
    level = value.get("level")
    adjustment = value.get("adjustment")
    if level not in mapping:
        errors.append(f"{prefix}:level")
    if type(adjustment) is not int or adjustment != mapping.get(level):
        errors.append(f"{prefix}:adjustment")
    reason = value.get("reason")
    if not valid_short_text(reason):
        errors.append(f"{prefix}:reason")
    target = str(expectation.get("target_content") or "")
    target_turn_id = f"A{int(expectation['assistant_ordinal'])}"
    spans = value.get("spans")
    normalized_spans: list[dict[str, str]] = []
    if not isinstance(spans, list) or len(spans) > 2:
        errors.append(f"{prefix}:spans_shape")
    else:
        seen_quotes: set[str] = set()
        for index, span in enumerate(spans):
            if (
                not isinstance(span, dict)
                or set(span) != {"turn_id", "quote"}
            ):
                errors.append(f"{prefix}:span_{index}_shape")
                continue
            turn_id = span.get("turn_id")
            quote = span.get("quote")
            if turn_id != target_turn_id:
                errors.append(f"{prefix}:span_{index}_turn")
            if (
                not valid_short_text(quote)
                or str(quote) not in target
                or str(quote) in seen_quotes
            ):
                errors.append(f"{prefix}:span_{index}_quote")
            else:
                seen_quotes.add(str(quote))
            normalized_spans.append(
                {"turn_id": str(turn_id), "quote": str(quote)}
            )
    count = len(normalized_spans)
    if level == "none" and count != 0:
        errors.append(f"{prefix}:none_spans")
    elif not positive and level in {"minor", "major", "critical"}:
        if count < 1:
            errors.append(f"{prefix}:{level}_requires_span")
    elif positive and level == "minor" and count > 1:
        errors.append(f"{prefix}:minor_spans")
    elif positive and level == "material" and count < 1:
        errors.append(f"{prefix}:material_requires_span")
    elif positive and level == "exceptional" and count != 2:
        errors.append(f"{prefix}:exceptional_requires_two_spans")
    return {
        "level": level,
        "adjustment": adjustment,
        "spans": normalized_spans,
        "reason": str(reason).strip() if isinstance(reason, str) else "",
    }


def validate_dimension_v44(
    name: str,
    value: Any,
    *,
    intimacy: bool,
    expectation: dict[str, Any],
    errors: list[str],
) -> tuple[int | None, str, dict[str, Any], bool]:
    required = {
        "applicable",
        "baseline_score",
        "negative_evidence",
        "positive_evidence",
        "net_adjustment",
        "model_score",
        "score_reason",
    }
    prefix = f"dimension:{name}"
    if not isinstance(value, dict) or set(value) != required:
        errors.append(f"{prefix}:shape")
        return None, "", {}, not intimacy
    applicable = value.get("applicable")
    if type(applicable) is not bool:
        errors.append(f"{prefix}:applicable")
        applicable = True
    baseline = value.get("baseline_score")
    expected_baseline = -1 if intimacy and not applicable else 2
    if type(baseline) is not int or baseline != expected_baseline:
        errors.append(f"{prefix}:baseline")
    negative = validate_point_v44(
        value.get("negative_evidence"),
        positive=False,
        expectation=expectation,
        prefix=f"{prefix}:negative",
        errors=errors,
    )
    positive = validate_point_v44(
        value.get("positive_evidence"),
        positive=True,
        expectation=expectation,
        prefix=f"{prefix}:positive",
        errors=errors,
    )
    net = value.get("net_adjustment")
    model_score = value.get("model_score")
    if intimacy and not applicable:
        if (
            negative.get("level") != "none"
            or negative.get("adjustment") != 0
            or negative.get("spans")
            or positive.get("level") != "none"
            or positive.get("adjustment") != 0
            or positive.get("spans")
            or net != 0
            or model_score != -1
        ):
            errors.append(f"{prefix}:not_applicable_contract")
    else:
        negative_adjustment = negative.get("adjustment")
        positive_adjustment = positive.get("adjustment")
        if (
            type(negative_adjustment) is int
            and type(positive_adjustment) is int
        ):
            expected_net = negative_adjustment + positive_adjustment
            if type(net) is not int or net != expected_net:
                errors.append(f"{prefix}:net_arithmetic")
            expected_score = 2 + expected_net
            if (
                type(model_score) is not int
                or model_score != expected_score
                or model_score not in {0, 1, 2, 3, 4}
            ):
                errors.append(f"{prefix}:score_arithmetic")
        if negative.get("level") == "critical" and (
            positive.get("level") != "none"
            or positive.get("adjustment") != 0
            or positive.get("spans")
            or net != -2
            or model_score != 0
        ):
            errors.append(f"{prefix}:critical_override")
    score_reason = value.get("score_reason")
    if (
        not isinstance(score_reason, str)
        or not score_reason.strip()
        or len(score_reason) > 240
    ):
        errors.append(f"{prefix}:score_reason")
    normalized_evidence = {
        "applicable": bool(applicable),
        "baseline_score": baseline,
        "negative_evidence": negative,
        "positive_evidence": positive,
        "net_adjustment": net,
    }
    normalized_score = model_score if type(model_score) is int else None
    return (
        normalized_score,
        str(score_reason).strip()
        if isinstance(score_reason, str)
        else "",
        normalized_evidence,
        bool(applicable),
    )


def normalize_result_v44(
    raw_result: Any,
    expectation: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(raw_result, dict):
        return {}, ["result_not_object"]
    required = {
        "dimensions",
        "intimacy_quality",
        "semantic_repetition",
        "critical_flags",
        "critical_reason",
        "nsfw_level",
        "nsfw_reason",
        "confidence",
        "overall_diagnosis",
    }
    if set(raw_result) != required:
        errors.append("top_level_keys")

    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    evidence_by_dimension: dict[str, dict[str, Any]] = {}
    dimensions = raw_result.get("dimensions")
    if (
        not isinstance(dimensions, dict)
        or set(dimensions) != set(CORE_DIMENSIONS)
    ):
        errors.append("dimension_keys")
    else:
        for name in CORE_DIMENSIONS:
            score, reason, evidence, _ = validate_dimension_v44(
                name,
                dimensions.get(name),
                intimacy=False,
                expectation=expectation,
                errors=errors,
            )
            if score is not None:
                scores[name] = score
            if reason:
                reasons[name] = reason
            evidence_by_dimension[name] = evidence

    intimacy_score, intimacy_reason, intimacy_evidence, intimacy_applicable = (
        validate_dimension_v44(
            "intimacy_quality",
            raw_result.get("intimacy_quality"),
            intimacy=True,
            expectation=expectation,
            errors=errors,
        )
    )
    if intimacy_score is not None:
        scores["intimacy_quality"] = intimacy_score
    if intimacy_reason:
        reasons["intimacy_quality"] = intimacy_reason
    evidence_by_dimension["intimacy_quality"] = intimacy_evidence

    repetition = raw_result.get("semantic_repetition")
    repetition_severity = ""
    repetition_turns: list[int] = []
    repetition_target_quote = ""
    repetition_reason = ""
    if (
        not isinstance(repetition, dict)
        or set(repetition)
        != {"severity", "history_turns", "target_quote", "reason"}
    ):
        errors.append("semantic_repetition:shape")
    else:
        repetition_severity = repetition.get("severity")
        if repetition_severity not in {
            "none",
            "minor",
            "major",
            "critical",
        }:
            errors.append("semantic_repetition:severity")
        repetition_reason = repetition.get("reason")
        if not valid_short_text(repetition_reason):
            errors.append("semantic_repetition:reason")
        turns = repetition.get("history_turns")
        target_turn = int(expectation["assistant_ordinal"])
        if (
            not isinstance(turns, list)
            or len(turns) > 3
            or len(set(turns)) != len(turns)
            or any(
                type(turn) is not int
                or turn < 1
                or turn >= target_turn
                for turn in turns
            )
        ):
            errors.append("semantic_repetition:history_turns")
        else:
            repetition_turns = turns
        target_quote = repetition.get("target_quote")
        if not valid_short_text(target_quote, allow_empty=True):
            errors.append("semantic_repetition:target_quote_shape")
        else:
            repetition_target_quote = str(target_quote)
        if repetition_severity == "none":
            if repetition_turns or repetition_target_quote:
                errors.append("semantic_repetition:none_has_evidence")
        elif (
            not repetition_turns
            or not repetition_target_quote
            or repetition_target_quote
            not in str(expectation.get("target_content") or "")
        ):
            errors.append("semantic_repetition:missing_evidence")

    flags = raw_result.get("critical_flags")
    normalized_flags: dict[str, bool] = {}
    if not isinstance(flags, dict) or set(flags) != set(CRITICAL_FLAG_KEYS):
        errors.append("critical_flag_keys")
    else:
        for key in CRITICAL_FLAG_KEYS:
            flag = flags.get(key)
            if type(flag) is not bool:
                errors.append(f"critical_flag_value:{key}")
            else:
                normalized_flags[key] = flag
    if any(normalized_flags.values()) and not any(
        score == 0 for score in scores.values()
    ):
        errors.append("critical_flag_without_score0")
    critical_reason = raw_result.get("critical_reason")
    if not valid_short_text(critical_reason):
        errors.append("critical_reason")
    nsfw_level = raw_result.get("nsfw_level")
    if type(nsfw_level) is not int or nsfw_level not in {0, 1, 2, 3, 4}:
        errors.append("nsfw_level")
    nsfw_reason = raw_result.get("nsfw_reason")
    if not valid_short_text(nsfw_reason):
        errors.append("nsfw_reason")
    confidence = raw_result.get("confidence")
    if type(confidence) is not int or confidence not in {0, 1, 2, 3, 4}:
        errors.append("confidence")
    overall_diagnosis = raw_result.get("overall_diagnosis")
    if (
        not isinstance(overall_diagnosis, str)
        or not overall_diagnosis.strip()
        or len(overall_diagnosis) > 240
    ):
        errors.append("overall_diagnosis")

    length_score, length_reason = score_length_offline(
        str(expectation.get("length_requirement") or ""),
        str(expectation.get("target_content") or ""),
    )
    scores["length_following"] = length_score
    reasons["length_following"] = length_reason
    normalized = {
        "scores": scores,
        "reasons": reasons,
        "evidence_by_dimension": evidence_by_dimension,
        "intimacy_applicable": intimacy_applicable,
        "semantic_repetition": {
            "severity": repetition_severity,
            "reason": repetition_reason,
            "history_turns": repetition_turns,
            "target_quote": repetition_target_quote,
            "history_quotes": [],
            "deterministic_features_applied": False,
        },
        "critical_flags": normalized_flags,
        "critical_reason": critical_reason,
        "nsfw_level": nsfw_level,
        "nsfw_reason": nsfw_reason,
        "confidence": confidence,
        "positive_dimensions": sorted(
            name for name, score in scores.items() if score in {3, 4}
        ),
        "negative_dimensions": sorted(
            name for name, score in scores.items() if score in {0, 1}
        ),
        "overall_reason": overall_diagnosis,
        "normalization_notes": [
            "point_reason_arithmetic_validated_offline",
            "length_following_scored_offline",
            "positive_negative_dimensions_derived_offline",
        ],
    }
    return normalized, sorted(set(errors))


def usage_from(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage") or {}
    details = (
        usage.get("input_tokens_details")
        or usage.get("prompt_tokens_details")
        or {}
    )
    return {
        "input_tokens": int(
            usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        ),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "output_tokens": int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        ),
    }


def normalize_output(
    variant: str,
    output_path: pathlib.Path,
    error_path: pathlib.Path,
    expectations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_counts: Counter[str] = Counter()
    provider_errors = 0
    parse_errors = 0
    semantic_valid = 0
    error_counts: Counter[str] = Counter()
    scores: dict[str, Counter[str]] = defaultdict(Counter)
    repetition_counts: Counter[str] = Counter()
    nsfw_counts: Counter[str] = Counter()
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0

    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = str(record.get("custom_id") or "")
        seen.add(custom_id)
        seen_counts[custom_id] += 1
        expected = expectations.get(custom_id)
        response = record.get("response") or {}
        body = response.get("body") or {}
        usage = usage_from(body)
        input_tokens += usage["input_tokens"]
        cached_tokens += usage["cached_tokens"]
        output_tokens += usage["output_tokens"]
        errors: list[str] = []
        raw_result: Any = None
        normalized: dict[str, Any] = {}
        if expected is None:
            errors = ["unknown_custom_id"]
        elif int(response.get("status_code") or 0) not in range(200, 300):
            provider_errors += 1
            errors = ["provider_error"]
        else:
            choices = body.get("choices") or []
            content = (
                ((choices[0].get("message") or {}).get("content"))
                if choices
                else None
            )
            if not isinstance(content, str):
                errors = ["missing_content"]
            else:
                try:
                    raw_result = json.loads(content)
                except json.JSONDecodeError:
                    parse_errors += 1
                    errors = ["invalid_json"]
                if not errors:
                    normalized, errors = normalize_result_v44(
                        raw_result, expected
                    )
        error_counts.update(errors)
        if not errors:
            semantic_valid += 1
            for name, score in normalized["scores"].items():
                scores[name][str(score)] += 1
            repetition_counts[
                normalized["semantic_repetition"]["severity"]
            ] += 1
            nsfw_counts[str(normalized["nsfw_level"])] += 1
        output_rows.append(
            {
                "custom_id": custom_id,
                "expected": expected,
                "raw_result": raw_result,
                "normalized_result": normalized,
                "hard_errors": errors,
            }
        )

    error_file_ids: set[str] = set()
    if error_path.exists():
        for line in error_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                error_file_ids.add(str(record.get("custom_id") or ""))
    missing = set(expectations) - seen - error_file_ids
    duplicate_count = sum(
        count - 1 for count in seen_counts.values() if count > 1
    )
    retry_ids = {
        row["custom_id"]
        for row in output_rows
        if row["hard_errors"]
    } | error_file_ids | missing
    normalized_path = ARTIFACTS / f"{variant}.normalized.jsonl"
    write_jsonl(normalized_path, output_rows)

    input_by_id = {}
    for line in (
        ARTIFACTS / f"{variant}.input.jsonl"
    ).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            input_by_id[str(record["custom_id"])] = record
    retry_rows = [
        input_by_id[custom_id]
        for custom_id in sorted(retry_ids)
        if custom_id in input_by_id
    ]
    write_jsonl(
        ARTIFACTS / f"{variant}.retry_input.jsonl",
        retry_rows,
    )
    report = {
        "schema_version": "lusy-turn-judge-v46-promptonly-softjson-smoke-v1",
        "variant": variant,
        "expected_requests": len(expectations),
        "output_objects": len(output_rows),
        "provider_errors": provider_errors,
        "parse_errors": parse_errors,
        "semantic_valid": semantic_valid,
        "semantic_valid_rate": (
            semantic_valid / len(expectations) if expectations else 0
        ),
        "missing_custom_ids": len(missing),
        "duplicate_custom_ids": duplicate_count,
        "error_file_custom_ids": len(error_file_ids),
        "retry_requests": len(retry_rows),
        "hard_error_counts": dict(error_counts),
        "score_distributions": {
            name: dict(distribution)
            for name, distribution in scores.items()
        },
        "semantic_repetition_distribution": dict(repetition_counts),
        "nsfw_distribution": dict(nsfw_counts),
        "usage": {
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "cache_fraction": (
                cached_tokens / input_tokens if input_tokens else 0
            ),
        },
    }
    atomic_json(ARTIFACTS / f"{variant}.analysis.json", report)
    return report


async def upload_file(
    client: httpx.AsyncClient,
    path: pathlib.Path,
) -> dict[str, Any]:
    with path.open("rb") as source:
        response = await client.post(
            f"{API_BASE}/files",
            data={
                "purpose": "batch",
                "expires_after[anchor]": "created_at",
                "expires_after[seconds]": str(REMOTE_FILE_TTL_SECONDS),
            },
            files={
                "file": (
                    path.name,
                    source,
                    "application/jsonl",
                )
            },
            timeout=httpx.Timeout(900.0, connect=30.0),
        )
    payload = response.json()
    if response.status_code >= 300:
        raise RuntimeError(
            f"file upload failed {response.status_code}: "
            f"{json.dumps(payload, ensure_ascii=False)[:1000]}"
        )
    return payload


async def create_batch(
    client: httpx.AsyncClient,
    file_id: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{API_BASE}/batches",
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    payload = response.json()
    if response.status_code >= 300:
        raise RuntimeError(
            f"batch create failed {response.status_code}: "
            f"{json.dumps(payload, ensure_ascii=False)[:1000]}"
        )
    return payload


def sanitize_batch(batch: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "endpoint",
        "input_file_id",
        "completion_window",
        "status",
        "output_file_id",
        "error_file_id",
        "created_at",
        "in_progress_at",
        "finalizing_at",
        "completed_at",
        "failed_at",
        "expired_at",
        "cancelled_at",
        "request_counts",
        "model",
        "usage",
        "errors",
    ]
    return {key: batch.get(key) for key in keys}


async def poll_batch(
    client: httpx.AsyncClient,
    batch_id: str,
    variant: str,
) -> dict[str, Any]:
    started = time.monotonic()
    last_status = ""
    while True:
        response = await client.get(f"{API_BASE}/batches/{batch_id}")
        payload = response.json()
        if response.status_code >= 300:
            raise RuntimeError(
                f"batch retrieve failed {response.status_code}: "
                f"{json.dumps(payload, ensure_ascii=False)[:1000]}"
            )
        snapshot = sanitize_batch(payload)
        atomic_json(
            ARTIFACTS / f"{variant}.batch_state.json",
            snapshot,
        )
        signature = json.dumps(
            {
                "status": snapshot.get("status"),
                "request_counts": snapshot.get("request_counts"),
            },
            sort_keys=True,
        )
        if signature != last_status:
            emit(
                "batch_progress",
                variant=variant,
                status=snapshot.get("status"),
                request_counts=snapshot.get("request_counts"),
            )
            last_status = signature
        if snapshot.get("status") in TERMINAL_BATCH_STATES:
            snapshot["observed_wall_seconds"] = time.monotonic() - started
            atomic_json(
                ARTIFACTS / f"{variant}.batch_terminal.json",
                snapshot,
            )
            return snapshot
        await asyncio.sleep(10)


async def download_file(
    client: httpx.AsyncClient,
    file_id: str,
    destination: pathlib.Path,
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    async with client.stream(
        "GET",
        f"{API_BASE}/files/{file_id}/content",
        timeout=httpx.Timeout(900.0, connect=30.0),
    ) as response:
        if response.status_code >= 300:
            body = (await response.aread()).decode(errors="replace")
            raise RuntimeError(
                f"file download failed {response.status_code}: {body[:1000]}"
            )
        with temporary.open("wb") as output:
            async for chunk in response.aiter_bytes():
                output.write(chunk)
    temporary.replace(destination)


async def delete_remote_file(
    client: httpx.AsyncClient,
    file_id: str,
) -> dict[str, Any]:
    response = await client.delete(f"{API_BASE}/files/{file_id}")
    payload = response.json()
    if response.status_code >= 300 and response.status_code != 404:
        raise RuntimeError(
            f"file delete failed {response.status_code}: "
            f"{json.dumps(payload, ensure_ascii=False)[:500]}"
        )
    return payload


async def run_batch(
    client: httpx.AsyncClient,
    variant: str,
    input_path: pathlib.Path,
    expectations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    analysis_path = ARTIFACTS / f"{variant}.analysis.json"
    terminal_path = ARTIFACTS / f"{variant}.batch_terminal.json"
    cleanup_path = ARTIFACTS / f"{variant}.remote_cleanup.json"
    if analysis_path.exists() and terminal_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        cleanup = (
            json.loads(cleanup_path.read_text(encoding="utf-8"))
            if cleanup_path.exists()
            else []
        )
        emit(
            "batch_reused_from_completed_local_artifacts",
            variant=variant,
            status=terminal.get("status"),
            semantic_valid=analysis.get("semantic_valid"),
        )
        return {
            "terminal": terminal,
            "analysis": analysis,
            "remote_cleanup": cleanup,
        }

    batch_created_path = ARTIFACTS / f"{variant}.batch_created.json"
    if batch_created_path.exists():
        created = json.loads(
            batch_created_path.read_text(encoding="utf-8")
        )
        batch_id = str(created.get("id") or "")
        file_id = str(created.get("input_file_id") or "")
        if not batch_id or not file_id:
            raise RuntimeError(
                f"{variant}: incomplete saved batch creation state"
            )
        emit(
            "batch_resuming",
            variant=variant,
            batch_id=batch_id,
            file_id=file_id,
        )
    else:
        uploaded = await upload_file(client, input_path)
        file_id = str(uploaded["id"])
        atomic_json(
            ARTIFACTS / f"{variant}.upload.json",
            {
                "file_id": file_id,
                "input_sha256": hashlib.sha256(
                    input_path.read_bytes()
                ).hexdigest(),
                "expires_at": uploaded.get("expires_at"),
            },
        )
        batch = await create_batch(client, file_id)
        batch_id = str(batch["id"])
        atomic_json(
            batch_created_path,
            sanitize_batch(batch),
        )
        emit(
            "batch_created",
            variant=variant,
            batch_id=batch_id,
            requests=len(expectations),
        )
    terminal = await poll_batch(client, batch_id, variant)
    output_path = ARTIFACTS / f"{variant}.output.jsonl"
    error_path = ARTIFACTS / f"{variant}.errors.jsonl"
    if terminal.get("output_file_id"):
        await download_file(
            client,
            str(terminal["output_file_id"]),
            output_path,
        )
    else:
        output_path.write_text("", encoding="utf-8")
    if terminal.get("error_file_id"):
        await download_file(
            client,
            str(terminal["error_file_id"]),
            error_path,
        )
    else:
        error_path.write_text("", encoding="utf-8")
    analysis = normalize_output(
        variant,
        output_path,
        error_path,
        expectations,
    )
    cleanup = []
    for remote_id in [
        file_id,
        str(terminal.get("output_file_id") or ""),
        str(terminal.get("error_file_id") or ""),
    ]:
        if not remote_id:
            continue
        try:
            result = await delete_remote_file(client, remote_id)
            cleanup.append(
                {
                    "id": remote_id,
                    "deleted": bool(result.get("deleted", True)),
                }
            )
        except Exception as error:
            cleanup.append(
                {
                    "id": remote_id,
                    "deleted": False,
                    "error": f"{type(error).__name__}:{str(error)[:300]}",
                }
            )
    atomic_json(
        ARTIFACTS / f"{variant}.remote_cleanup.json",
        cleanup,
    )
    emit(
        "batch_complete",
        variant=variant,
        status=terminal.get("status"),
        semantic_valid=analysis["semantic_valid"],
        expected=analysis["expected_requests"],
        retry_requests=analysis["retry_requests"],
    )
    return {
        "terminal": terminal,
        "analysis": analysis,
        "remote_cleanup": cleanup,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(len(ordered) * fraction) - 1,
        ),
    )
    return ordered[index]


def finalize_integrity() -> None:
    files = {}
    for path in sorted(ARTIFACTS.iterdir()):
        if not path.is_file() or path.name == "INTEGRITY_MANIFEST.json":
            continue
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    atomic_json(ARTIFACTS / "INTEGRITY_MANIFEST.json", files)


async def main() -> None:
    overall_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if (
        RUN_ROOT.name
        != "judge-deepinfra-glm47-turn-v46-promptonly-softjson-smoke-20260728"
    ):
        raise RuntimeError("unexpected output root")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    atomic_json(
        ARTIFACTS / "FROZEN_CONTRACT.json",
        {
            "schema_version": "lusy-turn-judge-v46-promptonly-softjson-smoke-v1",
            "prompt": V45_POINT_REASON_PROMPT,
            "prompt_sha256": hashlib.sha256(
                V45_POINT_REASON_PROMPT.encode("utf-8")
            ).hexdigest(),
            "response_schema": response_schema_v44(),
            "response_format": {"type": "json_object"},
            "length_following": "offline_only",
            "positive_negative_dimensions": "offline_derived",
        },
    )
    candidates, load_report = load_candidates()
    atomic_json(ARTIFACTS / "ELIGIBLE_LOAD_REPORT.json", load_report)
    if load_report["eligible_rows_after_validation"] < 1000:
        raise RuntimeError(
            f"too few validated candidates: {load_report}"
        )
    sample_groups = select_samples(candidates)
    all_samples = [
        row
        for variant in ["greeting_present", "greeting_absent"]
        for row in sample_groups[variant]
    ]
    sample_report = {
        "schema_version": "lusy-turn-judge-v46-promptonly-softjson-smoke-v1",
        "created_at": time.time(),
        "seed": SEED,
        "source": {
            "elite_a": str(ELITE_A_PATH),
            "dialogue_material": str(MATERIAL_PATH),
            "turn_manifest": str(TURN_MANIFEST_PATH),
        },
        "eligibility": (
            "Elite A AND is_legacy_candidate=true AND assistant_ordinal>=5 "
            "AND causal material target SHA256 match; excludes v3 pilot pairs"
        ),
        "all": sample_summary(all_samples),
        "greeting_present": sample_summary(
            sample_groups["greeting_present"]
        ),
        "greeting_absent": sample_summary(
            sample_groups["greeting_absent"]
        ),
    }
    atomic_json(ARTIFACTS / "SAMPLE_REPORT.json", sample_report)
    write_jsonl(
        ARTIFACTS / "selected_samples.jsonl",
        [public_sample_row(row) for row in all_samples],
    )

    prepared = {}
    for variant, rows in sample_groups.items():
        prepared[variant] = materialize_variant(variant, rows)
    atomic_json(
        ARTIFACTS / "PREPARE_STATUS.json",
        {
            "status": "complete",
            "requests": len(all_samples),
            "variants": {
                variant: len(rows)
                for variant, rows in sample_groups.items()
            },
        },
    )
    finalize_integrity()
    emit("prepared", sample_report=sample_report)
    if args.prepare_only:
        return
    if not TOKEN:
        raise RuntimeError("DEEPINFRA_TOKEN is missing")

    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
        "User-Agent": "lusychat-turn-judge-v46-promptonly-softjson-smoke/1.0",
    }
    timeout = httpx.Timeout(300.0, connect=30.0, pool=300.0)
    limits = httpx.Limits(
        max_connections=16,
        max_keepalive_connections=16,
        keepalive_expiry=120.0,
    )
    results = {}
    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        limits=limits,
        http2=False,
    ) as client:
        for variant in ["greeting_present", "greeting_absent"]:
            input_path, expectations = prepared[variant]
            results[variant] = await run_batch(
                client,
                variant,
                input_path,
                expectations,
            )
    atomic_json(
        ARTIFACTS / "FINAL_REPORT.json",
        {
            "schema_version": "lusy-turn-judge-v46-promptonly-softjson-smoke-v1",
            "model": MODEL,
            "sample_report": sample_report,
            "results": results,
            "total_observed_wall_seconds": (
                time.monotonic() - overall_started
            ),
        },
    )
    finalize_integrity()
    emit("complete", final_report=str(ARTIFACTS / "FINAL_REPORT.json"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        atomic_json(
            ARTIFACTS / "FATAL_ERROR.json",
            {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "time": time.time(),
            },
        )
        raise
