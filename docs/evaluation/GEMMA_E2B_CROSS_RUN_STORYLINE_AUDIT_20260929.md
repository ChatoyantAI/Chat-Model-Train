# Gemma E2B: why Storyline Quality keeps declining

Read-only audit on 2026-09-29. No new generation, paid judging, training, serving,
sampling, benchmark, or scoring changes were made. Evidence includes five saved
CoSER runs (999 successful scene results), actual transport requests, the pinned
worker implementation, and all 325,142 final training targets. Six paired full
trajectories were exported for close reading; critiques are observations by GLM,
not independently certified ground truth.

## Findings

The repeatable measured deficit is mostly additional logical contradictions,
not a general increase in repetition. Full fine-tuning improves several metrics
over LoRA but does not materially improve narrative coherence in this setup.
The observed behaviour is a mixture of real content errors, training-to-rollout
distribution differences, and a comparison that changes more than model weights.
There is no experiment here that assigns a causal percentage to each source.

All numbers below use the same 199 scenes, excluding failed case 67 in the new
full fine-tuning run. Previous runs are recomputed on that subset.

| Run | Average | Storyline Consistency | Anthropomorphism | Character Fidelity | Storyline Quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| IT repeat2, Sep 26 | 31.57 | 29.69 | 27.05 | 28.35 | 41.18 |
| IT fresh pair, Sep 27 | 31.56 | 29.69 | 26.84 | 29.22 | 40.49 |
| LoRA epoch 1 repeat2 | 27.65 | 26.94 | 27.59 | 23.75 | 32.30 |
| LoRA epoch 2 fresh pair | 27.92 | 25.88 | 28.11 | 25.07 | 32.61 |
| Full epoch 1, Sep 29 | 29.67 | 29.26 | 29.69 | 26.88 | 32.87 |

The companion JSON is authoritative for unrounded values. Full epoch 1 is lower
than IT on Storyline Quality in 121 scenes, higher in 74, tied in 4. A paired
scene bootstrap gives a mean difference -7.63, approximate 95% interval
[-11.25, -4.32]. This is conditional on these generated trajectories and judge
responses; it does not estimate judge/model rerun variance or remove protocol
confounding. Qwen's previously recorded no-repetition-penalty run was nearly
level with its IT comparison on this metric, so the deficit is not a universal
law of fine-tuning across every model we tested.

## What produces the numerical deficit

Original GLM Storyline Quality flaw types were grouped by their explicit type
labels. Variants of Logical Consistency and Flow & Progression are combined;
one mixed item in the full run is kept separate.

| Matched 199 scenes | IT | LoRA epoch 1 | LoRA epoch 2 | Full epoch 1 |
| --- | ---: | ---: | ---: | ---: |
| Logical consistency flaws | 323 | 521 | 531 | 562 |
| Flow/progression/redundancy flaws | 656 | 610 | 623 | 597 |
| Logical severity sum | 1,038 | 1,855 | 1,873 | 1,944 |
| Mean actor turns per scene | 12.66 | 17.17 | 16.73 | 17.14 |
| Mean words per actor turn | 43.05 | 28.80 | 28.76 | 30.01 |
| Estimated spoken words per actor turn | 11.58 | 7.27 | 8.11 | 8.19 |
| Estimated turns without speech | 3.73% | 13.50% | 13.04% | 11.05% |
| Scenes reaching the 20-turn simulator cap | 94 | 163 | 152 | 168 |
| Storyline Quality zero-score scenes | 7 | 27 | 29 | 23 |

Word/speech extraction is a regex heuristic: remove speaker and Speech labels,
bracketed thoughts, parenthesized/asterisk actions, then count words. A speechless
turn can be a legitimate meaningful action. These counts are not an independent
semantic quality judgment. Exact repeated speech is only 1.75% for IT, 2.14% for
full, and 1.56% for LoRA epoch 2; semantic repetition can escape exact matching.
This evidence does not support attributing the bulk of the gap to repetition.

Longer trajectories create more opportunities for errors. Even after a rough
normalization by actor turns, GLM marks 16.48 logical flaws per 100 full-model
turns versus 12.82 for IT. A flaw can span several turns, so this is an exposure
normalization, not a per-turn error probability.

The unchanged formula is
`clamp(100 - 5 * sum(integer severities) + 1.5 * actor_turns, 0, 100)`.
Before clipping, the full-minus-IT arithmetic decomposes into:

- Extra logical severities: -22.76 points per scene.
- Fewer flow/redundancy severities: +6.68 points.
- Extra actor turns: +6.72 points.
- One mixed flaw: -0.10 points.
- Different clipping effects: +1.83 points, yielding the observed -7.63.

This explains the scores mechanically; it is not a causal decomposition of
training. More turns actually receive a bonus, so a simple claim that long
dialogues are automatically scored lower is incorrect.

## Close reading: errors and improvements

### Case 196, A Game of Thrones: IT 95.5, full 15.0

IT keeps a broadly connected assessment of Jon, duty, and following Tormund,
though the prose is formulaic and 95.5 should not mean near-perfect literature.
The full model shifts a question about reaching the Wall into an unsupported
Winterfell massacre. Asked `Were they killed?`, it replies `No, they were killed`.
Later `bellyful of round pies` becomes `We'll be pies and pie crust soon enough`.
These are visible contradictions and broken semantic transitions; they do not
depend on removing any thought tags or exact reconstruction of the novel.

### Case 127, The Dark Tower: IT 81.0, full 19.0

The full model introduces wyverns and ultimately pushes "Patrick's chair" even
though Susannah is the wheelchair user and Patrick has just stood up. It repeats
the urgent need to leave while the world simulator escalates a supernatural
threat. IT also has generic escalating atmosphere and repetitions, but does not
make the same chair ownership error. The environment is generated by GLM, so
its invented events must not all be attributed to Gemma.

The judge also penalizes departure from the reference barn plan. That is a
separate issue: the current actor prompt does not contain the barn plan (see
the omitted thought/motivation field below). Such critique cannot automatically
be treated as failure to follow information the actor was actually given.

### Case 18, The Wheel of Time

In the previous LoRA epoch 2 snapshot, the last eight actor turns repeated
`We're here.` while actions changed. In the new full run, that exact loop is
gone: characters decide to leave, but introduce different issues, including
Faile thinking `That's the woman I fell in love with` about Perrin. A stable
average deficit does not imply the same exact error survived every checkpoint.

### Case 46, The Idiot: IT 53.5, full 88.5

IT cycles through storm/light/dawn metaphors and repeated comfort without a
concrete decision. Full asks whether the prince truly cares, tests his acceptance,
then leaves after continued reassurance. It still has weaknesses but the turns
are more connected to the relationship conflict. Fine-tuning does improve some
trajectories; dismissing the entire model would ignore these examples.

### Case 93, The Count of Monte Cristo: judge calibration warning

Full receives 85 despite gender/name confusion and saying someone is dead yet
can be saved from grief. The judge assigns severities 3+2+2+2 = 9; 20 actor
turns add 30, so `100 - 45 + 30 = 85`. High scores can coexist with glaring
errors because of flaw enumeration, severity judgment and the round bonus.
Conversely, a lower score cannot be trusted without reading its critique.

## Evaluation differences that prevent a weights-only conclusion

1. IT uses the native CoSER role prompt, explicit 60-word instruction and its
   formatting examples. HER uses the official single-turn critical rules,
   approximately 200 words, and role tags. Both actually generate much less
   than the token cap; the longer allowed length does not explain truncation.
2. IT merges consecutive messages in the frozen adapter. HER preserves them,
   retains role tags in self history, changes NSP prompt rules/parsing, and
   retains only the parsed next speaker in NSP self history. The environment
   prompt text has the same substantive requirements, but histories and
   scheduling differ. Full has fewer environment turns and reaches the 20-turn
   cap much more often. The resulting trajectory is not controlled solely by
   the tested model's weights.
3. The original native Agent calls `fix_repeation` on eligible non-NSP responses;
   the HER Agent does not. Its actual impact was not counted here. This is
   another code difference, not grounds to silently add a repetition penalty.
4. HER judge conversion adds speaker labels. IT's native judge serialization
   simply joins response contents; only 750/2,519 matched actor turns include
   an explicit matching `Name:` prefix, compared with 3,411/3,411 for full.
   Many IT turns leave the judge to infer the speaker. This can affect error
   attribution; the direction and size require rescoring the same trajectories.
5. Both judge views retain role thoughts. HER removes only system/native
   analysis and converts role tags to brackets/parentheses. Storyline Quality
   rubric text is identical, but complete judge prompts/JSON examples are not
   byte-identical. The premise that IT thoughts are hidden but HER thoughts
   are scored is false for these runs.

All 3,430 full-model requests and 2,528 IT requests explicitly set native
thinking false, supply no repetition penalty, and finish with `stop`, not
`length`. Both actor endpoints use temperature 0.7 and max_tokens 16384;
GLM uses temperature 0 and max_tokens 8192. No evidence points to token
truncation, FP8, file write speed, or HTTP errors as the score-gap explanation.
These Gemma E2B services are BF16. Failed case 67 is excluded from both sides.

## Newly verified missing context field

In all 200 locked test scenes, the 456 `key_characters` entries use `thought`;
none has a nonempty `motivation`. The simulation loop nevertheless looks up
`c.get('motivation', '')`. Both current protocols therefore omit this optional
inner-state block. Actual actor payloads and copied run data confirm the
mismatch, including case 127 (no inner-thought heading, no barn plan).

The pinned native implementation already contains this lookup; it was not
introduced by the new full-model config. Correcting it would be a new evaluation
variant that supplies more input, not an invisible repair to old scores.
Keep existing results frozen and label an ablation explicitly. It applies to
both models, so it alone does not prove why HER loses more, but HER training
explicitly contains `Recent Experience` and `Your Inner State`, unlike the
actual evaluation prompts. Unequal sensitivity to missing scaffolding is a
testable hypothesis.

## What the actual training data establishes

The full-model script loads the same tokenized source as the LoRA experiment:
`/workspace/training/gemma4-e2b-her-single-unsloth-b2-v1/data/train.arrow`.
Source text is `/workspace/data/her-single-no-system-5090-v1/train.jsonl`, from
HER official `HER_dataset_sft_single_turn.jsonl`, pinned revision
`fe15660adb112cdd4420d205c10f9f42d8d69c80`.

- 325,142 training examples, 27,617 source groups; last assistant + turn
  terminator loss, native thinking false, removed system-thinking blocks.
- 77.54% contain earlier assistant messages; mean 3.42 assistant messages,
  maximum 35. These are cumulative-history next-reply targets, not 325k
  independent one-line dialogues with no preceding conversation.
- Final targets average 40.10 words and about 15.27 estimated spoken words;
  7.55% have no speech by the extraction heuristic. The new free-running
  model averages only 30.01 total / 8.19 spoken words and 11.05% speechless.
- Mean supervised length is about 77.3 Gemma tokens; actual maximum input is
  4,884 tokens with no truncation. Configured 64K is capacity, not typical
  training sample length.
- Recorded full training uses fresh IT weights, active text full fine-tuning,
  learning rate 5e-5, global batch 32, and epoch-1 checkpoint 10161. The
  training script verifies the collator preserves the stored completion mask
  on ten preflight examples. This audit did not recompute every token label.

The first actual target already contains only a physical reaction and private
thought after another speaker's speech. This is legitimate literary training
data, but overproducing such turns during self-play can limit public exchange.
The generated speech deficit is larger than the training-target deficit, so
"the dataset is all short bad replies" is not supported either.

Training optimizes next replies given supplied correct histories; evaluation
feeds the model's own generated errors back through both characters and a world
simulator. Learning stylistic and emotional patterns without equally robust
state tracking is the leading behavioural hypothesis. It is not yet proven
that the dataset, learning rate, missing system reasoning, or loss policy is
the causal root. Switching to all-assistant loss blindly can reweight repeated
cumulative prefixes; it is not automatically more multi-turn training.

## Small controlled experiments to distinguish causes

1. Reuse saved trajectories: explicitly label every speaker on both sides and
   submit the same canonical judge prompt. This isolates presentation/judge
   effects without re-generating dialogues. Retain original scores separately.
2. Use a fixed, stratified development subset and a 2x2 matrix of IT/full weights
   versus CoSER/HER actor format. Hold simulator/NSP/history/length policy and
   canonical judge view fixed, so the variable is the actor format rather than
   swapping the entire official protocol. Repeat with fixed seeds and report
   scene-paired differences, not four unrelated averages.
3. Separately test the `thought` context mapping on/off for both weights, with
   an explicitly named protocol variant. Do not rewrite original benchmark
   results or claim the variant is untouched official evaluation.
4. Compare next-reply behaviour on the same fixed good history with free rollout.
   Track the first identity, knowledge, location or action-state error, not just
   the final score. This tests teacher-forcing versus self-generated-history
   exposure before deciding to change training data or loss.

No new inference is launched by this audit. A prompt/normalization fix should
not be used to erase observed content errors or to optimize only this test set.

## Evidence locations

- Aggregate statistics: sibling `GEMMA_E2B_CROSS_RUN_STORYLINE_AUDIT_20260929.json`.
- Local five-run result export and reproducible analysis:
  `.local/storyline-audit-20260929/{results.json,analyze.py,statistics.json}`.
- Complete six-case reading export:
  `.local/storyline-audit-20260929/complete-case-comparisons.md`.
- Actual wire samples and all-target training statistics:
  `.local/storyline-audit-20260929/{request-audit.json,training-statistics.json,training-samples.json}`.
- Remote exports: `/workspace/projects/public-evaluation/.local/storyline-audit-20260929/`.
- Original runs: `/workspace/runs/roleplay-evaluation/`, names in `results.json`.
- Relevant source: `her_official.py`, `her_coser.py`, `her_upstream/prompts.py`,
  and pinned CoSER `gca_evaluation/{main.py,agent.py,utils.py}`.
