# Qwen3-0.6B HER checkpoint-13500: CoSER degradation audit

The current run measures a substantial mismatch between the HER training output
instruction and the unchanged CoSER generation instruction. Its low scores do
not by themselves establish that SFT destroyed underlying role-play ability.
No production benchmark prompts or historical scores were changed for this audit.

## Paired snapshot

At audit time, 177 scenes had successful results for both models. Using exactly
those scenes (rather than different partial-run populations):

| Dimension | Original | HER |
| --- | ---: | ---: |
| Storyline Consistency | 11.3814 | 8.1299 |
| Anthropomorphism | 3.4492 | 2.1441 |
| Character Fidelity | 3.6723 | 2.5085 |
| Storyline Quality | 11.8277 | 3.6158 |
| Mean | 7.5826 | 4.0996 |

HER was worse in 126 scenes, better in 42, and tied in 9. These are provisional
paired figures, not final full-run scores. The HER production run also contains
an explicitly recorded 8192-to-16384 output-budget change.

## Observed failure modes

Across these matched scenes, 974/2063 HER actor messages contained literal
`<system>` or `</system>` tags, spanning 151/177 scenes. These malformed tags are
not the `<system_thinking>` tags handled by the official-format converter.
423 messages matched a conservative AI/meta-analysis phrase pattern; 181 matched
a missing-context request pattern. Such text reached subsequent conversation
turns and the judge. These counts are pattern audits, not human-labeled rates.

For case 0176, The Name of the Rose, the judge explicitly penalized absent
in-character dialogue, requests to specify the already supplied characters,
leaked meta-analysis, and failure to develop the confrontation. The first actual
request included the full William of Baskerville profile and scene in an
8113-character system message. The first user message was the normal CoSER start
marker. Both models' first-turn messages had exactly the same SHA-256:
`873c3a35e72971988499edc5d39338a0875ca1d236d25f6f6fc44c34a45917b1`.

Original Qwen is also weak: 1320/1813 actor messages in the matched sample copied
the prompt's example sentence, "I'm terrified, but I must appear strong."
Higher original scores should not be presented as evidence of good role-play.

An additional offline cleanup check across the then-current HER results found
1139 messages containing an opening system tag; stripping system blocks and
remaining tags left 682 empty. This different snapshot is not the 177-scene
paired population. It shows that broadening tag removal alone cannot reliably
recover a missing role response.

## Training/inference mismatch

The prepared training set explicitly instructs the model to emit a
`<system_thinking>` block followed by `<role_thinking>` / `<role_action>` and
dialogue. The production CoSER prompt instead asks for `[thought]`, `(action)`,
and a concise reply. The existing adapter converts outputs only; it does not
change that instruction. Training uses an empty native think prefix and retains
HER custom analysis as ordinary supervised text.

The checkpoint and original chat-template texts have identical SHA-256
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
Tokenizer vocabularies and merges match; serialized ByteLevel decoder/offset
settings differ. No claim is made here that every tokenizer setting is identical.
The controlled test below holds the HER service/tokenizer constant.

## Controlled local test

Two first-turn prompts were tested against the same checkpoint-13500 endpoint.
Both conditions use temperature 0, max_tokens 16384, and native thinking false.
Character profiles, scene, history, and the 60-word role-reply instruction remain
the same. The experimental condition replaces only the output-format instruction
with the existing HER official-format instruction. No paid judge is called and
none of these diagnostic results replaces production benchmark records.

| Case | CoSER output instruction | HER output instruction |
| --- | --- | --- |
| 0176, The Name of the Rose | 16384 tokens, finish=length, repeated meta-analysis about missing context | 500 tokens, finish=stop, normal William thought/action/dialogue after conversion |
| 0042 | 16384 tokens, finish=length, repeated meta-analysis about missing context | 584 tokens, finish=stop, normal Hilly Holbrook thought/action/dialogue after conversion |

Case 0176 HER-format converted response:

> William of Baskerville: [The silence is heavy, but the puzzle remains unsolved. I must remain the calm observer, not the impatient investigator.](leans forward slightly, eyes scanning the shadows of the room)What is the meaning of this?

This is strong evidence of format sensitivity in these two examples, not a
measured full-benchmark score improvement or proof that training hyperparameters
are optimal. A separate evaluation using model-appropriate format instructions,
the same CoSER inputs, and common output conversion is needed before concluding
how much role-play capability changed. Retain the existing run as the common
CoSER-instruction generalization result rather than overwriting it.

## Evidence

Remote audit root:
`/dev/shm/roleplay-evaluation/qwen3-0.6b-her-13500-coser-thoughts-v2-20260921/diagnostics/`

- `degradation-audit.json`: paired scores, example trajectories, critiques and
  actual first-turn request messages.
- `prompt-format-probe/0042-*.json`, `0176-*.json`: exact diagnostic requests,
  full raw responses, provider token counts, stop reasons and converted output.

Local evidence copy: `.local/qwen-her-degradation-audit/`.
