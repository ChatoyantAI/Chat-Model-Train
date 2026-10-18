# CoSER / HER alignment audit: corrections to interpretation

This read-only audit compares the executed paired runs, actual training examples,
and upstream HER commit `1e5fd856b9e23d3af41445e3c4ca734e24e012ef`. No new model
calls were sent and no evaluation source or historical result was changed.

## Finding

The current runner is a CoSER worker with selected HER format and judging
adaptations. It is not a complete reproduction of HER's CoSER implementation.
Using the same role tags does not establish matching multi-turn inputs.

| Layer | Executed paired protocol | HER upstream / training evidence |
| --- | --- | --- |
| Dataset | Pinned CoSER original 200 scenes | HER runner also contains adaptations for HER-format reference data; exact dataset equivalence was not established in this audit |
| Role output instruction | CoSER format block replaced with a HER-style no-system instruction | Actual HER `get_character_prompt` includes additional single-character, single-turn constraints |
| Length instruction | CoSER instruction: at most 60 words | HER `her_nosys` prompt: approximately 200 words |
| Own assistant history | Converted to square-bracket thoughts and parenthesized actions before the next actor call | HER benchmark keeps its own original response; actual training rows retain role tags |
| Other actors' history | Inner thoughts hidden; actions already converted to parentheses | HER benchmark removes inner thoughts while retaining role-action markup |
| Judge input | Converted CoSER representation, retaining role thoughts | HER also converts the judging view; this does not require converting the actor's own history |
| System thinking | Removed from instruction by user request; native thinking false | Intentional no-system variant, not reproduction of the system-thinking HER model |

The training example at row 1 of
`/workspace/data/her-single-no-system-5090-v1/train.jsonl` contains an earlier
assistant response by Dr. Roy Michaels with `<role_action>` and `<role_thinking>`.
Its user history retains `<role_action>`. Its no-system output instruction matches
our new format block, but matching that block alone does not align the whole
request.

A real repeated-run actor request is saved at:

`/workspace/runs/roleplay-evaluation/qwen3-0.6b-her-single-no-system-27000-coser-pair-repeat2-20260923/transport-journal/df/dff09934df33450dd8851f7eb7514f32c953b65e2c624b900a9111577156bc79/795aacb00fbb46a486ea4ebd8d238572.outcome.json`

Its instruction requests HER tags, while its preceding assistant messages contain
`Josh Bennett: [The words feel like stones in my throat...](stares...)` and
`Josh Bennett: [It makes no sense...](paces...)`, confirming the conversion occurs
before generation of the next turn, not only when preparing judge input.

## Upstream evidence

- Actual prompt constructor, including the `her_nosys` branch, single-role/turn
  constraints, and 200-word guidance:
  https://github.com/cydu24/HER/blob/1e5fd856b9e23d3af41445e3c4ca734e24e012ef/eval_code/benchmarks/multi_turn/coser/prompts.py#L6
- Actual benchmark invokes `get_character_prompt` from that module:
  https://github.com/cydu24/HER/blob/1e5fd856b9e23d3af41445e3c4ca734e24e012ef/eval_code/benchmarks/multi_turn/coser/benchmark.py#L431
- Own history preserves the response, other-character history hides thoughts:
  https://github.com/cydu24/HER/blob/1e5fd856b9e23d3af41445e3c4ca734e24e012ef/eval_code/benchmarks/multi_turn/coser/benchmark.py#L630
- Conversion to CoSER representation occurs in the judging view:
  https://github.com/cydu24/HER/blob/1e5fd856b9e23d3af41445e3c4ca734e24e012ef/eval_code/benchmarks/multi_turn/coser/benchmark.py#L704

Our earlier adapter copied the simpler output instruction from HER's utils helper.
That helper must not be treated as the complete prompt actually constructed by
the benchmark. The previous label `her-official-v1` overstates alignment.

## What this changes about the conclusion

The saved trajectories still demonstrate real contradictions, state drift and
speaker confusion in generated content. However, these observations do not
isolate a training failure: the model was also exposed to a different history
representation from its training data. The independent repeats measure
variability of this same adapted protocol, not robustness of the official HER
protocol. The scoring bias favoring repetitive low-content trajectories remains
a separate issue.

The next controlled validation should retain HER-formatted actor histories,
separate self/other/judge views, and explicitly choose the role prompt and length
instruction. Keep dataset, weights, judge and sampling fixed, preserve existing
results, and label any changed protocol separately. First compare a small fixed
set of scenes; neither the size nor direction of a score change has been measured
yet. Restoring system thinking is not required to fix the role-history mismatch.
