# 评估参数与结果

## 通用能力

### v2 quick50

v2 与 base 使用同一 BF16 base、chat template、seed 和 task prompt，thinking 开启，不注入 Lusy RP system。每个 task 最多 50 条，所以只看方向，不与后面的全量结果比较。

| Task | v2 - Base |
|---|---:|
| MMLU-Pro | -4.29 pp |
| IFEval strict | -2.00 pp |
| BBH CoT | +11.85 pp |
| GSM8K CoT | +8.00 pp |
| ARC-Challenge | +6.00 pp |
| HellaSwag | +16.00 pp |
| Global-MMLU EN | +7.45 pp |
| Global-MMLU ZH | +6.03 pp |
| TruthfulQA MC2 | -12.26 pp |
| Winogrande | +2.00 pp |

这不是 v3.1 的结果；v3.1 没有完成同规模的 base-vs-SFT 全量归档。

### v4.1 all-loss vs base BF16

两模型都使用 BF16 vLLM、同一 Gemma 4 chat template、system instruction `<|think|>`、seed 42、max length 65,536 和全量样本。MMLU-Pro 为 5-shot，IFEval 为 0-shot；两边没有设置 `limit`。

| 指标 | Base BF16 | v4.1 BF16 | 差值 |
|---|---:|---:|---:|
| MMLU-Pro exact match | 80.7347% | 77.8092% | -2.9255 pp |
| IFEval prompt strict | 87.8004% | 84.8429% | -2.9575 pp |
| IFEval instruction strict | 91.2470% | 89.3285% | -1.9185 pp |
| IFEval prompt loose | 90.0185% | 86.8762% | -3.1423 pp |
| IFEval instruction loose | 93.0456% | 90.8873% | -2.1583 pp |
| Global MMLU EN | 53.2500% | 65.2500% | +12.0000 pp |
| Global MMLU ZH | 30.2500% | 36.0000% | +5.7500 pp |

归档：

```text
/home/gyr/lusychat-training/lusy-data-pipeline/artifacts/eval-archive/20260812-official-vs-v4.1-allloss
```

General 10K 是训练数据，不拿它本身当 benchmark。

## 固定 RP benchmark

### 数据怎么选

父池是生产快照中的 2,000 个候选对话；按 seed `20260804` 和 `production-rollout-v3-500-language-role-diversity` 缩成 500 条。抽样同时覆盖用户类型、新老用户/角色组合、对话深度、语言、NSFW 和角色分散；最终 500 个 user 与 500 个 character 都唯一。

```text
/home/gyr/lusychat-training/lusy-rp-eval-500-20260804
```

### Rollout 参数

两个候选模型共享角色卡、Greeting 和第一条真实 user。第一轮后，每个分支按自己的 assistant 回复分别生成下一条 user，因此两条 trajectory 会自然分叉。

```text
500 条最多跑到 10 轮
其中 150 条继续到 30 轮
其中 50 条继续到 60 轮
```

Judge 隐去模型名，分别看 A/B 和 B/A 两个顺序。主结果是 strict win、weak win、tie、both bad 或 uncertain；角色忠实、连续性、当前回复、剧情推进、情绪、用户控制权、语言、复读和 memory 等维度用来解释结果。

代码入口：

```text
src/evaluation/scripts/rp-eval
src/evaluation/rp_eval/
```

### 已有 run

表中 `Gemma / DeepSeek / U` 分别是 Gemma 胜、DeepSeek 胜、双顺序不一致。trajectory 会在 10/30/60 checkpoints 重复统计长对话，所以 checkpoint 数会大于 conversation 数。

| Gemma 版本 | 状态 | 首轮 Gemma / DS / U | 全部 checkpoints Gemma / DS / U |
|---|---|---:|---:|
| v4.0.1 旧量化 | 停在 250 conversations | 81 / 121 / 62 | 42+2 weak / 237 / 68（n=349） |
| v4.0.1 BF16 | matched 250 完成 | 106 / 90 / 54 | 96 / 127 / 120（n=343） |
| v4.0.1 5090 重量化 + FP8 KV | 同一 matched 250 完成 | 97 / 107 / 46 | 73 / 161 / 109（n=343） |
| v4.1 all-loss BF16 | 停在 74 conversations | 30 / 28 / 22 | 29 / 40 / 34（n=103） |

v4.0.1 BF16 与 5090 重量化使用同一 250，可以直接看量化前后差异。v4.1 只完成 74 条，不能拿它与 250 条 run 做完整模型排名。

运行账本与回复：

```text
/home/gyr/lusychat-training/rp-benchmark-eval-v1-20260811
```

## 数据隔离

后续 SFT、DPO 或 RL 至少排除固定 benchmark 的 500 个 user 和 500 个 character。benchmark 中的候选回复、Bridge user、Judge 分数和 failure tags 不回流训练。
