"""Build the public V2 method document and append the exact reviewed rubric."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "docs" / "evaluation" / "ROLEPLAY_EVALUATION_DIMENSIONS_V0_2.md"
TARGET = ROOT / "docs" / "evaluation" / "ROLEPLAY_SINGLE_SAMPLE_FILTER_RUBRIC_V2.md"


PREAMBLE = """# 角色扮演单样本概率评分与过滤标准 V2

协议：`roleplay-single-sample-filter-v2.2`

量表来源：`roleplay-rubric-v0.18`

本版本用于评价一条候选回复本身，不比较 A/B。它保留完整原子量表的 0/1/2/3 与 N/A，不再把质量压成“通过／不通过”。Gate 与质量评分分成两次请求，评分结果在渠道支持时保留合法评分 Token 的概率分布。

## 一、为什么采用“分数 + 分布”

只保存模型最终选择的整数会把 `P(2)=0.51, P(3)=0.47` 和 `P(2)=0.98, P(3)=0.01` 都压成 2 分，损失了区分度。V2 同时保存：

- 模型实际选择的离散档位；
- 每个合法评分档位的条件概率；
- 由概率计算的连续期望分；
- N/A 概率、低质量概率和分布熵；
- 渠道无法可靠返回 logprobs 时的离散回退分。

这与 G-Eval 使用评分 Token 概率加权求期望分的方向一致。后续对 judge 分布的研究也发现，分布均值通常比只取众数更能保留排序信息。这里没有照搬论文的单一总分，而是把该推断方法应用到我们已经逐项审过的角色扮演量表。

参考：

- G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment: <https://aclanthology.org/2023.emnlp-main.153/>
- Improving LLM-as-a-Judge Inference with the Judgment Distribution: <https://aclanthology.org/2025.findings-emnlp.1259/>
- LLM-Rubric: A Multidimensional, Calibrated Approach to Automated Evaluation: <https://aclanthology.org/2024.acl-long.745/>

## 二、两阶段请求

### 阶段 A：硬 Gate

- `temperature=0`；
- 每项只输出布尔值；
- `true` 固定表示“明确发现该错误”，`false` 表示“没有发现该错误”；
- 不做安全政策审核；题材敏感、成人内容或校园背景不能单独触发 Gate；
- 任一 Gate 为 `true` 时停止本轮质量评分，进入独立 Gate 复核。

### 阶段 B：20 项质量评分

- 只有 Gate 全部为 `false` 才发送；
- 先按 H/C/M/L/N 维度分组，再把标准正文过长的组自动拆成小批；每个原子项的完整标准与案例仍放在同一请求中，不会被截断；这避免单个超长上下文或一个大 JSON 牵连全部 20 项；
- `temperature=1`、`top_p=1`，以便直接读取模型在当前采样分布下的评分概率；
- 输出只有固定结构的单字符代码：`A=0`、`B=1`、`C=2`、`D=3`、`X=N/A`；
- 不要求先输出理由或思维链，避免长解释改变后续评分 Token 的分布，也降低 JSON 失败率；
- 每项 Schema 都允许完整连续数字档位 A/B/C/D；该项允许不适用时再增加 X；
- 完整评分标准、边界和案例仍全部发送给模型。

Gate 请求与评分请求是独立的。各评分小批只负责列出的原子项。评分请求明确告诉模型 Gate 已经通过，不允许重新做 Gate 或安全判断。

## 三、概率与连续分的计算

对原子项 `i`，将 API 返回的合法评分代码 logprob 记为 `l_iv`。只在所有合法档位都出现在该 Token 位置的候选列表中、且渠道通过位置一致性检查时，才把分布标记为可用：

```text
p_iv = exp(l_iv) / Σ_u exp(l_iu)
```

如果该项允许 N/A：

```text
适用概率 A_i = 1 - p_i,N/A
条件期望分 E_i = Σ_{k∈数字档} k·p_ik / A_i
低质量概率 Q_i = Σ_{k<2} p_ik / A_i
```

如果该项不允许 N/A，则 `A_i=1`。原子项展示的 0–100 归一化分为：

```text
S_i = 100 · E_i / 3
```

所有原子项统一使用 `0/1/2/3`，因此 3 分始终映射为 100，2 分始终映射为 66.67，不再需要按不同量表上限换算。

### 聚合

1. 组内：H/C/M/L/N 各组对适用项目等权，`A_i` 作为软适用权重；
2. 组间：有有效项目的五个大类等权；
3. 总分：五组均值再映射到 0–100。

当前公式故意不使用 Battle 拟合权重。它先保留尽可能中性的、可解释的原始质量信号；等积累足够可靠的人类胜负关系后，再单独学习维度权重，不能把拟合权重反写成量表真理。

## 四、单条过滤规则

V2 首先保存完整分数，不把综合分本身冒充绝对真值。当前自动筛查触发条件是：

```text
A_i >= 0.5 且 Q_i >= 0.5  → 该原子项进入独立复核
```

即模型更倾向认为该项适用，并且在“适用”的前提下，更倾向认为它低于 2 分。若渠道没有可靠 logprobs，则回退为：离散分 0/1 进入复核，2/3 不进入，N/A 不作为缺陷。

主模型不能直接删除样本。第二个不同模型家族只复核被触发的 Gate 或原子项：

- 第二模型确认硬 Gate：`reject`；
- 第二模型确认原子项低于 2：`exclude`；
- 第二模型认为原子项更可能 N/A：`review`，留给人工或第三模型；
- 第二模型推翻全部疑点：`keep`。

综合质量分用于排序、分桶和后续阈值校准；单项低质量概率用于发现不可被平均分掩盖的局部缺陷。两者不能互相替代。

## 五、logprobs 可用性与失败降级

“API 返回了 logprobs”不等于“概率可用”。后台还检查：

1. 返回的是最终 JSON 内容对应的逐 Token 记录；
2. 每个 Token 自身 logprob 与 `top_logprobs` 中同 Token 的值一致；
3. 能定位到具体原子项的评分代码；
4. 该项全部合法评分代码都在候选列表中。

只返回部分合法档位时，部分概率仅保留作诊断，不计算期望分。出现渠道位置错配、完全不返回 logprobs 或合法档位不全时：

- 结构化评分仍然有效；
- 不把整条任务标成失败；
- 明确显示降级原因；
- 使用离散档位计算分数与复核条件。

因此 V2 的“评分成功”和“概率分布可用”是两个独立指标。

端点目录中的“支持 logprobs”只表示可以请求，不保证每次响应都覆盖评分 Token。若响应省略 JSON 项目键的 Token，后台仅在评分代码数量、顺序和解析结果全部吻合时采用严格顺序映射；某一评分代码的 logprob 缺失时，该项降级，不能用相邻 Token 或自报置信度代替。

## 六、当前状态与校准要求

这是一套可运行的实验协议，不宣称当前阈值已经由真人标签证明为最优。正式大规模过滤前，应在固定验证集上同时报告：

- 结构化输出成功率；
- 原子项 logprob 覆盖率；
- 与真人分数的加权 Krippendorff α；
- 单项 0/1 缺陷的 precision、recall；
- 连续总分对人工 Battle 胜负的 ROC-AUC、PR-AUC 和校准曲线；
- 不同模型、渠道和语言切片的稳定性。

在没有足够真人金标前，推荐把自动结果用于候选排序和“送复核”，不要用一个未经校准的总分阈值不可逆地批量删除数据。

---

# 附录：完整评分维度、档位、边界与案例

下文由当前线上量表源文件完整同步，不做摘要或二次改写。

"""


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    TARGET.write_text(PREAMBLE + source.rstrip() + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
