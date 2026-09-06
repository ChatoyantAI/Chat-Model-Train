# Roleplay Battle 可学习权重计算规范

对应实现：`src/evaluation/roleplay_annotation_platform/battle_weight_fit.py`

版本：`battle-weight-fit-v2`

## 1. 目的

这套方法用 LLM Judge 对候选 A、B 的原子评分，拟合原始线上用户在 Battle 中选择 A
还是 B。它的作用是：

- 找出哪些维度、哪些原子项更能解释线上胜负；
- 把一组评分换算成可比较的 A 胜率；
- 保留每个权重、N/A 和 Gate 的可解释性；
- 最终为数据筛选、Battle 排名和后续拟合提供特征。

它不是新的 Judge，也不重新生成评分。它只使用已经保存的 Judge 结果和原始线上选择。
当前版本不使用 Token logprobs。

## 2. 一个训练样本是什么

一个训练样本是一组原始线上 Battle：

```text
候选 A 的 Gate 状态和原子评分
候选 B 的 Gate 状态和原子评分
线上用户实际选择 A 或 B
```

标签定义为：

```text
y = 1  用户选择 A
y = 0  用户选择 B
```

只有以下结果进入拟合：

1. A、B 都通过 Gate，并且都有评分；
2. 只有一侧通过 Gate，另一侧被 Gate 淘汰。

以下结果不进入拟合：

- A、B 都被 Gate 淘汰；
- 任意一侧调用失败、结果无效、被取消或尚未完成；
- 没有原始线上 A/B 选择；
- A、B 的全部原子项都同时为 N/A。

一次请求失败不会被当成低分或输掉；它只会使这一组不能作为训练样本。

## 3. 原子分差归一化

拟合器按任务冻结量表读取每项最高档，兼容历史不同尺度；现行 v0.18 的全部项目均为
0/1/2/3。对原子项 `j` 定义归一化分差：

```text
delta_ij = (score_Aij - score_Bij) / max_score_j
```

`delta_ij` 的范围是 `[-1, 1]`：

- 大于 0：该项支持 A；
- 小于 0：该项支持 B；
- 等于 0：该项不提供方向。

这保证 0–2 量表从 0 到 2 的完整差距，与 0–3 量表从 0 到 3 的完整差距等价。

## 4. 可学习评分权重

每个原子项有一个最终有效权重 `beta_j`：

```text
beta_j >= 0
sum_j(beta_j) = 1
```

代码使用 softmax 保证非负与总和为 1。模型直接学习最终原子项权重，维度权重和维度内
权重由它唯一分解：

```text
dimension_weight_d = sum(beta_j), j 属于维度 d
within_dimension_weight_j = beta_j / dimension_weight_d
```

这样既能得到 C、H、L、M、N 的维度权重，也能得到 C1、C2 等原子项在本维度内的
权重，同时避免“维度参数 × 原子参数”带来的重复参数和不可辨识问题。

### 初始先验

先验不是“所有原子项等权”，而是：

1. 所有维度等权；
2. 同一维度内的所有原子项等权。

如果共有 `D` 个维度，维度 `d` 有 `n_d` 个原子项，则：

```text
prior_j = 1 / D / n_d
```

拟合可以把权重从先验拉开，但需要得到线上选择信号的支持。

## 5. 不使用模型自报置信度

评分请求不再要求模型输出 `high / medium / low confidence`，拟合也不读取历史结果中的
该字段。模型自己口头声明“有多确定”没有稳定校准，容易引入一个新的、难以解释的可学习
变量，因此权重只由实际档位差和线上 A/B 选择学习。

OpenRouter 返回的评分 Token 概率仍可单独用于对齐诊断，但当前拟合版本不使用它；无法
返回 Token 概率的渠道也不会因此受罚。只有某原子项全部合法档位的概率都出现在
top-logprobs 中时，才形成可用的评分分布；所选 Token 的绝对生成概率和缺档的部分分布
不能当作置信度。后台还要求每个位置的候选列表包含该位置实际生成的 Token 且 logprob
一致；位置错配的整段数据直接作废。Cloudflare 的 GLM 5.3 系列实测不满足该条件，当前
不请求其评分 Token 概率。

拟合报告必须把“双方都有原子评分的评分对”和“只有一边被 Gate 淘汰的配对”分开报告。
不能用 Gate 较容易预测所带来的提升掩盖原子分本身的低预测力。当前保守候选门槛为：
评分对五折 AUC 至少 0.65，且评分对五折方向命中率至少 62%。未同时达到时，报告状态为
`diagnostic_only`，参数只能用于发现量表问题，不能自动过滤、排序或替代线上投票；达到
门槛也仍需独立留出集确认。

每个原子项还报告：在多少评分对中适用、实际让 A/B 产生不同分数的比例，以及产生分差
时分差方向与线上胜者一致的比例。稀疏的硬错误项即使很少产生分差，也仍可能适合作为
过滤条件；但不能因为拟合得到一个权重，就把它解释成稳定的排序信号。

## 6. N/A 怎样参与

对每一组 Battle `i`、原子项 `j`，定义分母占位 `active_ij`：

```text
A、B 都是 N/A：active_ij = 0
其他情况：       active_ij = 1
```

具体处理如下：

| A | B | 分子贡献 | 分母权重 | 含义 |
|---|---|---:|---:|---|
| 数字 | 数字 | `beta × delta` | `beta` | 正常比较 |
| N/A | N/A | 0 | 0 | 该项对双方都不适用，删除 |
| N/A | 数字 | 0 | `beta` | 不武断判输赢，但把总分差向 0 收缩 |
| 数字 | N/A | 0 | `beta` | 同上 |

缺少或无法解析的旧评分记录按单边 N/A 的保守口径处理：保留分母，不提供方向。

双边都通过 Gate 时，整组的归一化加权分差为：

```text
                 sum_j(beta_j * delta_ij * comparable_ij)
D_i = ----------------------------------------------------------------
                         sum_j(beta_j * active_ij)
```

其中 `comparable_ij=1` 仅表示 A、B 都是合法数字分。该设计有两个效果：

- 双边 N/A 不会稀释其余真正适用的项；
- 单边 N/A 不偏向任何候选，但会降低这组比较的确定程度。

页面额外报告：

```text
有效权重覆盖 = 双边数字项权重 / active 分母权重
```

## 7. Gate 怎样参与

Gate 方向 `G_i` 定义为：

```text
G_i = +1   A 通过，B 淘汰
G_i = -1   A 淘汰，B 通过
G_i =  0   A、B 都通过
```

单边 Gate 样本不再使用原子分差，即 `D_i=0`。模型为 Gate 学习独立的非负系数
`gamma`，用来衡量“线上用户多大概率也选择通过 Gate 的一侧”。

这里必须区分两个用途：

- 数据筛选：Gate 是硬规则，被淘汰的一侧直接不能入选；
- 拟合线上选择：`gamma` 只用于计算概率和诊断用户选择与 Gate 的关系。

拟合出的 Gate 概率不能把硬 Gate 改成软规则。

## 8. 胜率公式

最终 logit 和 A 胜率为：

```text
z_i = temperature * D_i + gamma * G_i
P(A wins) = sigmoid(z_i)
```

`temperature > 0` 控制评分差转成胜率时有多陡；`gamma > 0` 是 Gate 系数。

模型故意不使用截距。交换 A、B 后，`D` 和 `G` 都变号，因此：

```text
P(A wins after swap) = 1 - P(A wins before swap)
```

不会因为候选恰好放在 A 位或 B 位就获得固定优势。

## 9. 优化目标

训练最小化：

```text
loss = binary_cross_entropy
     + 0.025 * KL(atom_weights || equal_dimension_prior)
     + 0.002 * scale_penalty
```

各项含义：

- `binary_cross_entropy`：预测胜率与原始线上选择之间的二元交叉熵；
- 权重正则：权重没有足够数据证据时靠近“维度等权、维度内等权”的先验；
- 尺度正则：temperature 靠近 5，Gate 系数靠近 1，防止概率过度极端。

实际的尺度正则是：

```text
(log(temperature) - log(5))^2 + log(gamma)^2
```

## 10. 优化器

当前实现只依赖 NumPy，使用自定义 Adam：

- 最大 1,400 步；
- 学习率 0.035；
- 梯度范数上限 10；
- 使用三组确定性初值，后两组对原子项权重加入固定种子的轻微扰动；
- 连续 220 步没有改善就提前停止；
- 三次重启选择目标函数最小的一次。

固定初值和固定随机种子使同一份输入能复现相同结果。

## 11. 五折验证

报告不能只看训练集。系统使用 Battle 的稳定标识（优先 `pair_group_id`）计算 SHA-256，
再对 5 取模，得到固定折号。每次用四折拟合，在未参与训练的一折上预测，最后把五折
的折外预测合并为一组指标。

这不是随机打乱后临时切分，因此重复拟合不会因为切分变化而得到另一套验证结果。

页面同时显示：

- 等维度、维度内等权基线；
- 全量训练集指标；
- 五折折外指标；
- 每一折学到的权重均值和标准差。

如果折外 AUC 没有超过基线，系统会明确提示该权重只适合诊断，不应直接定版。

## 12. 指标怎么算

### AUC

随机取一组“线上选 A”和一组“线上选 B”的 Battle，模型把前者预测得更偏向 A 的
概率。0.5 约等于随机，1.0 为完全排序正确。并列按半对计算。

### Direction accuracy

用 `z > 0` 预测 A、`z < 0` 预测 B：

- `excluding_ties`：去掉 `z=0` 后计算正确率；
- `ties_half`：`z=0` 按 0.5 个正确计算。

### Log loss

```text
-mean(y*log(p) + (1-y)*log(1-p))
```

它不仅要求方向正确，也惩罚“非常自信但预测错误”。越低越好。

### Brier score

```text
mean((p-y)^2)
```

衡量概率与实际结果的平方误差，越低越好。

## 13. 单组 Battle 如何复算

公开函数：

```python
from src.evaluation.roleplay_annotation_platform.battle_weight_fit import (
    score_battle_pair,
)

result = score_battle_pair(rubric, pair, fit_report["parameters"])
```

返回内容包括：

- `a_win_probability`：拟合模型给出的 A 胜率；
- `predicted_winner`：A、B 或 tie；
- `weighted_score_difference`：温度缩放前的归一化加权分差；
- `comparable_weight_coverage`：有双边数字分的有效权重覆盖；
- `dimension_contributions`：各维度对 A−B 分差的贡献；
- 双边 N/A、单边 N/A 数量；
- Gate 的硬筛选胜方和拟合概率。

## 14. 接口与持久化

```text
POST /api/llm-judge/battles/{job_id}/weight-fit
```

拟合当前 Battle 任务并保存报告。任务必须全部结束，且必须使用当前线上评分规则。

```text
GET /api/llm-judge/battles/{job_id}/weight-fit
```

读取已保存的参数、指标和诊断。

```text
GET /api/llm-judge/battles/{job_id}/pairs/{pair_seq}
```

如果当前任务已有拟合报告，会额外返回 `learned_comparison`，按上述完全相同的公式复算
这一组 Battle。

报告保存在结果数据库的 `llm_battle_weight_fits` 表中，以 `job_id` 为主键。重新拟合会
更新报告，不会修改 Judge 的原始评分或原始线上投票。

## 15. 怎样解释结果

权重大表示该原子项在当前训练数据中更能解释线上选择，不等于该项在理论上更重要。
至少同时检查：

1. 五折 AUC 是否高于等权基线；
2. 五折 log loss、Brier 是否改善；
3. 各折维度和原子权重是否稳定；
4. 单边 N/A 和失败样本是否过多；
5. 新的一批 Battle 上能否复现提升。

只有训练集变好而折外不变或变差，通常表示过拟合，不能据此直接决定生产权重。
