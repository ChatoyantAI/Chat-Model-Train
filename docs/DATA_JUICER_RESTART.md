# Data-Juicer v2.2 重启计划

## 起点

v2.2 于 2026-08-16 完成前置构造，数据窗口仍是 `[2026-03-20, 2026-07-20)` UTC。它不读取旧 MDS、Judge、RP 50K 或 repair 数据，而是从所有模型的当前有效消息版本重新构造完整对话。

```text
52,963,615 message versions
→ 51,554,445 text-trainable versions
→ 25,951,081 logical turns
→ 1,477,539 raw conversation bundles
→ 1,254,820 accepted dialogues
→ 222,719 quarantined dialogues
```

结构通过率为 84.93%。accepted 数据共保留 18,999,811 个 `user → assistant` turns。

222,719 条 quarantine 占 15.07%。reason 是 occurrence，一个对话可命中多个：

| reason | occurrences |
|---|---:|
| `untrainable_turn` | 146,418 |
| `wrong_station` | 77,873 |
| `empty_user_content` | 38,080 |
| `missing_user_event_id` | 35,717 |
| `user_version_anomaly` | 35,717 |
| `empty_assistant_content` | 16,677 |
| `assistant_version_anomaly` | 11,656 |
| `missing_assistant_event_id` | 11,646 |
| `missing_greeting` | 28 |
| `assistant_empty_after_cot_strip` | 22 |
| `current_version_causal_order_violation` | 1 |

## v2.2 已经做了什么

- 物理 message version 规范化；
- 每个 logical turn 恰好选择一个当前 user 和一个当前 assistant；
- 检查 event ID、因果顺序、空内容、媒体和版本异常；
- 统一构造 RP system、角色卡、persona、用户名和输出语言；
- 选择有效 memory，并只保留 boundary 后的完整 turns；
- 保留 Greeting、fork lineage、feedback 和 model/provider 审计字段；
- 去掉 assistant 开头完整闭合的 provider think block；
- 对 256 个 buckets 做全量结构验证并冻结 manifest。

v2.2 没有按模型来源硬筛。inactive conversation、deleted user、fork 以及不可靠的元数据标签也没有在构造层一刀切，而是保留成 audit 字段。

## v2.2 还没有做什么

1. PII、内容级未成年人风险和 provider/system 泄漏清理；
2. 对 exact/MinHash/SimHash 初筛结果做人工阈值校准和质量感知代表选择；
3. fork/shared-prefix 归并；
4. 基于已启动的 assistant/user turn embedding，完成单对话语义循环与跨对话模板候选判定；
5. 逐 turn RP 质量与整段连续性评估；
6. 点赞、点踩、regen、edit 的曝光校准；
7. language、NSFW、角色类型、长度和实体配比；
8. final-target 或 all-assistant loss 决策；
9. 下一版训练集物化。

因此 1,254,820 条只能称为候选母集。

## 接下来要补的筛选

### 1. 结构复验与基线画像

Data-Juicer 读写后重新验证 nested `messages`、role 顺序、Greeting index、event ID 唯一性和 message/turn accounting。使用真实训练 tokenizer 统计 tokens；同时统计语言、memory、fork、模型来源、反馈、用户和角色集中度。

输出：不可修改的 profile manifest，以及独立 benchmark 排除清单。

### 2. 确定性内容处理

只处理可证明的错误：PII 候选、残余宏、provider/system 泄漏、隐藏 CoT、控制字符和真实内容级安全风险。RP 的星号、引号、换行、状态栏和动作格式属于风格契约，不能直接套用通用文本 normalization。

任何清理都以 `messages` 为权威，然后重新派生扁平文本视图，避免三份表示不一致。

### 3. 分层去重

分别处理：

- 完整 dialogue exact/MinHash duplicate；
- fork 和 shared-prefix duplicate；
- 跨角色的 assistant 模板重复；
- 单对话内部循环和功能性复读。

Data-Juicer/MinHash/ANN 负责召回候选组，最终代表选择结合角色、Greeting、上下文、质量和 scene delta，不能随机保留一个。

### 4. RP 专用质量评估

逐 turn 检查角色忠实、对最新 user 的响应、上下文/关系/物品/地点连续性、用户控制权、语言和格式、复读、scene delta、节奏和纠错恢复。整段对话再聚合连续性和 trajectory health。

合格规则不能只看 mean，至少需要：

```text
critical error count = 0
每个有 loss 的 turn 通过硬合同
最差 turn 和 P10 turn 高于下限
session coherence 高于下限
不存在 major loop
```

在大规模运行前先冻结人工 gold，校准 Judge 的一致性、阈值和误杀率。不要重复旧链路“只有一半对话有 Judge、只有 Elite A 有 Turn Judge”的覆盖缺口。

### 5. 校准反馈信号

点赞只给对应展示回复提供弱正证据，no-feedback 是 unknown。dislike、regen、edit 和 correction 更适合 preference/repair lane。高 engagement 不能抵消角色崩坏、上下文矛盾、用户控制或复读硬失败。

反馈需要按 user、character、language、NSFW、turn depth、模型版本和曝光概率校准。

### 6. 在 quality-pass 池内选样

质量门通过后，才执行：

- language、NSFW、角色类型、剧情阶段和长度覆盖；
- user/character/conversation cap；
- 新老用户和角色组合覆盖；
- opening、early、middle、late 覆盖；
- semantic coverage 和 MMR/k-center 类选择；
- 最终 source/model 分布审计。

配比是质量合格池内的二级目标，不能让 liked、来源或 quota 绕过质量底线。

### 7. 冻结训练格式和评估隔离

在物化前决定监督单位：

- final-target-only：每个 target 一条记录；
- conversation all-loss：每条 trajectory 一条记录，Greeting 后多个 assistant 有 loss；
- 或基于 turn 质量选择可监督 assistant 的混合 mask。

同时冻结 benchmark user/character/message 排除清单，验证 train/val/benchmark 无实体和文本泄漏。

## 代码入口

`src/data_juicer/construction/` 保存 v2.2 构造相关代码；`src/data_juicer/independent_v1/` 保存已完成的第一轮全量 CPU 初筛。该版本把每条对话视为独立样本，不使用 fork 元数据改变初筛范围；exact 后的 MinHash、SimHash 是独立分支，复读算子只打分。`src/data_juicer/semantic_audit_v1/` 保存 18,958,570 个 turn pair 的 BGE-M3 assistant/user embedding 管线、断点续跑契约与可视化页面。后续版本不能覆盖 v2.2 构造或这些正式产物。

远端候选母集：

```text
/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816
```
