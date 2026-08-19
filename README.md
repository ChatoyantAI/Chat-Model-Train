# Chat Model Train

LusyChat 角色扮演模型的后训练工程仓库。这里集中维护数据提取与筛选、MDS/MMR 多样性选择、Dialogue/Turn Judge、通用能力锚点、SFT 消息物化和固定评测代码，同时记录每个数据版本、训练版本、评测结果及其远端产物位置。

仓库不保存生产聊天正文、用户/角色映射、大型 Parquet、模型权重或服务凭证；这些内容留在训练服务器的版本目录中，本仓库通过 manifest、hash 和路径文档关联。历史流程里没有保留下来的源码会明确标注，不用后来重写的代码冒充当时实现。

## 当前内容

| 内容 | 当前状态 | 入口 |
|---|---|---|
| 第一版 RP SFT 数据线 | 已完成并训练：`2.83M targets → pre-MDS 653K → MMR 419K targets → Judge → RP 50K` | 本 README「第一版训练数据的完整构造过程」 |
| General 能力锚点 | 已完成：公开通用候选 100K → 分层抽取 10K，与 RP 50K 合成 SFT 60K | [`src/general`](src/general/README.md) |
| Data-Juicer v2.2 重启线 | 已从全量消息构造 1,254,820 条结构合格对话；质量筛选、去重和配比尚未开始 | [`docs/DATA_JUICER_RESTART.md`](docs/DATA_JUICER_RESTART.md) |
| RP 模型评测 | 已保留固定 benchmark、候选模型客户端、Judge、评分和 ledger 代码 | [`docs/EVALUATION.md`](docs/EVALUATION.md) |

数据版本和模型版本的对应关系见 [`VERSION_HISTORY.md`](VERSION_HISTORY.md)。训练参数与结果见 [`docs/TRAINING.md`](docs/TRAINING.md)，远端数据位置见 [`docs/DATA_ARTIFACTS.md`](docs/DATA_ARTIFACTS.md)。

## 阅读导航

| 想了解什么 | 从哪里开始 |
|---|---|
| 第一版数据从生产消息到 SFT 60K 如何形成 | 继续阅读本 README 的 1–10 节 |
| 某个版本何时构造、用了哪版数据和模型 | [`VERSION_HISTORY.md`](VERSION_HISTORY.md) |
| 训练参数、tokens、steps、loss 和模型产物 | [`docs/TRAINING.md`](docs/TRAINING.md) |
| benchmark 协议和评测结果 | [`docs/EVALUATION.md`](docs/EVALUATION.md) |
| 远端 Parquet、report、manifest 和 hash 在哪里 | [`docs/DATA_ARTIFACTS.md`](docs/DATA_ARTIFACTS.md) |
| 每个脚本的输入、输出和用途 | [`src/README.md`](src/README.md) 及各子目录 README |
| 下一版为什么回到全量数据、后续准备怎么筛 | [`docs/DATA_JUICER_RESTART.md`](docs/DATA_JUICER_RESTART.md) |

## 仓库结构

```text
Chat-Model-Train/
├── README.md                         # 数据构造主线与关键口径
├── VERSION_HISTORY.md                # 数据、训练和评测版本记录
├── CONTRIBUTING.md                   # 仓库维护规则
├── requirements.txt                  # 历史 pipeline 的依赖并集（版本以 run manifest 为准）
├── docs/
│   ├── DATA_ARTIFACTS.md             # 远端数据、manifest 和 hash 路径
│   ├── TRAINING.md                   # 各训练版本的参数、tokens 和结果
│   ├── EVALUATION.md                 # 通用评测与 RP benchmark 参数、结果
│   └── DATA_JUICER_RESTART.md        # v2.2 后续筛选计划
└── src/
    ├── README.md                     # 每个目录和脚本的用途
    ├── preprocessing/                # 第一版：消息物化、技术门禁、因果上下文、语言过滤和实体 cap
    ├── mds/                          # 第一版：对话聚合、BGE-M3 embedding、KMeans 和 MMR
    ├── judging/
    │   ├── dialogue/                 # 对话级 Judge contract、调用、有效性检查和 tier 构造
    │   └── turn/                     # 逐回复 Judge、重试、聚合、审计和 selection policy 分析
    ├── general/                      # General 100K 构造、10K 分层抽样与验证
    ├── materialization/              # RP/General SFT 消息物化及 Prompt、Greeting、占位符修复
    ├── evaluation/                   # 固定 RP benchmark 的样本、候选模型、Judge 和 ledger
    └── data_juicer/
        └── construction/             # v2.2 全量消息规范化、fork lineage 和完整对话重建
```

第一版主链路是：

```text
生产消息快照
→ preprocessing 技术过滤与缩池
→ mds 对话聚合和多样性选择
→ dialogue judge 对话级质量分层
→ turn judge 逐回复质量评估
→ RP 50K 选择
→ materialization 生成 SFT messages
→ 与 general 10K 合并训练
```

`evaluation` 不参与训练数据筛选，负责在固定样本和协议上比较模型。`data_juicer` 是独立的新数据线，从全量消息重新构造对话，不读取第一版的 MDS、Judge 或 RP 50K 结果。

每次运行的数据、manifest、配置和模型留在远端版本目录。`requirements.txt` 是历史脚本用到的依赖并集；某次运行的准确依赖、参数和输入 hash 以对应 run manifest 为准。

## 第一版训练数据的完整构造过程

第一版并不是直接从 2.83M 条里取最高分的 50K。质量分只覆盖 MMR 后一半对话中的 Elite A；更早的缩池主要由技术门禁、实体 cap 和 MMR 决定。

### 总账

不同位置的数字单位不同：`message-version`、`assistant target`、`dialogue` 和 `SFT record` 不能直接相减。

| 流程位置 | 输入 → 输出 | 同单位保留率 | 这一层主要做什么 |
|---|---:|---:|---|
| 消息规范化 | 52,963,615 → 51,554,445 message versions | 97.34% | 删除失效版本、非文本媒体和空内容 |
| 技术候选 | 形成 2,834,682 assistant targets | — | 恢复当前 user、完整前缀、target 和业务元数据 |
| 严格语言过滤 | 2,834,682 → 2,672,966 targets | 94.30% | 删除高置信语言不匹配 |
| pre-MDS | 2,672,966 → 653,370 targets | 24.44% | conversation/user/character cap、精确历史重复处理 |
| 聚合 MDS universe | 653,364 targets → 189,321 dialogues | 单位变化 | 按对话、回放模式和 memory boundary 聚合 |
| KMeans + MMR | 189,321 → 115,746 dialogues | 61.14% | 分语言覆盖和对话多样性选择 |
| MMR 后展开 | 115,746 dialogues → 419,691 targets | 单位变化 | 合并 MMR neutral 与 protected/memory bypass |
| Dialogue Judge | 115,746 中抽取 57,874 dialogues | 50.00% | 对话整体质量评分；另一半没有 Judge 分数 |
| Dialogue Judge 有效结果 | 57,874 → 57,536 dialogues | 99.42% | 删除结构无效或请求失败 |
| Elite A | 57,536 → 15,772 dialogues | 27.41% | 选出进入逐 turn 评估的最高层 |
| Turn Judge | 302,523 → 302,500 turns | 99.99% | 因果逐回复质量评估与重试 |
| 重复评估行归并 | 302,500 → 296,959 native replies | 98.17% | 同一原生回复只保留一个最终选择候选 |
| 自动可选池 | 296,959 → 286,418 replies | 96.45% | 删除硬失败并隔离身份冲突 |
| 最终 RP | 286,418 → 50,000 records | 17.46% | 质量排序、配比、实体 cap、阶段覆盖 |
| 通用锚点 | 100,000 → 10,000 records | 10.00% | 按来源、任务、语言、长度分层抽取 |
| 第一版 SFT | RP 50,000 + General 10,000 | 83.33% / 16.67% | 按 conversation/group 隔离 train/val |

### 1. 冻结生产数据并保留版本历史

数据窗口为 `[2026-03-20, 2026-07-20)`，数据库时间按 UTC 解释。生产库只读导出后，所有处理都在冻结文件上完成。

第一版 SFT 候选范围只接受目标站点内 Public、有效的 root conversation；`is_fork_conversation=true`、`is_root_conversation=false` 或其它可确认的 non-root branch 在 target 构造前直接排除。因此老链路没有在 parent 与 child Fork 之间比较质量，也没有构造 lineage、计算 shared prefix 后择优保留：它采用的是“Fork 不进入候选池”，不是“Fork-aware 去重”。符合条件的 root parent 可以继续参与筛选，其 child branch 整体不参与。

冻结内容包括消息版本、对话、角色、角色卡、用户 persona、memory、edit/regen、点赞点踩、订阅证据和模型/provider 信息。消息按稳定 conversation bucket 分片，并保留原始 event ID，便于后续因果重建和审计。

行级只保留：

- 当前有效版本；
- `user` 或 `assistant` 文本；
- 非空内容；
- 无非文本媒体。

被 edit/regen 替代的旧回复不进入 SFT 正文，但 mutation 和 feedback 继续保留在 sidecar，用于 preference 数据与审计。

对应代码：[`src/preprocessing`](src/preprocessing/README.md)。

### 2. 从消息构造可训练 assistant targets

51,554,445 是物理 message versions，里面同时有 user、assistant、历史版本和不可训练消息；2,834,682 是 assistant targets，单位已经变化，所以这里不写一个误导性的保留率。做法是遍历当前有效 assistant event，每条都恢复 target 前的 user 和历史，并要求 target 前至少有 5 个完整、技术可用的 `user → assistant` turns。以下内容在这一层删除：

- dislike 对应的当前回复；
- edit/regen 后被替代的旧版本；
- user 或 assistant 缺失、空文本、版本冲突；
- 隐藏 reasoning 字段和开头完整闭合的 provider `<think>…</think>`；
- 剥离 think 后为空、think 标签不闭合或仍有高风险残留；
- 无法确定性修复的引号、星号或嵌套 RP 格式；
- 高置信 provider、平台、system prompt 或 AI 身份泄漏。

这一层形成 2,834,682 个技术候选。它并不是“所有模型等量进入”：

| 来源 | targets | 占比 |
|---|---:|---:|
| Flash neutral + liked Flash | 2,812,068 | 99.20% |
| Claude | 17,961 | 0.63% |
| 其他模型中被点赞且技术合格 | 4,653 | 0.16% |

因此第一版存在明确的 V4 Flash 来源偏置。后续 Judge 虽然对来源盲评，但无法恢复在这里没有进入候选池的其他模型回复。

### 3. 构造因果上下文，执行语言过滤和实体限流

每个 target 被物化为因果可见上下文：统一 RP system、角色卡、Greeting、persona、有效 memory、历史 user/assistant 和当前 user。当前 target 以及 target 之后的消息不会进入选择特征。

严格语言过滤只删高置信 user/assistant 语言不匹配：`2,834,682 → 2,672,966`，删除 161,716 条，保留 94.30%。其中 neutral Flash 有 2,647,320 条，按下面的固定顺序缩池：

| 顺序 | 参数 | 输入 → 输出 | 对上一步保留率 |
|---|---|---:|---:|
| 1 | 每个 conversation 最多 6 条 | 2,647,320 → 1,021,702 | 38.59% |
| 2 | 每个 user-character 最多 12 条 | 1,021,702 → 1,021,702 | 100.00% |
| 3 | 每个 user 最多 60 条 | 1,021,702 → 979,954 | 95.91% |
| 4 | 每个 character 最多占 neutral pool 的 0.5% | 979,954 → 628,403 | 64.13% |

同一 cap 内的排序依次看：target 后是否有真实 user 回复、target 当时是否有付费证据、稳定 hash、`target_id`。这一段没有 LLM 质量分。

Claude、liked strong、liked Flash 和有效 memory 等 protected 数据不完全走 neutral cap。把 25,235 条 protected core 加回后有 653,638 条；再删 207 条“target 与可见历史完全重复”和 61 条 post-dedupe character cap 超额，得到 653,370 条。相对语言过滤后的 2,672,966 条，总保留率为 24.44%。

这一层的排序主要使用“是否有真实后续 user”“target 时点是否有付费证据”和稳定 hash，还没有 LLM 内容质量分。这是旧链路后续需要重点修正的问题。

### 4. 用 MDS/MMR 控制对话多样性

pre-MDS 的 653,370 条先删除 privacy denylist 中 6 条，再按 `conversation_id + replay mode + memory id/boundary` 将 653,364 targets 聚合为 189,321 个 replay dialogues。每个对话构造三个不包含当前 assistant target 的视图：

- `U`：完整用户需求与轨迹；
- `R`：角色、场景、personality、tags 和 Greeting；
- `S`：target 前的近期剧情和当前 user。

三个 BGE-M3 embedding 归一化后按下面的权重拼接：

```text
V = L2Norm(concat(sqrt(0.745) × U, sqrt(0.005) × R, sqrt(0.250) × S))
```

21 种语言分开聚类。MiniBatchKMeans 使用 `K=1000`、seed `20260724`、batch size `4096`、`max_iter=100`、`n_init=3`。每个 cluster 的第一条取 centroid 最近项，之后按下面的 MMR 分数贪心选择；cluster 不超过 5 条时全部保留：

```text
0.5 × cosine(item, centroid)
- 0.5 × max_cosine(item, already_selected)
```

neutral pool 有 185,040 个 dialogues。主版本每个 cluster 取 `ceil(size × 60%)`，得到 111,465 个 neutral dialogues；再加 3,851 个 protected core 和 430 个 effective-memory bypass，得到 115,746 个 dialogues，随后展开为 419,691 个 targets。

| MMR retention | dialogues（含 bypass） | 展开 targets |
|---:|---:|---:|
| 50% | 97,172 | 355,963 |
| **60%** | **115,746** | **419,691** |
| 70% | 134,247 | 482,092 |
| 80% | 152,672 | 542,203 |

这一层优化的是**对话语义覆盖**，不是逐条回复质量；代码位于 [`src/mds`](src/mds/README.md)。

### 5. Dialogue Judge 评估对话整体质量

115,746 个 retention-60 UNION dialogues 中，111,465 条来自 neutral MMR，另外还有 3,851 条 protected core bypass 和 430 条 effective-memory/summary bypass。Dialogue Judge 没有简单随机切半，也没有按照 MMR rank 或已知质量信号取前一半，而是做了两次确定性的分层抽样：Qwen 与 GLM 各取 28,937 条，也就是各约占完整母集的 25%；两个集合互不重叠，不是同题双判。合计调用 57,874 条，结构有效 57,536 条，Judge 有效覆盖率为 49.71%。

抽样单位是完整 dialogue。每条对话按以下五个维度组成联合分层：

- `source_partition`：neutral MMR、protected core bypass、summary bypass；
- `language_group`：en、es、pt、fr、zh、de、ja、other；
- `representative_topic`：explicit、romance/suggestive、other RP、unknown；
- `replay_mode`：完整历史、bootstrap tail、effective-memory tail；
- `turn_bucket`：5–8、9–16、17–32、33–64、65+ 个可评分 assistant turns。

先用最大余数法按 `source_partition` 分配整数名额，再在每个来源内部按上述联合分层分配名额。层内使用固定 seed 对 `dialogue_id` 计算 `SHA256(seed + "\0" + dialogue_id)`，按 hash 排序后截取相应名额，因此相同输入可以复现完全相同的样本。第一批从完整母集抽取 28,937 条交给 Qwen；第二批先排除第一批，再从剩余 86,809 条中重新分层并抽取 28,937 条交给 GLM。第二批目标仍是完整母集的 25%，不是剩余数据的 25%。

| 批次 | 母集处理 | dialogues | scoreable turns |
|---|---|---:|---:|
| Qwen | 从完整 115,746 条中分层抽取第一批 | 28,937 | 491,155 |
| GLM | 排除第一批后，从剩余 86,809 条重新分层抽取 | 28,937 | 491,283 |
| 合计 | 两批交集为 0 | 57,874 | 982,438 |
| 未评估 | 未进入两个 Judge 批次 | 57,872 | — |

每条 dialogue 一旦入选，就保留其中所有可评分 assistant turns。两个 25% 分别向上取整，所以合计比严格的一半多 1 条；抽样阶段也没有向 Judge 暴露模型来源、点赞、reaction、订阅、edit/regen 或 MMR rank 等偏好信号。

这里的 dialogue 是同一个 root conversation 按 `replay_mode + memory_id + memory boundary` 聚合出的技术评估单元，不是 Fork branch。相同原生 assistant event 可能因不同 replay boundary 出现在多个技术 dialogue 中，Turn Judge 后会再按 `(conversation_id, assistant_event_id)` 归并；这只消除同一原生回复的重复技术展开，不能替代不同原生 conversation 之间的内容去重。

Judge 的核心维度均为 0–4 分：角色忠实、上下文连贯、对当前 user 的响应、剧情推进、写作和 trajectory health，另外记录 Greeting、质量下限、可用 turn 数、阶段证据和重复 flags。对话分先按下面的权重计算，再在各自 Judge family 内转成 rank percentile，避免直接混合 Qwen 与 GLM 的原始分：

```text
weighted_core =
    0.28 × character + 0.18 × context + 0.12 × user_relevance
  + 0.18 × progression + 0.12 × writing + 0.12 × trajectory

raw_quality =
    0.80 × weighted_core/4
  + 0.12 × quality_floor/4
  + 0.08 × usable_turn_bucket/4
```

进入可选层还要求：核心维度均不低于 2、quality floor 不低于 2、usable bucket 不低于 3、没有 critical/repeat block，也不需要 rejudge。

57,536 条有效结果按 gate 和 family 内分位得到：

- Elite A：15,772 dialogues；
- Strong B：4,228 dialogues；
- borderline 22,444；rejudge 4,868；hold 10,562。

Elite A 占有效 Judge 对话的 27.41%。Strong B 与 Elite A 合起来是 Best-20K，但只有 Elite A 进入下一步 Turn Judge。对应代码位于 [`src/judging/dialogue`](src/judging/README.md)。

### 6. Turn Judge 逐回复评估

Elite A 展开为其可见 trajectory 中的 302,523 个 assistant turns。这里会把 Dialogue endpoint 前面的历史 turns 一起展开，所以虽然最早的技术候选要求 endpoint 前有 5 个 clean turns，Turn Judge 池里仍然会出现真实的 opening reply。生产口径为 Turn Judge v4.7，模型是 GLM 4.7 Flash，Direct API 并发 64；两台 runner 按 dialogue 切成互不重叠的两半。Judge 只看当前回复及其之前的因果上下文，不读取后续 user，也不看来源、点赞和 tier。

核心检查包括：

- character fidelity；
- context coherence；
- current turn fit；
- relationship continuity；
- scene progression；
- language / roleplay quality；
- 条件性亲密内容是否合适；
- 语义复读与风险 flags。

每个维度输出 `reason + 0–4 分`，另外给 repetition、flags、NSFW 和置信度。首次结构有效 291,417 条（96.33%）；11,106 条重试后得到 302,500 条有效结果，仅 23 条持续无效。首次全量运行约 134.97 美元，prompt cache hit 约 73.12%。对应实现和聚合工具位于 [`src/judging/turn`](src/judging/README.md)。

### 7. 归并同一原生回复的重复评估行，并选出 RP 50K

这里的 302,500 是有效 Turn Judge evaluation rows，不等于 302,500 条不同的数据库回复。MDS 的技术 `dialogue_id` 由 `conversation_id + replay_mode + memory_id + memory boundary` 组成；同一个 root conversation 可以形成多个完整历史或 memory-tail 技术 dialogue。Turn Judge 会展开每个 Elite A 技术 dialogue 中可见的 assistant turns，因此同一条数据库 assistant event 可能在不同 replay context 下被评估多次。

评估阶段保留这些行是合理的，因为上下文连贯、memory 一致性、关系连续性和重复判断会随可见 context 改变；但最终选择时必须恢复到原生回复单位，避免一条回复因拥有多个 replay view 而获得多次进入 50K 的机会或被重复训练。因此按 `(conversation_id, assistant_increase_id)` 归并：

```text
302,500 Turn Judge rows
→ 296,959 unique native replies
→ 删除 10,494 hard failures
→ 47 条身份冲突进入人工复核
→ 286,418 automatically selectable
→ 50,000 selected
```

归并掉的 5,541 行是重复评估记录，不代表发现了 5,541 条文本抄袭，也不是 retry 重复；retry 已在 Turn Judge 聚合时处理。代表行不取 Judge 最高分，而是依次偏好：target body hash 与原始 text hash 一致、可见 context 更完整、stable hash 更靠前，避免多次评估带来“挑最高分”偏差。共有 233 个原生键出现不同 body hash，其中无法可靠解释的 47 条进入人工复核。

这一步只归并同一原生 `(conversation_id, assistant_increase_id)` 的技术评估视图。它不处理业务 Fork，不合并不同 conversation 中内容相似的回复，也不是 exact/MinHash/semantic 文本去重。

50K 占 286,418 条自动可选回复的 17.46%，不是均匀随机抽样。旧 selection report 记录的选择顺序是：

1. 硬门禁：任一核心维度为 0、critical flag、major/critical repetition 直接删除；L4 和身份冲突转人工复核。
2. 给每条回复标记 `opening / early / middle / late`。`ordinal=1` 是 opening，其余按对话内相对位置三等分。
3. 按 `quality band → 核心均分 → 最低维度 → confidence → soft flags → stable hash` 排序。
4. 先从优质长对话组成 `opening + middle + late` bundle，再补 NSFW、语言、来源、字数和阶段缺口。
5. 对已经入选 1–2 条的对话，优先补到 3 条且增加新 stage；最后用局部 swap 修正配额。

选择时的硬 cap：

```text
每 conversation <= 5 targets
每 character <= 400 targets
每 user <= 20 targets
```

NSFW 和 stage 是硬目标；language、word length 和来源是软配额。liked 数据在不破坏质量门和 cap 的前提下提高优先级。最终 selector 的独立源码没有保留下来，因此这些规则和结果可核对，但不能只靠本仓库重新生成完全相同的 50K ID 集合。

最终 RP 50K：

| 维度 | 实际分布 |
|---|---|
| conversations / characters / users | 12,378 / 1,813 / 11,182 |
| 每对话 targets | 最大 5，中位数 4；89.13% 对话贡献 3–5 条 |
| NSFW | L0 17,500；L1 12,500；L2 11,000；L3 9,000 |
| stage | opening 5,000；early 12,501；middle 16,249；late 16,250 |
| quality | A 48,954；B 746；C 300 |
| 主要语言 | en 31,064；es 9,319；pt 3,624；fr 2,589；zh 1,553 |
| 来源 | ordinary 48,248；Claude non-liked 1,095；liked non-Claude 656；Claude liked 1 |
| 回复长度 | 1–50: 131；51–99: 627；100–149: 4,179；150–200: 24,250；201–250: 15,386；251–399: 4,312；400+: 1,115 |

本仓库保留 Judge、聚合代码和远端 selection report；缺失的 selector 文件在 [`docs/DATA_ARTIFACTS.md`](docs/DATA_ARTIFACTS.md) 中单独标注。

### 8. 构造 General 100K，再分层抽取 10K

通用数据不是线上聊天，而是公开数据集构成的能力锚点，用于降低 RP 微调造成的灾难性遗忘。先确定性构造 100,000 条候选：

| 来源 | 100K 候选 | 最终 10K | 作用 |
|---|---:|---:|---|
| Aya | 27,000 | 2,700 | 多语言人工指令 |
| OASST2 | 23,000 | 2,300 | 人类排序的多轮对话 |
| Tulu instruction following | 15,000 | 1,500 | 约束遵循 |
| UltraChat | 15,000 | 1,500 | 广泛普通助手行为 |
| SmolTalk selected | 10,000 | 1,000 | 约束、改写、摘要 |
| OpenMathInstruct | 7,000 | 700 | 数学和标准答案来源 |
| BigCode SelfCodeAlign | 3,000 | 300 | 执行过滤代码 |

统一过滤要求：角色合法、对话交替、最后一条为 assistant、target 非空、长度不过界、无残留 think tag、无控制 token、无高置信 AI 身份声明，且 last-user prompt 与 target 分别全局精确去重。每个来源在过滤后按固定语言/config quota 和 deterministic key 取到上表数量；不是从 Hugging Face 原始顺序截前 N 条。只有最后一个 assistant 计算 loss。

10K 不是重新混配来源，而是每个 source 精确取 10%：Aya 2,700、OASST2 2,300、Tulu 1,500、UltraChat 1,500、SmolTalk 1,000、OpenMath 700、BigCode 300。同时在每个 source 内按 task、language 和 target length 分层，每个全局长度十分位最终各 1,000 条。构造与验证代码位于 [`src/general`](src/general/README.md)。

### 9. 物化为 60K SFT 数据

RP 记录使用：

```text
system（统一 RP 规则 + Character + 长度 + User）
→ assistant Greeting
→ 历史 user / assistant
→ 当前 user
→ 最终 assistant target
```

第一版 v3.1 每条记录只有最后一个 assistant `train=true`；system、Greeting、user 和历史 assistant 都是 masked context。General 同样只训练最后一个 assistant。

数据按 conversation/group 隔离：

| Split | RP | General | 合计 |
|---|---:|---:|---:|
| train | 49,000 | 9,800 | 58,800 |
| validation | 1,000 | 200 | 1,200 |
| 总计 | 50,000 | 10,000 | 60,000 |

物化阶段还修复了早期 Prompt 重复、Greeting 漏取、用户占位符残留和按真实 words 生成 Length Guidance 等问题。训练格式与版本演进见 [`docs/TRAINING.md`](docs/TRAINING.md)。

### 10. 训练和评估

下面是已经跑过的主要版本。它们都从 Gemma 4 26B-A4B IT 开始，使用 BF16 LoRA；详细参数、tokens 和结果见 [`docs/TRAINING.md`](docs/TRAINING.md)。

| 版本 | 数据 / loss | GPU / global batch | Epoch / LR | Steps | 结果 |
|---|---|---:|---|---:|---|
| v2 run4 | 旧 60K；final target | 2×H100 / 16 | 1 / LR 未单独冻结 | 3,675 | 34h55m；train 0.9711；eval 0.9281 |
| v3.1 run1 | RP 50K + General 10K；final target | 2×H100 / 16 | 1 / `5e-5` | 3,675 | 26h34m；train 0.9645；eval 0.9185 |
| v4.0.1 run1 | RP 45,166 + General 10K；final target | 3×H100 / 24 | 1 / `5e-5` | 2,254 | 19h10m；train 1.038；eval 0.9998 |
| v4.1 all-loss | RP 11,345 trajectories + General 10K；RP 多 assistant loss | 3×H100 / 24 | 1 / `1e-5` | 约 872 | BF16 merge 完成；完整 loss 曲线只在原训练日志/W&B |

v3.1 和 v4.0.1 的 LoRA 都是 dense/shared rank 32、alpha 64，routed experts rank 16、alpha 32；dropout 0。v4.1 保留同一 LoRA 结构，但把学习率降到 `1e-5`，因为 trainable tokens 从 v4.0.1 的 15.77M 增加到 56.75M。

v4.1 与原生 BF16 的同协议全量结果：MMLU-Pro `80.73% → 77.81%`，IFEval prompt strict `87.80% → 84.84%`，Global MMLU EN `53.25% → 65.25%`，ZH `30.25% → 36.00%`。完整表见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。


## Data-Juicer v2.2：从全量对话重新开始

2026-08-16 完成的新链路没有读取旧 MDS、Judge、50K 或 repair 数据，而是回到全量消息版本：

```text
52,963,615 message versions
→ 51,554,445 text-trainable versions
→ 25,951,081 logical turns
→ 1,477,539 raw conversation bundles
→ 1,254,820 structurally valid dialogues（84.93%）
```

222,719 条进入 structural quarantine（15.07%）。reason 可以重叠，主要 occurrence 是：`untrainable_turn` 146,418、`wrong_station` 77,873、`empty_user_content` 38,080、`missing_user_event_id/user_version_anomaly` 各 35,717、`empty_assistant_content` 16,677、`assistant_version_anomaly` 11,656、`missing_assistant_event_id` 11,646、`missing_greeting` 28、`assistant_empty_after_cot_strip` 22、因果顺序异常 1。

v2.2 纳入所有模型的当前有效回复，模型/provider 只保留为审计字段。它已经完成统一 system、角色卡、persona、memory、Greeting、fork lineage 和完整对话的重建，但只做了结构门禁；尚未执行内容安全、全局去重、逐轮质量 Judge、会话连续性、配比或训练 loss 决策。

准备接 Data-Juicer 的筛选顺序：

```text
结构复验和 tokenizer 统计
→ PII / CoT / provider 泄漏等确定性内容清理
→ dialogue、fork/shared-prefix、assistant 模板和单对话循环去重
→ RP 专用逐轮与会话级质量评估
→ 反馈信号校准
→ 在 quality-pass 池内做语言、NSFW、角色、长度和实体覆盖
→ 冻结 benchmark 隔离清单
→ 定义 final-target 或 all-assistant loss
→ 物化下一版训练集
```

具体输入、输出和待补 operator 见 [`docs/DATA_JUICER_RESTART.md`](docs/DATA_JUICER_RESTART.md)，构造代码位于 [`src/data_juicer`](src/data_juicer/README.md)。

## 远端数据

大数据在训练服务器，入口：

```text
/home/gyr/lusychat-training/lusy-data-pipeline/README.md
```

数据、报告、manifest 和哈希见 [`docs/DATA_ARTIFACTS.md`](docs/DATA_ARTIFACTS.md)。

## 改动时一起更新

- 数据数量或筛选规则变了：改本 README。
- 新增数据/训练/评测版本：改 [`VERSION_HISTORY.md`](VERSION_HISTORY.md)。
- 新增或移动脚本：改 [`src/README.md`](src/README.md) 和对应目录 README。
- 新增远端产物：改 [`docs/DATA_ARTIFACTS.md`](docs/DATA_ARTIFACTS.md)。
- 不提交聊天正文、用户/角色 ID 映射、权重、日志和凭证。

更多规则见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
