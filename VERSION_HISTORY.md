# 版本记录

这里同时记数据版本和模型训练版本。版本号相同不代表同一种东西，例如 `v4.0.1 dataset` 是训练输入，`v4.0.1 run1` 是用这份输入训练出的模型。

## 一览

| 日期 | 版本 | 类型 | 输入 | 主要变化 | 结果 |
|---|---|---|---|---|---|
| 2026-07-30 | RP 50K selection v2 | 数据选择 | 286,418 条自动可选 RP replies | 质量排序、NSFW/stage 配额、实体 cap | 50,000 RP records |
| 2026-07-30 | General 10K | 数据选择 | 100,000 条公开数据候选 | 每个 source 取 10%，再保 task/language/length 分层 | 10,000 General records |
| 日期未单独冻结 | model v2 run4 | 训练 | 旧 60K | final-target-only；prompt 有重复 | 完成；旧权重已清理 |
| 2026-08-02 | dataset/model v3.1 | 数据 + 训练 | RP 50K + General 10K | 修 prompt、Greeting、user 宏和 Length Guidance | LoRA、BF16 merge、NVFP4 完成 |
| 2026-08-09 | dataset v4.0 | 数据修复 | v3.1 | 删除 DROP conversations，应用 repair v2 | 55,166 records；没有单独训练 |
| 2026-08-09 | dataset/model v4.0.1 | 数据 + 训练 | v4.0 | 修正一个对话的 Thoughts 位置 | 三卡 LoRA、merge、量化和 RP replay |
| 2026-08-12 | dataset/model v4.1 | 数据 + 训练 | v4.0.1 | RP 合并为 trajectory，Greeting 后所有 assistant 有 loss | 三卡 LoRA、BF16 merge、通用全量评测 |
| 2026-08-16 | Data-Juicer dialogue v2.2 | 候选母集 | 全量当前消息版本 | 所有模型来源统一构造完整对话 | 1,254,820 条结构合格对话；未质量筛选 |

## RP 50K selection v2

输入不是全部生产聊天，而是 Turn Judge 后归并出的母池：

```text
302,500 valid Judge rows
→ 296,959 unique native replies
→ 286,418 automatically selectable replies
→ 50,000 selected records
```

关键参数：

```text
conversation cap = 5
character cap    = 400
user cap         = 20

NSFW quota       = 17,500 / 12,500 / 11,000 / 9,000（L0/L1/L2/L3）
stage quota      = 5,000 / 12,500 / 16,250 / 16,250（opening/early/middle/late）
```

结果：12,378 个 conversations、1,813 个 characters、11,182 个 users；50K 中 48,954 条为 quality A。执行脚本没有单独保留，结果、selection report 和输入输出 manifest 仍在远端。

## General 10K

100K 候选按固定来源数量构造，最终每个来源取 10%：

| 来源 | 候选 | 训练选择 |
|---|---:|---:|
| Aya | 27,000 | 2,700 |
| OASST2 | 23,000 | 2,300 |
| Tulu IF | 15,000 | 1,500 |
| UltraChat | 15,000 | 1,500 |
| SmolTalk | 10,000 | 1,000 |
| OpenMathInstruct | 7,000 | 700 |
| BigCode SelfCodeAlign | 3,000 | 300 |

每个全局 target-length decile 取 1,000 条，source 内继续保持 task 和 language 分层。最终 train/val 为 9,800/200。

## Model v2 run4

| 项目 | 值 |
|---|---|
| Base | Gemma 4 26B-A4B IT |
| 数据 | 旧 60K；train 58,800 |
| Loss | final-target-only |
| GPU | 2×H100 80GB；FSDP2 |
| Context | 64K |
| Global batch | 16 |
| Epoch / steps | 1 / 3,675 |
| Train input tokens | 约 401.77M |
| Runtime | 约 34h55m |
| Train / eval loss | 0.9711 / 0.9281 |

v2 的独立 YAML 和旧权重已经清理，因此 LR、optimizer 和 LoRA rank 不从后续版本反推。它做过每项 `limit=50` 的 quick check，结果在 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

## Dataset / Model v3.1

数据：RP 49,000/1,000 train/val，General 9,800/200，共 60,000。每条只训练最后一个 assistant。输入 314.36M tokens，其中 train split 308.08M；trainable tokens 17.14M，其中 train split 16.79M。最长 45,197 tokens，无截断。

| 参数 | 值 |
|---|---|
| GPU | 2×H100 80GB；FSDP2 |
| Precision / context | BF16 + TF32；65,536 |
| Batch | 1/GPU × 2 GPU × grad accumulation 8 = 16 |
| Epoch / steps | 1 / 3,675 |
| Optimizer | `adamw_torch_8bit` |
| LR | `5e-5`；cosine；warmup 0.03 |
| Regularization | weight decay 0.01；max grad norm 1.0；seed 42 |
| LoRA | dense/shared r32 α64；experts r16 α32；dropout 0 |

结果：26h34m，aggregate train loss 0.96452，final eval loss 0.91848，PPL 2.50549。相对 v2，train input tokens 少 23.3%，运行时间少约 24%。没有完成可归档的 v3.1 全量 base-vs-SFT 通用基准。

## Dataset v4.0 / v4.0.1

v4.0 从 v3.1 的 50K RP 中删除 1,098 个 DROP conversations，对保留对话应用 repair v2。RP 变为 45,166 条，保留 90.33%；General 10K 字节不变。v4.0.1 在 v4.0 上把一个对话中 14 个 Thoughts blocks 的位置修正到 assistant 回复末尾，数据数量不变。

```text
RP       44,275 train + 891 val     = 45,166
General   9,800 train + 200 val     = 10,000
Combined 54,075 train + 1,091 val   = 55,166
```

## Model v4.0.1 run1

| 参数 | 值 |
|---|---|
| Loss | final-target-only |
| GPU | 3×H100 80GB；FSDP2 |
| Precision / context | BF16 + TF32；65,536；无截断 |
| Batch | 1/GPU × 3 GPU × grad accumulation 8 = 24 |
| Epoch / steps | 1 / 2,254 |
| Optimizer / LR | `adamw_torch_8bit`；`5e-5`；cosine；warmup 0.03 |
| LoRA | dense/shared r32 α64；experts r16 α32；dropout 0 |

train input 270.49M tokens，trainable 15.77M，最长 47,665。结果：19h09m34s，final train loss 1.038，validation loss 0.9998，PPL 2.718；没有 OOM、NCCL error 或 traceback。

重复风险审计中，strict target candidates 从 v3.1 的 2.428% 降到 1.948%，命中 conversation 从 7.877% 降到 6.232%。这是重复风险召回率，不是整体 RP 分数。

## Dataset / Model v4.1 all-loss

v4.1 把 45,166 条 RP targets 按 conversation leaf trajectory 合并为 11,345 条 RP trajectories，再与 10K General 混合：

```text
RP trajectories  11,122 train + 223 val = 11,345
General rows       9,800 train + 200 val = 10,000
Combined rows     20,922 train + 423 val = 21,345
```

RP 的 System、User、Greeting 不计算 loss，Greeting 后每条 assistant 计算 loss；General 仍只训练最后一个 assistant。v4.0.1 的 45,166 条 RP targets 全部覆盖一次。train+val 输入 93.99M tokens，train split trainable tokens 56.75M，最长 46,830，无截断。

| 参数 | 值 |
|---|---|
| GPU / context | 3×H100 80GB；FSDP2；65,536 |
| Batch | global 24 |
| Epoch / steps | 1 / 约 872 |
| Optimizer / LR | `adamw_torch_8bit`；`1e-5`；cosine；warmup 0.03 |
| LoRA | dense/shared r32 α64；experts r16 α32；dropout 0 |

BF16 merged 模型已经生成。完整 872-step loss 曲线只保存在原训练日志和 W&B；当前归档没有单独冻结最终 loss，因此这里不写推测值。全量通用基准见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

## Data-Juicer dialogue v2.2

```text
52,963,615 message versions
→ 51,554,445 text-trainable versions
→ 25,951,081 logical turns
→ 1,477,539 raw dialogue bundles
→ 1,254,820 structurally accepted dialogues（84.93%）
```

这版没有沿用旧 MDS、Judge 或 50K 结果，也没有按模型来源筛选。当前只完成结构构造；质量过滤、去重、Judge、配比和训练格式还没定，因此没有对应模型结果。

## 路径

数据、配置、结果和 manifest 的远端位置统一登记在 [`docs/DATA_ARTIFACTS.md`](docs/DATA_ARTIFACTS.md)。
