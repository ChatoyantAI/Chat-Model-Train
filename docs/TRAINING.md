# 训练参数与结果

## 版本对比

| 版本 | Train 数据 | Loss policy | GPU / global batch | LR | Steps | Trainable tokens |
|---|---:|---|---:|---:|---:|---:|
| v2 run4 | 58,800 records | final target | 2×H100 / 16 | 未单独冻结 | 3,675 | 未单独冻结 |
| v3.1 run1 | 58,800 records | final target | 2×H100 / 16 | `5e-5` | 3,675 | 16.79M |
| v4.0.1 run1 | 54,075 records | final target | 3×H100 / 24 | `5e-5` | 2,254 | 15.77M |
| v4.1 run1 | 20,922 trajectories/rows | RP all assistant；General final target | 3×H100 / 24 | `1e-5` | 约 872 | 56.75M |

v4.1 records 最少，但有 loss 的 tokens 最多。比较训练强度时要看 trainable tokens，不能只看 records 或 steps。

## Loss mask

v2、v3.1、v4.0.1：

```text
System / Greeting / User / 历史 assistant = loss 0
最后一个 assistant target                 = loss 1
```

v4.1 RP：

```text
System / Greeting / User       = loss 0
Greeting 后每个 assistant      = loss 1
```

v4.1 General 仍是 final-target-only。

## 共同设置

四个版本都从 Gemma 4 26B-A4B IT 开始，训练 LoRA，不是 full fine-tuning，也不是 QLoRA/QAT。可核对到的后三版共同使用：BF16、TF32、FSDP2、context 65,536、micro batch 1/GPU、gradient accumulation 8、cosine schedule、warmup 0.03、weight decay 0.01、max grad norm 1.0、seed 42、不 packing、不 pad-to-64K、超长直接报错。

v3.1、v4.0.1、v4.1 的 LoRA：

| 模块 | Rank | Alpha | Dropout |
|---|---:|---:|---:|
| Attention q/k/v/o + shared MLP up/down/gate | 32 | 64 | 0 |
| Routed experts gate_up/down | 16 | 32 | 0 |

## v2 run4

```text
run name: gemma4-26b-a4b-lora-sft-60k-64k-v2-run4-fsdp2
```

| 项目 | 值 |
|---|---:|
| Train records | 58,800 |
| GPU / global batch | 2×H100 / 16 |
| Epoch / steps | 1 / 3,675 |
| Train input tokens | 约 401.77M |
| Runtime | 约 34h55m |
| Aggregate train loss | 0.9711 |
| Final eval loss | 0.9281 |
| Eval interval | 900 steps |

v2 的 YAML 和模型产物已经清理，LR、optimizer、LoRA rank 和 trainable token 总量没有独立记录。

## v3.1 run1

数据：

| Split | Records | Input tokens | Trainable tokens |
|---|---:|---:|---:|
| RP train | 49,000 | 302,952,225 | 14,702,984 |
| RP val | 1,000 | 6,174,694 | 299,425 |
| General train | 9,800 | 5,124,367 | 2,084,763 |
| General val | 200 | 109,927 | 48,275 |
| 合计 | 60,000 | 314,361,213 | 17,135,447 |

train split 是 308,076,592 input tokens、16,787,747 trainable tokens。最长记录 45,197 tokens，无截断。

| 参数 | 值 |
|---|---|
| GPU / distributed | 2×H100 80GB；Accelerate；FSDP2 |
| Global batch | 16 |
| Epoch / steps | 1 / 3,675 |
| Optimizer | `adamw_torch_8bit` |
| LR | `5e-5`；cosine；warmup 0.03 |
| Eval / save | 每 600 / 100 steps；最多 4 checkpoints |
| Config SHA-256 | `7f87f87613a95371750b05fe98cd1dfa2ac47a45b4a7a42581b421f01cc50161` |
| W&B run | `ej1japvf` |

| 结果 | 值 |
|---|---:|
| Runtime | 95,637 s；约 26h34m |
| Aggregate train loss | 0.96452 |
| Final logged batch loss | 1.00635 |
| Final eval loss / PPL | 0.91848 / 2.50549 |
| Train samples/s | 0.615 |
| Optimizer steps/s | 0.038 |

v3.1 相比 v2，train input 从约 401.77M 降到 308.08M，减少 23.3%；runtime 从约 34h55m 降到 26h34m。

## v4.0 / v4.0.1 数据

v4.0 在 v3.1 上删除 1,098 个 DROP conversations，应用 repair v2，并把同一被修 turn 在 target 和历史中的出现一起更新。v4.0.1 只追加一个 Thoughts 位置修正，数量不变。

| Split | Records | Input tokens | Trainable tokens |
|---|---:|---:|---:|
| RP train | 44,275 | 265,367,448 | 13,684,161 |
| RP val | 891 | 5,388,045 | 270,594 |
| General train | 9,800 | 5,124,367 | 2,084,763 |
| General val | 200 | 109,927 | 48,275 |
| 合计 | 55,166 | 275,989,787 | 16,087,793 |

General 文件与 v3.1 字节相同；RP conversation split leakage 为 0。

## v4.0.1 run1

| 参数 | 值 |
|---|---|
| GPU / distributed | 3×H100 80GB；Accelerate；FSDP2 |
| Global batch | 24 |
| Epoch / steps | 1 / 2,254 |
| Optimizer / LR | `adamw_torch_8bit` / `5e-5` |
| Eval / save | 每 600 / 100 steps；最多 4 checkpoints |
| Config SHA-256 | `b132a9dedfc0f19044077514a0cdd4402621216855f2df3e1f7ff12b8e9f994b` |
| Adapter SHA-256 | `7db068ba317d8282ff84667761030bffcfce1526ef8170a08732be1a0f0487a7` |
| W&B run | `s94r5lx4` |

train split 有 270,491,815 input tokens、15,768,924 trainable tokens；最长 47,665，无截断。

| 结果 | 值 |
|---|---:|
| Runtime | 68,980 s；约 19h09m34s |
| Completed steps | 2,254 / 2,254 |
| Final train loss | 1.038 |
| Final validation loss / PPL | 0.9998 / 2.718 |
| OOM / NCCL / traceback | 0 / 0 / 0 |

## v4.1 run1

v4.1 将 v4.0.1 的 45,166 条 RP targets 聚合成 11,345 条 conversation leaf trajectories。为避免早期 turn 看到未来信息，移除了 per-target Length Guidance 和 terminal dynamic Memory。

| 数据 | Train | Val | 合计 |
|---|---:|---:|---:|
| RP trajectories | 11,122 | 223 | 11,345 |
| General rows | 9,800 | 200 | 10,000 |
| Combined rows | 20,922 | 423 | 21,345 |
| RP supervised assistant messages | 181,067 | 3,644 | 184,711 |

train+val input 93,990,477 tokens；train split trainable 56,754,827 tokens；最长 46,830；截断 0。按监督 tokens，RP:General 约 26.2:1。

| 参数 | 值 |
|---|---|
| GPU / distributed | 3×H100 80GB；Accelerate；FSDP2 |
| Global batch | 24 |
| Epoch / steps | 1 / 约 872 |
| Optimizer / LR | `adamw_torch_8bit` / `1e-5` |
| Eval / save | 每 300 / 100 steps；最多 4 checkpoints |
| Config SHA-256 | `88eb844f9a5be87a7c5e3c8145dcdab372b0686b99d14a33d4fe18bea3a710fe` |
| W&B run | `52tui0x3` |

训练完成并生成 BF16 merged 模型。当前仓库记录没有冻结最终 loss；完整曲线在原训练日志和 W&B run。

## 远端路径

```text
# v3.1 dataset
/home/gyr/lusychat-training/lusy-sft-60k-20260802-v3.1

# v4.0.1 dataset
/home/gyr/lusychat-training/lusy-sft-55k-20260810-v4.0.1

# v4.1 构造、配置和启动记录
/home/gyr/lusychat-training/lusy-data-pipeline/code/training-v4.1-coser-allloss

# v4.1 H100 pretokenized data
/workspace/lusychat-training/data/lusy-sft-21k-64k-v4.1-coser-allloss-pretokenized

# v4.1 BF16 merged model
/workspace/lusychat-eval/models/gemma4-26b-a4b-lusy-sft-v4-1-coser-allloss-bf16-merged
```

完整版本顺序见 [`../VERSION_HISTORY.md`](../VERSION_HISTORY.md)，远端数据索引见 [`DATA_ARTIFACTS.md`](DATA_ARTIFACTS.md)。
