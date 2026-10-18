# 固定公开评测流程

更换候选模型复用同一流程：冻结输入 → 候选模型生成 → GLM 5.3 Flash 评分 → 分 benchmark 汇总。
恢复时先读取已成功的请求缓存与原始响应，只补缺失、失败或无效样本。
付费响应先持久保存再交付；派生成绩沿用已验证的 Gemma 异步 checkpoint 实现。

## 配置与入口

- `configs/evaluation/public_gemma_verified_v1.json`：固定调度、超时、异步写入配置和恢复模块 SHA256。
- `configs/evaluation/qwen35_public_pair.json`：本轮两个模型的名称、revision、服务端口与部署身份。
- `scripts/evaluation/run_qwen_pair_public.py`：共用 CharacterBench/CoSER 入口，模型从配置读取。
- `scripts/evaluation/launch_qwen_pair_public.sh`：同时启动本轮两模型，复用各自原运行目录。

候选服务准备完成后，换模型只改模型配置并指定新的结果目录；不修改冻结评测脚本。
同一候选模型恢复则使用原目录，避免重复计费。不同模型结果分别保存，不覆盖旧结果。
调用示例（Linux 评测环境）：

```bash
python scripts/evaluation/run_qwen_pair_public.py \
  --actor base \
  --models-config configs/evaluation/qwen35_public_pair.json \
  --runtime-config configs/evaluation/public_gemma_verified_v1.json \
  --run-root /workspace/runs/roleplay-evaluation/qwen35-base-public-20260918 \
  --execute
```

## 固定边界

温度 0.7、max_tokens 8192、CoSER 20 轮与固定 seed、输入、提示词、评分 schema、裁判和思考模式策略
沿用冻结协议。API/本地请求限额分别管理，两模型合计 API 128、本地 16；每模型分配 64/8。
CharacterBench/CoSER worker 使用冻结脚本的 32/8。RMT 的 96 worker 不用于这两个 benchmark。
预算保护关闭，仍保留真实费用、usage、响应、错误与重试证据。

未经用户明确要求，不调整上述参数或引入新的缓存/归档机制。
模型服务的部署与评测请求配置分别管理；模型目录完整性与服务身份在运行前检查。
每次部署保留实际流程配置、模型配置与代码哈希，已有协议与日志原样保留。
无响应且无法确认收费的在途请求隔离，不自动重发；缺失结果如实报告，不能标记全量完成。

本轮部署、模型校验和恢复细节见 `QWEN35_REGMIX_PUBLIC_20260918.md`。
