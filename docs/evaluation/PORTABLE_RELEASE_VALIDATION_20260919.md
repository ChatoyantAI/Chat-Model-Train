# 独立安装验证：2026-09-19

本次从GitHub新克隆 `codex/evaluation-portable`，未复制开发工作区的未跟踪代码。功能代码版本：`472289d`。

## 实际验证

- Linux/Python3.11 新虚拟环境，使用仓库锁定依赖安装；不含 torch/vLLM。
- 安装脚本自动初始化两套固定Git子模块；RMT/MiniMax复用经SHA-256校验的已有数据缓存。
- NLTK已有分词数据直接复用；修复重复拉取下载索引的卡顿，缺失资源下载有超时。
- 完成66项回归测试：0失败、0跳过，约19.5秒。
- 从统一 `init` 生成四套配置，`run --config` 自动准备四套小样本并 dry-run 成功；未发生成请求。
- `check --check-api` 验证 DeepSeek V4 Flash 与 GLM 5.3 Flash 的模型列表，只有 GET 请求。
- 将远端入口切到独立仓库后重放前次8条CharacterBench和1个CoSER完整场景：全部保持完成，新增API调用0，原始请求仍为60条。
- 冻结协议源码、评分schema、温度、最大输出长度、本地16/远程96并发和写盘参数保持。

## 清理与部署

本地开发分支清理30个旧入口/配置/配套文件，迁移仍有用的费用进程测试；13个未提交旧实验路径归档。
公开发布分支从最新 `origin/main` 建立，未带入同时开发的训练改动。
旧的原生四入口与模型专用配置已退出新任务路径。GitHub分支可直接克隆，主分支仍需PR合并。

远端完整仓库：`/workspace/projects/public-evaluation`。
固定链接：`/workspace/services/public-evaluation/current`；便捷入口：`bash /workspace/services/public-evaluation/run.sh`。
旧release和重复恢复启动器：`/workspace/archives/evaluation-legacy-20260919`。
历史训练、模型、评测缓存和结果没有删除。

原始安装日志、失败安装记录、通过的测试记录、免费端点检查、dry-run和缓存重放回执：
`/workspace/services/public-evaluation/validation/20260919/portable-clean-install/`。

这次是安装/配置/恢复验证，不是新的全量评测，没有新增付费补全。此前真实96并发测试记录见统一运行说明。
