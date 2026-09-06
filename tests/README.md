# 测试目录

测试按模块组织，例如 `tests/data/`、`tests/training/`、`tests/evaluation/` 和 `tests/common/`。

最低要求：

- 新功能覆盖核心成功路径和关键失败路径；
- 缺陷修复包含能复现问题的回归测试；
- 数据转换测试使用极小、固定、脱敏样例；
- 需要 GPU、网络或付费 API 的测试必须显式标记，默认测试不得意外调用生产服务；
- 随机测试设置种子，并在失败信息中输出复现条件。

每个模块 README 必须写出实际测试命令和验收门槛。

## 评估模块

```bash
python -m pip install -r src/evaluation/roleplay_annotation_platform/requirements.txt
python -m unittest discover -s tests/evaluation -v
```

覆盖 Elo、上下文还原、人工标注、LLM 请求 / 校验 / 重试、Battle 权重、概率评分、会话队列与排名、角色配额和 target 前缀。HTTP 使用 mock，临时数据均为虚构样例；无 GPU、生产数据或 API Key 要求。同步测试核对完整规则文档与生成 JSON，不能只修改其中一份。
