# 维护约定

## 仓库边界

本仓库用于保存可审计的训练数据代码与说明，不是数据仓库。禁止提交：

- 生产聊天正文或可还原用户身份的 sidecar；
- API Key、SSH 密码、私钥和 `.env`；
- Parquet、JSONL、模型权重、checkpoint、运行日志和批量评估结果；
- 临时 smoke、失败重试目录或本地绝对路径。

## 新增 pipeline 版本

每个正式版本至少应提供：

1. 输入数据版本、时间窗和单位；
2. 每个过滤器删除什么、删除数量和保留率；
3. 关键代码入口；
4. 输出 schema、manifest 和 SHA-256；
5. train/validation/benchmark 隔离方法；
6. 已知偏差和不能据此得出的结论。

运行产物放在远端版本化目录，仓库只登记路径。正式目录不可覆盖；重跑使用新版本或新 run 名。

## 文档口径

- `message-version`：数据库中的物理消息版本。
- `logical turn`：当前 user 与当前 assistant 组成的业务轮次。
- `assistant target`：以某条 assistant 回复为监督目标的候选。
- `dialogue`：按 conversation/replay/memory 聚合的对话单位。
- `SFT record`：最终交给训练器的一条 messages 记录。

数字变化只有在单位相同时才写保留率。单位改变时明确写“聚合”或“展开”，不写误导性的过滤百分比。

## 安全检查

提交前至少执行：

```bash
git diff --check
git grep -n -E 'sk-[A-Za-z0-9_-]{16,}|BEGIN (RSA |OPENSSH )?PRIVATE KEY'
git grep -n -E '[A-Za-z]:\\\\'
```

第二条允许命中环境变量名和占位符，但不得出现真实凭证；第三条不应命中仓库维护文档或默认运行路径。
