# Scoring Policy

## 候选与用户续接

第一轮两个候选接收相同 Greeting 和真实用户输入。后续每个分支分别把可见历史、候选回复和历史真实下一条用户消息交给 Luna。Luna 返回一条纯文本 user 消息，直接进入对应分支。

续接层没有 `KEEP`、`MINOR_EDIT`、`MAJOR_EDIT`、`UNBRIDGEABLE`，也没有 Verifier。Luna 输出只折叠空白为单行；不执行语义校验、自动修复或硬截断。续接文本和候选质量分开记录。

## Judge 隔离

Judge 只看规范化角色参考、可见对话和匿名 A/B 候选。它看不到模型名称、候选 system prompt、Luna 调用元数据或未来真实用户输入。

每个首轮和完整轨迹都执行两次评分，第二次交换 A/B：

- 两次选择同一实际模型：strict win
- 两次都是 TIE：tie
- 两次都是 BOTH_BAD：both_bad
- 一胜一平：weak win
- 其他不一致：uncertain

主指标是 pairwise outcome。十个 1..5 诊断维度为 character fidelity、continuity、relevance、progression、emotional fit、user agency、style、non-repetition、memory 和 future compatibility，不机械求和决定胜负。

## 多轮轨迹

Pilot 固定使用前 20 个会话、最多 10 个用户轮。正式计划为全部 500 个对话到第 10 用户轮、固定最长 150 个扩到 30、固定最长 50 个扩到 60。分支后的 user 文本允许不同，轨迹 Judge 评价整体 RP 体验。
