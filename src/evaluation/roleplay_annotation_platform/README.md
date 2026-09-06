# Roleplay evaluation platform

负责人：`@gaoxiaomo`；Reviewer：待指定。包含人工标注、投票回放、LLM Judge、多模型对齐、Judge Battle、独立单样本过滤 V2 和会话过滤。三个 Web 服务分别保存 Cookie 与结果库，不覆盖彼此的数据。

## 功能与评估单位

| 功能 | 入口 | 输出 |
|---|---|---|
| 人工标注 | `app:app`，`/` | 单回复评分；按用户名保存草稿、提交和进度 |
| 投票回放 | `/battle` | 原线上 A/B、用户选择；模型名称按需解盲 |
| LLM Judge | `/llm-judge` | 单回复 Gate、20 项分数、理由与证据 |
| 多模型对齐 | `/llm-compare` | 相同样本上多个 Judge 的逐项比较 |
| Judge Battle | `/llm-battle` | A/B 独立评分、分差、线上选择及权重拟合 |
| 单样本过滤 V2 | `filter_v2_app:app`，独立端口 | 离散 / 概率评分、0–100 汇总分及复核建议 |
| 会话过滤 | `conversation_filter_app:app`，独立端口 | 候选回复组的一套评分、问题定位、会话排名 |

入口均在本包；从仓库根目录运行时使用完整的 `src.evaluation.roleplay_annotation_platform.*` 模块名。

会话分不是逐轮独立分：阅读完整历史，只有 `scope=candidate` 的 Assistant 回复直接参与质量判断；`context_only` 历史与 Greeting 仅提供依据。会话模式不执行普通 Judge 的 Gate。

## 评分标准与文档

- [评估文档索引](../../../docs/evaluation/README.md)
- [完整单回复量表](../../../docs/evaluation/ROLEPLAY_EVALUATION_DIMENSIONS_V0_2.md)：现行 `roleplay-rubric-v0.18`，保留原文件名兼容已有引用；包含完整档位、边界和案例。
- [会话过滤规范](../../../docs/evaluation/ROLEPLAY_CONVERSATION_FILTER_V1.md)与[会话完整量表](../../../docs/evaluation/ROLEPLAY_CONVERSATION_FILTER_RUBRIC_V1.md)。
- [V2 概率评分规范](../../../docs/evaluation/ROLEPLAY_SINGLE_SAMPLE_FILTER_RUBRIC_V2.md)与[服务使用说明](../../../docs/evaluation/ROLEPLAY_SINGLE_SAMPLE_FILTER_V2_DEPLOYMENT.md)。
- [Battle 权重拟合公式](../../../docs/evaluation/ROLEPLAY_BATTLE_WEIGHT_FITTING_V1.md)，与 [Elo / Bradley–Terry 排名](../elo_ranking/README.md)是两个独立实现。
- [target 训练包规范](../../../docs/evaluation/ROLEPLAY_TARGET_EXPORT_V1.md)。

单回复量表包含 8 个错误 Gate、5 个大类、20 个原子项，均使用连续 `0/1/2/3`；允许不适用的项目另含 `N/A`。Gate 的 `true` 表示发现所述错误，`false` 表示未发现，不含 G-S。N1 不接受 N/A。2 表示完整做到，3 表示该项本身形成突出优点。

网页和请求共用 `rubric.json`，保存源文、版本与 SHA-256。同步命令：

```bash
python -m src.evaluation.roleplay_annotation_platform.build_rubric --input docs/evaluation/ROLEPLAY_EVALUATION_DIMENSIONS_V0_2.md --output src/evaluation/roleplay_annotation_platform/rubric.json
python -m src.evaluation.roleplay_annotation_platform.build_filter_v2_rubric
python -m src.evaluation.roleplay_annotation_platform.build_filter_v2_docs
python -m src.evaluation.roleplay_annotation_platform.build_conversation_docs
```

V2 从单回复量表复制质量项，保留自己的 Gate 协议；会话量表独立定义于 `conversation_filter.py`。不要仅改生成 JSON。历史评分携带原规则版本，不当作现行量表的新结果。

## 安装与运行

Python 3.10+。从仓库根目录创建并激活虚拟环境，再安装依赖：

```bash
python -m venv .venv
python -m pip install -r src/evaluation/roleplay_annotation_platform/requirements.txt
python -m src.evaluation.roleplay_annotation_platform.prepare_data --input outputs/evaluation/input/judge_inputs.jsonl.gz --output outputs/evaluation/data/samples.sqlite3
python -m uvicorn src.evaluation.roleplay_annotation_platform.app:app --host 127.0.0.1 --port 8766 --workers 1 --env-file configs/evaluation/annotation.env
```

将 [annotation.env.example](../../../configs/evaluation/annotation.env.example) 复制为同目录 `annotation.env`，填写路径；将 [llm-judge.env.example](llm-judge.env.example) 中需要的渠道配置加入该私有环境文件。真实 `.env` 不进 Git。`pyarrow` 用于 Parquet 构建与导出；Web 依赖 FastAPI、HTTPX、JSON Schema、NumPy、Uvicorn。

三个 `.service` 模板默认仓库位于 `%h/roleplay-evaluation`，虚拟环境在其中的 `.venv`。部署前修改路径，只启动需要的服务；均使用单个 Uvicorn worker，并发由应用队列管理。服务模板监听 `0.0.0.0`，本地开发默认回环地址。

## 输入与结果

输入构建见 [roleplay_judge_dataset](../roleplay_judge_dataset/README.md)，字段见[数据格式规范](../../../docs/evaluation/ROLEPLAY_EVALUATION_DATA_FORMAT_V0_1.md)。

| 文件 | 内容与访问边界 |
|---|---|
| `judge_inputs.jsonl.gz` | 单候选输入；只发送规范化上下文与回复，内部 ID 不进入 Prompt |
| `samples.sqlite3` | 样本序号及压缩输入索引 |
| `evaluation_labels.csv` | ID、配对、模型身份、线上选择与时间，仅解盲 / 离线分析使用 |
| `evaluation_context_audit.jsonl.gz` | 拼接来源审计，不进入盲评 |
| `annotations.sqlite3` | 人工结果、Judge 队列、逐次尝试、拟合报告 |
| `by_user/<username>/<seq>.json` | 人工草稿 / 提交快照 |

所有人工标注员按同一稳定样本顺序查看，不按用户名重新随机。第一条 Assistant 显示为 Greeting，目标回复单列。标准与案例在选分前可见，证据和原因可留空。普通盲评接口不返回模型、配对或线上选择；Battle 属于解盲页面。

```bash
python -m src.evaluation.roleplay_annotation_platform.export_results --database outputs/evaluation/results/annotations.sqlite3 --output outputs/evaluation/exports/annotations.jsonl
```

可加 `--username annotator_01`。导出含规则版本、用户名、序号、关联 ID、状态、Gate、scores、备注与时间戳。

## Judge 请求与后台队列

普通 Judge 与 Judge Battle 共用构造器。A/B 分别请求，不发送另一侧回复、来源模型、线上选择或人工评分。

1. 从 User 正文拆出平台 `<guideline>`，单列为 `platform_output_requirements`；N1 只评价用户意图，不检查平台建议字数。目标语言以平台指定语言为准。
2. 先发送 Gate 标准和 Schema，任一错误命中即 `discarded`。
3. 通过后发送统一说明、完整质量标准与案例及 20 项 Schema，每项输出 `value/evidence/reason`。
4. 校验 JSON、必需字段和合法档位。证据允许短引用、关键词、概括或空数组，不要求逐字匹配；不输出 `decision_step` 或自报 confidence。
5. 保存请求、结果、实际 provider、费用、延迟与错误。失败保留原始 content，不从 reason 猜分，不把失败当 0 或 N/A。

普通 Judge 默认不覆盖 temperature，不发 `max_tokens`。全局并发默认 64、范围 1–64，Judge / Compare / Battle 共享队列。普通校验失败最多两次；429 单独最多四次，默认约 5/15/45 秒退避并尊重 `Retry-After`。支持暂停、继续、停止和预算停止阈值；在途请求仍可能计费。

模型目录由服务器加载；`ROLEPLAY_LLM_MODEL_SOURCE=static` 使用静态列表。OpenRouter 按模型配置 provider；默认 GLM 固定 Cloudflare，其他模型使用独立优先顺序。TatuCloud 使用 `tatu::` 前缀区分同名模型，按渠道能力处理 Schema 与 SSE。DeepInfra 直连用于会话服务。

## 一致性、概率与拟合

- Gate 与原子评分分开统计。缺失只排除对应模型 / 样本 / 阶段，其余有效模型仍可比较。
- 分别展示两两同档、相差不超过一档和严重分歧。N/A 参与一致性，N/A 与数字不是相邻档。
- 全体同档 / 全体最大档差不超过 1 只在全部所选模型有结果的单元上计算；全体 N/A 计一致。
- 可按样本和维度集中查看所有模型的分数、理由与证据。
- 普通 Judge 仅在 token 位置正确且合法档位概率完整时计算分布与期望分；缺失概率不当作 100% 置信。当前 Cloudflare GLM 路由禁用评分 logprobs。
- Token 修正距离单独展示：`d_token = d_raw × sqrt(p_left × p_right)`，同时报告覆盖率和同一子集原始距离，不替代原始一致性。
- Battle 权重拟合只使用数字分差与线上选择，不使用自报 confidence 或 token logprobs；分别报告评分对 / 单边 Gate、基线、训练集与五折折外指标。未达规范中的验证阈值时仅用于诊断。

## 实现布局

| 文件组 | 职责 |
|---|---|
| `app.py`、`static/` | 人工标注、Battle、Judge、Compare 和拟合接口 |
| `llm_judge.py`、`llm_runtime.py`、`model_catalog.py` | 请求、校验、渠道、队列、费用与概率 |
| `build_rubric.py`、`rubric_migrations.py` | 标准构建及历史兼容 |
| `battle_weight_fit.py` | 可解释非负权重与胜率拟合 |
| `filter_v2_app.py`、`single_sample_filter_v2.py`、`static_filter_v2/` | 独立概率评分服务 |
| `conversation_filter*.py`、`conversation_platform_context.py`、`static_conversation/` | 会话索引、拼接、量表与后台服务 |
| `conversation_ranking.py`、`export_conversation_run.py` | 排名、汇总与冻结导出 |
| `select_ranked_conversation_targets.py`、`export_target_training_bundle.py` | 配额选择和 target 训练包 |
| `supervise_conversation_run.py` | 显式授权的有界技术失败恢复与终态导出，不默认启动 |

`single_sample_filter.py` 保留独立缺陷确认策略；`conversation_neutral_review.py` 保留中立文本复核接口。二者不在当前会话批量评分自动调用链中。

## 安全与产物管理

Key 只从服务端环境读取，预览脱敏。原始聊天、标签、数据库、凭据与完整请求不得放入静态目录或 Git。用户名登录不是密码认证；可访问页面的人可能创建付费任务。公网部署需另加访问控制与 HTTPS。

数据、日志与快照放在忽略的 `outputs/` 或外部存储；临时脚本留在 `tmp/`，旧草案留在 `.local/evaluation/archive/`。正式代码不得依赖这些目录。变更规则前备份结果库，按版本 / 哈希隔离，不混用旧任务续跑。

## 验证

```bash
python -m unittest discover -s tests/evaluation -v
python -m src.evaluation.roleplay_annotation_platform.build_conversation_docs --check
node --check src/evaluation/roleplay_annotation_platform/static/app.js
```

测试使用虚构数据和 HTTP mock，覆盖 Elo、上下文、盲态、Schema、重试、费用、N/A、概率、拟合、会话排名和 target 导出。默认不访问生产数据、GPU 或付费 API。提交前对三个静态目录下的所有 JavaScript 执行语法检查。
