# 角色扮演 Judge 数据格式 v0.1

## 1. 核心原则

Judge 不参与两两比较，也不需要知道 A、B。

Judge 每次只看到某一个候选在生成时对应的完整消息数组，以及这个候选生成的回复：

```text
按照采集期生产逻辑和冻结数据高保真重建的 input_messages（可以包含多条 system）
+ 本次模型生成的完整 assistant 消息
```

Judge 只评价最后这条生成消息。前面的 system、记忆、greeting 和对话历史都是评分依据，不是评分对象。

数据构造不能自行发明一种“适合评估”的上下文格式。它必须按照采集期线上
`get_input_messages()` / `_construct_message()` 的逻辑重建结果。若同一次线上比较中的两个候选因为模型上下文窗口不同而收到不同的 `input_messages`，则分别保留各自的输入，不能为了离线比较方便而强行合并成一份共同上下文。该结果不是生产环境保存下来的原始请求包。

模型同一次生成的状态栏、旁白、台词和多个段落共同组成一条完整消息，必须作为整体评分，不能拆成若干句分别评分。

## 2. Judge 输入

正式文件建议使用 `judge_inputs.jsonl.gz`，每行是一条可以独立评估的 JSON 数据。

```json
{
  "schema_version": 1,
  "evaluation_id": "opaque-random-id",
  "rubric_version": "roleplay-rubric-v0.1",
  "trigger_type": "message | continue | auto_reply",
  "input_messages": [
    {
      "role": "system",
      "content": "线上拼装后的主 system：模板、角色资料和 Persona"
    },
    {
      "role": "system",
      "content": "如果生成时存在：长期记忆 summary 与结构化状态"
    },
    {
      "role": "assistant",
      "content": "角色开场白"
    },
    {
      "role": "user",
      "content": "历史用户消息"
    },
    {
      "role": "assistant",
      "content": "历史 assistant 消息"
    },
    {
      "role": "user",
      "content": "触发本次生成的当前用户消息"
    }
  ],
  "output_message": {
    "role": "assistant",
    "content": "本次需要评分的完整模型输出"
  }
}
```

Judge 输入中不出现比较编号、候选编号、模型身份、左右位置或线上结果。

## 3. 必须复刻的线上拼接逻辑

以下逻辑以 ELO 采集期正式代码提交
`f8ab72c314fafa843677a94269e29dbdfb94f344` 为基准。当前线上提交
`39964e8a1ceb3f84ad37cd5b87b549934f590754` 在这条拼接链路上没有变化。

### 3.1 主 system

线上先根据 `chat_mod` 选择模板：

- `chat_mod <= 50`：`prompt_chat`；
- `50 < chat_mod < 450`：`prompt_scene`；
- `chat_mod >= 450`：`prompt_long_text`；
- 开启第三人称协作模式时改用对应 collaborative 模板；
- 命中测试账号 green prompt 时使用 green 模板；
- `prompt_priority=user_priority` 时使用对应 `_user_priority` 模板。

随后按线上代码依次执行：

1. 在模板最后一个 `**Output Format:**` 前插入 restricted-topics 规则；
2. 注入角色名、性别、年龄、人物描述、人格、场景、示例对话；
3. 根据当前 query 和最近消息激活 Character Card Lorebook 条目；
4. 注入目标语言、回复模式和 token 参数；
5. 解析角色名、用户昵称和用户代词占位符；
6. 在模板末尾追加 `<user_profile>` Persona 块，再做一次动态占位符替换。

Persona 块包含生成时有效且非空的昵称、性别、年龄、喜欢、厌恶、个人描述和置顶记忆。

不得直接把 `character_descriptions.prompt` 当作最终 system。普通聊天链路以代码仓库中的模板为起点，该数据库字段不是 `_construct_message()` 的最终输入。

### 3.2 长期记忆 system

主 system 之后可能追加第二条 `role=system` 消息。线上格式为：

```text
Long-term conversation memory:
{summary_text}

Structured continuity state:
{state_json}

Treat this memory as canonical background context. Raw dialogue below is the newest post-summary tail.
```

是否使用 summary 是候选模型相关的：

- `0 < context_window_k < 16` 的小上下文模型不使用 summary，只保留最近 25 个对话轮次；
- 其他模型如果存在有效 summary，只读取 `compressed_until_increase_id` 之后的原始消息，并在主 system 后加入上述 memory system；
- summary 行存在但 `summary_text` 与 `state_json` 都为空时，不增加 memory system；
- 没有 summary 且历史达到 bootstrap 条件时，线上同样只保留最近 25 个轮次，但不凭空生成 summary 文本。

因此 Judge 数据应针对每个候选分别生成 `input_messages`。本轮冻结数据中，有 123 个有效比较同时包含非空 summary 和 8K 上下文模型，这些比较不能让两个候选共用同一个输入数组。

### 3.3 角色开场白

- 角色设定；
- 用户 Persona；
- 角色世界设定或 Lorebook；
- 生成前最新的记忆摘要；
- 当前产品要求的回复规则。

不要让 Judge 自己从多个数据库字段猜测 system。数据构造程序应先完成拼装，再把最终 system 作为一条 `role=system` 消息交给 Judge。

角色开场白作为第一条 assistant 消息保留。它既是对话起点，也是判断后续回复是否延续状态栏、角色标签、动作标记和台词格式的依据。

线上优先使用 `conversations.initial_greeting` 的会话快照，否则回退到角色描述中的 greeting。拼装时会替换用户/角色占位符并移除 Markdown 链接。普通聊天会把它放在 system（以及可能存在的 memory system）之后、历史消息之前。

### 3.4 历史消息

开场白之后，按生成时的实际顺序放入恢复出的 user / assistant 消息。

- 查询 `status=1` 且非 Gift 的消息；
- 有 summary 时只取 summary 边界之后的消息，小上下文模型则从完整历史中保留最近 25 个轮次；
- 只保留线上认定为文本类的消息；
- 对历史用户图片按线上格式追加已完成的图片解析文本；
- 移除 Markdown 图片、Markdown 链接和裸 URL；
- 对历史 assistant 消息执行动态用户/角色占位符替换；
- 连续相同 role 的消息使用两个换行符合并；
- 如果处理后的历史最后一条是尚未回答的 user 消息，则先移除，避免与当前 query 重复；
- 应用生成前已经发生的编辑、重新生成和版本选择结果；
- 不包含本次待评输出之后才发生的编辑或选择；
- 不把待评输出提前混入历史。

### 3.5 当前触发与 guideline

普通聊天与 Auto Reply：最后一条 `input_messages` 使用当时实际提交的 query。

Continue：线上不会保持到最后一条真实消息就直接生成，而是临时追加一条固定的 Continue user 指令。它不落库，但属于模型真实收到的输入，因此必须在 Judge 的 `input_messages` 中复刻，同时用 `trigger_type=continue` 表明来源。

最后，线上根据 `chat_mod` 给当前 query 追加对应的 `<guideline>`，包括目标长度、对话/旁白形式、格式和输出语言。Judge 必须看到追加后的完整 user content，不能只看到用户原始正文。

Auto Reply 仍标记为 `trigger_type=auto_reply`，但不把“由自动回复生成器产生”之类的说明混进 query 正文。

### 3.6 最终顺序

```text
1. system：主模板 + restricted-topics + 角色卡 + Persona
2. system：长期记忆 summary + state（仅在该候选实际使用时存在）
3. assistant：会话 greeting（普通线上分支）
4. user / assistant：经过线上过滤、裁剪、替换与合并的历史
5. user：当前 query 或 Continue 指令 + 长度/格式 guideline
6. output_message：本次候选输出，单独保存，不放进 input_messages
```

## 4. 两条 Judge 输入示例

以下均为虚构数据，不是真实用户聊天。为了展示 Judge 不需要 A/B，两条数据只使用彼此无关的不透明 `evaluation_id`。

### 4.1 样本一

```json
{
  "schema_version": 1,
  "evaluation_id": "eval-6f2a901c",
  "rubric_version": "roleplay-rubric-v0.1",
  "trigger_type": "message",
  "input_messages": [
    {
      "role": "system",
      "content": "你正在扮演顾临，27岁，旧港档案馆夜班管理员。她谨慎、嘴硬，不愿直接承认自己需要帮助，说话简短。她与失踪的林夏是姐妹，随身携带一把黑伞。用户昵称是林，希望自己决定关键行动，不喜欢角色替自己说话。最新记忆：顾临和林在档案馆找到一张指向七码头的旧船票；顾临已把黑伞放在门边。"
    },
    {
      "role": "assistant",
      "content": "【地点：旧港档案馆｜情绪：戒备｜随身物品：黑伞】\n*顾临合上登记册。*\n“闭馆了。”"
    },
    {
      "role": "user",
      "content": "这张船票是林夏留下的吗？"
    },
    {
      "role": "assistant",
      "content": "【地点：旧港档案馆｜情绪：紧绷｜随身物品：黑伞（门边）】\n*顾临把船票翻到背面，指腹停在一行褪色的日期上。*\n“她的字。我不会认错。”"
    },
    {
      "role": "user",
      "content": "你刚才不是说她今晚没来过这里吗？"
    },
    {
      "role": "assistant",
      "content": "【地点：旧港档案馆｜情绪：迟疑｜随身物品：黑伞（门边）】\n*她把船票压回桌面。*\n“人没来。票是从归档箱里找到的。”"
    },
    {
      "role": "user",
      "content": "只告诉我她去了哪里，先别让她回来。"
    }
  ],
  "output_message": {
    "role": "assistant",
    "content": "【地点：旧港档案馆｜情绪：克制｜随身物品：黑伞（门边）】\n*顾临用指节敲了敲船票上的数字。*\n“七码头。她还在那里——至少现在是。”"
  }
}
```

Judge 只需要判断最后的 `output_message` 是否回应用户、符合角色、延续状态和格式等。

### 4.2 样本二

```json
{
  "schema_version": 1,
  "evaluation_id": "eval-c91d47b0",
  "rubric_version": "roleplay-rubric-v0.1",
  "trigger_type": "message",
  "input_messages": [
    {
      "role": "system",
      "content": "你正在扮演顾临，27岁，旧港档案馆夜班管理员。她谨慎、嘴硬，不愿直接承认自己需要帮助，说话简短。她与失踪的林夏是姐妹，随身携带一把黑伞。用户昵称是林，希望自己决定关键行动，不喜欢角色替自己说话。最新记忆：顾临和林在档案馆找到一张指向七码头的旧船票；顾临已把黑伞放在门边。"
    },
    {
      "role": "assistant",
      "content": "【地点：旧港档案馆｜情绪：戒备｜随身物品：黑伞】\n*顾临合上登记册。*\n“闭馆了。”"
    },
    {
      "role": "user",
      "content": "这张船票是林夏留下的吗？"
    },
    {
      "role": "assistant",
      "content": "【地点：旧港档案馆｜情绪：紧绷｜随身物品：黑伞（门边）】\n*顾临把船票翻到背面，指腹停在一行褪色的日期上。*\n“她的字。我不会认错。”"
    },
    {
      "role": "user",
      "content": "你刚才不是说她今晚没来过这里吗？"
    },
    {
      "role": "assistant",
      "content": "【地点：旧港档案馆｜情绪：迟疑｜随身物品：黑伞（门边）】\n*她把船票压回桌面。*\n“人没来。票是从归档箱里找到的。”"
    },
    {
      "role": "user",
      "content": "只告诉我她去了哪里，先别让她回来。"
    }
  ],
  "output_message": {
    "role": "assistant",
    "content": "【地点：旧港档案馆｜情绪：激动｜随身物品：黑伞】\n*门忽然被推开，林夏撑着伞走了进来。*\n“我回来了。”"
  }
}
```

这条 Judge 输入中仍然没有 A、B，也没有任何信息说明它与样本一来自同一上下文。Judge 只根据 system、对话历史和待评输出独立评分。

## 5. Judge 输出

Judge 输出只使用 `evaluation_id` 关联输入。

```json
{
  "evaluation_id": "eval-6f2a901c",
  "rubric_version": "roleplay-rubric-v0.1",
  "gate_results": [],
  "item_scores": [
    {
      "item_id": "N1",
      "item_name": "回应当前意图",
      "score": 2,
      "evidence": "回复给出‘七码头’，并说明林夏仍在那里",
      "reason": "回答了地点，也没有违反‘先别让她回来’"
    }
  ],
  "evaluation_notes": ""
}
```

正式输出必须包含评分标准中的全部评分项。没有检验机会的项目输出 `N/A`，不能省略。

## 6. Judge 永远看不到的映射表

数据构造完成时，另外生成 `evaluation_labels.csv`。它不进入 Judge 请求，也不提供给人工标注员。

```csv
evaluation_id,pair_group_id,model_id,model_name,human_selected,source_completed_at
eval-6f2a901c,hidden-pair-001,hidden-model-01,hidden-model-name-01,1,2026-08-28T10:00:00Z
eval-c91d47b0,hidden-pair-001,hidden-model-02,hidden-model-name-02,0,2026-08-28T10:00:01Z
```

这里不需要 A/B。`pair_group_id` 只在后处理阶段说明哪些输出来自同一次线上比较；`human_selected` 表示该次线上比较是否选择了当前输出。

如果以后让 11 个模型针对同一个上下文重新生成，可以用一个 `generation_group_id` 将 11 条评估结果关联起来，同样不需要 A/B。

## 7. 后处理

评分完成并冻结后：

```text
Judge 输出：evaluation_id + 分项分数
                ↓ 按 evaluation_id 合并
隔离映射：模型身份 + 原比较分组 + 线上选择
                ↓
比较同组输出的分数
拟合各评分项权重
汇总每个模型的平均得分
计算离线模型排名
与线上 ELO 排名比较
```

Judge 只负责“这条回复各项表现如何”。两条回复谁更好、某个模型总分多少、离线排名是否复现线上排名，全部是后处理程序的职责。

## 8. 实现约束

1. `evaluation_id` 必须是随机或不可反推出模型和比较关系的标识。
2. Judge 输入按随机顺序打散，不能让同一上下文的多个输出连续出现。
3. 每个候选分别复刻其线上 `input_messages`。只有当两个候选的 prompt 分支、上下文窗口和记忆裁剪结果相同时，才要求两份输入字节级一致；不得抹掉真实的候选输入差异。
4. 输入文件、Judge 输出和隔离映射分别保存，任何一步都不能覆盖原始冻结数据。
5. Judge 运行完成前，执行评分的进程不得读取隔离映射表。
6. 另存不提供给 Judge 的 `evaluation_context_audit.jsonl.gz`，至少记录生产代码提交、模板分支、`chat_mod`、prompt A/B 分组、候选上下文窗口、summary ID/边界、保留历史范围，以及每个输入字段来自冻结值、历史值、当前快照还是推断值。
7. 某个决定 prompt 的历史参数无法证明时，必须在审计文件标记，不能把“按默认值重建”描述成线上请求包的逐字快照。
