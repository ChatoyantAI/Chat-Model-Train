"""Frozen platform messages -> one context/target example; never calls a model."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

try:
    from .select_ranked_conversation_targets import (
        export_selection,
        dump,
        quantiles,
        sha256_file,
    )
except ImportError:
    from select_ranked_conversation_targets import (
        export_selection,
        dump,
        quantiles,
        sha256_file,
    )

VERSION = "roleplay-target-messages-v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def make_example(packet, candidate_id, ranking, source_context_hash):
    """Use only the frozen applicable system and strict message prefix before target."""
    if packet.get("platform_assembly", {}).get("version") != "platform-assembly-v1":
        raise ValueError("Expected the evaluated platform-assembled snapshot")
    if "prompt" in packet.get("role_profile", {}):
        raise ValueError("Legacy role prompt must not be part of this dataset")
    messages = packet["messages"]
    if len({m["id"] for m in messages}) != len(messages):
        raise ValueError("Duplicate source message IDs")
    matches = [
        (i, m)
        for i, m in enumerate(messages)
        if m.get("candidate_message_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError("Target must match exactly one frozen candidate")
    target_index, target = matches[0]
    if target["role"] != "assistant" or target["scope"] != "candidate":
        raise ValueError("Only coarse-filter Assistant candidates may be targets")
    systems = [s for s in packet["system_contexts"] if target["id"] in s["applies_to"]]
    if len(systems) != 1 or not systems[0].get("content"):
        raise ValueError("Target must have exactly one nonempty applicable SYSTEM")
    system = systems[0]
    requirements = target.get("platform_output_requirements", {})
    if requirements.get("response_language") != system["response_language"]:
        raise ValueError("Frozen output language and SYSTEM do not agree")
    prefix = messages[:target_index]
    if not prefix or prefix[-1]["role"] != "user":
        raise ValueError("Target must immediately follow a User message")
    conversation_id = packet["conversation_id"]
    if (
        ranking["conversation_id"] != conversation_id
        or ranking["message_id"] != candidate_id
    ):
        raise ValueError("Ranking reference mismatch")
    if ranking["target_increase_id"] != target["source_increase_id"]:
        raise ValueError("Source target position mismatch")

    prompt = [{"role": "system", "content": system["content"]}]
    greeting = packet.get("greeting", {})
    greeting_included = bool(greeting.get("content"))
    if greeting_included:
        prompt.append({"role": "assistant", "content": greeting["content"]})
    kept_prefix = []
    for message in prefix:
        if greeting_included and message.get("displayed_as_greeting"):
            continue
        if message["role"] not in ("user", "assistant") or not isinstance(
            message["content"], str
        ):
            raise ValueError("Unexpected message shape in frozen history")
        prompt.append({"role": message["role"], "content": message["content"]})
        kept_prefix.append(message)
    if not isinstance(target["content"], str) or not target["content"]:
        raise ValueError("Expected a nonempty string target")
    if prompt[-1]["role"] != "user":
        raise ValueError("Training messages must end with the current User query")
    metadata = {
        "selection_order": ranking["selection_order"],
        "conversation_id": conversation_id,
        "conversation_seq": ranking["conversation_seq"],
        "character_id": packet["character_id"],
        "user_key": packet["user_key"],
        "source_model": ranking["model"],
        "language": ranking["language"],
        "response_language": system["response_language"],
        "assistant_turn_id": target["id"],
        "target_increase_id": target["source_increase_id"],
        "query_increase_id": prefix[-1]["source_increase_id"],
        "source_context_hash": source_context_hash,
        "input_sha256": digest(prompt),
        "target_sha256": hashlib.sha256(target["content"].encode()).hexdigest(),
        "system_context_id": system["id"],
        "system_template": system.get("template"),
        "system_sha256": hashlib.sha256(system["content"].encode()).hexdigest(),
        "history_turn_ids": [m["id"] for m in kept_prefix],
        "history_source_increase_ids": [
            m.get("source_increase_id") for m in kept_prefix
        ],
        "background_assistant_count": sum(
            m["role"] == "assistant" and m["scope"] == "context_only"
            for m in kept_prefix
        ),
        "greeting_included": greeting_included,
        "greeting_source": greeting.get("source"),
        "historical_request_replay": False,
        "platform_assembly_version": packet["platform_assembly"]["version"],
        "platform_output_requirements": requirements,
        "conversation_mean_score": ranking["conversation_mean_score"],
        "conversation_numeric_items": ranking["conversation_numeric_items"],
        "conversation_na_items": ranking["conversation_na_items"],
        "conversation_low_items": ranking["conversation_low_items"],
        "conversation_major_issue": ranking["conversation_major_issue"],
        "target_independently_scored": False,
    }
    return {
        "schema_version": VERSION,
        "sample_id": candidate_id,
        "messages": prompt,
        "target": target["content"],
        "metadata": metadata,
    }


READER = '''"""Read the released dataset without external packages or network requests."""
import gzip
import json
from pathlib import Path

def iter_targets(root):
    root = Path(root)
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    for shard in manifest['shards']:
        with gzip.open(root / shard['path'], 'rb') as stream:
            for line in stream:
                yield json.loads(line)

def as_chat_messages(example):
    # This conversion does NOT configure training loss masks.
    # Configure your trainer to supervise ONLY the final appended Assistant.
    return [*example['messages'], {'role': 'assistant', 'content': example['target']}]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    total = sum(1 for _ in iter_targets(args.root))
    print(json.dumps({'records': total}))
'''

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": VERSION,
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "sample_id", "messages", "target", "metadata"],
    "properties": {
        "schema_version": {"const": VERSION},
        "sample_id": {"type": "string", "minLength": 1},
        "messages": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["role", "content"],
                "properties": {
                    "role": {"enum": ["system", "user", "assistant"]},
                    "content": {"type": "string"},
                },
            },
        },
        "target": {"type": "string", "minLength": 1},
        "metadata": {"type": "object"},
    },
}


def build_bundle(
    root,
    job_id,
    output,
    limit=50000,
    character_cap=200,
    shard_size=5000,
    previous_ranking=None,
):
    import sqlite3

    if shard_size < 1:
        raise ValueError("Positive shard size required")
    frozen = root / "exports" / job_id
    signature = {
        "schema_version": VERSION,
        "limit": limit,
        "character_cap": character_cap,
        "shard_size": shard_size,
        "job_id": job_id,
        "ranking_sha256": sha256_file(frozen / "ranking.json"),
        "builder_sha256": sha256_file(Path(__file__)),
        "selector_sha256": sha256_file(
            Path(__file__).with_name("select_ranked_conversation_targets.py")
        ),
    }
    if previous_ranking is not None:
        previous_ranking = Path(previous_ranking)
        signature["previous_ranking_sha256"] = sha256_file(previous_ranking)
    if output.exists():
        old = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
        if old["build_signature"] != signature:
            raise ValueError("Existing release differs; use a new version directory")
        verify_release(output)
        return old
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    for name in ("data", "audit", "docs", "tools"):
        (stage / name).mkdir()
    selection_dir = stage / "audit/selection"
    selection = export_selection(
        root, job_id, limit, selection_dir, character_cap=character_cap
    )
    with closing(
        sqlite3.connect(
            f"file:{(root / 'data/conversations-v1-platform.sqlite3').as_posix()}?mode=ro",
            uri=True,
        )
    ) as db:
        hashes = dict(db.execute("SELECT seq,context_hash FROM conversations"))
    with gzip.open(selection_dir / "target_ranking.jsonl.gz", "rb") as stream:
        ranks = {r["message_id"]: r for line in stream for r in [json.loads(line)]}
    selected_plan = json.loads(
        (selection_dir / "selected_conversations.json").read_text(encoding="utf-8")
    )
    shard_count = math.ceil(limit / shard_size)
    shards = []
    stats = Counter()
    input_chars, message_counts = [], []
    lookup = {}
    with ExitStack() as stack:
        streams = []
        for i in range(shard_count):
            path = f"data/targets-{i:05d}-of-{shard_count:05d}.jsonl.gz"
            streams.append(
                stack.enter_context(
                    gzip.open(stage / path, "wt", encoding="utf-8", compresslevel=6)
                )
            )
            shards.append({"path": path, "records": 0})
        contexts = stack.enter_context(
            gzip.open(selection_dir / "contexts.jsonl.gz", "rb")
        )
        for line in contexts:
            item = json.loads(line)
            packet = item["conversation"]
            seq = item["conversation_seq"]
            for target_id in item["selected_candidate_message_ids"]:
                rank = ranks[target_id]
                assert rank["selection_order"] == stats["targets"] + 1
                example = make_example(packet, target_id, rank, hashes[seq])
                shard_index = stats["targets"] // shard_size
                streams[shard_index].write(
                    json.dumps(example, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                shards[shard_index]["records"] += 1
                stats["targets"] += 1
                stats["with_greeting"] += example["metadata"]["greeting_included"]
                stats["with_background_assistant_history"] += (
                    example["metadata"]["background_assistant_count"] > 0
                )
                stats["with_defaulted_platform_settings"] += bool(
                    packet["platform_assembly"]["defaulted_settings"]
                )
                input_chars.append(sum(len(m["content"]) for m in example["messages"]))
                message_counts.append(len(example["messages"]))
                lookup[target_id] = example["metadata"]
            if (
                len(lookup)
                and len(lookup) // 10000
                != (len(lookup) - len(item["selected_candidate_message_ids"])) // 10000
            ):
                print(
                    json.dumps({"stage": "messages_written", "targets": len(lookup)}),
                    flush=True,
                )
    assert stats["targets"] == limit and len(lookup) == limit

    # Full reread: originals, strict context prefixes, exact frozen SYSTEM, and order.
    raw_targets = {}
    with gzip.open(selection_dir / "targets.jsonl.gz", "rb") as stream:
        for line in stream:
            row = json.loads(line)
            raw_targets[row["message_id"]] = row
    packets = {}
    with gzip.open(selection_dir / "contexts.jsonl.gz", "rb") as stream:
        for line in stream:
            record = json.loads(line)
            packets[record["conversation_seq"]] = record["conversation"]
    seen = set()
    characters = Counter()
    for shard in shards:
        count = 0
        with gzip.open(stage / shard["path"], "rb") as stream:
            for line in stream:
                ex = json.loads(line)
                mid = ex["sample_id"]
                meta = ex["metadata"]
                raw = raw_targets[mid]
                assert mid not in seen
                assert meta["selection_order"] == len(seen) + 1
                assert ex["target"] == raw["target_text"]
                assert meta["target_increase_id"] == raw["target_increase_id"]
                assert meta["query_increase_id"] == raw["query_increase_id"]
                assert (
                    meta["character_id"] == raw["character_id"]
                    and meta["language"] == raw["language"]
                )
                packet = packets[meta["conversation_seq"]]
                assert ex == make_example(
                    packet, mid, ranks[mid], hashes[meta["conversation_seq"]]
                )
                assert meta["input_sha256"] == digest(ex["messages"])
                assert set(ex["messages"][0]) == {"role", "content"}
                assert (
                    ex["messages"][0]["role"] == "system"
                    and ex["messages"][-1]["role"] == "user"
                )
                positions = {m["id"]: i for i, m in enumerate(packet["messages"])}
                assert all(
                    positions[m] < positions[meta["assistant_turn_id"]]
                    for m in meta["history_turn_ids"]
                )
                assert meta["assistant_turn_id"] not in meta["history_turn_ids"]
                characters[meta["character_id"]] += 1
                seen.add(mid)
                count += 1
        assert count == shard["records"]
        shard.update(
            bytes=(stage / shard["path"]).stat().st_size,
            sha256=sha256_file(stage / shard["path"]),
        )
    assert seen == set(ranks) and len(seen) == limit
    assert max(characters.values()) <= character_cap
    assert len(characters) == selection["selected_characters"]
    verification = {
        "status": "passed",
        "records_checked": len(seen),
        "unique_target_ids": len(seen),
        "max_targets_per_character": max(characters.values()),
        "role_cap": character_cap,
        "exact_original_target_match": True,
        "exact_frozen_system_match": True,
        "history_is_strict_prefix": True,
        "future_messages_injected": False,
        "current_target_event_in_input_messages": False,
        "query_platform_guideline_reappended": False,
        "legacy_role_prompt_used": False,
        "shard_counts_verified": True,
        "no_model_calls": True,
    }
    stats_report = json.loads(
        (selection_dir / "statistics.json").read_text(encoding="utf-8")
    )
    if previous_ranking is not None:
        with gzip.open(previous_ranking, "rb") as stream:
            old_ids = {json.loads(line)["message_id"] for line in stream}
        verification["overlap_with_previous_ranking"] = len(old_ids & seen)
        verification["replaced_from_previous_ranking"] = len(old_ids - seen)
    training_stats = {
        **dict(stats),
        "input_chars": quantiles(input_chars),
        "input_message_counts": quantiles(message_counts),
        "partial_conversations": sum(
            r["partial_conversation_selection"] for r in selected_plan
        ),
    }
    dump(stage / "audit/verification.json", verification)
    dump(stage / "audit/training_statistics.json", training_stats)
    dump(stage / "docs/schema.json", SCHEMA)
    (stage / "tools/read_targets.py").write_text(READER, encoding="utf-8")
    for name in (
        "export_target_training_bundle.py",
        "select_ranked_conversation_targets.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), stage / "tools" / name)
    selection["output_directory"] = str(output / "audit/selection")
    dump(selection_dir / "SUMMARY.json", selection)
    # Refresh the nested manifest after replacing its staging path with the final path.
    dump(
        selection_dir / "FILES.json",
        [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(selection_dir.iterdir())
            if p.is_file() and p.name != "FILES.json"
        ],
    )
    manifest = {
        "dataset_id": output.name,
        "schema_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_signature": signature,
        "source_job": job_id,
        "data_hash": selection["selection_signature"]["data_hash"],
        "rubric_hash": selection["selection_signature"]["rubric_hash"],
        "records": limit,
        "conversations": selection["selected_conversations"],
        "characters": selection["selected_characters"],
        "users": selection["selected_users"],
        "character_cap": character_cap,
        "cutoff_mean": selection["cutoff_mean"],
        "mean_conversation_score_weighted_by_target": stats_report[
            "target_weighted_conversation_means"
        ]["mean"],
        "quality_flags": selection["quality_flags"],
        "shards": shards,
        "split": "unsplit",
        "supervision": "target_only",
        "target_independently_scored": False,
        "historical_request_replay": False,
        "paid_requests": 0,
        "output_directory": str(output),
        "verification": "audit/verification.json",
    }
    dump(stage / "MANIFEST.json", manifest)
    write_docs(stage, output, manifest, stats_report, training_stats, verification)
    dump(
        stage / "FILES.json",
        [
            {
                "path": p.relative_to(stage).as_posix(),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in sorted(stage.rglob("*"))
            if p.is_file() and p != stage / "FILES.json"
        ],
    )
    verify_release(stage)
    # mkdtemp starts at 0700; release owner/group must be able to traverse audit too.
    for path in stage.rglob("*"):
        path.chmod(0o750 if path.is_dir() else 0o640)
    stage.chmod(0o750)
    stage.rename(output)
    return manifest


def verify_release(root):
    files = json.loads((root / "FILES.json").read_text(encoding="utf-8"))
    for entry in files:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path leaves release directory")
        if (
            path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("File checksum mismatch: " + entry["path"])
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "FILES.json"
    }
    if actual != {r["path"] for r in files}:
        raise ValueError("Untracked or missing release files")


def write_docs(stage, output, manifest, stats, training, verification):
    (stage / "README.md").write_text(
        f"""# 角色上限 {manifest["character_cap"]} · 前 {manifest["records"]:,} 条 target

这是版本化的数据交付，不替换原始粗筛池或已有发布目录。数据尚未划分训练/验证集，也未启动训练。

## 快速使用

主文件：`data/targets-00000-of-{len(manifest["shards"]):05d}.jsonl.gz` 至最后一片，共 {len(manifest["shards"])} 片。UTF-8，gzip 压缩 JSONL，每行一个独立 target 样本。按 MANIFEST.json 中的 shards 顺序读取，共 {manifest["records"]:,} 条，不要把 audit 中的原始数据再拼进来。

```python
import sys
sys.path.insert(0, '{output}/tools')
from read_targets import iter_targets
for row in iter_targets('{output}'):
    input_messages = row['messages']  # SYSTEM + Greeting + 历史 + 本轮 User
    label = row['target']            # 本轮 Assistant 原文，仅此处计算训练损失
```

`messages` 不含当前 target，也不含它之后的消息。`target` 是字符串，不是评分、不是候选列表。上下文中以前的 Assistant 回复和 Greeting 只作为输入，不能自动一起计入训练损失。metadata 是审计信息，不要发给待训练模型。

## 结果

- {manifest["records"]:,} 条 target / {manifest["conversations"]:,} 个会话 / {manifest["characters"]:,} 个角色 / {manifest["users"]:,} 个用户。
- 每个 character_id 跨语言、跨用户最多 {manifest["character_cap"]} 条。前 10 角色占 {stats["selected"]["concentration"]["character"]["top_share"]["10"]:.2%}。
- 截止会话均分 {manifest["cutoff_mean"]:.6f}/3；按 target 条数加权后的会话均分 {manifest["mean_conversation_score_weighted_by_target"]:.6f}/3。
- 与指定历史排名的 target 重合 {verification.get("overlap_with_previous_ranking", "未比较")} 条。
- 有 Greeting 的 target：{training["with_greeting"]:,}；输入中包含粗筛未选历史 Assistant 的 target：{training["with_background_assistant_history"]:,}。
- 没有新增 LLM 请求，费用为 0；完成逐条 target 原文、SYSTEM、前缀历史、唯一 ID、上限和文件哈希校验。

## 目录与管理

- `data/`：唯一训练主数据，分片 context + target。
- `MANIFEST.json`：版本、数量、来源任务/规则哈希、分片顺序、SHA-256。
- `FILES.json`：除自身之外全部文件的大小与 SHA-256。
- `docs/FORMAT.md` / `docs/schema.json`：字段和消息拼装说明。
- `docs/SELECTION.md`：评分排序、角色配额、边界与使用限制。
- `docs/DISTRIBUTIONS.md`：语言、角色、用户、模型与长度分布。
- `audit/selection/`：原粗筛 target Parquet/JSONL、排名、全会话上下文、分数/风险标记、完整统计；不是另一份训练 split。
- `audit/verification.json` / `audit/training_statistics.json`：逐条校验结果与输入规模。
- `tools/`：无需额外依赖的读取器，以及可重现导出代码。

保持本版本只读使用，不编辑分片。任何配额、评分筛选或拼装变更请新建版本目录，并保留旧版与新旧 manifest。导出先在 `.building-*` 目录完成，校验通过后原子改名；该正式目录存在即表示本次构建通过验证。未建立易变 latest 指针。

数据含原始角色扮演文本、角色资料及用户自填 Persona。只供授权团队使用；不放到无认证公网下载，不把敏感原文发到外部分析服务。user_key 是已有的用户散列，不是用户名，但仍应按可关联标识保护。目录权限 0750、文件 0640，所有者及所属组可读取；不向其他系统用户或公网开放。
""",
        encoding="utf-8",
    )
    (stage / "docs/FORMAT.md").write_text(
        """# 格式与拼装

## 每行结构

以下仅为字段示意，占位内容不是数据样例：

```json
{
  "schema_version": "roleplay-target-messages-v1",
  "sample_id": "原始 target message_id",
  "messages": [
    {"role": "system", "content": "已拼好的平台 SYSTEM 全文"},
    {"role": "assistant", "content": "角色资料中的已有 Greeting"},
    {"role": "user", "content": "此前用户消息"},
    {"role": "assistant", "content": "此前回复，仅作为上下文"},
    {"role": "user", "content": "当前用户原文，不追加平台 guideline"}
  ],
  "target": "当前 Assistant 回复原文",
  "metadata": {"target_independently_scored": false}
}
```

首条必须是 system，末条必须是当前 user。Greeting 存在时以 assistant 消息放在 system 后；没有真实资料则省略并标记，不编造开场。如果同一开场已在冻结历史中被标为 displayed_as_greeting，不再重复放入历史。

## SYSTEM 来源

按当前 target 的 A 回合 ID 匹配冻结 `system_contexts[].applies_to`，必须唯一匹配，完整复制对应 content；不从其他候选或另一语言复制。该 content 已通过旧平台共享的 `build_platform_system_prompt()` 拼装，含原模板分支、角色资料、Persona、语言等配置。不是 Judge 评分提示词，不包含评分标准、模型理由或最终分数。

沿用已送评的 `platform-assembly-v1` 快照，没有重新查询在线角色资料或再次生成。`character_descriptions.prompt` / `characters.prompt` 不参与拼装，也不重新拼进去。数据库里的角色介绍/性格等事实字段是允许的，不等于旧整段 prompt 字段。

目标语言以对应 SYSTEM 的 response_language 和该 target 的平台要求为准，不以用户正文的语言推断。旧 query 后附的 platform guideline 已在冻结索引中分离，训练输入不接回 query；平台输出要求作为 metadata.platform_output_requirements 保留，SYSTEM 原文不另行改写。

## 时间边界与训练损失

从冻结对话取当前 target 之前的严格消息前缀，保留此前所有 User/Assistant，包括粗筛未留下的历史回复。当前 target 只放 target 字段，后续消息完全不带。上下文中曾被评为候选的早期 Assistant 仍只是本条样本的输入。

不注入整段会话结束后的摘要，不重新截取线上记忆窗口，不做上下文长度裁剪。角色资料、Persona、配置和 Greeting 是既有资料快照，不能证明等同每个历史生成时刻的原始请求；即使严格删除后文，也不将该资料快照称为历史逐字重放。

使用 SFT 框架时，需要让 **messages 全部不计算 loss，仅 target 计算 loss**。`tools/read_targets.py` 的 as_chat_messages() 可追加最终 assistant 得到标准聊天数组，但不替你配置框架的 loss mask；直接使用“所有 assistant 都训练”的默认设置会误训练背景回复。切分验证集至少按 conversation_id 分组；若验证跨用户或跨角色泛化，还需按 user_key 或 character_id 分组。不要按 target 行随机拆分，同一会话的上下文相互重叠。

## 主要 metadata 字段

| 字段 | 含义 |
|---|---|
| selection_order | 本版本全局 1 起始顺序，分片顺序也是此顺序 |
| conversation_id / conversation_seq | 来源会话和平台展示序号 |
| character_id / user_key | 角色 ID、冻结用户 SHA-256 散列 |
| language / response_language | 来源 target 语言代码、平台规定输出语言名称 |
| source_model | 生成 target 的来源模型，不是 Judge 模型 |
| assistant_turn_id / target_increase_id / query_increase_id | 目标回合及源消息位置 |
| history_turn_ids / history_source_increase_ids | 进入本条输入的历史回合及源位置，不含 Greeting、SYSTEM |
| system_context_id / system_template / system_sha256 | 对应的冻结 SYSTEM、模板与正文哈希 |
| source_context_hash / input_sha256 / target_sha256 | 冻结整段上下文、此条消息数组、target 原文哈希 |
| conversation_mean_score | 会话组数字项均分，不是当前 target 独立得分 |
| conversation_numeric_items / conversation_na_items / conversation_low_items | 会话组的数字评分项、N/A 项、0 或 1 分项数量 |
| conversation_major_issue | 会话有重大问题标记，不表示本条必然有问题 |
| target_independently_scored | 固定 false，不冒充单条质量分 |

JSONL 用 gzip 解压后按换行读，每一行解析一个 JSON。正文包含换行和 Unicode，不能用字符串 splitlines() 对整份已解码文本粗切；可用 gzip.open(path, 'rb') 逐行读取。未使用模型专用聊天模板、特殊 token 或分词器，由训练框架按目标模型处理。
""",
        encoding="utf-8",
    )
    (stage / "docs/SELECTION.md").write_text(
        f"""# 筛选规则

来源任务 `{manifest["source_job"]}` 的冻结会话评分。所有数字项等权求均分，N/A 不按 0，也不计入分母。没有加入置信度、logprobs 或可学习权重。

排除未完成、没有完整合法评分、没有数字项、疑似整段拒评的会话。其他低分、警告、资料限制、major 问题仍保留标签，不在本次临时加一票否决。

按精确分数（评分和/数字项数）降序；同分按原会话 seq 升序。在这个顺序下选 target，同一 character_id 总计至多 {manifest["character_cap"]} 条，跨用户、跨语言统一计数。会话内按原回合顺序取；配额或总数到达时可能只取该会话的一部分。每个入选 target 独立保留完整前缀，未入选历史不会因此消失。

取满 {manifest["records"]:,} 条，截止均分 {manifest["cutoff_mean"]:.6f}。没有角色均等配额、用户上限、语言配额或最低均分硬阈值；本版本的每角色上限为 {manifest["character_cap"]}，不同数据和配额可以产生不同截止分。要改变质量门槛，应重新出版本，不暗中放宽。

## 使用限制

这是按会话分数并控制角色富集选出的候选，不是逐条审核通过证明。

- 来自有低分项的会话：{manifest["quality_flags"]["targets_from_low_score_conversations"]:,} 条。
- 来自有 major 问题标记的会话：{manifest["quality_flags"]["targets_from_major_issue_conversations"]:,} 条。
- 来自超过 10 项 N/A 的会话：{manifest["quality_flags"]["targets_from_more_than_10_na_conversations"]:,} 条。

以上标记不能机械当作每条 target 本身有问题，但均分会掩盖局部缺点，大量 N/A 的均分可比性较弱。配额只解决按 character_id 的集中，不等于题材、人设、表达和用户都均衡。不同 ID 的相似角色未合并；相同来源角色跨语言共享上限。

上下文完全复制既有资料和历史，target 原文不改写、不续写。没有生成新内容，也没有为规避模型拒评而重写提示词。训练使用前仍应按实际用途和安全要求决定风险标记的处理。

## 可重现

在原共享服务器中，使用独立 PyArrow 环境和 tools 内冻结脚本运行（必须使用新的输出路径）：

```bash
python -m src.evaluation.roleplay_annotation_platform.export_target_training_bundle --root RUN_ROOT --job-id {manifest["source_job"]} --limit {manifest["records"]} --character-cap {manifest["character_cap"]} --shard-size {manifest["build_signature"]["shard_size"]} --output NEW_VERSION_DIRECTORY
```

需要可读的原始粗筛 Parquet、冻结索引及评分数据库。代码不读取 API key、不访问 LLM。MANIFEST 固定来源数据、评分量表和构建代码哈希。同目录已有不同构建签名时拒绝覆盖。
""",
        encoding="utf-8",
    )
    lines = [
        "# 分布与规模",
        "",
        "所有占比除特别说明外都按 target 条数计算。原池为 229,160 条粗筛数据；模型是 target 的来源模型。",
        "",
    ]
    for name, key in [("语言", "language"), ("来源模型", "model")]:
        lines.extend(
            [
                f"## {name}",
                "",
                "| 标识 | 入选条数 | 入选占比 | 原池占比 |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in stats["distributions"][key]:
            if row["selected_targets"]:
                lines.append(
                    f"| {row['id']} | {row['selected_targets']:,} | {row['selected_share']:.2%} | {row['pool_share']:.2%} |"
                )
        lines.append("")
    lines.extend(["## 集中度", "", "| 指标 | 原池 | 入选集 |", "|---|---:|---:|"])
    for key, name in [("character", "角色"), ("user", "用户")]:
        before = stats["pool"]["concentration"][key]
        after = stats["selected"]["concentration"][key]
        lines.extend(
            [
                f"| {name}数 | {before['groups']:,} | {after['groups']:,} |",
                f"| 单个{name}最多条数 | {before['max_group']:,} | {after['max_group']:,} |",
            ]
        )
        for n in ("10", "50", "100"):
            lines.append(
                f"| 前{n}个{name}占比 | {before['top_share'][n]:.2%} | {after['top_share'][n]:.2%} |"
            )
    lines.extend(
        [
            "",
            "## 输入长度",
            "",
            "| 项目 | 中位数 | P95 | 最大值 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, v in [
        ("输入字符数（含 SYSTEM）", training["input_chars"]),
        ("输入消息数", training["input_message_counts"]),
        ("target 字符数", stats["selected"]["lengths"]["target_chars"]),
    ]:
        lines.append(
            f"| {name} | {v['median']:,.1f} | {v['p95']:,.1f} | {v['max']:,} |"
        )
    lines.extend(
        [
            "",
            "字符数不是 Token 数。训练前按实际 tokenizer 测长度；若裁剪上下文，应另存新版本并说明策略。完整角色/用户分布与分项评分分布见 audit/selection/statistics.json。",
            "",
        ]
    )
    (stage / "docs/DISTRIBUTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--character-cap", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument(
        "--previous-ranking",
        type=Path,
        help="Optional previous target_ranking.jsonl.gz for overlap audit",
    )
    args = parser.parse_args()
    os.umask(0o027)
    print(
        json.dumps(
            build_bundle(
                args.root,
                args.job_id,
                args.output,
                args.limit,
                args.character_cap,
                args.shard_size,
                args.previous_ranking,
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
