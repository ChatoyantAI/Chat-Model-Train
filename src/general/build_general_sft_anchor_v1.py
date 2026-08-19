#!/usr/bin/env python3
"""Build a deterministic 100k general SFT anchor set for LusyChat.

This script only downloads public Hugging Face datasets. It does not connect to
or write any LusyChat production service.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import gc
import hashlib
import heapq
import json
import math
import os
import pathlib
import random
import re
import shutil
import sys
import time
import unicodedata
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from datasets import load_dataset
from huggingface_hub import HfApi


SEED = "lusy-general-anchor-v1-20260724"
SCHEMA_VERSION = "lusy-general-sft-anchor-v1"
DEFAULT_ROOT = pathlib.Path("/home/gyr/lusychat-training/general-sft-anchor-v1")

SOURCE_TARGETS = collections.OrderedDict(
    [
        ("aya", 27_000),
        ("oasst2", 23_000),
        ("tulu_if", 15_000),
        ("ultrachat", 15_000),
        ("smoltalk", 10_000),
        ("openmath", 7_000),
        ("bigcode", 3_000),
    ]
)

LANGUAGE_TARGETS = {
    "en": 60_750,
    "es": 9_100,
    "pt": 5_800,
    "fr": 1_750,
    "de": 1_460,
    "zh": 5_300,
    "ja": 4_100,
    "ru": 3_290,
    "other": 8_450,
}

AYA_LANGUAGE_TARGETS = {
    "en": 2_850,
    "es": 2_800,
    "pt": 5_200,
    "fr": 950,
    "de": 160,
    "zh": 3_700,
    "ja": 4_100,
    "ru": 290,
    "other": 6_950,
}

OASST_LANGUAGE_TARGETS = {
    "en": 7_900,
    "es": 6_300,
    "pt": 600,
    "fr": 800,
    "de": 1_300,
    "zh": 1_600,
    "ja": 0,
    "ru": 3_000,
    "other": 1_500,
}

SMOLTALK_CONFIG_TARGETS = {
    "smol-constraints": 3_000,
    "smol-rewrite": 3_000,
    "smol-summarize": 4_000,
}

SOURCE_SPECS = {
    "aya": {
        "repo": "CohereLabs/aya_collection",
        "config": "aya_dataset",
        "split": "train",
        "license": "apache-2.0",
        "provenance": "human_annotated",
        "license_review": "declared_permissive",
    },
    "oasst2": {
        "repo": "OpenAssistant/oasst2",
        "config": "default",
        "split": "train+validation",
        "license": "apache-2.0",
        "provenance": "human_feedback",
        "license_review": "declared_permissive",
    },
    "tulu_if": {
        "repo": "allenai/tulu-3-sft-personas-instruction-following",
        "config": "default",
        "split": "train",
        "license": "odc-by-1.0",
        "provenance": "synthetic_verifiable_constraints",
        "license_review": "attribution_required",
    },
    "ultrachat": {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "config": "default",
        "split": "train_sft",
        "license": "mit",
        "provenance": "synthetic_chatgpt_filtered",
        "license_review": "dataset_card_permissive_output_terms_review",
    },
    "smoltalk": {
        "repo": "HuggingFaceTB/smoltalk",
        "config": "selected_configs",
        "split": "train",
        "license": "apache-2.0",
        "provenance": "synthetic_task_specific",
        "license_review": "selected_new_subsets_only",
    },
    "openmath": {
        "repo": "nvidia/OpenMathInstruct-2",
        "config": "default",
        "split": "train_1M",
        "license": "cc-by-4.0",
        "provenance": "synthetic_math_with_expected_answer",
        "license_review": "attribution_required",
        "selection_scope": "pinned_stream_shuffle_buffer_50000_scan_150000",
    },
    "bigcode": {
        "repo": "bigcode/self-oss-instruct-sc2-exec-filter-50k",
        "config": "default",
        "split": "train",
        "license": "odc-by",
        "provenance": "synthetic_execution_filtered_code",
        "license_review": "attribution_required",
    },
}

ISO_LANGUAGE_MAP = {
    "eng": "en",
    "en": "en",
    "spa": "es",
    "es": "es",
    "por": "pt",
    "pt": "pt",
    "pt-br": "pt",
    "fra": "fr",
    "fre": "fr",
    "fr": "fr",
    "deu": "de",
    "ger": "de",
    "de": "de",
    "zho": "zh",
    "cmn": "zh",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "jpn": "ja",
    "ja": "ja",
    "rus": "ru",
    "ru": "ru",
}

ROLE_MAP = {
    "user": "user",
    "human": "user",
    "prompter": "user",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
}

AI_IDENTITY_PATTERNS = [
    re.compile(r"\bas an ai(?: language model| assistant)?\b", re.I),
    re.compile(r"\bi(?:'m| am) (?:just )?an? (?:ai|chatbot|language model)\b", re.I),
    re.compile(r"\bmy (?:training data|developer|provider|system prompt)\b", re.I),
    re.compile(r"作为(?:一个)?(?:人工智能|ai|语言模型)", re.I),
    re.compile(r"我是(?:一个)?(?:人工智能|ai|语言模型|聊天机器人)", re.I),
    re.compile(r"como (?:una|un) (?:ia|inteligencia artificial|modelo de lenguaje)", re.I),
    re.compile(r"en tant qu['’](?:ia|intelligence artificielle|assistant ia)", re.I),
    re.compile(r"als (?:ki|künstliche intelligenz|sprachmodell)", re.I),
    re.compile(r"я (?:всего лишь )?(?:ии|искусственный интеллект|языковая модель)", re.I),
]

THINK_BLOCK_RE = re.compile(r"^\s*<think\b[^>]*>.*?</think>\s*", re.I | re.S)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.I)
MODEL_TOKEN_RE = re.compile(r"<\|(?:assistant|user|system|im_start|im_end)[^>]*\|>", re.I)
WHITESPACE_RE = re.compile(r"\s+")

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("source_dataset", pa.string(), nullable=False),
        pa.field("source_config", pa.string(), nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_license", pa.string(), nullable=False),
        pa.field("license_review", pa.string(), nullable=False),
        pa.field("provenance_type", pa.string(), nullable=False),
        pa.field("language_raw", pa.string(), nullable=False),
        pa.field("language_final", pa.string(), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field(
            "messages",
            pa.large_list(
                pa.struct(
                    [
                        pa.field("role", pa.string(), nullable=False),
                        pa.field("content", pa.large_string(), nullable=False),
                        pa.field("weight", pa.int8(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("target_text", pa.large_string(), nullable=False),
        pa.field("prompt_chars", pa.int32(), nullable=False),
        pa.field("target_chars", pa.int32(), nullable=False),
        pa.field("total_chars", pa.int32(), nullable=False),
        pa.field("conversation_turns", pa.int16(), nullable=False),
        pa.field("cot_removed", pa.bool_(), nullable=False),
        pa.field("explicit_cot_present", pa.bool_(), nullable=False),
        pa.field("ai_identity_break", pa.bool_(), nullable=False),
        pa.field("target_loss_only", pa.bool_(), nullable=False),
        pa.field("training_mode", pa.string(), nullable=False),
        pa.field("system_prompt_policy", pa.string(), nullable=False),
        pa.field("verification_type", pa.string(), nullable=False),
        pa.field("verification_payload_json", pa.large_string(), nullable=False),
        pa.field("source_metadata_json", pa.large_string(), nullable=False),
        pa.field("quality_flags", pa.list_(pa.string()), nullable=False),
        pa.field("prompt_fingerprint", pa.string(), nullable=False),
        pa.field("target_fingerprint", pa.string(), nullable=False),
        pa.field("deterministic_key", pa.uint64(), nullable=False),
        pa.field("quota_bucket", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_uint64(*parts: str) -> int:
    payload = "\x1f".join((SEED, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def normalize_for_dedupe(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_language(value: Any) -> str:
    if value is None:
        return "other"
    raw = str(value).strip().lower().replace("_", "-")
    if raw in ISO_LANGUAGE_MAP:
        return ISO_LANGUAGE_MAP[raw]
    head = raw.split("-", 1)[0]
    return ISO_LANGUAGE_MAP.get(head, "other")


def scale_quota(targets: dict[str, int], total: int) -> dict[str, int]:
    positive = {key: value for key, value in targets.items() if value > 0}
    denominator = sum(positive.values())
    raw = {key: total * value / denominator for key, value in positive.items()}
    scaled = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(scaled.values())
    for key in sorted(positive, key=lambda item: (-(raw[item] - scaled[item]), item))[:remainder]:
        scaled[key] += 1
    return scaled


def strip_explicit_think(text: str) -> tuple[str, bool, bool]:
    original = text
    text = THINK_BLOCK_RE.sub("", text, count=1)
    removed = text != original
    remains = bool(THINK_TAG_RE.search(text))
    return text.strip(), removed, remains


def has_ai_identity_break(text: str) -> bool:
    return any(pattern.search(text) for pattern in AI_IDENTITY_PATTERNS)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def normalize_task_type(value: Any, source_key: str) -> str:
    task_type = safe_text(value)
    if task_type.casefold() in {"", "-", "none", "null", "unknown"}:
        return "multilingual_general_instruction" if source_key == "aya" else "general_instruction"
    return task_type


def canonicalize_messages(raw_messages: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    output: list[dict[str, Any]] = []
    flags: list[str] = []
    seen_user = False
    seen_non_system = False

    for raw in raw_messages:
        role = ROLE_MAP.get(str(raw.get("role", "")).strip().lower())
        if role is None:
            flags.append("unsupported_role")
            return [], flags
        content = safe_text(raw.get("content", raw.get("text", "")))
        if not content:
            flags.append("empty_message")
            return [], flags
        if MODEL_TOKEN_RE.search(content):
            flags.append("model_control_token")
            return [], flags

        if role == "system":
            if seen_non_system:
                flags.append("late_system_message")
                return [], flags
        else:
            seen_non_system = True
        if role == "user":
            seen_user = True

        removed = False
        remains = False
        if role == "assistant":
            content, removed, remains = strip_explicit_think(content)
            if removed:
                flags.append("cot_removed")
            if remains:
                flags.append("explicit_cot_remaining")
                return [], flags
            if not content:
                flags.append("empty_after_cot_removal")
                return [], flags

        output.append({"role": role, "content": content, "weight": 0})

    if not seen_user or not output or output[-1]["role"] != "assistant":
        flags.append("invalid_terminal_role")
        return [], flags

    non_system_roles = [item["role"] for item in output if item["role"] != "system"]
    for previous, current in zip(non_system_roles, non_system_roles[1:]):
        if previous == current:
            flags.append("non_alternating_roles")
            return [], flags

    output[-1]["weight"] = 1
    return output, flags


def build_candidate(
    *,
    source_key: str,
    revision: str,
    source_id: Any,
    source_config: str,
    language_raw: Any,
    task_type: str,
    raw_messages: Iterable[dict[str, Any]],
    quota_bucket: str,
    verification_type: str = "none",
    verification_payload: dict[str, Any] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    spec = SOURCE_SPECS[source_key]
    messages, flags = canonicalize_messages(raw_messages)
    if not messages:
        return None, flags[-1] if flags else "invalid_messages"

    target = messages[-1]["content"]
    if has_ai_identity_break(target):
        return None, "ai_identity_break"
    if len(target) < 2:
        return None, "target_too_short"
    if len(target) > 24_000:
        return None, "target_too_long"

    prompt_messages = messages[:-1]
    prompt_chars = sum(len(message["content"]) for message in prompt_messages)
    total_chars = prompt_chars + len(target)
    if prompt_chars > 60_000 or total_chars > 80_000:
        return None, "context_too_long"

    previous_assistants = [
        normalize_for_dedupe(message["content"])
        for message in messages[:-1]
        if message["role"] == "assistant"
    ]
    target_norm = normalize_for_dedupe(target)
    if target_norm in previous_assistants:
        return None, "target_repeats_history_assistant"

    last_user = next(
        (message["content"] for message in reversed(messages[:-1]) if message["role"] == "user"),
        "",
    )
    if not last_user:
        return None, "missing_last_user"

    source_id_text = str(source_id)
    sample_hash = sha256_text(
        "\x1f".join((spec["repo"], source_config, source_id_text, last_user, target))
    )
    language_final = normalize_language(language_raw)
    deterministic_key = stable_uint64(source_key, source_config, source_id_text, sample_hash)
    cot_removed = "cot_removed" in flags

    row = {
        "sample_id": f"gen_{source_key}_{sample_hash[:24]}",
        "schema_version": SCHEMA_VERSION,
        "source_key": source_key,
        "source_dataset": spec["repo"],
        "source_config": source_config,
        "source_split": spec["split"],
        "source_revision": revision,
        "source_id": source_id_text,
        "source_license": spec["license"],
        "license_review": spec["license_review"],
        "provenance_type": spec["provenance"],
        "language_raw": safe_text(language_raw) or "unknown",
        "language_final": language_final,
        "task_type": normalize_task_type(task_type, source_key),
        "messages": messages,
        "target_text": target,
        "prompt_chars": prompt_chars,
        "target_chars": len(target),
        "total_chars": total_chars,
        "conversation_turns": sum(1 for message in messages if message["role"] == "assistant"),
        "cot_removed": cot_removed,
        "explicit_cot_present": False,
        "ai_identity_break": False,
        "target_loss_only": True,
        "training_mode": "general",
        "system_prompt_policy": "source_preserved_no_injection",
        "verification_type": verification_type,
        "verification_payload_json": json_dumps(verification_payload or {}),
        "source_metadata_json": json_dumps(source_metadata or {}),
        "quality_flags": sorted(set(flags)),
        "prompt_fingerprint": sha256_text(normalize_for_dedupe(last_user)),
        "target_fingerprint": sha256_text(target_norm),
        "deterministic_key": deterministic_key,
        "quota_bucket": quota_bucket,
        "split": "train",
    }
    return row, None


class StablePool:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self.heap: list[tuple[int, str, dict[str, Any]]] = []

    def add(self, item: dict[str, Any]) -> None:
        if self.capacity <= 0:
            return
        key = int(item["deterministic_key"])
        tie = item["sample_id"]
        entry = (-key, tie, item)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
            return
        worst_key = -self.heap[0][0]
        if key < worst_key:
            heapq.heapreplace(self.heap, entry)

    def rows(self) -> list[dict[str, Any]]:
        return sorted((entry[2] for entry in self.heap), key=lambda row: (row["deterministic_key"], row["sample_id"]))


class RejectionCounter:
    def __init__(self) -> None:
        self.counter: collections.Counter[str] = collections.Counter()
        self.seen = 0
        self.accepted = 0

    def reject(self, reason: str | None) -> None:
        self.seen += 1
        if reason is None:
            self.accepted += 1
        else:
            self.counter[reason] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "accepted_before_sampling": self.accepted,
            "rejected": dict(self.counter.most_common()),
        }


def resolve_revisions(api: HfApi) -> dict[str, str]:
    revisions: dict[str, str] = {}
    for source_key, spec in SOURCE_SPECS.items():
        repo = spec["repo"]
        if repo in revisions:
            continue
        log(f"resolving revision {repo}")
        info = api.dataset_info(repo, revision="main")
        revisions[repo] = info.sha
    return revisions


def dataset_stream(
    repo: str,
    config: str | None,
    split: str,
    revision: str,
    cache_dir: pathlib.Path,
) -> Iterable[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "path": repo,
        "split": split,
        "revision": revision,
        "streaming": True,
        "cache_dir": str(cache_dir),
    }
    if config and config != "default":
        kwargs["name"] = config
    return load_dataset(**kwargs)


def build_aya(root: pathlib.Path, revision: str, smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capacities = {
        language: min(target if smoke else target + max(300, math.ceil(target * 0.20)), target + 1_500)
        for language, target in AYA_LANGUAGE_TARGETS.items()
    }
    if smoke:
        smoke_quota = scale_quota(AYA_LANGUAGE_TARGETS, 100)
        capacities = {
            language: target + max(3, math.ceil(target * 0.25))
            for language, target in smoke_quota.items()
        }
    pools = {language: StablePool(capacity) for language, capacity in capacities.items()}
    stats = RejectionCounter()

    stream = dataset_stream(
        SOURCE_SPECS["aya"]["repo"],
        "aya_dataset",
        "train",
        revision,
        root / "01_raw_cache" / "hf",
    )
    for index, record in enumerate(stream):
        language_raw = record.get("language")
        bucket = normalize_language(language_raw)
        if bucket not in pools:
            bucket = "other"
        candidate, reason = build_candidate(
            source_key="aya",
            revision=revision,
            source_id=record.get("id", index),
            source_config="aya_dataset",
            language_raw=language_raw,
            task_type=safe_text(record.get("task_type")) or "general_instruction",
            raw_messages=[
                {"role": "user", "content": record.get("inputs")},
                {"role": "assistant", "content": record.get("targets")},
            ],
            quota_bucket=bucket,
            source_metadata={
                "dataset_name": record.get("dataset_name"),
                "sub_dataset_name": record.get("sub_dataset_name"),
                "script": record.get("script"),
            },
        )
        stats.reject(reason)
        if candidate is not None:
            pools[bucket].add(candidate)

    rows = [row for pool in pools.values() for row in pool.rows()]
    return rows, stats.as_dict()


def load_oasst_rows(root: pathlib.Path, revision: str) -> list[dict[str, Any]]:
    dataset = load_dataset(
        SOURCE_SPECS["oasst2"]["repo"],
        revision=revision,
        cache_dir=str(root / "01_raw_cache" / "hf"),
    )
    rows: list[dict[str, Any]] = []
    for split_name in dataset:
        for record in dataset[split_name]:
            row = dict(record)
            row["_split"] = split_name
            rows.append(row)
    return rows


def build_oasst2(root: pathlib.Path, revision: str, smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capacities = {
        language: target + max(500, math.ceil(target * 0.60))
        for language, target in OASST_LANGUAGE_TARGETS.items()
        if target > 0
    }
    if smoke:
        smoke_quota = scale_quota(OASST_LANGUAGE_TARGETS, 100)
        capacities = {
            language: target + max(5, math.ceil(target * 0.50))
            for language, target in smoke_quota.items()
        }
    pools = {language: StablePool(capacity) for language, capacity in capacities.items()}
    stats = RejectionCounter()
    rows = load_oasst_rows(root, revision)
    by_id = {str(row["message_id"]): row for row in rows}

    def accepted(row: dict[str, Any]) -> bool:
        if bool(row.get("deleted")):
            return False
        if row.get("review_result") is False:
            return False
        return True

    for index, target in enumerate(rows):
        if safe_text(target.get("role")).lower() != "assistant" or not accepted(target):
            continue
        rank = target.get("rank")
        if rank is not None and int(rank) != 0:
            continue

        path: list[dict[str, Any]] = []
        current: dict[str, Any] | None = target
        visited: set[str] = set()
        while current is not None and len(path) < 64:
            message_id = str(current.get("message_id"))
            if message_id in visited or not accepted(current):
                path = []
                break
            visited.add(message_id)
            path.append(current)
            parent_id = current.get("parent_id")
            if not parent_id:
                break
            current = by_id.get(str(parent_id))
            if current is None:
                path = []
                break
        if not path:
            stats.reject("broken_or_rejected_path")
            continue
        path.reverse()

        target_language_raw = target.get("lang")
        bucket = normalize_language(target_language_raw)
        if bucket not in pools:
            bucket = "other"
        path_languages = {
            normalize_language(node.get("lang"))
            for node in path
            if node.get("lang")
        }
        if any(language not in {bucket, "other"} for language in path_languages):
            stats.reject("path_language_mismatch")
            continue

        raw_messages = [
            {"role": node.get("role"), "content": node.get("text")}
            for node in path
        ]
        candidate, reason = build_candidate(
            source_key="oasst2",
            revision=revision,
            source_id=target.get("message_id", index),
            source_config="default",
            language_raw=target_language_raw,
            task_type="human_multiturn_dialogue",
            raw_messages=raw_messages,
            quota_bucket=bucket,
            verification_type="human_ranked",
            verification_payload={"rank": rank, "review_count": target.get("review_count")},
            source_metadata={
                "message_tree_id": target.get("message_tree_id"),
                "target_split": target.get("_split"),
                "synthetic": target.get("synthetic"),
            },
        )
        stats.reject(reason)
        if candidate is not None:
            pools[bucket].add(candidate)

    output = [row for pool in pools.values() for row in pool.rows()]
    return output, stats.as_dict()


def build_simple_messages_source(
    *,
    root: pathlib.Path,
    source_key: str,
    revision: str,
    config: str | None,
    split: str,
    target_capacity: int,
    smoke: bool,
    extractor: Any,
    shuffle_buffer: int | None = None,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capacity = min(100, target_capacity) if smoke else target_capacity + max(1_000, math.ceil(target_capacity * 0.20))
    pool = StablePool(capacity)
    stats = RejectionCounter()
    stream = dataset_stream(
        SOURCE_SPECS[source_key]["repo"],
        config,
        split,
        revision,
        root / "01_raw_cache" / "hf",
    )
    if shuffle_buffer:
        stream = stream.shuffle(
            seed=stable_uint64(source_key, "stream-shuffle") % (2**32),
            buffer_size=shuffle_buffer,
        )
    iterator = iter(stream)
    try:
        for index, record in enumerate(iterator):
            kwargs = extractor(record, index)
            candidate, reason = build_candidate(
                source_key=source_key,
                revision=revision,
                **kwargs,
            )
            stats.reject(reason)
            if candidate is not None:
                pool.add(candidate)
            if smoke and index >= 3_000:
                break
            if not smoke and max_records is not None and index + 1 >= max_records:
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
        del iterator
        del stream
        gc.collect()
    return pool.rows(), stats.as_dict()


def extract_tulu_if(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "source_id": record.get("id", index),
        "source_config": "default",
        "language_raw": "en",
        "task_type": "constraint_following",
        "raw_messages": record.get("messages") or [],
        "quota_bucket": "en",
        "verification_type": "declared_constraints",
        "verification_payload": {"constraints": record.get("constraints") or []},
        "source_metadata": {"prompt": record.get("prompt")},
    }


def extract_ultrachat(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "source_id": record.get("prompt_id", index),
        "source_config": "default",
        "language_raw": "en",
        "task_type": "general_multiturn_dialogue",
        "raw_messages": record.get("messages") or [],
        "quota_bucket": "en",
        "source_metadata": {"prompt_id": record.get("prompt_id")},
    }


def build_smoltalk(root: pathlib.Path, revision: str, smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    combined_stats: dict[str, Any] = {}
    task_map = {
        "smol-constraints": "constraint_following",
        "smol-rewrite": "rewrite",
        "smol-summarize": "summarization",
    }
    for config, target in SMOLTALK_CONFIG_TARGETS.items():
        capacity = min(50, target) if smoke else target + max(500, math.ceil(target * 0.20))
        pool = StablePool(capacity)
        stats = RejectionCounter()
        stream = dataset_stream(
            SOURCE_SPECS["smoltalk"]["repo"],
            config,
            "train",
            revision,
            root / "01_raw_cache" / "hf",
        )
        for index, record in enumerate(stream):
            candidate, reason = build_candidate(
                source_key="smoltalk",
                revision=revision,
                source_id=record.get("id", f"{config}:{index}"),
                source_config=config,
                language_raw="en",
                task_type=task_map[config],
                raw_messages=record.get("messages") or [],
                quota_bucket=config,
                source_metadata={"source": record.get("source")},
            )
            stats.reject(reason)
            if candidate is not None:
                pool.add(candidate)
            if smoke and index >= 2_000:
                break
        output.extend(pool.rows())
        combined_stats[config] = stats.as_dict()
    return output, combined_stats


def extract_openmath(record: dict[str, Any], index: int) -> dict[str, Any]:
    problem = record.get("problem")
    solution = record.get("generated_solution")
    expected = record.get("expected_answer")
    source_id = sha256_text(safe_text(problem))[:24] if problem else index
    return {
        "source_id": source_id,
        "source_config": "default",
        "language_raw": "en",
        "task_type": "math",
        "raw_messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ],
        "quota_bucket": "en",
        "verification_type": "source_expected_answer",
        "verification_payload": {
            "expected_answer": expected,
            "verification_status": "source_reported_not_rechecked",
        },
        "source_metadata": {"problem_source": record.get("problem_source")},
    }


def extract_bigcode(record: dict[str, Any], index: int) -> dict[str, Any]:
    source_id = record.get("sha1") or record.get("id", index)
    return {
        "source_id": source_id,
        "source_config": "default",
        "language_raw": "en",
        "task_type": "code",
        "raw_messages": [
            {"role": "user", "content": record.get("instruction")},
            {"role": "assistant", "content": record.get("response")},
        ],
        "quota_bucket": "en",
        "verification_type": "source_execution_filtered",
        "verification_payload": {"verification_status": "source_reported_execution_filter"},
        "source_metadata": {
            "seed_sha1": record.get("sha1"),
            "concepts": record.get("concepts"),
        },
    }


def write_rows_parquet(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=8,
        use_dictionary=True,
        write_statistics=True,
    )


def read_rows_parquet(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["task_type"] = normalize_task_type(row["task_type"], row["source_key"])
    return rows


def source_candidate_path(root: pathlib.Path, source_key: str) -> pathlib.Path:
    return root / "02_candidates" / f"{source_key}.candidates.parquet"


def build_or_load_sources(
    root: pathlib.Path,
    revisions: dict[str, str],
    smoke: bool,
    reuse: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_stats: dict[str, Any] = {}

    builders = {
        "aya": lambda revision: build_aya(root, revision, smoke),
        "oasst2": lambda revision: build_oasst2(root, revision, smoke),
        "tulu_if": lambda revision: build_simple_messages_source(
            root=root,
            source_key="tulu_if",
            revision=revision,
            config=None,
            split="train",
            target_capacity=SOURCE_TARGETS["tulu_if"],
            smoke=smoke,
            extractor=extract_tulu_if,
        ),
        "ultrachat": lambda revision: build_simple_messages_source(
            root=root,
            source_key="ultrachat",
            revision=revision,
            config=None,
            split="train_sft",
            target_capacity=SOURCE_TARGETS["ultrachat"],
            smoke=smoke,
            extractor=extract_ultrachat,
        ),
        "smoltalk": lambda revision: build_smoltalk(root, revision, smoke),
        "openmath": lambda revision: build_simple_messages_source(
            root=root,
            source_key="openmath",
            revision=revision,
            config=None,
            split="train_1M",
            target_capacity=SOURCE_TARGETS["openmath"],
            smoke=smoke,
            extractor=extract_openmath,
            shuffle_buffer=50_000,
            max_records=150_000,
        ),
        "bigcode": lambda revision: build_simple_messages_source(
            root=root,
            source_key="bigcode",
            revision=revision,
            config=None,
            split="train",
            target_capacity=SOURCE_TARGETS["bigcode"],
            smoke=smoke,
            extractor=extract_bigcode,
        ),
    }

    for source_key, builder in builders.items():
        path = source_candidate_path(root, source_key)
        if reuse and path.exists():
            log(f"loading cached candidates {source_key}")
            rows = read_rows_parquet(path)
            stats = {"reused_candidate_file": True, "candidate_rows": len(rows)}
        else:
            log(f"building source {source_key}")
            revision = revisions[SOURCE_SPECS[source_key]["repo"]]
            rows, stats = builder(revision)
            write_rows_parquet(path, rows)
            log(f"wrote candidates {source_key}: {len(rows)}")
        source_rows[source_key] = rows
        source_stats[source_key] = stats

    return source_rows, source_stats


def source_language_quota(source_key: str) -> dict[str, int] | None:
    if source_key == "aya":
        return AYA_LANGUAGE_TARGETS
    if source_key == "oasst2":
        return OASST_LANGUAGE_TARGETS
    return None


def select_final(
    source_rows: dict[str, list[dict[str, Any]]],
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    seen_prompt: set[str] = set()
    seen_target: set[str] = set()
    seen_pair: set[tuple[str, str]] = set()
    tree_counts: collections.Counter[str] = collections.Counter()
    selection_stats: dict[str, Any] = {}

    def is_duplicate(row: dict[str, Any]) -> str | None:
        pair = (row["prompt_fingerprint"], row["target_fingerprint"])
        if pair in seen_pair:
            return "duplicate_prompt_target"
        if row["prompt_fingerprint"] in seen_prompt:
            return "duplicate_prompt"
        if row["target_fingerprint"] in seen_target:
            return "duplicate_target"
        if row["source_key"] == "oasst2":
            metadata = json.loads(row["source_metadata_json"])
            tree_id = str(metadata.get("message_tree_id") or "")
            if tree_id and tree_counts[tree_id] >= 3:
                return "oasst_tree_cap"
        return None

    def commit(row: dict[str, Any]) -> None:
        final_rows.append(row)
        seen_prompt.add(row["prompt_fingerprint"])
        seen_target.add(row["target_fingerprint"])
        seen_pair.add((row["prompt_fingerprint"], row["target_fingerprint"]))
        if row["source_key"] == "oasst2":
            metadata = json.loads(row["source_metadata_json"])
            tree_id = str(metadata.get("message_tree_id") or "")
            if tree_id:
                tree_counts[tree_id] += 1

    for source_key, full_target in SOURCE_TARGETS.items():
        target = min(100, full_target) if smoke else full_target
        rows = sorted(source_rows[source_key], key=lambda row: (row["deterministic_key"], row["sample_id"]))
        reject_counts: collections.Counter[str] = collections.Counter()
        selected_before = len(final_rows)

        language_quota = source_language_quota(source_key)
        if language_quota is not None:
            scaled_quota = scale_quota(language_quota, target) if smoke else {
                language: value for language, value in language_quota.items() if value > 0
            }
            for language, quota in scaled_quota.items():
                count = 0
                for row in rows:
                    if row["language_final"] != language or count >= quota:
                        continue
                    duplicate_reason = is_duplicate(row)
                    if duplicate_reason:
                        reject_counts[duplicate_reason] += 1
                        continue
                    commit(row)
                    count += 1
                if count < quota:
                    raise RuntimeError(f"{source_key}/{language}: selected {count}, need {quota}")
        elif source_key == "smoltalk":
            scaled_config_targets = scale_quota(SMOLTALK_CONFIG_TARGETS, target) if smoke else SMOLTALK_CONFIG_TARGETS
            for config, quota in scaled_config_targets.items():
                count = 0
                for row in rows:
                    if row["source_config"] != config or count >= quota:
                        continue
                    duplicate_reason = is_duplicate(row)
                    if duplicate_reason:
                        reject_counts[duplicate_reason] += 1
                        continue
                    commit(row)
                    count += 1
                if count < quota:
                    raise RuntimeError(f"smoltalk/{config}: selected {count}, need {quota}")
        else:
            count = 0
            for row in rows:
                if count >= target:
                    break
                duplicate_reason = is_duplicate(row)
                if duplicate_reason:
                    reject_counts[duplicate_reason] += 1
                    continue
                commit(row)
                count += 1
            if count < target:
                raise RuntimeError(f"{source_key}: selected {count}, need {target}")

        selected_count = len(final_rows) - selected_before
        if not smoke and selected_count != full_target:
            raise RuntimeError(f"{source_key}: selected {selected_count}, expected {full_target}")
        selection_stats[source_key] = {
            "selected": selected_count,
            "selection_rejections": dict(reject_counts.most_common()),
        }

    final_rows.sort(key=lambda row: (row["source_key"], row["deterministic_key"], row["sample_id"]))
    return final_rows, selection_stats


def write_final_outputs(root: pathlib.Path, rows: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    output_dir = root / ("03_selected_smoke" if smoke else "03_selected")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    temp_dir = root / f".{output_dir.name}.tmp-{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    parquet_dir = temp_dir / "parquet"
    parquet_dir.mkdir()
    shard_size = 10_000
    parquet_paths: list[pathlib.Path] = []
    for shard_index, start in enumerate(range(0, len(rows), shard_size)):
        shard_rows = rows[start : start + shard_size]
        path = parquet_dir / f"part-{shard_index:05d}.parquet"
        write_rows_parquet(path, shard_rows)
        parquet_paths.append(path)

    jsonl_path = temp_dir / "general_sft_100k.openai.jsonl.zst"
    compressor = zstd.ZstdCompressor(level=10, threads=-1)
    with jsonl_path.open("wb") as raw:
        with compressor.stream_writer(raw) as writer:
            for row in rows:
                payload = {
                    "id": row["sample_id"],
                    "messages": row["messages"],
                    "metadata": {
                        "source_key": row["source_key"],
                        "source_dataset": row["source_dataset"],
                        "source_revision": row["source_revision"],
                        "source_id": row["source_id"],
                        "source_license": row["source_license"],
                        "language": row["language_final"],
                        "task_type": row["task_type"],
                        "verification_type": row["verification_type"],
                        "target_loss_only": True,
                    },
                }
                writer.write((json_dumps(payload) + "\n").encode("utf-8"))

    os.replace(temp_dir, output_dir)

    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(root))] = digest.hexdigest()
    return {"output_dir": str(output_dir), "sha256": hashes}


def build_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def count(field: str) -> dict[str, int]:
        return dict(collections.Counter(str(row[field]) for row in rows).most_common())

    target_lengths = sorted(row["target_chars"] for row in rows)
    total_lengths = sorted(row["total_chars"] for row in rows)

    def percentiles(values: list[int]) -> dict[str, int]:
        if not values:
            return {}
        result: dict[str, int] = {}
        for name, quantile in [("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]:
            index = min(len(values) - 1, math.floor((len(values) - 1) * quantile))
            result[name] = values[index]
        result["max"] = values[-1]
        return result

    return {
        "rows": len(rows),
        "source_distribution": count("source_key"),
        "language_distribution": count("language_final"),
        "task_distribution": count("task_type"),
        "license_distribution": count("source_license"),
        "provenance_distribution": count("provenance_type"),
        "verification_distribution": count("verification_type"),
        "cot_removed": sum(bool(row["cot_removed"]) for row in rows),
        "explicit_cot_present": sum(bool(row["explicit_cot_present"]) for row in rows),
        "ai_identity_break": sum(bool(row["ai_identity_break"]) for row in rows),
        "target_char_percentiles": percentiles(target_lengths),
        "total_char_percentiles": percentiles(total_lengths),
        "total_target_chars": sum(target_lengths),
        "total_chars": sum(total_lengths),
    }


def write_review_sample(root: pathlib.Path, rows: list[dict[str, Any]], smoke: bool) -> pathlib.Path:
    output = root / "05_review" / ("review_samples_smoke.jsonl" if smoke else "review_samples_500.jsonl")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row["source_key"], row["language_final"])].append(row)
    sample: list[dict[str, Any]] = []
    for key in sorted(grouped):
        candidates = sorted(grouped[key], key=lambda row: stable_uint64("review", row["sample_id"]))
        sample.extend(candidates[: min(10, len(candidates))])
    remaining = sorted(rows, key=lambda row: stable_uint64("review-fill", row["sample_id"]))
    seen = {row["sample_id"] for row in sample}
    target = min(100 if smoke else 500, len(rows))
    for row in remaining:
        if len(sample) >= target:
            break
        if row["sample_id"] not in seen:
            sample.append(row)
            seen.add(row["sample_id"])
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sample:
            handle.write(
                json_dumps(
                    {
                        "sample_id": row["sample_id"],
                        "source": row["source_key"],
                        "language": row["language_final"],
                        "task_type": row["task_type"],
                        "messages": row["messages"],
                        "review": {"decision": None, "notes": ""},
                    }
                )
                + "\n"
            )
    return output


def write_dataset_card(root: pathlib.Path, stats: dict[str, Any], revisions: dict[str, str]) -> None:
    lines = [
        "# LusyChat General SFT Anchor v1",
        "",
        "A deterministic 100,000-example general capability anchor pool for mixed SFT with LusyChat RP data.",
        "",
        "## Fixed mixture",
        "",
        "| Source | Rows | Purpose |",
        "|---|---:|---|",
        "| Aya human subset | 27,000 | multilingual instruction following |",
        "| OpenAssistant OASST2 | 23,000 | human-ranked multi-turn chat |",
        "| Tulu 3 instruction following | 15,000 | verifiable constraint following |",
        "| UltraChat 200k | 15,000 | broad English assistant behavior |",
        "| SmolTalk selected subsets | 10,000 | constraints, rewriting, summarization |",
        "| OpenMathInstruct-2 | 7,000 | math with expected-answer provenance |",
        "| BigCode SelfCodeAlign | 3,000 | execution-filtered code |",
        "",
        "All history messages have `weight=0`; only the final assistant has `weight=1`.",
        "No neutral system prompt is injected at this stage. Source system prompts are preserved.",
        "Provider reasoning fields are absent; complete leading `<think>...</think>` blocks are removed.",
        "",
        "## Revisions",
        "",
    ]
    for repo, revision in sorted(revisions.items()):
        lines.append(f"- `{repo}`: `{revision}`")
    lines.extend(
        [
            "",
            "## Final distribution",
            "",
            "```json",
            json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Important limitations",
            "",
            "- Public dataset cards and upstream output-model terms must remain attached to any commercial release review.",
            "- OpenMath expected answers and BigCode execution filtering are upstream claims; this build records but does not independently rerun every verifier.",
            "- The 100k pool is a candidate/training reservoir. Final epoch sampling should still be controlled by source and task weights.",
        ]
    )
    (root / "00_manifest" / "DATASET_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(rows: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    errors: list[str] = []
    expected = sum(min(100, value) for value in SOURCE_TARGETS.values()) if smoke else 100_000
    if len(rows) != expected:
        errors.append(f"row_count={len(rows)} expected={expected}")
    if not smoke:
        source_counts = collections.Counter(row["source_key"] for row in rows)
        if dict(source_counts) != dict(SOURCE_TARGETS):
            errors.append(f"source_distribution={dict(source_counts)}")
        language_counts = collections.Counter(row["language_final"] for row in rows)
        if dict(language_counts) != LANGUAGE_TARGETS:
            errors.append(f"language_distribution={dict(language_counts)}")
        smoltalk_counts = collections.Counter(
            row["source_config"] for row in rows if row["source_key"] == "smoltalk"
        )
        if dict(smoltalk_counts) != SMOLTALK_CONFIG_TARGETS:
            errors.append(f"smoltalk_distribution={dict(smoltalk_counts)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate_sample_id")
    prompt_fingerprints = [row["prompt_fingerprint"] for row in rows]
    if len(prompt_fingerprints) != len(set(prompt_fingerprints)):
        errors.append("duplicate_prompt_fingerprint")
    target_fingerprints = [row["target_fingerprint"] for row in rows]
    if len(target_fingerprints) != len(set(target_fingerprints)):
        errors.append("duplicate_target_fingerprint")
    for row in rows:
        weights = [message["weight"] for message in row["messages"]]
        if weights[-1] != 1 or any(weight != 0 for weight in weights[:-1]):
            errors.append(f"loss_mask:{row['sample_id']}")
            break
        if row["messages"][-1]["role"] != "assistant":
            errors.append(f"terminal_role:{row['sample_id']}")
            break
        if THINK_TAG_RE.search(row["target_text"]):
            errors.append(f"think_tag:{row['sample_id']}")
            break
        if has_ai_identity_break(row["target_text"]):
            errors.append(f"ai_identity:{row['sample_id']}")
            break
    return {"valid": not errors, "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: pathlib.Path = args.root
    for directory in ("00_manifest", "01_raw_cache", "02_candidates", "04_rejected", "05_review", "logs"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(root / "01_raw_cache" / "hf-home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "01_raw_cache" / "datasets"))

    api = HfApi(endpoint=args.hf_endpoint)
    revisions_path = root / "00_manifest" / "SOURCE_REVISIONS.json"
    if revisions_path.exists():
        revisions = json.loads(revisions_path.read_text(encoding="utf-8"))
    else:
        revisions = resolve_revisions(api)
        revisions_path.write_text(json.dumps(revisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    design = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "source_targets": SOURCE_TARGETS,
        "language_targets": LANGUAGE_TARGETS,
        "aya_language_targets": AYA_LANGUAGE_TARGETS,
        "oasst_language_targets": OASST_LANGUAGE_TARGETS,
        "smoltalk_config_targets": SMOLTALK_CONFIG_TARGETS,
        "source_specs": SOURCE_SPECS,
        "hf_endpoint": args.hf_endpoint,
        "smoke": args.smoke,
    }
    (root / "00_manifest" / "BUILD_DESIGN.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_rows, source_stats = build_or_load_sources(
        root,
        revisions,
        smoke=args.smoke,
        reuse=args.reuse_candidates,
    )
    rows, selection_stats = select_final(source_rows, smoke=args.smoke)
    validation = validate(rows, smoke=args.smoke)
    if not validation["valid"]:
        raise RuntimeError(f"validation failed: {validation}")

    stats = build_stats(rows)
    output_info = write_final_outputs(root, rows, smoke=args.smoke)
    review_path = write_review_sample(root, rows, smoke=args.smoke)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "source_revisions": revisions,
        "source_build_stats": source_stats,
        "selection_stats": selection_stats,
        "final_stats": stats,
        "validation": validation,
        "output": output_info,
        "review_sample": str(review_path),
        "database_accessed": False,
        "production_api_accessed": False,
        "repository_modified": False,
    }
    suffix = "_SMOKE" if args.smoke else ""
    (root / "00_manifest" / f"BUILD_REPORT{suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not args.smoke:
        write_dataset_card(root, stats, revisions)
    log(f"complete rows={len(rows)} output={output_info['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
