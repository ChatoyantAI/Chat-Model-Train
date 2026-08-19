"""Stable Arrow schemas and vectorized batch converters."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.compute as pc

MESSAGE_COLUMNS = [
    "increase_id", "id", "conversation_id", "user_id", "role", "content", "language",
    "create_time", "update_time", "status", "evaluation", "regenerated_times", "media_type",
    "media_id", "is_locked", "model", "model_id", "attached_image_url", "inner_thought",
    "inner_thought_content", "inner_thought_viewed", "inner_thought_placeholder",
]

MESSAGE_SCHEMA = pa.schema([
    pa.field("increase_id", pa.int64(), nullable=False),
    pa.field("id", pa.string()),
    pa.field("conversation_id", pa.string()),
    pa.field("user_id", pa.string()),
    pa.field("role", pa.int8()),
    pa.field("content", pa.large_string()),
    pa.field("language", pa.string()),
    pa.field("create_time", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("update_time", pa.timestamp("us", tz="UTC")),
    pa.field("status", pa.int8()),
    pa.field("evaluation", pa.int8()),
    pa.field("regenerated_times", pa.int32()),
    pa.field("media_type", pa.int16()),
    pa.field("media_id", pa.string()),
    pa.field("is_locked", pa.int8()),
    pa.field("model", pa.string()),
    pa.field("model_id", pa.int32()),
    pa.field("attached_image_url", pa.large_string()),
    pa.field("inner_thought", pa.int8()),
    pa.field("inner_thought_content", pa.large_string()),
    pa.field("inner_thought_viewed", pa.int8()),
    pa.field("inner_thought_placeholder", pa.large_string()),
], metadata={b"time_semantics": b"UTC", b"source": b"production-reader-stream"})


def _string(values: Sequence[str | None], large: bool = False) -> pa.Array:
    return pa.array(values, type=pa.large_string() if large else pa.string())


def _integer(values: Sequence[str | None], data_type: pa.DataType) -> pa.Array:
    parsed = [None if value in (None, "") else int(value) for value in values]
    return pa.array(parsed, type=data_type)


def _timestamp_utc(values: Sequence[str | None]) -> pa.Array:
    strings = pa.array(values, type=pa.string())
    naive = pc.strptime(strings, format="%Y-%m-%d %H:%M:%S", unit="us", error_is_null=False)
    return pc.assume_timezone(naive, "UTC")


def message_record_batch(rows: Sequence[Sequence[str | None]]) -> pa.RecordBatch:
    for index, row in enumerate(rows):
        if len(row) != len(MESSAGE_COLUMNS):
            raise ValueError(f"message row {index} has {len(row)} columns, expected {len(MESSAGE_COLUMNS)}")
    columns = list(zip(*rows)) if rows else [tuple() for _ in MESSAGE_COLUMNS]
    builders = {
        "increase_id": lambda v: _integer(v, pa.int64()),
        "id": _string,
        "conversation_id": _string,
        "user_id": _string,
        "role": lambda v: _integer(v, pa.int8()),
        "content": lambda v: _string(v, large=True),
        "language": _string,
        "create_time": _timestamp_utc,
        "update_time": _timestamp_utc,
        "status": lambda v: _integer(v, pa.int8()),
        "evaluation": lambda v: _integer(v, pa.int8()),
        "regenerated_times": lambda v: _integer(v, pa.int32()),
        "media_type": lambda v: _integer(v, pa.int16()),
        "media_id": _string,
        "is_locked": lambda v: _integer(v, pa.int8()),
        "model": _string,
        "model_id": lambda v: _integer(v, pa.int32()),
        "attached_image_url": lambda v: _string(v, large=True),
        "inner_thought": lambda v: _integer(v, pa.int8()),
        "inner_thought_content": lambda v: _string(v, large=True),
        "inner_thought_viewed": lambda v: _integer(v, pa.int8()),
        "inner_thought_placeholder": lambda v: _string(v, large=True),
    }
    arrays = [builders[name](columns[index]) for index, name in enumerate(MESSAGE_COLUMNS)]
    return pa.RecordBatch.from_arrays(arrays, schema=MESSAGE_SCHEMA)
