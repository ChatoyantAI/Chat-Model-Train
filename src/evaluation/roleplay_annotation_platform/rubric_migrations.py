from __future__ import annotations

from typing import Any


LEGACY_NARRATIVE_RUBRIC_VERSIONS = frozenset(
    {
        "roleplay-rubric-v0.3",
        "roleplay-rubric-v0.4",
        "roleplay-rubric-v0.5",
        "roleplay-rubric-v0.6",
    }
)

# These versions already use the current item IDs. Their answers can be carried
# forward unchanged when wording or score anchors are clarified.
PASSTHROUGH_RUBRIC_VERSIONS = frozenset({"roleplay-rubric-v0.7"})


def _record_value(record: Any) -> Any:
    return record.get("value") if isinstance(record, dict) else None


def _merged_value(old_n2: Any, old_n3: Any) -> int | str:
    n2_value = _record_value(old_n2)
    n3_value = _record_value(old_n3)
    if n2_value is None or n3_value is None:
        return n2_value if n2_value is not None else "N/A"
    if n2_value == "N/A" or n3_value == "N/A":
        return "N/A"
    numeric_n2 = int(n2_value)
    numeric_n3 = int(n3_value)
    if numeric_n2 == 0:
        result = 0
    elif numeric_n2 == 1 or numeric_n3 == 0:
        result = 1
    elif numeric_n3 == 3:
        result = 3
    else:
        result = 2
    return str(result) if isinstance(n2_value, str) else result


def _merged_field(old_n2: Any, old_n3: Any, field: str) -> Any:
    values = [
        record.get(field)
        for record in (old_n2, old_n3)
        if isinstance(record, dict) and record.get(field) not in (None, "", [])
    ]
    if not values:
        return (
            []
            if field == "evidence"
            and any(
                isinstance(record, dict) and isinstance(record.get(field), list)
                for record in (old_n2, old_n3)
            )
            else ""
        )
    if all(isinstance(value, list) for value in values):
        merged: list[Any] = []
        for value in values:
            for entry in value:
                if entry not in merged:
                    merged.append(entry)
        return merged
    strings: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in strings:
            strings.append(text)
    return "\n".join(strings)[:4000]


def merge_legacy_narrative_scores(scores: dict[str, Any]) -> dict[str, Any]:
    """Merge old N2/N3 and shift old N4/N5 into the v0.7 narrative IDs."""
    migrated = {
        item_id: value
        for item_id, value in scores.items()
        if item_id not in {"N2", "N3", "N4", "N5"}
    }
    old_n2 = scores.get("N2")
    old_n3 = scores.get("N3")
    if old_n2 is not None or old_n3 is not None:
        merged_record = dict(old_n2) if isinstance(old_n2, dict) else {}
        merged_record["value"] = _merged_value(old_n2, old_n3)
        merged_record["evidence"] = _merged_field(old_n2, old_n3, "evidence")
        if any(
            isinstance(record, dict) and "reason" in record
            for record in (old_n2, old_n3)
        ):
            merged_record["reason"] = _merged_field(old_n2, old_n3, "reason")
        migrated["N2"] = merged_record
    if "N4" in scores:
        migrated["N3"] = scores["N4"]
    if "N5" in scores:
        migrated["N4"] = scores["N5"]
    return migrated


def scores_for_current_rubric(
    scores: dict[str, Any] | None,
    source_rubric_version: str,
) -> dict[str, Any] | None:
    if scores is None:
        return None
    if source_rubric_version in LEGACY_NARRATIVE_RUBRIC_VERSIONS:
        return merge_legacy_narrative_scores(scores)
    return scores
