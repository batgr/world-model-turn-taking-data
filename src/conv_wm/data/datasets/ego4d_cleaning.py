"""Source-faithful targeted annotation cleaning for the Ego4D AV release."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path
from conv_wm.data.cleaning import (
    CleanedAnnotationTable,
    CleaningDecision,
    remove_rows_missing_required_fields,
    require_source_columns,
)

RULE_VERSION = "ego4d-annotation-cleaning-v1"

VOICE_COLUMNS = (
    "clip_uid",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "video_start_time",
    "video_end_time",
    "video_start_frame",
    "video_end_frame",
    "person_id",
)
TRANSCRIPTION_COLUMNS = (
    "clip_uid",
    "transcription",
    "start_time_sec",
    "end_time_sec",
    "person_id",
    "video_start_time",
    "video_start_frame",
    "video_end_time",
    "video_end_frame",
)
SOCIAL_COLUMNS = (
    "clip_uid",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "video_start_time",
    "video_end_time",
    "video_start_frame",
    "video_end_frame",
    "person",
    "target",
    "is_at_me",
)

VOICE_REQUIRED = VOICE_COLUMNS
TRANSCRIPTION_REQUIRED = TRANSCRIPTION_COLUMNS
SOCIAL_REQUIRED = tuple(column for column in SOCIAL_COLUMNS if column != "target")


def _collection(owner: Mapping[str, Any], key: str) -> list[Any]:
    value = owner.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Ego4D {key} must be a JSON array or null")
    return value


def iter_json_array(path: Path, key: str) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading the release at once."""
    decoder = json.JSONDecoder()
    buffer = ""
    found = False
    with path.open(encoding="utf-8") as source:
        while not found:
            chunk = source.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"{path}: top-level array {key!r} not found")
            buffer += chunk
            marker = f'"{key}"'
            marker_index = buffer.find(marker)
            if marker_index < 0:
                buffer = buffer[-len(marker) :]
                continue
            array_index = buffer.find("[", marker_index + len(marker))
            while array_index < 0:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    raise ValueError(f"{path}: array start for {key!r} not found")
                buffer += chunk
                array_index = buffer.find("[", marker_index + len(marker))
            buffer = buffer[array_index + 1 :]
            found = True

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    raise ValueError(f"{path}: incomplete JSON object in {key!r}")
                buffer += chunk
                continue
            if not isinstance(value, dict):
                raise TypeError(f"{path}: {key!r} contains a non-object value")
            yield value
            buffer = buffer[end:]


def extract_annotation_tables(
    videos: Iterable[Mapping[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Flatten only annotation collections affected by historical null filtering."""
    records: dict[str, list[dict[str, Any]]] = {
        "voice_segments": [],
        "transcriptions": [],
        "social_segments_talking": [],
        "social_segments_looking": [],
    }
    for video in videos:
        for clip_value in _collection(video, "clips"):
            if not isinstance(clip_value, Mapping):
                raise TypeError("Ego4D clips must be JSON objects")
            clip = clip_value
            clip_uid = clip.get("clip_uid")
            for person_value in _collection(clip, "persons"):
                if not isinstance(person_value, Mapping):
                    raise TypeError("Ego4D persons must be JSON objects")
                for segment_value in _collection(person_value, "voice_segments"):
                    if not isinstance(segment_value, Mapping):
                        raise TypeError("Ego4D voice segments must be JSON objects")
                    segment = dict(segment_value)
                    segment["clip_uid"] = clip_uid
                    segment["person_id"] = segment.pop("person", None)
                    records["voice_segments"].append(segment)
            for source_name in (
                "transcriptions",
                "social_segments_talking",
                "social_segments_looking",
            ):
                for annotation_value in _collection(clip, source_name):
                    if not isinstance(annotation_value, Mapping):
                        raise TypeError(
                            f"Ego4D {source_name} entries must be JSON objects"
                        )
                    annotation = dict(annotation_value)
                    annotation["clip_uid"] = clip_uid
                    records[source_name].append(annotation)
    return {
        "voice_segments": pd.DataFrame.from_records(
            records["voice_segments"], columns=VOICE_COLUMNS
        ),
        "transcriptions": pd.DataFrame.from_records(
            records["transcriptions"], columns=TRANSCRIPTION_COLUMNS
        ),
        "social_segments_talking": pd.DataFrame.from_records(
            records["social_segments_talking"], columns=SOCIAL_COLUMNS
        ),
        "social_segments_looking": pd.DataFrame.from_records(
            records["social_segments_looking"], columns=SOCIAL_COLUMNS
        ),
    }


def _clean_table(
    source: pd.DataFrame,
    *,
    name: str,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    decision: CleaningDecision,
    reason: str,
) -> CleanedAnnotationTable:
    require_source_columns(source, (*required, *optional), dataset="ego4d", name=name)
    cleaned, removed = remove_rows_missing_required_fields(source, required)
    return CleanedAnnotationTable(
        dataset="ego4d",
        name=name,
        output_key=f"{name}_clean",
        source_rows=len(source),
        table=cleaned,
        decision=decision,
        reason=reason,
        required_fields=required,
        optional_nullable_fields=optional,
        removed_by_reason=removed,
        cleaning_rule=RULE_VERSION,
        source_paths=(),
        transformations=(
            "flatten nested release records and attach parent clip_uid",
            "retain source timestamps without clipping or correction",
        ),
    )


def clean_annotation_tables(
    sources: Mapping[str, pd.DataFrame],
) -> list[CleanedAnnotationTable]:
    """Apply the reviewed required-field policy to flat Ego4D source tables."""
    voice = _clean_table(
        sources["voice_segments"],
        name="voice_segments",
        required=VOICE_REQUIRED,
        optional=(),
        decision="KEEP CURRENT FILTER",
        reason=(
            "Only structurally empty exploded-list placeholders were historically "
            "removed; the maintained rule names every required field explicitly."
        ),
    )
    transcriptions = _clean_table(
        sources["transcriptions"],
        name="transcriptions",
        required=TRANSCRIPTION_REQUIRED,
        optional=(),
        decision="KEEP CURRENT FILTER",
        reason=(
            "Only structurally empty exploded-list placeholders were historically "
            "removed; unknown person -1 is a valid retained identity sentinel."
        ),
    )
    talking = _clean_table(
        sources["social_segments_talking"],
        name="social_segments_talking",
        required=SOCIAL_REQUIRED,
        optional=("target",),
        decision="CHANGE FILTER",
        reason=(
            "target is nullable in the release schema; global null filtering discarded "
            "valid not-at-me intervals."
        ),
    )
    looking = _clean_table(
        sources["social_segments_looking"],
        name="social_segments_looking",
        required=SOCIAL_REQUIRED,
        optional=("target",),
        decision="CHANGE FILTER",
        reason=(
            "target is nullable and is_at_me is meaningful source payload; only rows "
            "without the required person identity or temporal coordinates are unusable."
        ),
    )
    talking_table = talking.table
    looking_table = looking.table
    return [
        voice,
        transcriptions,
        replace(
            talking,
            statistics={
                "unknown_person_rows": int(talking_table["person"].eq("-1").sum()),
                "target_available_rows": int(talking_table["target"].notna().sum()),
                "target_null_rows": int(talking_table["target"].isna().sum()),
                "is_at_me_true_rows": int(talking_table["is_at_me"].eq(True).sum()),
                "is_at_me_false_rows": int(talking_table["is_at_me"].eq(False).sum()),
            },
        ),
        replace(
            looking,
            statistics={
                "unknown_person_rows": int(looking_table["person"].eq("-1").sum()),
                "target_available_rows": int(looking_table["target"].notna().sum()),
                "target_null_rows": int(looking_table["target"].isna().sum()),
                "is_at_me_true_rows": int(looking_table["is_at_me"].eq(True).sum()),
                "is_at_me_false_rows": int(looking_table["is_at_me"].eq(False).sum()),
            },
        ),
    ]


def build_annotation_cleaning(cfg: DictConfig) -> list[CleanedAnnotationTable]:
    """Load the official Ego4D JSON release and derive reviewed interim tables."""
    paths = (
        get_path(cfg, "ego4d", "raw", "av_train"),
        get_path(cfg, "ego4d", "raw", "av_val"),
    )
    videos = (video for path in paths for video in iter_json_array(path, "videos"))
    results = clean_annotation_tables(extract_annotation_tables(videos))
    return [replace(result, source_paths=paths) for result in results]
