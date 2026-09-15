"""Source-faithful targeted annotation cleaning for EgoCom tables."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path
from conv_wm.data.cleaning import (
    CleanedAnnotationTable,
    remove_rows_missing_required_fields,
    require_source_columns,
)

RULE_VERSION = "egocom-annotation-cleaning-v1"

GROUND_TRUTH_COLUMNS = (
    "source_row",
    "conversation_id",
    "speaker_id",
    "startTime",
    "endTime",
    "word",
)
GROUND_TRUTH_REQUIRED = ("source_row", "conversation_id", "speaker_id")
GROUND_TRUTH_OPTIONAL = ("startTime", "endTime", "word")


def clean_ground_truth(
    source: pd.DataFrame,
    *,
    video_info: pd.DataFrame | None = None,
    source_path: Path = Path("ground_truth_transcriptions.csv"),
) -> CleanedAnnotationTable:
    """Preserve timed and untimed source tokens without grouping or text mutation."""
    source_columns = tuple(
        column for column in GROUND_TRUTH_COLUMNS if column != "source_row"
    )
    require_source_columns(
        source, source_columns, dataset="egocom", name="ground_truth"
    )
    ordered = cast(pd.DataFrame, source.loc[:, list(source_columns)]).copy()
    ordered.insert(0, "source_row", np.arange(len(ordered), dtype=np.int64))
    cleaned, removed = remove_rows_missing_required_fields(
        ordered, GROUND_TRUTH_REQUIRED
    )
    timed = cleaned["startTime"].notna() & cleaned["endTime"].notna()
    start_only = cleaned["startTime"].notna() & cleaned["endTime"].isna()
    untimed = cleaned["startTime"].isna() & cleaned["endTime"].isna()
    nonblank_word = cleaned["word"].fillna("").astype(str).str.strip().ne("")
    statistics = {
        "fully_timed_rows": int(timed.sum()),
        "start_only_rows": int(start_only.sum()),
        "fully_untimed_rows": int(untimed.sum()),
        "nonblank_fully_untimed_rows": int((untimed & nonblank_word).sum()),
        "nonblank_start_only_rows": int((start_only & nonblank_word).sum()),
    }
    if video_info is not None:
        require_source_columns(
            video_info,
            ("conversation_id", "video_speaker_id"),
            dataset="egocom",
            name="video_info",
        )
        available_povs = set(
            video_info.loc[:, ["conversation_id", "video_speaker_id"]].itertuples(
                index=False, name=None
            )
        )
        has_own_pov = [
            key in available_povs
            for key in cleaned.loc[:, ["conversation_id", "speaker_id"]].itertuples(
                index=False, name=None
            )
        ]
        n_with_pov = sum(has_own_pov)
        statistics |= {
            "participant_has_own_pov_true_rows": int(n_with_pov),
            "participant_has_own_pov_false_rows": int(len(has_own_pov) - n_with_pov),
        }
    return CleanedAnnotationTable(
        dataset="egocom",
        name="ground_truth",
        output_key="ground_truth_clean",
        source_rows=len(source),
        table=cleaned,
        decision="NO FILTER NEEDED",
        reason=(
            "Missing timestamps represent both formatting and valid untimed lexical or "
            "annotation content; source tokens are retained one-for-one."
        ),
        required_fields=GROUND_TRUTH_REQUIRED,
        optional_nullable_fields=GROUND_TRUTH_OPTIONAL,
        removed_by_reason=removed,
        cleaning_rule=RULE_VERSION,
        source_paths=(source_path,),
        transformations=(
            "add zero-based source_row to preserve source sequence",
            "preserve one output row per source row; do not aggregate equal intervals",
        ),
        statistics=statistics,
    )


def clean_video_info(
    source: pd.DataFrame, *, source_path: Path = Path("video_info.csv")
) -> CleanedAnnotationTable:
    """Derive the split label while retaining the source's POV identifier wording."""
    split_columns = ("train", "val", "test")
    require_source_columns(
        source,
        split_columns,
        dataset="egocom",
        name="video_info",
    )
    selected = cast(
        pd.DataFrame,
        source.loc[
            :, [column for column in source.columns if column not in split_columns]
        ],
    ).copy()
    split_flags = cast(pd.DataFrame, source.loc[:, list(split_columns)])
    split_array = split_flags.to_numpy(dtype=bool)
    if not bool(np.equal(split_array.sum(axis=1), 1).all()):
        raise ValueError(
            "egocom/video_info: exactly one split flag must be true per row"
        )
    labels = np.asarray(split_columns)[split_array.argmax(axis=1)]
    selected["split"] = pd.Series(labels, index=selected.index, dtype="string")
    required = (
        "video_id",
        "conversation_id",
        "video_speaker_id",
        "num_speakers",
        "duration_seconds",
        "video_name",
        "split",
    )
    require_source_columns(selected, required, dataset="egocom", name="video_info")
    return CleanedAnnotationTable(
        dataset="egocom",
        name="video_info",
        output_key="video_info_clean",
        source_rows=len(source),
        table=selected,
        decision="CHANGE FILTER",
        reason=(
            "No rows require filtering; retain video_speaker_id because it identifies "
            "the camera wearer/available POV rather than a complete participant registry."
        ),
        required_fields=required,
        optional_nullable_fields=tuple(
            column for column in selected.columns if column not in required
        ),
        removed_by_reason={},
        cleaning_rule=RULE_VERSION,
        source_paths=(source_path,),
        transformations=(
            "replace mutually exclusive train/val/test flags with split",
            "retain source-native video_speaker_id name",
        ),
    )


def build_annotation_cleaning(cfg: DictConfig) -> list[CleanedAnnotationTable]:
    """Load EgoCom CSV sources and derive the two maintained interim tables."""
    video_path = get_path(cfg, "egocom", "raw", "video_info")
    ground_truth_path = get_path(cfg, "egocom", "raw", "ground_truth")
    video_info = pd.read_csv(video_path)
    return [
        clean_video_info(video_info, source_path=video_path),
        clean_ground_truth(
            pd.read_csv(ground_truth_path),
            video_info=video_info,
            source_path=ground_truth_path,
        ),
    ]
