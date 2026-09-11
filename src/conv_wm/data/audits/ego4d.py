"""Structural audit orchestration for the Ego4D interim tables."""

import pandas as pd

from conv_wm.data.audits.pandera import audit_dataframe
from conv_wm.data.schemas import EGO4D_SCHEMAS
from conv_wm.data.validation import check_foreign_key


def _relation_result(orphan_rows: int) -> dict[str, object]:
    return {
        "valid": orphan_rows == 0,
        "n_failures": orphan_rows,
        "orphan_rows": orphan_rows,
    }


def audit_ego4d(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Validate Ego4D tables and defensible parent-child relationships."""
    table_results = {
        name: audit_dataframe(tables[name], schema)
        for name, schema in EGO4D_SCHEMAS.items()
    }

    clips = tables["clips_clean"]
    persons = tables["persons_clean"]
    tracking_paths = tables["tracking_paths_clean"]

    relation_specs = {
        "persons_clip_uid_in_clips": (
            tables["persons_clean"],
            ["clip_uid"],
            clips,
            ["clip_uid"],
        ),
        "tracking_paths_clip_uid_in_clips": (
            tracking_paths,
            ["clip_uid"],
            clips,
            ["clip_uid"],
        ),
        "tracks_clip_uid_in_clips": (
            tables["tracks_clean"],
            ["clip_uid"],
            clips,
            ["clip_uid"],
        ),
        "voice_segments_clip_uid_in_clips": (
            tables["voice_segments_clean"],
            ["clip_uid"],
            clips,
            ["clip_uid"],
        ),
        "transcriptions_clip_uid_in_clips": (
            tables["transcriptions_clean"],
            ["clip_uid"],
            clips,
            ["clip_uid"],
        ),
        "social_talking_clip_uid_in_clips": (
            tables["social_segments_talking_clean"],
            ["clip_uid"],
            clips,
            ["clip_uid"],
        ),
        "social_looking_clip_uid_in_clips": (
            tables["social_segments_looking_clean"],
            ["clip_uid"],
            clips,
            ["clip_uid"],
        ),
        "tracking_paths_person_in_persons": (
            tracking_paths,
            ["clip_uid", "person_id"],
            persons,
            ["clip_uid", "person_id"],
        ),
        "tracks_path_in_tracking_paths": (
            tables["tracks_clean"],
            ["clip_uid", "person_id", "track_id"],
            tracking_paths,
            ["clip_uid", "person_id", "track_id"],
        ),
        "voice_segments_person_in_persons": (
            tables["voice_segments_clean"],
            ["clip_uid", "person_id"],
            persons,
            ["clip_uid", "person_id"],
        ),
    }
    relations = {
        name: _relation_result(check_foreign_key(*spec))
        for name, spec in relation_specs.items()
    }

    return {
        "valid": all(result["valid"] for result in table_results.values())
        and all(result["valid"] for result in relations.values()),
        "tables": table_results,
        "relations": relations,
    }
