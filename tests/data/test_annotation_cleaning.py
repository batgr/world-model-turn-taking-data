from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from conv_wm.data import datasets
from conv_wm.data.cleaning import (
    CleanedAnnotationTable,
    remove_rows_missing_required_fields,
    run_annotation_cleaning,
)
from conv_wm.data.datasets.ego4d_cleaning import (
    CLIP_COLUMNS,
    MISSING_VOICE_COLUMNS,
    PERSON_COLUMNS,
    SOCIAL_COLUMNS,
    TRANSCRIPTION_COLUMNS,
    VOICE_COLUMNS,
    clean_annotation_tables,
    extract_annotation_tables,
)
from conv_wm.data.datasets.egocom_cleaning import clean_ground_truth
from conv_wm.data.datasets.spec import DatasetSpec


def _voice_row(**overrides):
    row = {
        "clip_uid": "clip-1",
        "start_time": 0.0,
        "end_time": 1.0,
        "start_frame": 0,
        "end_frame": 30,
        "video_start_time": 10.0,
        "video_end_time": 11.0,
        "video_start_frame": 300,
        "video_end_frame": 330,
        "person_id": "1",
    }
    row.update(overrides)
    return row


def _transcription_row(**overrides):
    row = {
        "clip_uid": "clip-1",
        "transcription": "hello",
        "start_time_sec": 0.0,
        "end_time_sec": 1.0,
        "person_id": "1",
        "video_start_time": 10.0,
        "video_start_frame": 300,
        "video_end_time": 11.0,
        "video_end_frame": 330,
    }
    row.update(overrides)
    return row


def _social_row(**overrides):
    row = {
        "clip_uid": "clip-1",
        "start_time": 0.0,
        "end_time": 1.0,
        "start_frame": 0,
        "end_frame": 30,
        "video_start_time": 10.0,
        "video_end_time": 11.0,
        "video_start_frame": 300,
        "video_end_frame": 330,
        "person": "1",
        "target": "0",
        "is_at_me": True,
    }
    row.update(overrides)
    return row


def _clip_row(**overrides):
    row = {
        "split": "train",
        "clip_uid": "clip-1",
        "source_clip_uid": "source-1",
        "video_uid": "video-1",
        "video_start_sec": 10.0,
        "video_end_sec": 20.0,
        "video_start_frame": 300,
        "video_end_frame": 600,
        "clip_start_sec": 0,
        "clip_end_sec": 10.0,
        "clip_start_frame": 0,
        "clip_end_frame": 300,
        "valid": True,
    }
    row.update(overrides)
    return row


def _clean_ego4d(
    *, talking, looking=None, voice=None, transcriptions=None, clips=None, missing=None
):
    sources = {
        "clips": pd.DataFrame.from_records(
            clips or [_clip_row()], columns=CLIP_COLUMNS
        ),
        "persons": pd.DataFrame.from_records(
            [{"clip_uid": "clip-1", "person_id": "0", "is_camera_wearer": True}],
            columns=PERSON_COLUMNS,
        ),
        "missing_voice_segments": pd.DataFrame.from_records(
            missing or [], columns=MISSING_VOICE_COLUMNS
        ).astype({"start_time": float, "end_time": float}),
        "voice_segments": pd.DataFrame.from_records(
            voice or [_voice_row()], columns=VOICE_COLUMNS
        ),
        "transcriptions": pd.DataFrame.from_records(
            transcriptions or [_transcription_row()], columns=TRANSCRIPTION_COLUMNS
        ),
        "social_segments_talking": pd.DataFrame.from_records(
            talking, columns=SOCIAL_COLUMNS
        ),
        "social_segments_looking": pd.DataFrame.from_records(
            looking or [_social_row()], columns=SOCIAL_COLUMNS
        ),
    }
    return {result.name: result for result in clean_annotation_tables(sources)}


def test_talking_nullable_target_false_label_and_unknown_person_survive():
    results = _clean_ego4d(
        talking=[_social_row(person="-1", target=None, is_at_me=False)]
    )

    talking = results["social_segments_talking"]

    assert len(talking.table) == 1
    assert talking.table.loc[0, "person"] == "-1"
    assert pd.isna(talking.table.loc[0, "target"])
    assert not bool(talking.table.loc[0, "is_at_me"])
    assert talking.rows_removed == 0
    assert (
        talking.to_summary(Path("out.parquet"))[
            "rows_retained_with_nullable_optional_fields"
        ]
        == 1
    )


def test_talking_missing_required_time_is_removed_and_accounted():
    results = _clean_ego4d(
        talking=[_social_row(start_time=None), _social_row(target=None)]
    )

    talking = results["social_segments_talking"]

    assert len(talking.table) == 1
    assert talking.removed_by_reason == {"missing_required_field:start_time": 1}
    talking.validate_accounting()


def test_looking_preserves_nullable_payload_and_removes_missing_identity():
    results = _clean_ego4d(
        talking=[_social_row()],
        looking=[_social_row(target=None, is_at_me=False), _social_row(person=None)],
    )

    looking = results["social_segments_looking"]

    assert len(looking.table) == 1
    assert "target" in looking.table
    assert "is_at_me" in looking.table
    assert looking.removed_by_reason == {"missing_required_field:person": 1}


def test_voice_and_transcription_keep_valid_unknown_identity():
    results = _clean_ego4d(
        talking=[_social_row()],
        voice=[_voice_row(person_id="-1")],
        transcriptions=[_transcription_row(person_id="-1")],
    )

    assert results["voice_segments"].table.loc[0, "person_id"] == "-1"
    assert results["transcriptions"].table.loc[0, "person_id"] == "-1"


def test_egocom_ground_truth_preserves_untimed_and_optional_null_rows():
    source = pd.DataFrame(
        {
            "conversation_id": ["c1", "c1", "c1"],
            "speaker_id": [1, 3, 3],
            "startTime": [0.0, None, 1.0],
            "endTime": [0.5, None, None],
            "word": ["hello", "untimed", None],
        }
    )

    result = clean_ground_truth(source)

    assert len(result.table) == len(source)
    assert result.table["source_row"].tolist() == [0, 1, 2]
    assert result.table.loc[1, "speaker_id"] == 3
    assert result.rows_removed == 0
    result.validate_accounting()


def test_required_field_accounting_assigns_each_row_once():
    table = pd.DataFrame({"identity": [None, None, "ok"], "start": [None, 0.0, None]})

    cleaned, reasons = remove_rows_missing_required_fields(table, ("identity", "start"))

    assert cleaned.empty
    assert reasons == {
        "missing_required_field:identity": 2,
        "missing_required_field:start": 1,
    }


def test_null_nested_annotation_collection_is_structurally_empty():
    tables = extract_annotation_tables(
        [
            {
                "clips": [
                    {
                        "clip_uid": "clip-1",
                        "persons": [],
                        "transcriptions": None,
                        "social_segments_talking": None,
                        "social_segments_looking": None,
                    }
                ]
            }
        ]
    )

    assert len(tables["clips"]) == 1
    assert tables["clips"].loc[0, "clip_uid"] == "clip-1"
    assert all(table.empty for name, table in tables.items() if name != "clips")


def test_clip_validity_wearer_and_missing_voice_regions_are_extracted():
    tables = extract_annotation_tables(
        [
            {
                "split": "val",
                "clips": [
                    {
                        **{k: v for k, v in _clip_row().items() if k != "split"},
                        "valid": False,
                        "missing_voice_segments": [
                            {"person": "0", "start_time": 1.0, "end_time": 2.0}
                        ],
                        "persons": [
                            {
                                "person_id": "0",
                                "camera_wearer": True,
                                "voice_segments": [],
                            },
                            {"person_id": "1", "camera_wearer": False},
                        ],
                    }
                ],
            }
        ]
    )

    clip = tables["clips"].loc[0]
    assert (clip["split"], clip["valid"], clip["video_start_sec"]) == (
        "val",
        False,
        10.0,
    )
    assert tables["persons"]["is_camera_wearer"].tolist() == [True, False]
    missing = tables["missing_voice_segments"].loc[0]
    assert (missing["person_id"], missing["start_time"], missing["end_time"]) == (
        "0",
        1.0,
        2.0,
    )
    results = {r.name: r for r in clean_annotation_tables(tables)}
    assert results["clips"].statistics == {"invalid_clip_rows": 1}
    assert results["missing_voice_segments"].output_rows == 1


def test_cleaning_orchestration_writes_accounted_table_and_report(tmp_path):
    def cleaner(_):
        return [
            CleanedAnnotationTable(
                dataset="synthetic",
                name="events",
                output_key="events_clean",
                source_rows=2,
                table=pd.DataFrame({"id": [1]}),
                decision="CHANGE FILTER",
                reason="one missing required identity",
                required_fields=("id",),
                optional_nullable_fields=(),
                removed_by_reason={"missing_required_field:id": 1},
                cleaning_rule="synthetic-v1",
                source_paths=(tmp_path / "raw.csv",),
            )
        ]

    reports = tmp_path / "reports"
    cfg = OmegaConf.create(
        {
            "paths": {
                stage: str(tmp_path / stage)
                for stage in ("raw", "interim", "validated", "processed", "model_ready")
            }
            | {"reports": str(reports)},
            "datasets": {
                "synthetic": {
                    "interim": str(tmp_path / "interim"),
                    "files": {"events_clean": "events_clean.parquet"},
                }
            },
        }
    )
    datasets.register(DatasetSpec(name="synthetic", annotation_cleaner=cleaner))
    try:
        outputs = run_annotation_cleaning(cfg, dataset_names=["synthetic"])
    finally:
        datasets.unregister("synthetic")

    account = outputs.summary["datasets"]["synthetic"]["tables"]["events"]
    assert account["source_rows"] == 2
    assert account["output_rows"] == 1
    assert account["rows_removed"] == 1
    assert (tmp_path / "interim" / "events_clean.parquet").exists()
    assert outputs.summary_path == reports / "cleaning" / "annotations" / "summary.json"
