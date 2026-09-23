"""The control focal voice-state build: transform, provenance and immutability."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
from state_layers import TIMELINE, build_control_state, config_for, write_native_state

from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.pipeline.control_state import BRIDGED_GAPS_TABLE, TIMELINE_TABLE
from conv_wm.data.vocal.action_grid import DECISION_STEP_S
from conv_wm.data.vocal.control_state import (
    CONTROL_STATE_SCHEMA_VERSION,
    TransformKind,
)
from conv_wm.data.vocal.native_state import NATIVE_STATE_SCHEMA_VERSION

STEP = DECISION_STEP_S


def _states(timeline: pd.DataFrame):
    return [
        (row.canonical_start_s, row.canonical_end_s, row.voice_state)
        for row in timeline.itertuples()
    ]


def test_sub_step_silences_are_bridged_and_everything_else_is_kept(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "ego4d")

    [output] = build_control_state(cfg, "ego4d")

    assert _states(output.timeline) == [
        (0.0, 0.34, "SILENT"),
        (0.34, 0.62, "SPEAKING"),
        (0.62, 1.0, "SILENT"),  # 380 ms: representable, kept
        (1.0, 2.4, "SPEAKING"),  # the 10 ms and 40 ms gaps are bridged
        (2.4, 2.9, "UNKNOWN"),
        (2.9, 3.5, "SILENT"),
        (3.5, 3.52, "SPEAKING"),  # the short burst survives
        (3.52, 3.97, "SILENT"),
    ]
    bridged = output.timeline.loc[output.timeline["transform_kind"].notna()]
    assert len(bridged) == 1
    assert bridged["transform_kind"].iloc[0] == str(
        TransformKind.SUB_STEP_SILENCE_BRIDGE
    )
    assert bridged["bridged_gap_count"].iloc[0] == 2
    assert bridged["bridged_gap_total_duration_s"].iloc[0] == pytest.approx(0.05)
    assert bridged["source_native_index_first"].iloc[0] == 3
    assert bridged["source_native_index_last"].iloc[0] == 7


def test_bridged_gaps_are_recorded_one_row_each(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "egocom")

    [output] = build_control_state(cfg, "egocom")

    assert output.bridged_gaps["gap_duration_s"].tolist() == [
        pytest.approx(0.01),
        pytest.approx(0.04),
    ]
    assert output.bridged_gaps["native_index"].tolist() == [4, 6]
    assert output.bridged_gaps["recording_id"].eq("rec-1").all()
    assert output.bridged_gaps_path.exists()
    assert output.bridged_gaps_path.name == BRIDGED_GAPS_TABLE


def test_the_native_artifact_is_never_modified(tmp_path):
    cfg = config_for(tmp_path)
    native_path = write_native_state(cfg, "ego4d")
    before = hashlib.sha256(native_path.read_bytes()).hexdigest()
    native_before = pd.read_parquet(native_path)

    [output] = build_control_state(cfg, "ego4d")

    assert hashlib.sha256(native_path.read_bytes()).hexdigest() == before
    pd.testing.assert_frame_equal(pd.read_parquet(native_path), native_before)
    assert output.timeline_path != native_path
    assert output.timeline_path.name == TIMELINE_TABLE
    assert output.report["transform"]["native_artifact_modified"] is False


def test_timeline_duration_and_window_are_preserved(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "ego4d")

    [output] = build_control_state(cfg, "ego4d")
    summary = output.summary.iloc[0]
    statistics = output.report["statistics"]

    assert summary["canonical_start_s"] == 0.0
    assert summary["canonical_end_s"] == pytest.approx(3.97)
    durations = (
        output.timeline["canonical_end_s"] - output.timeline["canonical_start_s"]
    )
    assert durations.sum() == pytest.approx(3.97)
    # bridging moves exactly the bridged duration from SILENT to SPEAKING
    assert statistics["control_speaking_duration_s"] == pytest.approx(
        statistics["native_speaking_duration_s"]
        + statistics["bridged_gap_total_duration_s"]
    )
    assert statistics["control_silent_duration_s"] == pytest.approx(
        statistics["native_silent_duration_s"]
        - statistics["bridged_gap_total_duration_s"]
    )


def test_statistics_report_the_before_and_after_of_the_transform(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "egocom")

    [output] = build_control_state(cfg, "egocom")
    statistics = output.report["statistics"]

    assert statistics["native_interval_count"] == len(TIMELINE)
    assert statistics["control_interval_count"] == 8
    assert statistics["native_transition_count"] == 9
    assert statistics["control_transition_count"] == 5
    assert statistics["removed_transition_count"] == 4
    assert statistics["bridged_gap_count"] == 2
    assert statistics["bridged_gap_total_duration_s"] == pytest.approx(0.05)
    assert statistics["bridged_gap_duration_s"]["max"] == pytest.approx(0.04)
    assert statistics["bridged_gap_duration_histogram_s"] == {
        "[0.010,0.020)": 1,
        "[0.040,0.050)": 1,
    }
    assert statistics["recordings_with_bridged_gap"] == 1
    assert statistics["max_bridged_gaps_in_one_interval"] == 2


def test_build_is_deterministic_and_records_its_lineage(tmp_path):
    cfg = config_for(tmp_path)
    native_path = write_native_state(cfg, "ego4d")

    first = build_control_state(cfg, "ego4d")[0]
    second = build_control_state(cfg, "ego4d")[0]

    pd.testing.assert_frame_equal(first.timeline, second.timeline)
    report = json.loads(first.report_path.read_text())
    assert report["control_state_schema_version"] == CONTROL_STATE_SCHEMA_VERSION
    assert report["native_state_schema_version"] == NATIVE_STATE_SCHEMA_VERSION
    assert report["decision_step_s"] == STEP
    assert report["lineage_chain"] == [
        "native annotation",
        "native focal voice state",
        "control focal voice state",
        "vocal action grid",
    ]
    transform = report["transform"]
    assert transform["transform_kind"] == str(TransformKind.SUB_STEP_SILENCE_BRIDGE)
    assert transform["condition"] == "gap_duration_s < decision_step_s"
    assert transform["condition_is_strict"] is True
    assert transform["speech_bursts_filtered"] is False
    assert transform["native_timestamps_modified"] is False
    inputs = report["input_artifacts"]
    assert inputs["native_focal_voice_state_timeline"]["path"] == str(native_path)
    assert (
        inputs["native_focal_voice_state_timeline"]["sha256"]
        == hashlib.sha256(native_path.read_bytes()).hexdigest()
    )
    assert inputs["native_focal_voice_state_report"]["sha256"]
    assert (
        report["output_artifacts"]["timeline"]["sha256"]
        == hashlib.sha256(first.timeline_path.read_bytes()).hexdigest()
    )
    assert report["config_checksum"]
    assert {"git_commit", "git_dirty", "command", "created_at"} <= set(report)
    assert "ego4d upstream limitation" in report["limitations"]
    assert (
        report["source_annotation_versions"]["annotation_schema_version"]
        == "ego4d-test-v1"
    )
    assert (
        report["output_artifacts"]["timeline"]["sha256"]
        == second.report["output_artifacts"]["timeline"]["sha256"]
    )
    assert report["config_checksum"] == second.report["config_checksum"]


def test_stale_or_missing_native_state_is_refused(tmp_path):
    cfg = config_for(tmp_path)

    with pytest.raises(MissingPrerequisiteError):
        build_control_state(cfg, "ego4d")

    path = write_native_state(cfg, "ego4d")
    frame = pd.read_parquet(path)
    frame.loc[0, "canonical_end_s"] = 0.5  # rewrite the timeline behind its report
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="does not match"):
        build_control_state(cfg, "ego4d")


def test_every_dataset_is_transformed_by_the_same_rule(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "ego4d")
    write_native_state(cfg, "egocom")

    ego4d, egocom = build_control_state(cfg, "all")

    shared = ["canonical_start_s", "canonical_end_s", "voice_state", "transform_kind"]
    pd.testing.assert_frame_equal(ego4d.timeline[shared], egocom.timeline[shared])
    assert ego4d.timeline["dataset"].eq("ego4d").all()
    assert egocom.timeline["sync_group_id"].eq("group-1").all()
    assert ego4d.timeline["sync_group_id"].isna().all()
