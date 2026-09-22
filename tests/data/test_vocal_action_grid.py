"""The vocal action grid build: invariants, lineage and dataset independence."""

from __future__ import annotations

import ast
import importlib.util
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from state_layers import build_control_state, config_for, write_native_state

from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.vocal.action_grid import (
    ACTION_SCHEMA_VERSION,
    ACTIONS,
    DECISION_STEP_S,
    Action,
    MaskReason,
)
from conv_wm.data.vocal.control_state import CONTROL_STATE_SCHEMA_VERSION
from conv_wm.data.vocal.native_state import NATIVE_STATE_SCHEMA_VERSION
from conv_wm.data.vocal_action_grid import (
    GRID_TABLE,
    check_invariants,
    run_vocal_action_grid_build,
)

STEP = DECISION_STEP_S
ACOUSTIC_PACKAGES = {"torch", "torchaudio", "silero_vad", "onnxruntime", "pyannote"}


def _build(cfg, dataset="all"):
    """Build the control layer the grid consumes, then the grid itself."""
    build_control_state(cfg, dataset)
    return run_vocal_action_grid_build(cfg, dataset=dataset, stream=io.StringIO())


def _by_index(grid: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Slot rows keyed by decision index, as plain Python values."""
    records: list[dict[str, Any]] = [
        {str(key): value for key, value in row.items()}
        for row in grid.to_dict(orient="records")
    ]
    return {int(row["decision_index"]): row for row in records}


def test_grid_labels_states_actions_and_masks_of_a_complete_timeline(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "ego4d")

    [output] = _build(cfg, "ego4d")
    rows = _by_index(output.grid)

    assert rows[0]["mask_reason"] == str(MaskReason.RECORDING_START)
    assert (rows[2]["focal_state_before"], rows[2]["action"]) == ("SILENT", "NO_EVENT")
    assert rows[3]["action"] == str(Action.ONSET)
    assert rows[3]["tau_s"] == pytest.approx(0.04)
    assert rows[3]["event_time_s"] == pytest.approx(0.34)
    assert rows[6]["action"] == str(Action.OFFSET)
    assert rows[6]["tau_s"] == pytest.approx(0.02)
    # an event exactly on a grid point: tau = 0 in that slot
    assert rows[10]["action"] == str(Action.ONSET)
    assert rows[10]["tau_s"] == pytest.approx(0.0)
    # the 10 ms cross-slot gap and the 40 ms same-slot gap are both bridged
    # upstream: the speech they interrupted is now one uninterrupted episode
    assert [rows[index]["action"] for index in (13, 14, 18)] == ["NO_EVENT"] * 3
    assert all(
        rows[index]["focal_state_before"] == "SPEAKING" for index in (13, 14, 18)
    )
    # UNKNOWN opens exactly on t_24 and closes exactly on t_29
    assert rows[23]["action"] == "NO_EVENT"
    assert rows[24]["mask_reason"] == str(MaskReason.UNKNOWN_WITHIN_SLOT)
    assert rows[25]["mask_reason"] == str(MaskReason.UNKNOWN_STATE)
    assert rows[29]["mask_reason"] == str(MaskReason.UNKNOWN_STATE)
    assert rows[30]["action"] == "NO_EVENT"
    assert rows[30]["focal_state_before"] == "SILENT"
    # the 20 ms speech burst is kept, so its two events still collide in one slot
    assert pd.isna(rows[35]["action"])
    assert rows[35]["mask_reason"] == str(MaskReason.COMPOUND_TRANSITION)
    # the trailing partial slot [3.9, 4.0) is not emitted
    assert max(rows) == 38
    dropped = float(output.summary["trailing_duration_dropped_s"].iloc[0])
    assert dropped == pytest.approx(0.07)
    assert 0.0 <= dropped < STEP


def test_invariants_hold_and_are_enforced(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "ego4d")

    [output] = _build(cfg, "ego4d")
    grid = output.grid

    check_invariants(grid)
    valid = grid.loc[grid["action_valid"]]
    masked = grid.loc[~grid["action_valid"]]
    assert set(valid["action"]) <= set(ACTIONS)
    assert masked["action"].isna().all() and masked["mask_reason"].notna().all()
    assert valid["mask_reason"].isna().all()
    no_event = valid.loc[valid["action"].eq("NO_EVENT")]
    assert no_event["tau_s"].isna().all() and no_event["event_time_s"].isna().all()
    events = valid.loc[valid["action"].ne("NO_EVENT")]
    assert ((events["tau_s"] >= 0) & (events["tau_s"] < STEP)).all()
    assert not grid.duplicated(["recording_id", "decision_index"]).any()
    assert np.allclose(grid["decision_time_s"], grid["decision_index"] * STEP)

    broken = grid.copy()
    broken.loc[broken.index[0], "action"] = "TAKE"
    broken.loc[broken.index[0], "action_valid"] = True
    with pytest.raises(ValueError, match="outside the vocabulary"):
        check_invariants(broken)


def test_statistics_report_the_compounds_bridging_left_behind(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "egocom")

    [output] = _build(cfg, "egocom")
    statistics = output.report["statistics"]

    assert statistics["total_slots"] == len(output.grid)
    assert statistics["valid_action_slots"] + statistics["masked_slots"] == len(
        output.grid
    )
    # only the short speech burst still collides in one slot
    assert statistics["compound_slot_count"] == 1
    assert statistics["compound_transition_count"] == 2
    assert statistics["compound_transitions_per_slot"] == {"2": 1}
    assert statistics["compound_patterns"] == {"SILENT-SPEAKING-SILENT": 1}
    assert statistics["compound_pattern_summary"] == {
        "SILENT-SPEAKING-SILENT": 1,
        "SPEAKING-SILENT-SPEAKING": 0,
        "other": 0,
    }
    # the control timeline holds no sub-Δ silence at all any more
    assert statistics["sub_delta_gap_analysis"]["sub_delta_gap_count"] == 0
    assert (
        statistics["source_transition_count"]
        == statistics["represented_transition_count"]
        + statistics["masked_transition_count"]
        + statistics["transitions_outside_grid_count"]
    )
    assert statistics["masked_transition_count"] == 2  # the burst's ONSET + OFFSET
    assert set(statistics["mask_counts_by_reason"]) == {
        "recording_start",
        "unknown_state",
        "unknown_within_slot",
        "compound_transition",
        "invalid_timeline",
    }
    assert statistics["tau_s_onset"]["min"] >= 0.0
    assert statistics["tau_s_offset"]["max"] < STEP
    assert len(output.compound_slots) == 1
    assert output.compound_slots["event_types"].iloc[0] == [
        str(Action.ONSET),
        str(Action.OFFSET),
    ]
    assert output.compound_slots["pattern"].iloc[0] == "SILENT-SPEAKING-SILENT"
    assert output.sub_delta_gaps.empty


def test_the_report_states_what_bridging_changed_in_the_grid(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "egocom")

    [output] = _build(cfg, "egocom")
    comparison = output.report["native_grid_comparison"]
    native, control = comparison["native"], comparison["control"]

    # same slots on both sides: bridging removes boundaries, never time
    assert native["total_slots"] == control["total_slots"]
    # the 40 ms same-slot gap was a compound slot; the burst still is
    assert native["compound_slot_count"] == 2
    assert control["compound_slot_count"] == 1
    assert comparison["delta"]["compound_slot_count"] == -1
    assert native["compound_pattern_summary"]["SPEAKING-SILENT-SPEAKING"] == 1
    assert control["compound_pattern_summary"]["SPEAKING-SILENT-SPEAKING"] == 0
    assert native["sub_delta_gap_analysis"]["sub_delta_gap_count"] == 2
    assert control["sub_delta_gap_analysis"]["sub_delta_gap_count"] == 0
    assert control["source_transition_count"] < native["source_transition_count"]
    assert control["valid_action_ratio"] > native["valid_action_ratio"]

    transform = output.report["control_transform"]
    assert transform["transform_kind"] == "sub_step_silence_bridge"
    assert transform["bridged_gap_count"] == 2
    assert transform["native_transition_count"] == 9
    assert transform["control_transition_count"] == 5
    assert (
        transform["native_transition_count"] - transform["control_transition_count"]
        == 4
    )


def test_output_is_identical_for_the_same_timeline_under_any_dataset(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "ego4d")
    write_native_state(cfg, "egocom")

    ego4d, egocom = _build(cfg, "all")
    shared = [
        "decision_index",
        "decision_time_s",
        "slot_end_s",
        "focal_state_before",
        "action",
        "action_valid",
        "event_time_s",
        "tau_s",
        "mask_reason",
    ]

    pd.testing.assert_frame_equal(ego4d.grid[shared], egocom.grid[shared])
    assert ego4d.grid["dataset"].eq("ego4d").all()
    assert egocom.grid["dataset"].eq("egocom").all()
    assert egocom.grid["sync_group_id"].eq("group-1").all()
    assert ego4d.grid["sync_group_id"].isna().all()


def test_build_is_deterministic_and_records_lineage(tmp_path):
    cfg = config_for(tmp_path)
    native_path = write_native_state(cfg, "ego4d")

    first = _build(cfg, "ego4d")[0]
    second = _build(cfg, "ego4d")[0]

    pd.testing.assert_frame_equal(first.grid, second.grid)
    report = json.loads(first.report_path.read_text())
    assert report["decision_step_s"] == STEP
    assert report["action_schema_version"] == ACTION_SCHEMA_VERSION
    assert report["source_control_state_schema_version"] == CONTROL_STATE_SCHEMA_VERSION
    assert report["source_native_state_schema_version"] == NATIVE_STATE_SCHEMA_VERSION
    assert report["action_space"] == list(ACTIONS)
    assert report["lineage_chain"] == [
        "native annotation",
        "native focal voice state",
        "control focal voice state",
        "vocal action grid",
    ]
    inputs = report["input_artifacts"]
    assert inputs["control_focal_voice_state_timeline"]["sha256"]
    assert inputs["control_focal_voice_state_report"]["sha256"]
    assert inputs["native_focal_voice_state_timeline"]["path"] == str(native_path)
    assert inputs["dataset_manifest"]["sha256"]
    assert report["output_artifacts"]["grid"]["sha256"]
    assert report["config_checksum"]
    assert (
        report["source_annotation_versions"]["annotation_schema_version"]
        == "ego4d-test-v1"
    )
    assert report["source_annotation_versions"]["native_state_git_commit"] == "abc123"
    assert "ego4d upstream limitation" in report["limitations"]
    assert report["policy"]["sub_delta_gaps_merged_in_this_build"] is False
    assert report["policy"]["sub_delta_gaps_bridged_upstream"] is True
    assert report["policy"]["short_speaking_bursts_filtered"] is False
    assert {"git_commit", "git_dirty", "command", "created_at"} <= set(report)
    assert (
        report["output_artifacts"]["grid"]["sha256"]
        == second.report["output_artifacts"]["grid"]["sha256"]
    )


def test_stale_or_missing_control_state_is_refused(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "ego4d")

    with pytest.raises(MissingPrerequisiteError):
        run_vocal_action_grid_build(cfg, dataset="ego4d", stream=io.StringIO())

    [control] = build_control_state(cfg, "ego4d")
    frame = pd.read_parquet(control.timeline_path)
    frame.loc[0, "canonical_end_s"] = 0.5  # rewrite the timeline behind its report
    frame.to_parquet(control.timeline_path, index=False)

    with pytest.raises(ValueError, match="does not match"):
        run_vocal_action_grid_build(cfg, dataset="ego4d", stream=io.StringIO())


def test_a_native_timeline_rewritten_behind_the_control_layer_is_refused(tmp_path):
    cfg = config_for(tmp_path)
    native_path = write_native_state(cfg, "ego4d")
    build_control_state(cfg, "ego4d")

    frame = pd.read_parquet(native_path)
    frame.loc[0, "canonical_end_s"] = 0.5
    frame.to_parquet(native_path, index=False)

    with pytest.raises(ValueError, match="does not match"):
        run_vocal_action_grid_build(cfg, dataset="ego4d", stream=io.StringIO())


def test_invalid_timeline_is_fully_masked(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(
        cfg,
        "ego4d",
        timeline=[(0.0, 0.5, "SILENT"), (0.6, 1.0, "SPEAKING")],  # gap at 0.5
    )

    [output] = _build(cfg, "ego4d")

    assert not output.grid["action_valid"].any()
    assert output.grid["mask_reason"].eq(str(MaskReason.INVALID_TIMELINE)).all()
    assert output.summary["invalid_timeline_reason"].iloc[0] == (
        "gap or overlap between intervals"
    )


def _import_closure(module: str) -> set[str]:
    """Every module reachable from ``module`` by static imports."""
    seen: set[str] = set()
    queue = [module]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        if not name.startswith("conv_wm"):
            continue  # third-party packages are leaves for this check
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError):  # pragma: no cover - defensive
            continue
        if found is None or found.origin is None or not found.origin.endswith(".py"):
            continue
        tree = ast.parse(Path(found.origin).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                queue.append(node.module)
    return seen


def test_build_needs_no_acoustic_dependency():
    closure = _import_closure("conv_wm.data.vocal_action_grid")

    assert not {name for name in closure if name.split(".")[0] in ACOUSTIC_PACKAGES}
    assert "conv_wm.data.vocal_action_grid" in closure
    assert not {
        name
        for name in closure
        if name.startswith("conv_wm.data.vocal.")
        and name.split(".")[-1] in {"detector", "audio", "identity", "coverage"}
    }
    assert "conv_wm.data.vocal.native_state" in closure
    assert "conv_wm.data.vocal.control_state" in closure


def test_grid_file_is_written_per_dataset(tmp_path):
    cfg = config_for(tmp_path)
    write_native_state(cfg, "egocom")

    [output] = _build(cfg, "egocom")

    expected = Path(cfg.paths.processed) / "vocal_action_grid" / "egocom" / GRID_TABLE
    assert output.grid_path == expected and expected.exists()
    assert not (Path(cfg.paths.processed) / "vocal_action_grid" / "ego4d").exists()


def test_compound_slot_records_every_event_including_one_on_the_grid_point(tmp_path):
    cfg = config_for(tmp_path)
    # ONSET exactly on t_12 (1.2 s, where 12 * 0.1 overshoots) and OFFSET 1 ms later
    write_native_state(
        cfg,
        "ego4d",
        timeline=[
            (0.0, 1.2, "SILENT"),
            (1.2, 1.201, "SPEAKING"),
            (1.201, 2.0, "SILENT"),
        ],
    )

    [output] = _build(cfg, "ego4d")
    compound = output.compound_slots

    assert len(compound) == 1
    row = compound.iloc[0]
    assert row["decision_index"] == 12
    assert row["transition_count"] == 2
    assert row["event_times_s"] == [pytest.approx(1.2), pytest.approx(1.201)]
    assert row["event_types"] == [str(Action.ONSET), str(Action.OFFSET)]
    assert row["state_before"] == "SILENT" and row["state_after"] == "SILENT"
    assert row["pattern"] == "SILENT-SPEAKING-SILENT"
