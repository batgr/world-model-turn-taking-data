"""The model-ready window index: anchor validity, boundaries, classes, splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from state_layers import (
    build_action_grid,
    config_for,
    long_timeline,
    write_native_state,
    write_recording_splits,
)

from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.pipeline.model_ready import (
    CANONICAL_SPLITS,
    INDEX_COLUMNS,
    INDEX_TABLE,
    METADATA_FILE,
    SplitSpec,
    apply_splits,
    assign_conversations,
    build_index,
    contract_checks,
    run_model_ready_build,
    split_leakage,
)
from conv_wm.data.vocal.windows import (
    FUTURE_STEPS,
    MAX_CONTEXT_STEPS,
    MIN_CONTEXT_STEPS,
    SampleClass,
    WindowSpec,
)


def _grid(cfg, dataset="ego4d", timeline=None, recording="rec-1"):
    write_native_state(cfg, dataset, timeline or long_timeline(), recording=recording)
    [output] = build_action_grid(cfg, dataset)
    return output.grid


def _read_grid(cfg, dataset):
    return pd.read_parquet(
        Path(cfg.paths.processed)
        / "vocal_action_grid"
        / dataset
        / "vocal_action_grid.parquet"
    )


def _build(cfg, dataset="all", spec=None):
    return run_model_ready_build(cfg, dataset=dataset, spec=spec)


def test_windows_never_cross_a_session_boundary(tmp_path):
    cfg = config_for(tmp_path)
    first = _grid(cfg, "ego4d", recording="rec-a")
    second = _grid(cfg, "egocom", recording="rec-b")
    grid = pd.concat([first, second], ignore_index=True)

    index = build_index(grid, dataset="mixed")

    # every window lies inside the rows of its own recording
    ordered = grid.sort_values(
        ["recording_id", "decision_index"], kind="stable", ignore_index=True
    )
    bounds = ordered.reset_index().groupby("recording_id")["index"].agg(["min", "max"])
    anchor_row = index["anchor_row"].to_numpy()
    first_row = index["recording_id"].map(bounds["min"]).to_numpy()
    last_row = index["recording_id"].map(bounds["max"]).to_numpy()

    assert (anchor_row - index["max_context_steps"].to_numpy() + 1 >= first_row).all()
    assert (anchor_row + index["future_steps"].to_numpy() <= last_row).all()
    assert set(index["recording_id"]) == {"rec-a", "rec-b"}


def test_windows_never_cross_a_discontinuity_inside_a_session(tmp_path):
    cfg = config_for(tmp_path)
    grid = _grid(cfg, "ego4d")
    # drop one slot in the middle: the two sides become separate segments
    hole = int(grid["decision_index"].iloc[200])
    punctured = grid.loc[~grid["decision_index"].eq(hole)].reset_index(drop=True)

    index = build_index(punctured, dataset="ego4d")

    assert index["segment_id"].nunique() == 2
    anchor = index["anchor_idx"].to_numpy()
    context_start = anchor - index["max_context_steps"].to_numpy() + 1
    future_end = anchor + index["future_steps"].to_numpy()

    assert not ((context_start <= hole) & (hole <= future_end)).any()


def test_an_anchor_without_enough_context_is_rejected(tmp_path):
    cfg = config_for(tmp_path)
    grid = _grid(cfg, "ego4d")

    index = build_index(grid, dataset="ego4d")

    first_index = int(grid["decision_index"].min())
    assert index["anchor_idx"].min() == first_index + MIN_CONTEXT_STEPS - 1
    assert (index["max_context_steps"] >= MIN_CONTEXT_STEPS).all()
    # the context an anchor supports grows with its position, capped at 50 steps
    assert (index["max_context_steps"] <= MAX_CONTEXT_STEPS).all()
    assert index["max_context_steps"].iloc[0] == MIN_CONTEXT_STEPS
    assert index["max_context_steps"].max() == MAX_CONTEXT_STEPS
    deep = index.loc[index["anchor_idx"] >= first_index + MAX_CONTEXT_STEPS]
    assert deep["max_context_steps"].eq(MAX_CONTEXT_STEPS).all()


def test_an_anchor_without_a_complete_future_is_rejected(tmp_path):
    cfg = config_for(tmp_path)
    grid = _grid(cfg, "ego4d")

    index = build_index(grid, dataset="ego4d")

    last_index = int(grid["decision_index"].max())
    assert index["anchor_idx"].max() == last_index - FUTURE_STEPS
    assert index["future_steps"].eq(FUTURE_STEPS).all()


def test_validity_thresholds_decide_trainability_not_existence(tmp_path):
    cfg = config_for(tmp_path)
    grid = _grid(cfg, "ego4d")
    # the first slot of a recording is always masked (recording_start)
    assert not bool(grid["action_valid"].iloc[0])

    tolerant = build_index(grid, dataset="ego4d", spec=WindowSpec())
    strict = build_index(
        grid, dataset="ego4d", spec=WindowSpec(min_context_valid_ratio=1.0)
    )

    # the default 0.90 threshold tolerates that one masked slot in a 10-step context
    assert tolerant["is_trainable"].all()
    # a threshold of 1.0 rejects every anchor whose context still reaches slot 0,
    # and rejected anchors stay in the index with their measured ratio
    rejected = strict.loc[~strict["is_trainable"]]
    assert len(strict) == len(tolerant)
    assert len(rejected) == MAX_CONTEXT_STEPS - MIN_CONTEXT_STEPS + 1
    assert (rejected["context_valid_ratio"] < 1.0).all()
    assert strict.loc[strict["is_trainable"], "context_valid_ratio"].eq(1.0).all()
    assert strict["future_valid_ratio"].eq(1.0).all()


def test_event_and_background_classification_follows_the_action_vocabulary(tmp_path):
    cfg = config_for(tmp_path)
    grid = _grid(cfg, "ego4d")

    index = build_index(grid, dataset="ego4d")

    assert set(index["sample_class"]) == {
        str(SampleClass.EVENT),
        str(SampleClass.BACKGROUND),
    }
    events = index.loc[index["sample_class"].eq(str(SampleClass.EVENT))]
    background = index.loc[index["sample_class"].eq(str(SampleClass.BACKGROUND))]
    assert (events["future_event_count"] > 0).all()
    assert background["future_event_count"].eq(0).all()

    # cross-check against the grid: the future window of an event sample holds
    # at least one action that is not NO_EVENT
    ordered = grid.sort_values(
        ["recording_id", "decision_index"], kind="stable", ignore_index=True
    )
    row = events.iloc[0]
    future = ordered.iloc[row["anchor_row"] + 1 : row["anchor_row"] + 1 + FUTURE_STEPS]
    assert future["action"].isin(["ONSET", "OFFSET"]).any()


def test_upstream_splits_are_propagated_per_session_and_normalised(tmp_path):
    cfg = config_for(tmp_path)
    grid = pd.concat(
        [
            _grid(cfg, "ego4d", recording="rec-a"),
            _grid(cfg, "egocom", recording="rec-b"),
        ],
        ignore_index=True,
    )
    index = build_index(grid, dataset="mixed")

    assigned = apply_splits(index, pd.Series({"rec-a": "train", "rec-b": "val"}))

    assert assigned.loc[assigned["recording_id"].eq("rec-a"), "split"].eq("train").all()
    # the release spells it "val"; the artifact always says "validation"
    assert (
        assigned.loc[assigned["recording_id"].eq("rec-b"), "split"]
        .eq("validation")
        .all()
    )
    assert set(assigned["split"]) <= set(CANONICAL_SPLITS)
    assert split_leakage(assigned) == {}


def test_no_session_appears_in_more_than_one_split(tmp_path):
    cfg = config_for(tmp_path)
    grid = pd.concat(
        [
            _grid(cfg, "ego4d", recording="rec-a"),
            _grid(cfg, "egocom", recording="rec-b"),
        ],
        ignore_index=True,
    )

    assigned = apply_splits(build_index(grid, dataset="mixed"), None)

    per_session = assigned.groupby("recording_id")["split"].nunique()
    assert per_session.eq(1).all()
    # a conversation spanning two splits is reported, never repaired
    grouped = assigned.copy()
    grouped["split"] = ["train"] * (len(grouped) - 1) + ["test"]
    grouped["conversation_id"] = "conversation-1"
    assert split_leakage(grouped) == {"conversation-1": ["test", "train"]}


def test_the_fallback_split_is_session_level_and_deterministic():
    groups = [f"conversation-{number:02d}" for number in range(20)]

    first = assign_conversations(groups, SplitSpec(seed=7))
    again = assign_conversations(list(reversed(groups)), SplitSpec(seed=7))
    other = assign_conversations(groups, SplitSpec(seed=8))

    assert first == again  # row order cannot change the assignment
    assert first != other  # but the seed can
    assert set(first.values()) == set(CANONICAL_SPLITS)
    assert sum(split == "train" for split in first.values()) == 16
    assert sum(split == "validation" for split in first.values()) == 2
    assert sum(split == "test" for split in first.values()) == 2


def test_the_fallback_split_keeps_a_conversation_whole(tmp_path):
    cfg = config_for(tmp_path)
    grid = pd.concat(
        [
            _grid(cfg, "egocom", recording="rec-a"),
            _grid(cfg, "egocom", recording="rec-b"),
        ],
        ignore_index=True,
    )
    index = build_index(grid, dataset="egocom")
    # both point-of-view videos record the same conversation
    index["conversation_id"] = "conversation-1"

    assigned = apply_splits(index, None, spec=SplitSpec(seed=3))

    assert assigned["split"].nunique() == 1
    assert split_leakage(assigned) == {}


def test_build_writes_the_index_and_reports_its_statistics(tmp_path):
    cfg = config_for(tmp_path)
    _grid(cfg, "ego4d")
    write_recording_splits(cfg, "ego4d", {"rec-1": "train"})

    [output] = _build(cfg, "ego4d")
    statistics = output.report["statistics"]

    expected = Path(cfg.paths.model_ready) / "ego4d" / INDEX_TABLE
    assert output.index_path == expected and expected.exists()
    assert statistics["recording_count"] == 1
    assert statistics["sample_count"] == int(output.index["is_trainable"].sum())
    assert sum(statistics["class_counts"].values()) == statistics["sample_count"]
    assert statistics["window_geometry"] == {
        "decision_step_s": 0.1,
        "min_context_steps": MIN_CONTEXT_STEPS,
        "max_context_steps": MAX_CONTEXT_STEPS,
        "future_steps": FUTURE_STEPS,
        "min_context_seconds": pytest.approx(1.0),
        "max_context_seconds": pytest.approx(5.0),
        "future_seconds": pytest.approx(1.0),
    }
    assert all(statistics["contract_checks"].values())
    assert contract_checks(output.index, _read_grid(cfg, "ego4d"), WindowSpec()) == {
        "no_recording_in_multiple_splits": True,
        "every_anchor_has_minimum_context": True,
        "every_anchor_has_full_future": True,
        "no_anchor_crosses_a_recording_boundary": True,
    }
    report = json.loads(output.report_path.read_text())
    assert report["lineage_chain"][-1] == "model-ready window index"
    assert report["input_artifacts"]["vocal_action_grid"]["sha256"]
    assert report["output_artifacts"]["index"]["sha256"]
    assert report["config_checksum"]
    assert report["policy"]["windows_cross_recording_boundary"] is False
    assert not output.index.duplicated(["sample_id"]).any()
    assert output.index["split"].eq("train").all()
    assert output.index["split_source"].eq("upstream").all()
    assert statistics["split_counts"] == {"train": statistics["sample_count"]}
    assert statistics["split_leakage"] == {}

    # the index stays an index: no materialized window, no feature array
    assert set(output.index.columns) == set(INDEX_COLUMNS)
    assert not any(
        output.index[column]
        .map(lambda value: isinstance(value, (list, np.ndarray)))
        .any()
        for column in output.index.columns
    )

    metadata = json.loads(output.metadata_path.read_text())
    assert output.metadata_path.name == METADATA_FILE
    assert output.metadata_path.parent == output.index_path.parent
    assert metadata["grid"]["frequency_hz"] == 10.0
    assert metadata["grid"]["timestep_seconds"] == 0.1
    assert metadata["windows"]["materialized"] is False
    assert metadata["windows"]["min_context_steps"] == MIN_CONTEXT_STEPS
    assert metadata["windows"]["max_context_steps"] == MAX_CONTEXT_STEPS
    assert metadata["windows"]["future_steps"] == FUTURE_STEPS
    assert metadata["validity_thresholds"] == {
        "min_context_valid_ratio": 0.9,
        "min_future_valid_ratio": 1.0,
    }
    assert metadata["splits"]["names"] == list(CANONICAL_SPLITS)
    assert metadata["counts"]["recordings"] == 1
    assert metadata["source"]["vocal_action_grid"]["sha256"]
    assert all(metadata["contract_checks"].values())


def test_a_stale_or_missing_action_grid_is_refused(tmp_path):
    cfg = config_for(tmp_path)

    with pytest.raises(MissingPrerequisiteError):
        _build(cfg, "ego4d")

    _grid(cfg, "ego4d")
    write_recording_splits(cfg, "ego4d", {"rec-1": "train"})
    grid_path = (
        Path(cfg.paths.processed)
        / "vocal_action_grid"
        / "ego4d"
        / "vocal_action_grid.parquet"
    )
    frame = pd.read_parquet(grid_path)
    frame.loc[0, "decision_index"] = 999  # rewrite the grid behind its report
    frame.to_parquet(grid_path, index=False)

    with pytest.raises(ValueError, match="does not match"):
        _build(cfg, "ego4d")
