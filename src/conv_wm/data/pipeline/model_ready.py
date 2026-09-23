"""Build the model-ready window index from the vocal action grid.

The artifact is an **index**, not a copy of the grid: one row per valid anchor,
saying where a window can be extracted and what it contains. The grid stays the
single source of per-step data, and a separate modelling repository
reconstructs windows from it.

```text
vocal action grid -> model-ready window index -> WindowDataset -> DataLoader
```
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import PROJECT_ROOT, pipeline_paths
from conv_wm.data import datasets
from conv_wm.data.pipeline.action_grid import (
    GRID_TABLE,
)
from conv_wm.data.pipeline.action_grid import (
    PROCESSED_ROOT as GRID_PROCESSED_ROOT,
)
from conv_wm.data.pipeline.action_grid import (
    REPORT_FILE as GRID_REPORT_FILE,
)
from conv_wm.data.pipeline.action_grid import (
    REPORT_ROOT as GRID_REPORT_ROOT,
)
from conv_wm.data.pipeline.control_state import config_checksum
from conv_wm.data.pipeline_inputs import (
    CheckedArtifact,
    artifact_reference,
    load_checked_artifact,
    sha256_file,
)
from conv_wm.data.vocal.action_grid import ACTION_SCHEMA_VERSION, DECISION_STEP_S
from conv_wm.data.vocal.windows import (
    GRID_SORT_COLUMNS,
    WINDOW_SCHEMA_VERSION,
    SampleClass,
    WindowSpec,
    canonical_grid,
    classify,
    encode_actions,
    is_trainable,
    segment_anchors,
    segment_starts,
)
from conv_wm.provenance import collect_provenance
from conv_wm.reports import JsonDict, write_summary, write_table

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = 1
REPORT_ROOT = Path("model_ready")
INDEX_TABLE = "index.parquet"
METADATA_FILE = "metadata.json"
SUMMARY_TABLE = "summary.parquet"
REPORT_FILE = "report.json"

GRID_BUILD_COMMAND = "conv-wm build vocal-action-grid"
DEFAULT_SPEC = WindowSpec()
"""The prototype window geometry, used unless a caller passes its own."""
DEFAULT_SPLIT_SPEC_SEED = 0
"""Seed of the fallback recording-level split, when a dataset carries none."""
INDEX_COLUMNS = (
    "sample_id",
    "dataset",
    "recording_id",
    "conversation_id",
    "view_id",
    "wearer_id",
    "split",
    "segment_id",
    "anchor_idx",
    "anchor_time",
    "anchor_row",
    "max_context_steps",
    "future_steps",
    "context_valid_ratio",
    "future_valid_ratio",
    "future_event_count",
    "sample_class",
    "is_trainable",
    "split_source",
    "window_schema_version",
    "action_schema_version",
)
LINEAGE_CHAIN = (
    "native annotation",
    "native focal voice state",
    "control focal voice state",
    "vocal action grid",
    "model-ready window index",
)


@dataclass(frozen=True)
class ModelReadyOutputs:
    """Index, per-recording summary and report written for one dataset."""

    dataset: str
    index: pd.DataFrame
    summary: pd.DataFrame
    report: JsonDict
    index_path: Path
    metadata_path: Path
    summary_path: Path
    report_path: Path


def supported_datasets() -> tuple[str, ...]:
    """Registered datasets whose action grid this stage can window."""
    return tuple(spec.name for spec in datasets.iter_specs() if spec.native_focal_voice)


def selected_datasets(dataset: str) -> tuple[str, ...]:
    """Expand the CLI selector (``all`` or one supported dataset)."""
    supported = supported_datasets()
    if dataset == "all":
        return supported
    if dataset not in supported:
        raise ValueError(
            f"dataset must be one of {[*supported, 'all']}, got {dataset!r}"
        )
    return (dataset,)


def load_action_grid_input(cfg: DictConfig, dataset: str) -> CheckedArtifact:
    """Read one dataset's action grid, refusing an input its report disowns."""
    paths = pipeline_paths(cfg)
    return load_checked_artifact(
        dataset=dataset,
        table_path=paths.processed / GRID_PROCESSED_ROOT / dataset / GRID_TABLE,
        report_path=paths.reports / GRID_REPORT_ROOT / dataset / GRID_REPORT_FILE,
        schema_key="action_schema_version",
        expected_version=ACTION_SCHEMA_VERSION,
        artifact_key="grid",
        produce_with=f"{GRID_BUILD_COMMAND} --dataset {dataset}",
    )


def recording_splits(cfg: DictConfig, dataset: str) -> pd.Series | None:
    """The dataset's own ``recording_id -> split`` map, or ``None`` when it has none."""
    loader = datasets.get(dataset).recording_splits
    if loader is None:
        return None
    table = loader(cfg)
    return (
        pd.Series(
            table["split"].to_numpy(),
            index=pd.Index(table["recording_id"].to_numpy(), name="recording_id"),
            dtype="string",
        )
        .groupby(level=0)
        .first()
    )


def build_index(
    grid: pd.DataFrame, *, dataset: str, spec: WindowSpec = DEFAULT_SPEC
) -> pd.DataFrame:
    """One row per structurally valid anchor of ``grid``, in canonical order.

    Anchors are computed per segment (a run of consecutive ``decision_index``
    values of one recording), so no window crosses a recording boundary or a hole
    inside a recording. ``anchor_row`` addresses the grid returned by
    :func:`~conv_wm.data.vocal.windows.canonical_grid`.
    """
    ordered = canonical_grid(grid)
    if ordered.empty:
        return _empty_index()
    recording = ordered["recording_id"].to_numpy()
    decision_index = ordered["decision_index"].to_numpy(dtype=np.int64)
    action_valid = ordered["action_valid"].to_numpy(dtype=bool)
    action_codes = encode_actions(ordered["action"])
    recording_breaks = np.flatnonzero(recording[1:] != recording[:-1]) + 1
    frames: list[pd.DataFrame] = []
    for first, last in zip(
        np.concatenate([[0], recording_breaks]),
        np.concatenate([recording_breaks, [len(ordered)]]),
        strict=True,
    ):
        identity = ordered.iloc[first]
        starts = segment_starts(decision_index[first:last]) + first
        for segment, (begin, end) in enumerate(
            zip(starts, np.concatenate([starts[1:], [last]]), strict=True)
        ):
            anchors = segment_anchors(
                action_valid=action_valid[begin:end],
                action_codes=action_codes[begin:end],
                spec=spec,
            )
            if not len(anchors.position):
                continue
            rows = begin + anchors.position
            frames.append(
                pd.DataFrame(
                    {
                        "recording_id": identity["recording_id"],
                        "conversation_id": identity["sync_group_id"]
                        if pd.notna(identity["sync_group_id"])
                        else identity["recording_id"],
                        "view_id": identity["view_id"],
                        "wearer_id": identity["wearer_id"],
                        "segment_id": segment,
                        "anchor_idx": decision_index[rows],
                        "anchor_time": ordered["decision_time_s"].to_numpy()[rows],
                        "anchor_row": rows,
                        "max_context_steps": anchors.max_context_steps,
                        "future_steps": spec.future_steps,
                        "context_valid_ratio": anchors.context_valid_ratio,
                        "future_valid_ratio": anchors.future_valid_ratio,
                        "future_event_count": anchors.future_event_count,
                        "sample_class": classify(anchors.future_event_count),
                        "is_trainable": is_trainable(anchors, spec),
                    }
                )
            )
    if not frames:
        return _empty_index()
    index = pd.concat(frames, ignore_index=True)
    index["dataset"] = dataset
    index["sample_id"] = (
        index["recording_id"].astype(str) + "#" + index["anchor_idx"].astype(str)
    )
    index["split"] = pd.Series(pd.NA, index=index.index, dtype="string")
    index["split_source"] = pd.Series(pd.NA, index=index.index, dtype="string")
    index["window_schema_version"] = WINDOW_SCHEMA_VERSION
    index["action_schema_version"] = ACTION_SCHEMA_VERSION
    for column in (
        "sample_id",
        "dataset",
        "recording_id",
        "conversation_id",
        "view_id",
        "wearer_id",
        "sample_class",
    ):
        index[column] = index[column].astype("string")
    return index[list(INDEX_COLUMNS)]


def _empty_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(
                dtype="string"
                if "id" in column or column in {"split", "sample_class"}
                else float
            )
            for column in INDEX_COLUMNS
        }
    )


CANONICAL_SPLITS = ("train", "validation", "test")
"""The split names this stage emits, whatever an upstream release calls them."""

SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation"}
"""Upstream spellings normalised onto :data:`CANONICAL_SPLITS`."""


@dataclass(frozen=True)
class SplitSpec:
    """How recordings are assigned to splits when a dataset carries no split.

    Fractions are of *conversations*, not of samples: every recording of a
    group lands in the same split, so no conversation is ever cut across two.
    """

    seed: int = 0
    train_fraction: float = 0.8
    validation_fraction: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError(
                f"train_fraction must be in (0, 1), got {self.train_fraction}"
            )
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in [0, 1), got {self.validation_fraction}"
            )
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must leave room for test")

    @property
    def test_fraction(self) -> float:
        """The remaining fraction of groups."""
        return 1.0 - self.train_fraction - self.validation_fraction


def canonical_split(value: object) -> str | None:
    """One upstream split label normalised, or ``None`` when it is absent."""
    if value is None:
        return None
    name = str(value).strip().lower()
    if name in {"", "nan", "none", "<na>"}:
        return None
    return SPLIT_ALIASES.get(name, name)


def assign_conversations(groups: Sequence[str], spec: SplitSpec) -> dict[str, str]:
    """Deterministically assign conversations to the canonical splits.

    Groups are sorted before shuffling, so the assignment depends only on the
    set of groups and ``spec.seed`` — never on row order or on the corpus the
    build happened to see first.
    """
    ordered = sorted({str(group) for group in groups})
    if not ordered:
        return {}
    shuffled = list(
        np.random.default_rng(spec.seed).permutation(np.asarray(ordered, dtype=object))
    )
    train_end = round(len(shuffled) * spec.train_fraction)
    validation_end = train_end + round(len(shuffled) * spec.validation_fraction)
    assignment = dict.fromkeys(shuffled[:train_end], "train")
    assignment |= dict.fromkeys(shuffled[train_end:validation_end], "validation")
    assignment |= dict.fromkeys(shuffled[validation_end:], "test")
    return {str(group): split for group, split in assignment.items()}


def apply_splits(
    index: pd.DataFrame,
    splits: pd.Series | None,
    *,
    spec: SplitSpec | None = None,
) -> pd.DataFrame:
    """Give every sample the split of its recording, normalised to canonical names.

    An upstream per-recording assignment is preferred and never recomputed; the
    recordings it does not cover are assigned deterministically at *group* level
    from ``spec.seed``. Splitting individual anchors is never possible here:
    the assignment is a map from group or recording, applied to whole recordings.
    """
    if index.empty:
        return index
    spec = spec or SplitSpec()
    assigned = index.copy()
    upstream = (
        assigned["recording_id"].map(splits).map(canonical_split)
        if splits is not None
        else pd.Series(None, index=assigned.index, dtype="object")
    )
    missing = upstream.isna()
    if missing.any():
        fallback = assign_conversations(
            assigned.loc[missing, "conversation_id"].tolist(), spec
        )
        upstream = upstream.where(
            ~missing, assigned["conversation_id"].astype(str).map(fallback)
        )
    assigned["split"] = upstream.astype("string")
    assigned["split_source"] = pd.Series(
        np.where(missing, "deterministic", "upstream"), index=assigned.index
    ).astype("string")
    return assigned


def split_leakage(index: pd.DataFrame) -> dict[str, list[str]]:
    """Conversation groups whose recordings do not all share one split.

    EgoCom splits point-of-view videos, several of which record the same
    conversation; a group spanning two splits would leak between them. Reported,
    never silently repaired.
    """
    if index.empty or index["split"].isna().all():
        return {}
    per_group = index.groupby("conversation_id", observed=True)["split"].nunique(
        dropna=True
    )
    leaking = per_group.loc[per_group > 1].index
    return {
        str(group): sorted(
            str(value)
            for value in index.loc[index["conversation_id"].eq(group), "split"]
            .dropna()
            .unique()
        )
        for group in leaking
    }


def contract_checks(
    index: pd.DataFrame, grid: pd.DataFrame, spec: WindowSpec
) -> dict[str, bool]:
    """Re-verify the index's promises against the grid it was built from.

    Cheap and independent of the code that produced the index, so a regression
    in anchor generation shows up as a failed check in the report rather than
    as a silently wrong dataset.
    """
    if index.empty:
        return dict.fromkeys(
            (
                "no_recording_in_multiple_splits",
                "every_anchor_has_minimum_context",
                "every_anchor_has_full_future",
                "no_anchor_crosses_a_recording_boundary",
            ),
            True,
        )
    ordered = canonical_grid(grid)
    bounds = ordered.groupby("recording_id", observed=True)["decision_index"].agg(
        ["min", "max"]
    )
    recording = index["recording_id"]
    first = recording.map(bounds["min"]).to_numpy()
    last = recording.map(bounds["max"]).to_numpy()
    anchor = index["anchor_idx"].to_numpy()
    context_start = anchor - index["max_context_steps"].to_numpy() + 1
    future_end = anchor + index["future_steps"].to_numpy()
    rows = ordered["recording_id"].to_numpy()
    return {
        "no_recording_in_multiple_splits": bool(
            index.groupby("recording_id", observed=True)["split"]
            .nunique(dropna=False)
            .le(1)
            .all()
        )
        and not split_leakage(index),
        "every_anchor_has_minimum_context": bool(
            (index["max_context_steps"] >= spec.min_context_steps).all()
            and (index["max_context_steps"] <= spec.max_context_steps).all()
        ),
        "every_anchor_has_full_future": bool((future_end <= last).all()),
        "no_anchor_crosses_a_recording_boundary": bool(
            (context_start >= first).all()
            and (future_end <= last).all()
            and np.array_equal(
                rows[index["anchor_row"].to_numpy()], recording.to_numpy()
            )
        ),
    }


def _summary_table(index: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """One row per recording: anchors, classes, splits and trainability."""
    if index.empty:
        return pd.DataFrame(columns=["dataset", "recording_id"])
    trainable = index.loc[index["is_trainable"]]
    grouped = index.groupby("recording_id", observed=True)
    summary = pd.DataFrame(
        {
            "dataset": dataset,
            "anchor_count": grouped.size(),
            "trainable_count": trainable.groupby("recording_id", observed=True)
            .size()
            .reindex(grouped.size().index, fill_value=0),
            "conversation_id": grouped["conversation_id"].first(),
            "view_id": grouped["view_id"].first(),
            "wearer_id": grouped["wearer_id"].first(),
            "split": grouped["split"].first(),
            "segment_count": grouped["segment_id"].nunique(),
            "first_anchor_idx": grouped["anchor_idx"].min(),
            "last_anchor_idx": grouped["anchor_idx"].max(),
            "mean_context_valid_ratio": grouped["context_valid_ratio"].mean(),
        }
    )
    for name in SampleClass:
        selected = trainable.loc[trainable["sample_class"].eq(str(name))]
        summary[f"{name}_count"] = (
            selected.groupby("recording_id", observed=True)
            .size()
            .reindex(summary.index, fill_value=0)
        )
    return summary.reset_index().sort_values(
        "recording_id", kind="stable", ignore_index=True
    )


def _value_counts(values: pd.Series) -> dict[str, int]:
    """Integer value counts, keyed by the value, in increasing order."""
    if values.empty:
        return {}
    counted = values.value_counts().sort_index()
    return {
        str(int(key)): int(count)
        for key, count in zip(
            counted.index.to_numpy(dtype=np.int64),
            counted.to_numpy(dtype=np.int64),
            strict=True,
        )
    }


def _class_counts(index: pd.DataFrame) -> dict[str, int]:
    return {
        str(name): int(index["sample_class"].eq(str(name)).sum())
        for name in SampleClass
    }


def _statistics(
    index: pd.DataFrame, grid: pd.DataFrame, spec: WindowSpec
) -> dict[str, object]:
    """The numbers a prototype training run needs before it starts."""
    trainable = index.loc[index["is_trainable"]] if not index.empty else index
    counts = _class_counts(trainable) if not trainable.empty else _class_counts(index)
    event = counts[str(SampleClass.EVENT)]
    background = counts[str(SampleClass.BACKGROUND)]
    return {
        "grid_slot_count": len(grid),
        "recording_count": int(index["recording_id"].nunique())
        if not index.empty
        else 0,
        "conversation_count": int(index["conversation_id"].nunique())
        if not index.empty
        else 0,
        "anchor_count": len(index),
        "sample_count": len(trainable),
        "rejected_by_validity_count": len(index) - len(trainable),
        "class_counts": counts,
        "event_ratio": event / len(trainable) if len(trainable) else None,
        "event_background_ratio": event / background if background else None,
        "split_counts": (
            {
                str(key): int(value)
                for key, value in trainable["split"].value_counts(dropna=False).items()
            }
            if not trainable.empty
            else {}
        ),
        "split_class_counts": (
            {
                str(key): _class_counts(group)
                for key, group in trainable.groupby(
                    "split", dropna=False, observed=True
                )
            }
            if not trainable.empty
            else {}
        ),
        "split_leakage": split_leakage(index),
        "recordings_per_split": (
            {
                str(key): int(group["recording_id"].nunique())
                for key, group in trainable.groupby(
                    "split", dropna=False, observed=True
                )
            }
            if not trainable.empty
            else {}
        ),
        "contract_checks": contract_checks(index, grid, spec),
        "split_source_recordings": (
            {
                str(key): int(group["recording_id"].nunique())
                for key, group in index.groupby(
                    "split_source", dropna=False, observed=True
                )
            }
            if not index.empty
            else {}
        ),
        "max_context_steps_distribution": _value_counts(trainable["max_context_steps"]),
        "context_valid_ratio_mean": float(trainable["context_valid_ratio"].mean())
        if not trainable.empty
        else None,
        "segments_per_session_max": int(
            index.groupby("recording_id", observed=True)["segment_id"].nunique().max()
        )
        if not index.empty
        else 0,
        "window_geometry": {
            "decision_step_s": DECISION_STEP_S,
            "min_context_steps": spec.min_context_steps,
            "max_context_steps": spec.max_context_steps,
            "future_steps": spec.future_steps,
            "min_context_seconds": spec.min_context_seconds,
            "max_context_seconds": spec.max_context_seconds,
            "future_seconds": spec.future_seconds,
        },
    }


def _dataset_metadata(
    source: CheckedArtifact,
    *,
    statistics: dict[str, object],
    spec: WindowSpec,
    split_spec: SplitSpec,
    index_path: Path,
    provenance_time: str,
    git_commit: str | None,
) -> JsonDict:
    """The self-contained card shipped next to the index.

    It describes the artifact for whoever loads it — the modelling repository
    or a Hugging Face dataset card — without requiring this repository's
    reports directory.
    """
    geometry = statistics["window_geometry"]
    assert isinstance(geometry, dict)
    return {
        "dataset": source.dataset,
        "dataset_version": f"window-index-v{WINDOW_SCHEMA_VERSION}",
        "window_schema_version": WINDOW_SCHEMA_VERSION,
        "source_action_schema_version": ACTION_SCHEMA_VERSION,
        "generated_at_utc": provenance_time,
        "git_commit": git_commit,
        "files": {
            "index": index_path.name,
            "sequences": str(source.table_path),
        },
        "grid": {
            "frequency_hz": round(1.0 / DECISION_STEP_S, 6),
            "timestep_seconds": DECISION_STEP_S,
            "slot_count": statistics["grid_slot_count"],
            "sequence_key": list(GRID_SORT_COLUMNS),
        },
        "windows": {
            "min_context_steps": geometry["min_context_steps"],
            "max_context_steps": geometry["max_context_steps"],
            "future_steps": geometry["future_steps"],
            "min_context_seconds": geometry["min_context_seconds"],
            "max_context_seconds": geometry["max_context_seconds"],
            "future_seconds": geometry["future_seconds"],
            "context": "[anchor_idx - L + 1, anchor_idx] for any L in [min, max_context_steps]",
            "future": "[anchor_idx + 1, anchor_idx + future_steps]",
            "materialized": False,
        },
        "validity_thresholds": {
            "min_context_valid_ratio": spec.min_context_valid_ratio,
            "min_future_valid_ratio": spec.min_future_valid_ratio,
        },
        "splits": {
            "names": list(CANONICAL_SPLITS),
            "level": "conversation (never an individual anchor)",
            "fallback_seed": split_spec.seed,
            "anchors": statistics["split_counts"],
            "recordings": statistics["recordings_per_split"],
            "sessions_by_source": statistics["split_source_recordings"],
            "note": (
                "a split absent from this listing is absent from the release "
                "upstream; it is never synthesised from another split"
            ),
        },
        "counts": {
            "recordings": statistics["recording_count"],
            "groups": statistics["conversation_count"],
            "anchors": statistics["anchor_count"],
            "trainable_anchors": statistics["sample_count"],
            "classes": statistics["class_counts"],
        },
        "how_to_reconstruct_a_window": (
            "Read the sequences table, keep the rows of index.recording_id "
            "(its recording_id), order them by decision_index: the anchor is the "
            "row with decision_index == anchor_idx, the context is the L rows "
            "ending there and the future is the next future_steps rows. "
            "anchor_row is that row's position in the sequences table sorted by "
            f"{list(GRID_SORT_COLUMNS)}."
        ),
        "source": {
            "vocal_action_grid": {
                "path": str(source.table_path),
                "sha256": source.table_sha256,
            },
            "annotation_versions": source.report.get("source_annotation_versions", {}),
        },
        "contract_checks": statistics["contract_checks"],
    }


def _build_dataset(
    source: CheckedArtifact,
    *,
    cfg: DictConfig,
    spec: WindowSpec,
    split_spec: SplitSpec,
    command: str,
) -> ModelReadyOutputs:
    paths = pipeline_paths(cfg)
    index = apply_splits(
        build_index(source.table, dataset=source.dataset, spec=spec),
        recording_splits(cfg, source.dataset),
        spec=split_spec,
    )
    summary = _summary_table(index, source.dataset)
    statistics = _statistics(index, source.table, spec)

    processed_dir = paths.model_ready / source.dataset
    report_dir = paths.reports / REPORT_ROOT / source.dataset
    index_path = write_table(processed_dir / INDEX_TABLE, index)
    summary_path = write_table(report_dir / SUMMARY_TABLE, summary)

    provenance = collect_provenance()
    uv_lock = PROJECT_ROOT / "uv.lock"
    grid_report = source.report
    payload: JsonDict = {
        "build_type": "model_ready_window_index",
        "schema_version": REPORT_SCHEMA_VERSION,
        "window_schema_version": WINDOW_SCHEMA_VERSION,
        "source_action_schema_version": ACTION_SCHEMA_VERSION,
        "dataset": source.dataset,
        "created_at": provenance.generated_at_utc,
        "git_commit": provenance.git_commit,
        "git_dirty": provenance.git_dirty,
        "command": command,
        "config_checksum": config_checksum(cfg),
        "lineage_chain": list(LINEAGE_CHAIN),
        "window_spec": {
            "min_context_steps": spec.min_context_steps,
            "max_context_steps": spec.max_context_steps,
            "future_steps": spec.future_steps,
            "min_context_valid_ratio": spec.min_context_valid_ratio,
            "min_future_valid_ratio": spec.min_future_valid_ratio,
            "feature_columns": list(spec.feature_columns),
        },
        "anchor_contract": {
            "context": "[anchor_idx - L + 1, anchor_idx], L in [min, max_context_steps]",
            "future": "[anchor_idx + 1, anchor_idx + future_steps]",
            "boundaries": (
                "anchors are per segment (a run of consecutive decision_index of "
                "one recording), so no window crosses a recording or a hole"
            ),
            "context_valid_ratio": "measured over the largest context the anchor allows",
            "is_trainable": "both validity ratios satisfy the configured thresholds",
            "sample_class": (
                f"event when the future logs one of {list(SampleClass)} actions "
                "other than NO_EVENT, background otherwise"
            ),
            "split": (
                "the dataset's own per-recording assignment, normalised to "
                f"{list(CANONICAL_SPLITS)}; recordings it does not cover are "
                "assigned deterministically at conversation-group level"
            ),
        },
        "split_spec": {
            "names": list(CANONICAL_SPLITS),
            "level": "conversation",
            "seed": split_spec.seed,
            "train_fraction": split_spec.train_fraction,
            "validation_fraction": split_spec.validation_fraction,
            "test_fraction": split_spec.test_fraction,
            "upstream_preferred": True,
        },
        "input_artifacts": {
            "vocal_action_grid": {
                "path": str(source.table_path),
                "sha256": source.table_sha256,
            },
            "vocal_action_grid_report": {
                "path": str(source.report_path),
                "sha256": source.report_sha256,
            },
            "control_focal_voice_state_timeline": grid_report.get(
                "input_artifacts", {}
            ).get("control_focal_voice_state_timeline"),
        },
        "source_annotation_versions": grid_report.get("source_annotation_versions", {}),
        "uv_lock_checksum": sha256_file(uv_lock) if uv_lock.exists() else None,
        "output_artifacts": {
            "index": artifact_reference(index_path),
            "summary": artifact_reference(summary_path),
        },
        "statistics": statistics,
        "policy": {
            "sliding_windows_materialized": False,
            "grid_features_duplicated": False,
            "windows_cross_recording_boundary": False,
            "windows_cross_segment_boundary": False,
            "upstream_splits_recomputed": False,
            "anchors_split_individually": False,
            "classes_rebalanced": False,
            "future_horizon_varies": False,
        },
        "limitations": [
            (
                "The prototype exposes the vocal action grid only: there is no "
                "multimodal feature column yet, so a window's per-step observation "
                "is the focal state and action sequence."
            ),
            (
                "context_valid_ratio is measured over the largest context an anchor "
                "allows; a shorter sampled context may have a different ratio, which "
                "is why every sample carries its per-step validity mask."
            ),
            *grid_report.get("limitations", []),
        ],
    }
    report = write_summary(report_dir / REPORT_FILE, payload, provenance=provenance)
    metadata_path = processed_dir / METADATA_FILE
    write_summary(
        metadata_path,
        _dataset_metadata(
            source,
            statistics=statistics,
            spec=spec,
            split_spec=split_spec,
            index_path=index_path,
            provenance_time=provenance.generated_at_utc,
            git_commit=provenance.git_commit,
        ),
        provenance=provenance,
    )
    return ModelReadyOutputs(
        source.dataset,
        index,
        summary,
        report,
        index_path,
        metadata_path,
        summary_path,
        report_dir / REPORT_FILE,
    )


def run_model_ready_build(
    cfg: DictConfig,
    *,
    dataset: str = "all",
    spec: WindowSpec | None = None,
    split_spec: SplitSpec | None = None,
    command: str | None = None,
) -> list[ModelReadyOutputs]:
    """Build the model-ready window index of every selected dataset."""
    geometry = spec or DEFAULT_SPEC
    splits = split_spec or SplitSpec(seed=DEFAULT_SPLIT_SPEC_SEED)
    invoked = command or f"conv-wm build model-ready --dataset {dataset}"
    outputs: list[ModelReadyOutputs] = []
    for name in selected_datasets(dataset):
        source = load_action_grid_input(cfg, name)
        logger.info("%s: %d grid slots", name, len(source.table))
        outputs.append(
            _build_dataset(
                source,
                cfg=cfg,
                spec=geometry,
                split_spec=splits,
                command=invoked,
            )
        )
    return outputs


def format_outputs(outputs: list[ModelReadyOutputs]) -> str:
    """The per-dataset summary and contract report a generation run prints."""
    lines: list[str] = []
    for output in outputs:
        statistics = output.report["statistics"]
        geometry = statistics["window_geometry"]
        counts = statistics["class_counts"]
        ratio = statistics["event_background_ratio"]
        lines.append(f"[{output.dataset}]")
        lines.append(f"Recordings: {statistics['recording_count']}")
        for split in CANONICAL_SPLITS:
            lines.append("")
            lines.append(f"{split.capitalize()}:")
            lines.append(
                f"  recordings: {statistics['recordings_per_split'].get(split, 0)}"
            )
            lines.append(f"  anchors: {statistics['split_counts'].get(split, 0)}")
        lines.append("")
        lines.append(f"Event anchors: {counts[str(SampleClass.EVENT)]}")
        lines.append(f"Background anchors: {counts[str(SampleClass.BACKGROUND)]}")
        lines.append(
            "Event/background ratio: "
            + (f"{ratio:.4f}" if ratio is not None else "n/a")
        )
        lines.append(
            f"Rejected by validity: {statistics['rejected_by_validity_count']} "
            f"of {statistics['anchor_count']} anchors"
        )
        lines.append("")
        lines.append(
            f"Context support: {geometry['min_context_seconds']:.1f}-"
            f"{geometry['max_context_seconds']:.1f} s"
        )
        lines.append(f"Future horizon: {geometry['future_seconds']:.1f} s")
        lines.append(f"Grid: {round(1.0 / geometry['decision_step_s'])} Hz")
        lines.append("")
        for check, passed in statistics["contract_checks"].items():
            lines.append(
                f"  [{'ok' if passed else 'FAILED'}] {check.replace('_', ' ')}"
            )
        if statistics["split_leakage"]:
            lines.append(f"  [FAILED] split leakage: {statistics['split_leakage']}")
        lines.append("")
        lines.append(f"  index:    {output.index_path}")
        lines.append(f"  metadata: {output.metadata_path}")
        lines.append(f"  report:   {output.report_path}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "CANONICAL_SPLITS",
    "INDEX_COLUMNS",
    "INDEX_TABLE",
    "METADATA_FILE",
    "ModelReadyOutputs",
    "SplitSpec",
    "apply_splits",
    "assign_conversations",
    "build_index",
    "canonical_split",
    "contract_checks",
    "format_outputs",
    "load_action_grid_input",
    "recording_splits",
    "run_model_ready_build",
    "selected_datasets",
    "split_leakage",
    "supported_datasets",
]
