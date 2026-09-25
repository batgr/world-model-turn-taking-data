"""Label coverage and sanity statistics: an audit of the built label stores.

For every registry label and every dataset: whether the dataset can support
it (from declared facts), whether it is materialized, its source kind,
modalities and time reference, and — when materialized — the share of units
that carry a value (*coverage*), the share that is valid (*valid*: a true
companion ``*_valid`` column, or a non-null value when there is none) and the
complement (*missing*). Nested values are counted per element (one element
per participant, subframe or horizon).

It also reports the obvious sanity statistics (speaking and overlap time,
event counts, floor transfers, turn durations, time-to-next distributions,
labels per modality). This is an audit: it never modifies a label.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from conv_wm.data.labels import store
from conv_wm.data.labels.catalog import REGISTRY
from conv_wm.data.labels.registry import (
    Availability,
    LabelSpec,
    Modality,
    Table,
)

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
VALIDITY_COLUMNS = (
    "valid",
    "context_valid",
    "timing_valid",
    "speech_rate_valid",
    "resolved",
)


def _leaves(array: pa.ChunkedArray | pa.Array) -> tuple[pa.Array, np.ndarray]:
    """Every leaf value of a (possibly nested) column and its null mask.

    A null parent list stands for its missing elements: a null fixed-size list
    for ``list_size`` of them, a null variable-size list (read back from Parquet
    with no element) for the most common length of its non-null siblings.
    """
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    null = array.is_null().to_numpy(zero_copy_only=False)
    while isinstance(array, (pa.ListArray, pa.FixedSizeListArray)):
        if isinstance(array, pa.FixedSizeListArray):
            size = array.type.list_size
            values = array.values
            null = np.repeat(null, size) | values.is_null().to_numpy(
                zero_copy_only=False
            )
            array = values
            continue
        offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        lengths = np.diff(offsets)
        present = lengths[~null]
        fill = int(np.bincount(present).argmax()) if len(present) else 1
        units = np.where(null, np.maximum(lengths, fill), lengths)
        first = np.repeat(np.cumsum(units) - units, units)
        within = np.arange(int(units.sum())) - first
        parent = np.repeat(np.arange(len(units)), units)
        source = offsets[:-1][parent] + within
        placeholder = null[parent] | (within >= lengths[parent])
        indices = pa.array(np.where(placeholder, 0, source), mask=placeholder)
        values = (
            array.values.take(indices)
            if len(array.values)
            else pa.nulls(len(indices), array.type.value_type)
        )
        null = placeholder | values.is_null().to_numpy(zero_copy_only=False)
        array = values
    return array, null


def _true_count(values: pa.Array, null: np.ndarray) -> int:
    flags = np.asarray(
        values.fill_null(False).to_numpy(zero_copy_only=False), dtype=bool
    )
    return int((flags & ~null).sum())


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def quantiles(values: np.ndarray) -> dict[str, float] | None:
    """Named quantiles of the finite values, or ``None`` when there are none."""
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        f"q{int(q * 100):02d}": float(v)
        for q, v in zip(QUANTILES, np.quantile(values, QUANTILES), strict=True)
    }


def _label_coverage(spec: LabelSpec, table: pa.Table) -> dict[str, Any]:
    if spec.row_filter is not None:
        column, value = spec.row_filter
        table = table.filter(pc.field(column) == value)
        rows = table.num_rows
        validity = next((c for c in VALIDITY_COLUMNS if c in spec.columns), None)
        valid = rows
        if validity and rows:
            valid = _true_count(*_leaves(table.column(validity)))
        return {
            "units": rows,
            "unit": "row",
            "coverage": 1.0 if rows else None,
            "valid": _share(valid, rows),
            "missing": _share(rows - valid, rows),
        }
    values, null = _leaves(table.column(spec.columns[0]))
    units = len(null)
    present = int((~null).sum())
    companion = next(
        (c for c in spec.columns[1:] if c.endswith(("valid", "valid_subframes"))), None
    )
    if companion is not None:
        valid = _true_count(*_leaves(table.column(companion)))
    elif pa.types.is_boolean(values.type) and spec.name.endswith("valid"):
        valid = _true_count(values, null)
    else:
        valid = present
    return {
        "units": units,
        "unit": "element" if units != table.num_rows else "row",
        "coverage": _share(present, units),
        "valid": _share(valid, units),
        "missing": _share(units - valid, units),
    }


def dataset_coverage(
    dataset_dir: Path, provides: frozenset[str], specs: Sequence[LabelSpec] = REGISTRY
) -> dict[str, Any]:
    """Coverage of every label for one dataset's store (reads only what it measures)."""
    manifests = {
        str(spec.extractor): store.read_manifest(dataset_dir, spec.extractor)
        for spec in specs
        if spec.extractor is not None
    }
    cache: dict[tuple[str, str, tuple[str, ...]], pa.Table] = {}
    labels: dict[str, Any] = {}
    for spec in specs:
        entry: dict[str, Any] = {
            "family": spec.family,
            "source_kind": str(spec.source_kind),
            "modalities": [str(m) for m in spec.modalities],
            "level": str(spec.level),
            "time_reference": spec.time_reference,
            "availability": str(spec.availability),
            "supported": spec.supported_by(provides),
            "materialized": False,
        }
        manifest = manifests.get(str(spec.extractor)) if spec.extractor else None
        if spec.availability is Availability.UNSUPPORTED:
            entry["reason"] = spec.unsupported_reason
        elif manifest is None:
            entry["reason"] = "extractor not built"
        elif spec.name not in manifest.get("materialized_labels", ()):
            entry["reason"] = manifest.get("unavailable_labels", {}).get(
                spec.name, "not built"
            )
        else:
            assert spec.table is not None and spec.extractor is not None
            columns = tuple(spec.columns) + (
                (spec.row_filter[0],) if spec.row_filter else ()
            )
            key = (str(spec.extractor), str(spec.table), columns)
            if key not in cache:
                cache[key] = pq.read_table(
                    dataset_dir / str(spec.extractor) / store.table_file(spec.table),
                    columns=list(dict.fromkeys(columns)),
                )
            entry["materialized"] = True
            entry.update(_label_coverage(spec, cache[key]))
        labels[spec.name] = entry
    return {
        "labels": labels,
        "sanity": sanity(dataset_dir),
        "per_modality": _per_modality(labels),
    }


def _per_modality(labels: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for modality in Modality:
        chosen = [e for e in labels.values() if str(modality) in e["modalities"]]
        result[str(modality)] = {
            "registered": len(chosen),
            "supported": sum(bool(e["supported"]) for e in chosen),
            "materialized": sum(bool(e["materialized"]) for e in chosen),
        }
    return result


def _read(
    dataset_dir: Path, extractor: str, table: Table, columns: list[str]
) -> pa.Table | None:
    path = dataset_dir / extractor / store.table_file(table)
    if not path.exists():
        return None
    available = set(pq.read_schema(path).names)
    if not set(columns) <= available:
        return None
    return pq.read_table(path, columns=columns)


def sanity(dataset_dir: Path) -> dict[str, Any]:
    """Headline statistics of the speech extractor (empty when it was not built)."""
    result: dict[str, Any] = {}
    grid = _read(
        dataset_dir,
        "speech",
        Table.GRID,
        [
            "ego_speaking",
            "others_active",
            "active_speaker_count_subframes",
            "time_to_next_ego_onset",
            "time_to_next_other_onset",
        ],
    )
    if grid is not None and grid.num_rows:
        leaves, null = _leaves(grid.column("active_speaker_count_subframes"))
        count = np.asarray(
            leaves.fill_null(0).to_numpy(zero_copy_only=False), dtype=float
        )
        known = ~null
        result["grid_rows"] = grid.num_rows
        result["ego_speaking_share_of_known_cells"] = _mean_known(
            grid.column("ego_speaking")
        )
        result["others_active_share_of_known_cells"] = _mean_known(
            grid.column("others_active")
        )
        result["overlap_share_of_known_subframes"] = (
            float((count[known] >= 2).mean()) if known.any() else None
        )
        result["silence_share_of_known_subframes"] = (
            float((count[known] == 0).mean()) if known.any() else None
        )
        for name in ("time_to_next_ego_onset", "time_to_next_other_onset"):
            values = grid.column(name).to_numpy(zero_copy_only=False).astype(float)
            result[f"{name}_s"] = quantiles(values)
            result[f"{name}_censored_share"] = float(np.isnan(values).mean())
    events = _read(
        dataset_dir,
        "speech",
        Table.EVENTS,
        ["event_type", "is_ego", "fto_s", "onset_type"],
    )
    if events is not None:
        frame = events.to_pandas()
        result["onset_count"] = int((frame.event_type == "onset").sum())
        result["offset_count"] = int((frame.event_type == "offset").sum())
        result["ego_onset_count"] = int(
            ((frame.event_type == "onset") & frame.is_ego).sum()
        )
        onset_types = frame.loc[
            frame.event_type == "onset", "onset_type"
        ].value_counts()
        result["onset_types"] = {str(k): int(v) for k, v in onset_types.items()}
        transfers = frame.loc[frame.event_type == "floor_change", "fto_s"].to_numpy(
            float
        )
        result["floor_transfer_count"] = len(transfers)
        result["floor_transfer_fto_s"] = quantiles(transfers)
        result["floor_transfer_overlap_share"] = (
            float((transfers < 0).mean()) if len(transfers) else None
        )
    segments = _read(
        dataset_dir,
        "speech",
        Table.SEGMENTS,
        ["segment_type", "duration_s", "overlap_type", "silence_type"],
    )
    if segments is not None:
        frame = segments.to_pandas()
        turns = frame.loc[frame.segment_type == "turn", "duration_s"].to_numpy(float)
        result["turn_count"] = len(turns)
        result["turn_duration_s"] = quantiles(turns)
        overlaps = frame.loc[
            frame.segment_type == "overlap", "overlap_type"
        ].value_counts()
        result["overlap_types"] = {str(k): int(v) for k, v in overlaps.items()}
        silences = frame.loc[
            frame.segment_type == "silence", "silence_type"
        ].value_counts()
        result["silence_types"] = {str(k): int(v) for k, v in silences.items()}
    return result


def _mean_known(column: pa.ChunkedArray) -> float | None:
    """Share of true values among the non-null ones."""
    values, null = _leaves(column)
    known = int((~null).sum())
    return _true_count(values, null) / known if known else None


def coverage_report(
    stores: Mapping[str, Path], provides: Mapping[str, frozenset[str]]
) -> dict[str, Any]:
    """Coverage of every dataset plus the dataset-independent label counts."""
    datasets = {
        name: dataset_coverage(path, provides.get(name, frozenset()))
        for name, path in sorted(stores.items())
    }
    counts: dict[str, int] = {}
    for spec in REGISTRY:
        key = str(spec.source_kind) if spec.extractor else "unsupported"
        counts[key] = counts.get(key, 0) + 1
    return {
        "label_count": len(REGISTRY),
        "counts_by_kind": counts,
        "datasets": datasets,
    }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f} %"


def coverage_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable rendering of :func:`coverage_report`."""
    names = list(report["datasets"])
    lines = [
        "# Label coverage report",
        "",
        "Generated by `conv-wm audit labels`. An audit: nothing here modifies a label.",
        "",
        f"{report['label_count']} registered labels: "
        + ", ".join(f"{k} {v}" for k, v in sorted(report["counts_by_kind"].items())),
        "",
        "| label | kind | modalities | "
        + " | ".join(
            f"{n} status | {n} coverage | {n} valid | {n} missing" for n in names
        )
        + " |",
        "| --- | --- | --- | "
        + " | ".join("--- | --- | --- | ---" for _ in names)
        + " |",
    ]
    for spec in REGISTRY:
        cells = []
        for name in names:
            entry = report["datasets"][name]["labels"][spec.name]
            status = (
                "built"
                if entry["materialized"]
                else ("supported, not built" if entry["supported"] else "unsupported")
            )
            cells += [
                status,
                _percent(entry.get("coverage")),
                _percent(entry.get("valid")),
                _percent(entry.get("missing")),
            ]
        lines.append(
            f"| `{spec.name}` | {spec.source_kind} | {', '.join(map(str, spec.modalities))} | "
            + " | ".join(cells)
            + " |"
        )
    for name in names:
        lines += ["", f"## Sanity statistics: {name}", "", "```json"]
        lines.append(
            json.dumps(report["datasets"][name]["sanity"], indent=2, sort_keys=True)
        )
        lines += [
            "```",
            "",
            f"Labels per modality: {report['datasets'][name]['per_modality']}",
        ]
    return "\n".join(lines) + "\n"


__all__ = [
    "coverage_markdown",
    "coverage_report",
    "dataset_coverage",
    "quantiles",
    "sanity",
]
