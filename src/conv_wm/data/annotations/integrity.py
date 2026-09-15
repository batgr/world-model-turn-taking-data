"""Generic integrity checks driven by annotation-source declarations.

Every check detects and classifies; nothing is corrected. Results are typed and
serializable, and each anomaly carries the severity the declaration assigned.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from conv_wm.data.annotations.spec import (
    AnnotationSourceSpec,
    CrossSourceComparison,
    DurationBounds,
    EntityReference,
    MediaReference,
    Severity,
    TemporalFields,
)
from conv_wm.data.validation import check_foreign_key

MEDIA_SOURCE = "media"


@dataclass(frozen=True)
class Anomaly:
    """One violated (or informational) constraint on a source."""

    constraint: str
    severity: Severity
    count: int
    total: int
    detail: str = ""

    @property
    def fraction(self) -> float:
        """``count / total`` (0 when the source is empty)."""
        return self.count / self.total if self.total else 0.0

    def to_row(self) -> dict[str, Any]:
        """Flatten into a table row."""
        row = asdict(self)
        row["severity"] = str(self.severity)
        row["fraction"] = self.fraction
        return row


@dataclass(frozen=True)
class TemporalIntegrityResult:
    """Outcome of the temporal checks of one source."""

    n_rows: int
    n_with_start: int
    n_with_end: int
    n_non_finite: int
    n_negative_start: int
    n_start_after_end: int
    n_zero_duration: int
    duration_quantiles: dict[str, float]
    """Quantiles of ``end - start`` in the source's own unit (intervals only)."""
    n_out_of_bounds: int | None
    """Rows ending beyond the referenced duration (``None`` without bounds)."""
    n_unbounded: int | None
    """Rows whose referenced duration could not be resolved."""
    max_overshoot: float | None
    """Largest excess beyond the referenced duration, in the source's unit."""

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping."""
        return asdict(self)


@dataclass(frozen=True)
class ReferenceIntegrityResult:
    """Outcome of one entity or media reference check."""

    kind: str
    columns: tuple[str, ...]
    target: str
    n_rows: int
    n_unknown: int
    """Rows whose key is a declared 'unknown' sentinel."""
    n_orphans: int
    """Rows with a concrete key that does not exist in the target."""
    severity: Severity

    @property
    def valid(self) -> bool:
        """No orphan rows."""
        return self.n_orphans == 0

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping."""
        document = asdict(self)
        document["severity"] = str(self.severity)
        document["valid"] = self.valid
        return document


@dataclass(frozen=True)
class SourceIntegrityReport:
    """Everything the audit established about one annotation source."""

    dataset: str
    name: str
    description: str
    scope: str
    dimensions: tuple[str, ...]
    provenance: str
    n_rows: int
    n_columns: int
    temporal_coordinates: dict[str, Any] | None
    media_relation: dict[str, Any] | None
    entity_relations: list[dict[str, Any]]
    value_fields: tuple[str, ...]
    confidence_field: str | None
    temporal: TemporalIntegrityResult | None
    references: list[ReferenceIntegrityResult]
    anomalies: list[Anomaly]
    known_limitations: tuple[str, ...]
    load_error: str | None = None

    @property
    def valid(self) -> bool:
        """No error-severity anomaly and no load failure."""
        return self.load_error is None and not any(
            anomaly.severity == Severity.ERROR and anomaly.count > 0
            for anomaly in self.anomalies
        )

    @property
    def downstream_suitability(self) -> str:
        """Coarse verdict: ``usable``, ``usable_with_caveats`` or ``blocked``."""
        if not self.valid:
            return "blocked"
        if (
            any(
                anomaly.severity == Severity.WARNING and anomaly.count > 0
                for anomaly in self.anomalies
            )
            or self.known_limitations
        ):
            return "usable_with_caveats"
        return "usable"

    def to_dict(self) -> dict[str, Any]:
        """JSON layout of the annotation contract for this source."""
        return {
            "dataset": self.dataset,
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "dimensions": list(self.dimensions),
            "provenance": self.provenance,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "temporal_coordinates": self.temporal_coordinates,
            "media_relation": self.media_relation,
            "entity_relations": self.entity_relations,
            "value_fields": list(self.value_fields),
            "confidence_field": self.confidence_field,
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "references": [reference.to_dict() for reference in self.references],
            "anomalies": [anomaly.to_row() for anomaly in self.anomalies],
            "known_limitations": list(self.known_limitations),
            "valid": self.valid,
            "downstream_suitability": self.downstream_suitability,
            "load_error": self.load_error,
        }

    def to_row(self) -> dict[str, Any]:
        """One-line summary row for the sources table."""
        return {
            "dataset": self.dataset,
            "source": self.name,
            "scope": self.scope,
            "dimensions": ",".join(self.dimensions),
            "provenance": self.provenance,
            "n_rows": self.n_rows,
            "n_error_anomalies": sum(
                1
                for a in self.anomalies
                if a.severity == Severity.ERROR and a.count > 0
            ),
            "n_warning_anomalies": sum(
                1
                for a in self.anomalies
                if a.severity == Severity.WARNING and a.count > 0
            ),
            "valid": self.valid,
            "downstream_suitability": self.downstream_suitability,
            "load_error": self.load_error,
        }


@dataclass(frozen=True)
class CrossSourceResult:
    """Exact interval identity and overlap agreement per shared entity key."""

    name: str
    left_source: str
    right_source: str
    n_left: int
    n_right: int
    left_exact_identity_fraction: float
    """Fraction of left intervals exactly present in the right source."""
    right_exact_identity_fraction: float
    n_left_exact: int
    n_right_exact: int
    left_covered_fraction: float
    """Fraction of left intervals overlapping at least one right interval."""
    right_covered_fraction: float
    n_left_uncovered: int
    n_right_uncovered: int
    n_shared_keys: int
    n_left_only_keys: int
    n_right_only_keys: int
    overlap_tolerance: float
    description: str = ""

    def to_row(self) -> dict[str, Any]:
        """Flatten into a table row."""
        return asdict(self)


@dataclass
class _AnomalyCollector:
    anomalies: list[Anomaly] = field(default_factory=list)

    def add(
        self,
        constraint: str,
        severity: Severity,
        count: int,
        total: int,
        detail: str = "",
    ) -> None:
        self.anomalies.append(Anomaly(constraint, severity, count, total, detail))


def _numeric(table: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(table[column], errors="coerce")


def _join_keys(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> pd.Series:
    """Composite string key per row (empty frames yield an empty string Series)."""
    if frame.empty:
        return pd.Series(dtype=str, index=frame.index)
    return frame[list(columns)].astype(str).agg("/".join, axis=1)


def audit_temporal(
    spec: TemporalFields,
    table: pd.DataFrame,
    *,
    reference_durations: pd.Series | None,
    collector: _AnomalyCollector,
) -> TemporalIntegrityResult:
    """Check finiteness, ordering, duration and bounds of the temporal columns.

    ``reference_durations`` is aligned with ``table`` (NaN where unresolved) and
    expressed in the source's own unit.
    """
    n = len(table)
    start = _numeric(table, spec.start)
    end = _numeric(table, spec.end) if spec.end else None
    with_start = start.notna()
    with_end = end.notna() if end is not None else pd.Series(False, index=table.index)
    raw_start = table[spec.start]
    non_finite = int((raw_start.notna() & ~np.isfinite(start.fillna(0))).sum())
    if end is not None:
        non_finite += int((table[spec.end].notna() & ~np.isfinite(end.fillna(0))).sum())  # type: ignore[index]
    negative = int((start < 0).sum())
    n_missing = int((~with_start).sum() + ((~with_end).sum() if end is not None else 0))
    collector.add(
        "missing_timestamps",
        Severity.INFO if spec.nullable else Severity.WARNING,
        n_missing,
        n * (2 if end is not None else 1),
        "declared nullable" if spec.nullable else "",
    )
    collector.add("non_finite_timestamps", Severity.ERROR, non_finite, n)
    collector.add(
        "negative_start",
        Severity.WARNING,
        negative,
        n,
        "negative is not automatically invalid; check the declared origin",
    )
    start_after_end = 0
    zero_duration = 0
    quantiles: dict[str, float] = {}
    if end is not None:
        both = with_start & with_end
        durations = (end - start)[both]
        start_after_end = int((durations < 0).sum())
        zero_duration = int((durations == 0).sum())
        if len(durations):
            quantiles = {
                name: float(durations.quantile(q))
                for name, q in (
                    ("min", 0.0),
                    ("p05", 0.05),
                    ("median", 0.5),
                    ("p95", 0.95),
                    ("max", 1.0),
                )
            }
        collector.add(
            "start_after_end", Severity.ERROR, start_after_end, int(both.sum())
        )
        collector.add("zero_duration", Severity.INFO, zero_duration, int(both.sum()))

    n_out_of_bounds: int | None = None
    n_unbounded: int | None = None
    max_overshoot: float | None = None
    if reference_durations is not None:
        last = end if end is not None else start
        resolvable = last.notna()
        unresolved = resolvable & reference_durations.isna()
        overshoot = (last - reference_durations)[
            resolvable & reference_durations.notna()
        ]
        out = overshoot > spec.out_of_bounds_tolerance
        n_out_of_bounds = int(out.sum())
        n_unbounded = int(unresolved.sum())
        max_overshoot = float(overshoot.max()) if len(overshoot) else None
        collector.add(
            "end_beyond_reference_duration",
            Severity.WARNING,
            n_out_of_bounds,
            int(resolvable.sum()),
            f"tolerance {spec.out_of_bounds_tolerance} {spec.unit}",
        )
        collector.add(
            "unresolved_reference_duration",
            Severity.WARNING,
            n_unbounded,
            int(resolvable.sum()),
        )
    return TemporalIntegrityResult(
        n_rows=n,
        n_with_start=int(with_start.sum()),
        n_with_end=int(with_end.sum()),
        n_non_finite=non_finite,
        n_negative_start=negative,
        n_start_after_end=start_after_end,
        n_zero_duration=zero_duration,
        duration_quantiles=quantiles,
        n_out_of_bounds=n_out_of_bounds,
        n_unbounded=n_unbounded,
        max_overshoot=max_overshoot,
    )


def audit_entity_reference(
    reference: EntityReference,
    table: pd.DataFrame,
    target: pd.DataFrame,
    *,
    collector: _AnomalyCollector,
) -> ReferenceIntegrityResult:
    """Count rows whose key is absent from the target, ignoring 'unknown' sentinels."""
    columns = list(reference.columns)
    keys = table[columns].astype(str)
    unknown = pd.Series(False, index=table.index)
    for column in columns:
        unknown |= keys[column].isin(reference.unknown_values)
    concrete = table.loc[~unknown]
    orphans = check_foreign_key(
        concrete, columns, target, list(reference.target_columns)
    )
    collector.add(
        f"dangling_{reference.kind}_reference:{','.join(columns)}",
        reference.severity,
        orphans,
        len(concrete),
        f"target {reference.target_source}",
    )
    return ReferenceIntegrityResult(
        kind=str(reference.kind),
        columns=reference.columns,
        target=reference.target_source,
        n_rows=len(table),
        n_unknown=int(unknown.sum()),
        n_orphans=orphans,
        severity=reference.severity,
    )


def audit_media_reference(
    reference: MediaReference,
    table: pd.DataFrame,
    media: pd.DataFrame,
    *,
    collector: _AnomalyCollector,
) -> ReferenceIntegrityResult:
    """Count rows naming a media file that the media metadata table does not list."""
    media_keys = media["relative_path"].map(reference.media_key).astype(str)
    target = pd.DataFrame({"media_key": media_keys.unique()})
    child = pd.DataFrame({"media_key": _join_keys(table, reference.columns)})
    orphans = check_foreign_key(child, ["media_key"], target, ["media_key"])
    collector.add(
        f"dangling_media_reference:{','.join(reference.columns)}",
        reference.severity,
        orphans,
        len(table),
        "target media metadata table",
    )
    return ReferenceIntegrityResult(
        kind="media",
        columns=reference.columns,
        target=MEDIA_SOURCE,
        n_rows=len(table),
        n_unknown=0,
        n_orphans=orphans,
        severity=reference.severity,
    )


def resolve_reference_durations(
    bounds: DurationBounds,
    table: pd.DataFrame,
    *,
    tables: Mapping[str, pd.DataFrame],
    media: pd.DataFrame | None,
    media_key: MediaReference | None,
) -> pd.Series:
    """Per-row reference duration from the declared bounds source (NaN if unresolved)."""
    if bounds.source == MEDIA_SOURCE:
        if media is None or media_key is None:
            return pd.Series(np.nan, index=table.index)
        durations = media[["audio_duration_sec", "video_duration_sec"]].min(axis=1)
        target = pd.DataFrame(
            {
                "_key": media["relative_path"].map(media_key.media_key).astype(str),
                "_duration": durations.to_numpy(),
            }
        ).drop_duplicates("_key")
        keys = _join_keys(table, media_key.columns)
    else:
        source = tables[bounds.source]
        if bounds.duration_column is not None:
            duration = pd.to_numeric(source[bounds.duration_column], errors="coerce")
        else:
            duration = pd.to_numeric(
                source[bounds.end_column], errors="coerce"
            ) - pd.to_numeric(  # type: ignore[index]
                source[bounds.start_column],
                errors="coerce",  # type: ignore[index]
            )
        target = pd.DataFrame(
            {
                "_key": source[list(bounds.target_key_columns)]
                .astype(str)
                .agg("/".join, axis=1),
                "_duration": duration.to_numpy(),
            }
        ).drop_duplicates("_key")
        keys = _join_keys(table, bounds.key_columns)
    lookup = target.set_index("_key")["_duration"]
    return pd.Series(keys.map(lookup).to_numpy(dtype=float), index=table.index)


def audit_source(
    spec: AnnotationSourceSpec,
    table: pd.DataFrame,
    *,
    tables: Mapping[str, pd.DataFrame],
    media: pd.DataFrame | None,
) -> SourceIntegrityReport:
    """Run every declared check on one loaded source."""
    collector = _AnomalyCollector()
    missing_columns = _declared_columns(spec) - set(table.columns)
    collector.add(
        "missing_declared_columns",
        Severity.ERROR,
        len(missing_columns),
        len(_declared_columns(spec)),
        ",".join(sorted(missing_columns)),
    )
    if spec.identity and not (set(spec.identity) - set(table.columns)):
        duplicates = int(table.duplicated(list(spec.identity), keep=False).sum())
        collector.add("duplicate_identity", Severity.ERROR, duplicates, len(table))

    references: list[ReferenceIntegrityResult] = []
    if (
        spec.media_reference
        and media is not None
        and not (set(spec.media_reference.columns) - set(table.columns))
    ):
        references.append(
            audit_media_reference(
                spec.media_reference, table, media, collector=collector
            )
        )
    for reference in spec.entity_references:
        if (
            set(reference.columns) - set(table.columns)
            or reference.target_source not in tables
        ):
            continue
        references.append(
            audit_entity_reference(
                reference, table, tables[reference.target_source], collector=collector
            )
        )

    temporal: TemporalIntegrityResult | None = None
    if spec.temporal and not (_temporal_columns(spec.temporal) - set(table.columns)):
        durations = (
            resolve_reference_durations(
                spec.bounds,
                table,
                tables=tables,
                media=media,
                media_key=spec.media_reference,
            )
            if spec.bounds is not None
            else None
        )
        temporal = audit_temporal(
            spec.temporal, table, reference_durations=durations, collector=collector
        )
    return SourceIntegrityReport(
        dataset=spec.dataset,
        name=spec.name,
        description=spec.description,
        scope=str(spec.scope),
        dimensions=spec.dimensions,
        provenance=str(spec.provenance),
        n_rows=len(table),
        n_columns=len(table.columns),
        temporal_coordinates=_temporal_coordinates(spec),
        media_relation=_media_relation(spec),
        entity_relations=[
            {
                "kind": str(reference.kind),
                "columns": list(reference.columns),
                "target": reference.target_source,
                "target_columns": list(reference.target_columns),
                "unknown_values": list(reference.unknown_values),
            }
            for reference in spec.entity_references
        ],
        value_fields=spec.value_fields,
        confidence_field=spec.confidence_field,
        temporal=temporal,
        references=references,
        anomalies=collector.anomalies,
        known_limitations=spec.known_limitations,
    )


def failed_source(
    spec: AnnotationSourceSpec, error: Exception
) -> SourceIntegrityReport:
    """Report for a source that could not be loaded."""
    return SourceIntegrityReport(
        dataset=spec.dataset,
        name=spec.name,
        description=spec.description,
        scope=str(spec.scope),
        dimensions=spec.dimensions,
        provenance=str(spec.provenance),
        n_rows=0,
        n_columns=0,
        temporal_coordinates=_temporal_coordinates(spec),
        media_relation=_media_relation(spec),
        entity_relations=[],
        value_fields=spec.value_fields,
        confidence_field=spec.confidence_field,
        temporal=None,
        references=[],
        anomalies=[],
        known_limitations=spec.known_limitations,
        load_error=f"{type(error).__name__}: {error}",
    )


def _declared_columns(spec: AnnotationSourceSpec) -> set[str]:
    columns: set[str] = set(spec.identity) | set(spec.value_fields)
    if spec.temporal:
        columns |= _temporal_columns(spec.temporal)
    if spec.media_reference:
        columns |= set(spec.media_reference.columns)
    for reference in spec.entity_references:
        columns |= set(reference.columns)
    if spec.bounds:
        columns |= set(spec.bounds.key_columns)
    if spec.confidence_field:
        columns.add(spec.confidence_field)
    return columns


def _temporal_columns(temporal: TemporalFields) -> set[str]:
    return {temporal.start} | ({temporal.end} if temporal.end else set())


def _temporal_coordinates(spec: AnnotationSourceSpec) -> dict[str, Any] | None:
    if spec.temporal is None:
        return None
    return {
        "start": spec.temporal.start,
        "end": spec.temporal.end,
        "unit": str(spec.temporal.unit),
        "origin": str(spec.temporal.origin),
        "nullable": spec.temporal.nullable,
        "bounds": None
        if spec.bounds is None
        else {
            "source": spec.bounds.source,
            "key_columns": list(spec.bounds.key_columns),
            "duration_column": spec.bounds.duration_column,
            "start_column": spec.bounds.start_column,
            "end_column": spec.bounds.end_column,
        },
    }


def _media_relation(spec: AnnotationSourceSpec) -> dict[str, Any] | None:
    if spec.media_reference is None:
        return None
    return {
        "columns": list(spec.media_reference.columns),
        "target": "media metadata table",
    }


def _interval_frame(
    table: pd.DataFrame,
    spec: AnnotationSourceSpec,
    key_columns: tuple[str, ...],
    *,
    required_non_null: tuple[str, ...] = (),
    excluded_key_values: tuple[str, ...] = (),
) -> pd.DataFrame:
    assert spec.temporal is not None and spec.temporal.end is not None
    selected = pd.Series(True, index=table.index)
    for column in (*key_columns, *required_non_null):
        selected &= table[column].notna()
    if excluded_key_values:
        for column in key_columns:
            selected &= ~table[column].astype(str).isin(excluded_key_values)
    table = table.loc[selected]
    frame = pd.DataFrame(
        {
            "_key": _join_keys(table, key_columns),
            "_start": _numeric(table, spec.temporal.start),
            "_end": _numeric(table, spec.temporal.end),
        }
    )
    return frame.dropna().reset_index(drop=True)


def _covered_mask(
    left: pd.DataFrame, right: pd.DataFrame, *, tolerance: float
) -> np.ndarray:
    """For each left interval, whether any right interval with the same key overlaps it."""
    covered = np.zeros(len(left), dtype=bool)
    right_groups = {key: group for key, group in right.groupby("_key")}
    for key, group in left.groupby("_key"):
        other = right_groups.get(key)
        if other is None:
            continue
        starts = other["_start"].to_numpy()
        ends = other["_end"].to_numpy()
        for index, (start, end) in zip(
            group.index, zip(group["_start"], group["_end"], strict=True), strict=True
        ):
            covered[index] = bool(
                np.any((starts < end + tolerance) & (ends > start - tolerance))
            )
    return covered


def _exact_mask(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    """For each left interval, whether its key and bounds occur exactly on the right."""
    right_intervals = set(
        right.loc[:, ["_key", "_start", "_end"]].itertuples(index=False, name=None)
    )
    return np.fromiter(
        (
            interval in right_intervals
            for interval in left.loc[:, ["_key", "_start", "_end"]].itertuples(
                index=False, name=None
            )
        ),
        dtype=bool,
        count=len(left),
    )


def compare_sources(
    comparison: CrossSourceComparison,
    left_spec: AnnotationSourceSpec,
    right_spec: AnnotationSourceSpec,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> CrossSourceResult:
    """Exact-identity and interval-overlap coverage in both directions."""
    left_frame = _interval_frame(
        left,
        left_spec,
        comparison.left_key_columns,
        required_non_null=comparison.left_required_non_null,
        excluded_key_values=comparison.excluded_key_values,
    )
    right_frame = _interval_frame(
        right,
        right_spec,
        comparison.right_key_columns,
        required_non_null=comparison.right_required_non_null,
        excluded_key_values=comparison.excluded_key_values,
    )
    left_exact = _exact_mask(left_frame, right_frame)
    right_exact = _exact_mask(right_frame, left_frame)
    left_covered = _covered_mask(
        left_frame, right_frame, tolerance=comparison.overlap_tolerance
    )
    right_covered = _covered_mask(
        right_frame, left_frame, tolerance=comparison.overlap_tolerance
    )
    left_keys = set(left_frame["_key"])
    right_keys = set(right_frame["_key"])
    return CrossSourceResult(
        name=comparison.name,
        left_source=comparison.left_source,
        right_source=comparison.right_source,
        n_left=len(left_frame),
        n_right=len(right_frame),
        left_exact_identity_fraction=float(left_exact.mean())
        if len(left_frame)
        else math.nan,
        right_exact_identity_fraction=float(right_exact.mean())
        if len(right_frame)
        else math.nan,
        n_left_exact=int(left_exact.sum()),
        n_right_exact=int(right_exact.sum()),
        left_covered_fraction=float(left_covered.mean())
        if len(left_frame)
        else math.nan,
        right_covered_fraction=float(right_covered.mean())
        if len(right_frame)
        else math.nan,
        n_left_uncovered=int((~left_covered).sum()),
        n_right_uncovered=int((~right_covered).sum()),
        n_shared_keys=len(left_keys & right_keys),
        n_left_only_keys=len(left_keys - right_keys),
        n_right_only_keys=len(right_keys - left_keys),
        overlap_tolerance=comparison.overlap_tolerance,
        description=comparison.description,
    )
