"""Annotation integrity audit over every registered dataset.

For each declared annotation source: load it, run the generic temporal,
identity, media-reference and entity-reference checks, then compare the
sources a dataset declared comparable. Writes a machine-readable contract per
source plus flat tables of sources, anomalies and cross-source agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data import datasets
from conv_wm.data.annotations import (
    CrossSourceResult,
    SourceIntegrityReport,
    audit_source,
    compare_sources,
    failed_source,
)
from conv_wm.data.audits.errors import MissingPrerequisiteError
from conv_wm.data.audits.media_metadata import MEDIA_METADATA_TABLE
from conv_wm.data.datasets.spec import DatasetSpec
from conv_wm.reports import JsonDict, write_summary, write_table

OUTPUT_DIR = Path("annotations")


@dataclass(frozen=True)
class DatasetAnnotationResult:
    """Integrity outcome of one dataset."""

    dataset: str
    sources: list[SourceIntegrityReport]
    comparisons: list[CrossSourceResult]

    @property
    def valid(self) -> bool:
        """Every source is free of error-severity anomalies."""
        return all(source.valid for source in self.sources)

    def to_dict(self) -> JsonDict:
        """JSON layout: contracts keyed by source name plus comparisons."""
        return {
            "valid": self.valid,
            "sources": {source.name: source.to_dict() for source in self.sources},
            "cross_source": [comparison.to_row() for comparison in self.comparisons],
        }


@dataclass(frozen=True)
class AnnotationOutputs:
    """Artifacts written by one run."""

    results: list[DatasetAnnotationResult]
    summary: JsonDict
    output_dir: Path

    @property
    def valid(self) -> bool:
        """Whether every audited dataset passed."""
        return bool(self.summary["valid"])

    @property
    def summary_path(self) -> Path:
        """Location of the annotation integrity contract."""
        return self.output_dir / "annotation_integrity.json"


def audit_dataset_annotations(
    spec: DatasetSpec,
    tables: dict[str, pd.DataFrame],
    *,
    media: pd.DataFrame | None,
    load_errors: dict[str, Exception] | None = None,
) -> DatasetAnnotationResult:
    """Run the declared checks of ``spec`` on already loaded ``tables``."""
    load_errors = load_errors or {}
    dataset_media = (
        media.loc[media["dataset"].eq(spec.name)] if media is not None else None
    )
    reports: list[SourceIntegrityReport] = []
    for source in spec.annotations.sources:
        if source.name in load_errors:
            reports.append(failed_source(source, load_errors[source.name]))
            continue
        reports.append(
            audit_source(
                source, tables[source.name], tables=tables, media=dataset_media
            )
        )
    comparisons = [
        compare_sources(
            comparison,
            spec.annotations.source(comparison.left_source),
            spec.annotations.source(comparison.right_source),
            tables[comparison.left_source],
            tables[comparison.right_source],
        )
        for comparison in spec.annotations.comparisons
        if comparison.left_source in tables and comparison.right_source in tables
    ]
    return DatasetAnnotationResult(spec.name, reports, comparisons)


def load_annotation_tables(
    spec: DatasetSpec, cfg: DictConfig
) -> tuple[dict[str, pd.DataFrame], dict[str, Exception]]:
    """Load every declared source; failures are returned, not raised."""
    tables: dict[str, pd.DataFrame] = {}
    errors: dict[str, Exception] = {}
    for source in spec.annotations.sources:
        try:
            tables[source.name] = source.load(cfg)
        except Exception as exc:  # noqa: BLE001 - one unreadable source must not abort the audit
            errors[source.name] = exc
    return tables, errors


def build_annotation_summary(results: list[DatasetAnnotationResult]) -> JsonDict:
    """Combine per-dataset results into the contract document."""
    return {
        "audit_type": "annotation_integrity",
        "valid": all(result.valid for result in results),
        "datasets": {result.dataset: result.to_dict() for result in results},
        "interpretation": {
            "correction_applied": False,
            "severity_semantics": {
                "error": "blocks downstream use until resolved",
                "warning": "usable with caveats; documented in the contract",
                "info": "expected property, recorded for completeness",
            },
            "cross_source_agreement": (
                "interval overlap per shared entity; disagreement is reported, not judged"
            ),
        },
    }


def _flat_tables(results: list[DatasetAnnotationResult]) -> dict[str, pd.DataFrame]:
    sources = [source.to_row() for result in results for source in result.sources]
    anomalies = [
        {"dataset": source.dataset, "source": source.name, **anomaly.to_row()}
        for result in results
        for source in result.sources
        for anomaly in source.anomalies
    ]
    comparisons = [
        {"dataset": result.dataset, **comparison.to_row()}
        for result in results
        for comparison in result.comparisons
    ]
    return {
        "annotation_sources": pd.DataFrame.from_records(sources),
        "annotation_anomalies": pd.DataFrame.from_records(anomalies),
        "annotation_cross_source": pd.DataFrame.from_records(comparisons),
    }


def run_annotation_audit(
    cfg: DictConfig, *, dataset_names: list[str] | None = None
) -> AnnotationOutputs:
    """Audit every registered dataset that declares annotation sources."""
    paths = pipeline_paths(cfg)
    media_path = paths.reports / MEDIA_METADATA_TABLE
    if not media_path.exists():
        raise MissingPrerequisiteError(media_path, produce_with="conv-wm audit media")
    media = pd.read_parquet(media_path)
    names = dataset_names if dataset_names is not None else datasets.names()
    results = []
    for name in names:
        spec = datasets.get(name)
        if not spec.annotations.sources:
            continue
        tables, errors = load_annotation_tables(spec, cfg)
        results.append(
            audit_dataset_annotations(spec, tables, media=media, load_errors=errors)
        )
    summary = build_annotation_summary(results)
    output_dir = paths.reports / OUTPUT_DIR
    for name, table in _flat_tables(results).items():
        write_table(output_dir / f"{name}.parquet", table)
    outputs = AnnotationOutputs(results, summary, output_dir)
    write_summary(outputs.summary_path, summary)
    return outputs


def format_annotation_summary(outputs: AnnotationOutputs) -> str:
    """Human-readable per-source verdicts for the console."""
    lines: list[str] = []
    for result in outputs.results:
        lines.append(result.dataset)
        for source in result.sources:
            errors = [
                a for a in source.anomalies if a.severity == "error" and a.count > 0
            ]
            warnings = [
                a for a in source.anomalies if a.severity == "warning" and a.count > 0
            ]
            status = "FAIL" if not source.valid else "PASS"
            detail = f"{len(errors)} errors, {len(warnings)} warnings"
            if source.load_error:
                detail = source.load_error
            lines.append(
                f"  {source.name} [{source.scope}] rows={source.n_rows}: {status} "
                f"({detail}) -> {source.downstream_suitability}"
            )
        for comparison in result.comparisons:
            lines.append(
                f"  {comparison.name}: {comparison.left_source} covered "
                f"{comparison.left_covered_fraction:.1%}, {comparison.right_source} covered "
                f"{comparison.right_covered_fraction:.1%}"
            )
    lines.append(f"overall: {'PASS' if outputs.valid else 'FAIL'}")
    lines.append(f"report: {outputs.summary_path}")
    return "\n".join(lines)
