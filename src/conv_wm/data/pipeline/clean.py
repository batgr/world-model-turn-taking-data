"""Generic orchestration and accounting for targeted annotation cleaning.

Cleaning is deliberately separate from integrity auditing: dataset rules validate
and transform source tables, while this module checks their accounting and writes
derived interim tables plus one provenance-stamped report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import get_path, pipeline_paths
from conv_wm.reports import JsonDict, write_summary, write_table

CleaningDecision = Literal[
    "KEEP CURRENT FILTER",
    "CHANGE FILTER",
    "NO FILTER NEEDED",
    "UNRESOLVED",
]

REPORT_PATH = Path("cleaning") / "annotations" / "summary.json"


@dataclass(frozen=True)
class CleanedAnnotationTable:
    """One derived table and complete row-level transformation accounting."""

    dataset: str
    name: str
    output_key: str
    source_rows: int
    table: pd.DataFrame
    decision: CleaningDecision
    reason: str
    required_fields: tuple[str, ...]
    optional_nullable_fields: tuple[str, ...]
    removed_by_reason: Mapping[str, int]
    cleaning_rule: str
    source_paths: tuple[Path, ...]
    transformations: tuple[str, ...] = ()
    statistics: Mapping[str, object] | None = None

    @property
    def output_rows(self) -> int:
        """Number of rows in the derived table."""
        return len(self.table)

    @property
    def rows_removed(self) -> int:
        """Source rows not present in the derived table."""
        return self.source_rows - self.output_rows

    def validate_accounting(self) -> None:
        """Require every removed row to have exactly one declared reason."""
        if self.rows_removed < 0:
            raise ValueError(f"{self.dataset}/{self.name}: output exceeds source rows")
        accounted = sum(self.removed_by_reason.values())
        if accounted != self.rows_removed:
            raise ValueError(
                f"{self.dataset}/{self.name}: {self.rows_removed} rows removed but "
                f"{accounted} accounted for"
            )
        missing = set(self.required_fields) - set(self.table.columns)
        if missing:
            raise ValueError(
                f"{self.dataset}/{self.name}: required output columns missing: "
                f"{sorted(missing)}"
            )
        residual: dict[str, int] = {}
        for column in self.required_fields:
            count = int(self.table[column].isna().sum())
            if count:
                residual[column] = count
        if residual:
            raise ValueError(
                f"{self.dataset}/{self.name}: required values remain null: {residual}"
            )

    def to_summary(self, output_path: Path) -> JsonDict:
        """Machine-readable account for this table."""
        optional_counts = {
            column: int(self.table[column].isna().sum())
            for column in self.optional_nullable_fields
        }
        optional_mask = pd.Series(False, index=self.table.index)
        for column in self.optional_nullable_fields:
            optional_mask |= self.table[column].isna()
        optional_rows = int(optional_mask.sum())
        return {
            "decision": self.decision,
            "reason": self.reason,
            "source_paths": [str(path) for path in self.source_paths],
            "output_path": str(output_path),
            "source_rows": self.source_rows,
            "output_rows": self.output_rows,
            "rows_removed": self.rows_removed,
            "rows_retained_with_nullable_optional_fields": optional_rows,
            "nullable_optional_field_counts": optional_counts,
            "removed_by_reason": dict(self.removed_by_reason),
            "required_fields": list(self.required_fields),
            "optional_nullable_fields": list(self.optional_nullable_fields),
            "cleaning_rule": self.cleaning_rule,
            "transformations": list(self.transformations),
            "statistics": dict(self.statistics or {}),
        }


@dataclass(frozen=True)
class AnnotationCleaningOutputs:
    """Artifacts produced by a complete annotation-cleaning run."""

    tables: tuple[CleanedAnnotationTable, ...]
    summary: JsonDict
    summary_path: Path


def require_source_columns(
    table: pd.DataFrame, columns: Sequence[str], *, dataset: str, name: str
) -> None:
    """Fail before transformation when a declared source column is absent."""
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{dataset}/{name}: source columns missing: {sorted(missing)}")


def remove_rows_missing_required_fields(
    table: pd.DataFrame, required_fields: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove unusable rows, assigning each row to its first missing required field.

    The ordered, field-specific mask avoids a global null policy: nulls in every
    column not named in ``required_fields`` are preserved.
    """
    keep = pd.Series(True, index=table.index)
    reasons: dict[str, int] = {}
    for column in required_fields:
        missing = keep & table[column].isna()
        count = int(missing.sum())
        if count:
            reasons[f"missing_required_field:{column}"] = count
            keep &= ~missing
    return table.loc[keep].reset_index(drop=True), reasons


def run_annotation_cleaning(
    cfg: DictConfig, *, dataset_names: list[str] | None = None
) -> AnnotationCleaningOutputs:
    """Run registered dataset cleaners and write interim tables and their report."""
    from conv_wm.data import datasets

    names = dataset_names if dataset_names is not None else datasets.names()
    results: list[CleanedAnnotationTable] = []
    for name in names:
        spec = datasets.get(name)
        if spec.annotation_cleaner is None:
            continue
        results.extend(spec.annotation_cleaner(cfg))

    output_paths: dict[tuple[str, str], Path] = {}
    for result in results:
        result.validate_accounting()
        output = get_path(cfg, result.dataset, "interim", result.output_key)
        write_table(output, result.table)
        output_paths[(result.dataset, result.name)] = output

    payload: JsonDict = {
        "cleaning_type": "annotation_cleaning",
        "datasets": {
            dataset: {
                "tables": {
                    result.name: result.to_summary(
                        output_paths[(result.dataset, result.name)]
                    )
                    for result in results
                    if result.dataset == dataset
                }
            }
            for dataset in sorted({result.dataset for result in results})
        },
    }
    summary_path = pipeline_paths(cfg).reports / REPORT_PATH
    summary = write_summary(
        summary_path,
        payload,
        parameters={"datasets": names},
    )
    return AnnotationCleaningOutputs(tuple(results), summary, summary_path)


def format_annotation_cleaning_summary(outputs: AnnotationCleaningOutputs) -> str:
    """Format table counts and the report path for the command line."""
    lines = [
        f"{table.dataset}/{table.name}: {table.source_rows} -> {table.output_rows} "
        f"({table.rows_removed} removed)"
        for table in outputs.tables
    ]
    lines.append(f"report: {outputs.summary_path}")
    return "\n".join(lines)
