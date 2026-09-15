"""Structural validation: Pandera contracts and relations declared by each dataset."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from conv_wm.config import pipeline_paths
from conv_wm.data import datasets
from conv_wm.data.audits.pandera import audit_dataframe
from conv_wm.data.datasets.spec import DatasetSpec, StructuralSpec
from conv_wm.data.validation import check_foreign_key
from conv_wm.reports import JsonDict, write_summary

REPORT_PATH = Path("structural") / "structural_audit.json"


@dataclass(frozen=True)
class TableAuditResult:
    """Outcome of validating one table against its Pandera contract."""

    name: str
    valid: bool
    shape: tuple[int, int]
    n_failures: int
    failure_cases: list[dict[str, Any]]


@dataclass(frozen=True)
class RelationAuditResult:
    """Outcome of checking one declared relation."""

    name: str
    valid: bool
    n_failures: int
    orphan_rows: int
    """Child rows whose key is absent from the parent table."""


@dataclass(frozen=True)
class DatasetStructureResult:
    """Structural verdict of one dataset."""

    dataset: str
    valid: bool
    tables: list[TableAuditResult]
    relations: list[RelationAuditResult]

    def to_dict(self) -> JsonDict:
        """JSON layout: ``tables`` and ``relations`` keyed by name."""
        return {
            "valid": self.valid,
            "tables": {
                table.name: _without_name(asdict(table)) for table in self.tables
            },
            "relations": {
                relation.name: _without_name(asdict(relation))
                for relation in self.relations
            },
        }


def _without_name(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key != "name"}


def audit_dataset_structure(
    spec: DatasetSpec, tables: Mapping[str, pd.DataFrame]
) -> DatasetStructureResult:
    """Validate every declared table and relation of ``spec`` on loaded ``tables``."""
    structure = spec.structure
    if structure is None:
        return DatasetStructureResult(spec.name, True, [], [])
    missing = {table.name for table in structure.tables} - set(tables)
    if missing:
        raise KeyError(f"{spec.name}: tables not loaded: {sorted(missing)}")
    table_results = []
    for table in structure.tables:
        outcome = audit_dataframe(tables[table.name], table.schema)
        table_results.append(
            TableAuditResult(
                name=table.name,
                valid=outcome.valid,
                shape=outcome.shape,
                n_failures=outcome.n_failures,
                failure_cases=outcome.failure_cases,
            )
        )
    relation_results = []
    for relation in structure.relations:
        orphans = check_foreign_key(
            tables[relation.child_table],
            list(relation.child_columns),
            tables[relation.parent_table],
            list(relation.parent_columns),
        )
        relation_results.append(
            RelationAuditResult(
                name=relation.name,
                valid=orphans == 0,
                n_failures=orphans,
                orphan_rows=orphans,
            )
        )
    return DatasetStructureResult(
        dataset=spec.name,
        valid=all(t.valid for t in table_results)
        and all(r.valid for r in relation_results),
        tables=table_results,
        relations=relation_results,
    )


def load_structural_tables(
    structure: StructuralSpec, cfg: DictConfig
) -> dict[str, pd.DataFrame]:
    """Load every table declared by ``structure`` from the configured paths."""
    return {table.name: table.load(cfg) for table in structure.tables}


@dataclass(frozen=True)
class StructuralOutputs:
    """Report written by one run."""

    results: list[DatasetStructureResult]
    report: JsonDict
    report_path: Path

    @property
    def valid(self) -> bool:
        """Whether every audited dataset passed."""
        return bool(self.report["valid"])


def build_structural_report(results: list[DatasetStructureResult]) -> JsonDict:
    """Combine per-dataset results into the report layout."""
    return {
        "audit_type": "structural_validation",
        "valid": all(result.valid for result in results),
        "datasets": {result.dataset: result.to_dict() for result in results},
    }


def run_structural_audit(
    cfg: DictConfig, *, dataset_names: list[str] | None = None
) -> StructuralOutputs:
    """Validate every registered dataset that declares structure and write the report."""
    names = dataset_names if dataset_names is not None else datasets.names()
    results = []
    for name in names:
        spec = datasets.get(name)
        if spec.structure is None:
            continue
        results.append(
            audit_dataset_structure(spec, load_structural_tables(spec.structure, cfg))
        )
    report = build_structural_report(results)
    report_path = pipeline_paths(cfg).reports / REPORT_PATH
    write_summary(report_path, report)
    return StructuralOutputs(results, report, report_path)


def format_structural_summary(outputs: StructuralOutputs) -> str:
    """Human-readable PASS/FAIL lines for the console."""
    lines = []
    for result in outputs.results:
        lines.append(result.dataset)
        for table in result.tables:
            status = "PASS" if table.valid else "FAIL"
            lines.append(
                f"  table {table.name}: {status} ({table.n_failures} failures)"
            )
        for relation in result.relations:
            status = "PASS" if relation.valid else "FAIL"
            lines.append(
                f"  relation {relation.name}: {status} ({relation.n_failures} failures)"
            )
    lines.append(f"overall: {'PASS' if outputs.valid else 'FAIL'}")
    lines.append(f"report: {outputs.report_path}")
    return "\n".join(lines)
