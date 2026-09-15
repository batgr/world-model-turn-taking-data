"""Typed description of what a dataset contributes to the generic pipeline.

A :class:`DatasetSpec` is the single place where dataset-specific knowledge is
declared. Generic audits consult it through the registry instead of branching
on dataset names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa
from omegaconf import DictConfig

from conv_wm.config import Stage, get_path
from conv_wm.data.media.audio.decode import DecodeValidationCase
from conv_wm.data.media.audio.interpretation import KnownBoundaryGrid

TableLoader = Callable[[DictConfig], pd.DataFrame]
"""Load one table of a dataset from the configured paths."""


def csv_table(dataset: str, stage: Stage, key: str) -> TableLoader:
    """Loader for a CSV artifact declared in the dataset ``files`` configuration."""
    return lambda cfg: pd.read_csv(get_path(cfg, dataset, stage, key))


def parquet_table(dataset: str, stage: Stage, key: str) -> TableLoader:
    """Loader for a Parquet artifact declared in the dataset ``files`` configuration."""
    return lambda cfg: pd.read_parquet(get_path(cfg, dataset, stage, key))


@dataclass(frozen=True)
class TableSpec:
    """One structurally validated table of a dataset."""

    name: str
    schema: pa.DataFrameSchema
    """Enforceable contract: columns, dtypes, nullability, unique keys, checks."""
    load: TableLoader
    description: str = ""


@dataclass(frozen=True)
class RelationSpec:
    """A foreign-key style relation between two tables of the same dataset."""

    name: str
    child_table: str
    child_columns: tuple[str, ...]
    parent_table: str
    parent_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.child_columns) != len(self.parent_columns):
            raise ValueError(
                f"Relation {self.name}: child and parent key lengths differ."
            )


@dataclass(frozen=True)
class StructuralSpec:
    """Tables and relations a dataset submits to structural validation."""

    tables: tuple[TableSpec, ...]
    relations: tuple[RelationSpec, ...] = ()

    def __post_init__(self) -> None:
        names = {table.name for table in self.tables}
        if len(names) != len(self.tables):
            raise ValueError("StructuralSpec table names must be unique.")
        for relation in self.relations:
            unknown = {relation.child_table, relation.parent_table} - names
            if unknown:
                raise ValueError(
                    f"Relation {relation.name} references unknown tables {unknown}."
                )

    def table(self, name: str) -> TableSpec:
        """Look up a table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(name)


DecodeCaseSelector = Callable[[pd.DataFrame], list[DecodeValidationCase]]
"""Given this dataset's rows of the audio file table, propose extra decode windows."""

SummarySectionBuilder = Callable[[pd.DataFrame, pd.DataFrame], Mapping[str, object]]
"""Given this dataset's file table and decode validation table, build a summary section."""


@dataclass(frozen=True)
class AudioInterpretation:
    """Dataset knowledge applied *after* generic audio timeline analysis."""

    known_boundary_grid: KnownBoundaryGrid | None = None
    """Periodic recording joins (e.g. a stitch grid) used to relabel events."""
    extra_decode_cases: DecodeCaseSelector | None = None
    """Additional decoded-validation windows the dataset wants covered."""
    summary_section: SummarySectionBuilder | None = None
    """Dataset-specific characterization added to the population summary."""


@dataclass(frozen=True)
class DatasetSpec:
    """Registration record of one dataset."""

    name: str
    """Dataset key: the directory name below the raw root, lower-cased."""
    description: str = ""
    structure: StructuralSpec | None = None
    """Structural contracts; ``None`` when the dataset has no tabular annotations."""
    audio: AudioInterpretation = field(default_factory=AudioInterpretation)

    def __post_init__(self) -> None:
        if self.name != self.name.lower() or not self.name:
            raise ValueError("DatasetSpec.name must be a non-empty lower-case key.")
