"""Typed description of what a dataset contributes to the generic pipeline.

A :class:`DatasetSpec` is the single place where dataset-specific knowledge is
declared. Generic audits consult it through the registry instead of branching
on dataset names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from conv_wm.data.media.audio.decode import DecodeValidationCase
from conv_wm.data.media.audio.interpretation import KnownBoundaryGrid

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
    audio: AudioInterpretation = field(default_factory=AudioInterpretation)

    def __post_init__(self) -> None:
        if self.name != self.name.lower() or not self.name:
            raise ValueError("DatasetSpec.name must be a non-empty lower-case key.")
