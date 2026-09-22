"""Native focal vocal state -> control focal vocal state at resolution Δ.

The native timeline is what the annotations say; the control timeline is what a
controller running at ``decision_step_s`` can represent. The two differ by one
quantization rule, applied here and nowhere else:

```text
SPEAKING -> SILENT for g < Δ -> SPEAKING   becomes   SPEAKING continuous
```

This is **sub-step silence bridging**: a statement about the controller's
temporal resolution, not a correction of the annotation. The native layer is
immutable and this module never writes to it.

The symmetric rule is deliberately **not** applied: a short ``SPEAKING`` burst
between two silences is kept, whatever its duration. Short vocalizations are
signal; when they put two transitions in one slot the action grid masks that
slot as ``compound_transition``, exactly as before.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

from conv_wm.data.vocal.action_grid import BOUNDARY_TOLERANCE_S, DECISION_STEP_S
from conv_wm.data.vocal.native_state import NativeStateInterval, VoiceState

CONTROL_STATE_SCHEMA_VERSION = 1
"""Bumped when the columns or the transform semantics of the control layer change."""


class TransformKind(StrEnum):
    """Which control transform produced an interval that is not a native copy."""

    SUB_STEP_SILENCE_BRIDGE = "sub_step_silence_bridge"


@dataclass(frozen=True)
class BridgedGap:
    """One native sub-Δ ``SILENT`` gap absorbed into a continuous ``SPEAKING``."""

    dataset: str
    recording_id: str
    native_index: int
    """Position of the bridged SILENT interval in the recording's native timeline."""
    gap_start_s: float
    gap_end_s: float
    gap_duration_s: float

    def to_row(self) -> dict[str, Any]:
        """Stable Parquet row."""
        return asdict(self)


@dataclass(frozen=True)
class ControlStateInterval:
    """One row of the control timeline: a native copy, or a bridged ``SPEAKING``.

    ``source_native_index_first``/``_last`` address the native intervals this
    row was built from, in the recording's native timeline order, so every
    control row can be traced back without re-deriving the transform.
    """

    dataset: str
    recording_id: str
    sync_group_id: str | None
    view_id: str
    wearer_id: str
    canonical_start_s: float
    canonical_end_s: float
    voice_state: VoiceState
    source_kind: str
    source_annotation_id: str | None
    annotation_schema_version: str
    source_native_index_first: int
    source_native_index_last: int
    transform_kind: str | None
    bridged_gap_count: int
    bridged_gap_total_duration_s: float
    native_state_schema_version: int
    control_state_schema_version: int = CONTROL_STATE_SCHEMA_VERSION

    @property
    def duration_s(self) -> float:
        """Interval length in seconds."""
        return self.canonical_end_s - self.canonical_start_s

    def to_row(self) -> dict[str, Any]:
        """Stable Parquet row: enums as strings, one column per field."""
        row = asdict(self)
        row["voice_state"] = str(self.voice_state)
        row["source_kind"] = str(self.source_kind)
        return row


def _abuts(
    previous: NativeStateInterval, current: NativeStateInterval, tolerance_s: float
) -> bool:
    return abs(current.canonical_start_s - previous.canonical_end_s) <= tolerance_s


def is_bridgeable(
    before: NativeStateInterval,
    gap: NativeStateInterval,
    after: NativeStateInterval,
    *,
    step_s: float = DECISION_STEP_S,
    tolerance_s: float = BOUNDARY_TOLERANCE_S,
) -> bool:
    """Whether ``gap`` is a sub-step silence framed by speech, hence bridgeable.

    The condition is strict — ``gap_duration_s < decision_step_s`` — so a gap
    of exactly Δ is representable and is kept. ``tolerance_s`` is the tolerance
    already used for timestamp comparisons: it decides equality, it never
    modifies a timestamp. ``UNKNOWN`` is never bridged across, because both
    frames must be ``SPEAKING``.
    """
    return (
        before.voice_state == VoiceState.SPEAKING
        and gap.voice_state == VoiceState.SILENT
        and after.voice_state == VoiceState.SPEAKING
        and _abuts(before, gap, tolerance_s)
        and _abuts(gap, after, tolerance_s)
        and gap.duration_s < step_s - tolerance_s
    )


def _annotation_ids(rows: Sequence[NativeStateInterval]) -> str | None:
    ids = {
        part
        for row in rows
        if row.source_annotation_id
        for part in str(row.source_annotation_id).split("|")
    }
    return "|".join(sorted(ids)) if ids else None


def build_control_timeline(
    rows: Sequence[NativeStateInterval],
    *,
    step_s: float = DECISION_STEP_S,
    tolerance_s: float = BOUNDARY_TOLERANCE_S,
) -> tuple[list[ControlStateInterval], list[BridgedGap]]:
    """Bridge every sub-step silence of one recording's native timeline.

    ``rows`` must be one recording's native intervals in timeline order. The
    result covers exactly the same span: bridging only removes interior
    boundaries, so the total duration and the two endpoints are unchanged.
    Successive micro-gaps chain into a single ``SPEAKING`` interval in one
    deterministic left-to-right pass.
    """
    control: list[ControlStateInterval] = []
    bridged: list[BridgedGap] = []
    index = 0
    while index < len(rows):
        first = last = index
        gaps: list[BridgedGap] = []
        while last + 2 < len(rows) and is_bridgeable(
            rows[last],
            rows[last + 1],
            rows[last + 2],
            step_s=step_s,
            tolerance_s=tolerance_s,
        ):
            gap = rows[last + 1]
            gaps.append(
                BridgedGap(
                    dataset=gap.dataset,
                    recording_id=gap.recording_id,
                    native_index=last + 1,
                    gap_start_s=gap.canonical_start_s,
                    gap_end_s=gap.canonical_end_s,
                    gap_duration_s=gap.duration_s,
                )
            )
            last += 2
        source = rows[first]
        merged = rows[first : last + 1]
        control.append(
            ControlStateInterval(
                dataset=source.dataset,
                recording_id=source.recording_id,
                sync_group_id=source.sync_group_id,
                view_id=source.view_id,
                wearer_id=source.wearer_id,
                canonical_start_s=source.canonical_start_s,
                canonical_end_s=rows[last].canonical_end_s,
                voice_state=source.voice_state,
                source_kind=str(source.source_kind),
                source_annotation_id=_annotation_ids(merged)
                if gaps
                else source.source_annotation_id,
                annotation_schema_version=source.annotation_schema_version,
                source_native_index_first=first,
                source_native_index_last=last,
                transform_kind=str(TransformKind.SUB_STEP_SILENCE_BRIDGE)
                if gaps
                else None,
                bridged_gap_count=len(gaps),
                bridged_gap_total_duration_s=sum(gap.gap_duration_s for gap in gaps),
                native_state_schema_version=source.native_state_schema_version,
            )
        )
        bridged.extend(gaps)
        index = last + 1
    return control, bridged


def resolved_transition_count(states: Sequence[str]) -> int:
    """Number of ``SILENT`` <-> ``SPEAKING`` changes along a state sequence.

    Changes to or from ``UNKNOWN`` are not resolved transitions: they can never
    log an action, so they are excluded from the before/after comparison.
    """
    unknown = str(VoiceState.UNKNOWN)
    return sum(
        before != after and before != unknown and after != unknown
        for before, after in pairwise(states)
    )


__all__ = [
    "CONTROL_STATE_SCHEMA_VERSION",
    "BridgedGap",
    "ControlStateInterval",
    "TransformKind",
    "build_control_timeline",
    "is_bridgeable",
    "resolved_transition_count",
]
