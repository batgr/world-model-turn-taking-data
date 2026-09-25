"""Native social annotations (Looking-At-Me, Talking-To-Me, face tracks) on the grid.

Semantics follow the Ego4D Social benchmark rather than a local reading:

* **Looking At Me** is a frame-level label of a *tracked* face. A subframe of
  a participant is ``True`` inside a positive native segment, ``False`` where
  the face is tracked (or a native negative segment covers it) without a
  positive one, and ``null`` where the face is not tracked: not annotated is
  never negative.
* **Talking To Me** is an utterance-level label of a speaking person: the
  value of the native talking segment covering the subframe, ``null``
  outside talking segments.
* The ``anyone_*`` aggregates are three-valued: ``True`` if any participant
  is ``True``; ``False`` only when at least one relevant participant exists
  and every relevant participant is known ``False``; ``null`` otherwise.

Native rows whose person does not resolve to a participant (e.g. ``-1``) are
kept in the segments table with ``resolved = False`` and never painted on the
grid. The raw ``target`` field is preserved as ``annotation_target``: its
semantics are undocumented, so it is never promoted to an addressee.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from conv_wm.data.labels.gridding import (
    GridFrame,
    IntervalCoverage,
    fixed,
    nullable_vectors,
    per_participant,
)
from conv_wm.data.labels.timeline import EPSILON_S, SPEAKING, RecordingStructure

SOCIAL_SEGMENT_SCHEMA = pa.schema(
    [
        ("recording_id", pa.string()),
        ("segment_type", pa.string()),
        ("segment_id", pa.int64()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("duration_s", pa.float64()),
        ("participant_index", pa.int16()),
        ("participant_id", pa.string()),
        ("person", pa.string()),
        ("kind", pa.string()),
        ("is_at_me", pa.bool_()),
        ("annotation_target", pa.string()),
        ("resolved", pa.bool_()),
        ("start_frame", pa.int64()),
        ("end_frame", pa.int64()),
        ("track_id", pa.string()),
        ("first_frame", pa.int64()),
        ("last_frame", pa.int64()),
        ("frame_count", pa.int64()),
    ]
)


def _optional_bool(value: Any) -> bool | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"invalid native boolean {value!r}")


def _optional_str(value: Any) -> str | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return str(value)


def social_grid(
    structure: RecordingStructure,
    frame: GridFrame,
    social: pd.DataFrame,
    tracks: pd.DataFrame,
) -> dict[str, pa.Array]:
    """Grid columns of the social extractor for one recording."""
    ids = structure.pieces.participant_ids
    index = {pid: i for i, pid in enumerate(ids)}
    count, size, participants = len(frame.decision_index), frame.subframes, len(ids)
    sub_start = frame.subframe_start
    sub_end = sub_start + frame.subframe_s
    hidden = (
        IntervalCoverage(
            [(u.start_s, u.end_s) for u in structure.recording.unknown]
        ).between(sub_start, sub_end)
        > EPSILON_S
    )

    def coverage(rows: pd.DataFrame) -> np.ndarray:
        return (
            IntervalCoverage(
                list(zip(rows["start_s"], rows["end_s"], strict=True))
            ).between(sub_start, sub_end)
            > EPSILON_S
        )

    shape = (count, size, participants)
    tracked = np.zeros(shape, dtype=bool)
    box_sum = np.zeros((count, participants, 4))
    box_count = np.zeros((count, participants))
    resolved_tracks = (
        tracks[tracks["participant_id"].isin(index)] if len(tracks) else tracks
    )
    if len(resolved_tracks):
        rows, subs, inside = frame.locate(resolved_tracks["time_s"].to_numpy(float))
        who = resolved_tracks["participant_id"].map(index).to_numpy(int)
        tracked[rows[inside], subs[inside], who[inside]] = True
        boxes = resolved_tracks[["x", "y", "width", "height"]].to_numpy(float)
        np.add.at(box_sum, (rows[inside], who[inside]), boxes[inside])
        np.add.at(box_count, (rows[inside], who[inside]), 1.0)

    looking = np.zeros(shape, dtype=bool)
    looking_known = tracked.copy()
    looking_any = np.zeros(shape, dtype=bool)
    talking = np.zeros(shape, dtype=bool)
    talking_known = np.zeros(shape, dtype=bool)
    talking_any = np.zeros(shape, dtype=bool)
    for pid, p in index.items():
        own = social[social["participant_id"].eq(pid)] if len(social) else social
        if not len(own):
            continue
        at_me = own["is_at_me"].map(_optional_bool)
        for kind, value, known, covered in (
            ("looking", looking, looking_known, looking_any),
            ("talking", talking, talking_known, talking_any),
        ):
            rows = own[own["kind"].eq(kind)]
            flags = at_me[rows.index]
            covered[:, :, p] = coverage(rows)
            positive = coverage(rows[flags.eq(True)])
            negative = coverage(rows[flags.eq(False)])
            value[:, :, p] = positive
            known[:, :, p] |= positive | negative

    speaking = np.stack(
        [
            IntervalCoverage(
                [
                    (s, e)
                    for s, e, state in zip(
                        structure.pieces.starts,
                        structure.pieces.ends,
                        structure.pieces.states[:, p],
                        strict=True,
                    )
                    if state == SPEAKING
                ]
            ).between(sub_start, sub_end)
            > EPSILON_S
            for p in range(participants)
        ],
        axis=2,
    )
    columns: dict[str, pa.Array] = {}
    for name, value, known in (
        ("looking_at_wearer", looking, looking_known),
        ("talking_to_wearer", talking, talking_known),
    ):
        valid = known & ~hidden[:, :, None]
        columns[f"{name}_subframes"] = per_participant(
            value.transpose(0, 2, 1), (~valid).transpose(0, 2, 1), pa.bool_()
        )
        columns[f"{name}_valid_subframes"] = per_participant(
            valid.transpose(0, 2, 1), None, pa.bool_()
        )
    for name, value, known, relevant in (
        ("looking_at_wearer", looking, looking_known, tracked | looking_any),
        ("talking_to_wearer", talking, talking_known, talking_any | speaking),
    ):
        positive = (value & known).any(axis=2)
        negative = relevant.any(axis=2) & (~relevant | (known & ~value)).all(axis=2)
        valid = (positive | negative) & ~hidden
        columns[f"anyone_{name}_subframes"] = fixed(positive, ~valid, size, pa.bool_())
        columns[f"anyone_{name}_valid_subframes"] = fixed(valid, None, size, pa.bool_())
    columns["face_tracked_subframes"] = per_participant(
        tracked.transpose(0, 2, 1),
        np.repeat(hidden[:, None, :], participants, axis=1),
        pa.bool_(),
    )
    mean = np.divide(
        box_sum,
        box_count[:, :, None],
        out=np.zeros_like(box_sum),
        where=box_count[:, :, None] > 0,
    )
    boxes = nullable_vectors(
        mean.astype(np.float32).reshape(-1, 4),
        (box_count == 0).reshape(-1),
        pa.float32(),
    )
    offsets = pa.array(np.arange(count + 1, dtype=np.int32) * participants)
    columns["face_track_bbox"] = pa.ListArray.from_arrays(offsets, boxes)
    columns["participant_ids"] = pa.array(
        [list(ids)] * count, type=pa.list_(pa.string())
    )
    return columns


def social_segment_rows(
    structure: RecordingStructure,
    social: pd.DataFrame,
    tracks: pd.DataFrame,
    frame_rate_hz: float | None,
) -> list[dict[str, Any]]:
    """Native social segments (resolved or not) and one row per face track."""
    ids = structure.pieces.participant_ids
    index = {pid: i for i, pid in enumerate(ids)}
    recording_id = structure.recording.recording_id
    rows: list[dict[str, Any]] = []
    ordered = (
        social.sort_values(["start_s", "kind", "person", "source_row"], kind="stable")
        if len(social)
        else social
    )
    for ordinal, item in enumerate(ordered.to_dict(orient="records")):
        pid = _optional_str(item["participant_id"])
        resolved = pid is not None and pid in index
        rows.append(
            {
                "recording_id": recording_id,
                "segment_type": "social_segment",
                "segment_id": ordinal,
                "start_s": float(item["start_s"]),
                "end_s": float(item["end_s"]),
                "duration_s": float(item["end_s"]) - float(item["start_s"]),
                "participant_index": index[pid]
                if resolved and pid is not None
                else None,
                "participant_id": pid if resolved else None,
                "person": _optional_str(item["person"]),
                "kind": str(item["kind"]),
                "is_at_me": _optional_bool(item["is_at_me"]),
                "annotation_target": _optional_str(item["annotation_target"]),
                "resolved": resolved,
                "start_frame": _optional_int(item["start_frame"]),
                "end_frame": _optional_int(item["end_frame"]),
            }
        )
    if len(tracks):
        step = 1.0 / frame_rate_hz if frame_rate_hz else 0.0
        grouped = tracks.groupby(["participant_id", "track_id"], sort=True)
        summary = grouped.agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            frame_count=("frame", "size"),
            start_s=("time_s", "min"),
            end_s=("time_s", "max"),
        ).reset_index()
        summary = summary.sort_values(
            ["start_s", "participant_id", "track_id"], kind="stable"
        )
        for ordinal, item in enumerate(summary.to_dict(orient="records")):
            pid = str(item["participant_id"])
            end_s = float(item["end_s"]) + step
            rows.append(
                {
                    "recording_id": recording_id,
                    "segment_type": "face_track",
                    "segment_id": ordinal,
                    "start_s": float(item["start_s"]),
                    "end_s": end_s,
                    "duration_s": end_s - float(item["start_s"]),
                    "participant_index": index.get(pid),
                    "participant_id": pid if pid in index else None,
                    "resolved": pid in index,
                    "track_id": str(item["track_id"]),
                    "first_frame": int(item["first_frame"]),
                    "last_frame": int(item["last_frame"]),
                    "frame_count": int(item["frame_count"]),
                }
            )
    return rows


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


__all__ = ["SOCIAL_SEGMENT_SCHEMA", "social_grid", "social_segment_rows"]
