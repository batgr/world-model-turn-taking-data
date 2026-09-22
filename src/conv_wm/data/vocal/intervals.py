"""Closed-open time intervals ``[start, end)`` as ``(n, 2)`` float arrays, in seconds."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

Intervals = np.ndarray
"""Shape ``(n, 2)``, sorted by start, non-overlapping unless stated otherwise."""


def as_intervals(pairs: Iterable[Sequence[float]] | np.ndarray) -> Intervals:
    """Validate and sort ``(start, end)`` pairs; drops empty or reversed pairs."""
    array = np.asarray(
        list(pairs) if not isinstance(pairs, np.ndarray) else pairs, dtype=float
    )
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("intervals must have shape (n, 2)")
    array = array[np.isfinite(array).all(axis=1) & (array[:, 1] > array[:, 0])]
    return array[np.argsort(array[:, 0], kind="stable")]


def merge(intervals: Intervals, *, gap_s: float = 0.0) -> Intervals:
    """Union of intervals, also joining those separated by at most ``gap_s``."""
    intervals = as_intervals(intervals)
    if len(intervals) == 0:
        return intervals
    merged = [intervals[0].copy()]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append(np.array([start, end]))
    return np.asarray(merged, dtype=float)


def dilate(intervals: Intervals, tolerance_s: float) -> Intervals:
    """Widen every interval by ``tolerance_s`` on both sides and re-merge."""
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be >= 0")
    intervals = as_intervals(intervals)
    if len(intervals) == 0:
        return intervals
    widened = intervals + np.array([-tolerance_s, tolerance_s])
    return merge(widened)


def clip(intervals: Intervals, start_s: float, end_s: float) -> Intervals:
    """Restrict intervals to ``[start_s, end_s)``."""
    intervals = as_intervals(intervals)
    if len(intervals) == 0:
        return intervals
    clipped = np.column_stack(
        [np.maximum(intervals[:, 0], start_s), np.minimum(intervals[:, 1], end_s)]
    )
    return clipped[clipped[:, 1] > clipped[:, 0]]


def total_duration(intervals: Intervals) -> float:
    """Sum of lengths of (merged) intervals."""
    merged = merge(intervals)
    return float((merged[:, 1] - merged[:, 0]).sum()) if len(merged) else 0.0


def overlap_duration(start_s: float, end_s: float, reference: Intervals) -> float:
    """Length of ``[start_s, end_s)`` covered by the union of ``reference``."""
    reference = merge(reference)
    if len(reference) == 0:
        return 0.0
    overlaps = np.minimum(reference[:, 1], end_s) - np.maximum(reference[:, 0], start_s)
    return float(np.clip(overlaps, 0.0, None).sum())


def overlap_ratio(start_s: float, end_s: float, reference: Intervals) -> float:
    """Fraction of ``[start_s, end_s)`` covered by ``reference`` (0 for empty segments)."""
    length = end_s - start_s
    if length <= 0:
        return 0.0
    return min(1.0, overlap_duration(start_s, end_s, reference) / length)


def subtract(intervals: Intervals, reference: Intervals) -> Intervals:
    """Parts of ``intervals`` not covered by ``reference``."""
    intervals = merge(intervals)
    reference = merge(reference)
    if len(intervals) == 0 or len(reference) == 0:
        return intervals
    remaining: list[list[float]] = []
    for start, end in intervals:
        cursor = start
        for ref_start, ref_end in reference:
            if ref_end <= cursor:
                continue
            if ref_start >= end:
                break
            if ref_start > cursor:
                remaining.append([cursor, ref_start])
            cursor = max(cursor, ref_end)
            if cursor >= end:
                break
        if cursor < end:
            remaining.append([cursor, end])
    return as_intervals(remaining)


def nearest_boundary_offsets(
    start_s: float, end_s: float, reference: Intervals
) -> tuple[float, float]:
    """Signed offsets ``(start - nearest ref start, end - nearest ref end)`` in seconds.

    Returns ``(nan, nan)`` when ``reference`` is empty.
    """
    reference = as_intervals(reference)
    if len(reference) == 0:
        return float("nan"), float("nan")
    start_offset = start_s - reference[np.abs(reference[:, 0] - start_s).argmin(), 0]
    end_offset = end_s - reference[np.abs(reference[:, 1] - end_s).argmin(), 1]
    return float(start_offset), float(end_offset)
