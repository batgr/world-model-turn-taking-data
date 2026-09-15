"""Packet timeline normalization and container-level (packet-clock) metrics."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from conv_wm.data.media.audio.records import (
    PacketClockMetrics,
    PacketCoverage,
    PacketDurationMetrics,
)


def ticks_to_samples(values: np.ndarray, samples_per_tick: Fraction) -> np.ndarray:
    """Convert integer time-base ticks to samples through an exact rational ratio."""
    return (
        values.astype(np.float64)
        * samples_per_tick.numerator
        / samples_per_tick.denominator
    )


@dataclass(frozen=True)
class PacketTimeline:
    """Integer PTS/duration arrays of one audio stream plus its clock ratios.

    ``pts`` and ``duration`` keep ``NaN`` where ffprobe reported no value; the
    ``*_present`` masks say which entries are real. Steps are only formed between
    adjacent packets that both carry a PTS.
    """

    pts: pd.Series
    duration: pd.Series
    time_base: Fraction
    samples_per_tick: Fraction
    sample_rate_hz: int

    @classmethod
    def from_packets(
        cls,
        packets: list[dict[str, object]],
        *,
        sample_rate_hz: int,
        time_base: str,
    ) -> PacketTimeline:
        """Normalize raw ffprobe packet dictionaries."""
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        time_base_fraction = Fraction(time_base)
        return cls(
            pts=pd.to_numeric(
                pd.Series([packet.get("pts") for packet in packets], dtype=object),
                errors="coerce",
            ),
            duration=pd.to_numeric(
                pd.Series([packet.get("duration") for packet in packets], dtype=object),
                errors="coerce",
            ),
            time_base=time_base_fraction,
            samples_per_tick=Fraction(sample_rate_hz) * time_base_fraction,
            sample_rate_hz=sample_rate_hz,
        )

    @property
    def n_packets(self) -> int:
        """Number of packets, including ones without PTS or duration."""
        return len(self.pts)

    @property
    def pts_present(self) -> np.ndarray:
        """Boolean mask of packets that carry a PTS."""
        return self.pts.notna().to_numpy()

    @property
    def duration_present(self) -> np.ndarray:
        """Boolean mask of packets that carry a duration."""
        return self.duration.notna().to_numpy()

    @property
    def pts_ticks(self) -> np.ndarray:
        """Integer PTS array with missing values filled by zero (mask them!)."""
        return self.pts.fillna(0).astype(np.int64).to_numpy()

    @property
    def duration_ticks(self) -> np.ndarray:
        """Integer duration array with missing values filled by zero (mask them!)."""
        return self.duration.fillna(0).astype(np.int64).to_numpy()

    @property
    def step_packet_indices(self) -> np.ndarray:
        """Indices ``i`` such that packets ``i`` and ``i + 1`` both carry a PTS."""
        present = self.pts_present
        return np.flatnonzero(present[:-1] & present[1:])

    def pts_steps_ticks(self) -> np.ndarray:
        """``pts[i + 1] - pts[i]`` for each valid step, in time-base ticks."""
        indices = self.step_packet_indices
        pts = self.pts_ticks
        return pts[indices + 1] - pts[indices]

    def coverage(self) -> PacketCoverage:
        """Field coverage and monotonicity of the packet PTS sequence."""
        n = self.n_packets
        both = self.pts_present & self.duration_present
        valid_pts = self.pts.dropna().astype(np.int64).to_numpy()
        diffs = np.diff(valid_pts) if len(valid_pts) > 1 else np.array([], dtype=int)
        return PacketCoverage(
            n_packets=n,
            n_valid_packets=int(both.sum()),
            timestamp_coverage=float(self.pts_present.mean()) if n else 0.0,
            duration_coverage=float(self.duration_present.mean()) if n else 0.0,
            timestamp_duration_coverage=float(both.mean()) if n else 0.0,
            pts_monotonic=bool(len(diffs) and np.all(diffs > 0)),
            n_non_positive_pts_steps=int((diffs <= 0).sum()),
            samples_per_time_base_tick_numerator=self.samples_per_tick.numerator,
            samples_per_time_base_tick_denominator=self.samples_per_tick.denominator,
        )


@dataclass(frozen=True)
class PacketClockErrors:
    """Per-packet ``next_pts - pts - duration`` for packets carrying a duration."""

    packet_indices: np.ndarray
    errors_samples: np.ndarray

    def metrics(self) -> PacketClockMetrics:
        """Aggregate container self-consistency counts."""
        errors = self.errors_samples
        return PacketClockMetrics(
            n_valid_packet_timeline_steps=len(errors),
            packet_timeline_gap_count=int((errors > 0).sum()),
            packet_timeline_overlap_count=int((errors < 0).sum()),
            packet_max_abs_timeline_error_samples=(
                float(np.abs(errors).max()) if len(errors) else float("nan")
            ),
        )


def packet_clock_errors(timeline: PacketTimeline) -> PacketClockErrors:
    """Measure whether the container's own packet durations tile the PTS axis."""
    steps = timeline.step_packet_indices
    with_duration = steps[timeline.duration_present[:-1][steps]]
    pts = timeline.pts_ticks
    error_ticks = (
        pts[with_duration + 1]
        - pts[with_duration]
        - timeline.duration_ticks[with_duration]
    )
    return PacketClockErrors(
        packet_indices=with_duration,
        errors_samples=ticks_to_samples(error_ticks, timeline.samples_per_tick),
    )


def packet_duration_metrics(
    timeline: PacketTimeline,
    *,
    reference_decoded_samples: int,
    threshold_samples: float,
) -> PacketDurationMetrics:
    """Describe the packet-duration distribution and its long-duration fence.

    Abnormally long durations are counted as observations; they do not classify
    events (see :mod:`conv_wm.data.media.audio.pcm_clock`).
    """
    valid = timeline.duration.dropna().astype(np.int64).to_numpy()
    if not len(valid):
        return PacketDurationMetrics.empty()
    durations = ticks_to_samples(valid, timeline.samples_per_tick)
    values, counts = np.unique(valid, return_counts=True)
    nominal = float(Fraction(int(values[counts.argmax()])) * timeline.samples_per_tick)
    robust_nominal = float(np.median(durations))
    mad = float(np.median(np.abs(durations - robust_nominal)))
    long_threshold = max(
        robust_nominal + max(6.0 * mad, 0.5 * robust_nominal),
        reference_decoded_samples + threshold_samples,
    )
    deviation = durations - reference_decoded_samples
    return PacketDurationMetrics(
        nominal_packet_duration_samples=nominal,
        robust_nominal_packet_duration_samples=robust_nominal,
        packet_duration_mad_samples=mad,
        long_packet_duration_threshold_samples=long_threshold,
        min_packet_duration_samples=float(durations.min()),
        median_packet_duration_samples=float(np.median(durations)),
        max_packet_duration_samples=float(durations.max()),
        n_distinct_packet_durations=len(values),
        n_packet_duration_variations=int(
            (~np.isclose(deviation, 0.0, atol=1e-12)).sum()
        ),
        max_abs_packet_duration_deviation_samples=float(np.abs(deviation).max()),
        n_abnormally_long_packet_durations=int((durations > long_threshold).sum()),
    )


def step_duration_samples(timeline: PacketTimeline) -> np.ndarray:
    """Duration (samples) of the packet that starts each valid PTS step, NaN if absent."""
    steps = timeline.step_packet_indices
    result = np.full(len(steps), np.nan)
    present = timeline.duration_present[steps]
    result[present] = ticks_to_samples(
        timeline.duration_ticks[steps[present]], timeline.samples_per_tick
    )
    return result
