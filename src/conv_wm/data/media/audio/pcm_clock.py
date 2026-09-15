"""PCM-vs-PTS clock measurements and event classification.

The PCM clock is what a naïve consumer sees: decoded frames of a fixed nominal
size concatenated back to back. Comparing each PTS step with that nominal size
reveals where native timestamps and concatenated PCM diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from conv_wm.data.media.audio.packets import (
    PacketClockErrors,
    PacketTimeline,
    ticks_to_samples,
)
from conv_wm.data.media.audio.records import (
    AudioTimelineEvent,
    AudioTimelineParameters,
    DropoutAccounting,
    EventCategory,
    EventDirection,
    EventType,
    PacketDurationMetrics,
    PcmClockMetrics,
    ticks_to_seconds,
)


@dataclass(frozen=True)
class PcmClockSeries:
    """Per-step PCM clock errors and their cumulative sum, in samples."""

    step_packet_indices: np.ndarray
    """Packet index before each step (see ``PacketTimeline.step_packet_indices``)."""
    next_pts: np.ndarray
    """PTS (ticks) of the packet after each step."""
    step_errors: np.ndarray
    """``pts_step - reference_decoded_samples`` per step."""
    cumulative_drift: np.ndarray
    step_duration_samples: np.ndarray
    """Duration of the packet starting each step, NaN when absent."""

    @property
    def n_steps(self) -> int:
        """Number of valid PTS steps."""
        return len(self.step_errors)


def pcm_clock_series(
    timeline: PacketTimeline,
    *,
    reference_decoded_samples: int,
    step_duration_samples: np.ndarray,
) -> PcmClockSeries:
    """Compare each PTS step with the nominal decoded frame size."""
    steps = timeline.step_packet_indices
    step_samples = ticks_to_samples(
        timeline.pts_steps_ticks(), timeline.samples_per_tick
    )
    errors = step_samples - reference_decoded_samples
    return PcmClockSeries(
        step_packet_indices=steps,
        next_pts=timeline.pts_ticks[steps + 1],
        step_errors=errors,
        cumulative_drift=np.cumsum(errors),
        step_duration_samples=step_duration_samples,
    )


def pcm_clock_metrics(
    series: PcmClockSeries,
    *,
    time_base: Fraction,
    sample_rate_hz: int,
    threshold_samples: float,
) -> PcmClockMetrics:
    """Aggregate step errors and cumulative drift into file-level metrics."""
    errors = series.step_errors
    cumulative = series.cumulative_drift
    abs_cumulative = np.abs(cumulative)
    if len(errors):
        peak = int(abs_cumulative.argmax())
        final_drift = float(cumulative[-1])
        max_abs_cumulative = float(abs_cumulative[peak])
        peak_time = ticks_to_seconds(int(series.next_pts[peak]), time_base)
        max_abs_step = float(np.abs(errors).max())
        p95 = float(np.quantile(np.abs(errors), 0.95))
    else:
        final_drift = max_abs_cumulative = peak_time = max_abs_step = p95 = float("nan")
    n_variations = int((~np.isclose(errors, 0.0, atol=1e-12)).sum())
    return PcmClockMetrics(
        n_valid_pts_steps=len(errors),
        n_pcm_step_variations=n_variations,
        max_abs_pcm_step_error_samples=max_abs_step,
        pcm_step_error_p95_abs_samples=p95,
        final_cumulative_pcm_drift_samples=final_drift,
        final_cumulative_pcm_drift_ms=1000.0 * final_drift / sample_rate_hz,
        max_abs_cumulative_pcm_drift_samples=max_abs_cumulative,
        max_abs_cumulative_pcm_drift_ms=1000.0 * max_abs_cumulative / sample_rate_hz,
        max_abs_cumulative_pcm_drift_time_sec=peak_time,
        has_timestamp_jitter=n_variations > 0,
        has_small_bounded_timestamp_jitter=bool(
            n_variations > 0
            and max_abs_step <= threshold_samples
            and max_abs_cumulative <= threshold_samples
        ),
        has_significant_transient_pcm_offset=bool(
            max_abs_cumulative > threshold_samples
        ),
        has_significant_final_pcm_offset=bool(abs(final_drift) > threshold_samples),
    )


@dataclass(frozen=True)
class CandidatePersistence:
    """Dropout candidates (positive steps above threshold) and their persistence.

    For a candidate at step ``k`` with excess ``e``, the residual is the cumulative
    drift minus the pre-step baseline over the next ``window_steps`` steps. The
    candidate persists when the residual never drops below
    ``persistence_fraction * e`` inside that window.
    """

    positions: np.ndarray
    excess: np.ndarray
    min_residual: np.ndarray
    end_residual: np.ndarray
    window_truncated: np.ndarray
    persistent: np.ndarray

    def accounting(self) -> DropoutAccounting:
        """Count candidates by outcome."""
        persistent = int(self.persistent.sum())
        return DropoutAccounting(
            n_dropout_candidate_steps=len(self.positions),
            n_persistent_dropout_steps=persistent,
            n_compensated_candidate_steps=len(self.positions) - persistent,
        )

    def is_persistent(self, n_steps: int) -> np.ndarray:
        """Boolean mask over all steps marking persistent candidates."""
        mask = np.zeros(n_steps, dtype=bool)
        mask[self.positions] = self.persistent
        return mask

    def offset_of(self, position: int) -> int:
        """Index into the candidate arrays for a step position."""
        return int(np.flatnonzero(self.positions == position)[0])


def candidate_persistence(
    series: PcmClockSeries,
    *,
    threshold_samples: float,
    window_steps: int,
    persistence_fraction: float,
) -> CandidatePersistence:
    """Test each positive step above threshold against its compensation window."""
    positions = np.flatnonzero(series.step_errors > threshold_samples)
    n_steps = series.n_steps
    cumulative = series.cumulative_drift
    excess = series.step_errors[positions]
    baseline = np.where(positions > 0, cumulative[np.maximum(positions - 1, 0)], 0.0)
    window_end = np.minimum(positions + window_steps, n_steps - 1)
    min_residual = np.empty(len(positions))
    end_residual = np.empty(len(positions))
    for offset, (position, stop) in enumerate(zip(positions, window_end, strict=True)):
        residual = cumulative[position : stop + 1] - baseline[offset]
        min_residual[offset] = residual.min()
        end_residual[offset] = residual[-1]
    return CandidatePersistence(
        positions=positions,
        excess=excess,
        min_residual=min_residual,
        end_residual=end_residual,
        window_truncated=positions + window_steps > n_steps - 1,
        persistent=min_residual >= persistence_fraction * excess,
    )


def packet_clock_events(
    errors: PacketClockErrors,
    timeline: PacketTimeline,
    *,
    durations: PacketDurationMetrics,
    reference_decoded_samples: int,
) -> list[AudioTimelineEvent]:
    """One event per packet whose duration does not reach the next PTS."""
    events: list[AudioTimelineEvent] = []
    nan = float("nan")
    for position in np.flatnonzero(errors.errors_samples != 0):
        error = float(errors.errors_samples[position])
        index = int(errors.packet_indices[position])
        event_pts = int(timeline.pts_ticks[index + 1])
        time_sec = ticks_to_seconds(event_pts, timeline.time_base)
        events.append(
            AudioTimelineEvent(
                packet_index=index,
                next_packet_index=index + 1,
                event_type=(
                    EventType.PACKET_TIMELINE_GAP
                    if error > 0
                    else EventType.PACKET_TIMELINE_OVERLAP
                ),
                event_category=EventCategory.PACKET_TIMELINE_ERROR,
                event_direction=EventDirection.GAP
                if error > 0
                else EventDirection.OVERLAP,
                event_pts=event_pts,
                event_time_sec=time_sec,
                event_end_time_sec=time_sec,
                n_steps=1,
                magnitude_samples=abs(error),
                packet_timeline_error_samples=error,
                first_pcm_step_error_samples=nan,
                max_abs_pcm_step_error_samples=nan,
                net_pcm_drift_samples=nan,
                robust_nominal_packet_duration_samples=(
                    durations.robust_nominal_packet_duration_samples
                ),
                long_packet_duration_threshold_samples=(
                    durations.long_packet_duration_threshold_samples
                ),
                n_abnormally_long_packets=0,
                max_packet_duration_samples=nan,
                n_candidate_steps=0,
                n_compensated_candidate_steps=0,
                dropout_duration_samples=nan,
                dropout_duration_ms=nan,
                dropout_total_excess_samples=nan,
                dropout_total_excess_ms=nan,
                dropout_min_residual_in_window_samples=nan,
                dropout_residual_at_window_end_samples=nan,
                dropout_window_truncated=False,
                reference_decoded_samples=reference_decoded_samples,
            )
        )
    return events


def pcm_clock_events(
    series: PcmClockSeries,
    persistence: CandidatePersistence,
    *,
    time_base: Fraction,
    sample_rate_hz: int,
    parameters: AudioTimelineParameters,
    durations: PacketDurationMetrics,
) -> list[AudioTimelineEvent]:
    """Segment non-zero PCM steps into episodes and classify them.

    Consecutive non-zero steps form an episode. A persistent candidate becomes
    its own single-step ``AUDIO_DROPOUT`` event and splits the episode around it.
    Remaining segments are dropped when both their largest step and their
    largest excursion stay within the threshold; otherwise they are
    ``COMPENSATED_TIMESTAMP_CADENCE`` when they contain compensated candidates
    and ``PCM_TIMESTAMP_VARIATION`` otherwise.
    """
    errors = series.step_errors
    nonzero = np.flatnonzero(~np.isclose(errors, 0.0, atol=1e-12))
    if not len(nonzero):
        return []
    threshold = parameters.threshold_samples(sample_rate_hz)
    candidate_mask = errors > threshold
    persistent_mask = persistence.is_persistent(series.n_steps)
    long_mask = np.isfinite(series.step_duration_samples) & (
        series.step_duration_samples > durations.long_packet_duration_threshold_samples
    )
    nan = float("nan")

    def build(
        segment: np.ndarray,
        *,
        event_type: EventType,
        category: EventCategory,
        direction: EventDirection,
        dropout: dict[str, float | bool] | None,
    ) -> AudioTimelineEvent:
        first, last = int(segment[0]), int(segment[-1])
        segment_errors = errors[segment]
        segment_durations = series.step_duration_samples[segment]
        candidates = segment[candidate_mask[segment]]
        dropout = dropout or {}
        return AudioTimelineEvent(
            packet_index=int(series.step_packet_indices[first]),
            next_packet_index=int(series.step_packet_indices[first] + 1),
            event_type=event_type,
            event_category=category,
            event_direction=direction,
            event_pts=int(series.next_pts[first]),
            event_time_sec=ticks_to_seconds(int(series.next_pts[first]), time_base),
            event_end_time_sec=ticks_to_seconds(int(series.next_pts[last]), time_base),
            n_steps=len(segment),
            magnitude_samples=float(np.abs(segment_errors).max()),
            packet_timeline_error_samples=nan,
            first_pcm_step_error_samples=float(segment_errors[0]),
            max_abs_pcm_step_error_samples=float(np.abs(segment_errors).max()),
            net_pcm_drift_samples=float(segment_errors.sum()),
            robust_nominal_packet_duration_samples=(
                durations.robust_nominal_packet_duration_samples
            ),
            long_packet_duration_threshold_samples=(
                durations.long_packet_duration_threshold_samples
            ),
            n_abnormally_long_packets=int(long_mask[segment].sum()),
            max_packet_duration_samples=(
                float(np.nanmax(segment_durations))
                if np.isfinite(segment_durations).any()
                else nan
            ),
            n_candidate_steps=len(candidates),
            n_compensated_candidate_steps=int((~persistent_mask[candidates]).sum()),
            dropout_duration_samples=float(dropout.get("excess", nan)),
            dropout_duration_ms=float(dropout.get("excess", nan))
            * 1000.0
            / sample_rate_hz,
            dropout_total_excess_samples=float(dropout.get("excess", nan)),
            dropout_total_excess_ms=float(dropout.get("excess", nan))
            * 1000.0
            / sample_rate_hz,
            dropout_min_residual_in_window_samples=float(
                dropout.get("min_residual", nan)
            ),
            dropout_residual_at_window_end_samples=float(
                dropout.get("end_residual", nan)
            ),
            dropout_window_truncated=bool(dropout.get("truncated", False)),
            reference_decoded_samples=parameters.reference_decoded_samples,
        )

    events: list[AudioTimelineEvent] = []
    split_points = np.flatnonzero(np.diff(series.step_packet_indices[nonzero]) != 1) + 1
    for group in np.split(nonzero, split_points):
        cursor = 0
        segments: list[tuple[bool, np.ndarray]] = []
        for boundary in np.flatnonzero(persistent_mask[group]):
            if boundary > cursor:
                segments.append((False, group[cursor:boundary]))
            segments.append((True, group[boundary : boundary + 1]))
            cursor = boundary + 1
        if cursor < len(group):
            segments.append((False, group[cursor:]))

        for is_dropout, segment in segments:
            if is_dropout:
                offset = persistence.offset_of(int(segment[0]))
                events.append(
                    build(
                        segment,
                        event_type=EventType.AUDIO_DROPOUT,
                        category=EventCategory.AUDIO_DROPOUT,
                        direction=EventDirection.GAP,
                        dropout={
                            "excess": float(persistence.excess[offset]),
                            "min_residual": float(persistence.min_residual[offset]),
                            "end_residual": float(persistence.end_residual[offset]),
                            "truncated": bool(persistence.window_truncated[offset]),
                        },
                    )
                )
                continue
            segment_errors = errors[segment]
            excursion = np.cumsum(segment_errors)
            if (
                float(np.abs(segment_errors).max()) <= threshold
                and float(np.abs(excursion).max()) <= threshold
            ):
                continue
            if np.all(segment_errors > 0):
                direction = EventDirection.GAP
            elif np.all(segment_errors < 0):
                direction = EventDirection.OVERLAP
            else:
                direction = EventDirection.MIXED
            events.append(
                build(
                    segment,
                    event_type=EventType.PCM_STEP_EPISODE,
                    category=(
                        EventCategory.COMPENSATED_TIMESTAMP_CADENCE
                        if candidate_mask[segment].any()
                        else EventCategory.PCM_TIMESTAMP_VARIATION
                    ),
                    direction=direction,
                    dropout=None,
                )
            )
    events.sort(key=lambda event: event.packet_index)
    return events
