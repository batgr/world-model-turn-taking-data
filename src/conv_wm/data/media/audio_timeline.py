"""Audio packet-clock and decoded-PCM timeline measurements."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from conv_wm.data.media.ffprobe import probe_json


def probe_audio_frames(
    path: Path,
    *,
    read_intervals: str | None = None,
) -> pd.DataFrame:
    """Probe decoded frames from the first audio stream."""
    command = ["-select_streams", "a:0"]
    if read_intervals is not None:
        command += ["-read_intervals", read_intervals]
    command += [
        "-show_frames",
        "-show_entries",
        "frame=pts,pts_time,duration,duration_time,nb_samples",
    ]
    return pd.DataFrame(probe_json(path, command).get("frames", []))


def probe_audio_packets(path: Path) -> list[dict]:
    """Probe all packets from the first audio stream."""
    payload = probe_json(
        path,
        [
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_entries",
            (
                "packet=pts,pts_time,duration,duration_time:"
                "packet_side_data=side_data_type,skip_samples,discard_padding,"
                "skip_reason,discard_reason"
            ),
        ],
    )
    return payload.get("packets", [])


def extract_skip_samples(packets: list[dict]) -> pd.DataFrame:
    """Extract FFmpeg ``Skip Samples`` packet side data."""
    records: list[dict[str, object]] = []
    for packet_index, packet in enumerate(packets):
        for side_data in packet.get("side_data_list", []):
            if side_data.get("side_data_type") != "Skip Samples":
                continue
            records.append(
                {
                    "packet_index": packet_index,
                    "pts": packet.get("pts"),
                    "pts_time": packet.get("pts_time"),
                    "skip_samples": int(side_data.get("skip_samples", 0)),
                    "discard_padding": int(side_data.get("discard_padding", 0)),
                    "skip_reason": side_data.get("skip_reason"),
                    "discard_reason": side_data.get("discard_reason"),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_decoded_sample_counts(frames: pd.DataFrame) -> dict[str, object]:
    """Infer the nominal decoded frame size from observed ``nb_samples`` values."""
    samples = pd.to_numeric(
        frames.get("nb_samples", pd.Series(dtype=float)),
        errors="coerce",
    )
    valid = samples.dropna().astype(int)
    if valid.empty:
        return {
            "n_decoded_frames": len(frames),
            "n_valid_decoded_sample_counts": 0,
            "reference_decoded_samples": None,
            "reference_method": "mode_nb_samples",
            "decoded_sample_count_distribution": {},
            "n_960_sample_frames": 0,
        }

    counts = valid.value_counts().sort_index()
    max_count = int(counts.max())
    reference = int(counts[counts == max_count].index.min())
    distribution = {str(int(value)): int(count) for value, count in counts.items()}
    return {
        "n_decoded_frames": len(frames),
        "n_valid_decoded_sample_counts": len(valid),
        "reference_decoded_samples": reference,
        "reference_method": "mode_nb_samples",
        "decoded_sample_count_distribution": distribution,
        "n_non_reference_decoded_frames": int(valid.ne(reference).sum()),
        "n_960_sample_frames": int(valid.eq(960).sum()),
    }


def _ticks_to_samples(values: np.ndarray, samples_per_tick: Fraction) -> np.ndarray:
    """Convert integer time-base ticks to samples from an exact rational ratio."""
    return (
        values.astype(np.float64)
        * samples_per_tick.numerator
        / samples_per_tick.denominator
    )


def analyze_audio_timeline(
    frames: pd.DataFrame,
    *,
    sample_rate_hz: int,
    time_base: str,
    tolerance_samples: float = 1.0,
) -> dict[str, object]:
    """Measure decoded-frame PTS continuity against decoded sample counts."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    pts = pd.to_numeric(frames.get("pts", pd.Series(dtype=float)), errors="coerce")
    nb_samples = pd.to_numeric(
        frames.get("nb_samples", pd.Series(dtype=float)),
        errors="coerce",
    )
    valid_mask = pts.notna() & nb_samples.notna()
    valid_pts = pts[valid_mask].reset_index(drop=True)
    valid_samples = nb_samples[valid_mask].reset_index(drop=True)
    base = {
        "n_frames": len(frames),
        "n_valid_frames": int(valid_mask.sum()),
        "timestamp_coverage": float(pts.notna().mean()) if len(frames) else 0.0,
        "sample_count_coverage": (
            float(nb_samples.notna().mean()) if len(frames) else 0.0
        ),
    }
    if len(valid_pts) < 2:
        return {**base, "timeline_class": "insufficient_data"}

    samples_per_tick = Fraction(sample_rate_hz) * Fraction(time_base)
    pts_steps = np.diff(valid_pts.astype(np.int64).to_numpy())
    pts_step_samples = _ticks_to_samples(pts_steps, samples_per_tick)
    frame_samples = valid_samples.iloc[:-1].to_numpy(dtype=float)
    gap_samples = pts_step_samples - frame_samples
    n_gaps = int((gap_samples > tolerance_samples).sum())
    n_overlaps = int((gap_samples < -tolerance_samples).sum())
    continuous = n_gaps == 0 and n_overlaps == 0
    return {
        **base,
        "timeline_class": "continuous" if continuous else "gap_or_overlap",
        "first_pts": int(valid_pts.iloc[0]),
        "last_pts": int(valid_pts.iloc[-1]),
        "median_nb_samples": float(valid_samples.median()),
        "min_gap_samples": float(gap_samples.min()),
        "max_gap_samples": float(gap_samples.max()),
        "max_abs_gap_samples": float(np.abs(gap_samples).max()),
        "n_gaps": n_gaps,
        "n_overlaps": n_overlaps,
    }


def effective_audio_bounds(
    frames: pd.DataFrame,
    *,
    sample_rate_hz: int,
    time_base: str,
) -> tuple[float, float]:
    """Return first start and last end on the decoded audio PTS timeline."""
    pts = pd.to_numeric(frames.get("pts", pd.Series(dtype=float)), errors="coerce")
    nb_samples = pd.to_numeric(
        frames.get("nb_samples", pd.Series(dtype=float)),
        errors="coerce",
    )
    valid = pts.notna() & nb_samples.notna()
    if not valid.any():
        return np.nan, np.nan
    starts = pts[valid] * float(Fraction(time_base))
    ends = starts + nb_samples[valid] / sample_rate_hz
    return float(starts.min()), float(ends.max())


def _packet_summary_base(
    *,
    n_packets: int,
    pts: pd.Series,
    duration: pd.Series,
    samples_per_tick: Fraction,
) -> dict[str, object]:
    both = pts.notna() & duration.notna()
    valid_pts = pts.dropna().astype(np.int64).to_numpy()
    return {
        "n_packets": n_packets,
        "n_valid_packets": int(both.sum()),
        "timestamp_coverage": float(pts.notna().mean()) if n_packets else 0.0,
        "duration_coverage": float(duration.notna().mean()) if n_packets else 0.0,
        "timestamp_duration_coverage": (float(both.mean()) if n_packets else 0.0),
        "pts_monotonic": bool(np.all(np.diff(valid_pts) > 0))
        if len(valid_pts) > 1
        else False,
        "n_non_positive_pts_steps": int((np.diff(valid_pts) <= 0).sum())
        if len(valid_pts) > 1
        else 0,
        "samples_per_time_base_tick_numerator": samples_per_tick.numerator,
        "samples_per_time_base_tick_denominator": samples_per_tick.denominator,
    }


def _candidate_persistence(
    *,
    candidate_positions: np.ndarray,
    step_errors: np.ndarray,
    cumulative_drift: np.ndarray,
    window_steps: int,
    persistence_fraction: float,
) -> dict[str, np.ndarray]:
    """Test whether each positive PCM-clock step survives its compensation window.

    A candidate step raises the cumulative PCM-vs-PTS drift by ``excess``. The
    residual is the drift measured relative to the pre-step baseline over the next
    ``window_steps`` steps. If the residual never falls below
    ``persistence_fraction * excess`` inside the window, the discontinuity is
    persistent; otherwise later short steps compensate it locally, which is a
    timestamp cadence rather than missing audio.
    """
    n_steps = len(step_errors)
    excess = step_errors[candidate_positions]
    baseline = np.where(
        candidate_positions > 0,
        cumulative_drift[np.maximum(candidate_positions - 1, 0)],
        0.0,
    )
    window_end = np.minimum(candidate_positions + window_steps, n_steps - 1)
    min_residual = np.empty(len(candidate_positions))
    end_residual = np.empty(len(candidate_positions))
    for offset, (position, stop) in enumerate(
        zip(candidate_positions, window_end, strict=True)
    ):
        residual = cumulative_drift[position : stop + 1] - baseline[offset]
        min_residual[offset] = residual.min()
        end_residual[offset] = residual[-1]
    return {
        "excess": excess,
        "min_residual": min_residual,
        "end_residual": end_residual,
        "window_truncated": candidate_positions + window_steps > n_steps - 1,
        "persistent": min_residual >= persistence_fraction * excess,
    }


def _pcm_episode_events(
    *,
    step_packet_indices: np.ndarray,
    next_pts: np.ndarray,
    step_duration_samples: np.ndarray,
    step_errors: np.ndarray,
    cumulative_drift: np.ndarray,
    time_base: Fraction,
    threshold_samples: float,
    robust_nominal_duration_samples: float,
    long_duration_threshold_samples: float,
    reference_decoded_samples: int,
    sample_rate_hz: int,
    window_steps: int,
    persistence_fraction: float,
) -> list[dict[str, object]]:
    """Turn PCM-clock steps into diagnostic events.

    Consecutive non-zero steps form an episode. Every positive step above the
    drift threshold is a dropout *candidate*; a candidate whose excess persists
    beyond the compensation window becomes its own single-step ``audio_dropout``
    event and splits the surrounding episode. Remaining episodes are
    ``compensated_timestamp_cadence`` when they contain compensated candidates and
    ``pcm_timestamp_variation`` otherwise. Abnormally long packet durations are
    recorded as an observation on every event and never decide the category.
    """
    nonzero = np.flatnonzero(~np.isclose(step_errors, 0.0, atol=1e-12))
    if not len(nonzero):
        return []
    candidate_mask = step_errors > threshold_samples
    candidate_positions = np.flatnonzero(candidate_mask)
    persistence = _candidate_persistence(
        candidate_positions=candidate_positions,
        step_errors=step_errors,
        cumulative_drift=cumulative_drift,
        window_steps=window_steps,
        persistence_fraction=persistence_fraction,
    )
    persistent_mask = np.zeros(len(step_errors), dtype=bool)
    persistent_mask[candidate_positions] = persistence["persistent"]
    candidate_lookup = {
        int(position): offset for offset, position in enumerate(candidate_positions)
    }
    long_mask = np.isfinite(step_duration_samples) & (
        step_duration_samples > long_duration_threshold_samples
    )

    def base_event(first: int, last: int, group: np.ndarray) -> dict[str, object]:
        errors = step_errors[group]
        durations = step_duration_samples[group]
        candidates = group[candidate_mask[group]]
        return {
            "packet_index": int(step_packet_indices[first]),
            "next_packet_index": int(step_packet_indices[first] + 1),
            "event_pts": int(next_pts[first]),
            "event_time_sec": float(Fraction(int(next_pts[first])) * time_base),
            "event_end_time_sec": float(Fraction(int(next_pts[last])) * time_base),
            "n_steps": len(group),
            "magnitude_samples": float(np.abs(errors).max()),
            "packet_timeline_error_samples": np.nan,
            "first_pcm_step_error_samples": float(errors[0]),
            "max_abs_pcm_step_error_samples": float(np.abs(errors).max()),
            "net_pcm_drift_samples": float(errors.sum()),
            "robust_nominal_packet_duration_samples": robust_nominal_duration_samples,
            "long_packet_duration_threshold_samples": long_duration_threshold_samples,
            "n_abnormally_long_packets": int(long_mask[group].sum()),
            "max_packet_duration_samples": (
                float(np.nanmax(durations)) if np.isfinite(durations).any() else np.nan
            ),
            "n_candidate_steps": len(candidates),
            "n_compensated_candidate_steps": int((~persistent_mask[candidates]).sum()),
            "reference_decoded_samples": reference_decoded_samples,
        }

    events: list[dict[str, object]] = []
    split_points = np.flatnonzero(np.diff(step_packet_indices[nonzero]) != 1) + 1
    for group in np.split(nonzero, split_points):
        # Persistent candidates become single-step events and split the episode.
        boundaries = np.flatnonzero(persistent_mask[group])
        segments: list[tuple[str, np.ndarray]] = []
        cursor = 0
        for boundary in boundaries:
            if boundary > cursor:
                segments.append(("episode", group[cursor:boundary]))
            segments.append(("dropout", group[boundary : boundary + 1]))
            cursor = boundary + 1
        if cursor < len(group):
            segments.append(("episode", group[cursor:]))

        for kind, segment in segments:
            errors = step_errors[segment]
            first = int(segment[0])
            last = int(segment[-1])
            if kind == "dropout":
                offset = candidate_lookup[first]
                excess = float(persistence["excess"][offset])
                events.append(
                    {
                        **base_event(first, last, segment),
                        "event_type": "audio_dropout",
                        "event_category": "audio_dropout",
                        "event_direction": "gap",
                        "dropout_duration_samples": excess,
                        "dropout_duration_ms": 1000.0 * excess / sample_rate_hz,
                        "dropout_total_excess_samples": excess,
                        "dropout_total_excess_ms": 1000.0 * excess / sample_rate_hz,
                        "dropout_min_residual_in_window_samples": float(
                            persistence["min_residual"][offset]
                        ),
                        "dropout_residual_at_window_end_samples": float(
                            persistence["end_residual"][offset]
                        ),
                        "dropout_window_truncated": bool(
                            persistence["window_truncated"][offset]
                        ),
                    }
                )
                continue

            excursion = np.cumsum(errors)
            if (
                float(np.abs(errors).max()) <= threshold_samples
                and float(np.abs(excursion).max()) <= threshold_samples
            ):
                continue
            if np.all(errors > 0):
                direction = "gap"
            elif np.all(errors < 0):
                direction = "overlap"
            else:
                direction = "mixed"
            has_compensated_candidate = bool(candidate_mask[segment].any())
            events.append(
                {
                    **base_event(first, last, segment),
                    "event_type": "pcm_step_episode",
                    "event_category": (
                        "compensated_timestamp_cadence"
                        if has_compensated_candidate
                        else "pcm_timestamp_variation"
                    ),
                    "event_direction": direction,
                    "dropout_duration_samples": np.nan,
                    "dropout_duration_ms": np.nan,
                    "dropout_total_excess_samples": np.nan,
                    "dropout_total_excess_ms": np.nan,
                    "dropout_min_residual_in_window_samples": np.nan,
                    "dropout_residual_at_window_end_samples": np.nan,
                    "dropout_window_truncated": False,
                }
            )
    events.sort(key=lambda event: event["packet_index"])
    return events


def analyze_audio_packet_timeline(
    packets: list[dict],
    *,
    sample_rate_hz: int,
    time_base: str,
    reference_decoded_samples: int,
    drift_threshold_ms: float = 1.0,
    compensation_window_sec: float = 2.0,
    persistence_fraction: float = 0.5,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Measure packet timing, duration variation, and PCM-vs-PTS clock offset.

    Packet-clock errors compare each PTS step with the current packet duration.
    PCM-clock errors compare each PTS step with the independently established
    decoded sample count. Calculations start from integer PTS and an exact
    ``Fraction(time_base)``; floats are used only in report values.

    Every positive PCM-clock step strictly greater than ``drift_threshold_ms`` is
    a dropout candidate. It is reported as ``audio_dropout`` only when the
    cumulative drift stays at or above ``persistence_fraction`` of the step
    excess for ``compensation_window_sec`` after the step. Candidates that are
    compensated inside that window are timestamp cadence, not missing audio.
    All threshold comparisons are strict (``>``).
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if reference_decoded_samples <= 0:
        raise ValueError("reference_decoded_samples must be positive.")
    if drift_threshold_ms <= 0:
        raise ValueError("drift_threshold_ms must be positive.")
    if compensation_window_sec <= 0:
        raise ValueError("compensation_window_sec must be positive.")
    if not 0 < persistence_fraction <= 1:
        raise ValueError("persistence_fraction must be in (0, 1].")

    n_packets = len(packets)
    pts = pd.to_numeric(
        pd.Series([packet.get("pts") for packet in packets], dtype=object),
        errors="coerce",
    )
    duration = pd.to_numeric(
        pd.Series([packet.get("duration") for packet in packets], dtype=object),
        errors="coerce",
    )
    time_base_fraction = Fraction(time_base)
    samples_per_tick = Fraction(sample_rate_hz) * time_base_fraction
    summary = _packet_summary_base(
        n_packets=n_packets,
        pts=pts,
        duration=duration,
        samples_per_tick=samples_per_tick,
    )
    threshold_samples = sample_rate_hz * drift_threshold_ms / 1000.0
    window_steps = max(
        1,
        int(
            np.ceil(
                compensation_window_sec * sample_rate_hz / reference_decoded_samples
            )
        ),
    )
    summary.update(
        {
            "reference_decoded_samples": reference_decoded_samples,
            "drift_threshold_ms": drift_threshold_ms,
            "drift_threshold_samples": threshold_samples,
            "threshold_comparison": "strictly_greater",
            "compensation_window_sec": compensation_window_sec,
            "compensation_window_steps": window_steps,
            "persistence_fraction": persistence_fraction,
        }
    )
    if n_packets < 2:
        return {**summary, "timeline_status": "insufficient_data"}, pd.DataFrame()

    pts_array = pts.fillna(0).astype(np.int64).to_numpy()
    duration_array = duration.fillna(0).astype(np.int64).to_numpy()
    adjacent_pts = pts.notna().to_numpy()[:-1] & pts.notna().to_numpy()[1:]
    step_packet_indices = np.flatnonzero(adjacent_pts)
    pts_steps_ticks = (
        pts_array[step_packet_indices + 1] - pts_array[step_packet_indices]
    )
    pts_step_samples = _ticks_to_samples(pts_steps_ticks, samples_per_tick)
    next_pts = pts_array[step_packet_indices + 1]

    packet_step_mask = duration.notna().to_numpy()[:-1][step_packet_indices]
    packet_indices = step_packet_indices[packet_step_mask]
    packet_error_ticks = (
        pts_array[packet_indices + 1]
        - pts_array[packet_indices]
        - duration_array[packet_indices]
    )
    packet_errors = _ticks_to_samples(packet_error_ticks, samples_per_tick)
    packet_gap_mask = packet_errors > 0
    packet_overlap_mask = packet_errors < 0

    pcm_step_errors = pts_step_samples - reference_decoded_samples
    cumulative_drift = np.cumsum(pcm_step_errors)
    abs_cumulative = np.abs(cumulative_drift)
    max_cumulative_position = int(abs_cumulative.argmax()) if len(abs_cumulative) else 0
    final_drift = float(cumulative_drift[-1]) if len(cumulative_drift) else np.nan
    max_abs_cumulative = (
        float(abs_cumulative[max_cumulative_position])
        if len(abs_cumulative)
        else np.nan
    )

    valid_durations = duration.dropna().astype(np.int64).to_numpy()
    duration_samples = _ticks_to_samples(valid_durations, samples_per_tick)
    if len(valid_durations):
        values, counts = np.unique(valid_durations, return_counts=True)
        nominal_ticks = int(values[counts.argmax()])
        nominal_duration = float(Fraction(nominal_ticks) * samples_per_tick)
        duration_deviation = duration_samples - reference_decoded_samples
        robust_nominal_duration = float(np.median(duration_samples))
        duration_mad = float(
            np.median(np.abs(duration_samples - robust_nominal_duration))
        )
        long_duration_threshold = max(
            robust_nominal_duration
            + max(6.0 * duration_mad, 0.5 * robust_nominal_duration),
            reference_decoded_samples + threshold_samples,
        )
        duration_metrics = {
            "nominal_packet_duration_samples": nominal_duration,
            "robust_nominal_packet_duration_samples": robust_nominal_duration,
            "packet_duration_mad_samples": duration_mad,
            "long_packet_duration_threshold_samples": long_duration_threshold,
            "min_packet_duration_samples": float(duration_samples.min()),
            "median_packet_duration_samples": float(np.median(duration_samples)),
            "max_packet_duration_samples": float(duration_samples.max()),
            "n_distinct_packet_durations": len(values),
            "n_packet_duration_variations": int(
                (~np.isclose(duration_deviation, 0.0, atol=1e-12)).sum()
            ),
            "max_abs_packet_duration_deviation_samples": float(
                np.abs(duration_deviation).max()
            ),
            "n_abnormally_long_packet_durations": int(
                (duration_samples > long_duration_threshold).sum()
            ),
        }
    else:
        robust_nominal_duration = np.nan
        long_duration_threshold = np.nan
        duration_metrics = {
            "nominal_packet_duration_samples": np.nan,
            "robust_nominal_packet_duration_samples": np.nan,
            "packet_duration_mad_samples": np.nan,
            "long_packet_duration_threshold_samples": np.nan,
            "min_packet_duration_samples": np.nan,
            "median_packet_duration_samples": np.nan,
            "max_packet_duration_samples": np.nan,
            "n_distinct_packet_durations": 0,
            "n_packet_duration_variations": 0,
            "max_abs_packet_duration_deviation_samples": np.nan,
            "n_abnormally_long_packet_durations": 0,
        }

    skip_data = extract_skip_samples(packets)
    skip_samples = int(skip_data["skip_samples"].sum()) if not skip_data.empty else 0
    discard_padding = (
        int(skip_data["discard_padding"].sum()) if not skip_data.empty else 0
    )
    n_pcm_variations = int((~np.isclose(pcm_step_errors, 0.0, atol=1e-12)).sum())
    candidate_positions = np.flatnonzero(pcm_step_errors > threshold_samples)
    persistence = _candidate_persistence(
        candidate_positions=candidate_positions,
        step_errors=pcm_step_errors,
        cumulative_drift=cumulative_drift,
        window_steps=window_steps,
        persistence_fraction=persistence_fraction,
    )
    n_persistent = int(persistence["persistent"].sum())
    max_abs_pcm_step = (
        float(np.abs(pcm_step_errors).max()) if len(pcm_step_errors) else np.nan
    )
    summary.update(
        {
            "timeline_status": "measured",
            "n_valid_pts_steps": len(pts_step_samples),
            "n_valid_packet_timeline_steps": len(packet_errors),
            "packet_timeline_gap_count": int(packet_gap_mask.sum()),
            "packet_timeline_overlap_count": int(packet_overlap_mask.sum()),
            "packet_max_abs_timeline_error_samples": (
                float(np.abs(packet_errors).max()) if len(packet_errors) else np.nan
            ),
            **duration_metrics,
            "n_pcm_step_variations": n_pcm_variations,
            "max_abs_pcm_step_error_samples": max_abs_pcm_step,
            "pcm_step_error_p95_abs_samples": (
                float(np.quantile(np.abs(pcm_step_errors), 0.95))
                if len(pcm_step_errors)
                else np.nan
            ),
            "final_cumulative_pcm_drift_samples": final_drift,
            "final_cumulative_pcm_drift_ms": (1000.0 * final_drift / sample_rate_hz),
            "max_abs_cumulative_pcm_drift_samples": max_abs_cumulative,
            "max_abs_cumulative_pcm_drift_ms": (
                1000.0 * max_abs_cumulative / sample_rate_hz
            ),
            "max_abs_cumulative_pcm_drift_time_sec": (
                float(
                    Fraction(int(next_pts[max_cumulative_position]))
                    * time_base_fraction
                )
                if len(next_pts)
                else np.nan
            ),
            "has_timestamp_jitter": n_pcm_variations > 0,
            "has_small_bounded_timestamp_jitter": bool(
                n_pcm_variations > 0
                and max_abs_pcm_step <= threshold_samples
                and max_abs_cumulative <= threshold_samples
            ),
            "has_significant_transient_pcm_offset": bool(
                max_abs_cumulative > threshold_samples
            ),
            "has_significant_final_pcm_offset": bool(
                abs(final_drift) > threshold_samples
            ),
            "n_dropout_candidate_steps": len(candidate_positions),
            "n_persistent_dropout_steps": n_persistent,
            "n_compensated_candidate_steps": len(candidate_positions) - n_persistent,
            "skip_samples": skip_samples,
            "discard_padding_samples": discard_padding,
        }
    )

    events: list[dict[str, object]] = []
    for position in np.flatnonzero(packet_errors != 0):
        error = float(packet_errors[position])
        index = int(packet_indices[position])
        event_pts = int(pts_array[index + 1])
        events.append(
            {
                "packet_index": index,
                "next_packet_index": index + 1,
                "event_type": (
                    "packet_timeline_gap" if error > 0 else "packet_timeline_overlap"
                ),
                "event_category": "packet_timeline_error",
                "event_direction": "gap" if error > 0 else "overlap",
                "event_pts": event_pts,
                "event_time_sec": float(Fraction(event_pts) * time_base_fraction),
                "event_end_time_sec": float(Fraction(event_pts) * time_base_fraction),
                "n_steps": 1,
                "magnitude_samples": abs(error),
                "packet_timeline_error_samples": error,
                "first_pcm_step_error_samples": np.nan,
                "max_abs_pcm_step_error_samples": np.nan,
                "net_pcm_drift_samples": np.nan,
                "robust_nominal_packet_duration_samples": (robust_nominal_duration),
                "long_packet_duration_threshold_samples": (long_duration_threshold),
                "n_abnormally_long_packets": 0,
                "max_packet_duration_samples": np.nan,
                "n_candidate_steps": 0,
                "n_compensated_candidate_steps": 0,
                "dropout_duration_samples": np.nan,
                "dropout_duration_ms": np.nan,
                "dropout_total_excess_samples": np.nan,
                "dropout_total_excess_ms": np.nan,
                "dropout_min_residual_in_window_samples": np.nan,
                "dropout_residual_at_window_end_samples": np.nan,
                "dropout_window_truncated": False,
                "reference_decoded_samples": reference_decoded_samples,
            }
        )
    step_duration_samples = np.full(len(step_packet_indices), np.nan)
    valid_step_durations = duration.notna().to_numpy()[step_packet_indices]
    step_duration_samples[valid_step_durations] = _ticks_to_samples(
        duration_array[step_packet_indices[valid_step_durations]],
        samples_per_tick,
    )
    events.extend(
        _pcm_episode_events(
            step_packet_indices=step_packet_indices,
            next_pts=next_pts,
            step_duration_samples=step_duration_samples,
            step_errors=pcm_step_errors,
            cumulative_drift=cumulative_drift,
            time_base=time_base_fraction,
            threshold_samples=threshold_samples,
            robust_nominal_duration_samples=robust_nominal_duration,
            long_duration_threshold_samples=long_duration_threshold,
            reference_decoded_samples=reference_decoded_samples,
            sample_rate_hz=sample_rate_hz,
            window_steps=window_steps,
            persistence_fraction=persistence_fraction,
        )
    )
    return summary, pd.DataFrame.from_records(events)
