"""Decoded-frame evidence: nominal frame size, continuity and bounds.

Decoding is the ground truth for how many PCM samples a packet yields. It is
slow, so it is applied to short windows selected from the packet-level audit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from conv_wm.data.media.audio.packets import ticks_to_samples
from conv_wm.data.media.audio.probe import probe_audio_frames
from conv_wm.data.media.audio.records import DecodedValidationRecord
from conv_wm.data.media.audio.summary import regime


@dataclass(frozen=True)
class DecodedSampleCounts:
    """Distribution of ``nb_samples`` over decoded frames."""

    n_decoded_frames: int
    n_valid_decoded_sample_counts: int
    reference_decoded_samples: int | None
    """Modal frame size (smallest mode on ties), ``None`` without valid frames."""
    distribution: dict[str, int]
    n_non_reference_decoded_frames: int
    n_960_sample_frames: int
    """AAC-LD/ELD frames are 960 samples; their presence changes the reference."""

    reference_method: str = "mode_nb_samples"


def summarize_decoded_sample_counts(frames: pd.DataFrame) -> DecodedSampleCounts:
    """Infer the nominal decoded frame size from observed ``nb_samples``."""
    samples = pd.to_numeric(
        frames.get("nb_samples", pd.Series(dtype=float)), errors="coerce"
    )
    valid = samples.dropna().astype(int)
    if valid.empty:
        return DecodedSampleCounts(len(frames), 0, None, {}, 0, 0)
    counts = valid.value_counts().sort_index()
    reference = int(counts[counts == int(counts.max())].index.min())
    return DecodedSampleCounts(
        n_decoded_frames=len(frames),
        n_valid_decoded_sample_counts=len(valid),
        reference_decoded_samples=reference,
        distribution={str(int(value)): int(count) for value, count in counts.items()},  # type: ignore[call-overload]
        n_non_reference_decoded_frames=int(valid.ne(reference).sum()),
        n_960_sample_frames=int(valid.eq(960).sum()),
    )


def decoded_frame_continuity(
    frames: pd.DataFrame,
    *,
    sample_rate_hz: int,
    time_base: str,
    tolerance_samples: float = 1.0,
) -> dict[str, object]:
    """Compare decoded-frame PTS steps with each frame's own sample count.

    Returns ``timeline_class`` ``continuous`` / ``gap_or_overlap`` /
    ``insufficient_data`` plus gap statistics in samples. This is the local,
    decoded counterpart of the packet-level PCM clock and is used only on short
    validation windows.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    pts = pd.to_numeric(frames.get("pts", pd.Series(dtype=float)), errors="coerce")
    nb_samples = pd.to_numeric(
        frames.get("nb_samples", pd.Series(dtype=float)), errors="coerce"
    )
    valid_mask = pts.notna() & nb_samples.notna()
    valid_pts = pts[valid_mask].reset_index(drop=True)
    valid_samples = nb_samples[valid_mask].reset_index(drop=True)
    base: dict[str, object] = {
        "n_frames": len(frames),
        "n_valid_frames": int(valid_mask.sum()),
        "timestamp_coverage": float(pts.notna().mean()) if len(frames) else 0.0,
        "sample_count_coverage": float(nb_samples.notna().mean())
        if len(frames)
        else 0.0,
    }
    if len(valid_pts) < 2:
        return {**base, "timeline_class": "insufficient_data"}
    samples_per_tick = Fraction(sample_rate_hz) * Fraction(time_base)
    pts_steps = np.diff(valid_pts.astype(np.int64).to_numpy())
    gap_samples = ticks_to_samples(pts_steps, samples_per_tick) - valid_samples.iloc[
        :-1
    ].to_numpy(dtype=float)
    n_gaps = int((gap_samples > tolerance_samples).sum())
    n_overlaps = int((gap_samples < -tolerance_samples).sum())
    return {
        **base,
        "timeline_class": "continuous"
        if n_gaps == 0 and n_overlaps == 0
        else "gap_or_overlap",
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
    """Return first start and last end (seconds) on the decoded audio PTS timeline."""
    pts = pd.to_numeric(frames.get("pts", pd.Series(dtype=float)), errors="coerce")
    nb_samples = pd.to_numeric(
        frames.get("nb_samples", pd.Series(dtype=float)), errors="coerce"
    )
    valid = pts.notna() & nb_samples.notna()
    if not valid.any():
        return np.nan, np.nan
    starts = pts[valid] * float(Fraction(time_base))
    ends = starts + nb_samples[valid] / sample_rate_hz
    return float(starts.min()), float(ends.max())


def decode_window_interval(
    event_time_sec: float | None, *, half_width_sec: float = 1.0
) -> str:
    """ffprobe ``-read_intervals`` spec around ``event_time_sec`` (or the stream start)."""
    if event_time_sec is None or pd.isna(event_time_sec):
        return f"%+{half_width_sec:.6f}"
    start = max(0.0, float(event_time_sec) - half_width_sec)
    return f"{start:.6f}%+{2 * half_width_sec:.6f}"


@dataclass(frozen=True)
class DecodeValidationCase:
    """A short window selected for decoded validation."""

    case: str
    selection_reason: str
    dataset: str
    relative_path: str
    event_time_sec: float | None


def validate_decoded_window(
    case: DecodeValidationCase,
    *,
    path: Path,
    file_row: pd.Series,
) -> DecodedValidationRecord:
    """Decode one window and compare its frame sizes with the packet-level reference.

    ``file_row`` is the file's row of the packet timeline table; the record
    repeats the packet-level context so the validation table is self-contained.
    """
    interval = decode_window_interval(case.event_time_sec)
    base = {
        "case": case.case,
        "selection_reason": case.selection_reason,
        "dataset": case.dataset,
        "relative_path": case.relative_path,
        "requested_interval": interval,
        "event_time_sec": case.event_time_sec,
        "audio_codec": str(file_row["audio_codec"]),
        "sample_rate_hz": int(file_row["sample_rate_hz"]),
        "time_base": str(file_row["time_base"]),
        "packet_nominal_duration_samples": float(
            file_row["nominal_packet_duration_samples"]
        ),
        "robust_nominal_packet_duration_samples": float(
            file_row["robust_nominal_packet_duration_samples"]
        ),
        "long_packet_duration_threshold_samples": float(
            file_row["long_packet_duration_threshold_samples"]
        ),
        "packet_max_abs_duration_deviation_samples": float(
            file_row["max_abs_packet_duration_deviation_samples"]
        ),
        "skip_samples": int(file_row["skip_samples"]),
        "discard_padding_samples": int(file_row["discard_padding_samples"]),
    }
    try:
        frames = probe_audio_frames(path, read_intervals=interval)
        decoded = summarize_decoded_sample_counts(frames)
        continuity = decoded_frame_continuity(
            frames,
            sample_rate_hz=int(file_row["sample_rate_hz"]),
            time_base=str(file_row["time_base"]),
        )
    except Exception as exc:  # noqa: BLE001 - one bad window must not abort the audit
        return DecodedValidationRecord(**base, probe_ok=False, probe_error=str(exc))
    return DecodedValidationRecord(
        **base,
        probe_ok=True,
        probe_error=None,
        n_decoded_frames=decoded.n_decoded_frames,
        n_valid_decoded_sample_counts=decoded.n_valid_decoded_sample_counts,
        reference_decoded_samples=decoded.reference_decoded_samples,
        reference_method=decoded.reference_method,
        n_non_reference_decoded_frames=decoded.n_non_reference_decoded_frames,
        n_960_sample_frames=decoded.n_960_sample_frames,
        decoded_sample_count_distribution_json=json.dumps(
            decoded.distribution, sort_keys=True
        ),
        reference_matches_mode=(
            decoded.reference_decoded_samples
            == int(file_row["reference_decoded_samples"])
        ),
        decoded_pts_max_abs_step_error_samples=float(
            continuity.get("max_abs_gap_samples", np.nan)  # type: ignore[arg-type]
        ),
    )


def _first_row(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[0]


def select_decode_validation_cases(
    files: pd.DataFrame,
    events: pd.DataFrame,
    *,
    extra_cases: list[DecodeValidationCase] | None = None,
) -> list[DecodeValidationCase]:
    """Choose short windows whose decoded frames validate the packet-level reading.

    Generic selection covers, deterministically: the start and end of the
    quietest file of every observed dataset/sample-rate regime, the file with the
    most small bounded variations, the largest event of each interpreted category
    (known boundary, codec boundary, compensated cadence), the longest dropout per
    regime and the two largest cumulative offsets. Datasets may add
    ``extra_cases`` (for instance a systematic profile of a regime they know).
    """
    valid = files.loc[files["probe_ok"]]
    cases: list[DecodeValidationCase] = []

    def add(
        name: str, reason: str, row: pd.Series, event_time: float | None = None
    ) -> None:
        cases.append(
            DecodeValidationCase(
                case=name,
                selection_reason=reason,
                dataset=str(row["dataset"]),
                relative_path=str(row["relative_path"]),
                event_time_sec=event_time,
            )
        )

    for key, subset in valid.groupby(["dataset", "sample_rate_hz"]):
        dataset, rate = regime(key)
        ordered = subset.sort_values(
            [
                "max_abs_cumulative_pcm_drift_samples",
                "max_abs_pcm_step_error_samples",
                "relative_path",
            ]
        )
        row = _first_row(ordered)
        name = f"{dataset}_{rate}_regime"
        add(name, "dataset_sample_rate_regime", row)
        add(
            f"{name}_end",
            "dataset_sample_rate_end_boundary",
            row,
            float(row["audio_duration_sec"]),
        )

    jitter = valid.loc[valid["has_small_bounded_timestamp_jitter"]].sort_values(
        ["n_pcm_step_variations", "relative_path"], ascending=[False, True]
    )
    if not jitter.empty:
        row = _first_row(jitter)
        add(
            "small_bounded_jitter",
            "largest_count_of_small_bounded_pts_variations",
            row,
            float(row["max_abs_cumulative_pcm_drift_time_sec"]),
        )

    if not events.empty:
        for category in (
            "stitch_related_pcm_discontinuity",
            "codec_boundary_pcm_event",
        ):
            candidates = events.loc[events["event_category"].eq(category)].sort_values(
                ["magnitude_samples", "relative_path", "event_time_sec"],
                ascending=[False, True, True],
            )
            if not candidates.empty:
                event = candidates.iloc[0]
                row = _first_row(
                    valid.loc[valid["relative_path"].eq(event["relative_path"])]
                )
                add(category, category, row, float(event["event_time_sec"]))
        for category, sort_key in (
            ("compensated_timestamp_cadence", "max_abs_pcm_step_error_samples"),
            ("audio_dropout", "dropout_duration_ms"),
        ):
            subset = events.loc[events["event_category"].eq(category)]
            for (dataset, rate), candidates in subset.groupby(
                ["dataset", "sample_rate_hz"]
            ):
                event = candidates.sort_values(
                    [sort_key, "relative_path", "event_time_sec"],
                    ascending=[False, True, True],
                ).iloc[0]
                row = _first_row(
                    valid.loc[valid["relative_path"].eq(event["relative_path"])]
                )
                add(
                    f"{category}_{dataset}_{rate}",
                    category,
                    row,
                    float(event["event_time_sec"]),
                )

    largest = valid.sort_values(
        ["max_abs_cumulative_pcm_drift_samples", "relative_path"],
        ascending=[False, True],
    ).head(2)
    for index, (_, row) in enumerate(largest.iterrows(), start=1):
        add(
            f"largest_pcm_offset_{index}",
            "population_max_abs_cumulative_pcm_drift",
            row,
            float(row["max_abs_cumulative_pcm_drift_time_sec"]),
        )
    cases.extend(extra_cases or [])
    return cases
