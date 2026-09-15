"""Population runner for the authoritative B3 audio timeline audit."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from conv_wm.config import load_config
from conv_wm.data.media.audio_timeline import (
    analyze_audio_packet_timeline,
    analyze_audio_timeline,
    probe_audio_frames,
    probe_audio_packets,
    summarize_decoded_sample_counts,
)
from conv_wm.reports import write_summary

DRIFT_THRESHOLD_MS = 1.0
# A positive PCM-clock step is an audio dropout only if its excess persists
# (>= PERSISTENCE_FRACTION of the step) through the next COMPENSATION_WINDOW_SEC.
COMPENSATION_WINDOW_SEC = 2.0
PERSISTENCE_FRACTION = 0.5
# Every threshold comparison in this runner is strict: value > threshold.
THRESHOLD_COMPARISON = "strictly_greater"
EGO4D_STITCH_PERIOD_SEC = 300.0
STITCH_TOLERANCE_PACKETS = 2.0
EARLY_BOUNDARY_EXTRA_FRAMES = 2
MAX_WORKERS = 4

# The population metadata contains AAC only. Targeted ``-show_frames`` validation
# across every dataset/sample-rate regime establishes 1024 as the decoded-frame
# mode; the runner emits that validation alongside the packet audit.
DECODED_SAMPLE_REFERENCES = {
    "aac": {
        "samples": 1024,
        "source": "targeted_show_frames_mode",
    }
}

EVENT_COLUMNS = [
    "dataset",
    "relative_path",
    "audio_codec",
    "sample_rate_hz",
    "packet_index",
    "next_packet_index",
    "event_type",
    "event_category",
    "event_direction",
    "event_pts",
    "event_time_sec",
    "event_end_time_sec",
    "n_steps",
    "magnitude_samples",
    "packet_timeline_error_samples",
    "first_pcm_step_error_samples",
    "max_abs_pcm_step_error_samples",
    "net_pcm_drift_samples",
    "robust_nominal_packet_duration_samples",
    "long_packet_duration_threshold_samples",
    "n_abnormally_long_packets",
    "max_packet_duration_samples",
    "n_candidate_steps",
    "n_compensated_candidate_steps",
    "dropout_duration_samples",
    "dropout_duration_ms",
    "dropout_total_excess_samples",
    "dropout_total_excess_ms",
    "dropout_min_residual_in_window_samples",
    "dropout_residual_at_window_end_samples",
    "dropout_window_truncated",
    "reference_decoded_samples",
    "nearest_stitch_index",
    "nearest_stitch_time_sec",
    "distance_to_stitch_sec",
    "stitch_tolerance_sec",
    "near_known_stitch_boundary",
    "early_boundary_tolerance_sec",
    "has_early_boundary_context",
]


def audit_parameters() -> dict[str, object]:
    """Settings that materially affect how the audio timeline report is read."""
    return {
        "drift_threshold_ms": DRIFT_THRESHOLD_MS,
        "threshold_comparison": THRESHOLD_COMPARISON,
        "compensation_window_sec": COMPENSATION_WINDOW_SEC,
        "persistence_fraction": PERSISTENCE_FRACTION,
        "decoded_sample_references": DECODED_SAMPLE_REFERENCES,
    }


def _decoded_sample_reference(codec: str) -> tuple[int, str]:
    try:
        reference = DECODED_SAMPLE_REFERENCES[codec]
    except KeyError as exc:
        raise ValueError(
            f"No validated decoded-sample reference for audio codec: {codec}"
        ) from exc
    return int(reference["samples"]), str(reference["source"])


def final_drift_threshold_flags(
    final_drift_ms: float,
    *,
    video_fps: float,
) -> dict[str, float | bool]:
    """Return file-level final-drift flags; every comparison is strict (``>``).

    A file whose absolute final drift equals a threshold exactly is *not* counted
    above it.
    """
    if video_fps <= 0:
        raise ValueError("video_fps must be positive")
    absolute_ms = abs(final_drift_ms)
    half_video_frame_ms = 500.0 / video_fps
    return {
        "abs_final_cumulative_pcm_drift_ms": absolute_ms,
        "half_video_frame_ms": half_video_frame_ms,
        "final_pcm_drift_exceeds_1_ms": absolute_ms > 1.0,
        "final_pcm_drift_exceeds_half_video_frame": (absolute_ms > half_video_frame_ms),
        "final_pcm_drift_exceeds_100_ms": absolute_ms > 100.0,
        "final_pcm_drift_exceeds_500_ms": absolute_ms > 500.0,
    }


def annotate_audio_events(
    events: pd.DataFrame,
    *,
    dataset: str,
    sample_rate_hz: int,
    reference_decoded_samples: int,
    skip_samples: int,
) -> pd.DataFrame:
    """Add codec-boundary and dataset-specific stitch interpretation."""
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    result = events.copy()
    if "event_category" not in result:
        result["event_category"] = "pcm_timestamp_variation"
    event_time = pd.to_numeric(result["event_time_sec"], errors="coerce")
    event_end_time = pd.to_numeric(
        result.get("event_end_time_sec", event_time),
        errors="coerce",
    )
    packet_duration_sec = reference_decoded_samples / sample_rate_hz
    stitch_tolerance_sec = STITCH_TOLERANCE_PACKETS * packet_duration_sec

    if dataset == "ego4d":
        stitch_index = np.maximum(
            np.rint(event_time / EGO4D_STITCH_PERIOD_SEC),
            1,
        ).astype(int)
        stitch_time = stitch_index * EGO4D_STITCH_PERIOD_SEC
        stitch_distance = (event_time - stitch_time).abs()
        result["nearest_stitch_index"] = stitch_index
        result["nearest_stitch_time_sec"] = stitch_time
        result["distance_to_stitch_sec"] = stitch_distance
        result["near_known_stitch_boundary"] = stitch_distance <= stitch_tolerance_sec
    else:
        result["nearest_stitch_index"] = pd.NA
        result["nearest_stitch_time_sec"] = np.nan
        result["distance_to_stitch_sec"] = np.nan
        result["near_known_stitch_boundary"] = False

    early_boundary_tolerance_sec = (
        skip_samples + EARLY_BOUNDARY_EXTRA_FRAMES * reference_decoded_samples
    ) / sample_rate_hz
    result["stitch_tolerance_sec"] = stitch_tolerance_sec
    result["early_boundary_tolerance_sec"] = early_boundary_tolerance_sec
    result["has_early_boundary_context"] = event_time.ge(0) & event_time.le(
        early_boundary_tolerance_sec
    )

    is_packet_error = result["event_type"].str.startswith("packet_timeline_")
    is_pcm_event = result["event_type"].isin(["pcm_step_episode", "audio_dropout"])
    # A persistent positive step inside the AAC priming context has a known codec
    # cause, so the codec-boundary label takes precedence over ``audio_dropout``.
    is_early = (
        is_pcm_event
        & result["has_early_boundary_context"]
        & event_end_time.le(early_boundary_tolerance_sec)
        & (skip_samples > 0)
    )
    is_dropout = result["event_category"].eq("audio_dropout") & ~is_early
    is_stitch = (
        result["event_type"].eq("pcm_step_episode")
        & result["near_known_stitch_boundary"]
        & ~is_dropout
        & ~is_early
    )
    result.loc[is_early, "event_category"] = "codec_boundary_pcm_event"
    result.loc[is_stitch, "event_category"] = "stitch_related_pcm_discontinuity"
    result.loc[is_packet_error, "event_category"] = "packet_timeline_error"
    return result.reindex(columns=EVENT_COLUMNS)


def audit_audio_file(row, *, raw_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    """Run a full packet scan for one media file without modifying it."""
    path = raw_root / row.relative_path
    sample_rate_hz = int(row.audio_sample_rate_hz)
    codec = str(row.audio_codec)
    time_base = str(row.audio_time_base)
    reference_samples, reference_source = _decoded_sample_reference(codec)
    base = {
        "dataset": row.dataset,
        "relative_path": row.relative_path,
        "audio_codec": codec,
        "sample_rate_hz": sample_rate_hz,
        "time_base": time_base,
        "audio_duration_sec": float(row.audio_duration_sec),
        "video_avg_frame_rate": float(row.video_avg_frame_rate),
        "reference_decoded_samples_source": reference_source,
    }
    try:
        packets = probe_audio_packets(path)
        summary, events = analyze_audio_packet_timeline(
            packets,
            sample_rate_hz=sample_rate_hz,
            time_base=time_base,
            reference_decoded_samples=reference_samples,
            drift_threshold_ms=DRIFT_THRESHOLD_MS,
            compensation_window_sec=COMPENSATION_WINDOW_SEC,
            persistence_fraction=PERSISTENCE_FRACTION,
        )
        events = events.assign(
            dataset=row.dataset,
            relative_path=row.relative_path,
            audio_codec=codec,
            sample_rate_hz=sample_rate_hz,
        )
        events = annotate_audio_events(
            events,
            dataset=row.dataset,
            sample_rate_hz=sample_rate_hz,
            reference_decoded_samples=reference_samples,
            skip_samples=int(summary["skip_samples"]),
        )
        dropout_events = events.loc[events["event_category"].eq("audio_dropout")]
        final_drift_flags = final_drift_threshold_flags(
            float(summary["final_cumulative_pcm_drift_ms"]),
            video_fps=float(row.video_avg_frame_rate),
        )
        file_record = {
            **base,
            "probe_ok": True,
            "probe_error": None,
            **summary,
            "has_stitch_related_event": bool(
                events["event_category"].eq("stitch_related_pcm_discontinuity").any()
            ),
            "has_early_boundary_event": bool(
                events["event_category"].eq("codec_boundary_pcm_event").any()
            ),
            "has_audio_dropout": not dropout_events.empty,
            "audio_dropout_event_count": len(dropout_events),
            "audio_dropout_packet_count": int(dropout_events["n_steps"].sum()),
            "has_compensated_timestamp_cadence": bool(
                events["event_category"].eq("compensated_timestamp_cadence").any()
            ),
            "max_audio_dropout_duration_ms": (
                float(dropout_events["dropout_duration_ms"].max())
                if not dropout_events.empty
                else 0.0
            ),
            "total_audio_dropout_excess_ms": (
                float(dropout_events["dropout_total_excess_ms"].sum())
                if not dropout_events.empty
                else 0.0
            ),
            **final_drift_flags,
        }
        return file_record, events
    except Exception as exc:  # noqa: BLE001
        # Population boundary: one corrupt file must not abort all 369 probes.
        return (
            {
                **base,
                "probe_ok": False,
                "probe_error": str(exc),
                "n_packets": 0,
                "n_valid_packets": 0,
                "has_timestamp_jitter": False,
                "has_small_bounded_timestamp_jitter": False,
                "has_significant_transient_pcm_offset": False,
                "has_significant_final_pcm_offset": False,
                "has_stitch_related_event": False,
                "has_early_boundary_event": False,
                "has_audio_dropout": False,
                "audio_dropout_event_count": 0,
                "audio_dropout_packet_count": 0,
                "has_compensated_timestamp_cadence": False,
                "n_dropout_candidate_steps": 0,
                "n_persistent_dropout_steps": 0,
                "n_compensated_candidate_steps": 0,
                "n_abnormally_long_packet_durations": 0,
                "max_audio_dropout_duration_ms": 0.0,
                "total_audio_dropout_excess_ms": 0.0,
                "abs_final_cumulative_pcm_drift_ms": np.nan,
                "half_video_frame_ms": 500.0 / float(row.video_avg_frame_rate),
                "final_pcm_drift_exceeds_1_ms": False,
                "final_pcm_drift_exceeds_half_video_frame": False,
                "final_pcm_drift_exceeds_100_ms": False,
                "final_pcm_drift_exceeds_500_ms": False,
                "skip_samples": 0,
                "discard_padding_samples": 0,
            },
            pd.DataFrame(columns=EVENT_COLUMNS),
        )


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def _quantiles(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        name: float(values.quantile(quantile))
        for name, quantile in [
            ("min", 0.0),
            ("p05", 0.05),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p95", 0.95),
            ("max", 1.0),
        ]
    }


def build_dataset_sample_rate_summary(
    files: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Summarize final PCM drift and dropout burden at the requested grain."""
    result: dict[str, dict[str, object]] = {}
    for (dataset, sample_rate), subset in files.groupby(["dataset", "sample_rate_hz"]):
        valid = subset.loc[subset["probe_ok"]]
        dropouts = events.loc[
            events["dataset"].eq(dataset)
            & events["sample_rate_hz"].eq(sample_rate)
            & events["event_category"].eq("audio_dropout")
        ]
        result.setdefault(str(dataset), {})[str(int(sample_rate))] = {
            "n_files": len(subset),
            "n_probe_errors": int((~subset["probe_ok"]).sum()),
            "final_pcm_drift_ms_quantiles": _quantiles(
                valid["final_cumulative_pcm_drift_ms"]
            ),
            "abs_final_pcm_drift_ms_quantiles": _quantiles(
                valid["abs_final_cumulative_pcm_drift_ms"]
            ),
            "final_pcm_drift_threshold_counts": {
                "comparison": THRESHOLD_COMPARISON,
                "above_1_ms": int(valid["final_pcm_drift_exceeds_1_ms"].sum()),
                "above_half_video_frame": int(
                    valid["final_pcm_drift_exceeds_half_video_frame"].sum()
                ),
                "above_100_ms": int(valid["final_pcm_drift_exceeds_100_ms"].sum()),
                "above_500_ms": int(valid["final_pcm_drift_exceeds_500_ms"].sum()),
            },
            "audio_dropouts": {
                "n_files": int(valid["has_audio_dropout"].sum()),
                "n_events": len(dropouts),
                "n_persistent_steps": int(dropouts["n_steps"].sum()),
                "n_candidate_steps": int(valid["n_dropout_candidate_steps"].sum()),
                "n_compensated_candidate_steps": int(
                    valid["n_compensated_candidate_steps"].sum()
                ),
                "n_files_with_compensated_cadence": int(
                    valid["has_compensated_timestamp_cadence"].sum()
                ),
                "n_abnormally_long_packets_observed": int(
                    valid["n_abnormally_long_packet_durations"].sum()
                ),
                "dropout_duration_ms_quantiles": _quantiles(
                    dropouts["dropout_duration_ms"]
                ),
                "event_total_excess_ms_quantiles": _quantiles(
                    dropouts["dropout_total_excess_ms"]
                ),
            },
        }
    return result


def characterize_ego4d_44100(
    files: pd.DataFrame,
    decode_validation: pd.DataFrame | None,
) -> dict[str, object]:
    """Describe the two observed 44.1 kHz timing regimes without correcting them."""
    subset = files.loc[files["dataset"].eq("ego4d") & files["sample_rate_hz"].eq(44100)]
    dense = subset.loc[
        subset["robust_nominal_packet_duration_samples"]
        < 0.95 * subset["reference_decoded_samples"]
    ]
    sparse = subset.drop(index=dense.index)
    decoded = (
        decode_validation.loc[
            decode_validation["selection_reason"].eq("ego4d_44100_systematic_profile")
        ]
        if decode_validation is not None and not decode_validation.empty
        else pd.DataFrame()
    )
    return {
        "n_files": len(subset),
        "n_dense_compensating_duration_files": len(dense),
        "n_sparse_near_1024_duration_files": len(sparse),
        "dense_robust_nominal_duration_samples": _quantiles(
            dense["robust_nominal_packet_duration_samples"]
        ),
        "sparse_robust_nominal_duration_samples": _quantiles(
            sparse["robust_nominal_packet_duration_samples"]
        ),
        "final_pcm_drift_ms_quantiles": _quantiles(
            subset["final_cumulative_pcm_drift_ms"]
        ),
        "max_abs_cumulative_pcm_drift_ms_quantiles": _quantiles(
            subset["max_abs_cumulative_pcm_drift_ms"]
        ),
        "targeted_decode": {
            "n_files": (
                int(decoded["relative_path"].nunique()) if not decoded.empty else 0
            ),
            "n_windows": len(decoded),
            "n_frames": (
                int(decoded["n_valid_decoded_sample_counts"].fillna(0).sum())
                if not decoded.empty
                else 0
            ),
            "n_non_reference_frames": (
                int(decoded["n_non_reference_decoded_frames"].fillna(0).sum())
                if not decoded.empty
                else 0
            ),
            "n_960_sample_frames": (
                int(decoded["n_960_sample_frames"].fillna(0).sum())
                if not decoded.empty
                else 0
            ),
        },
        "interpretation": (
            "Four files use dense compensating short/long PTS increments; six "
            "are predominantly 1024-sample steps with sparse deviations. Targeted "
            "decoding retains 1024 samples per AAC frame, so no correction is "
            "inferred. Native PTS must be preserved."
        ),
    }


def build_packet_summary(
    files: pd.DataFrame,
    events: pd.DataFrame,
    decode_validation: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build the concise population-level JSON summary."""
    datasets: dict[str, object] = {}
    for dataset, subset in files.groupby("dataset"):
        valid = subset.loc[subset["probe_ok"]]
        datasets[str(dataset)] = {
            "n_files": len(subset),
            "n_probe_errors": int((~subset["probe_ok"]).sum()),
            "sample_rate_counts": _counts(valid["sample_rate_hz"]),
            "codec_counts": _counts(valid["audio_codec"]),
            "n_files_with_packet_timeline_gaps": int(
                valid["packet_timeline_gap_count"].gt(0).sum()
            ),
            "n_files_with_packet_timeline_overlaps": int(
                valid["packet_timeline_overlap_count"].gt(0).sum()
            ),
            "n_files_with_monotonic_pts": int(valid["pts_monotonic"].sum()),
            "n_files_with_full_timestamp_coverage": int(
                valid["timestamp_coverage"].eq(1.0).sum()
            ),
            "n_files_with_full_duration_coverage": int(
                valid["duration_coverage"].eq(1.0).sum()
            ),
            "n_files_with_timestamp_jitter": int(valid["has_timestamp_jitter"].sum()),
            "n_files_with_small_bounded_timestamp_jitter": int(
                valid["has_small_bounded_timestamp_jitter"].sum()
            ),
            "n_files_with_significant_transient_pcm_offset": int(
                valid["has_significant_transient_pcm_offset"].sum()
            ),
            "n_files_with_significant_final_pcm_offset": int(
                valid["has_significant_final_pcm_offset"].sum()
            ),
            "n_files_with_stitch_related_events": int(
                valid["has_stitch_related_event"].sum()
            ),
            "n_files_with_early_boundary_events": int(
                valid["has_early_boundary_event"].sum()
            ),
            "n_files_with_audio_dropouts": int(valid["has_audio_dropout"].sum()),
            "n_audio_dropout_events": int(valid["audio_dropout_event_count"].sum()),
            "n_files_with_compensated_timestamp_cadence": int(
                valid["has_compensated_timestamp_cadence"].sum()
            ),
            "n_dropout_candidate_steps": int(valid["n_dropout_candidate_steps"].sum()),
            "n_compensated_candidate_steps": int(
                valid["n_compensated_candidate_steps"].sum()
            ),
            "n_files_with_skip_samples": int(valid["skip_samples"].gt(0).sum()),
            "n_files_with_discard_padding": int(
                valid["discard_padding_samples"].gt(0).sum()
            ),
            "skip_samples_counts": _counts(valid["skip_samples"]),
            "discard_padding_samples_counts": _counts(valid["discard_padding_samples"]),
            "max_abs_packet_timeline_error_samples": float(
                valid["packet_max_abs_timeline_error_samples"].max()
            ),
            "max_abs_pcm_step_error_samples": float(
                valid["max_abs_pcm_step_error_samples"].max()
            ),
            "max_abs_cumulative_pcm_drift_ms": float(
                valid["max_abs_cumulative_pcm_drift_ms"].max()
            ),
            "max_abs_final_pcm_drift_ms": float(
                valid["final_cumulative_pcm_drift_ms"].abs().max()
            ),
        }

    summary: dict[str, object] = {
        "audit_type": "full_audio_packet_and_pcm_clock_audit",
        "n_files": len(files),
        "n_probe_errors": int((~files["probe_ok"]).sum()),
        "n_events": len(events),
        "drift_threshold_ms": DRIFT_THRESHOLD_MS,
        "datasets": datasets,
        "by_dataset_sample_rate": build_dataset_sample_rate_summary(files, events),
        "ego4d_44100_characterization": characterize_ego4d_44100(
            files,
            decode_validation,
        ),
        "event_type_counts": _counts(events["event_type"]) if not events.empty else {},
        "event_category_counts": _counts(events["event_category"])
        if not events.empty
        else {},
        "ego4d_stitch_events_by_sample_rate": _counts(
            events.loc[
                events["event_category"].eq("stitch_related_pcm_discontinuity"),
                "sample_rate_hz",
            ]
        )
        if not events.empty
        else {},
        "ego4d_stitch_files_by_sample_rate": {
            str(sample_rate): int(group["relative_path"].nunique())
            for sample_rate, group in events.loc[
                events["event_category"].eq("stitch_related_pcm_discontinuity")
            ].groupby("sample_rate_hz")
        }
        if not events.empty
        else {},
        "interpretation": {
            "packet_clock_metric": "next_pts-current_pts-current_packet_duration",
            "pcm_clock_metric": "next_pts-current_pts-reference_decoded_samples",
            "decoded_reference_source": "targeted_show_frames_mode",
            "temporal_source_of_truth": "native_pts",
            "automatic_audio_correction": False,
            "ego4d_observed_stitch_grid_sec": EGO4D_STITCH_PERIOD_SEC,
            "threshold_comparison": THRESHOLD_COMPARISON,
            "audio_dropout_definition": (
                "positive PCM-clock step > drift_threshold whose cumulative excess "
                "stays >= persistence_fraction of the step for compensation_window_sec"
            ),
            "compensation_window_sec": COMPENSATION_WINDOW_SEC,
            "persistence_fraction": PERSISTENCE_FRACTION,
            "compensated_timestamp_cadence_definition": (
                "episode with candidate steps that are all compensated inside the "
                "window; not missing audio"
            ),
            "abnormally_long_packet_role": "observation only; never decides category",
            "long_packet_nominal_duration_method": "per_file_median",
            "long_packet_duration_fence": (
                "max(median + max(6*MAD, 0.5*median), decoded_reference + 1ms)"
            ),
            "file_time_of_max_abs_cumulative_pcm_drift": (
                "max_abs_cumulative_pcm_drift_time_sec"
            ),
        },
    }
    return summary


def _add_case(
    cases: list[dict[str, object]],
    *,
    name: str,
    reason: str,
    file_row,
    event_time_sec: float | None = None,
) -> None:
    cases.append(
        {
            "case": name,
            "selection_reason": reason,
            "dataset": file_row.dataset,
            "relative_path": file_row.relative_path,
            "event_time_sec": event_time_sec,
        }
    )


def select_decode_validation_cases(
    files: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Choose deterministic regimes and anomalies for targeted frame decoding."""
    valid = files.loc[files["probe_ok"]].copy()
    cases: list[dict[str, object]] = []

    for dataset, rate, name in [
        ("egocom", 44100, "clean_egocom"),
        ("ego4d", 32000, "ego4d_32000_regime"),
        ("ego4d", 44100, "ego4d_44100_regime"),
        ("ego4d", 48000, "ego4d_48000_regime"),
    ]:
        subset = valid.loc[
            valid["dataset"].eq(dataset) & valid["sample_rate_hz"].eq(rate)
        ].sort_values(
            [
                "max_abs_cumulative_pcm_drift_samples",
                "max_abs_pcm_step_error_samples",
                "relative_path",
            ]
        )
        if not subset.empty:
            file_row = next(subset.itertuples(index=False))
            _add_case(
                cases,
                name=name,
                reason="dataset_sample_rate_regime",
                file_row=file_row,
            )
            _add_case(
                cases,
                name=f"{name}_end",
                reason="dataset_sample_rate_end_boundary",
                file_row=file_row,
                event_time_sec=float(file_row.audio_duration_sec),
            )

    ego4d_44100 = valid.loc[
        valid["dataset"].eq("ego4d") & valid["sample_rate_hz"].eq(44100)
    ].sort_values("relative_path")
    for file_row in ego4d_44100.itertuples(index=False):
        file_id = Path(file_row.relative_path).stem[:8]
        for window, event_time_sec in [
            ("start", 0.0),
            ("middle", float(file_row.audio_duration_sec) / 2.0),
            ("max_offset", float(file_row.max_abs_cumulative_pcm_drift_time_sec)),
            ("end", float(file_row.audio_duration_sec)),
        ]:
            _add_case(
                cases,
                name=f"ego4d_44100_{file_id}_{window}",
                reason="ego4d_44100_systematic_profile",
                file_row=file_row,
                event_time_sec=event_time_sec,
            )

    jitter = valid.loc[valid["has_small_bounded_timestamp_jitter"]].sort_values(
        ["n_pcm_step_variations", "relative_path"], ascending=[False, True]
    )
    if not jitter.empty:
        jitter_row = next(jitter.itertuples(index=False))
        _add_case(
            cases,
            name="small_bounded_jitter",
            reason="largest_count_of_small_bounded_pts_variations",
            file_row=jitter_row,
            event_time_sec=float(jitter_row.max_abs_cumulative_pcm_drift_time_sec),
        )

    if not events.empty:
        for category, name in [
            ("stitch_related_pcm_discontinuity", "stitch_related_event"),
            ("codec_boundary_pcm_event", "early_codec_boundary_event"),
        ]:
            candidates = events.loc[events["event_category"].eq(category)].sort_values(
                ["magnitude_samples", "relative_path", "event_time_sec"],
                ascending=[False, True, True],
            )
            if not candidates.empty:
                event = candidates.iloc[0]
                file_row = next(
                    valid.loc[
                        valid["relative_path"].eq(event["relative_path"])
                    ].itertuples(index=False)
                )
                _add_case(
                    cases,
                    name=name,
                    reason=category,
                    file_row=file_row,
                    event_time_sec=float(event["event_time_sec"]),
                )

        cadence_events = events.loc[
            events["event_category"].eq("compensated_timestamp_cadence")
        ]
        for (dataset, sample_rate), candidates in cadence_events.groupby(
            ["dataset", "sample_rate_hz"]
        ):
            event = candidates.sort_values(
                ["max_abs_pcm_step_error_samples", "relative_path", "event_time_sec"],
                ascending=[False, True, True],
            ).iloc[0]
            file_row = next(
                valid.loc[valid["relative_path"].eq(event["relative_path"])].itertuples(
                    index=False
                )
            )
            _add_case(
                cases,
                name=f"compensated_cadence_{dataset}_{int(sample_rate)}",
                reason="compensated_timestamp_cadence",
                file_row=file_row,
                event_time_sec=float(event["event_time_sec"]),
            )

        dropout_events = events.loc[events["event_category"].eq("audio_dropout")]
        for (dataset, sample_rate), candidates in dropout_events.groupby(
            ["dataset", "sample_rate_hz"]
        ):
            event = candidates.sort_values(
                ["dropout_duration_ms", "relative_path", "event_time_sec"],
                ascending=[False, True, True],
            ).iloc[0]
            file_row = next(
                valid.loc[valid["relative_path"].eq(event["relative_path"])].itertuples(
                    index=False
                )
            )
            _add_case(
                cases,
                name=f"audio_dropout_{dataset}_{int(sample_rate)}",
                reason="audio_dropout",
                file_row=file_row,
                event_time_sec=float(event["event_time_sec"]),
            )

    largest = valid.sort_values(
        ["max_abs_cumulative_pcm_drift_samples", "relative_path"],
        ascending=[False, True],
    ).head(2)
    for index, file_row in enumerate(largest.itertuples(index=False), start=1):
        _add_case(
            cases,
            name=f"largest_pcm_offset_{index}",
            reason="population_max_abs_cumulative_pcm_drift",
            file_row=file_row,
            event_time_sec=float(file_row.max_abs_cumulative_pcm_drift_time_sec),
        )
    return pd.DataFrame.from_records(cases)


def run_targeted_decode_validation(
    cases: pd.DataFrame,
    *,
    files: pd.DataFrame,
    raw_root: Path,
) -> pd.DataFrame:
    """Decode short, selected windows to validate packet-level interpretation."""
    records: list[dict[str, object]] = []
    files_by_path = files.set_index("relative_path")
    for case in cases.itertuples(index=False):
        file_row = files_by_path.loc[case.relative_path]
        event_time = case.event_time_sec
        if pd.isna(event_time):
            interval = "%+1.000000"
        else:
            interval = f"{max(0.0, float(event_time) - 1.0):.6f}%+2.000000"
        base = {
            "case": case.case,
            "selection_reason": case.selection_reason,
            "dataset": case.dataset,
            "relative_path": case.relative_path,
            "requested_interval": interval,
            "event_time_sec": event_time,
            "audio_codec": file_row.audio_codec,
            "sample_rate_hz": int(file_row.sample_rate_hz),
            "time_base": file_row.time_base,
            "packet_nominal_duration_samples": (
                file_row.nominal_packet_duration_samples
            ),
            "robust_nominal_packet_duration_samples": (
                file_row.robust_nominal_packet_duration_samples
            ),
            "long_packet_duration_threshold_samples": (
                file_row.long_packet_duration_threshold_samples
            ),
            "packet_max_abs_duration_deviation_samples": (
                file_row.max_abs_packet_duration_deviation_samples
            ),
            "skip_samples": int(file_row.skip_samples),
            "discard_padding_samples": int(file_row.discard_padding_samples),
        }
        try:
            frames = probe_audio_frames(
                raw_root / case.relative_path,
                read_intervals=interval,
            )
            decoded = summarize_decoded_sample_counts(frames)
            continuity = analyze_audio_timeline(
                frames,
                sample_rate_hz=int(file_row.sample_rate_hz),
                time_base=str(file_row.time_base),
            )
            distribution = decoded.pop("decoded_sample_count_distribution")
            records.append(
                {
                    **base,
                    "probe_ok": True,
                    "probe_error": None,
                    **decoded,
                    "decoded_sample_count_distribution_json": json.dumps(
                        distribution,
                        sort_keys=True,
                    ),
                    "reference_matches_mode": (
                        decoded["reference_decoded_samples"]
                        == int(file_row.reference_decoded_samples)
                    ),
                    "decoded_pts_max_abs_step_error_samples": continuity.get(
                        "max_abs_gap_samples",
                        np.nan,
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    **base,
                    "probe_ok": False,
                    "probe_error": str(exc),
                    "reference_matches_mode": False,
                }
            )
    return pd.DataFrame.from_records(records)


def validate_media_metadata(media_metadata: pd.DataFrame) -> None:
    required = {
        "dataset",
        "relative_path",
        "audio_codec",
        "audio_sample_rate_hz",
        "audio_time_base",
        "audio_duration_sec",
        "video_avg_frame_rate",
    }
    missing = required - set(media_metadata.columns)
    if missing:
        raise ValueError(f"Missing required media metadata columns: {sorted(missing)}")
    unsupported = set(media_metadata["audio_codec"].dropna()) - set(
        DECODED_SAMPLE_REFERENCES
    )
    if unsupported:
        raise ValueError(
            f"Audio codecs need decoded-sample validation: {sorted(unsupported)}"
        )


def main() -> None:
    cfg = load_config()
    raw_root = Path(cfg.paths.raw)
    reports_root = Path(cfg.paths.reports)
    metadata_path = (
        reports_root / "temporal" / "media_metadata" / "media_metadata.parquet"
    )
    output_dir = reports_root / "temporal" / "audio_timeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Media metadata report not found: {metadata_path}")

    media_metadata = pd.read_parquet(metadata_path)
    validate_media_metadata(media_metadata)
    file_records: list[dict[str, object]] = []
    event_frames: list[pd.DataFrame] = []
    rows = list(media_metadata.itertuples(index=False))
    total = len(rows)
    audit_one = partial(audit_audio_file, raw_root=raw_root)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(audit_one, rows)
        for index, (file_record, events) in enumerate(results, start=1):
            print(f"\rPacket scan {index}/{total}", end="", flush=True)
            file_records.append(file_record)
            if not events.empty:
                event_frames.append(events)
    print()

    files = (
        pd.DataFrame.from_records(file_records)
        .sort_values(["dataset", "relative_path"])
        .reset_index(drop=True)
    )
    events = (
        pd.concat(event_frames, ignore_index=True)
        .sort_values(["dataset", "relative_path", "event_time_sec", "event_type"])
        .reset_index(drop=True)
        if event_frames
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    cases = select_decode_validation_cases(files, events)
    decode_validation = run_targeted_decode_validation(
        cases,
        files=files,
        raw_root=raw_root,
    )

    files_path = output_dir / "audio_packet_timeline_files.parquet"
    events_path = output_dir / "audio_packet_timeline_events.parquet"
    summary_path = output_dir / "audio_packet_timeline_summary.json"
    decode_path = output_dir / "audio_decode_validation.parquet"
    decode_summary_path = output_dir / "audio_decode_validation_summary.json"
    files.to_parquet(files_path, index=False)
    events.to_parquet(events_path, index=False)
    decode_validation.to_parquet(decode_path, index=False)

    summary = build_packet_summary(files, events, decode_validation)
    write_summary(summary_path, summary, parameters=audit_parameters())
    decode_summary = {
        "n_cases": len(decode_validation),
        "n_probe_ok": int(decode_validation["probe_ok"].sum()),
        "n_reference_matches": int(decode_validation["reference_matches_mode"].sum()),
        "n_960_sample_frames": int(
            decode_validation["n_960_sample_frames"].fillna(0).sum()
        ),
        "case_counts": _counts(decode_validation["case"]),
        "ego4d_44100": {
            "n_files": int(
                decode_validation.loc[
                    decode_validation["selection_reason"].eq(
                        "ego4d_44100_systematic_profile"
                    ),
                    "relative_path",
                ].nunique()
            ),
            "n_windows": int(
                decode_validation["selection_reason"]
                .eq("ego4d_44100_systematic_profile")
                .sum()
            ),
            "n_non_reference_decoded_frames": int(
                decode_validation.loc[
                    decode_validation["selection_reason"].eq(
                        "ego4d_44100_systematic_profile"
                    ),
                    "n_non_reference_decoded_frames",
                ]
                .fillna(0)
                .sum()
            ),
        },
    }
    write_summary(decode_summary_path, decode_summary, parameters=audit_parameters())

    print(f"files: {files_path}")
    print(f"events: {events_path}")
    print(f"summary: {summary_path}")
    print(f"targeted decode: {decode_path}")
    print(f"probe success: {int(files['probe_ok'].sum())}/{len(files)}")
    print(
        "packet timeline errors:",
        int(files["packet_timeline_gap_count"].sum()),
        "gaps /",
        int(files["packet_timeline_overlap_count"].sum()),
        "overlaps",
    )


if __name__ == "__main__":
    main()
