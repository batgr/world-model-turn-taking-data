"""Population-level summary of the audio timeline audit."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any, SupportsInt, cast

import pandas as pd

from conv_wm.data.media.audio.records import EventCategory
from conv_wm.reports import JsonDict


def regime(key: Hashable) -> tuple[str, int]:
    """Typed ``(dataset, sample_rate_hz)`` from a two-column groupby key."""
    dataset, rate = cast(tuple[object, SupportsInt], key)
    return str(dataset), int(rate)


QUANTILES: tuple[tuple[str, float], ...] = (
    ("min", 0.0),
    ("p05", 0.05),
    ("p25", 0.25),
    ("median", 0.5),
    ("p75", 0.75),
    ("p95", 0.95),
    ("max", 1.0),
)


def value_counts(series: pd.Series) -> dict[str, int]:
    """Counts keyed by stringified value, JSON-ready."""
    return {str(key): int(value) for key, value in series.value_counts().items()}


def quantiles(series: pd.Series) -> dict[str, float]:
    """Seven fixed quantiles of a numeric series (empty mapping when no data)."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {name: float(values.quantile(q)) for name, q in QUANTILES}


def dropout_section(valid: pd.DataFrame, dropouts: pd.DataFrame) -> JsonDict:
    """Dropout burden for one group of files, keeping candidates and observations apart."""
    return {
        "n_files": int(valid["has_audio_dropout"].sum()),
        "n_events": len(dropouts),
        "n_persistent_steps": int(dropouts["n_steps"].sum())
        if not dropouts.empty
        else 0,
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
        "dropout_duration_ms_quantiles": quantiles(
            dropouts.get("dropout_duration_ms", pd.Series(dtype=float))
        ),
    }


def by_dataset_and_sample_rate(
    files: pd.DataFrame,
    events: pd.DataFrame,
    *,
    threshold_comparison: str,
) -> dict[str, dict[str, Any]]:
    """Final-drift distributions and dropout burden per dataset and sample rate."""
    result: dict[str, dict[str, Any]] = {}
    for key, subset in files.groupby(["dataset", "sample_rate_hz"]):
        dataset, sample_rate = regime(key)
        valid = subset.loc[subset["probe_ok"]]
        dropouts = (
            events.loc[
                events["dataset"].eq(dataset)
                & events["sample_rate_hz"].eq(sample_rate)
                & events["event_category"].eq(EventCategory.AUDIO_DROPOUT)
            ]
            if not events.empty
            else pd.DataFrame(columns=events.columns)
        )
        result.setdefault(dataset, {})[str(sample_rate)] = {
            "n_files": len(subset),
            "n_probe_errors": int((~subset["probe_ok"]).sum()),
            "final_pcm_drift_ms_quantiles": quantiles(
                valid["final_cumulative_pcm_drift_ms"]
            ),
            "abs_final_pcm_drift_ms_quantiles": quantiles(
                valid["abs_final_cumulative_pcm_drift_ms"]
            ),
            "final_pcm_drift_threshold_counts": {
                "comparison": threshold_comparison,
                "above_1_ms": int(valid["final_pcm_drift_exceeds_1_ms"].sum()),
                "above_half_video_frame": int(
                    valid["final_pcm_drift_exceeds_half_video_frame"].sum()
                ),
                "above_100_ms": int(valid["final_pcm_drift_exceeds_100_ms"].sum()),
                "above_500_ms": int(valid["final_pcm_drift_exceeds_500_ms"].sum()),
            },
            "audio_dropouts": dropout_section(valid, dropouts),
        }
    return result


def dataset_overview(files: pd.DataFrame) -> dict[str, JsonDict]:
    """Per-dataset counts of files, regimes and file-level flags."""
    overview: dict[str, JsonDict] = {}
    for dataset, subset in files.groupby("dataset"):
        valid = subset.loc[subset["probe_ok"]]
        overview[str(dataset)] = {
            "n_files": len(subset),
            "n_probe_errors": int((~subset["probe_ok"]).sum()),
            "sample_rate_counts": value_counts(valid["sample_rate_hz"]),
            "codec_counts": value_counts(valid["audio_codec"]),
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
            "skip_samples_counts": value_counts(valid["skip_samples"]),
            "discard_padding_samples_counts": value_counts(
                valid["discard_padding_samples"]
            ),
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
    return overview


INTERPRETATION: JsonDict = {
    "packet_clock_metric": "next_pts - current_pts - current_packet_duration",
    "pcm_clock_metric": "next_pts - current_pts - reference_decoded_samples",
    "temporal_source_of_truth": "native_pts",
    "automatic_audio_correction": False,
    "audio_dropout_definition": (
        "positive PCM-clock step > drift_threshold whose cumulative excess stays "
        ">= persistence_fraction of the step for compensation_window_sec"
    ),
    "compensated_timestamp_cadence_definition": (
        "episode whose candidate steps are all compensated inside the window; "
        "not missing audio"
    ),
    "abnormally_long_packet_role": "observation only; never decides a category",
    "long_packet_nominal_duration_method": "per_file_median",
    "long_packet_duration_fence": "max(median + max(6*MAD, 0.5*median), reference + threshold)",
    "file_time_of_max_abs_cumulative_pcm_drift": "max_abs_cumulative_pcm_drift_time_sec",
    "known_boundary_grids": "declared per dataset; see parameters.datasets",
}


def build_audio_timeline_summary(
    files: pd.DataFrame,
    events: pd.DataFrame,
    *,
    threshold_comparison: str,
    dataset_sections: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Population summary; ``dataset_sections`` carries dataset-provided analyses."""
    stitch = (
        events.loc[events["event_category"].eq(EventCategory.STITCH_RELATED)]
        if not events.empty
        else pd.DataFrame(columns=events.columns)
    )
    return {
        "audit_type": "audio_timeline",
        "n_files": len(files),
        "n_probe_errors": int((~files["probe_ok"]).sum()),
        "n_events": len(events),
        "datasets": dataset_overview(files),
        "by_dataset_sample_rate": by_dataset_and_sample_rate(
            files, events, threshold_comparison=threshold_comparison
        ),
        "event_type_counts": value_counts(events["event_type"])
        if not events.empty
        else {},
        "event_category_counts": value_counts(events["event_category"])
        if not events.empty
        else {},
        "known_boundary_events_by_dataset_sample_rate": {
            "{}/{}".format(*regime(key)): len(group)
            for key, group in stitch.groupby(["dataset", "sample_rate_hz"])
        },
        "dataset_sections": dict(dataset_sections or {}),
        "interpretation": INTERPRETATION,
    }


def build_decode_validation_summary(
    decode_validation: pd.DataFrame,
) -> dict[str, object]:
    """Summary of the decoded validation windows."""
    if decode_validation.empty:
        return {
            "n_cases": 0,
            "n_probe_ok": 0,
            "n_reference_matches": 0,
            "n_960_sample_frames": 0,
        }
    return {
        "n_cases": len(decode_validation),
        "n_probe_ok": int(decode_validation["probe_ok"].sum()),
        "n_reference_matches": int(decode_validation["reference_matches_mode"].sum()),
        "n_non_reference_decoded_frames": int(
            decode_validation["n_non_reference_decoded_frames"].fillna(0).sum()
        ),
        "n_960_sample_frames": int(
            decode_validation["n_960_sample_frames"].fillna(0).sum()
        ),
        "case_counts": value_counts(decode_validation["selection_reason"]),
    }
