"""Ego4D AV benchmark: dataset-specific knowledge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from conv_wm.data.datasets.spec import AudioInterpretation, DatasetSpec
from conv_wm.data.media.audio.decode import DecodeValidationCase
from conv_wm.data.media.audio.interpretation import KnownBoundaryGrid
from conv_wm.data.media.audio.summary import quantiles

STITCH_PERIOD_SEC = 300.0
"""Ego4D recordings are joined every 300 s; audio events cluster on that grid."""

SYSTEMATIC_PROFILE_SAMPLE_RATE = 44100
"""Regime whose dense short/long PTS cadence deserves a full decoded profile."""


def systematic_profile_cases(files: pd.DataFrame) -> list[DecodeValidationCase]:
    """Four decoded windows per 44.1 kHz file: start, middle, drift peak, end."""
    cases: list[DecodeValidationCase] = []
    subset = files.loc[files["sample_rate_hz"].eq(SYSTEMATIC_PROFILE_SAMPLE_RATE)]
    for _, row in subset.sort_values("relative_path").iterrows():
        file_id = Path(str(row["relative_path"])).stem[:8]
        duration = float(row["audio_duration_sec"])
        for window, event_time in (
            ("start", 0.0),
            ("middle", duration / 2.0),
            ("max_offset", float(row["max_abs_cumulative_pcm_drift_time_sec"])),
            ("end", duration),
        ):
            cases.append(
                DecodeValidationCase(
                    case=f"ego4d_44100_{file_id}_{window}",
                    selection_reason="ego4d_44100_systematic_profile",
                    dataset="ego4d",
                    relative_path=str(row["relative_path"]),
                    event_time_sec=event_time,
                )
            )
    return cases


def characterize_44100_regime(
    files: pd.DataFrame, decode_validation: pd.DataFrame
) -> dict[str, object]:
    """Describe the two observed 44.1 kHz timing regimes without correcting them."""
    subset = files.loc[files["sample_rate_hz"].eq(SYSTEMATIC_PROFILE_SAMPLE_RATE)]
    dense = subset.loc[
        subset["robust_nominal_packet_duration_samples"]
        < 0.95 * subset["reference_decoded_samples"]
    ]
    sparse = subset.drop(index=dense.index)
    decoded = (
        decode_validation.loc[
            decode_validation["selection_reason"].eq("ego4d_44100_systematic_profile")
        ]
        if not decode_validation.empty
        else pd.DataFrame()
    )
    return {
        "n_files": len(subset),
        "n_dense_compensating_cadence_files": len(dense),
        "n_sparse_near_reference_files": len(sparse),
        "dense_robust_nominal_duration_samples": quantiles(
            dense["robust_nominal_packet_duration_samples"]
        ),
        "sparse_robust_nominal_duration_samples": quantiles(
            sparse["robust_nominal_packet_duration_samples"]
        ),
        "final_pcm_drift_ms_quantiles": quantiles(
            subset["final_cumulative_pcm_drift_ms"]
        ),
        "max_abs_cumulative_pcm_drift_ms_quantiles": quantiles(
            subset["max_abs_cumulative_pcm_drift_ms"]
        ),
        "targeted_decode": {
            "n_files": int(decoded["relative_path"].nunique())
            if not decoded.empty
            else 0,
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
        },
        "interpretation": (
            "Dense files use compensating short/long PTS increments; sparse files "
            "are predominantly reference-sized steps. Decoded frames stay at the "
            "reference size, so no correction is inferred. Native PTS remains the clock."
        ),
    }


EGO4D = DatasetSpec(
    name="ego4d",
    description="Ego4D v2 audio-visual diarization benchmark clips (video_540ss).",
    audio=AudioInterpretation(
        known_boundary_grid=KnownBoundaryGrid(period_sec=STITCH_PERIOD_SEC),
        extra_decode_cases=systematic_profile_cases,
        summary_section=characterize_44100_regime,
    ),
)
