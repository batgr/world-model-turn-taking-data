"""Population runner for the B4 technical A/V container-timeline audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from conv_wm.config import load_config
from conv_wm.data.media.av_sync import compute_av_sync_metadata
from conv_wm.reports import write_summary


def build_av_sync_summary(sync: pd.DataFrame) -> dict[str, object]:
    """Summarize measured start offsets and end/duration deltas by dataset."""
    datasets: dict[str, object] = {}
    for dataset, group in sync.groupby("dataset"):
        datasets[str(dataset)] = {
            "n_files": len(group),
            "n_zero_start_offsets": int(group["av_start_offset_sec"].eq(0).sum()),
            "max_abs_start_offset_sec": float(group["av_start_offset_sec"].abs().max()),
            "av_end_delta_sec": {
                "min": float(group["av_end_delta_sec"].min()),
                "median": float(group["av_end_delta_sec"].median()),
                "max": float(group["av_end_delta_sec"].max()),
                "max_abs": float(group["abs_av_end_delta_sec"].max()),
            },
        }
    all_starts_zero = bool(sync["av_start_offset_sec"].eq(0).all())
    return {
        "audit_type": "technical_av_container_timeline",
        "n_files": len(sync),
        "n_zero_start_offsets": int(sync["av_start_offset_sec"].eq(0).sum()),
        "all_start_offsets_zero": all_starts_zero,
        "datasets": datasets,
        "interpretation": {
            "audio_time_source": "native_pts",
            "video_time_source": "native_pts",
            "global_av_offset_correction": False,
            "duration_delta_is_sync_failure": False,
            "perceptual_av_sync": "deferred",
            "fine_av_drift_correction": "deferred",
        },
    }


def main() -> None:
    cfg = load_config()
    reports_root = Path(cfg.paths.reports)
    metadata_path = (
        reports_root / "temporal" / "media_metadata" / "media_metadata.parquet"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Media metadata report not found: {metadata_path}")

    output_dir = reports_root / "temporal" / "av_sync"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(metadata_path)
    sync = compute_av_sync_metadata(metadata).sort_values(["dataset", "relative_path"])
    summary = build_av_sync_summary(sync)

    files_path = output_dir / "av_sync_files.parquet"
    summary_path = output_dir / "av_sync_summary.json"
    sync.to_parquet(files_path, index=False)
    write_summary(summary_path, summary)

    print(f"files: {files_path}")
    print(f"summary: {summary_path}")
    print(
        "zero start offsets:",
        f"{summary['n_zero_start_offsets']}/{summary['n_files']}",
    )
    print("global correction: not applied")


if __name__ == "__main__":
    main()
