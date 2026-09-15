from __future__ import annotations

from pathlib import Path

import pandas as pd

from conv_wm.config import load_config
from conv_wm.data.media.video_timeline import (
    analyze_frame_timeline,
    audit_sampled_video_timeline,
    probe_video_timestamps,
)
from conv_wm.reports import write_summary


def main() -> None:
    cfg = load_config()

    raw_root = Path(cfg.paths.raw)
    reports_root = Path(cfg.paths.reports)

    metadata_path = (
        reports_root / "temporal" / "media_metadata" / "media_metadata.parquet"
    )

    output_dir = reports_root / "temporal" / "video_timeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(metadata_path)

    sampled_records: list[dict[str, object]] = []

    print(f"Auditing sampled video timelines ({len(metadata)} files)...")

    for index, row in enumerate(
        metadata.itertuples(index=False),
        start=1,
    ):
        print(
            f"\r  {index}/{len(metadata)}",
            end="",
            flush=True,
        )

        path = raw_root / row.relative_path

        sampled_records.extend(
            audit_sampled_video_timeline(
                dataset=row.dataset,
                relative_path=row.relative_path,
                path=path,
                duration_sec=float(row.video_duration_sec),
                fps=float(row.video_avg_frame_rate),
                time_base=row.video_time_base,
            )
        )

    print()

    sampled = pd.DataFrame(sampled_records)

    sampled_path = output_dir / "sampled_timeline.parquet"
    sampled.to_parquet(
        sampled_path,
        index=False,
    )

    # Aggregate sampled results at file level.
    file_summary = sampled.groupby(
        ["dataset", "relative_path"],
        as_index=False,
    ).agg(
        n_windows=(
            "window",
            "count",
        ),
        all_probes_ok=(
            "probe_ok",
            "all",
        ),
        all_windows_cfr_consistent=(
            "timeline_class",
            lambda values: bool((values == "cfr_consistent").all()),
        ),
        min_timestamp_coverage=(
            "timestamp_coverage",
            "min",
        ),
        max_delta_error_ticks=(
            "max_delta_error_ticks",
            "max",
        ),
        max_cumulative_drift_sec=(
            "max_abs_cumulative_drift_sec",
            "max",
        ),
        total_duplicate_timestamps=(
            "n_duplicate_timestamps",
            "sum",
        ),
        total_non_positive_deltas=(
            "n_non_positive_deltas",
            "sum",
        ),
    )

    # Anything not clean in the sampled audit receives a full scan.
    suspects = file_summary.loc[~file_summary["all_windows_cfr_consistent"]]

    full_scan_records: list[dict[str, object]] = []

    if not suspects.empty:
        print(f"Full-scanning {len(suspects)} suspect video(s)...")

        metadata_by_path = metadata.set_index("relative_path")

        for suspect in suspects.itertuples(index=False):
            relative_path = suspect.relative_path
            row = metadata_by_path.loc[relative_path]

            path = raw_root / relative_path

            try:
                frames = probe_video_timestamps(path)

                analysis = analyze_frame_timeline(
                    frames,
                    fps=float(row["video_avg_frame_rate"]),
                    time_base=row["video_time_base"],
                )

                full_scan_records.append(
                    {
                        "dataset": row["dataset"],
                        "relative_path": relative_path,
                        "probe_ok": True,
                        "probe_error": None,
                        **analysis,
                    }
                )

            # One bad file must not abort the audit: record the failure and go on.
            except Exception as exc:  # noqa: BLE001
                full_scan_records.append(
                    {
                        "dataset": row["dataset"],
                        "relative_path": relative_path,
                        "probe_ok": False,
                        "probe_error": str(exc),
                        "timeline_class": ("probe_error"),
                    }
                )

    full_scan = pd.DataFrame(full_scan_records)

    if not full_scan.empty:
        full_scan_path = output_dir / "full_scan_suspects.parquet"
        full_scan.to_parquet(
            full_scan_path,
            index=False,
        )

    file_summary_path = output_dir / "file_summary.parquet"
    file_summary.to_parquet(
        file_summary_path,
        index=False,
    )

    n_consistent = int(file_summary["all_windows_cfr_consistent"].sum())

    if full_scan.empty:
        n_irregular_full = 0
    else:
        n_irregular_full = int((full_scan["timeline_class"] != "cfr_consistent").sum())

    summary = {
        "n_files": len(file_summary),
        "n_sampled_windows": len(sampled),
        "n_files_sampled_consistent": n_consistent,
        "n_sampled_suspects": len(suspects),
        "n_full_scanned": len(full_scan),
        "n_irregular_after_full_scan": (n_irregular_full),
    }

    summary_path = output_dir / "summary.json"
    write_summary(summary_path, summary)

    print()
    print("Video timeline audit")
    print(f"  files: {summary['n_files']}")
    print(
        "  sampled windows:",
        summary["n_sampled_windows"],
    )
    print(
        "  sampled consistent:",
        f"{n_consistent}/{len(file_summary)}",
    )
    print(
        "  suspects:",
        summary["n_sampled_suspects"],
    )
    print(
        "  full scans:",
        summary["n_full_scanned"],
    )
    print(
        "  irregular:",
        summary["n_irregular_after_full_scan"],
    )

    overall = n_consistent == len(file_summary) and n_irregular_full == 0

    print(f"overall: {'PASS' if overall else 'REVIEW'}")
    print(f"report: {summary_path}")


if __name__ == "__main__":
    main()
