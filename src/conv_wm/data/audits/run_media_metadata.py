from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conv_wm.config import load_config
from conv_wm.data.media.ffprobe import FFprobeError
from conv_wm.data.media.metadata import extract_media_metadata


def main() -> None:
    cfg = load_config()

    raw_root = Path(cfg.paths.raw)
    reports_root = Path(cfg.paths.reports)

    manifest_path = reports_root / "manifest" / "raw_manifest.parquet"

    output_dir = reports_root / "temporal" / "media_metadata"

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(manifest_path)

    media = manifest[manifest["file_type"].isin(["video", "audio"])].copy()

    records: list[dict] = []

    for row in media.itertuples(index=False):
        path = raw_root / row.relative_path

        try:
            record = extract_media_metadata(
                dataset=row.dataset,
                relative_path=row.relative_path,
                path=path,
            )

            record["probe_error"] = None

        except FFprobeError as exc:
            record = {
                "dataset": row.dataset,
                "relative_path": row.relative_path,
                "probe_ok": False,
                "probe_error": str(exc),
            }

        records.append(record)

    metadata = pd.DataFrame(records)

    # Descriptive derived fields.
    if {
        "audio_duration_sec",
        "video_duration_sec",
    }.issubset(metadata.columns):
        metadata["av_duration_delta_sec"] = (
            metadata["audio_duration_sec"] - metadata["video_duration_sec"]
        )

    # Minimal B1 QC.
    metadata["qc_probe_ok"] = metadata["probe_ok"].fillna(False)

    metadata["qc_has_video"] = metadata["n_video_streams"].fillna(0) >= 1

    metadata["qc_has_audio"] = metadata["n_audio_streams"].fillna(0) >= 1

    metadata["qc_positive_video_duration"] = (
        metadata["video_duration_sec"].fillna(0) > 0
    )

    metadata["qc_positive_audio_duration"] = (
        metadata["audio_duration_sec"].fillna(0) > 0
    )

    metadata_path = output_dir / "media_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)

    summary = {
        "n_files": len(metadata),
        "n_probe_ok": int(metadata["qc_probe_ok"].sum()),
        "n_with_video": int(metadata["qc_has_video"].sum()),
        "n_with_audio": int(metadata["qc_has_audio"].sum()),
        "datasets": {
            str(dataset): int(count)
            for dataset, count in metadata["dataset"].value_counts().items()
        },
    }

    summary_path = output_dir / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for dataset, group in metadata.groupby("dataset"):
        print(dataset)
        print(f"  files: {len(group)}")
        print(
            "  probe:",
            f"{int(group['qc_probe_ok'].sum())}/{len(group)} PASS",
        )
        print(
            "  video:",
            f"{int(group['qc_has_video'].sum())}/{len(group)}",
        )
        print(
            "  audio:",
            f"{int(group['qc_has_audio'].sum())}/{len(group)}",
        )

    overall = bool(
        metadata[
            [
                "qc_probe_ok",
                "qc_has_video",
                "qc_has_audio",
                "qc_positive_video_duration",
                "qc_positive_audio_duration",
            ]
        ].all(axis=None)
    )

    print(f"overall: {'PASS' if overall else 'FAIL'}")
    print(f"metadata: {metadata_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
