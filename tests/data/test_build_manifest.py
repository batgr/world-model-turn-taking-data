from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from conv_wm.data.build_manifest import run


def test_run_builds_manifest_from_global_raw_root(tmp_path: Path):
    raw_root = tmp_path / "raw"

    egocom = raw_root / "EgoCom"
    ego4d = raw_root / "Ego4D"

    egocom.mkdir(parents=True)
    ego4d.mkdir(parents=True)

    (egocom / "video.mp4").write_bytes(b"video")
    (ego4d / "annotations.json").write_text("{}")

    output = tmp_path / "reports" / "manifest" / "raw_manifest.parquet"

    cfg = OmegaConf.create(
        {
            "paths": {
                "raw": str(raw_root),
            },
            "manifest": {
                "compute_checksum": False,
                "output": str(output),
            },
        }
    )

    run(cfg)

    assert output.exists()

    manifest = pd.read_parquet(output)

    assert len(manifest) == 2
    assert set(manifest["dataset"]) == {"egocom", "ego4d"}
