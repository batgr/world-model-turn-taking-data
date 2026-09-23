"""EgoCom splits, keyed by the session of the vocal pipeline (the POV video).

``video_info_clean`` carries the split derived by the cleaning step from the
source's mutually exclusive ``train`` / ``val`` / ``test`` flags.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from conv_wm.data.datasets.spec import parquet_table


def load_egocom_recording_splits(cfg: DictConfig) -> pd.DataFrame:
    """One row per POV video: its ``recording_id`` and the split it belongs to."""
    info = parquet_table("egocom", "interim", "video_info_clean")(cfg)
    return pd.DataFrame(
        {
            "recording_id": info["video_name"].astype("string"),
            "split": info["split"].astype("string"),
        }
    )


__all__ = ["load_egocom_recording_splits"]
