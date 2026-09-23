"""Ego4D release splits, keyed by the session of the vocal pipeline (the clip).

``clips_clean`` carries the split its video was published in (``train`` or
``val``; the benchmark publishes no labelled test split). The split is the
release's own, never recomputed here.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from conv_wm.data.datasets.spec import parquet_table


def load_ego4d_recording_splits(cfg: DictConfig) -> pd.DataFrame:
    """One row per clip: its ``recording_id`` and the release split it belongs to."""
    clips = parquet_table("ego4d", "interim", "clips_clean")(cfg)
    return pd.DataFrame(
        {
            "recording_id": clips["clip_uid"].astype("string"),
            "split": clips["split"].astype("string"),
        }
    )


__all__ = ["load_ego4d_recording_splits"]
