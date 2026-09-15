"""EgoCom: dataset-specific knowledge."""

from __future__ import annotations

from conv_wm.data.datasets.spec import DatasetSpec

EGOCOM = DatasetSpec(
    name="egocom",
    description="EgoCom multi-person egocentric conversations (240p, 20-minute videos).",
)
