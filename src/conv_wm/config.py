from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"

Stage = Literal[
    "raw",
    "interim",
    "validated",
    "processed",
    "model_ready",
    "reports",
]


def get_path(cfg, dataset: str, stage: Stage, key: str) -> Path:
    """Return the configured path for a dataset artifact.

    Args:
        cfg: Loaded project configuration.
        dataset: Dataset name, e.g. ``"egocom"`` or ``"ego4d"``.
        stage: Pipeline stage containing the artifact.
        key: Artifact key defined in the dataset ``files`` configuration.

    Returns:
        Path to the requested artifact.

    Raises:
        KeyError: If the dataset, stage, or file key does not exist.
    """
    ds = cfg.datasets[dataset]

    if key not in ds.files:
        raise KeyError(f"Unknown file key '{key}' for dataset '{dataset}'.")

    return Path(ds[stage]) / ds.files[key]


def load_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return OmegaConf.load(path)
