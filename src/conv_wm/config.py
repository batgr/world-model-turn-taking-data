"""Project configuration: YAML loading and typed path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf

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


def get_path(cfg: DictConfig, dataset: str, stage: Stage, key: str) -> Path:
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


def load_config(config_path: str | Path | None = None) -> DictConfig:
    """Load the project configuration (``conf/config.yaml`` by default)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return config


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved stage roots from the ``paths`` section of the configuration."""

    raw: Path
    interim: Path
    validated: Path
    processed: Path
    model_ready: Path
    reports: Path


def pipeline_paths(cfg: DictConfig) -> PipelinePaths:
    """Resolve the stage roots of ``cfg`` into a typed object."""
    return PipelinePaths(
        raw=Path(cfg.paths.raw),
        interim=Path(cfg.paths.interim),
        validated=Path(cfg.paths.validated),
        processed=Path(cfg.paths.processed),
        model_ready=Path(cfg.paths.model_ready),
        reports=Path(cfg.paths.reports),
    )
