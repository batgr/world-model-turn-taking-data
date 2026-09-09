from pathlib import Path

import hydra
from omegaconf import DictConfig

from conv_wm.data.manifest import build_manifest, save_manifest


def run(cfg: DictConfig) -> None:
    """Build and persist the raw dataset manifest from configuration."""
    manifest = build_manifest(
        Path(cfg.paths.raw),
        compute_checksum=cfg.manifest.compute_checksum,
    )

    save_manifest(
        manifest,
        Path(cfg.manifest.output),
    )


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for manifest generation."""
    run(cfg)


if __name__ == "__main__":
    main()
