"""Compatibility wrapper; the maintained entry point is ``conv-wm audit manifest``."""

from __future__ import annotations

from omegaconf import DictConfig

from conv_wm.config import load_config
from conv_wm.data.audits.manifest import run_manifest_audit


def run(cfg: DictConfig) -> None:
    """Build and persist the raw dataset manifest from configuration."""
    run_manifest_audit(cfg)


def main() -> None:
    """Build the manifest with the default configuration."""
    outputs = run_manifest_audit(load_config())
    print(f"manifest: {outputs.manifest_path}")


if __name__ == "__main__":
    main()
