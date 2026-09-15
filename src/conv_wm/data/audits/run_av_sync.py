"""Compatibility wrapper; the maintained entry point is ``conv-wm audit sync``."""

from __future__ import annotations

from conv_wm.config import load_config
from conv_wm.data.audits.av_sync import run_av_sync_audit


def main() -> None:
    """Run the technical A/V alignment audit with the default configuration."""
    outputs = run_av_sync_audit(load_config())
    print(f"summary: {outputs.summary_path}")


if __name__ == "__main__":
    main()
