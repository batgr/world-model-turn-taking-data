"""Compatibility wrapper; the maintained entry point is ``conv-wm audit media``."""

from __future__ import annotations

from conv_wm.config import load_config
from conv_wm.data.audits.media_metadata import run_media_metadata_audit


def main() -> None:
    """Run the media metadata audit with the default configuration."""
    outputs = run_media_metadata_audit(load_config())
    print(f"summary: {outputs.summary_path}")


if __name__ == "__main__":
    main()
