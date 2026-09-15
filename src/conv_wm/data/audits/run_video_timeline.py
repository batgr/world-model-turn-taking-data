"""Compatibility wrapper; the maintained entry point is ``conv-wm audit video``."""

from __future__ import annotations

from conv_wm.config import load_config
from conv_wm.data.audits.video_timeline import run_video_timeline_audit


def main() -> None:
    """Run the video timeline audit with the default configuration."""
    outputs = run_video_timeline_audit(load_config())
    print(f"summary: {outputs.summary_path}")


if __name__ == "__main__":
    main()
