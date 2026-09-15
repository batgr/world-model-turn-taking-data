"""Compatibility wrapper; the maintained entry point is ``conv-wm audit audio``."""

from __future__ import annotations

from conv_wm.config import load_config
from conv_wm.data.audits.audio_timeline import run_audio_timeline_audit


def main() -> None:
    """Run the audio timeline audit with the default configuration."""
    outputs = run_audio_timeline_audit(load_config())
    print(f"summary: {outputs.summary_path}")


if __name__ == "__main__":
    main()
