"""Compatibility wrapper; the maintained entry point is ``conv-wm audit structure``."""

from __future__ import annotations

from conv_wm.config import load_config
from conv_wm.data.audits.structural import (
    format_structural_summary,
    run_structural_audit,
)


def main() -> None:
    """Run the structural validation with the default configuration."""
    print(format_structural_summary(run_structural_audit(load_config())))


if __name__ == "__main__":
    main()
