"""Errors shared by population audits."""

from __future__ import annotations

from pathlib import Path


class MissingPrerequisiteError(FileNotFoundError):
    """A required upstream artifact is absent; the message says how to produce it."""

    def __init__(self, artifact: Path, *, produce_with: str) -> None:
        self.artifact = artifact
        self.produce_with = produce_with
        super().__init__(
            f"Required artifact not found: {artifact}\nProduce it with: {produce_with}"
        )
