import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from conv_wm.data.vocal.config import VocalCoverageConfig
from conv_wm.data.vocal.identity import IdentityPipeline


@pytest.fixture
def config() -> VocalCoverageConfig:
    return VocalCoverageConfig()


@pytest.fixture
def annotation_pipeline(config) -> IdentityPipeline:
    """Ego4D-style attribution: annotations only, everything else unresolved."""
    return IdentityPipeline(("focal_annotation", "other_annotation"), config.identity)
