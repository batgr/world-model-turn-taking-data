import numpy as np
import pytest

from conv_wm.data.vocal.identity import MultiDeviceEnergyState


def test_energy_state_converts_canonical_time_to_local_indices_once():
    state = MultiDeviceEnergyState(
        frame_s=0.1,
        canonical_offset_s=10.0,
        focal_db=np.array([10.0, 10.0, 10.0, 10.0]),
        others_db={"other": np.array([2.0, 2.0, 2.0, 2.0])},
        lags_frames={"other": 0},
        correlations={"other": 1.0},
        excluded_views=(),
    )

    assert state.dominance_db(10.0, 10.2) == pytest.approx(8.0)
