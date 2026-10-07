from conv_wm.config import get_path, load_config


def test_load_config():
    cfg = load_config()

    assert "egocom" in cfg.datasets
    assert "ego4d" in cfg.datasets


def test_get_raw_path():
    cfg = load_config()

    path = get_path(cfg, dataset="egocom", stage="raw", key="ground_truth")

    assert path.name == "ground_truth_transcriptions.csv"


def test_overrides_set_values_and_refuse_unknown_keys():
    import pytest

    from conv_wm.config import load_config

    cfg = load_config(
        overrides=["grid.decision_step_s=0.08", "labels.subframes_per_step=2"]
    )
    assert cfg.grid.decision_step_s == 0.08
    assert cfg.labels.subframes_per_step == 2  # typed, not a string

    with pytest.raises(KeyError, match="labels.subframe_per_step"):
        load_config(overrides=["labels.subframe_per_step=2"])  # typo
    with pytest.raises(ValueError, match="key=value"):
        load_config(overrides=["labels.subframes_per_step"])
