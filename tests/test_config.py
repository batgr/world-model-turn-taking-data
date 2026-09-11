from conv_wm.config import get_path, load_config


def test_load_config():
    cfg = load_config()

    assert "egocom" in cfg.datasets
    assert "ego4d" in cfg.datasets


def test_get_raw_path():
    cfg = load_config()

    path = get_path(
        cfg,
        dataset="egocom",
        stage="raw",
        key="ground_truth",
    )

    assert path.name == "ground_truth_transcriptions.csv"
