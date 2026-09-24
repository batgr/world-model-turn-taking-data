"""The generated registry page matches the registry: docs cannot drift."""

from conv_wm.data.labels.docs import DOC_PATH, registry_markdown


def test_the_committed_registry_page_is_up_to_date():
    assert DOC_PATH.read_text(encoding="utf-8") == registry_markdown(), (
        "run: uv run conv-wm labels docs"
    )


def test_every_label_is_documented():
    from conv_wm.data.labels.catalog import REGISTRY

    page = DOC_PATH.read_text(encoding="utf-8")
    for spec in REGISTRY:
        assert f"### `{spec.name}`" in page
