import io

from conv_wm.data.audits.population import run_population


def _audit(item: int) -> str:
    if item == 2:
        raise RuntimeError("boom")
    return f"ok-{item}"


def test_run_population_keeps_order_and_converts_errors():
    stream = io.StringIO()

    results = run_population(
        [1, 2, 3],
        _audit,
        on_error=lambda item, exc: f"failed-{item}-{exc}",
        label="probe",
        stream=stream,
    )

    assert results == ["ok-1", "failed-2-boom", "ok-3"]
    assert "probe 3/3" in stream.getvalue()


def test_run_population_threaded_keeps_order():
    results = run_population(
        list(range(20)),
        _audit,
        on_error=lambda item, exc: "failed",
        label="probe",
        max_workers=4,
        stream=None,
    )

    assert results[:2] == ["ok-0", "ok-1"]
    assert results[2] == "failed"
    assert results[3:] == [f"ok-{i}" for i in range(3, 20)]
