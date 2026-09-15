"""Shared mechanics of population audits: iterate, capture failures, report progress.

Scientific logic never lives here. A population audit supplies an ``audit``
callable for one item and an ``on_error`` factory that turns an exception into
an explicit failed record, so error semantics are visible at the call site.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TextIO


def run_population[Item, Record](
    items: Sequence[Item],
    audit: Callable[[Item], Record],
    *,
    on_error: Callable[[Item, Exception], Record],
    label: str,
    max_workers: int = 1,
    stream: TextIO | None = sys.stderr,
) -> list[Record]:
    """Apply ``audit`` to every item, in input order, converting exceptions to records.

    ``max_workers > 1`` runs items in a thread pool (suited to subprocess-bound
    work such as ffprobe); results keep the input order either way. Progress is
    written to ``stream`` as ``<label> i/n``; pass ``None`` to silence it.
    """

    def guarded(item: Item) -> Record:
        try:
            return audit(item)
        except Exception as exc:  # noqa: BLE001 - population boundary: record and continue
            return on_error(item, exc)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                _with_progress(
                    executor.map(guarded, items),
                    total=len(items),
                    label=label,
                    stream=stream,
                )
            )
    return list(
        _with_progress(
            map(guarded, items), total=len(items), label=label, stream=stream
        )
    )


def _with_progress[Record](
    iterator: Iterable[Record], *, total: int, label: str, stream: TextIO | None
) -> Iterable[Record]:
    for index, record in enumerate(iterator, start=1):
        if stream is not None:
            print(f"\r{label} {index}/{total}", end="", file=stream, flush=True)
        yield record
    if stream is not None and total:
        print(file=stream)
