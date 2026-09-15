"""Small reporting wrapper around Pandera validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors


@dataclass(frozen=True)
class PanderaResult:
    """Lazy validation outcome of one table, JSON-serializable via ``to_dict``."""

    valid: bool
    shape: tuple[int, int]
    n_failures: int
    failure_cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping with ``shape`` as a list."""
        document = asdict(self)
        document["shape"] = list(self.shape)
        return document


def audit_dataframe(df: pd.DataFrame, schema: pa.DataFrameSchema) -> PanderaResult:
    """Validate a dataframe lazily and collect every failure case."""
    shape = (int(df.shape[0]), int(df.shape[1]))
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as error:
        failure_cases = json.loads(error.failure_cases.to_json(orient="records"))
        return PanderaResult(False, shape, len(failure_cases), failure_cases)
    return PanderaResult(True, shape, 0, [])
