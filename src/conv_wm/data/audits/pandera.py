"""Small reporting wrapper around Pandera validation."""

import json

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors


def audit_dataframe(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
) -> dict[str, object]:
    """Validate a dataframe lazily and return a JSON-serializable result."""
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as error:
        failure_cases = json.loads(error.failure_cases.to_json(orient="records"))
        return {
            "valid": False,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "n_failures": len(failure_cases),
            "failure_cases": failure_cases,
        }

    return {
        "valid": True,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "n_failures": 0,
        "failure_cases": [],
    }
