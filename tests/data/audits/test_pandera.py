import json

import pandas as pd
import pandera.pandas as pa

from conv_wm.data.audits.pandera import audit_dataframe


def test_audit_dataframe_returns_json_serializable_success():
    schema = pa.DataFrameSchema({"id": pa.Column(int, unique=True)})

    result = audit_dataframe(pd.DataFrame({"id": [1, 2]}), schema).to_dict()

    assert result == {
        "valid": True,
        "shape": [2, 1],
        "n_failures": 0,
        "failure_cases": [],
    }
    json.dumps(result)


def test_audit_dataframe_collects_lazy_failures():
    schema = pa.DataFrameSchema(
        {"id": pa.Column(int, unique=True, checks=pa.Check.gt(0))}
    )

    result = audit_dataframe(pd.DataFrame({"id": [0, 0]}), schema)

    assert result.valid is False
    assert result.n_failures >= 2
    assert result.failure_cases
    json.dumps(result.to_dict())
