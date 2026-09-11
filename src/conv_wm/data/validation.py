"""Cross-table validation helpers not provided by Pandera."""

import pandas as pd


def check_foreign_key(
    child_df: pd.DataFrame,
    child_columns: list[str],
    parent_df: pd.DataFrame,
    parent_columns: list[str],
) -> int:
    """Return the number of child rows whose key is absent from the parent."""
    if len(child_columns) != len(parent_columns):
        raise ValueError("child_columns and parent_columns must have the same length")

    if any(column not in child_df.columns for column in child_columns):
        return 0

    if any(column not in parent_df.columns for column in parent_columns):
        return 0

    parent_keys = (
        parent_df[parent_columns]
        .drop_duplicates()
        .rename(columns=dict(zip(parent_columns, child_columns, strict=True)))
    )
    merged = child_df[child_columns].merge(
        parent_keys,
        on=child_columns,
        how="left",
        indicator=True,
    )
    return int(merged["_merge"].eq("left_only").sum())
