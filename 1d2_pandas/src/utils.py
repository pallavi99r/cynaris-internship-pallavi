
from pathlib import Path
from typing import Callable

import pandas as pd


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a Pandas DataFrame."""
    return pd.read_csv(path, **kwargs)


def preview(df: pd.DataFrame, n: int = 10) -> None:
    """Display DataFrame shape, data types, and first n rows."""
    print("\nDataset shape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print(f"\nFirst {n} rows:")
    print(df.head(n))


def filter_rows(
    df: pd.DataFrame,
    condition: Callable
) -> pd.DataFrame:
    """Filter rows using a Boolean condition."""
    return df.loc[condition(df)].copy()


def group_agg(
    df: pd.DataFrame,
    by: list[str],
    agg_map: dict
) -> pd.DataFrame:
    """Group data and calculate aggregate values."""
    return (
        df.groupby(by, as_index=False)
        .agg(agg_map)
    )


def merge_dfs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | list[str],
    how: str = "inner"
) -> pd.DataFrame:
    """Merge two DataFrames."""
    return pd.merge(left, right, on=on, how=how)


def pivot_table(
    df: pd.DataFrame,
    index: str | list[str],
    columns: str,
    values: str,
    aggfunc: str = "sum"
) -> pd.DataFrame:
    """Create a pivot table."""
    return pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=0
    ).reset_index()


def export(
    df: pd.DataFrame,
    out_dir: Path,
    stem: str
) -> dict:
    """Export DataFrame to CSV and Parquet and compare sizes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{stem}.csv"
    parquet_path = out_dir / f"{stem}.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    return {
        "csv_bytes": csv_path.stat().st_size,
        "parquet_bytes": parquet_path.stat().st_size,
    }
