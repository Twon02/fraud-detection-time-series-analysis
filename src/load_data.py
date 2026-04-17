from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import PAYSIM_DTYPES, RAW_DATA_PATH


def load_sample(nrows: int = 10_000, path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        parent = path.parent
        available = sorted(p.name for p in parent.iterdir())[:50] if parent.exists() else []
        raise FileNotFoundError(
            f"Raw data was not found at {path}.\n"
            f"Available files in {parent}: {available}"
        )
    return pd.read_csv(path, nrows=nrows, dtype=PAYSIM_DTYPES)


def load_columns(
    columns: list[str] | None = None,
    nrows: int | None = None,
    path: str | Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    return pd.read_csv(path, usecols=columns, nrows=nrows, dtype=PAYSIM_DTYPES)


def load_in_chunks(chunksize: int = 250_000, path: str | Path = RAW_DATA_PATH):
    return pd.read_csv(path, dtype=PAYSIM_DTYPES, chunksize=chunksize)


def get_basic_schema(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=1_000)
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype_inferred": df.dtypes.astype(str).values,
        }
    )
