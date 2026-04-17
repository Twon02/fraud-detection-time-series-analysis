from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float32",
    "nameOrig": "string",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "nameDest": "string",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def load_full_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load the full dataset with optimized dtypes.

    Use this only if your machine can comfortably hold the dataset in memory.
    """
    path = Path(path)
    return pd.read_csv(path, dtype=DTYPES)


def load_in_chunks(path: str | Path, chunksize: int = 250_000) -> Iterable[pd.DataFrame]:
    """
    Stream the dataset in chunks for memory-safe audit operations.
    """
    path = Path(path)
    return pd.read_csv(path, dtype=DTYPES, chunksize=chunksize)


def get_row_count(path: str | Path) -> int:
    """
    Count data rows without loading the full dataset into memory.
    Subtract 1 for the header row.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def count_missing_values_chunked(path: str | Path, chunksize: int = 250_000) -> pd.Series:
    """
    Count missing values across the full dataset using chunked processing.
    """
    total_missing = None

    for chunk in load_in_chunks(path, chunksize=chunksize):
        chunk_missing = chunk.isnull().sum()

        if total_missing is None:
            total_missing = chunk_missing
        else:
            total_missing = total_missing.add(chunk_missing, fill_value=0)

    return total_missing.astype("int64").sort_values(ascending=False)


def count_exact_duplicate_rows_chunked(path: str | Path, chunksize: int = 250_000) -> int:
    """
    Count exact duplicate rows across the full dataset using row hashes.

    This is more memory-safe than loading the entire dataset and calling duplicated().
    It still uses memory proportional to the number of unique row hashes seen.
    """
    seen_hashes = set()
    duplicate_count = 0

    for chunk in load_in_chunks(path, chunksize=chunksize):
        row_hashes = pd.util.hash_pandas_object(chunk, index=False)

        for row_hash in row_hashes.to_numpy():
            if row_hash in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(row_hash)

    return duplicate_count


def compute_validation_flags(df: pd.DataFrame, tolerance: float = 1e-3) -> pd.DataFrame:
    """
    Add rule-based validation flags to a DataFrame.

    Important:
    Some balance relationships in PaySim-style data are known to be imperfect,
    so these should be treated as suspiciousness checks, not absolute truth.
    """
    out = df.copy()

    out["flag_amount_negative"] = out["amount"] < 0
    out["flag_amount_zero"] = out["amount"] == 0

    for col in BALANCE_COLUMNS:
        out[f"flag_{col}_negative"] = out[col] < 0

    # Sender-side balance consistency:
    # expected newbalanceOrig ~= oldbalanceOrg - amount
    out["sender_balance_error"] = (
        out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
    ).abs()

    out["flag_sender_balance_mismatch"] = out["sender_balance_error"] > tolerance

    # Destination-side balance consistency:
    # expected newbalanceDest ~= oldbalanceDest + amount
    out["dest_balance_error"] = (
        out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    ).abs()

    out["flag_dest_balance_mismatch"] = out["dest_balance_error"] > tolerance

    # Zero-balance edge cases that are often suspicious in this dataset
    out["flag_org_balance_zero_before_nonzero_amount"] = (
        (out["oldbalanceOrg"] == 0) & (out["amount"] > 0)
    )

    out["flag_dest_balance_zero_before_nonzero_amount"] = (
        (out["oldbalanceDest"] == 0) & (out["amount"] > 0)
    )

    # Merchant-like destination heuristic
    out["flag_dest_is_merchant"] = out["nameDest"].astype("string").str.startswith("M", na=False)

    return out


def summarize_validation_flags(df_flagged: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize all flag columns as counts and percentages.
    """
    flag_cols = [col for col in df_flagged.columns if col.startswith("flag_")]

    summary = pd.DataFrame({
        "flag_name": flag_cols,
        "flag_count": [int(df_flagged[col].sum()) for col in flag_cols],
        "flag_rate": [float(df_flagged[col].mean()) for col in flag_cols],
    }).sort_values(["flag_count", "flag_rate"], ascending=False)

    return summary.reset_index(drop=True)


def summarize_flags_by_type_and_fraud(df_flagged: pd.DataFrame) -> pd.DataFrame:
    """
    Show how suspicious flags distribute by transaction type and fraud label.
    """
    flag_cols = [col for col in df_flagged.columns if col.startswith("flag_")]

    grouped = (
        df_flagged
        .groupby(["type", "isFraud"], observed=True)[flag_cols]
        .mean()
        .reset_index()
    )

    return grouped


def get_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute class counts and rates for isFraud.
    """
    counts = df["isFraud"].value_counts(dropna=False).sort_index()
    rates = df["isFraud"].value_counts(normalize=True, dropna=False).sort_index()

    result = pd.DataFrame({
        "isFraud": counts.index.astype(int),
        "count": counts.values,
        "rate": rates.values,
    })

    return result


def get_type_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count transactions, fraud count, and fraud rate by transaction type.
    """
    summary = (
        df.groupby("type", observed=True)
          .agg(
              txn_count=("type", "size"),
              fraud_count=("isFraud", "sum"),
              avg_amount=("amount", "mean"),
              median_amount=("amount", "median"),
          )
          .reset_index()
    )

    summary["fraud_rate"] = summary["fraud_count"] / summary["txn_count"]
    return summary.sort_values("fraud_rate", ascending=False)


def profile_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a compact min/max/profile table for numeric columns.
    """
    rows = []

    for col in NUMERIC_COLUMNS:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
        })

    return pd.DataFrame(rows)


def run_chunked_quality_audit(path: str | Path, chunksize: int = 250_000) -> Dict[str, pd.DataFrame | int | pd.Series]:
    """
    Run a full-dataset quality audit in a memory-aware way.

    Returns a dictionary of outputs that can be saved or displayed.
    """
    path = Path(path)

    row_count = 0
    chunk_count = 0

    missing_total = None
    duplicate_hashes_seen = set()
    duplicate_rows = 0

    flag_count_accumulator = None
    class_count_accumulator = None
    type_summary_accumulator = None

    for chunk in load_in_chunks(path, chunksize=chunksize):
        chunk_count += 1
        row_count += len(chunk)

        # Missing values
        chunk_missing = chunk.isnull().sum()
        if missing_total is None:
            missing_total = chunk_missing
        else:
            missing_total = missing_total.add(chunk_missing, fill_value=0)

        # Exact duplicate rows via hashes
        row_hashes = pd.util.hash_pandas_object(chunk, index=False)
        for row_hash in row_hashes.to_numpy():
            if row_hash in duplicate_hashes_seen:
                duplicate_rows += 1
            else:
                duplicate_hashes_seen.add(row_hash)

        # Validation flags
        flagged = compute_validation_flags(chunk)
        chunk_flag_summary = summarize_validation_flags(flagged)

        if flag_count_accumulator is None:
            flag_count_accumulator = chunk_flag_summary.copy()
        else:
            flag_count_accumulator = (
                flag_count_accumulator[["flag_name", "flag_count"]]
                .merge(
                    chunk_flag_summary[["flag_name", "flag_count"]],
                    on="flag_name",
                    how="outer",
                    suffixes=("_old", "_new"),
                )
                .fillna(0)
            )
            flag_count_accumulator["flag_count"] = (
                flag_count_accumulator["flag_count_old"] + flag_count_accumulator["flag_count_new"]
            )
            flag_count_accumulator = flag_count_accumulator[["flag_name", "flag_count"]]

        # Class balance
        chunk_class = chunk["isFraud"].value_counts().sort_index()
        if class_count_accumulator is None:
            class_count_accumulator = chunk_class
        else:
            class_count_accumulator = class_count_accumulator.add(chunk_class, fill_value=0)

        # Type risk summary
        chunk_type_summary = (
            chunk.groupby("type", observed=True)
                 .agg(
                     txn_count=("type", "size"),
                     fraud_count=("isFraud", "sum"),
                     amount_sum=("amount", "sum"),
                 )
                 .reset_index()
        )

        if type_summary_accumulator is None:
            type_summary_accumulator = chunk_type_summary
        else:
            type_summary_accumulator = (
                pd.concat([type_summary_accumulator, chunk_type_summary], ignore_index=True)
                  .groupby("type", observed=True, as_index=False)
                  .agg(
                      txn_count=("txn_count", "sum"),
                      fraud_count=("fraud_count", "sum"),
                      amount_sum=("amount_sum", "sum"),
                  )
            )

    missing_total = missing_total.astype("int64").sort_values(ascending=False)

    flag_count_accumulator["flag_rate"] = flag_count_accumulator["flag_count"] / row_count
    flag_count_accumulator = flag_count_accumulator.sort_values("flag_count", ascending=False).reset_index(drop=True)

    class_balance = pd.DataFrame({
        "isFraud": class_count_accumulator.index.astype(int),
        "count": class_count_accumulator.values.astype(int),
    })
    class_balance["rate"] = class_balance["count"] / row_count

    type_summary_accumulator["fraud_rate"] = (
        type_summary_accumulator["fraud_count"] / type_summary_accumulator["txn_count"]
    )
    type_summary_accumulator["avg_amount"] = (
        type_summary_accumulator["amount_sum"] / type_summary_accumulator["txn_count"]
    )
    type_summary_accumulator = type_summary_accumulator.sort_values("fraud_rate", ascending=False)

    audit_summary = pd.DataFrame({
        "metric": [
            "row_count",
            "chunk_count",
            "duplicate_rows",
            "fraud_count",
            "fraud_rate",
        ],
        "value": [
            row_count,
            chunk_count,
            duplicate_rows,
            int(class_balance.loc[class_balance["isFraud"] == 1, "count"].sum()),
            float(class_balance.loc[class_balance["isFraud"] == 1, "rate"].sum()),
        ]
    })

    return {
        "audit_summary": audit_summary,
        "missing_values": missing_total,
        "duplicate_rows": duplicate_rows,
        "flag_summary": flag_count_accumulator,
        "class_balance": class_balance,
        "type_risk_summary": type_summary_accumulator,
    }


def save_audit_outputs(outputs: Dict[str, pd.DataFrame | int | pd.Series], output_dir: str | Path) -> None:
    """
    Save audit results to CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{key}.csv", index=False)
        elif isinstance(value, pd.Series):
            value.to_csv(output_dir / f"{key}.csv", header=True)
        else:
            pd.DataFrame({"value": [value]}).to_csv(output_dir / f"{key}.csv", index=False)