from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import add_time_helpers
from src.utils import classification_metrics, ensure_directory, safe_divide


def _with_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df if {"hour", "day"}.issubset(df.columns) else add_time_helpers(df)


def build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = _with_time_columns(df)
    return pd.DataFrame(
        {
            "metric": [
                "total_transactions",
                "total_fraud_transactions",
                "fraud_rate",
                "total_amount",
                "average_amount",
                "median_amount",
                "min_step",
                "max_step",
                "total_hours",
                "total_days_covered",
            ],
            "value": [
                len(out),
                int(out["isFraud"].sum()),
                float(out["isFraud"].mean()),
                float(out["amount"].sum()),
                float(out["amount"].mean()),
                float(out["amount"].median()),
                int(out["step"].min()),
                int(out["step"].max()),
                int(out["step"].max()),
                int(out["day"].max() + 1),
            ],
        }
    )


def build_amount_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    amounts = df["amount"]
    return pd.DataFrame(
        {
            "statistic": ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"],
            "value": [
                float(amounts.min()),
                float(amounts.quantile(0.01)),
                float(amounts.quantile(0.05)),
                float(amounts.quantile(0.25)),
                float(amounts.median()),
                float(amounts.quantile(0.75)),
                float(amounts.quantile(0.95)),
                float(amounts.quantile(0.99)),
                float(amounts.max()),
            ],
        }
    )


def build_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("type", observed=True)
        .agg(
            txn_count=("type", "size"),
            fraud_count=("isFraud", "sum"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            q95_amount=("amount", lambda s: float(s.quantile(0.95))),
            q99_amount=("amount", lambda s: float(s.quantile(0.99))),
        )
        .reset_index()
    )

    out["fraud_rate"] = safe_divide(out["fraud_count"], out["txn_count"])
    out["share_of_all_txns"] = safe_divide(out["txn_count"], out["txn_count"].sum())
    out["share_of_all_amount"] = safe_divide(out["total_amount"], out["total_amount"].sum())
    out["share_of_all_fraud"] = safe_divide(out["fraud_count"], out["fraud_count"].sum())
    return out.sort_values("fraud_rate", ascending=False).reset_index(drop=True)


def build_amount_by_fraud(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.assign(fraud_label=np.where(df["isFraud"] == 1, "Fraud", "Non-Fraud"))
        .groupby("fraud_label", observed=True)
        .agg(
            txn_count=("fraud_label", "size"),
            avg_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            q95_amount=("amount", lambda s: float(s.quantile(0.95))),
            q99_amount=("amount", lambda s: float(s.quantile(0.99))),
            total_amount=("amount", "sum"),
        )
        .reset_index()
    )
    return out


def build_fraud_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fraud_label"] = np.where(out["isFraud"] == 1, "Fraud", "Non-Fraud")
    summary = (
        out.groupby("fraud_label", observed=True)
        .agg(
            txn_count=("fraud_label", "size"),
            avg_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            avg_oldbalanceOrg=("oldbalanceOrg", "mean"),
            avg_newbalanceOrig=("newbalanceOrig", "mean"),
            avg_oldbalanceDest=("oldbalanceDest", "mean"),
            avg_newbalanceDest=("newbalanceDest", "mean"),
            median_oldbalanceOrg=("oldbalanceOrg", "median"),
            median_newbalanceOrig=("newbalanceOrig", "median"),
            median_oldbalanceDest=("oldbalanceDest", "median"),
            median_newbalanceDest=("newbalanceDest", "median"),
            avg_step=("step", "mean"),
            median_step=("step", "median"),
        )
        .reset_index()
    )
    return summary[
        [
            "txn_count",
            "avg_amount",
            "median_amount",
            "avg_oldbalanceOrg",
            "avg_newbalanceOrig",
            "avg_oldbalanceDest",
            "avg_newbalanceDest",
            "median_oldbalanceOrg",
            "median_newbalanceOrig",
            "median_oldbalanceDest",
            "median_newbalanceDest",
            "avg_step",
            "median_step",
            "fraud_label",
        ]
    ]


def build_flag_vs_actual_counts(df: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(
        df["isFlaggedFraud"],
        df["isFraud"],
        margins=True,
    ).reset_index()


def build_flagged_fraud_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    return classification_metrics(df["isFraud"], df["isFlaggedFraud"])


def build_flagged_vs_actual_by_type(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("type", observed=True)
        .agg(
            txn_count=("type", "size"),
            fraud_count=("isFraud", "sum"),
            flagged_count=("isFlaggedFraud", "sum"),
        )
        .reset_index()
    )
    out["fraud_rate"] = safe_divide(out["fraud_count"], out["txn_count"])
    out["flag_rate"] = safe_divide(out["flagged_count"], out["txn_count"])
    return out


def build_account_summary(
    df: pd.DataFrame,
    account_column: str,
    amount_label: str,
) -> pd.DataFrame:
    out = (
        df.groupby(account_column, observed=True)
        .agg(
            txn_count=(account_column, "size"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            fraud_count=("isFraud", "sum"),
            first_step=("step", "min"),
            last_step=("step", "max"),
        )
        .reset_index()
        .rename(columns={"total_amount": amount_label})
    )
    out["fraud_rate"] = safe_divide(out["fraud_count"], out["txn_count"])
    out["active_span_steps"] = out["last_step"] - out["first_step"]
    return out


def build_period_monitoring(
    df: pd.DataFrame,
    period_column: str,
    rolling_window: int,
    output_column: str | None = None,
) -> pd.DataFrame:
    output_column = output_column or period_column
    out = (
        df.groupby(period_column, observed=True)
        .agg(
            txn_count=("step", "size"),
            fraud_count=("isFraud", "sum"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            fraud_amount=("amount", lambda s: float(s[df.loc[s.index, "isFraud"] == 1].sum())),
        )
        .reset_index()
        .sort_values(period_column)
    )
    if output_column != period_column:
        out = out.rename(columns={period_column: output_column})
    period_column = output_column

    out["fraud_rate"] = safe_divide(out["fraud_count"], out["txn_count"])
    out["fraud_amount_share"] = safe_divide(out["fraud_amount"], out["total_amount"])
    out[f"txn_count_roll_mean_{rolling_window}"] = out["txn_count"].rolling(rolling_window, min_periods=1).mean()
    out[f"fraud_count_roll_mean_{rolling_window}"] = out["fraud_count"].rolling(rolling_window, min_periods=1).mean()
    out[f"fraud_rate_roll_mean_{rolling_window}"] = out["fraud_rate"].rolling(rolling_window, min_periods=1).mean()
    out[f"fraud_rate_roll_std_{rolling_window}"] = (
        out["fraud_rate"].rolling(rolling_window, min_periods=2).std().fillna(0.0)
    )
    if period_column == "hour":
        out[f"avg_amount_roll_mean_{rolling_window}"] = out["avg_amount"].rolling(rolling_window, min_periods=1).mean()

    std_column = f"fraud_rate_roll_std_{rolling_window}"
    mean_column = f"fraud_rate_roll_mean_{rolling_window}"
    out["fraud_rate_zscore_like"] = safe_divide(
        out["fraud_rate"] - out[mean_column],
        out[std_column],
        fill_value=0.0,
    )

    if period_column == "hour":
        out["fraud_count_spike_flag"] = out["fraud_count"] > (
            out[f"fraud_count_roll_mean_{rolling_window}"] + 3 * out["fraud_count"].rolling(rolling_window, min_periods=2).std().fillna(0.0)
        )
    out["fraud_rate_spike_flag"] = out["fraud_rate_zscore_like"] > 3
    return out.reset_index(drop=True)


def build_type_monitoring(df: pd.DataFrame, period_column: str) -> pd.DataFrame:
    out = (
        df.groupby([period_column, "type"], observed=True)
        .agg(
            txn_count=("type", "size"),
            fraud_count=("isFraud", "sum"),
            total_amount=("amount", "sum"),
        )
        .reset_index()
        .sort_values([period_column, "type"])
    )
    out["fraud_rate"] = safe_divide(out["fraud_count"], out["txn_count"])
    return out


def run_monitoring_report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = _with_time_columns(df)
    out = out.assign(monitor_hour=out["step"], monitor_day=(out["step"] - 1) // 24)
    return {
        "overall_summary": build_overall_summary(out),
        "amount_distribution_summary": build_amount_distribution_summary(out),
        "type_summary": build_type_summary(out),
        "amount_by_fraud": build_amount_by_fraud(out),
        "fraud_profile": build_fraud_profile(out),
        "flag_vs_actual_counts": build_flag_vs_actual_counts(out),
        "flagged_fraud_evaluation": build_flagged_fraud_evaluation(out),
        "flagged_vs_actual_by_type": build_flagged_vs_actual_by_type(out),
        "origin_account_summary": build_account_summary(out, "nameOrig", "total_sent"),
        "destination_account_summary": build_account_summary(out, "nameDest", "total_received"),
        "hourly_monitoring": build_period_monitoring(out, "monitor_hour", rolling_window=24, output_column="hour"),
        "daily_monitoring": build_period_monitoring(out, "monitor_day", rolling_window=7, output_column="day"),
        "hourly_type_monitoring": build_type_monitoring(
            out[["monitor_hour", "type", "amount", "isFraud"]].rename(columns={"monitor_hour": "hour"}),
            "hour",
        ),
        "daily_type_monitoring": build_type_monitoring(
            out[["monitor_day", "type", "amount", "isFraud"]].rename(columns={"monitor_day": "day"}),
            "day",
        ),
    }


def save_monitoring_outputs(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = ensure_directory(output_dir)
    for name, table in outputs.items():
        table.to_csv(Path(output_dir) / f"{name}.csv", index=False)
