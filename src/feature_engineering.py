from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import safe_divide


CORE_FEATURE_COLUMNS = [
    "orig_txn_count_prior",
    "orig_amount_prior_sum",
    "orig_avg_amount_prior",
    "orig_prev_step",
    "orig_time_since_prev",
    "orig_prev_amount",
    "orig_amount_to_oldbalance_ratio",
    "orig_balance_drop",
    "orig_balance_drop_ratio",
    "orig_amount_vs_prior_avg",
    "dest_txn_count_prior",
    "dest_amount_prior_sum",
    "dest_avg_amount_prior",
    "dest_prev_step",
    "dest_time_since_prev",
    "amount_vs_dest_prior_avg",
    "orig_txn_count_last_1",
    "orig_amount_sum_last_1",
    "orig_txn_count_last_6",
    "orig_amount_sum_last_6",
    "orig_txn_count_last_24",
    "orig_amount_sum_last_24",
]


FEATURE_INVENTORY_DETAILS = {
    "orig_txn_count_prior": (
        "Number of prior transactions completed by the sender",
        "Bursty origin accounts can indicate scripted or mule-like behavior",
    ),
    "orig_amount_prior_sum": (
        "Total historical amount sent by the origin account before the current row",
        "Very high prior throughput can separate active operational accounts from typical customers",
    ),
    "orig_avg_amount_prior": (
        "Average historical outgoing amount for the sender before the current row",
        "Fraud often appears as a sharp deviation from a sender's prior size pattern",
    ),
    "orig_prev_step": (
        "Time step of the sender's previous transaction",
        "Provides a direct reference point for sender recency",
    ),
    "orig_time_since_prev": (
        "Gap between the current transaction and the sender's previous transaction",
        "Short time gaps can expose sudden bursts of activity",
    ),
    "orig_prev_amount": (
        "Amount of the sender's previous transaction",
        "Lets the model compare the current amount with the most recent behavior",
    ),
    "orig_amount_to_oldbalance_ratio": (
        "Current amount divided by the sender's pre-transaction balance",
        "Near-drain transactions are often operationally risky",
    ),
    "orig_balance_drop": (
        "Observed sender balance drop during the transaction",
        "Large balance drops can highlight account-draining behavior",
    ),
    "orig_balance_drop_ratio": (
        "Sender balance drop as a share of old balance",
        "Normalizes the drop so large and small accounts can be compared fairly",
    ),
    "orig_amount_vs_prior_avg": (
        "Current amount relative to the sender's historical average amount",
        "Large jumps above normal can represent anomalous transfer behavior",
    ),
    "dest_txn_count_prior": (
        "Number of prior transactions received by the destination account",
        "New or lightly used destinations may carry higher transfer risk",
    ),
    "dest_amount_prior_sum": (
        "Total amount previously received by the destination account",
        "Helps distinguish established destinations from newly activated sinks",
    ),
    "dest_avg_amount_prior": (
        "Average historical incoming amount for the destination account",
        "Sudden jumps against a destination's normal intake may be suspicious",
    ),
    "dest_prev_step": (
        "Time step of the destination account's previous observed transaction",
        "Tracks recent destination-side activity",
    ),
    "dest_time_since_prev": (
        "Gap since the destination account's previous transaction",
        "Rapid reuse of a destination account can signal suspicious concentration",
    ),
    "amount_vs_dest_prior_avg": (
        "Current amount relative to the destination's prior average received amount",
        "Detects inflows that are unusually large for that destination",
    ),
    "orig_txn_count_last_1": (
        "Sender transaction count in the immediately preceding step",
        "Captures ultra-short burstiness",
    ),
    "orig_amount_sum_last_1": (
        "Sender amount total in the immediately preceding step",
        "Flags accounts that already moved significant value in the last hour",
    ),
    "orig_txn_count_last_6": (
        "Sender transaction count over the prior 6 steps",
        "Highlights multi-hour clustering",
    ),
    "orig_amount_sum_last_6": (
        "Sender amount total over the prior 6 steps",
        "Measures recent sender throughput over a medium window",
    ),
    "orig_txn_count_last_24": (
        "Sender transaction count over the prior 24 steps",
        "Creates a daily-style activity baseline",
    ),
    "orig_amount_sum_last_24": (
        "Sender amount total over the prior 24 steps",
        "Shows whether the sender has been moving unusual daily value",
    ),
}


def sort_transactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_id"] = np.arange(len(out))
    return out.sort_values(["step", "row_id"]).reset_index(drop=True)


def add_time_helpers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # PaySim `step` is already an hourly sequence, so we keep two distinct
    # temporal views:
    # - `hour`: hour-of-day style position (1-24)
    # - `day`: zero-based day bucket aligned to the saved monitoring outputs
    out["hour"] = ((out["step"] - 1) % 24) + 1
    out["day"] = (out["step"] - 1) // 24
    out["is_transfer"] = (out["type"] == "TRANSFER").astype("int8")
    out["is_cash_out"] = (out["type"] == "CASH_OUT").astype("int8")
    return out


def add_sender_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("nameOrig", sort=False)

    out["orig_txn_count_prior"] = grouped.cumcount()
    out["orig_amount_cum"] = grouped["amount"].cumsum()
    out["orig_amount_prior_sum"] = out["orig_amount_cum"] - out["amount"]
    out["orig_avg_amount_prior"] = safe_divide(
        out["orig_amount_prior_sum"],
        out["orig_txn_count_prior"],
        fill_value=np.nan,
    )
    out["orig_prev_step"] = grouped["step"].shift(1)
    out["orig_time_since_prev"] = out["step"] - out["orig_prev_step"]
    out["orig_prev_amount"] = grouped["amount"].shift(1)
    return out.drop(columns=["orig_amount_cum"])


def add_balance_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["orig_amount_to_oldbalance_ratio"] = safe_divide(
        out["amount"],
        out["oldbalanceOrg"],
        fill_value=np.nan,
    )
    out["orig_balance_drop"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["orig_balance_drop_ratio"] = safe_divide(
        out["orig_balance_drop"],
        out["oldbalanceOrg"],
        fill_value=np.nan,
    )
    out["orig_amount_vs_prior_avg"] = safe_divide(
        out["amount"],
        out["orig_avg_amount_prior"],
        fill_value=np.nan,
    )
    return out


def add_destination_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("nameDest", sort=False)

    out["dest_txn_count_prior"] = grouped.cumcount()
    out["dest_amount_cum"] = grouped["amount"].cumsum()
    out["dest_amount_prior_sum"] = out["dest_amount_cum"] - out["amount"]
    out["dest_avg_amount_prior"] = safe_divide(
        out["dest_amount_prior_sum"],
        out["dest_txn_count_prior"],
        fill_value=np.nan,
    )
    out["dest_prev_step"] = grouped["step"].shift(1)
    out["dest_time_since_prev"] = out["step"] - out["dest_prev_step"]
    out["amount_vs_dest_prior_avg"] = safe_divide(
        out["amount"],
        out["dest_avg_amount_prior"],
        fill_value=np.nan,
    )
    return out.drop(columns=["dest_amount_cum"])


def add_sender_window_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (1, 6, 24),
) -> pd.DataFrame:
    out = df.copy()
    orig_step_agg = (
        out.groupby(["nameOrig", "step"], as_index=False)
        .agg(
            step_txn_count=("amount", "size"),
            step_amount_sum=("amount", "sum"),
        )
        .sort_values(["nameOrig", "step"])
        .reset_index(drop=True)
    )

    grouped = orig_step_agg.groupby("nameOrig", sort=False)
    shifted_count = grouped["step_txn_count"].shift(1)
    shifted_amount = grouped["step_amount_sum"].shift(1)

    for window in windows:
        orig_step_agg[f"orig_txn_count_last_{window}"] = (
            shifted_count.groupby(orig_step_agg["nameOrig"], sort=False)
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        orig_step_agg[f"orig_amount_sum_last_{window}"] = (
            shifted_amount.groupby(orig_step_agg["nameOrig"], sort=False)
            .rolling(window=window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    merge_columns = ["nameOrig", "step"]
    for window in windows:
        merge_columns.extend(
            [
                f"orig_txn_count_last_{window}",
                f"orig_amount_sum_last_{window}",
            ]
        )

    return out.merge(orig_step_agg[merge_columns], on=["nameOrig", "step"], how="left")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in CORE_FEATURE_COLUMNS if column in df.columns]


def build_model_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (1, 6, 24),
) -> pd.DataFrame:
    out = sort_transactions(df)
    out = add_time_helpers(out)
    out = add_sender_history_features(out)
    out = add_balance_ratio_features(out)
    out = add_destination_history_features(out)
    out = add_sender_window_features(out, windows=windows)
    return out


def build_feature_inventory() -> pd.DataFrame:
    rows = []
    for feature_name in CORE_FEATURE_COLUMNS:
        what_it_measures, why_it_matters = FEATURE_INVENTORY_DETAILS[feature_name]
        rows.append(
            {
                "feature_name": feature_name,
                "what_it_measures": what_it_measures,
                "why_it_might_indicate_risk": why_it_matters,
                "strictly_past_only": True,
                "safe_for_modeling": True,
            }
        )
    return pd.DataFrame(rows)


def save_feature_artifacts(
    df: pd.DataFrame,
    features_path: str | Path,
    inventory_path: str | Path,
) -> None:
    features_path = Path(features_path)
    inventory_path = Path(inventory_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path, index=False)
    build_feature_inventory().to_csv(inventory_path, index=False)
