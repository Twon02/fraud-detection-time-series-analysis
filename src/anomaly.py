from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import CORE_FEATURE_COLUMNS
from src.modeling import time_based_split
from src.utils import safe_divide


DEFAULT_ANOMALY_FEATURES = [
    "step",
    "hour",
    "day",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "is_transfer",
    "is_cash_out",
] + CORE_FEATURE_COLUMNS


def get_anomaly_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in DEFAULT_ANOMALY_FEATURES if column in df.columns]


def prepare_anomaly_input(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = feature_columns or get_anomaly_feature_columns(df)
    working = df.copy()
    working = working.replace([np.inf, -np.inf], np.nan)
    return working, feature_columns


def fit_isolation_forest(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    contamination: float = 0.01,
    random_state: int = 42,
) -> tuple[Pipeline, list[str]]:
    working, feature_columns = prepare_anomaly_input(df, feature_columns)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    contamination=contamination,
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(working[feature_columns])
    return pipeline, feature_columns


def score_anomalies(
    df: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
    top_n: int = 5_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working, _ = prepare_anomaly_input(df, feature_columns)
    scores = -model.score_samples(working[feature_columns])
    alerts = working.copy()
    alerts["anomaly_score"] = scores
    alerts["anomaly_rank"] = alerts["anomaly_score"].rank(method="first", ascending=False).astype(int)
    alerts = alerts.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    top_alerts = alerts.head(top_n).copy()
    return alerts, top_alerts


def summarize_topk_precision(
    scored_df: pd.DataFrame,
    top_ks: tuple[int, ...] = (100, 500, 1_000, 5_000),
) -> pd.DataFrame:
    rows = []
    for top_k in top_ks:
        subset = scored_df.head(top_k)
        fraud_hits = int(subset["isFraud"].sum())
        rows.append(
            {
                "top_k": top_k,
                "fraud_hits": fraud_hits,
                "precision_at_k": float(safe_divide(fraud_hits, top_k)),
                "recall_at_k": float(safe_divide(fraud_hits, scored_df["isFraud"].sum())),
            }
        )
    return pd.DataFrame(rows)


def summarize_alert_segments(scored_df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    out = scored_df.copy()
    out["alert_decile"] = pd.qcut(
        out["anomaly_score"].rank(method="first"),
        q=bins,
        labels=[f"D{i}" for i in range(1, bins + 1)],
    )
    summary = (
        out.groupby("alert_decile", observed=True)
        .agg(
            txn_count=("alert_decile", "size"),
            fraud_count=("isFraud", "sum"),
            avg_anomaly_score=("anomaly_score", "mean"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )
    summary["fraud_rate"] = safe_divide(summary["fraud_count"], summary["txn_count"])
    return summary.sort_values("avg_anomaly_score", ascending=False).reset_index(drop=True)


def run_anomaly_detection(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    contamination: float = 0.01,
    top_n: int = 5_000,
) -> dict[str, pd.DataFrame]:
    working, feature_columns = prepare_anomaly_input(df, feature_columns)
    train_df, valid_df, test_df = time_based_split(working)
    fit_df = pd.concat([train_df, valid_df], ignore_index=True)

    model, feature_columns = fit_isolation_forest(
        df=fit_df,
        feature_columns=feature_columns,
        contamination=contamination,
    )
    scored_df, top_alerts = score_anomalies(test_df, model, feature_columns, top_n=top_n)
    return {
        "anomaly_split_summary": pd.DataFrame(
            {
                "split": ["fit", "test"],
                "row_count": [len(fit_df), len(test_df)],
                "fraud_count": [int(fit_df["isFraud"].sum()), int(test_df["isFraud"].sum())],
                "min_step": [int(fit_df["step"].min()), int(test_df["step"].min())],
                "max_step": [int(fit_df["step"].max()), int(test_df["step"].max())],
            }
        ),
        "anomaly_topk_summary": summarize_topk_precision(scored_df),
        "anomaly_decile_summary": summarize_alert_segments(scored_df),
        "anomaly_top_alerts": top_alerts,
    }
