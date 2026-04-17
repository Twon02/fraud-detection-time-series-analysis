from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import CORE_FEATURE_COLUMNS
from src.utils import safe_divide


DEFAULT_MODEL_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "step",
    "hour",
    "day",
    "is_transfer",
    "is_cash_out",
] + CORE_FEATURE_COLUMNS


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in DEFAULT_MODEL_FEATURES if column in df.columns]


def time_based_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = df.sort_values(["step", "row_id"] if "row_id" in df.columns else ["step"]).reset_index(drop=True)
    unique_steps = np.sort(working["step"].unique())
    train_idx = min(int(len(unique_steps) * train_fraction), len(unique_steps) - 1)
    valid_idx = min(int(len(unique_steps) * (train_fraction + validation_fraction)), len(unique_steps) - 1)
    train_cut = unique_steps[train_idx]
    valid_cut = unique_steps[valid_idx]

    train_df = working[working["step"] <= train_cut].copy()
    valid_df = working[(working["step"] > train_cut) & (working["step"] <= valid_cut)].copy()
    test_df = working[working["step"] > valid_cut].copy()
    return train_df, valid_df, test_df


def build_training_sample(
    df: pd.DataFrame,
    target_column: str = "isFraud",
    max_rows: int = 300_000,
    nonfraud_ratio: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    fraud = df[df[target_column] == 1]
    nonfraud = df[df[target_column] == 0]

    max_nonfraud = max(0, min(len(nonfraud), len(fraud) * nonfraud_ratio, max_rows - len(fraud)))
    nonfraud_sample = nonfraud.sample(n=max_nonfraud, random_state=random_state) if max_nonfraud < len(nonfraud) else nonfraud

    sampled = (
        pd.concat([fraud, nonfraud_sample], ignore_index=True)
        .sort_values(["step", "row_id"] if "row_id" in df.columns else ["step"])
        .reset_index(drop=True)
    )
    return sampled


def build_model_suite(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=14,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def select_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = safe_divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1], fill_value=0.0)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx])


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        {
            "model": [model_name],
            "threshold": [threshold],
            "roc_auc": [roc_auc_score(y_true, probabilities)],
            "average_precision": [average_precision_score(y_true, probabilities)],
            "precision": [float(safe_divide(((y_true == 1) & (predictions == 1)).sum(), (predictions == 1).sum()))],
            "recall": [float(safe_divide(((y_true == 1) & (predictions == 1)).sum(), (y_true == 1).sum()))],
            "f1": [f1_score(y_true, predictions, zero_division=0)],
            "alert_rate": [float(predictions.mean())],
        }
    )


def run_modeling_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "isFraud",
) -> dict[str, pd.DataFrame]:
    feature_columns = feature_columns or get_model_feature_columns(df)
    working = df.replace([np.inf, -np.inf], np.nan).copy()
    train_df, valid_df, test_df = time_based_split(working)
    sampled_train = build_training_sample(train_df, target_column=target_column)

    metric_tables = []
    scored_test_frames = []

    for model_name, pipeline in build_model_suite().items():
        pipeline.fit(sampled_train[feature_columns], sampled_train[target_column])

        valid_probabilities = pipeline.predict_proba(valid_df[feature_columns])[:, 1]
        test_probabilities = pipeline.predict_proba(test_df[feature_columns])[:, 1]
        threshold = select_best_threshold(valid_df[target_column], valid_probabilities)

        metric_tables.append(
            evaluate_model(model_name, test_df[target_column], test_probabilities, threshold)
        )

        scored = test_df[["step", target_column]].copy()
        scored["model"] = model_name
        scored["predicted_probability"] = test_probabilities
        scored["predicted_label"] = (test_probabilities >= threshold).astype(int)
        scored_test_frames.append(scored)

    split_summary = pd.DataFrame(
        {
            "split": ["train_full", "train_sample", "validation", "test"],
            "row_count": [len(train_df), len(sampled_train), len(valid_df), len(test_df)],
            "fraud_count": [
                int(train_df[target_column].sum()),
                int(sampled_train[target_column].sum()),
                int(valid_df[target_column].sum()),
                int(test_df[target_column].sum()),
            ],
        }
    )
    split_summary["fraud_rate"] = safe_divide(split_summary["fraud_count"], split_summary["row_count"])

    return {
        "model_split_summary": split_summary,
        "model_metrics": pd.concat(metric_tables, ignore_index=True).sort_values(
            ["average_precision", "roc_auc"],
            ascending=False,
        ),
        "model_test_scored_rows": pd.concat(scored_test_frames, ignore_index=True),
    }
