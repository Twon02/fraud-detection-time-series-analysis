from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "paysim dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"


PAYSIM_DTYPES = {
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


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(
    numerator: Any,
    denominator: Any,
    fill_value: float = 0.0,
) -> np.ndarray | pd.Series:
    numerator_array = np.asarray(numerator, dtype="float64")
    denominator_array = np.asarray(denominator, dtype="float64")
    result = np.full(numerator_array.shape, fill_value, dtype="float64")

    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=denominator_array != 0,
    )

    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index, name=numerator.name)
    return result


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    true_positives = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negatives = int(((y_true == 1) & (y_pred == 0)).sum())
    true_negatives = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    false_positive_rate = safe_divide(false_positives, false_positives + true_negatives)
    alert_rate = safe_divide(y_pred.sum(), len(y_pred))

    return pd.DataFrame(
        {
            "metric": [
                "true_positives",
                "false_positives",
                "false_negatives",
                "true_negatives",
                "precision",
                "recall",
                "false_positive_rate",
                "alert_rate",
            ],
            "value": [
                true_positives,
                false_positives,
                false_negatives,
                true_negatives,
                float(np.asarray(precision).item()),
                float(np.asarray(recall).item()),
                float(np.asarray(false_positive_rate).item()),
                float(np.asarray(alert_rate).item()),
            ],
        }
    )
