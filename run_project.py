from __future__ import annotations

import argparse

import pandas as pd

from src.anomaly import run_anomaly_detection
from src.feature_engineering import build_model_features, save_feature_artifacts
from src.monitoring import run_monitoring_report, save_monitoring_outputs
from src.preprocessing import run_chunked_quality_audit, save_audit_outputs
from src.modeling import run_modeling_pipeline
from src.utils import OUTPUTS_DIR, PAYSIM_DTYPES, PROCESSED_DIR, RAW_DATA_PATH, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more phases of the PaySim fraud monitoring project.")
    parser.add_argument(
        "--phase",
        choices=["audit", "monitoring", "features", "anomaly", "modeling", "all"],
        default="all",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for development runs.")
    return parser.parse_args()


def load_raw_frame(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH, dtype=PAYSIM_DTYPES, nrows=nrows)


def load_or_build_features(nrows: int | None = None) -> pd.DataFrame:
    features_path = PROCESSED_DIR / "model_features.parquet"
    if features_path.exists():
        return pd.read_parquet(features_path)
    return build_model_features(load_raw_frame(nrows=nrows))


def run_phase_audit() -> None:
    outputs = run_chunked_quality_audit(RAW_DATA_PATH)
    save_audit_outputs(outputs, OUTPUTS_DIR / "phase_2_audit")


def run_phase_monitoring(nrows: int | None = None) -> None:
    outputs = run_monitoring_report(load_raw_frame(nrows=nrows))
    save_monitoring_outputs(outputs, OUTPUTS_DIR / "phase_3_eda_monitoring")


def run_phase_features(nrows: int | None = None) -> pd.DataFrame:
    features = build_model_features(load_raw_frame(nrows=nrows))
    save_feature_artifacts(
        features,
        PROCESSED_DIR / "model_features.parquet",
        OUTPUTS_DIR / "feature_inventory.csv",
    )
    return features


def run_phase_anomaly(features: pd.DataFrame) -> None:
    output_dir = ensure_directory(OUTPUTS_DIR / "phase_5_anomaly_detection")
    for name, table in run_anomaly_detection(features).items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def run_phase_modeling(features: pd.DataFrame) -> None:
    output_dir = ensure_directory(OUTPUTS_DIR / "phase_6_modeling")
    for name, table in run_modeling_pipeline(features).items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    args = parse_args()

    if args.phase in {"audit", "all"}:
        run_phase_audit()

    features = None

    if args.phase in {"monitoring", "all"}:
        run_phase_monitoring(nrows=args.nrows)

    if args.phase in {"features", "all"}:
        features = run_phase_features(nrows=args.nrows)

    if args.phase in {"anomaly", "modeling"}:
        features = load_or_build_features(nrows=args.nrows)

    if args.phase == "all" and features is None:
        features = load_or_build_features(nrows=args.nrows)

    if args.phase in {"anomaly", "all"} and features is not None:
        run_phase_anomaly(features)

    if args.phase in {"modeling", "all"} and features is not None:
        run_phase_modeling(features)


if __name__ == "__main__":
    main()
