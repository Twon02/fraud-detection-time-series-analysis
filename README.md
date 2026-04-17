# PaySim Fraud Risk Monitoring Project

This repository contains an end-to-end fraud risk monitoring workflow built on the 6.36 million-row PaySim transaction dataset. The project is organized as a hybrid notebook-plus-source-code portfolio: notebooks tell the analytical story, while the `src/` modules hold the reusable logic for loading data, auditing quality, engineering history-aware features, detecting anomalies, and training supervised models.

## Project Goals

- audit the raw dataset before modeling
- validate full-dataset quality at scale
- describe fraud behavior with business-relevant monitoring tables
- engineer historically safe transaction features
- test an unsupervised anomaly-detection layer
- train time-aware supervised fraud models
- close with a stakeholder-ready business summary

## Repository Structure

```text
data/
  raw/                         Raw PaySim CSV
  processed/                   Engineered parquet datasets
  outputs/
    phase_2_audit/             Full-dataset audit outputs
    phase_3_eda_monitoring/    EDA and monitoring outputs
    phase_5_anomaly_detection/ Anomaly scoring outputs
    phase_6_modeling/          Supervised modeling outputs
notebooks/
  01_data_audit.ipynb
  02_data_quality_audit.ipynb
  03_eda_and_monitoring.ipynb
  04_feature_engineering.ipynb
  05_anomaly_detection.ipynb
  06_modeling.ipynb
  07_business_summary.ipynb
src/
  load_data.py
  preprocessing.py
  feature_engineering.py
  monitoring.py
  anomaly.py
  modeling.py
  utils.py
run_project.py                 End-to-end phase runner
```

## Notebook Flow

### `01_data_audit.ipynb`

Performs an initial sample-based audit to confirm the dataset structure, schema, and raw column behavior before heavier full-scale work.

### `02_data_quality_audit.ipynb`

Runs the full-dataset quality audit using chunked processing. This phase checks missing values, duplicate rows, class balance, rule-based validation flags, and transaction-type fraud concentration.

### `03_eda_and_monitoring.ipynb`

Builds descriptive EDA outputs and monitoring tables. This notebook focuses on fraud versus non-fraud comparisons, temporal monitoring, type-level summaries, and evaluation of the built-in `isFlaggedFraud` rule.

### `04_feature_engineering.ipynb`

Creates historically safe sender and destination behavior features, including prior transaction counts, prior amount baselines, recency features, and sender rolling-window activity summaries.

### `05_anomaly_detection.ipynb`

Uses the engineered feature set to run an isolation-forest anomaly workflow and produces ranked alert tables, decile summaries, and precision-at-k style outputs.

### `06_modeling.ipynb`

Runs time-aware supervised fraud modeling with reusable pipelines from `src/modeling.py`. The notebook compares baseline model families on held-out data and saves the scored outputs for downstream review.

### `07_business_summary.ipynb`

Reads the saved project outputs and translates the technical work into a business-facing summary with findings, implications, and next-step recommendations.

## Source Modules

### `src/load_data.py`

Reusable data-loading helpers for samples, column-restricted reads, chunked reads, and schema inspection.

### `src/preprocessing.py`

Full-dataset audit helpers, validation rules, chunked quality checks, and CSV export logic for the phase 2 audit.

### `src/feature_engineering.py`

Time-aware feature creation functions. These features are designed to use only information available before each transaction, which helps reduce future leakage risk.

### `src/monitoring.py`

Reusable builders for overall summaries, amount-distribution summaries, fraud-type summaries, flag evaluation tables, account-level summaries, and hourly or daily monitoring tables.

### `src/anomaly.py`

Unsupervised anomaly scoring logic built around an isolation forest, along with ranked alert summaries and anomaly-decile analysis.

### `src/modeling.py`

Time-based train or validation or test splitting, training-sample construction for the imbalanced fraud problem, threshold selection, and baseline supervised model comparison.

### `src/utils.py`

Shared project paths, dtypes, directory creation, numerically safe division, and simple classification metric helpers.

## Key Findings From Current Outputs

The saved phase 2 and phase 3 outputs already show several strong business findings:

- the dataset contains `6,362,620` rows with no missing values and no duplicate rows in the saved audit outputs
- fraud is extremely rare at roughly `0.129%` of all transactions
- fraud is concentrated in `TRANSFER` and `CASH_OUT`
- the built-in `isFlaggedFraud` rule has perfect precision in the saved outputs but extremely low recall, which means it misses almost all fraud
- large-dataset handling matters throughout the workflow because even descriptive analysis can become expensive at this scale

## Running The Project

Install the dependencies listed in `requirements.txt`, then run phases as needed:

```bash
python3 run_project.py --phase audit
python3 run_project.py --phase monitoring
python3 run_project.py --phase features
python3 run_project.py --phase anomaly
python3 run_project.py --phase modeling
python3 run_project.py --phase all
```

For lighter development runs, you can limit the raw CSV load:

```bash
python3 run_project.py --phase monitoring --nrows 200000
```

## Design Decisions

- relative project-root-aware paths are used throughout the new code and notebooks
- the notebook sequence now matches the intended handoff structure: anomaly detection, modeling, then business summary
- reusable logic lives in `src/` so the notebooks can stay readable and presentation-ready
- phase outputs are saved to disk so later notebooks can build on prior work without rerunning heavy computations

## Current Assumptions

- `data/raw/paysim dataset.csv` is the main raw input
- `data/processed/model_features.parquet` is the preferred feature input for notebooks 05 and 06 when available
- the anomaly and modeling outputs are produced after phase 4 feature engineering

## Next Improvement Ideas

- add formal tests for the reusable source modules
- add calibration and cost-sensitive threshold analysis for the supervised models
- add a small dashboard layer on top of the saved CSV outputs
- add notebook execution snapshots once the local Python environment includes the required dependencies
