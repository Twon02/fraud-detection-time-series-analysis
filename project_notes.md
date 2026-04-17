# Project Handoff Notes

## Project Direction

We are continuing the fraud risk monitoring project as a hybrid notebook-and-codebase workflow.

## Intended Notebook Structure

1. `01_data_audit.ipynb`
2. `02_data_quality_audit.ipynb`
3. `03_eda_and_monitoring.ipynb`
4. `04_feature_engineering.ipynb`
5. `05_anomaly_detection.ipynb`
6. `06_modeling.ipynb`
7. `07_business_summary.ipynb`

## Confirmed Progress

- Notebook `01` handles the sample-based audit
- Notebook `02` handles the full-scale data-quality audit on the full PaySim dataset
- Notebook `03` is the EDA and monitoring phase
- Feature engineering belongs in notebook `04`

## Working Rules

- keep the workflow modular and scalable
- prefer reusable code in `src/` over long notebook-only logic
- keep notebook paths relative and project-root aware
- write notebooks as a professional end-to-end story
- place detailed markdown explanations before the main code steps
- preserve privacy by avoiding unnecessary absolute local paths in notebook outputs
- keep the portfolio presentation strong and explicit about large-dataset handling

## Immediate Scope Used For This Refresh

This repository refresh completed the missing downstream phases by:

- keeping notebook `03` focused on EDA and monitoring
- preserving notebook `04` as the feature-engineering phase
- creating notebook `05` for anomaly detection
- creating notebook `06` for supervised modeling
- creating notebook `07` for the business summary
- filling the empty source modules needed to support those notebooks
- adding a project-level `README.md` and a reproducible `run_project.py` runner
