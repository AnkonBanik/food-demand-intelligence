# Food Demand Intelligence

**An end-to-end machine learning and business intelligence project for forecasting weekly food demand.**

Food Demand Intelligence explores how historical order patterns can be transformed into predictive insights to support meal planning, inventory management, and operational decision-making.

The project combines data engineering, exploratory data analysis, statistical forecasting, machine learning, reproducible experimentation, and an upcoming interactive analytics application.

**Project status:** Active development — first machine-learning benchmark completed.

## 1. Project Objectives

The project aims to:

- Analyze historical food demand across fulfillment centers and meal categories.
- Forecast weekly order quantities at the center-meal level.
- Compare statistical forecasting methods with machine-learning models.
- Evaluate forecasting performance across multiple prediction horizons.
- Investigate demand variability and model prediction errors.
- Develop reproducible and testable ML workflows.
- Present forecasting insights through an interactive business intelligence dashboard.

The longer-term objective is to demonstrate both machine-learning engineering and business analytics capabilities through a publicly reviewable portfolio project.

## 2. Machine Learning Results

The first controlled Random Forest experiment outperformed the last-observed-demand baseline on all three overall evaluation metrics.

| Evaluation metric | Statistical baseline | Random Forest | Improvement |
|---|---:|---:|---:|
| MAE | 110.3106 | **99.1353** | 10.13% |
| RMSE | 214.2156 | **178.1079** | 16.86% |
| RMSLE | 0.7665 | **0.7319** | 4.51% |

**Evaluation methodology**

- Training cutoff: Week 135
- Validation period: Weeks 136–145
- Forecast horizon: 10 weeks
- Validation observations: 32,821
- Random Forest training sample: 100,000 historical examples
- Random seed: 42

Lower metric values indicate better predictive performance.

These results represent a controlled historical experiment using a single fixed-origin validation window. They do not establish production forecasting performance.

## 3. Dataset

**Source:** Kaggle — Food Demand Forecasting dataset.

The training data contains historical weekly food order observations, including:

| Field | Description |
|---|---|
| `week` | Historical reporting week |
| `center_id` | Fulfillment center identifier |
| `meal_id` | Meal identifier |
| `checkout_price` | Checkout price |
| `base_price` | Base price |
| `emailer_for_promotion` | Email promotion indicator |
| `homepage_featured` | Homepage promotion indicator |
| `num_orders` | Recorded order quantity |

Additional datasets provide meal-category and fulfillment-center information.

The historical training dataset contains 456,548 observations.

Raw data files are not included in this repository. Users must obtain the dataset separately and place the required files in `data/raw/`.

## 4. Technical Architecture

The forecasting workflow follows these stages:

1. **Data ingestion:** Load historical orders and reference datasets.
2. **Data validation:** Check schema, duplicates, missing values and chronological gaps.
3. **Exploratory analysis:** Examine demand distributions and operational patterns.
4. **Feature engineering:** Generate calendar-aligned historical demand features.
5. **Forecast preparation:** Construct supervised examples for multiple forecast horizons.
6. **Model training:** Train statistical and machine-learning forecasting approaches.
7. **Model evaluation:** Calculate MAE, RMSE and RMSLE.
8. **Experiment tracking:** Save configurations, model results, dependency versions and dataset fingerprints.
9. **Business intelligence:** Present findings through an interactive dashboard (planned).

### Current technologies

| Area | Technologies |
|---|---|
| Programming | Python |
| Data processing | pandas, NumPy |
| Visualization | Matplotlib |
| Machine learning | scikit-learn |
| Testing | Python unittest |
| Reproducibility | Make, JSON experiment manifests |
| Version control | Git, GitHub |
| Application interface | Streamlit (planned) |

## 5. Forecasting Methodology

### Statistical baseline

The initial benchmark predicts demand using the latest recorded order quantity for each center-meal combination before the forecast origin.

For combinations without matching training history, the model applies a fallback strategy:

1. Historical demand from the same center-meal combination.
2. Meal-level historical median.
3. Global historical median.

All fallback statistics are calculated from training-period data.

### Random Forest

The first machine-learning model uses a direct multi-horizon regression approach.

Historical training examples contain:

- Center ID
- Meal ID
- Forecast horizon
- Demand at the historical forecast origin
- Calendar-aligned lagged demand
- Four-week historical demand average
- Historical observation count

Missing feature values are handled by a median imputer fitted on training data.

The controlled experiment used:

| Hyperparameter | Value |
|---|---:|
| Trees | 30 |
| Maximum depth | 12 |
| Minimum samples per leaf | 5 |
| Parallel jobs | 2 |
| Random seed | 42 |

### Leakage prevention

Forecasting features are derived from observations available at or before each forecast origin.

For the final validation window, historical demand is restricted to week 135 or earlier. Actual demand from weeks 136–145 is used for evaluation, not for building model inputs.

The current evaluation uses observed validation rows to identify which center-meal-week combinations to score. A deployment system would require an independently supplied forecasting schedule.

## 6. Project Setup

### Requirements

- Python 3.11 or 3.12
- Git
- Make (for the convenience commands)
- Original Kaggle dataset

### Clone the repository

```bash
git clone https://github.com/AnkonBanik/food-demand-intelligence.git
cd food-demand-intelligence
```

The repository's `main` branch may not contain the latest development work until its feature branches have been merged.

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### Prepare the dataset

Place these files inside `data/raw/`:

```text
data/raw/
├── train.csv
├── meal_info.csv
└── fulfilment_center_info.csv
```

The `data/` directory is excluded from version control to prevent accidental publication of raw datasets and generated artifacts.

## 7. Running the Project

The repository provides a Makefile for common development and evaluation tasks.

| Command | Purpose |
|---|---|
| `make help` | Display available commands |
| `make test` | Run all automated tests |
| `make baseline` | Evaluate the statistical baseline |
| `make random-forest` | Train and evaluate the controlled Random Forest experiment |
| `make resource-check` | Assess training dataset size and memory |
| `make manifest` | Generate the Random Forest experiment manifest |

For example:

```bash
make test
make baseline
make random-forest
make manifest
```

The commands use the active Python environment. To specify a different Python interpreter, provide `PYTHON=/path/to/python`.

The Random Forest run rebuilds historical training examples, samples 100,000 examples deterministically, trains the configured model, and generates validation metrics. Full training can require substantial computing resources.

## 8. Automated Testing

The project has **49 passing automated tests** at the latest verified checkpoint.

Tests cover:

- Dataset loading and integrity
- Duplicate and missing-record detection
- Calendar-aligned lag generation
- Historical feature engineering
- Cold-start forecasting
- Forecast-origin leakage prevention
- Chronological training separation
- Inference feature preparation
- Random Forest training and predictions
- Forecast evaluation metrics
- Experiment manifest generation

Run the complete suite:

```bash
make test
```

Automated tests provide evidence that expected software behavior is maintained. They do not guarantee forecasting accuracy or eliminate every possible data-quality risk.

## 9. Experiment Tracking and Documentation

The project maintains machine-readable experiment reports and human-readable analysis.

Important files include:

- [`docs/experiment_log.md`](docs/experiment_log.md) — experimental methodology, benchmark results, findings and limitations.
- [`docs/data_quality_report.md`](docs/data_quality_report.md) — historical data-quality findings.
- [`experiments/manifests/RF-001.json`](experiments/manifests/RF-001.json) — reproducibility metadata for the controlled Random Forest experiment.

Generated benchmark files are saved under `data/processed/benchmarks/` and are not committed to Git.

The experiment manifest records model parameters, evaluation results, dependency versions and dataset SHA-256 fingerprints. The Git revision captured during manifest generation does not independently prove the exact original training-time revision.

## 10. Project Roadmap

| Phase | Development status |
|---|---|
| Project foundation and repository setup | Completed |
| Data engineering and exploratory analysis | Completed |
| Statistical forecasting baseline | Completed |
| Random Forest first benchmark | Completed |
| Reproducibility documentation | In progress |
| Rolling-origin validation and model diagnostics | Planned |
| XGBoost, LightGBM and CatBoost benchmarking | Planned |
| Advanced forecasting model research | Planned |
| Production-oriented ML workflows | Planned |
| Interactive Streamlit dashboard | Planned |
| Final deployment and portfolio publication | Planned |

Future model comparisons will use consistent evaluation rules and will report limitations alongside results.

## 11. Current Limitations

- The dataset is historical and does not represent a live operational feed.
- Forecasting results currently come from one fixed-origin validation period.
- Random Forest has been evaluated using a controlled sample of historical training examples.
- Missing recorded observations are not automatically interpreted as zero demand.
- The present feature set focuses on historical demand rather than externally validated future promotion or price information.
- Model performance has not been tested in production.
- The interactive application and automated deployment workflow are not yet implemented.

## 12. Business Value

Accurate demand forecasting can potentially support:

- Better food preparation planning
- Reduced overproduction and shortages
- Improved inventory allocation
- Fulfillment-center demand monitoring
- More informed operational decisions

These are potential applications of forecasting. Specific financial or operational benefits have not yet been measured using this dataset.

## 13. Project Development Philosophy

Food Demand Intelligence is developed with an emphasis on:

- Modular, maintainable Python code
- Reproducible experiments
- Chronological evaluation
- Automated testing
- Transparent modeling assumptions
- Explainable business analytics
- Privacy-conscious GitHub publishing

The project is actively evolving from an experimental forecasting workflow toward a complete machine-learning and business intelligence application.
