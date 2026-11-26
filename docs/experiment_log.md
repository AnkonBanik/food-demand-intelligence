# Food Demand Intelligence — Experiment Log

## Project objective

Forecast weekly food demand at the fulfillment-center and meal level using historical order data.

The project compares statistical baselines with machine-learning approaches while maintaining chronological data separation.

## Experiment 001 — Last Observed Demand Baseline

**Method:** Fixed-origin statistical forecasting

**Forecast origin:** Week 135  
**Validation period:** Weeks 136–145  
**Validation observations:** 32,821

### Forecasting strategy

- Use the most recent recorded demand for each center-meal combination from the training period.
- Use the historical meal-level median for combinations without a matching training history.
- Use the global training median when meal-level history is unavailable.
- Keep the forecast fixed across the ten-week validation horizon.
- Do not use validation-period actual demand to generate predictions.

### Results

| Metric | Value |
|---|---:|
| MAE | 110.3106 |
| RMSE | 214.2156 |
| RMSLE | 0.7665 |

Eight validation observations required the meal-level fallback.

## Experiment 002 — Random Forest (Controlled)

**Method:** Direct multi-horizon Random Forest regression

**Code reference:** `576edd4`  
**Training cutoff:** Week 135  
**Validation period:** Weeks 136–145  
**Validation observations:** 32,821

### Experiment configuration

| Parameter | Value |
|---|---:|
| Training examples sampled | 100,000 |
| Random seed | 42 |
| Trees | 30 |
| Maximum tree depth | 12 |
| Minimum samples per leaf | 5 |
| Parallel jobs | 2 |
| Forecast horizon | 10 weeks |
| Feature count | 8 |

The sample was drawn deterministically from the historical training examples.

### Features

- Center identifier
- Meal identifier
- Forecast horizon
- Demand at the forecast origin
- Historical demand at additional calendar-aligned lags
- Mean demand across the latest four calendar weeks
- Number of available observations across those four weeks

Missing numerical features are imputed using training-fitted median statistics.

### Results

| Metric | Baseline | Random Forest | Relative improvement |
|---|---:|---:|---:|
| MAE | 110.3106 | 99.1353 | 10.13% |
| RMSE | 214.2156 | 178.1079 | 16.86% |
| RMSLE | 0.7665 | 0.7319 | 4.51% |

### Horizon-level observations

- Horizon 4 (week 139) had the lowest reported Random Forest MAE: 90.50.
- Horizon 6 (week 141) had the highest MAE: 111.76.
- Horizon 3 (week 138) had the highest RMSLE: 0.8189.
- Horizon 10 (week 145) had an MAE of approximately 106.75 over 3,268 observations.

### Interpretation

Random Forest outperformed the statistical baseline across all three overall evaluation metrics.

However, forecast accuracy varied across horizons. Additional experiments are needed to assess generalization across multiple forecast origins.

## Resource assessment

The full historical feature-generation process produced:

| Metric | Value |
|---|---:|
| Supervised training examples | 3,798,738 |
| Feature columns | 8 |
| DataFrame memory estimate | 260.84 MB |

The controlled experiment used 100,000 sampled examples to limit initial computational requirements.

## Reproducibility

Run the automated tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
```

Run the baseline:

```bash
PYTHONPATH=src python -m food_demand.modeling.evaluate_baseline
```

Run the controlled Random Forest experiment:

```bash
PYTHONPATH=src python -m food_demand.modeling.run_rf_experiment
```

Run the resource assessment:

```bash
PYTHONPATH=src python -m food_demand.modeling.resource_check
```

These commands require an environment with the project dependencies installed and the original datasets placed in the expected `data/raw/` location.

## Limitations

1. Results are based on one fixed-origin validation window rather than multiple rolling-origin evaluations.
2. The Random Forest experiment uses a sample of historical training examples.
3. Validation records identify which center-meal combinations are evaluated; a production system requires an independently supplied forecast schedule.
4. Missing historical observations are not assumed to represent zero demand.
5. The current benchmark does not establish real-world deployment performance.
6. Additional tests, model comparisons, and operational validation are required.

## Planned experiments

- Evaluate performance against the baseline separately for every forecast horizon.
- Investigate high-error weeks and center-meal combinations.
- Compare different training sample sizes.
- Benchmark XGBoost, LightGBM, and CatBoost.
- Introduce rolling-origin validation.
- Develop a reproducible model-comparison report and business-facing dashboard.

## Proposed Experiment 003 — Random Forest Hyperparameter Research (RF-002)

**Status:** Executed — results recorded in the section below

**Objective:** Investigate whether alternative Random Forest configurations improve forecasting accuracy while maintaining reproducibility and computational efficiency.

### Evaluation design

| Setting | Value |
|---|---:|
| Training cutoff | Week 115 |
| Validation period | Weeks 116–125 |
| Forecast horizon | 10 weeks |
| Training sample target | 100,000 |
| Random seed | 42 |
| Feature count | 8 |
| Parallel jobs | 2 |

Training examples will be generated exclusively from historical observations up to week 115. Validation features will use information available at the fixed forecast origin.

### Candidate configurations

| Configuration | Trees | Maximum depth | Minimum samples per leaf |
|---|---:|---:|---:|
| RF-002-A | 30 | 8 | 5 |
| RF-002-B | 30 | 12 | 3 |
| RF-002-C | 50 | 12 | 5 |
| RF-002-D | 50 | 16 | 3 |

All other experimental conditions will remain constant.


python3 - << 'EOF'
from pathlib import Path
p = Path("docs/experiment_log.md")
s = p.read_text(encoding="utf-8")
line = "**Primary selection metric:** RMSLE (MAE breaks ties). Declared before execution.\n\n"
assert s.count("### Evaluation metrics") == 1
p.write_text(s.replace("### Evaluation metrics", line + "### Evaluation metrics"), encoding="utf-8")
EOF
git diff docs/experiment_log.md | grep "^+" | grep -i "primary selection"


### Evaluation metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Root Mean Squared Logarithmic Error (RMSLE)
- Individual forecast-horizon performance
- Training time and resource requirements

### Experimental controls

1. Use the same historical dataset and feature definitions.
2. Apply deterministic sampling with random seed 42.
3. Keep the training and validation periods identical across candidates.
4. Do not use validation actual demand when constructing inference features.
5. Save each candidate's configuration and results separately.
6. Preserve RF-001 reports, manifests and model artifacts.

### Final evaluation policy

Weeks 136–145 remain reserved for final model comparison. They must not be used to select RF-002 hyperparameters.

After tuning, the selected configuration will be evaluated against the existing RF-001 benchmark using the original evaluation period.

### Expected deliverables

- Reusable experiment execution code
- Automated tests for experiment configuration and output separation
- Individual experiment reports
- RF-002 experiment manifest
- Updated experiment log with results and limitations

No candidate is considered production-approved based solely on this tuning experiment.

## Experiment 003 Results — RF-002 Hyperparameter Research

**Status:** Completed

### Tuning results (training through week 115, validation weeks 116–125)

Validation observations: 32,834. Training sample: 100,000 examples. Random seed: 42.
Candidates ranked by RMSLE, the metric declared before execution.

| Rank | Candidate | Trees | Max depth | Min leaf | MAE | RMSE | RMSLE | Training time (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | RF-002-C | 50 | 12 | 5 | 124.6470 | 277.3994 | 0.7483 | 3.48 |
| 2 | RF-002-B | 30 | 12 | 3 | 126.0486 | 279.7240 | 0.7527 | 2.11 |
| 3 | RF-002-D | 50 | 16 | 3 | 126.2268 | 277.6537 | 0.7540 | 4.40 |
| 4 | RF-002-A | 30 | 8 | 5 | 125.5089 | 282.1643 | 0.7614 | 1.48 |

RF-002-C had the lowest MAE, RMSE and RMSLE. The MAE difference between the best and worst candidate was about 1.3%.

### Final evaluation (training through week 135, evaluation weeks 136–145)

RF-002-C was selected from the tuning ranking and evaluated once on the reserved weeks.

| Model | MAE | RMSE | RMSLE |
|---|---:|---:|---:|
| Last-observed baseline | 110.3106 | 214.2156 | 0.7665 |
| RF-001 | 99.1353 | 178.1079 | 0.7319 |
| RF-002-C | 98.8656 | 177.6465 | 0.7312 |

Relative to RF-001, RF-002-C reduced MAE by 0.27%, RMSE by 0.26% and RMSLE by 0.10%.
Relative to the baseline, it reduced MAE by 10.38%, RMSE by 17.07% and RMSLE by 4.61%.

### Interpretation

Changing tree count, depth and leaf size within the tested range produced only small differences.
The improvement over RF-001 is small, was measured at a single forecast origin, and has not been tested for statistical significance. It should not be treated as a reliable improvement.

The current feature set appears to limit accuracy more than Random Forest hyperparameters do. Future experiments should investigate additional information, particularly known future price and promotion values.

### Limitations

1. Tuning metrics (weeks 116–125) and final metrics (weeks 136–145) come from different windows and are not directly comparable.
2. The final comparison uses one forecast origin.
3. Weeks 136–145 were not used to select RF-002 hyperparameters, but they had previously been used to evaluate RF-001. Further model comparisons should use rolling-origin validation.
4. The training sample was limited to 100,000 examples.

### Reproducibility

- Reports: `data/processed/benchmarks/rf002/` and `data/processed/benchmarks/rf002_final.json` (not committed to Git).
- Manifests: `experiments/manifests/RF-002-A.json` to `RF-002-D.json` and `RF-002-C-FINAL.json`.
- Commands: `make rf-tuning` and `make rf-final`.
