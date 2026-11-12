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
