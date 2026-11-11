# Food Demand Intelligence — Data Quality Report

## Dataset overview

| Metric | Value |
|---|---:|
| Training records | 456,548 |
| First week | 1 |
| Last week | 145 |
| Fulfillment centers | 77 |
| Meals | 51 |
| Center-meal combinations | 3,597 |
| Duplicate center-meal-week rows | 0 |
| Recorded zero-order observations | 0 |
| Gap events | 21,504 |
| Missing weeks between observations | 40,548 |
| Shortest series | 1 |
| Longest series | 145 |
| Median series length | 140.0 |

## Missing values

| Column | Missing values |
|---|---:|
| `id` | 0 |
| `week` | 0 |
| `center_id` | 0 |
| `meal_id` | 0 |
| `checkout_price` | 0 |
| `base_price` | 0 |
| `emailer_for_promotion` | 0 |
| `homepage_featured` | 0 |
| `num_orders` | 0 |
| `category` | 0 |
| `cuisine` | 0 |
| `city_code` | 0 |
| `region_code` | 0 |
| `center_type` | 0 |
| `op_area` | 0 |

## Forecasting considerations

- Demand is recorded weekly.
- Some center-meal series have gaps between observations.
- Missing observations must not automatically be treated as zero demand.
- Historical lag features must use exact calendar-week alignment.
- Model evaluation must respect chronological order.

## Reproduce the validation

From the project root:

```bash
PYTHONPATH=src python -m food_demand.data.validation
```

Run the automated tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
```

Use a Python environment with the project dependencies installed.

## Limitations

- This report describes the available historical training dataset.
- The checks do not establish whether missing weeks represent zero demand.
- Passing unit tests does not guarantee complete dataset quality.
