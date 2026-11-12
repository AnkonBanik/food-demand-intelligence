
# Food Demand Intelligence
# Reproducible project execution commands

PYTHON ?= python3
export PYTHONPATH := src

.PHONY: help test baseline random-forest resource-check manifest

help:
	@echo "Food Demand Intelligence"
	@echo ""
	@echo "Available commands:"
	@echo "  make test            Run automated tests"
	@echo "  make baseline        Evaluate statistical baseline"
	@echo "  make random-forest   Run controlled Random Forest experiment"
	@echo "  make resource-check  Inspect ML training requirements"
	@echo "  make manifest        Generate Random Forest experiment manifest"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

baseline:
	$(PYTHON) -m food_demand.modeling.evaluate_baseline

random-forest:
	$(PYTHON) -m food_demand.modeling.run_rf_experiment

resource-check:
	$(PYTHON) -m food_demand.modeling.resource_check

manifest:
	$(PYTHON) -m food_demand.modeling.experiment_manifest
