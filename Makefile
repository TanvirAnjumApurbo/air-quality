# =============================================================================
# Short-Horizon PM2.5 Forecasting for Dhaka -- reproducible pipeline
#
#   make all        reproduce every result from scratch
#   make help       list targets
#
# Windows: GNU make ships with MSYS2 at C:\msys64\usr\bin\make.exe, or use the
# PowerShell shim `.\make.ps1 <target>` which mirrors these targets exactly.
# =============================================================================

ifeq ($(OS),Windows_NT)
	PY := .venv/Scripts/python.exe
	VENV_PY := $(PY)
else
	PY := .venv/bin/python
	VENV_PY := $(PY)
endif

SCRIPTS := scripts
.DEFAULT_GOAL := help

# Seeds/horizons/paths are NOT defined here -- they live in config.yaml.
CONFIG := config.yaml

.PHONY: help env check discover data audit features test baselines deep \
        classify green eval figures report all lint fmt clean clean-results

help:  ## List available targets
	@echo "Targets:"
	@echo "  env         Create .venv and install pinned requirements"
	@echo "  check       Phase 0: environment / CUDA / disk check"
	@echo "  discover    Phase 1: list Dhaka OpenAQ monitors (STOPS for review)"
	@echo "  data        Phase 1: fetch OpenAQ + NASA POWER + UCI Beijing"
	@echo "  audit       Phase 1: build reports/DATA_AUDIT.md  (HARD GATE)"
	@echo "  features    Phase 2: build features + chronological splits"
	@echo "  test        Phase 2: leakage + unit tests (must pass before modelling)"
	@echo "  baselines   Phase 3: Tier 1 baselines + Tier 2 classical ML"
	@echo "  deep        Phase 4: Tier 3 sequence models, all seeds"
	@echo "  classify    Phase 5: AQI-category classifier"
	@echo "  green       Phase 5: complexity, latency, energy, CO2e"
	@echo "  eval        Phase 5: stratified eval, skill scores, Diebold-Mariano"
	@echo "  figures     Phase 6: all figures (png + pdf, 300 dpi)"
	@echo "  report      Phase 6: RESULTS.md + abstract_facts.json"
	@echo "  all         Everything above, in order"
	@echo "  lint        ruff check + format check"
	@echo "  clean       Remove caches and checkpoints (keeps raw data)"

env:  ## Create virtualenv and install dependencies
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip setuptools wheel
	$(VENV_PY) -m pip install torch --index-url https://download.pytorch.org/whl/cu128
	$(VENV_PY) -m pip install -r requirements.txt

check:  ## Phase 0 environment check
	$(PY) $(SCRIPTS)/00_check_env.py

discover:  ## Phase 1 OpenAQ location discovery (prints table, downloads nothing)
	$(PY) $(SCRIPTS)/01_discover_openaq.py --config $(CONFIG)

data:  ## Phase 1 fetch all three data sources
	$(PY) $(SCRIPTS)/02_fetch_data.py --config $(CONFIG)

audit:  ## Phase 1 data audit (hard gate before modelling)
	$(PY) $(SCRIPTS)/03_data_audit.py --config $(CONFIG)

features:  ## Phase 2 features + chronological splits
	$(PY) $(SCRIPTS)/04_build_features.py --config $(CONFIG)

test:  ## Phase 2 leakage and unit tests
	$(PY) -m pytest tests -v

baselines:  ## Phase 3 Tier 1 + Tier 2
	$(PY) $(SCRIPTS)/05_run_baselines.py --config $(CONFIG)

deep:  ## Phase 4 Tier 3 sequence models (resumable)
	$(PY) $(SCRIPTS)/06_train_sequence.py --config $(CONFIG) --resume auto

classify:  ## Phase 5 AQI-category classifier
	$(PY) $(SCRIPTS)/07_train_classifier.py --config $(CONFIG) --resume auto

green:  ## Phase 5 complexity / latency / energy / CO2e
	$(PY) $(SCRIPTS)/08_green_measure.py --config $(CONFIG)

eval:  ## Phase 5 stratified evaluation + significance tests
	$(PY) $(SCRIPTS)/09_evaluate.py --config $(CONFIG)

figures:  ## Phase 6 figures
	$(PY) $(SCRIPTS)/10_make_figures.py --config $(CONFIG)

report:  ## Phase 6 RESULTS.md + abstract_facts.json
	$(PY) $(SCRIPTS)/11_make_report.py --config $(CONFIG)

all: check data audit features test baselines deep classify green eval figures report  ## Full pipeline

lint:  ## ruff
	$(PY) -m ruff check src scripts tests
	$(PY) -m ruff format --check src scripts tests

fmt:  ## ruff format (writes)
	$(PY) -m ruff format src scripts tests
	$(PY) -m ruff check --fix src scripts tests

clean:  ## Remove caches, checkpoints and __pycache__ (raw data untouched)
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PY) -c "import shutil; shutil.rmtree('data/interim/cache', ignore_errors=True)"
	$(PY) -c "import shutil; shutil.rmtree('data/interim/checkpoints', ignore_errors=True)"
	$(PY) -c "import shutil; shutil.rmtree('.ruff_cache', ignore_errors=True); shutil.rmtree('.pytest_cache', ignore_errors=True)"

clean-results:  ## Remove generated results (forces full recompute)
	$(PY) -c "import shutil; shutil.rmtree('results/tables', ignore_errors=True); shutil.rmtree('results/figures', ignore_errors=True)"
