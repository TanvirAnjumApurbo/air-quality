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
# Cross-city generalisation check. Same pipeline, same grid, same seeds; only
# the record, one meteorological driver and the season definition differ.
CONFIG_B := config_beijing.yaml

.PHONY: help env check discover data audit features test baselines deep \
        classify green eval figures report all lint fmt clean clean-results \
        beijing beijing-post cross-city tune ablation donor-configs donors \
        replication donor-list everything

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
	@echo "  beijing     Cross-city: whole pipeline again on the Beijing record"
	@echo "  beijing-post  Beijing stages 08-11 only (after a resumed sweep)"
	@echo "  cross-city  Rank-transfer comparison (needs 'all' and 'beijing')"
	@echo "  everything  all + beijing + cross-city + report (final artefacts)"
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

# ---- cross-city generalisation check ---------------------------------------
# Reuses the Beijing frame already downloaded by `make data`; every stage after
# 12_prepare_beijing is the identical script run against the other config.
beijing:  ## Cross-city: run the whole pipeline again on the Beijing record
	$(PY) $(SCRIPTS)/12_prepare_beijing.py --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/03_data_audit.py      --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/04_build_features.py  --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/05_run_baselines.py   --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/06_train_sequence.py  --config $(CONFIG_B) --resume auto
	$(PY) $(SCRIPTS)/07_train_classifier.py --config $(CONFIG_B) --resume auto
	$(PY) $(SCRIPTS)/08_green_measure.py   --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/09_evaluate.py        --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/10_make_figures.py    --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/11_make_report.py     --config $(CONFIG_B)

# Needed on its own because a resumed sweep updates results_beijing.json but
# leaves the evaluation and report stale -- which is how Beijing ended up with
# no bootstrap CIs and no MCS while the primary city had both.
beijing-post:  ## Beijing stages 08-11 only, after a resumed sweep
	$(PY) $(SCRIPTS)/08_green_measure.py   --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/09_evaluate.py        --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/10_make_figures.py    --config $(CONFIG_B)
	$(PY) $(SCRIPTS)/11_make_report.py     --config $(CONFIG_B)

# Replication donors for the gap-injection experiment: further stations of the
# same UCI archive, so nothing is downloaded again. 06_train_sequence is
# deliberately absent -- the ablation's sequence specs are fixed in the config
# and only tier2 needs a donor-specific fit, which is what makes a donor cheap.
DONORS := $(shell sed -n 's/^[[:space:]]*-[[:space:]]*slug:[[:space:]]*//p' donors.yaml)

donor-configs:  ## Generate config/donors/*.yaml from donors.yaml
	$(PY) $(SCRIPTS)/14_make_donor_configs.py

donor-list:  ## Print the donor slugs parsed from donors.yaml
	@test -n "$(DONORS)" || (echo "no donor slugs parsed from donors.yaml" && false)
	@echo "donors: $(DONORS)"

donors: donor-configs donor-list  ## Prep + gap injection for every replication donor
	@for slug in $(DONORS); do \
	  echo "===== donor: $$slug ====="; \
	  $(PY) $(SCRIPTS)/02_fetch_data.py       --config config/donors/$$slug.yaml --skip openaq power; \
	  $(PY) $(SCRIPTS)/12_prepare_beijing.py  --config config/donors/$$slug.yaml --source data/interim/donors/$$slug/uci_beijing_hourly.parquet; \
	  $(PY) $(SCRIPTS)/03_data_audit.py       --config config/donors/$$slug.yaml; \
	  $(PY) $(SCRIPTS)/04_build_features.py   --config config/donors/$$slug.yaml; \
	  $(PY) $(SCRIPTS)/05_run_baselines.py    --config config/donors/$$slug.yaml; \
	  $(PY) $(SCRIPTS)/16_gap_injection.py    --config config/donors/$$slug.yaml --profile-config $(CONFIG) --progress plain; \
	  $(PY) $(SCRIPTS)/17_ablation_analysis.py --config config/donors/$$slug.yaml; \
	done
	$(PY) $(SCRIPTS)/18_donor_replication.py --config $(CONFIG_B)

replication:  ## Cross-donor comparison only (requires `ablation` and `donors`)
	$(PY) $(SCRIPTS)/18_donor_replication.py --config $(CONFIG_B)

cross-city:  ## Rank-transfer comparison (requires `all` and `beijing` first)
	$(PY) $(SCRIPTS)/13_cross_city.py --config $(CONFIG) --config-b $(CONFIG_B)

tune:  ## Choose the sequence training recipe on validation loss, before `deep`
	$(PY) $(SCRIPTS)/06_train_sequence.py --config $(CONFIG) --tune --progress plain

# The gap-injection experiment degrades the COMPARISON city (the near-complete
# record) using the PRIMARY city's gap-length distribution, so both configs are
# passed. Needs `beijing` first: hyperparameters are taken from its results.
ablation:  ## Gap-injection experiment + its figure (requires `beijing` first)
	$(PY) $(SCRIPTS)/16_gap_injection.py    --config $(CONFIG_B) --profile-config $(CONFIG) --progress plain
	$(PY) $(SCRIPTS)/17_ablation_analysis.py --config $(CONFIG_B)

# `report` is repeated last on purpose: 13_cross_city writes the comparison into
# results.json, and only a report generated afterwards carries section 7.
everything: all beijing cross-city ablation report  ## Every result in the paper

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
