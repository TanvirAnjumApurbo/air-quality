# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` covers setup, the phase-by-phase pipeline table, training flags and the
five methodological rules. Read it first. This file covers what the README does not:
the architecture you have to read several files to see, and the invariants that will
silently corrupt a result if you break them.

## Commands

**Always invoke Python through the venv.** Bare `python` on this box has no
dependencies installed and fails at `import pandas`.

```powershell
.venv/Scripts/python.exe scripts/11_make_report.py --config config.yaml
```

```powershell
.\make.ps1 help            # PowerShell shim; `make help` if GNU make is on PATH
.\make.ps1 all             # full Dhaka pipeline
.\make.ps1 beijing         # same pipeline again on the cross-city record
.\make.ps1 beijing-post    # Beijing 08-11 only, after a resumed sweep
.\make.ps1 cross-city      # rank-transfer comparison (needs `all` and `beijing`)
.\make.ps1 ablation        # gap-injection experiment (needs `beijing`)
.\make.ps1 everything      # all + beijing + cross-city + ablation + report
```

Lint and format (ruff config lives in `pyproject.toml`; docstrings and type
annotations on public functions are enforced by `D` and `ANN` rules):

```powershell
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check src scripts tests
.venv/Scripts/python.exe -m ruff format src scripts tests     # writes
```

Tests — all 37 live in `tests/test_leakage.py`:

```powershell
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m pytest tests -m leakage           # leakage guards only
.venv/Scripts/python.exe -m pytest tests -k test_no_valid_row_spans_a_gap -v
```

Markers are declared in `pyproject.toml` under `--strict-markers`: `leakage`,
`slow`, `network`. Four tests are negative controls — they inject the forbidden
mistake and assert the guard fires. If you relax a guard, those fail, which is the
point.

## Architecture

### The pipeline is city-agnostic; the config selects the city

`config.yaml` (Dhaka, primary) and `config_beijing.yaml` (cross-city check) are
interchangeable inputs to the *same* scripts. Every script takes `--config` and
resolves all paths, seeds, horizons and output locations through it. Only
`12_prepare_beijing.py` is city-specific — it reshapes the UCI record into the
schema the shared feature builder expects, so stages 03 through 11 run unchanged
against either config.

`config_beijing.yaml` was derived by copying `config.yaml`. **Any key you add to one
is silently inherited by the other with the wrong value.** This has already caused
four defects: both configs writing reports to the same path (the comparison city
overwrote the primary city's `RESULTS.md` and `DATA_AUDIT.md`), Beijing carrying
Dhaka's coordinates, Dhaka's OpenAQ location IDs, and Dhaka's study title. Dead keys
on the Beijing path are now `null` so they fail loudly rather than resolving to
Dhaka. When you touch either config, diff both.

### `results/results.json` is the only source of reported numbers

`src/results.py` defines the contract. Every modelling stage calls `load_results` →
mutate → `save_results`, which merges rather than overwrites, so a partial re-run
updates only what it recomputed. `upsert_run` keys records on
`(model, variant, horizon_h, seed)`.

`11_make_report.py` **reads** `results.json` and writes nothing that is not already
in it — if a phase has not run, its section is omitted rather than invented. Never
add a number to `RESULTS.md` or `abstract_facts.json` that does not come from
`results.json`. The Beijing run writes to `results/results_beijing.json` and its
reports to `reports/beijing/`.

There is exactly one other source, and it is deliberate: §8 of `RESULTS.md` reads
`results/ablation_gap_injection.json`, which `17_ablation_analysis.py` generates
and nothing writes by hand. It cannot live in `results.json` — the experiment runs
under the *donor* city's config, so it would land in `results_beijing.json`, while
the injected gap profile and the claim both belong to the primary city. The rule
that matters (no transcribed numbers) still holds. Do not add a third source.

### Data flow

```
02_fetch_data      -> data/interim/*.parquet        (target, meteorology, QC ledger)
03_data_audit      -> reports/DATA_AUDIT.md          HARD GATE, prints a verdict
04_build_features  -> data/processed/features.parquet + features_meta.json + scaler.json
05..07 (models)    -> results.json runs[] + data/interim/predictions/
08..09 (green/eval)-> results.json green/significance/stratified
10, 11             -> results/figures, results/tables, reports/
13_cross_city      -> results.json cross_city{}      (must run before the final report)
```

`features.parquet` is the handoff. It carries every predictor, a `split` column, and
one `valid_h{h}` boolean mask per horizon. `src/models/data.py` is the only reader
models go through: `get_split_arrays` materialises `SplitArrays` for tabular models,
`build_sequence_index` produces the window index for recurrent ones.

### The sequence tier receives channels, not the tabular predictor set

`src/models/data.py::build_sequence_index` calls `sequence_channel_columns`, not
`feature_columns`. Tabular models get all 102 engineered predictors because each row
must stand alone; a recurrent model reads the window itself, so those columns are
largely a restatement of what it already sees — `pm25_lag_24` at *t* IS the `pm25`
channel at *t−24*, inside a 48-hour window. Supplying both made the input 48×102 of
heavily collinear columns and handicapped the sequence tier in a comparison against
the tier those same columns help. The default is now 18 contemporaneous channels.

`features.sequence_channels.mode: engineered` restores the old behaviour and is a
**reported ablation arm, not dead code** — which representation suits which model
class is a result. `derived_history_columns` generates the excluded names from the
same config keys that build them, so the two cannot drift; a test asserts the two
sets partition the tabular predictors exactly.

Consequence to remember: the parameter budget bites differently at 18 channels.
`lstm_h128_l1` was 118,913 parameters and excluded; it is now 75,905 and runs.

### The ablation is a controlled experiment, and its controls are load-bearing

`scripts/16_gap_injection.py` is the paper's central claim. Three controls are not
optional detail — remove any one and it stops measuring fragmentation:

1. **Both arms remove an identical hour count.** Only the arrangement differs, so
   data volume is held constant. Without the contiguous arm, every result is
   equally explained by "less data".
2. **The test period is never degraded** (`protect_from=val_end`). Every cell is
   scored on the same rows, so RMSE is comparable across the grid.
3. **The optimizer-step budget is equalised, not the epoch budget.** Fragmentation
   shrinks the training set: at a fixed 60 epochs a fragmented cell takes 16
   steps/epoch against a contiguous cell's 65. Epochs, patience and `min_epochs`
   are rescaled per cell toward `target_optimizer_steps`. This was not theoretical —
   before the fix DLinear hit the 60-epoch cap still underfitting at val 2.0, and
   with the budget equalised it reaches 1.35 and keeps falling.

`src/features/pipeline.py::build_feature_matrix` exists so the ablation rebuilds each
degraded matrix through the *same* code as `04_build_features.py`. The boundary purge
and per-horizon validity masks are where the leakage rules live; a second copy would
be a second definition of a usable row. The extraction was verified to reproduce the
existing `features.parquet`, `scaler.json` and `features_meta.json` byte for byte.

### Tier vocabulary

Every run record carries a `tier`, and selection logic branches on it:

- **tier1** — persistence, seasonal-naive, climatology, SARIMAX. No hyperparameters.
- **tier2** — Ridge, RandomForest, XGBoost, LightGBM via `RandomizedSearchCV` +
  `TimeSeriesSplit` over train+val.
- **tier3** — GRU/LSTM under a 100k-parameter budget, 5 seeds each.

### Leakage rules are structural, not conventional

The five rules in the README are enforced at these points:

- `load_config` raises if `split.shuffle` or `task.allow_future_meteorology` is true.
  These are config-level invariants, not runtime options.
- `src/features/build_features.py` builds per-horizon validity from
  `pos_in_run >= max_lag`, `(run_len - pos_in_run) > h`, and a genuinely observed
  (never imputed) target. Rejection counts are reported, not silently dropped.
- `src/eval/split.py::TrainOnlyScaler` raises unless `split_name == "train"`, and
  raises on transform-before-fit.
- Oracle-meteorology columns exist but are prefixed `oracle_` and excluded from
  `feature_columns` unless explicitly requested; they are an upper bound, never a
  headline.

## Invariants that will corrupt results if broken

**Never write `config.yaml` with `yaml.safe_dump`.** It discards all 212 comment
lines, which hold the citations, source quotations, and the record of two verified
API discrepancies. `04_build_features.py::write_back_boundaries` does a targeted
regex line edit and verifies the round-trip. That is deliberate, not laziness.

**Never select a model on test RMSE.** `_best_per_horizon` in `11_make_report.py`
ranks tier3 by mean validation loss, tier2 by CV score, and tier1 by test RMSE only
because those have no hyperparameters and therefore no selection to bias. Ranking
candidates by test error and then reporting that error is circular and biases the
headline downward by the spread of the pool.

**Citations gate the numbers that depend on them.** `config.yaml` holds a
`citations:` block where every externally-sourced constant carries
`_status: VERIFIED | PENDING_VERIFICATION` (currently 10 verified, 0 pending).
`Config.require_verified()` raises on an unverified value. A pending citation must
leave the cell blank and flagged — never fill in a plausible figure. This applies to
AQI breakpoints, the Bangladesh PM2.5 standard, grid carbon intensity, and season
boundaries.

**Predictions are bounded to `[0, qc.pm25.sanity_cap_ugm3]` identically for every
model** (`src/models/data.py::invert`), with clip counts logged. The target is
modelled on a `log1p` scale; unbounded `expm1` inversion produced impossible
concentrations that dominated squared error.

**If a result improves sharply, suspect leakage before celebrating.** SARIMAX once
showed +0.44 skill because `get_prediction(dynamic=False)` returns 1-step-ahead
forecasts that were being stamped onto `t+h`. The honest value is −0.103. The exact
state-space projection now lives in
`src/models/baselines.py::h_step_ahead_from_filtered_state`.

## Operational constraints

- **Disk.** Every writing script calls `check_disk_space` and refuses below
  `runtime.min_free_disk_gb` (5 GB). A previous run filled the system drive. `data/`
  is gitignored; do not cache multi-GB intermediates.
- **Thermals.** This box has thrown CPU-overheat faults. Keep any individual
  training run under `runtime.max_minutes_per_run` (10 min); the trainer warns and
  stops rather than pushing past it.
- **Windows file locks.** Antivirus intermittently holds checkpoint handles and
  killed one sweep at run 191/675. `sequence.py::_atomic_save` retries with backoff.
  Always use `--resume auto`.
- **CUDA.** Blackwell (sm_120) needs the cu128 build. `torch.cuda.is_available()`
  returning `True` is not evidence training will run — `resolve_device` and
  `00_check_env.py` launch a real kernel.
- `.gitignore` anchors `/data/`, not `data/`. Unanchored, it also matched
  `src/data/` and dropped the package from a commit.

## Provenance

`results.json` stamps `git_commit` at the time the *results* were computed, not when
the report was rendered — so `RESULTS.md` can legitimately show an older, `-dirty`
hash than `HEAD`. That is intended. Current values are `26d8a27…-dirty` (Dhaka,
stamped when the sweep ran) and `548e95d…-dirty` (Beijing, restamped when
`beijing-post` re-ran stages 08–11 against the same records); making them clean
requires re-running the sweeps, which was considered and declined.

Note the asymmetry: `save_results` restamps `git_commit` on **every** write, so any
downstream stage that touches a city's `results.json` moves that city's hash forward
without a single model having been retrained. The hash records when the file was last
written, not when the numbers in it were produced.
