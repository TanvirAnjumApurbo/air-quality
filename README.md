# Short-Horizon PM2.5 Forecasting for Dhaka

**A Lightweight Sequence Model on Fused OpenAQ and NASA POWER Data**

Can a compact recurrent model (GRU/LSTM, ≤100k parameters) forecasting Dhaka
PM2.5 at 1–24 hour horizons beat honest classical baselines when air-quality
history is fused with meteorological drivers — and what is the
accuracy-per-parameter / accuracy-per-gram-of-CO₂ tradeoff?

**Contributions**

1. A reproducible, Bangladesh-specific multi-horizon PM2.5 forecaster benchmarked
   against the baselines air-quality papers usually omit — persistence,
   seasonal-naïve, and climatology — with a **skill score vs persistence** as a
   headline column.
2. A green-AI efficiency analysis: accuracy vs parameter count vs *measured*
   training energy vs estimated CO₂e, recomputed under Bangladesh's grid carbon
   intensity.

A secondary AQI-category classification task (next-day health advisory) is built
from the same pipeline.

---

## 1. Setup

Requires Python 3.11 or 3.12. This repo was built and run on Python 3.12.10.

### Windows (this box: RTX 5060 Ti 8 GB, Blackwell / sm_120)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

# CUDA 12.8 build. Older wheels install fine and then fail at the first kernel
# launch with "no kernel image is available for execution on the device".
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If your system drive is tight, redirect the pip cache off it first:

```powershell
$env:PIP_CACHE_DIR = "F:\BUP\.pipcache"
```

### Verify the environment

```powershell
.\.venv\Scripts\python.exe scripts\00_check_env.py
```

This prints Python/torch versions, `torch.cuda.is_available()`, the device name
and compute capability, host RAM, and free disk on the project drive — and then
**actually launches a CUDA kernel**, because on Blackwell `is_available()`
returning `True` is not evidence that training will run.

### API keys

Only OpenAQ *discovery* needs a key (bulk download from the S3 archive does not).

```powershell
Copy-Item .env.example .env      # then paste your key into .env
```

`.env` is gitignored. NASA POWER and UCI need no credentials.

---

## 2. Running the pipeline

`make` (GNU make, e.g. from MSYS2) and the PowerShell shim `make.ps1` expose the
same targets:

```powershell
.\make.ps1 help          # or:  make help
```

| Phase | Command | Deliverable |
|---|---|---|
| 0 | `.\make.ps1 check` | environment / CUDA / disk report |
| 1 | `.\make.ps1 discover` | OpenAQ Dhaka monitor table — **downloads nothing** |
| 1 | `.\make.ps1 data` | OpenAQ + NASA POWER + UCI Beijing into `data/raw` |
| 1 | `.\make.ps1 audit` | `reports/DATA_AUDIT.md` — **hard gate before modelling** |
| 2 | `.\make.ps1 features` | features + chronological splits |
| 2 | `.\make.ps1 test` | leakage tests (must pass) |
| 3 | `.\make.ps1 baselines` | Tier 1 baselines + Tier 2 classical ML |
| 4 | `.\make.ps1 deep` | Tier 3 GRU/LSTM, all seeds |
| 5 | `.\make.ps1 classify` / `green` / `eval` | classifier, energy, stratified eval |
| 6 | `.\make.ps1 figures` / `report` | figures, `RESULTS.md`, `abstract_facts.json` |

`make all` reproduces everything from scratch with fixed seeds. The git commit
hash is written into `results/results.json`.

---

## 3. Training commands (run these yourself)

Every training entrypoint prints live progress, checkpoints each epoch, and
resumes automatically from the last checkpoint if interrupted.

### Sequence models (Tier 3)

```powershell
# full sweep, all architectures x horizons x seeds, resumes if interrupted
.\.venv\Scripts\python.exe scripts\06_train_sequence.py --config config.yaml --resume auto

# a single configuration
.\.venv\Scripts\python.exe scripts\06_train_sequence.py --config config.yaml `
    --arch gru --hidden 64 --layers 1 --window 48 --horizon 24 --seed 42

# restart from scratch, ignoring existing checkpoints
.\.venv\Scripts\python.exe scripts\06_train_sequence.py --config config.yaml --resume never

# resume one interrupted run explicitly
.\.venv\Scripts\python.exe scripts\06_train_sequence.py --config config.yaml --resume always `
    --arch gru --hidden 64 --layers 1 --window 48 --horizon 24 --seed 42
```

### AQI classifier (secondary task)

```powershell
.\.venv\Scripts\python.exe scripts\07_train_classifier.py --config config.yaml --resume auto
```

### Useful flags

| Flag | Effect |
|---|---|
| `--resume {auto,never,always}` | checkpoint policy; `auto` resumes if `last.ckpt` exists |
| `--device {auto,cuda,cpu}` | override `runtime.device` |
| `--no-cache` | disable the in-RAM tensor cache (lower memory, slower) |
| `--cache-gb N` | override `runtime.ram_cache.budget_gb` |
| `--progress {tqdm,plain}` | `plain` prints one line per epoch — better for log files |
| `--max-minutes N` | abort a run that exceeds the thermal budget |
| `--dry-run` | build the sweep and print it without training |

### Performance notes

- **RAM cache.** Windowed tensors are materialised once into pinned host memory
  (budget in `runtime.ram_cache.budget_gb`, default 16 GB of the box's 24 GB) and
  persisted as `.npy` memmaps under `data/interim/cache`, so re-runs and
  additional seeds skip windowing entirely. With the dataset resident,
  `num_workers=0` is faster than any worker count — there is nothing to prefetch.
- **Mixed precision** (`bf16`) is on by default and safe on Blackwell.
- **Thermals.** This box has thrown CPU-overheat faults on long runs. Each
  individual run is sized to stay under `runtime.max_minutes_per_run` (10 min);
  the trainer warns and stops rather than pushing past it.
- **Disk.** Every writing script asserts ≥5 GB free before it starts.
  `data/` is gitignored; no multi-GB intermediates are cached.

---

## 4. Methodological rules (enforced in code, covered by tests)

These are the failure modes that get air-quality papers torn apart. Each has an
assertion or a test in `tests/`.

1. **No future meteorology.** Forecasting PM2.5 at *t+h* uses meteorology only up
   to and including *t*. An oracle-met variant exists but is labelled an upper
   bound and never reported as the headline model.
2. **Split before fitting anything.** Chronological only. Scalers and imputers fit
   on train, then transform val/test. No shuffling, ever.
3. **No interpolation across split boundaries.** Imputation is backward-looking
   (limited forward-fill) or fitted per split.
4. **Gap-aware windowing.** Any window whose internal timestamps are not
   contiguous is rejected; the count of rejected windows is reported.
5. **Direct multi-horizon.** A separate head/model per *h* ∈ {1, 3, 6, 12, 24}. No
   recursive rollout.

Split boundary dates are recorded in `config.yaml` and printed in every table
caption.

---

## 5. Layout

```
config.yaml         ALL hyperparameters, paths, seeds, horizons, citations
Makefile / make.ps1 pipeline targets
src/
  data/       fetch_openaq.py, fetch_power.py, fetch_uci_fallback.py
  features/   build_features.py
  models/     baselines.py, trees.py, sequence.py, classifier.py
  eval/       metrics.py, split.py, evaluate.py
  green/      energy.py, complexity.py
  viz/        figures.py, tables.py
scripts/      00_check_env.py … 11_make_report.py
results/      tables/*.csv + *.tex (booktabs), figures/*.png + *.pdf (300 dpi),
              logs/, results.json  <- single source of truth
reports/      DATA_AUDIT.md, RESULTS.md, abstract_facts.json
data/         raw/ interim/ processed/   (gitignored)
```

Nothing is hardcoded in a script: horizons, lags, split dates, seeds, model
sizes and learning rates all live in `config.yaml`.

---

## 6. Honesty notes

- **CodeCarbon reports estimates, not metered measurements**, and falls back to
  modelled values when it cannot read Intel RAPL counters (the normal case on
  Windows). This is stated in `RESULTS.md` and in the paper's limitations.
- Constants requiring a citation — AQI breakpoints, Bangladesh grid carbon
  intensity, the national PM2.5 standard, monsoon month boundaries — are held in
  `config.yaml` under `citations:` with an explicit
  `VERIFIED` / `PENDING_VERIFICATION` status. **A pending value blocks the number
  that depends on it**; the pipeline leaves the cell blank and flags it rather
  than filling in an invented figure.
- Every filtering and imputation decision is logged with a count of affected rows.

## 7. Data sources

| Source | Role | Access |
|---|---|---|
| OpenAQ S3 archive | PM2.5 target | `s3://openaq-data-archive`, unsigned |
| OpenAQ v3 REST API | monitor discovery only | free API key in `.env` |
| NASA POWER hourly point | meteorological drivers | no key; values in **UTC**, `-999` = missing |
| UCI Beijing Multi-Site (id 501) | fallback + cross-city check | `ucimlrepo` |
