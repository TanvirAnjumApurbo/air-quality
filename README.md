# Short-Horizon PM2.5 Forecasting for Dhaka

**A Lightweight Sequence Model on Fused OpenAQ and NASA POWER Data**

Can a compact recurrent model (GRU/LSTM/DLinear/NLinear, ≤100k parameters)
forecasting Dhaka PM2.5 at 1–24 hour horizons beat honest classical baselines when
air-quality history is fused with meteorological drivers — and **does record
fragmentation, rather than model class, decide which method wins?**

**Contributions**

1. **A controlled gap-injection experiment** (§7 below; `RESULTS.md` §8), which is
   the study's central claim. A near-complete record is degraded two ways that
   remove an *identical* number of observed hours and differ only in arrangement,
   so fragmentation is manipulated with data volume held constant. The sequence
   tier loses −0.0394 skill to arrangement alone (50 matched pairs, Wilcoxon,
   Holm p < 0.0001), and it is the only model family whose gap is significant on
   **all three** donor records.

   Two qualifications are reported with it rather than left for a referee.
   Significance against a family's *own* null on every donor is not a contrast
   between families, and when the families are differenced within matched
   (coverage, seed, donor) triples the sequence tier separates from the linear
   and climatological families but **not from the tree ensembles**
   (−0.0102, p = 0.15); the conjunction it would need — worse than every other
   family — does not hold pooled or on any donor alone. What does hold is a
   dose–response: the sequence gap steepens by −0.0226 per 10 percentage points
   of coverage lost (Holm p = 0.016) where the tree gap is flat.

2. **A mechanism for that result, and a law** (`RESULTS.md` §8/§9). A row is
   scored only if it carries the feature set's full backward reach of unbroken
   history, so a gap costs the hours it removes **plus the reach behind it**.
   The cost therefore follows the *number* of gaps, not their length — which is
   the arm contrast in closed form. Written as
   `usable ≈ O·exp(β₀ − αRk/O)` and fitted on one donor's 101 cells at R = 192,
   it predicts **404 held-out cells at R² = 0.982** (median error 3.5%). Those
   cells are two different extrapolations: 202 from two other stations at the
   same radius, which asks whether the constants travel between records, and 202
   from the *fitting* station rebuilt at R = 72 and R = 48, which asks whether R
   is a factor of the form or a scale the constants absorbed. It holds on both
   (R² 0.964–0.983). Scoring that second group at the fitted radius instead
   drops the pooled R² to 0.461, which is the difference the radius term is
   carrying. Its α lands inside the 95% interval of the share of gaps outliving
   the forward-fill limit on both cities — so the constant is the imputation
   policy, not a free parameter.
3. **A corrected benchmark protocol.** Non-degenerate sequence inputs, modern
   linear baselines (DLinear/NLinear, Zeng et al. 2023), block-bootstrap CIs,
   Holm–Bonferroni across the Diebold–Mariano family (one test per horizon, best
   sequence against best classical — five tests, not a full pairwise matrix), and
   a Model Confidence Set — so the paper states which models are
   *indistinguishable* from the best rather than over-reading a rank order.

   The horizon matters and is stated rather than selected: the sequence tier wins
   significantly at h = 24, but **lightgbm beats it significantly at h = 1 and
   h = 12**. Any single-horizon headline is a choice, and this one is reported
   next to the four horizons that do not support it.
4. **Multi-horizon forecasting benchmarked against the baselines air-quality papers
   usually omit** — persistence, seasonal-naïve, climatology, SARIMAX — with a
   skill score vs persistence as a headline column.
5. **Forecast availability as a reported quantity, and what the lookback costs**
   (`RESULTS.md` §8/§9). Every accuracy number in this study — and in the
   air-quality forecasting literature it sits in — is conditional on the model
   being able to forecast at all, and that condition is not usually reported.
   Here it is 76.4% of test hours on the primary record and 64–73% on the
   near-complete comparison stations, so **coverage and availability order these
   records in opposite directions**. Capping the backward reach at 24 h costs
   nothing measurable in accuracy on the hours both can serve (no
   Holm-corrected Diebold–Mariano test rejects) while raising availability to
   97.6%, which over all hours is worth +0.016 skill. The configured 168-hour
   reach buys no accuracy and costs fourteen points of availability.

6. **A cross-city rank-transfer check** (`RESULTS.md` §7) reporting that the ranking
   does *not* transfer between Dhaka and Beijing (Spearman +0.20), together with the
   evidence that the comparison city cannot support a ranking at all — its 95% Model
   Confidence Set retains 9 of 9 candidates, persistence included. The coverage
   contrast that motivates the comparison is shown there to be the wrong summary
   statistic: the two records are far closer in availability than in coverage, and
   ordered the other way.

A green-AI efficiency analysis (accuracy vs parameters vs estimated training CO₂e)
and a secondary AQI-category classification task are built from the same pipeline
and reported as subsections, not headline claims.

**What the results currently say.** At 24 h the selected `gru_h64_l1` (16,193
parameters) reaches RMSE 57.24 µg/m³ [52.38, 62.17] against random forest's 58.50,
Diebold–Mariano p = 0.0123 (Holm-adjusted 0.037). The 95% Model Confidence Set
nevertheless retains 5 of 9 candidates, so the sequence model and the tree
ensembles are **not** separable on this record — the honest reading is a tier-level
result, not an architecture-level one. See `reports/RESULTS.md`, which is generated
and holds every number.

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

| Phase | Command | Script | Deliverable |
|---|---|---|---|
| 0 | `.\make.ps1 check` | `00_check_env.py` | environment / CUDA / disk report |
| 1 | `.\make.ps1 discover` | `01_discover_openaq.py` | OpenAQ Dhaka monitor table — **downloads nothing** |
| 1 | `.\make.ps1 data` | `02_fetch_data.py` | OpenAQ + NASA POWER + UCI Beijing into `data/raw` |
| 1 | `.\make.ps1 audit` | `03_data_audit.py` | `reports/DATA_AUDIT.md` — **hard gate before modelling** |
| 2 | `.\make.ps1 features` | `04_build_features.py` | features + chronological splits |
| 2 | `.\make.ps1 test` | `pytest tests` | 81 leakage + unit tests (must pass) |
| 3 | `.\make.ps1 baselines` | `05_run_baselines.py` | Tier 1 baselines + Tier 2 classical ML |
| 4 | `.\make.ps1 deep` | `06_train_sequence.py` | Tier 3 GRU/LSTM/DLinear/NLinear, all seeds |
| 5 | `.\make.ps1 classify` | `07_train_classifier.py` | AQI-category classifier |
| 5 | `.\make.ps1 green` | `08_green_measure.py` | params, MACs, latency, energy, CO₂e |
| 5 | `.\make.ps1 eval` | `09_evaluate.py` | stratified metrics + Diebold-Mariano |
| 6 | `.\make.ps1 figures` | `10_make_figures.py` | all figures, 300 dpi PNG + PDF |
| 6 | `.\make.ps1 stability` | `19_selection_stability.py` | tier-3 selection stability — reads existing runs, no refit |
| 6 | `.\make.ps1 report` | `11_make_report.py` | `RESULTS.md`, `abstract_facts.json` |

`make all` runs every row above in order, with fixed seeds. The git commit hash is
written into `results/results.json`.

### Cross-city, ablation and replication targets

| Command | Script(s) | Deliverable |
|---|---|---|
| `.\make.ps1 beijing` | `12_prepare_beijing.py` then 03–11 | the whole pipeline again on the Beijing record |
| `.\make.ps1 beijing-post` | 08–11 | Beijing's downstream stages only, after a resumed sweep |
| `.\make.ps1 cross-city` | `13_cross_city.py` | rank-transfer comparison → §6 (needs `all` and `beijing`) |
| `.\make.ps1 ablation` | `16_gap_injection.py`, `17_ablation_analysis.py` | the gap-injection grid → §7 (needs `beijing`) |
| `.\make.ps1 donor-configs` | `14_make_donor_configs.py` | `config/donors/*.yaml` from `donors.yaml` |
| `.\make.ps1 donor-list` | — | prints the donor slugs; 2 s, run this before `donors` |
| `.\make.ps1 donors` | prep + `16`/`17` per donor | replication grids — **~3 h per donor** |
| `.\make.ps1 replication` | `18_donor_replication.py` | cross-donor comparison → §7b (needs ≥2 grids) |
| `.\make.ps1 everything` | all of the above | every result in the paper |

### Availability, the lookback frontier and the mediation test

| Command | Script(s) | Deliverable |
|---|---|---|
| `.\make.ps1 law` | `20_missingness_law.py` | amplification law, availability audit, the decision rule → §9 (**free**, seconds) |
| `.\make.ps1 lookback-configs` | `15_make_lookback_configs.py` | `config/lookback/*.yaml`, one per sterilisation radius |
| `.\make.ps1 frontier` | `21_lookback_frontier.py`, `22_availability_frontier.py` | the three arms refitted and scored honestly (**TRAINS**) |
| `.\make.ps1 frontier-analysis` | `22_availability_frontier.py` | re-score an existing frontier; no refit |
| `.\make.ps1 mediation` | `23_mediation.py` | does the arm gap shrink with the radius (needs the radius grids) |

`law` reads records and grids already on disk and trains nothing, so it can be run
at any point. Run it **after** `frontier` if you want `fig16`'s third panel: the
first two panels are identities the record alone supplies, the third is the
measured outcome and is omitted rather than invented when the arms have not run.

### Run order matters

- `08_green_measure.py` and `09_evaluate.py` read the sequence-model checkpoints,
  so `06_train_sequence.py` must have run first.
- `19_selection_stability.py` must precede `11_make_report.py`, which renders its
  subsection. `all` and both Beijing chains already order them.
- **Always re-run `17_ablation_analysis.py` after `16_gap_injection.py`.** `16`
  rebuilds its output file on every cell and drops the `analysis` block `17` wrote
  — deliberately, because an analysis computed over a different cell set is worse
  than none, since it looks finished.
- `20_missingness_law.py` is free but not order-free: its decision figure draws the
  *measured* payoff from both cities' `availability_frontier*.json`, so run it after
  `22_availability_frontier.py` on both cities to get the full figure. Re-run it for
  each city — the payload it writes is per-city and `11_make_report.py` reads its
  own city's copy.
- `11_make_report.py` reads `results/results.json` and writes nothing that is not
  already in it — if a phase has not run, the corresponding section is omitted
  rather than invented.

---

## 3. Training commands (run these yourself)

Every training entrypoint prints live progress, checkpoints each epoch, and
resumes automatically from the last checkpoint if interrupted.

### Sequence models (Tier 3)

**Choose the training recipe first.** `--tune` searches learning rate, weight
decay and the Huber delta on a reduced grid, scored on **validation loss only**,
and prints a ready-to-paste block. Paste it into `models.sequence.train` in
*both* configs before the sweep — the comparison city inherits the recipe on
purpose. The full grid and every candidate's score are written to
`results.json` under `sequence_recipe_search`, so the search is reportable
rather than something that happened offstage.

```powershell
.\.venv\Scripts\python.exe scripts\06_train_sequence.py --config config.yaml --tune --progress plain
```

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
6. **Never select a model on test error.** Tier 3 is ranked by mean validation loss,
   Tier 2 by cross-validation score, and Tier 1 by test RMSE *only* because those
   have no hyperparameters and therefore no selection to bias. Ranking candidates by
   test error and then reporting that error is circular and biases the headline
   downward by the spread of the pool.

Split boundary dates are recorded in `config.yaml` and printed in every table
caption.

**Rule 6 has a cost, and the study measures it rather than assuming it away.**
`19_selection_stability.py` re-runs the selection rule on single seeds, on
leave-one-seed-out subsets, and on 2,000 seed resamples. At 24 h on Dhaka no single
seed picks the reported architecture on its own and 10 candidates win at least once
— the *identity* of the winner is not stable. The consequence is bounded: selecting
on validation rather than test costs +0.05 RMSE, because the candidates validation
cannot separate are near-ties, which is what the Model Confidence Set says too. On
Beijing at 24 h the same diagnostic reads the other way (regret +6.63 RMSE, Spearman
ρ = −0.25 against the test ranking), which is why no architecture-level claim is made
from that record.

---

## 5. Layout

```text
config.yaml         ALL hyperparameters, paths, seeds, horizons, citations (Dhaka)
config_beijing.yaml the cross-city record; derived by copying config.yaml
config/donors/      generated per-donor configs — never hand-edited
donors.yaml         the donor registry, and what a donor does and does not prove
Makefile / make.ps1 pipeline targets
src/
  data/       fetch_openaq.py, fetch_power.py, fetch_uci_fallback.py
  features/   build_features.py, pipeline.py, gap_injection.py
  models/     baselines.py, trees.py, sequence.py, classifier.py, data.py
  eval/       metrics.py, split.py, evaluate.py, ablation.py
  green/      energy.py, complexity.py
  viz/        figures.py, tables.py
scripts/      00_check_env.py … 19_selection_stability.py
results/      tables/*.csv + *.tex (booktabs), figures/*.png + *.pdf (300 dpi),
              logs/, results.json  <- single source of truth
              ablation_gap_injection*.json, donor_replication.json  <- see below
reports/      DATA_AUDIT.md, RESULTS.md, abstract_facts.json
data/         raw/ interim/ processed/   (gitignored)
```

Nothing is hardcoded in a script: horizons, lags, split dates, seeds, model
sizes and learning rates all live in `config.yaml`.

Two structural points worth knowing before editing:

- **`results.json` is the only source of reported numbers.** The gap-injection
  experiment is the one deliberate exception, because it spans several donor
  configs and would otherwise land in whichever donor ran last. It owns
  `ablation_gap_injection*.json` (one per donor) and `donor_replication.json`. Both
  are generated; neither is ever written by hand, and the rule that matters — no
  transcribed numbers — still holds.
- **`src/features/pipeline.py::build_feature_matrix` is shared** by
  `04_build_features.py` and the ablation, so a degraded matrix is rebuilt through
  exactly the same code. The boundary purge and per-horizon validity masks are where
  the leakage rules live; a second copy would be a second definition of a usable row.

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

## 7. The gap-injection experiment

Two cities that differ in coverage, span, climate and instrument at once cannot
establish what causes a ranking to change. This experiment makes fragmentation the
manipulated variable on a single record.

> **Read the observational contrast carefully — it runs the other way, and the
> gradient it runs along is weaker than it looks.** Forecast availability, not
> coverage, is what reaches the model: 76.4% here against 72.5% on the comparison
> station, a four-point gap where coverage shows sixteen — and pointing the other
> way. See `RESULTS.md` §8.
>
> The sequence tier ranks **1** on the *more fragmented* record (Dhaka, 82.3% coverage)
> and **3** on the near-complete one (Beijing, 98.9%). That is the opposite of what
> a fragmentation account predicts, and `RESULTS.md` §7 reports it rather than
> setting it aside. The two-city contrast confounds fragmentation with a harder
> 24-hour problem, a shorter test period and a seasonal transition; it has no valid
> counterfactual. The injection experiment does, which is why the mechanism is
> tested here instead of inferred from the pair.

A near-complete donor is degraded to a series of coverage levels by two arms
that remove **exactly the same number of observed hours** and differ only in how
those hours are arranged:

- **fragmented** — many short outages, lengths drawn from Dhaka's own empirical
  gap-length distribution (2,074 gaps, 64% of them a single hour, longest 2,712 h);
- **contiguous** — the same hour count as a few long blocks, spread evenly so the
  arms differ in contiguity and not in seasonal composition.

Three controls make the comparison mean what it claims:

1. **The test period is never degraded**, so every cell is scored on identical
   rows and RMSE stays comparable across the whole grid.
2. **Hyperparameters are fixed** at the undegraded record's selections. Re-tuning
   per cell would let search compensate for the damage.
3. **The optimizer-step budget is equalised**, not the epoch budget. Fragmentation
   shrinks the training set, so at a fixed 60 epochs a fragmented cell takes 16
   steps/epoch where a contiguous one takes 65 — four times fewer gradient
   updates. Epochs, patience and the minimum-epoch floor are rescaled per cell so
   the degradation cannot be undertraining in disguise.

```powershell
# --profile-config names the record whose gap-length distribution is injected;
# --config names the donor being degraded. They are different cities on purpose.
.\.venv\Scripts\python.exe scripts\16_gap_injection.py --config config_beijing.yaml `
    --profile-config config.yaml --progress plain
.\.venv\Scripts\python.exe scripts\17_ablation_analysis.py --config config_beijing.yaml
```

Or `.\make.ps1 ablation`, which passes both. The grid is 101 cells (5 coverage
levels × 2 arms × 10 injection seeds, plus one undegraded reference) and takes
about 2.5 h. It is resumable and skips cells already recorded in the file named by
`ablation.gap_injection.output_name`. Each cell's feature matrix is deleted after
scoring unless `--keep-cells` is given, because the full grid would otherwise hold
several GB.

**Result.** Each (coverage level, injection seed) is one matched pair across the two
arms, giving 50 pairs per family; the representative model per family is fixed on the
undegraded record and never re-chosen per arm, so the difference cannot absorb a
change of model. Wilcoxon signed-rank, Holm-corrected across families:

| Family | Mean arm gap | 95% CI | p (Holm) |
|---|---|---|---|
| sequence | −0.0394 | [−0.0565, −0.0236] | <0.0001 |
| naive | −0.0358 | [−0.0609, −0.0140] | 0.0086 |
| trees | −0.0263 | [−0.0393, −0.0125] | 0.0003 |
| linear | −0.0016 | [−0.0081, +0.0047] | 0.9542 |

Negative means fragmentation costs that family more than the same hours removed
contiguously.

### 7b. Does it replicate on another record?

One record cannot distinguish a property of fragmentation from a property of that
station, so the experiment is repeated on further donors.

```powershell
.\make.ps1 donor-list      # confirm it will iterate before starting hours of work
.\make.ps1 donors          # ~3 h per donor
.\make.ps1 replication     # cross-donor comparison, seconds
```

Donors are registered in `donors.yaml`; `14_make_donor_configs.py` generates their
configs. Replication is deliberately strict — same sign **and** Holm significance on
every donor:

| Family | Wanliu | Dingling | Dongsi | Replicates |
|---|---|---|---|---|
| sequence | −0.0394\* | −0.0200\* | −0.0273\* | **yes** |
| trees | −0.0263\* | −0.0120 | −0.0179\* | no |
| linear | −0.0016 | −0.0298\* | −0.0205\* | no |
| naive | −0.0358\* | −0.0024 | −0.0211\* | no |

The naive row is the internal control and the reason more than one donor was run:
models that never read the training record should not care how it is arranged. On
Wanliu alone it collapses to −0.0958 at 75% coverage, which would have argued that
fragmentation degrades *anything* estimated from the record. It does not replicate
(Dingling −0.0038, Dongsi −0.0373), so that collapse is a property of Wanliu.

**Scope limit, and it matters.** All donors are stations of the *same* UCI Beijing
Multi-Site archive (id 501) — one four-year window, one regional weather regime,
spatially correlated PM2.5. This is a **station-robustness check**, not a multi-city
panel. It can falsify a donor-specific effect, which is the cheapest way to kill a
wrong claim; it cannot establish generality.

## 8. Data sources

| Source | Role | Access |
|---|---|---|
| OpenAQ S3 archive | PM2.5 target | `s3://openaq-data-archive`, unsigned |
| OpenAQ v3 REST API | monitor discovery only | free API key in `.env` |
| NASA POWER hourly point | meteorological drivers | no key; values in **UTC**, `-999` = missing |
| UCI Beijing Multi-Site (id 501) | fallback, cross-city check, **and all three gap-injection donors** | `ucimlrepo` |

The donors are three stations of that one archive (Wanliu, Dingling, Dongsi), which
is why §7b is a station-robustness check and not a multi-city panel — see
`donors.yaml`, which records the same caveat next to the registry itself.
