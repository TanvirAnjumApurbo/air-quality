# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`README.md` covers setup, the phase-by-phase pipeline table, every make target,
training flags, the six methodological rules, the gap-injection experiment and its
donor replication. Read it first. This file covers what the README does not: the
architecture you have to read several files to see, and the invariants that will
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
.\make.ps1 donor-list      # print the donor slugs; 2 s, run before `donors`
.\make.ps1 donors          # prep + gap injection for every replication donor
.\make.ps1 replication     # cross-donor comparison only (needs `ablation` + `donors`)
.\make.ps1 stability       # tier-3 selection stability; reads existing runs, no refit
.\make.ps1 everything      # all + beijing + cross-city + ablation + report
.\make.ps1 law             # amplification law + availability audit; free, seconds
.\make.ps1 frontier        # lookback frontier; TRAINS, then scores it
.\make.ps1 lookback-configs # generate config/lookback/*.yaml
.\make.ps1 mediation       # arm gap against radius (needs the radius grids)
```

`stability` (19) must precede `report` (11), which renders its subsection; `all`
and both Beijing chains already order them.

**`donors` is the expensive target: budget ~3 h per donor**, measured — the
101-cell grid took 154 min (Dingling) and 162 min (Dongsi), plus ~23 min of tier-2
fitting each. Two donors is most of a night. It is also the only one of these that
trains. `stability`, `replication` and `cross-city` are pure analysis over records
already on disk and finish in seconds; `ablation` on an existing grid resumes every
cell and returns in 0.0 min, which is what a correct no-op looks like here.

Lint and format (ruff config lives in `pyproject.toml`; docstrings and type
annotations on public functions are enforced by `D` and `ANN` rules):

```powershell
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check src scripts tests
.venv/Scripts/python.exe -m ruff format src scripts tests     # writes
```

Tests — all 43 live in `tests/test_leakage.py`:

```powershell
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m pytest tests -m leakage           # leakage guards only
.venv/Scripts/python.exe -m pytest tests -k test_no_valid_row_spans_a_gap -v
```

Markers are declared in `pyproject.toml` under `--strict-markers`: `leakage`,
`slow`, `network`. Seven tests (`def test_control_*`) are negative controls — they
inject the forbidden mistake and assert the guard fires. If you relax a guard,
those fail, which is the point. When you add a guard, add its negative control in
the same commit; a guard with no control is untested and will silently rot.

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

The gap-injection experiment is the one deliberate exception, and it owns two
files rather than one: §8 of `RESULTS.md` reads
`results/ablation_gap_injection*.json` (one per donor, named by
`ablation.gap_injection.output_name`), and §8b reads
`results/donor_replication.json`. Both are generated — by `17_ablation_analysis.py`
and `18_donor_replication.py` — and neither is ever written by hand. They cannot
live in `results.json`: the experiment spans several donor configs, so it would
land in whichever donor happened to run last, while the injected gap profile and
the claim both belong to the primary city. The rule that matters (no transcribed
numbers) still holds. Do not add a source outside this experiment.

`16_gap_injection.py` **rebuilds its output file from scratch on every cell**, which
drops the `analysis` block `17` wrote. That is intentional — an analysis computed
over a different cell set is worse than none, because it looks finished. Always
re-run `17` after `16`. `11_make_report.py` warns rather than silently omitting §8
when it finds cells with no analysis, because omitting it once renumbered
Limitations over the top of the study's central section.

### Replication donors are generated configs, not hand-maintained ones

`donors.yaml` is the registry; `14_make_donor_configs.py` generates
`config/donors/<slug>.yaml` from `config_beijing.yaml` by section-scoped regex line
edits, then verifies the result parses, differs in exactly the intended keys, and
preserves the comment count. Section scoping is required, not defensive: `tables`
and `figures` each appear at indent 2 under **both** `paths` and `output`, so an
unscoped replacement hits two lines and the verifier refuses.

Each donor needs its own `ablation.gap_injection.output_name`. Two donors sharing
one output path is the defect this repo hits most often, and here it is worst:
`16`'s resume keys on `(arm, coverage, injection_seed)` and knows nothing about
which *record* produced a cell, so a shared path makes every cell read as "already
recorded" and the run reports a complete grid for a station it never touched. `16`
now refuses to resume when the payload's `donor` label disagrees with the config's
station, and that check is fatal rather than a warning.

**Donors are further stations of one archive** (UCI Beijing Multi-Site, id 501):
one four-year window, one weather regime, spatially correlated PM2.5. It is a
station-robustness check that can falsify a donor-specific effect — which is the
cheapest way to kill a wrong claim — and it is **not** evidence of generality.
Never describe it as a multi-city panel. `donors.yaml` says this too; keep both.

`src/eval/ablation.py` holds the shared paired-test code (`paired_arm_gaps`,
`test_arm_gaps`, `family_contrasts`, `dose_response`, `resume_identity_problem`,
the family map) so `17` and `18` cannot compute the arm gap two different ways and
make a disagreement between donors look real.

**The family formerly called `naive` is `climatological`, and it was never a
control.** `paired_arm_gaps` picks each family's representative by highest skill on
the reference cell; persistence scores identically zero there by construction, so
climatology always wins the slot — and climatology is fitted on the *degraded*
training split. There is no useful no-training-data control for skill in this
design: because the test period is never degraded, any model reading nothing from
training has an arm gap of exactly zero in every cell, so it cannot vary and
cannot falsify anything. `17` asserts that invariance by equality instead, which
is stronger than a test that can only fail to reject.

**Significance on every donor is not a contrast between families.** The
replication rule tests each family against its own null; the claim that the
sequence tier is the family fragmentation hurts *most* is a comparison, and
`family_contrasts` is the test of it — differencing families within matched
(coverage, seed, donor) triples, as an intersection-union test because the claim
is a conjunction. It does not currently reject: the sequence tier separates from
`linear` and `climatological` but not from `trees`.

### Data flow

```text
02_fetch_data      -> data/interim/*.parquet        (target, meteorology, QC ledger)
03_data_audit      -> reports/DATA_AUDIT.md          HARD GATE, prints a verdict
04_build_features  -> data/processed/features.parquet + features_meta.json + scaler.json
05..07 (models)    -> results.json runs[] + data/interim/predictions/
08..09 (green/eval)-> results.json green/significance/stratified
10, 11             -> results/figures, results/tables, reports/
13_cross_city      -> results.json cross_city{}      (must run before the final report)
14_make_donor_cfgs -> config/donors/*.yaml           (from donors.yaml)
15_make_lookback   -> config/lookback/*.yaml        (one per sterilisation radius)
16_gap_injection   -> results/ablation_gap_injection*.json  cells{}   (one per donor)
17_ablation_analys -> same file, analysis{}          (ALWAYS re-run after 16)
18_donor_replicat  -> results/donor_replication.json (needs >=2 donor grids)
19_selection_stab  -> results.json selection_stability[]  (before 11; §3 renders it)
20_missingness_law -> results/missingness_law.json  FREE, no training
21_lookback_front  -> results.json runs[] tagged experiment=lookback_frontier
                      + results/lookback_frontier{,_predictions}.{json,npz}   TRAINS
22_availability    -> results/availability_frontier.json  (needs 21)
23_mediation       -> results/mediation_<slug>.json  (needs >=2 radius grids)
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

### The lookback cap sets the supervision floor, and they can be decoupled

`features.lookback_h` caps the backward reach of every engineered history
feature; `features.history_floor_h` sets how much unbroken history a row must
carry to be scored. Both are `null` by default, and with both null the build is
**bit-identical** to what it was before they existed — every `valid_h{h}` mask
elementwise equal, every numeric column unchanged. Keep it that way: the null
path is the headline pipeline.

They are normally the same number, because a row cannot support a feature that
reaches further back than its own history. Setting the floor *above* the cap
decouples them, which is the identical-row-count control arm of the frontier —
the same device the ablation uses when it holds the removed hour count constant.
Setting it *below* raises, and must keep raising: the deepest lag would be
NaN-because-outside-the-run, and `get_split_arrays` replaces NaN with 0.0 **after**
scaling, i.e. with the training mean, with no warning. The negative control
`test_control_short_floor_feeds_the_train_mean_to_the_model` demonstrates the
harm the guard prevents.

`capped_lookbacks` is the single place the cap is applied. Four readers —
`add_target_history`, `add_met_history`, `derived_history_columns` and
`max_backward_dependency` — consume the same four config lists, and applying the
cap in some but not others would make the built columns disagree with the names
generated to exclude them from the sequence channels. That is exactly the drift
`derived_history_columns` exists to prevent.

**A cap does not change the sequence tier's input by one channel.**
`sequence_channel_columns` already excludes every derived-history column, so for
tier 3 the cap moves only the validity floor. Tier-3 runs are a function of the
**sterilisation radius**

```text
R = max(lookback_h, window_h - 1) + horizon_h
```

and nothing else, so configurations sharing a radius share a run. This also means
a naive lookback sweep is a trap: at the ablation's 48-hour window, caps of 48, 24
and 12 all collapse onto the same experiment.

**Two radii that share a window will resume each other's checkpoints unless the
cell tree is separated.** `train_one` resumes by `(run_tag, run_id)` and `run_id`
encodes the window but not the lookback, while the cell directory derives from
`paths.data_interim`. `16_gap_injection.py` therefore puts a capped run under
`ablation_R{radius}` and leaves an uncapped run on the historical `ablation` path
so existing grids still resume. Removing that would let the shorter radius
continue from the longer one's weights, silently.

### Availability is a first-class metric, and RMSE across arms is not comparable

`src/eval/availability.py` defines the **fixed evaluation universe**: scoreable at
all, in the split, purged at `UNIVERSE_FLOOR_H` (168 h, the deepest floor in the
sweep) so every arm's served set is a subset of one common set. A model that
declines the hard hours will always look good on the hours it accepts, so
`rmse_served` must never be compared between configurations with different
availability. The two comparisons that *are* valid, both produced by
`22_availability_frontier.py`, are the common subset and `all_hours_skill`.

`all_hours_skill` scores a fallback cascade over the universe and **raises** if the
chain is not ordered by declared history requirement. A chain is a selection, and
ordering it by test error is rule 6 one level up. It also raises if the reference
is undefined anywhere in the universe: persistence is the denominator precisely
because it needs no history and is therefore defined everywhere, and a denominator
that moved with the arm would not be a denominator.

### Side experiments write into `results.json` and must not reach the headline

The frontier writes runs into the city's own `results.json`, tagged
`experiment: "lookback_frontier"` with a lookback-tagged variant. Variant tagging
prevents a collision; it does **not** prevent `_best_per_horizon` from *selecting*
one of those runs as the headline model. `src/results.py::main_runs` is the
chokepoint every consumer of `runs[]` goes through, and a test greps the repo for
survivors. Nine call sites is nine places to forget.

### Tier vocabulary

Every run record carries a `tier`, and selection logic branches on it:

- **tier1** — persistence, seasonal-naive, climatology, SARIMAX. No hyperparameters.
- **tier2** — Ridge, RandomForest, XGBoost, LightGBM via `RandomizedSearchCV` +
  `TimeSeriesSplit` over train+val.
- **tier3** — GRU, LSTM, DLinear and NLinear under a 100k-parameter budget, 5 seeds
  each. The two linear architectures (Zeng et al., AAAI 2023) are baselines a
  time-series reviewer checks for first, and a linear model beating the recurrent
  one is a result, not a bug.

### Figures are sized for the printed column and carry no titles

`src/viz/figures.py::setup_style` is the only place figure typography is set, and
every figure script calls it. Three conventions hold across all 13 figures:

- **No `set_title` and no `suptitle`.** The title goes in the LaTeX caption. A title
  drawn into the artefact is a second, unversioned copy that drifts from the caption.
  Where a panel's identity is data-bearing rather than recoverable from its axes —
  the tiers in fig05, the model families in fig12 — it rides on a `(a)`-style
  `panel_label` or on the legend *title*, not on a chart title.
- **Two widths only**, `COL_SINGLE` (3.5 in) and `COL_DOUBLE` (7.16 in), the narrower
  of {Elsevier 90/190 mm, IEEE 3.5/7.16 in} so one artefact fits either template at
  `width=\linewidth`. Never rescale in LaTeX: `font_size: 9` in `config.yaml` is
  chosen against the printed column, and scaling the figure unpicks that.
- **Units in mathtext**, via `UNIT_PM25`. The literal `µg/m³` needs glyphs the serif
  stack does not always carry, and a missing glyph prints as a tofu box rather than
  failing — the previous fig08 shipped with one in its axis label.

**`Axes.add_artist` clips what it is given to the axes patch.** A second legend
anchored outside the axes therefore renders invisible *and* is dropped from the tight
bounding box, silently. fig08 needs three legends; they are `fig.legend(...)` with
`bbox_transform=ax.transAxes`, which coexist in `fig.legends` and survive the tight
bbox. Reach for `fig.legend`, not `ax.add_artist`, whenever a legend sits outside.

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

**No `make.ps1` parameter may be named after a PowerShell automatic variable.**
`param([string[]]$Args)` binds — the value is there in `$PSBoundParameters` — but
`$Args` read by name returns the *automatic* variable, which is empty because
everything bound to a declared parameter. `& $Py $path @Args` therefore splatted
nothing, and from the first commit until `HEAD` **every `make.ps1` step ran with no
arguments**, falling back to its argparse `--config` default. It hid for the whole
project because those defaults match the `all` target; it surfaced only when
`donors` silently re-ran the *primary* city (`03_data_audit` reporting 79,216 rows
— Dhaka's, not a donor's) and `16_gap_injection` resumed off Wanliu's grid in
0.0 min. `beijing` and `beijing-post` had the same defect and would have written
Dhaka results into Beijing's slot; those stages were only ever safe because they
were run as explicit `python scripts/…` commands. The parameter is now `$StepArgs`,
and `test_make_shim_does_not_declare_a_powershell_automatic_variable` fails if any
reserved name comes back.

**Never write `config.yaml` with `yaml.safe_dump`.** It discards all 335 comment
lines (323 in `config_beijing.yaml`), which hold the citations, source quotations,
and the record of two verified API discrepancies.
`04_build_features.py::write_back_boundaries` and `14_make_donor_configs.py` both do
targeted regex line edits and verify the round-trip. That is deliberate, not
laziness.

**Never select a model on test RMSE.** `_best_per_horizon` in `11_make_report.py`
ranks tier3 by mean validation loss, tier2 by CV score, and tier1 by test RMSE only
because those have no hyperparameters and therefore no selection to bias. Ranking
candidates by test error and then reporting that error is circular and biases the
headline downward by the spread of the pool.

`19_selection_stability.py` is the one place that reads test error *next to* the
selection rule, because selection regret is defined against it. Its `_select` takes
validation loss only and regret is computed after the choice is fixed; a leakage
test hands `_select` a frame whose test column is poisoned to invert the ranking and
asserts the choice does not move. Preserve that separation if you touch the file.
The diagnostic exists because the winner's *identity* is unstable — at h=24 on Dhaka
no single seed picks the reported architecture alone and 10 candidates win at least
once — while the *cost* is +0.05 RMSE. Report tier-level claims, not
architecture-level ones.

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
- **`16_gap_injection.py` exits non-zero after a grid that trained, and the grid is
  fine.** Measured on three grids: the script logs `gap-injection grid complete in
  N min`, prints its closing lines, writes the payload — and then returns 127 with
  no traceback, during interpreter shutdown. A no-op run that skips every cell
  exits 0, so it correlates with having trained, not with the result. Neither CUDA
  teardown, nor `n_jobs=-1` sklearn, nor LightGBM, nor XGBoost reproduces it alone.
  **Verify a grid by its contents, never by the exit code**: cell count, and
  whether the cells differ from the previous run where they should. Any driver
  script should retry rather than abort — a second invocation finds the cells
  already recorded and returns 0 in seconds.

- **CUDA.** Blackwell (sm_120) needs the cu128 build. `torch.cuda.is_available()`
  returning `True` is not evidence training will run — `resolve_device` and
  `00_check_env.py` launch a real kernel.
- `.gitignore` anchors `/data/`, not `data/`. Unanchored, it also matched
  `src/data/` and dropped the package from a commit.

## Provenance

`results.json` stamps `git_commit` at the time the *results* were computed, not when
the report was rendered — so `RESULTS.md` can legitimately show an older, `-dirty`
hash than `HEAD`. That is intended. Both cities currently read `1afdd57…-dirty`,
stamped when `19_selection_stability.py` last wrote them during the figure restyle,
**not** when their sweeps ran; the sweeps themselves predate that by days. Making the
hashes clean requires re-running the sweeps, which was considered and declined.

Note the asymmetry: `save_results` restamps `git_commit` on **every** write, so any
downstream stage that touches a city's `results.json` moves that city's hash forward
without a single model having been retrained. The hash records when the file was last
written, not when the numbers in it were produced.
