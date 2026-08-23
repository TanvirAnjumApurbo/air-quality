r"""Phase 8c: is a shorter reach actually better, or just scored on easier hours?

``21_lookback_frontier.py`` reports each arm's RMSE on the rows that arm can
serve. Those numbers are not comparable, and the whole point of this contribution
is that nobody notices: the status-quo arm is scored on 83.5% of the test hours
and the shortest arm on 97.6%, so the shorter arm's apparent advantage could be
nothing but an easier denominator. A model that refuses the hard hours will
always look good on the hours it accepts.

Two answers, and both are reported because they answer different questions.

**Common subset.** Restrict every arm to the rows the *deepest* arm can serve.
Identical hours, identical difficulty, so a Diebold-Mariano test on that subset
asks whether the shorter reach is better at forecasting rather than better at
declining. This is the conservative comparison and it favours nothing.

**All hours.** Score every arm over the fixed universe, with a declared fallback
answering the rows it cannot serve. This is what a practitioner actually
experiences, because an hour with no forecast is not an hour without error -- it
is an hour someone must fall back on something worse.

The fallback chain is ordered by history requirement, never by test error, and
``all_hours_skill`` refuses any other order: a chain is a selection, and ranking
a selection by test error is the circularity rule 6 exists to prevent.

The exact decomposition ``MSE_A - MSE_C = (MSE_A - MSE_B) + (MSE_B - MSE_C)``
separates supervision volume from feature richness. It is done in MSE because
RMSE differences are not additive, and displayed as RMSE.

Run::

    python scripts/22_availability_frontier.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.eval.metrics import (
    all_hours_skill,
    block_bootstrap_rmse,
    diebold_mariano,
    holm_bonferroni,
    rmse,
)
from src.utils import load_config, setup_logging
from src.viz.figures import COL_DOUBLE, panel_label, save_figure, setup_style
from src.viz.tables import write_table

NL = chr(10)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def load_bundle(results_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read the frontier payload and its prediction bundle.

    Args:
        results_dir: Directory holding the frontier outputs.

    Returns:
        ``(payload, predictions)``.

    Raises:
        FileNotFoundError: If either artefact is absent.
    """
    payload_path = results_dir / "lookback_frontier.json"
    bundle_path = results_dir / "lookback_frontier_predictions.npz"
    if not payload_path.exists() or not bundle_path.exists():
        raise FileNotFoundError(
            f"run 21_lookback_frontier.py first; missing {payload_path.name} or {bundle_path.name}"
        )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    with np.load(bundle_path, allow_pickle=False) as z:
        predictions = {k: z[k] for k in z.files}
    return payload, predictions


def split_key(key: str) -> tuple[str, str]:
    """Split a ``model|arm`` bundle key.

    Args:
        key: Bundle key.

    Returns:
        ``(model, arm)``.
    """
    model, _, arm = key.rpartition("|")
    return model, arm


def main() -> int:
    """Score the frontier honestly and draw it."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "22_availability_frontier")
    results_dir = cfg.path_for("results")

    payload, bundle = load_bundle(results_dir)
    y_true = bundle["y_true"]
    reference = bundle["reference"]
    horizon = int(payload["horizon_h"])
    arms = {a["label"]: a for a in payload["arms"]}
    ref_rmse = rmse(y_true, reference)
    log.info("universe %d hours, persistence RMSE %.4f", y_true.size, ref_rmse)

    model_keys = [k for k in bundle if k not in ("universe_index", "y_true", "reference")]
    by_model: dict[str, dict[str, np.ndarray]] = {}
    for key in model_keys:
        model, arm = split_key(key)
        by_model.setdefault(model, {})[arm] = bundle[key]

    # ---- the common subset: every arm on the deepest arm's rows -------------
    # "Deepest" is decided by the declared history requirement, not by which arm
    # happens to serve fewest rows on this record -- a rule that read the data
    # would be a selection.
    deepest_arm = "C"
    common = np.ones(y_true.size, dtype=bool)
    for arms_pred in by_model.values():
        if deepest_arm in arms_pred:
            common &= ~np.isnan(arms_pred[deepest_arm])
    log.info("common subset: %d of %d universe hours", int(common.sum()), y_true.size)

    rows: list[dict] = []
    dm_rows: list[dict] = []
    for model, arms_pred in sorted(by_model.items()):
        if deepest_arm not in arms_pred:
            continue
        base = arms_pred[deepest_arm]
        for arm, pred in sorted(arms_pred.items()):
            ok = common & ~np.isnan(pred)
            if ok.sum() < 30:
                continue
            ci = block_bootstrap_rmse(y_true[ok], pred[ok])
            row = {
                "model": model,
                "arm": arm,
                "lookback_h": arms.get(arm, {}).get("lookback_h"),
                "n_common": int(ok.sum()),
                "rmse_common": rmse(y_true[ok], pred[ok]),
                "ci_low": ci.lower,
                "ci_high": ci.upper,
                "mse_common": float(np.mean((y_true[ok] - pred[ok]) ** 2)),
            }
            rows.append(row)
            # Persistence and climatology do not read the feature set, so their
            # predictions are identical across arms and the loss differential is
            # identically zero. Including them would pad the Holm family with
            # degenerate tests and make every real one harder to reject.
            varies = not np.allclose(pred[ok], base[ok], rtol=0.0, atol=1e-12, equal_nan=True)
            if arm != deepest_arm and varies:
                dm = diebold_mariano(
                    y_true[ok],
                    pred[ok],
                    base[ok],
                    horizon=horizon,
                    name_a=arm,
                    name_b=deepest_arm,
                )
                dm_rows.append(
                    {
                        "model": model,
                        "arm": arm,
                        "vs": deepest_arm,
                        "dm_stat": dm.statistic,
                        "better": dm.better,
                        "p_raw": dm.p_value,
                    }
                )

    common_table = pd.DataFrame(rows)
    dm_table = pd.DataFrame(dm_rows)
    if not dm_table.empty:
        adjusted = holm_bonferroni(dm_table["p_raw"].tolist(), alpha=args.alpha)
        dm_table["p_holm"] = [r["p_adjusted"] for r in adjusted]
        dm_table["significant"] = [r["reject"] for r in adjusted]

    # ---- the exact decomposition -------------------------------------------
    decomp_rows = []
    mse = common_table.set_index(["model", "arm"])["mse_common"].to_dict()
    for model in sorted(by_model):
        for cap in sorted({a["lookback_h"] for a in payload["arms"] if a["lookback_h"]}):
            keys = [(model, "C"), (model, f"B{cap}"), (model, f"A{cap}")]
            if not all(k in mse for k in keys):
                continue
            mse_c, mse_b, mse_a = (mse[k] for k in keys)
            decomp_rows.append(
                {
                    "model": model,
                    "lookback_h": cap,
                    "rmse_C": float(np.sqrt(mse_c)),
                    "rmse_B": float(np.sqrt(mse_b)),
                    "rmse_A": float(np.sqrt(mse_a)),
                    "total_mse_delta": mse_a - mse_c,
                    "volume_mse_delta": mse_a - mse_b,
                    "richness_mse_delta": mse_b - mse_c,
                    "residual": (mse_a - mse_c) - ((mse_a - mse_b) + (mse_b - mse_c)),
                }
            )
    decomp = pd.DataFrame(decomp_rows)
    if not decomp.empty:
        worst = float(decomp["residual"].abs().max())
        log.info("decomposition residual max |.| = %.3e (must be ~0 by construction)", worst)

    # ---- all hours, with the declared fallback ------------------------------
    all_hours_rows = []
    for model, arms_pred in sorted(by_model.items()):
        need = {}
        for arm in arms_pred:
            cap = arms.get(arm, {}).get("lookback_h")
            floor = arms.get(arm, {}).get("floor_h")
            need[arm] = int(floor if floor is not None else (cap if cap is not None else 168))
        for arm, pred in sorted(arms_pred.items()):
            served = {arm: pred, "persistence": reference}
            order = [arm, "persistence"]
            requirement = {arm: need[arm], "persistence": 0}
            single = all_hours_skill(
                y_true, served, order=order, history_requirement=requirement, reference=reference
            )
            all_hours_rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "policy": "single",
                    "availability": single.availability,
                    "rmse_served": single.rmse_served,
                    "rmse_all_hours": single.rmse_all_hours,
                    "skill_all_hours": single.skill_all_hours,
                }
            )

        # The cascade: deepest arm first, then progressively shallower, then
        # persistence. Ordered by history requirement and checked to be so.
        chain = sorted(arms_pred, key=lambda a: -need[a])
        served = {a: arms_pred[a] for a in chain}
        served["persistence"] = reference
        requirement = {a: need[a] for a in chain}
        requirement["persistence"] = 0
        cascade = all_hours_skill(
            y_true,
            served,
            order=[*chain, "persistence"],
            history_requirement=requirement,
            reference=reference,
        )
        all_hours_rows.append(
            {
                "model": model,
                "arm": "cascade",
                "policy": "cascade",
                "availability": 1.0,
                "rmse_served": cascade.rmse_served,
                "rmse_all_hours": cascade.rmse_all_hours,
                "skill_all_hours": cascade.skill_all_hours,
                "fallback_share": json.dumps(
                    {k: round(v, 4) for k, v in cascade.fallback_share.items()}
                ),
            }
        )
    all_hours = pd.DataFrame(all_hours_rows)

    # ---- tables -------------------------------------------------------------
    write_table(
        cfg,
        common_table.drop(columns=["mse_common"]),
        "lookback_common_subset",
        caption=(
            "Every arm scored on the rows the deepest arm can serve, so the hours are "
            "identical and no arm benefits from declining the hard ones. This is the "
            "conservative comparison: any advantage here is an advantage at forecasting, "
            "not at abstaining."
        ),
        float_format="%.4f",
    )
    if not dm_table.empty:
        write_table(
            cfg,
            dm_table,
            "lookback_dm_tests",
            caption=(
                "Diebold-Mariano on the common subset, each capped arm against the "
                "status quo, Holm-corrected across the family. A negative statistic "
                "favours the capped arm."
            ),
            float_format="%.4f",
        )
    if not decomp.empty:
        write_table(
            cfg,
            decomp,
            "lookback_decomposition",
            caption=(
                "Exact decomposition of the capped arm's change in mean squared error "
                "into supervision volume (A minus B, same predictors, more rows) and "
                "feature richness (B minus C, same rows, fewer predictors). The residual "
                "is zero by construction and is printed as a check on the arithmetic, "
                "not as a finding."
            ),
            float_format="%.4f",
        )
    write_table(
        cfg,
        all_hours,
        "lookback_all_hours",
        caption=(
            "Skill over the whole evaluation universe rather than each arm's own "
            "servable subset. Under the single policy an arm answers where it can and "
            "persistence answers the rest; under the cascade a shorter-reach arm answers "
            "the hours a deeper one refuses. An hour with no forecast is not an hour "
            "without error."
        ),
        float_format="%.4f",
    )

    # ---- figure -------------------------------------------------------------
    palette = setup_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 3.0))
    ax = axes[0]
    single = all_hours[all_hours["policy"] == "single"]
    for i, (model, sub) in enumerate(single.groupby("model")):
        sub = sub.sort_values("availability")
        ax.plot(
            sub["availability"],
            sub["skill_all_hours"],
            marker="o",
            ms=3.5,
            lw=1.0,
            color=palette[i % len(palette)],
            label=model,
        )
    ax.set_xlabel("Forecast availability")
    ax.set_ylabel("All-hours skill vs persistence")
    ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC", fontsize=6.4)
    panel_label(ax, "(a)")

    ax = axes[1]
    if not decomp.empty:
        labels, volume, richness = [], [], []
        for r in decomp.itertuples():
            labels.append(f"{r.model}{NL}L={r.lookback_h}")
            volume.append(r.volume_mse_delta)
            richness.append(r.richness_mse_delta)
        x = np.arange(len(labels))
        ax.bar(x - 0.19, volume, width=0.36, color=palette[0], label="supervision volume")
        ax.bar(x + 0.19, richness, width=0.36, color=palette[1], label="feature richness")
        ax.axhline(0.0, color="#666666", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.6)
        ax.set_ylabel("Change in MSE")
        ax.legend(
            frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC", fontsize=6.4
        )
    panel_label(ax, "(b)")
    fig.tight_layout(w_pad=1.1)
    written = save_figure(cfg, fig, "fig15_availability_frontier")
    log.info("wrote %s", ", ".join(p.name for p in written))
    plt.close(fig)

    # Are the hours a deep-reach model cannot serve harder or easier than the ones
    # it can? It is not rhetorical: all-hours RMSE comes out BELOW served RMSE on
    # this record, which looks like an error until you see that the reference does
    # better on the unserved hours than on the served ones. Reported so the reader
    # is not left to wonder, and computed rather than asserted.
    deep_pred = next(
        (p for k, p in bundle.items() if k.endswith(f"|{deepest_arm}") and np.isnan(p).any()),
        None,
    )
    unserved_note = {}
    if deep_pred is not None:
        served_rows = ~np.isnan(deep_pred)
        if served_rows.any() and (~served_rows).any():
            unserved_note = {
                "n_served": int(served_rows.sum()),
                "n_unserved": int((~served_rows).sum()),
                "reference_rmse_on_served": rmse(y_true[served_rows], reference[served_rows]),
                "reference_rmse_on_unserved": rmse(y_true[~served_rows], reference[~served_rows]),
                "observed_mean_on_served": float(np.mean(y_true[served_rows])),
                "observed_mean_on_unserved": float(np.mean(y_true[~served_rows])),
            }
            harder = (
                unserved_note["reference_rmse_on_unserved"]
                > unserved_note["reference_rmse_on_served"]
            )
            unserved_note["unserved_hours_are_harder"] = bool(harder)
            log.info(
                "reference RMSE on served %.4f vs unserved %.4f -- the hours the deepest "
                "arm cannot reach are %s",
                unserved_note["reference_rmse_on_served"],
                unserved_note["reference_rmse_on_unserved"],
                "harder" if harder else "EASIER, not harder",
            )

    out = results_dir / "availability_frontier.json"
    out.write_text(
        json.dumps(
            {
                "horizon_h": horizon,
                "n_universe": int(y_true.size),
                "n_common": int(common.sum()),
                "reference_rmse_universe": ref_rmse,
                "common_subset": common_table.to_dict(orient="records"),
                "dm_tests": dm_table.to_dict(orient="records"),
                "decomposition": decomp.to_dict(orient="records"),
                "all_hours": all_hours.to_dict(orient="records"),
                "unserved_hours": unserved_note,
                "comparability_note": (
                    "rmse_served is not comparable across arms: a deeper arm is scored "
                    "on fewer, better-covered hours. The common subset and the all-hours "
                    "columns are the two comparisons that are."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(NL + "=" * 78)
    print(f"COMMON SUBSET — every arm on the same {int(common.sum())} hours")
    print("=" * 78)
    print(
        common_table.drop(columns=["mse_common"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    if not dm_table.empty:
        print(NL + "Diebold-Mariano against the status quo (negative favours the cap)")
        print(dm_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not decomp.empty:
        print(NL + "Decomposition (MSE)")
        print(decomp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    # ---- the verdict, stated rather than left to be inferred ---------------
    if not dm_table.empty and not all_hours.empty:
        # Direction matters and "differs" hides it: the only significant result
        # on this record favours the STATUS QUO, so reporting a bare count would
        # read as support for the caps when it is the opposite.
        sig = dm_table[dm_table["significant"]]
        sig_for_cap = sig[sig["better"] != deepest_arm]
        sig_for_status_quo = sig[sig["better"] == deepest_arm]
        single = all_hours[all_hours["policy"] == "single"]
        best = single.loc[single.groupby("model")["skill_all_hours"].idxmax()]
        status_quo = single[single["arm"] == deepest_arm].set_index("model")["skill_all_hours"]
        gains = {
            r.model: float(r.skill_all_hours) - float(status_quo.get(r.model, np.nan))
            for r in best.itertuples()
            if r.arm != deepest_arm
        }
        print(NL + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        print(
            f"  On the common {int(common.sum())} hours, {len(sig_for_cap)} arm(s) beat the "
            f"status quo under Holm and {len(sig_for_status_quo)} lose to it."
        )
        if not len(sig_for_cap):
            print("  The shorter reach is not better at forecasting. It is better at answering.")
        for r in sig_for_status_quo.itertuples():
            print(
                f"    {r.model} at {r.arm} is significantly WORSE than the status quo "
                f"(p={r.p_holm:.4f}) -- a shorter reach costs this model real accuracy"
            )
        for model, gain in sorted(gains.items(), key=lambda kv: -kv[1]):
            row = best[best["model"] == model].iloc[0]
            print(
                f"    {model:<14} best arm {row['arm']:<4} "
                f"availability {row['availability']:.4f} "
                f"all-hours skill {row['skill_all_hours']:+.4f} "
                f"({gain:+.4f} vs status quo)"
            )

    print(NL + "All hours")
    print(
        all_hours.drop(columns=[c for c in ("fallback_share",) if c in all_hours]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
