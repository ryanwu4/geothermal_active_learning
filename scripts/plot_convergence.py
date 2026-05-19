#!/usr/bin/env python3
"""Walk an AL run directory and produce convergence / calibration plots.

Reads:
  <run_root>/state.json
  <run_root>/iter_NNNN/per_candidate_metrics.json   (one per completed iter)
  <run_root>/manifests/manifest_iter_NNNN.json      (selected batches)
  <run_root>/done.json                              (optional)

Writes PNGs into <run_root>/plots/. Run idempotently — safe to re-run after
each iteration to refresh the dashboard.

Style matches the existing inference scripts (Manim-flavored dark theme) so
the output blends with figures from inference/run_ensemble_active_learning.py
and analysis/cmaes_optuna/.

Usage:
    python scripts/plot_convergence.py /scratch/users/rwu4/al_runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Manim-flavored palette (mirrors inference/run_ensemble_active_learning.py).
MANIM_BG = "#000000"
MANIM_BLUE = "#58C4DD"
MANIM_ORANGE = "#FF9000"
MANIM_GREEN = "#83C167"
MANIM_RED = "#FC6255"
MANIM_PURPLE = "#9A72AC"
MANIM_WHITE = "#FFFFFF"
MANIM_GREY = "#888888"

FONT_SIZE = 14
TITLE_SIZE = 16
TICK_SIZE = 12

# Dodge offset for split-by-kind plots; ±KIND_DODGE around each integer iter.
KIND_DODGE = 0.20
KIND_COLORS = {"frontier": MANIM_BLUE, "adversarial": MANIM_RED}
KIND_OFFSETS = {"frontier": -KIND_DODGE, "adversarial": +KIND_DODGE}


def _set_style() -> None:
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.facecolor": MANIM_BG,
        "axes.facecolor": MANIM_BG,
        "axes.edgecolor": MANIM_WHITE,
        "axes.labelcolor": MANIM_WHITE,
        "xtick.color": MANIM_WHITE,
        "ytick.color": MANIM_WHITE,
        "text.color": MANIM_WHITE,
        "legend.facecolor": "#111111",
        "legend.edgecolor": MANIM_GREY,
        "axes.grid": True,
        "grid.color": "#222222",
        "grid.linestyle": "-",
    })


def _style_ax(ax) -> None:
    ax.set_facecolor(MANIM_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(MANIM_WHITE)


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=MANIM_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def _load_state(run_root: Path) -> dict:
    with open(run_root / "state.json", "r") as f:
        return json.load(f)


def _load_per_candidate(run_root: Path, iteration: int) -> dict | None:
    p = run_root / f"iter_{iteration:04d}" / "per_candidate_metrics.json"
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def _all_candidate_rows(run_root: Path, history: list[dict]) -> list[dict]:
    rows = []
    for rec in history:
        it = rec["iteration"]
        payload = _load_per_candidate(run_root, it)
        if payload is None:
            continue
        for c in payload.get("candidates", []):
            row = dict(c)
            row["iteration"] = it
            rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_best_revenue(history: list[dict], rows: list[dict], out_dir: Path) -> None:
    """Best-so-far line with dodged frontier/adversarial violins + scatter.

    Shows both the headline best-so-far metric AND the per-iteration full revenue
    distribution split by acquisition kind — so it's visible whether the batch
    is shifting upward or only outliers are.
    """
    best_iters = [r["iteration"] for r in history if r.get("best_real_revenue") is not None]
    best_revs = [r["best_real_revenue"] for r in history if r.get("best_real_revenue") is not None]
    if not best_iters and not rows:
        return

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=MANIM_BG)
    _style_ax(ax)

    rng = np.random.default_rng(0)
    iters = sorted({r["iteration"] for r in rows})

    seen_kinds: set[str] = set()
    for kind, color in KIND_COLORS.items():
        offset = KIND_OFFSETS[kind]
        for it in iters:
            vals = [r["real_revenue"] for r in rows
                    if r["iteration"] == it and r.get("kind") == kind
                    and r.get("real_revenue") is not None]
            if not vals:
                continue
            x_center = it + offset
            if len(vals) >= 3 and len(set(vals)) > 1:
                parts = ax.violinplot(vals, positions=[x_center], widths=KIND_DODGE * 1.6,
                                      showmeans=False, showmedians=False, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.35)
            jitter = rng.uniform(-KIND_DODGE * 0.35, KIND_DODGE * 0.35, size=len(vals))
            ax.scatter([x_center + j for j in jitter], vals,
                       s=22, color=color, alpha=0.7,
                       edgecolors="none",
                       label=kind.capitalize() if kind not in seen_kinds else None)
            seen_kinds.add(kind)
            ax.scatter([x_center], [float(np.mean(vals))],
                       marker="D", s=55, facecolor="none",
                       edgecolors=MANIM_ORANGE, linewidth=1.8, zorder=4)

    if best_iters:
        ax.plot(best_iters, best_revs, color=MANIM_WHITE, marker="o", lw=2,
                markersize=6, label="Best so far", zorder=5)

    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Discounted revenue")
    ax.set_title("Best Intersect-true revenue across iterations (orange ◇ = per-kind mean)")
    if iters:
        ax.set_xticks(iters)
    ax.legend(loc="lower right")
    _save(fig, out_dir / "best_revenue_so_far.png")


def plot_calibration_metrics(history: list[dict], out_dir: Path) -> None:
    """Per-iteration MAPE + signed bias, frontier vs adversarial breakdown."""
    iters = [r["iteration"] for r in history]
    mape = [r.get("batch_mape") for r in history]
    signed = [r.get("batch_signed_pct_bias") for r in history]
    front = [r.get("frontier_mape") for r in history]
    adv = [r.get("adversarial_mape") for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=MANIM_BG)
    _style_ax(axes[0])
    _style_ax(axes[1])

    axes[0].plot(iters, [v * 100 if v is not None else np.nan for v in mape],
                 color=MANIM_BLUE, marker="o", lw=2, label="Batch MAPE")
    axes[0].plot(iters, [v * 100 if v is not None else np.nan for v in front],
                 color=MANIM_GREEN, marker="s", lw=2, label="Frontier MAPE")
    axes[0].plot(iters, [v * 100 if v is not None else np.nan for v in adv],
                 color=MANIM_RED, marker="^", lw=2, label="Adversarial MAPE")
    axes[0].set_xlabel("AL iteration")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_title("Calibration: |pred − real| / |real|")
    axes[0].legend(loc="best")

    axes[1].axhline(0, color=MANIM_GREY, lw=1, ls="--")
    axes[1].plot(iters, [v * 100 if v is not None else np.nan for v in signed],
                 color=MANIM_ORANGE, marker="o", lw=2, label="Signed bias")
    axes[1].set_xlabel("AL iteration")
    axes[1].set_ylabel("Signed % bias")
    axes[1].set_title("Bias: positive = surrogate over-predicts (exploitation)")
    axes[1].legend(loc="best")

    _save(fig, out_dir / "calibration_over_iterations.png")


def plot_predicted_vs_real_scatter(rows: list[dict], out_dir: Path) -> None:
    """Predicted vs real revenue across all iterations, colored by iter."""
    if not rows:
        return
    iters = sorted({r["iteration"] for r in rows})
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=MANIM_BG)
    _style_ax(ax)

    for it in iters:
        sub = [r for r in rows if r["iteration"] == it]
        pred = np.array([r["predicted_revenue"] for r in sub])
        real = np.array([r["real_revenue"] for r in sub])
        if len(iters) > 1:
            color = cmap((it - iters[0]) / max(1, iters[-1] - iters[0]))
        else:
            color = MANIM_BLUE
        ax.scatter(real, pred, s=40, color=color, alpha=0.85,
                   edgecolors="white", linewidth=0.4,
                   label=f"iter {it}" if it in (iters[0], iters[-1]) or len(iters) <= 4 else None)

    all_pred = np.array([r["predicted_revenue"] for r in rows])
    all_real = np.array([r["real_revenue"] for r in rows])
    lo = float(min(all_pred.min(), all_real.min()))
    hi = float(max(all_pred.max(), all_real.max()))
    ax.plot([lo, hi], [lo, hi], color=MANIM_WHITE, lw=1, ls="--", label="y = x")

    ax.set_xlabel("Intersect-true revenue")
    ax.set_ylabel("Surrogate-predicted revenue")
    ax.set_title("Predicted vs real, colored by AL iteration")
    ax.legend(loc="best")
    _save(fig, out_dir / "scatter_pred_vs_real.png")


def plot_per_geology_mape_heatmap(history: list[dict], out_dir: Path) -> None:
    """Heatmap of per-geology MAPE across iterations."""
    rows = [r for r in history if r.get("per_geology")]
    if not rows:
        return
    geo_keys = sorted({k for r in rows for k in r["per_geology"].keys()},
                      key=lambda x: int(x) if x.lstrip("-").isdigit() else 0)
    iters = [r["iteration"] for r in rows]
    matrix = np.full((len(geo_keys), len(iters)), np.nan)
    for j, r in enumerate(rows):
        for i, g in enumerate(geo_keys):
            v = r["per_geology"].get(g, {}).get("mape")
            if v is not None:
                matrix[i, j] = v * 100  # to %

    fig, ax = plt.subplots(figsize=(max(8, len(iters) * 0.6), max(4, len(geo_keys) * 0.4)),
                           facecolor=MANIM_BG)
    _style_ax(ax)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels(iters)
    ax.set_yticks(range(len(geo_keys)))
    ax.set_yticklabels([f"geo {g}" for g in geo_keys])
    ax.set_xlabel("AL iteration")
    ax.set_title("Per-geology MAPE (%)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=MANIM_WHITE)
    cbar.set_label("MAPE (%)", color=MANIM_WHITE)
    _save(fig, out_dir / "per_geology_mape_heatmap.png")


def plot_real_revenue_distribution(rows: list[dict], out_dir: Path) -> None:
    """Box+strip plot of real revenue per iteration: track frontier movement."""
    if not rows:
        return
    iters = sorted({r["iteration"] for r in rows})
    data = [[r["real_revenue"] for r in rows if r["iteration"] == it] for it in iters]

    fig, ax = plt.subplots(figsize=(max(8, len(iters) * 0.6), 5), facecolor=MANIM_BG)
    _style_ax(ax)
    bp = ax.boxplot(data, positions=iters, widths=0.6, patch_artist=True,
                    showmeans=True,
                    boxprops=dict(facecolor="#1A1A1A", edgecolor=MANIM_BLUE, linewidth=1.5),
                    whiskerprops=dict(color=MANIM_BLUE, linewidth=1.5),
                    capprops=dict(color=MANIM_BLUE, linewidth=1.5),
                    medianprops=dict(color=MANIM_ORANGE, linewidth=2),
                    meanprops=dict(marker="D", markerfacecolor=MANIM_ORANGE,
                                   markeredgecolor=MANIM_ORANGE, markersize=6),
                    flierprops=dict(marker="o", markerfacecolor=MANIM_RED,
                                    markersize=4, markeredgecolor="none"))
    # Overlay strip plot for visibility on small N.
    for it, vals in zip(iters, data):
        x = np.full(len(vals), it) + np.random.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(x, vals, s=15, color=MANIM_GREEN, alpha=0.6, edgecolors="none")

    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Intersect-true revenue (per submitted candidate)")
    ax.set_title("Distribution of batch revenues across iterations")
    _save(fig, out_dir / "real_revenue_distribution.png")


def plot_topk_mean_gap(rows: list[dict], out_dir: Path) -> None:
    """Per-iter gap between top candidate and mean/median of the batch.

    Shrinking = batches concentrating on good placements. Growing = exploration
    is dragging the mean down (still diverse).
    """
    if not rows:
        return
    iters = sorted({r["iteration"] for r in rows})
    gap_mean = []
    gap_median = []
    plotted_iters = []
    for it in iters:
        vals = [r["real_revenue"] for r in rows
                if r["iteration"] == it and r.get("real_revenue") is not None]
        if len(vals) < 3:
            continue
        top = float(np.max(vals))
        gap_mean.append(top - float(np.mean(vals)))
        gap_median.append(top - float(np.median(vals)))
        plotted_iters.append(it)
    if not plotted_iters:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=MANIM_BG)
    _style_ax(ax)
    ax.plot(plotted_iters, gap_mean, color=MANIM_ORANGE, marker="o", lw=2,
            label="top1 − mean")
    ax.plot(plotted_iters, gap_median, color=MANIM_BLUE, marker="s", lw=2,
            label="top1 − median")
    ax.axhline(0, color=MANIM_GREY, lw=1, ls="--")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Revenue gap")
    ax.set_title("Top-1 vs batch center: convergence diagnostic")
    ax.legend(loc="best")
    _save(fig, out_dir / "topk_mean_gap.png")


def plot_pred_real_kde_shift(rows: list[dict], out_dir: Path, n_recent: int = 5) -> None:
    """KDE overlay of predicted vs real revenue for the most recent N iters.

    Solid = real, dashed = predicted. Color steps grey → blue with newest on top.
    Reveals distribution-level miscalibration batch MAPE collapses into a scalar.
    """
    if not rows:
        return
    try:
        from scipy.stats import gaussian_kde
    except Exception as e:
        print(f"  skipping pred_real_kde_shift — scipy unavailable: {e}")
        return

    iters = sorted({r["iteration"] for r in rows})
    recent = iters[-n_recent:]
    if not recent:
        return

    all_vals = [r[k] for r in rows
                if r["iteration"] in recent
                for k in ("real_revenue", "predicted_revenue")
                if r.get(k) is not None]
    if len(all_vals) < 2:
        return
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.05 * max(1e-9, hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 400)

    cmap = plt.get_cmap("Blues")
    n = len(recent)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=MANIM_BG)
    _style_ax(ax)
    for i, it in enumerate(recent):
        frac = 0.35 + 0.65 * (i / max(1, n - 1))
        color = cmap(frac)
        real_vals = [r["real_revenue"] for r in rows
                     if r["iteration"] == it and r.get("real_revenue") is not None]
        pred_vals = [r["predicted_revenue"] for r in rows
                     if r["iteration"] == it and r.get("predicted_revenue") is not None]
        if len(real_vals) >= 2 and len(set(real_vals)) > 1:
            ax.plot(xs, gaussian_kde(real_vals)(xs), color=color, lw=2,
                    label=f"iter {it} real")
        if len(pred_vals) >= 2 and len(set(pred_vals)) > 1:
            ax.plot(xs, gaussian_kde(pred_vals)(xs), color=color, lw=2, ls="--",
                    label=f"iter {it} pred")
    ax.set_xlabel("Discounted revenue")
    ax.set_ylabel("Density")
    ax.set_title(f"Predicted (--) vs real (—) revenue KDEs, last {n} iters")
    ax.legend(loc="best", ncol=2)
    _save(fig, out_dir / "pred_real_kde_shift.png")


# -----------------------------------------------------------------------------
# Holdout (train/val/test) metrics — read enriched per-case predictions from
# either train.py's new CSV format (y_true/y_pred columns) or
# eval_per_geology/enrich_predictions.py output.
# -----------------------------------------------------------------------------


def _load_holdout_rows_for_iter(run_root: Path, iteration: int) -> list[dict] | None:
    import csv as _csv
    plots_dir = (run_root / "models" / f"iter_{iteration:04d}"
                 / "geothermal_hetero_gnn" / "run_00" / "plots")
    if plots_dir.exists():
        any_csv = plots_dir / "test_predictions.csv"
        if any_csv.exists():
            with open(any_csv) as f:
                header = next(_csv.reader(f), [])
            if "y_true" in header and "y_pred" in header:
                geo = _maybe_resolve_geology_for_iter(run_root, iteration, plots_dir)
                rows: list[dict] = []
                for split in ("train", "val", "test"):
                    p = plots_dir / f"{split}_predictions.csv"
                    if not p.exists():
                        continue
                    with open(p) as f:
                        for r in _csv.DictReader(f):
                            cid = r["case_id"]
                            yt = float(r["y_true"])
                            yp = float(r["y_pred"])
                            rows.append({
                                "iteration": iteration,
                                "split": r["split"],
                                "case_id": cid,
                                "geology_index": geo.get(cid),
                                "y_true": yt,
                                "y_pred": yp,
                                "abs_err": abs(yp - yt),
                            })
                return rows
    enriched = run_root / "eval_per_geology" / f"iter_{iteration:04d}" / "enriched_predictions.csv"
    if enriched.exists():
        rows = []
        with open(enriched) as f:
            for r in _csv.DictReader(f):
                rows.append({
                    "iteration": iteration,
                    "split": r["split"],
                    "case_id": r["case_id"],
                    "geology_index": int(r["geology_index"]) if r.get("geology_index") not in (None, "") else None,
                    "y_true": float(r["y_true"]),
                    "y_pred": float(r["y_pred"]),
                    "abs_err": float(r["abs_err"]),
                })
        return rows
    return None


def _maybe_resolve_geology_for_iter(run_root: Path, iteration: int, plots_dir: Path) -> dict[str, int]:
    import csv as _csv
    cache_dir = run_root / "eval_per_geology" / f"iter_{iteration:04d}"
    cache_csv = cache_dir / "enriched_predictions.csv"
    if cache_csv.exists():
        out: dict[str, int] = {}
        with open(cache_csv) as f:
            for r in _csv.DictReader(f):
                if r.get("geology_index") not in (None, ""):
                    out[r["case_id"]] = int(r["geology_index"])
        if out:
            return out
    h5_path = run_root / "current_compiled.h5"
    if not h5_path.exists():
        return {}
    case_ids: list[str] = []
    for split in ("train", "val", "test"):
        p = plots_dir / f"{split}_predictions.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for r in _csv.DictReader(f):
                case_ids.append(r["case_id"])
    if not case_ids:
        return {}
    try:
        import sys
        sys.path.insert(0, "/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate")
        from geothermal.data import resolve_geology_indices  # type: ignore
        geo = resolve_geology_indices(case_ids, h5_path)
        if geo is None:
            return {}
        return {cid: int(g) for cid, g in zip(case_ids, geo)}
    except Exception as e:
        print(f"  warning: geology resolution failed for iter {iteration}: {e}")
        return {}


def _all_holdout_rows(run_root: Path, history: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for rec in history:
        it = rec["iteration"]
        per_iter = _load_holdout_rows_for_iter(run_root, it)
        if per_iter:
            rows.extend(per_iter)
    return rows


def _mape_clipped_p99(abs_err: np.ndarray, y_true: np.ndarray) -> float:
    yt = np.abs(y_true)
    if yt.size == 0:
        return float("nan")
    mask = yt > 0.01 * np.mean(yt)
    if not np.any(mask):
        return float("nan")
    raw = abs_err[mask] / np.maximum(yt[mask], 1e-12) * 100.0
    cap = np.percentile(raw, 99)
    return float(np.mean(np.clip(raw, a_min=None, a_max=cap)))


def plot_holdout_mape_over_iters(holdout_rows: list[dict], out_dir: Path) -> None:
    """Aggregate train/val/test MAPE (clipped @ p99) per iter, FULL dataset.

    Complements the acquired-only ``calibration_over_iterations``: this is the
    apples-to-apples comparison to the surrogate's benchmark MAPE.
    """
    if not holdout_rows:
        return
    iters = sorted({r["iteration"] for r in holdout_rows})
    split_colors = {"train": MANIM_BLUE, "val": MANIM_ORANGE, "test": MANIM_GREEN}

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=MANIM_BG)
    _style_ax(ax)
    for split, color in split_colors.items():
        ys = []
        xs = []
        last_n = 0
        for it in iters:
            sub = [r for r in holdout_rows
                   if r["iteration"] == it and r["split"] == split]
            if len(sub) < 5:
                continue
            ae = np.array([r["abs_err"] for r in sub])
            yt = np.array([r["y_true"] for r in sub])
            ys.append(_mape_clipped_p99(ae, yt))
            xs.append(it)
            last_n = len(sub)
        if xs:
            ax.plot(xs, ys, color=color, marker="o", lw=2,
                    label=f"{split} (n≈{last_n})")
    ax.axhline(4.03, color=MANIM_RED, ls="--", lw=1,
               label="FINDINGS benchmark geo-8 test MAPE (4.03%)")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Holdout MAPE clipped @ p99 (%)")
    ax.set_title("Holdout MAPE over iterations (FULL dataset, train.py splits)")
    ax.set_xticks(iters)
    ax.legend(loc="best")
    _save(fig, out_dir / "holdout_mape_over_iterations.png")


def plot_holdout_per_geology_heatmap(holdout_rows: list[dict], out_dir: Path) -> None:
    """Per-(geology, iter) test-split MAPE heatmap on the FULL dataset."""
    if not holdout_rows:
        return
    test_rows = [r for r in holdout_rows if r["split"] == "test"
                 and r.get("geology_index") is not None]
    if not test_rows:
        return
    iters = sorted({r["iteration"] for r in test_rows})
    geos = sorted({int(r["geology_index"]) for r in test_rows})
    matrix = np.full((len(geos), len(iters)), np.nan)
    for j, it in enumerate(iters):
        for i, g in enumerate(geos):
            sub = [r for r in test_rows
                   if r["iteration"] == it and int(r["geology_index"]) == g]
            if len(sub) < 3:
                continue
            ae = np.array([r["abs_err"] for r in sub])
            yt = np.array([r["y_true"] for r in sub])
            matrix[i, j] = _mape_clipped_p99(ae, yt)

    fig, ax = plt.subplots(
        figsize=(max(8, len(iters) * 0.7), max(4, len(geos) * 0.35)),
        facecolor=MANIM_BG)
    _style_ax(ax)
    vmax = max(15.0, float(np.nanpercentile(matrix, 95)) if np.isfinite(matrix).any() else 15.0)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest",
                   vmin=0, vmax=vmax)
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels(iters)
    ax.set_yticks(range(len(geos)))
    ax.set_yticklabels([f"geo {g}" for g in geos])
    ax.set_xlabel("AL iteration")
    ax.set_title("Per-geology TEST MAPE (full dataset, clipped @ p99)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=MANIM_WHITE)
    cbar.set_label("MAPE (%)", color=MANIM_WHITE)
    _save(fig, out_dir / "holdout_per_geology_mape_heatmap.png")


def plot_holdout_pred_vs_real_scatter(holdout_rows: list[dict], out_dir: Path) -> None:
    """Pred vs real, FULL dataset, latest iter, colored by split, geo-8 highlighted."""
    if not holdout_rows:
        return
    iters = sorted({r["iteration"] for r in holdout_rows})
    latest = iters[-1]
    sub = [r for r in holdout_rows if r["iteration"] == latest]
    if not sub:
        return
    split_colors = {"train": MANIM_BLUE, "val": MANIM_ORANGE, "test": MANIM_GREEN}

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=MANIM_BG)
    _style_ax(ax)
    MUSD = 1e6
    for split, color in split_colors.items():
        ss = [r for r in sub if r["split"] == split]
        if not ss:
            continue
        yt = np.array([r["y_true"] for r in ss]) / MUSD
        yp = np.array([r["y_pred"] for r in ss]) / MUSD
        is_g8 = np.array([r.get("geology_index") == 8 for r in ss])
        ax.scatter(yt[~is_g8], yp[~is_g8], s=18, alpha=0.45, color=color,
                   edgecolors="none",
                   label=f"{split} non-geo8 (n={int((~is_g8).sum())})")
        if is_g8.any():
            ax.scatter(yt[is_g8], yp[is_g8], s=42, alpha=0.85, color=color,
                       edgecolors=MANIM_RED, linewidth=1.4,
                       label=f"{split} geo8 (n={int(is_g8.sum())})")
    all_yt = np.array([r["y_true"] for r in sub]) / MUSD
    all_yp = np.array([r["y_pred"] for r in sub]) / MUSD
    lo = float(min(all_yt.min(), all_yp.min()))
    hi = float(max(all_yt.max(), all_yp.max()))
    ax.plot([lo, hi], [lo, hi], color=MANIM_WHITE, lw=1, ls="--", label="y = x")
    ax.set_xlabel("Actual revenue (M$)")
    ax.set_ylabel("Predicted revenue (M$)")
    ax.set_title(f"Holdout pred vs real, iter {latest} (geo-8 highlighted with red edge)")
    ax.legend(loc="upper left", fontsize=10)
    _save(fig, out_dir / "holdout_pred_vs_real_scatter.png")


# -----------------------------------------------------------------------------
# Well position plots — from acquire/iter_NNNN/snapshots_json/*.json.
# -----------------------------------------------------------------------------


def _load_well_coord_rows(run_root: Path, history: list[dict]) -> list[dict]:
    """For every selected candidate, read its snapshot JSON and emit one row per well."""
    rows: list[dict] = []
    for rec in history:
        it = rec["iteration"]
        payload = _load_per_candidate(run_root, it)
        if payload is None:
            continue
        snap_dir = run_root / "acquire" / f"iter_{it:04d}" / "snapshots_json"
        if not snap_dir.is_dir():
            continue
        for c in payload.get("candidates", []):
            snap_id = c.get("snapshot_id")
            if not snap_id:
                continue
            snap_path = snap_dir / f"{snap_id}.json"
            if not snap_path.exists():
                continue
            try:
                with open(snap_path) as f:
                    snap = json.load(f)
            except Exception:
                continue
            for w in snap.get("wells", []):
                rows.append({
                    "iteration": it,
                    "snapshot_id": snap_id,
                    "kind": c.get("kind"),
                    "geology_index": c.get("geology_index"),
                    "real_revenue": c.get("real_revenue"),
                    "well_id": w.get("well_id"),
                    "well_type": w.get("type"),
                    "x": float(w["x"]),
                    "y": float(w["y"]),
                    "z": float(w.get("z", 0.0)),
                })
    return rows


def plot_well_position_heatmaps(rows: list[dict], out_dir: Path) -> None:
    """Two-panel hexbin: injector / producer (x,y) across all submitted candidates.

    Style mirrors ``inference/run_ensemble_active_learning.py``'s ensemble_heatmaps.
    """
    if not rows:
        return
    inj = [(r["x"], r["y"]) for r in rows if r.get("well_type") == "injector"]
    prd = [(r["x"], r["y"]) for r in rows if r.get("well_type") == "producer"]
    if not inj and not prd:
        return
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    extent = [min(xs), max(xs), min(ys), max(ys)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=MANIM_BG)
    for ax, points, cmap, label, n in (
        (axes[0], inj, "Blues", "Injectors", len(inj)),
        (axes[1], prd, "Oranges", "Producers", len(prd)),
    ):
        _style_ax(ax)
        if points:
            ix, iy = zip(*points)
            hb = ax.hexbin(ix, iy, gridsize=22, cmap=cmap,
                           extent=extent, mincnt=1, alpha=0.95)
            cb = fig.colorbar(hb, ax=ax)
            cb.ax.tick_params(colors=MANIM_WHITE)
            cb.set_label("Count", color=MANIM_WHITE)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{label} — {n} placements across {len({r['iteration'] for r in rows})} iters")
    fig.suptitle("Well placement heatmap (all submitted candidates, all iters)",
                 color=MANIM_WHITE)
    _save(fig, out_dir / "well_position_heatmap.png")


def plot_well_position_heatmaps_by_kind(rows: list[dict], out_dir: Path) -> None:
    """2×2 hexbin: rows={frontier,adversarial} × cols={injector,producer}."""
    if not rows:
        return
    kinds = ("frontier", "adversarial")
    types = (("injector", "Blues"), ("producer", "Oranges"))
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    extent = [min(xs), max(xs), min(ys), max(ys)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 13), facecolor=MANIM_BG)
    for i, kind in enumerate(kinds):
        for j, (wt, cmap) in enumerate(types):
            ax = axes[i, j]
            _style_ax(ax)
            pts = [(r["x"], r["y"]) for r in rows
                   if r.get("kind") == kind and r.get("well_type") == wt]
            if pts:
                ix, iy = zip(*pts)
                hb = ax.hexbin(ix, iy, gridsize=22, cmap=cmap,
                               extent=extent, mincnt=1, alpha=0.95)
                cb = fig.colorbar(hb, ax=ax)
                cb.ax.tick_params(colors=MANIM_WHITE)
                cb.set_label("Count", color=MANIM_WHITE)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{kind} {wt} (n={len(pts)})")
    fig.suptitle("Well placement heatmap by kind × type", color=MANIM_WHITE)
    _save(fig, out_dir / "well_position_heatmap_by_kind.png")


def plot_well_position_heatmaps_over_iters(rows: list[dict], out_dir: Path,
                                            n_panels: int = 6) -> None:
    """Filmstrip of injector heatmaps at evenly-spaced iterations."""
    if not rows:
        return
    iters = sorted({r["iteration"] for r in rows})
    if len(iters) < 2:
        return
    if len(iters) <= n_panels:
        picked = iters
    else:
        idxs = np.linspace(0, len(iters) - 1, n_panels).round().astype(int)
        picked = [iters[i] for i in idxs]

    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    extent = [min(xs), max(xs), min(ys), max(ys)]

    cols = min(len(picked), 3)
    n_rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 5, n_rows * 4.5),
                             facecolor=MANIM_BG, squeeze=False)
    for k, it in enumerate(picked):
        ax = axes[k // cols, k % cols]
        _style_ax(ax)
        pts = [(r["x"], r["y"]) for r in rows
               if r["iteration"] == it and r.get("well_type") == "injector"]
        if pts:
            ix, iy = zip(*pts)
            hb = ax.hexbin(ix, iy, gridsize=18, cmap="Blues",
                           extent=extent, mincnt=1, alpha=0.95)
            fig.colorbar(hb, ax=ax).set_label("Count", color=MANIM_WHITE)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"iter {it} injectors (n={len(pts)})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    for k in range(len(picked), n_rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.suptitle("Injector placement evolution across AL iterations",
                 color=MANIM_WHITE)
    _save(fig, out_dir / "well_position_heatmap_over_iters.png")


def plot_best_well_config(run_root: Path, well_rows: list[dict], out_dir: Path,
                          z_slice: int = 30) -> None:
    """Top-1 well config across all iters, over a single-Z slice of log PermX."""
    if not well_rows:
        return
    by_snap: dict[tuple[int, str], dict] = {}
    for r in well_rows:
        sid = r["snapshot_id"]
        if r.get("real_revenue") is None:
            continue
        key = (r["iteration"], sid)
        if key not in by_snap:
            by_snap[key] = {
                "iteration": r["iteration"],
                "snapshot_id": sid,
                "kind": r["kind"],
                "real_revenue": r["real_revenue"],
                "geology_index": r["geology_index"],
                "wells": [],
            }
        by_snap[key]["wells"].append(r)
    if not by_snap:
        return
    _, best = max(by_snap.items(), key=lambda kv: kv[1]["real_revenue"])
    best_sid = best["snapshot_id"]
    wells = best["wells"]

    snap_json_path = (run_root / "acquire" / f"iter_{best['iteration']:04d}"
                      / "snapshots_json" / f"{best_sid}.json")
    geology_file = None
    geology_name = "(unknown)"
    if snap_json_path.exists():
        try:
            with open(snap_json_path) as f:
                snap = json.load(f)
            geology_file = snap.get("geology_file")
            geology_name = snap.get("geology_name", "(unknown)")
        except Exception:
            pass

    background = None
    nx = ny = None
    if geology_file and Path(geology_file).exists():
        try:
            import h5py
            with h5py.File(geology_file, "r") as gf:
                permx = gf["Input"]["PermX"]
                active = gf["Input"]["IsActive"]
                k = max(0, min(permx.shape[0] - 1, int(z_slice)))
                slab = permx[k, :, :].astype(float)
                mask = active[k, :, :] > 0
            with np.errstate(invalid="ignore"):
                slab_masked = np.where(mask & (slab > 0), slab, np.nan)
                background = np.log10(slab_masked)
            ny, nx = background.shape
            z_slice = k
        except Exception as e:
            print(f"  warning: could not load geology background: {e}")

    fig, ax = plt.subplots(figsize=(9, 8), facecolor=MANIM_BG)
    _style_ax(ax)
    if background is not None:
        im = ax.imshow(background, origin="lower", cmap="viridis", alpha=0.7,
                       extent=[0, nx, 0, ny])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"log10(PermX) at z = {z_slice}", color=MANIM_WHITE)
        cbar.ax.tick_params(colors=MANIM_WHITE)

    xs_all = [w["x"] for w in wells]
    ys_all = [w["y"] for w in wells]
    if background is None:
        ax.set_xlim(min(xs_all) - 2, max(xs_all) + 2)
        ax.set_ylim(min(ys_all) - 2, max(ys_all) + 2)
    seen = {"injector": False, "producer": False}
    for w in wells:
        is_inj = (w["well_type"] == "injector")
        ax.scatter(
            w["x"], w["y"],
            marker="^" if is_inj else "v",
            color=MANIM_BLUE if is_inj else MANIM_ORANGE,
            s=180, alpha=0.95, edgecolors=MANIM_WHITE, linewidths=1.5,
            label=("Injector" if is_inj else "Producer") if not seen[w["well_type"]] else None,
            zorder=5,
        )
        seen[w["well_type"]] = True
        ax.annotate(str(w["well_id"]), (w["x"], w["y"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color=MANIM_WHITE, zorder=6)

    ax.set_xlabel("x  (≈ j grid index)")
    ax.set_ylabel("y  (≈ i grid index)")
    ax.set_title(
        f"Best well config — iter {best['iteration']}, {best['kind']}, "
        f"geo {best['geology_index']} ({geology_name})\n"
        f"INTERSECT discounted revenue = {best['real_revenue']/1e6:.1f} M$"
    )
    ax.legend(loc="upper right")
    _save(fig, out_dir / "best_well_config.png")


def plot_training_growth(history: list[dict], out_dir: Path) -> None:
    """Twin-y: training set size vs val_loss, both over iterations."""
    iters = [r["iteration"] for r in history]
    n_train = [r.get("n_train_samples") for r in history]
    val_loss = [r.get("train_val_loss") for r in history]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=MANIM_BG)
    _style_ax(ax)
    ax.plot(iters, n_train, color=MANIM_BLUE, marker="o", lw=2, label="Train set size")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Train set size", color=MANIM_BLUE)
    ax.tick_params(axis="y", labelcolor=MANIM_BLUE)

    ax2 = ax.twinx()
    _style_ax(ax2)
    ax2.set_facecolor(MANIM_BG)
    has_loss = [v is not None for v in val_loss]
    if any(has_loss):
        ax2.plot(
            [it for it, h in zip(iters, has_loss) if h],
            [v for v, h in zip(val_loss, has_loss) if h],
            color=MANIM_ORANGE, marker="s", lw=2, label="Val loss",
        )
    ax2.set_ylabel("Val loss (target units)", color=MANIM_ORANGE)
    ax2.tick_params(axis="y", labelcolor=MANIM_ORANGE)
    ax.set_title("Training set growth and surrogate val_loss across iterations")
    _save(fig, out_dir / "training_growth.png")


def plot_wallclock_breakdown(history: list[dict], out_dir: Path) -> None:
    """Stacked bar of acquire/train/ingest minutes per iteration."""
    iters = [r["iteration"] for r in history]
    acq = [r.get("wallclock_acquire_min") or 0 for r in history]
    train = [r.get("wallclock_train_min") or 0 for r in history]
    ingest = [r.get("wallclock_ingest_min") or 0 for r in history]

    fig, ax = plt.subplots(figsize=(max(8, len(iters) * 0.6), 5), facecolor=MANIM_BG)
    _style_ax(ax)
    width = 0.7
    ax.bar(iters, acq, width, color=MANIM_GREEN, label="Acquire")
    ax.bar(iters, train, width, bottom=acq, color=MANIM_BLUE, label="Train")
    ax.bar(iters, ingest, width,
           bottom=[a + t for a, t in zip(acq, train)],
           color=MANIM_ORANGE, label="Ingest")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Wallclock (min)")
    ax.set_title("Per-phase wallclock per iteration (acquire / train / ingest)")
    ax.legend(loc="upper right")
    _save(fig, out_dir / "wallclock_breakdown.png")


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path,
                        help="AL run root (parent of state.json)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: <run_root>/plots)")
    args = parser.parse_args()

    state = _load_state(args.run_root)
    history = state.get("history", [])
    if not history:
        print("ERROR: no history yet in state.json")
        return 1

    out_dir = args.out_dir or (args.run_root / "plots")
    _set_style()

    print(f"Generating plots from {args.run_root} -> {out_dir}")
    rows = _all_candidate_rows(args.run_root, history)
    plot_best_revenue(history, rows, out_dir)
    plot_calibration_metrics(history, out_dir)
    plot_per_geology_mape_heatmap(history, out_dir)
    plot_training_growth(history, out_dir)
    plot_wallclock_breakdown(history, out_dir)
    plot_predicted_vs_real_scatter(rows, out_dir)
    plot_real_revenue_distribution(rows, out_dir)
    plot_topk_mean_gap(rows, out_dir)
    plot_pred_real_kde_shift(rows, out_dir)
    holdout_rows = _all_holdout_rows(args.run_root, history)
    plot_holdout_mape_over_iters(holdout_rows, out_dir)
    plot_holdout_per_geology_heatmap(holdout_rows, out_dir)
    plot_holdout_pred_vs_real_scatter(holdout_rows, out_dir)
    well_rows = _load_well_coord_rows(args.run_root, history)
    plot_well_position_heatmaps(well_rows, out_dir)
    plot_well_position_heatmaps_by_kind(well_rows, out_dir)
    plot_well_position_heatmaps_over_iters(well_rows, out_dir)
    plot_best_well_config(args.run_root, well_rows, out_dir)

    print(f"\nDone. {len(history)} iteration(s), {len(rows)} candidate rows. Plots in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
