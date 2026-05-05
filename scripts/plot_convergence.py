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


def plot_best_revenue(history: list[dict], out_dir: Path) -> None:
    """Best Intersect-true revenue so far vs iteration. Headline metric."""
    iters = [r["iteration"] for r in history if r.get("best_real_revenue") is not None]
    revs = [r["best_real_revenue"] for r in history if r.get("best_real_revenue") is not None]
    if not iters:
        return
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=MANIM_BG)
    _style_ax(ax)
    ax.plot(iters, revs, color=MANIM_BLUE, marker="o", lw=2, label="Best so far")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Discounted revenue")
    ax.set_title("Best Intersect-true revenue across iterations")
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
    plot_best_revenue(history, out_dir)
    plot_calibration_metrics(history, out_dir)
    plot_per_geology_mape_heatmap(history, out_dir)
    plot_training_growth(history, out_dir)
    plot_wallclock_breakdown(history, out_dir)

    rows = _all_candidate_rows(args.run_root, history)
    plot_predicted_vs_real_scatter(rows, out_dir)
    plot_real_revenue_distribution(rows, out_dir)

    print(f"\nDone. {len(history)} iteration(s), {len(rows)} candidate rows. Plots in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
