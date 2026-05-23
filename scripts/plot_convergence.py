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
KIND_DODGE = 0.12  # tighter dodge so 4 kinds fit per iter
KIND_COLORS = {
    "frontier": MANIM_BLUE,
    "adversarial": MANIM_RED,
    "exploit": MANIM_GREEN,
    "cma": MANIM_PURPLE,
}
KIND_OFFSETS = {
    "frontier": -0.30,
    "adversarial": -0.10,
    "exploit": +0.10,
    "cma": +0.30,
}


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
    # Local-hybrid driver writes ``state_mirror.json`` (pulled from Sherlock);
    # the canonical Sherlock orchestrator writes ``state.json``. Try both so
    # the same plotting CLI works in either mode.
    for name in ("state.json", "state_mirror.json"):
        p = run_root / name
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"No state.json or state_mirror.json in {run_root}")


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

    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=MANIM_BG)
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


def plot_kind_revenue_summary(rows: list[dict], out_dir: Path) -> None:
    """Per-kind best-so-far running max real_revenue, one line per kind.

    Skips kinds with no candidates (so legacy 2-kind runs render cleanly with
    only frontier+adversarial lines).
    """
    if not rows:
        return
    iters = sorted({r["iteration"] for r in rows})
    if not iters:
        return

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=MANIM_BG)
    _style_ax(ax)

    plotted_any = False
    for kind, color in KIND_COLORS.items():
        kind_rows = [r for r in rows
                     if r.get("kind") == kind
                     and r.get("real_revenue") is not None]
        if not kind_rows:
            continue
        running_max = -np.inf
        xs: list[int] = []
        ys: list[float] = []
        for it in iters:
            vals = [r["real_revenue"] for r in kind_rows if r["iteration"] == it]
            if not vals:
                continue
            running_max = max(running_max, float(np.max(vals)))
            xs.append(it)
            ys.append(running_max)
        if not xs:
            continue
        ax.plot(xs, ys, color=color, marker="o", lw=2,
                label=kind.capitalize())
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Best real revenue so far (per kind)")
    ax.set_title("Per-kind best-so-far Intersect-true revenue")
    if iters:
        ax.set_xticks(iters)
    ax.legend(loc="lower right")
    _save(fig, out_dir / "kind_revenue_summary.png")


# -----------------------------------------------------------------------------
# Per-geology kind-attribution plots
#
# These answer "which acquisition kind discovers the best well config for each
# geology, at each iteration?" — a finer question than "which kind has the
# global max revenue?" (latter is dominated by whichever geology is most
# lucrative). See the suite of 5 plots below; each one is a different lens on
# the same per-(geology, iter) winner attribution.
# -----------------------------------------------------------------------------


def _per_cell_best_by_kind(rows: list[dict]) -> dict[tuple[int, int], dict[str, float]]:
    """Returns ``{(geo_idx, iter): {kind: best_real_revenue}}``.

    Only kinds present in ``KIND_COLORS`` are bucketed; rows with missing or
    non-finite ``real_revenue`` are silently skipped. A (geo, iter) cell with
    no finite candidates is absent from the result (not present as an empty
    dict) so callers can skip it.
    """
    out: dict[tuple[int, int], dict[str, float]] = {}
    for r in rows:
        rr = r.get("real_revenue")
        if rr is None:
            continue
        try:
            rrf = float(rr)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rrf):
            continue
        kind = r.get("kind")
        if kind not in KIND_COLORS:
            continue
        try:
            key = (int(r["geology_index"]), int(r["iteration"]))
        except (KeyError, TypeError, ValueError):
            continue
        bucket = out.setdefault(key, {})
        if rrf > bucket.get(kind, -np.inf):
            bucket[kind] = rrf
    return out


def _per_cell_winner(rows: list[dict]) -> dict[tuple[int, int], str]:
    """Returns ``{(geo, iter): winning_kind}`` based on per-cell best real revenue."""
    bests = _per_cell_best_by_kind(rows)
    return {
        cell: max(by_kind.items(), key=lambda kv: kv[1])[0]
        for cell, by_kind in bests.items()
        if by_kind
    }


def plot_cumulative_wins_per_kind(rows: list[dict], out_dir: Path) -> None:
    """Bar chart: across all (geology, iter) cells, how many did each kind win?

    The "headline" attribution plot — single bar per kind, height = cell count.
    A kind that produces the best real-revenue config in many cells across the
    run is the one earning its keep.
    """
    if not rows:
        return
    winners = _per_cell_winner(rows)
    if not winners:
        return
    counts = {k: 0 for k in KIND_COLORS}
    for w in winners.values():
        counts[w] = counts.get(w, 0) + 1

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=MANIM_BG)
    _style_ax(ax)
    kinds = list(KIND_COLORS.keys())
    heights = [counts[k] for k in kinds]
    colors = [KIND_COLORS[k] for k in kinds]
    bars = ax.bar(kinds, heights, color=colors, edgecolor=MANIM_WHITE, linewidth=0.5)
    total = sum(heights)
    for bar, h in zip(bars, heights):
        if total > 0:
            pct = 100.0 * h / total
            label = f"{h}  ({pct:.0f}%)"
        else:
            label = str(h)
        ax.text(bar.get_x() + bar.get_width() / 2, h, label,
                ha="center", va="bottom", color=MANIM_WHITE, fontsize=TICK_SIZE)
    ax.set_xlabel("Acquisition kind")
    ax.set_ylabel("# (geology, iter) cells won")
    ax.set_title(f"Cumulative wins per kind ({total} cells total = "
                 f"{len({c[0] for c in winners})} geologies × "
                 f"{len({c[1] for c in winners})} iters)")
    _save(fig, out_dir / "wins_cumulative_per_kind.png")


def plot_per_geology_winner_heatmap(rows: list[dict], out_dir: Path) -> None:
    """2D categorical heatmap: rows=geologies, cols=iters, cell color=winning kind.

    The most information-dense view of per-cell attribution. Reveals patterns
    a bar chart hides:
      * geologies where one kind wins consistently (a horizontal band of color)
      * iters where one kind suddenly takes over (a vertical band)
      * whether the winner shifts from frontier → exploit as priors accumulate
    """
    if not rows:
        return
    winners = _per_cell_winner(rows)
    if not winners:
        return
    geos = sorted({c[0] for c in winners})
    iters = sorted({c[1] for c in winners})
    kinds = list(KIND_COLORS.keys())
    kind_to_idx = {k: i for i, k in enumerate(kinds)}
    # -1 = missing cell; encoded as a dim grey via a separate background fill.
    grid = np.full((len(geos), len(iters)), -1, dtype=np.int8)
    for (g, it), winner in winners.items():
        i = geos.index(g)
        j = iters.index(it)
        grid[i, j] = kind_to_idx[winner]

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([KIND_COLORS[k] for k in kinds])
    bounds = np.arange(-0.5, len(kinds) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(
        figsize=(max(8, 0.55 * len(iters) + 3), max(5, 0.35 * len(geos) + 2)),
        facecolor=MANIM_BG,
    )
    _style_ax(ax)
    # Background fill for missing cells (no candidate of any kind in that cell).
    ax.set_facecolor("#181818")
    masked = np.ma.masked_where(grid < 0, grid)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest", origin="upper")
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels(iters)
    ax.set_yticks(range(len(geos)))
    ax.set_yticklabels([f"geo {g}" for g in geos])
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Geology")
    ax.set_title("Per-cell winner: which kind produced the best real revenue?")
    # Custom discrete colorbar with kind labels.
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(kinds)),
                        boundaries=bounds, fraction=0.04, pad=0.02)
    cbar.ax.set_yticklabels([k.capitalize() for k in kinds])
    cbar.ax.tick_params(colors=MANIM_WHITE)
    _save(fig, out_dir / "per_geology_winner_heatmap.png")


def plot_per_geology_faceted_best(rows: list[dict], out_dir: Path) -> None:
    """Small-multiples grid: one subplot per geology, 4 lines per panel (one
    per kind) showing per-kind running max real revenue within that geology.

    Use this to drill into *why* a kind won (or lost) a specific (geo, iter)
    cell on the winner heatmap. Lines that stay parallel = kinds are equally
    competitive on that geology; lines that diverge = one kind found a regime
    the others miss.
    """
    if not rows:
        return
    geos = sorted({int(r["geology_index"]) for r in rows
                   if r.get("geology_index") is not None})
    iters = sorted({int(r["iteration"]) for r in rows
                    if r.get("iteration") is not None})
    if not geos or not iters:
        return
    bests = _per_cell_best_by_kind(rows)

    cols = min(5, len(geos))
    rows_n = int(np.ceil(len(geos) / cols))
    fig, axes = plt.subplots(
        rows_n, cols,
        figsize=(cols * 3.2, rows_n * 2.6),
        facecolor=MANIM_BG, squeeze=False, sharex=True,
    )
    for gi, geo in enumerate(geos):
        ax = axes[gi // cols, gi % cols]
        _style_ax(ax)
        any_plotted = False
        for kind, color in KIND_COLORS.items():
            xs: list[int] = []
            ys: list[float] = []
            running = -np.inf
            for it in iters:
                v = bests.get((geo, it), {}).get(kind)
                if v is None:
                    continue
                running = max(running, float(v))
                xs.append(it)
                ys.append(running)
            if xs:
                ax.plot(xs, ys, color=color, lw=1.5, marker="o", ms=2.5,
                        label=kind if gi == 0 else None)
                any_plotted = True
        ax.set_title(f"geo {geo}", color=MANIM_WHITE, fontsize=TICK_SIZE)
        ax.tick_params(axis="both", labelsize=8)
        if not any_plotted:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    color=MANIM_GREY, transform=ax.transAxes)
    # Hide unused panels.
    for k in range(len(geos), rows_n * cols):
        axes[k // cols, k % cols].axis("off")
    # Single shared legend at the top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, [lbl.capitalize() for lbl in labels],
                   loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.01), facecolor="#111111")
    fig.suptitle("Per-geology best real revenue per kind (running max)",
                 color=MANIM_WHITE, y=1.04)
    fig.text(0.5, 0.02, "AL iteration", ha="center", color=MANIM_WHITE)
    fig.text(0.005, 0.5, "Best real revenue per kind", va="center",
             rotation="vertical", color=MANIM_WHITE)
    _save(fig, out_dir / "per_geology_faceted_best.png")


def plot_win_rate_over_iters(rows: list[dict], out_dir: Path) -> None:
    """Line chart: at each iter, share of geologies won by each kind.

    The temporal version of ``plot_cumulative_wins_per_kind``. Tells you
    whether the answer shifts as priors accumulate — e.g., does exploit's
    share rise across iters, or does frontier stay dominant?
    """
    if not rows:
        return
    winners = _per_cell_winner(rows)
    if not winners:
        return
    iters = sorted({c[1] for c in winners})
    kinds = list(KIND_COLORS.keys())
    # Per-iter share per kind.
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=MANIM_BG)
    _style_ax(ax)
    for kind in kinds:
        ys: list[float] = []
        for it in iters:
            cells_this_iter = [w for (g, i), w in winners.items() if i == it]
            if not cells_this_iter:
                ys.append(np.nan)
                continue
            ys.append(100.0 * sum(1 for w in cells_this_iter if w == kind) / len(cells_this_iter))
        ax.plot(iters, ys, color=KIND_COLORS[kind], marker="o", lw=2,
                label=kind.capitalize())
    ax.axhline(25.0, color=MANIM_GREY, lw=1, ls="--", alpha=0.6, zorder=0)
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Share of geologies won (%)")
    ax.set_title("Per-iter win share by kind (dashed line = 25% baseline)")
    ax.set_ylim(0, 100)
    ax.set_xticks(iters)
    ax.legend(loc="upper right")
    _save(fig, out_dir / "wins_share_over_iters.png")


def plot_per_geology_best_so_far_facets(rows: list[dict], out_dir: Path) -> None:
    """Small-multiples grid: one line per geology, showing the running max of
    real revenue across ALL kinds combined. Each point on the line is colored
    by which kind produced that step's improvement, so you can see which
    algorithm earned each geology's progress.

    Distinct from ``plot_per_geology_faceted_best`` (4 per-kind lines per
    panel showing kind-specific running maxes); this view collapses the kinds
    into one cohort-wide running max curve per geology and tags each step.
    """
    if not rows:
        return
    geos = sorted({int(r["geology_index"]) for r in rows
                   if r.get("geology_index") is not None})
    iters = sorted({int(r["iteration"]) for r in rows
                    if r.get("iteration") is not None})
    if not geos or not iters:
        return

    # Build per-geology per-iter (best_real_value, kind_responsible).
    per_geo_iter: dict[tuple[int, int], tuple[float, str]] = {}
    for r in rows:
        rv = r.get("real_revenue")
        if rv is None:
            continue
        try:
            rvf = float(rv)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rvf):
            continue
        try:
            key = (int(r["geology_index"]), int(r["iteration"]))
        except (KeyError, TypeError, ValueError):
            continue
        kind = r.get("kind", "")
        prev = per_geo_iter.get(key)
        if prev is None or rvf > prev[0]:
            per_geo_iter[key] = (rvf, kind)

    cols = min(5, len(geos))
    rows_n = int(np.ceil(len(geos) / cols))
    fig, axes = plt.subplots(
        rows_n, cols,
        figsize=(cols * 3.4, rows_n * 2.8),
        facecolor=MANIM_BG, squeeze=False, sharex=True,
    )
    for gi, geo in enumerate(geos):
        ax = axes[gi // cols, gi % cols]
        _style_ax(ax)
        running = -np.inf
        xs: list[int] = []
        ys: list[float] = []
        marker_colors: list[str] = []
        prev_running = -np.inf
        improvement_iter: int | None = None
        improvement_val: float | None = None
        for it in iters:
            cell = per_geo_iter.get((geo, it))
            if cell is None:
                continue
            val, kind = cell
            if val > running:
                running = val
                # Improvement happened — color this marker by the kind that
                # caused it. Plateau steps (val ≤ running) use grey.
                marker_colors.append(KIND_COLORS.get(kind, MANIM_GREY))
                if improvement_iter is None or running > (improvement_val or -np.inf):
                    improvement_iter = it
                    improvement_val = running
            else:
                marker_colors.append(MANIM_GREY)
            xs.append(it)
            ys.append(running)
            prev_running = running
        if xs:
            # Underlying connecting line (subtle white).
            ax.plot(xs, ys, color=MANIM_WHITE, lw=1.0, alpha=0.55, zorder=2)
            # Markers colored by improvement-kind.
            ax.scatter(xs, ys, c=marker_colors, s=42, zorder=3,
                       edgecolors="none")
            # Dotted vertical line at the lifetime-best iter.
            if improvement_iter is not None:
                ax.axvline(improvement_iter, color=MANIM_ORANGE,
                           lw=1.2, ls=":", alpha=0.7, zorder=1)
            ax.set_title(
                f"geo {geo} (best={running:.2e} @ iter {improvement_iter})",
                color=MANIM_WHITE, fontsize=TICK_SIZE,
            )
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    color=MANIM_GREY, transform=ax.transAxes)
            ax.set_title(f"geo {geo}", color=MANIM_WHITE, fontsize=TICK_SIZE)
        ax.tick_params(axis="both", labelsize=8)
    # Hide unused panels.
    for k in range(len(geos), rows_n * cols):
        axes[k // cols, k % cols].axis("off")

    # Build a single shared legend at the top showing the kind colormap.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=color,
               label=kind.capitalize(), markersize=8)
        for kind, color in KIND_COLORS.items()
    ]
    legend_handles.append(
        Line2D([], [], marker="o", linestyle="", color=MANIM_GREY,
               label="No improvement", markersize=8)
    )
    legend_handles.append(
        Line2D([], [], color=MANIM_ORANGE, lw=1.5, ls=":",
               label="Lifetime-best iter")
    )
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=len(legend_handles),
               bbox_to_anchor=(0.5, 1.02), facecolor="#111111")
    fig.suptitle("Per-geology best real revenue so far "
                 "(marker color = improvement source)",
                 color=MANIM_WHITE, y=1.06)
    fig.text(0.5, 0.01, "AL iteration", ha="center", color=MANIM_WHITE)
    fig.text(0.005, 0.5, "Best real revenue so far", va="center",
             rotation="vertical", color=MANIM_WHITE)
    _save(fig, out_dir / "per_geology_best_so_far_facets.png")


def plot_best_revenue_with_geo_updates(
    history: list[dict], rows: list[dict], out_dir: Path
) -> None:
    """Dual-axis plot: bar = how many geologies hit their lifetime best at
    this iter; lines = per-geology best real revenue so far, one line per
    geology colored by geology id (continuous viridis colormap).

    The bar at the back shows where the cohort is *discovering* new bests;
    the lines on top show *which* geologies are doing the discovering and
    at what revenue level. A geology whose line is flat across iters has
    plateaued; one whose line steps up at the same iter as a tall bar is
    contributing to that bar's count.
    """
    if not rows:
        return

    # Build per-geology running max real revenue across iters.
    geos = sorted({int(r["geology_index"]) for r in rows
                   if r.get("geology_index") is not None})
    iters = sorted({int(r["iteration"]) for r in rows
                    if r.get("iteration") is not None})
    if not geos or not iters:
        return

    # Per (geo, iter): best real revenue seen at that iter (across all kinds).
    per_geo_iter: dict[tuple[int, int], float] = {}
    for r in rows:
        rv = r.get("real_revenue")
        if rv is None:
            continue
        try:
            rvf = float(rv)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(rvf):
            continue
        try:
            key = (int(r["geology_index"]), int(r["iteration"]))
        except (KeyError, TypeError, ValueError):
            continue
        if rvf > per_geo_iter.get(key, -np.inf):
            per_geo_iter[key] = rvf

    # Per geology: running max across iters, plus EVERY iter at which the
    # per-geo running max stepped up (a new best-so-far). Each step-up is a
    # discovery event and contributes to the bar count for that iter.
    geo_lines: dict[int, tuple[list[int], list[float]]] = {}
    step_up_iters: list[int] = []
    for geo in geos:
        xs: list[int] = []
        ys: list[float] = []
        running = -np.inf
        for it in iters:
            v = per_geo_iter.get((geo, it))
            if v is None:
                continue
            if v > running:
                running = v
                step_up_iters.append(it)
            xs.append(it)
            ys.append(running)
        if xs:
            geo_lines[geo] = (xs, ys)

    bar_vals = [sum(1 for s in step_up_iters if s == it) for it in iters]

    fig, ax_bar = plt.subplots(figsize=(13, 7), facecolor=MANIM_BG)
    _style_ax(ax_bar)
    ax_bar.bar(
        iters, bar_vals,
        color=MANIM_PURPLE, alpha=0.45, edgecolor=MANIM_PURPLE,
        linewidth=1.0, label="Geos with new best-so-far at this iter",
        zorder=1,
    )
    ax_bar.set_xlabel("AL iteration")
    ax_bar.set_ylabel("# geologies with new best-so-far at this iter",
                     color=MANIM_PURPLE)
    ax_bar.tick_params(axis="y", colors=MANIM_PURPLE)
    for it, h in zip(iters, bar_vals):
        if h > 0:
            ax_bar.text(it, h + 0.05, str(h), ha="center",
                        color=MANIM_PURPLE, fontsize=TICK_SIZE)
    ax_bar.set_ylim(0, max(max(bar_vals, default=0) + 1, 2))
    ax_bar.set_xticks(iters)

    ax_line = ax_bar.twinx()
    ax_line.set_facecolor(MANIM_BG)
    # Continuous colormap keyed by geology index — readable up to ~20 geologies.
    cmap = plt.get_cmap("viridis")
    geo_min, geo_max = min(geos), max(geos)
    geo_span = max(1, geo_max - geo_min)
    for geo in geos:
        if geo not in geo_lines:
            continue
        xs, ys = geo_lines[geo]
        color = cmap((geo - geo_min) / geo_span)
        ax_line.plot(xs, ys, color=color, lw=1.6, marker="o", ms=4,
                     alpha=0.95, zorder=3)
    ax_line.set_ylabel("Best real revenue so far (per geology)",
                      color=MANIM_WHITE)
    ax_line.tick_params(axis="y", colors=MANIM_WHITE)
    ax_line.spines["right"].set_color(MANIM_WHITE)
    ax_line.spines["left"].set_color(MANIM_PURPLE)
    ax_line.grid(False)

    # Discrete-style colorbar to map color → geology index.
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=geo_min, vmax=geo_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_line, pad=0.08, fraction=0.04,
                        ticks=geos)
    cbar.ax.set_yticklabels([str(g) for g in geos])
    cbar.ax.tick_params(colors=MANIM_WHITE, labelsize=9)
    cbar.set_label("Geology index", color=MANIM_WHITE)

    ax_bar.set_title(
        f"Per-geology best real revenue vs iter "
        f"({len(geo_lines)} geologies, "
        f"bar = # geos with new best-so-far at that iter)"
    )

    # Combined legend (one entry for bar; per-geology lines explained by
    # colorbar so we don't list 15 line entries).
    h1, l1 = ax_bar.get_legend_handles_labels()
    ax_bar.legend(h1, l1, loc="upper left", facecolor="#111111")

    _save(fig, out_dir / "best_revenue_with_geo_updates.png")


def plot_per_kind_gap_distribution(rows: list[dict], out_dir: Path) -> None:
    """Violin: for each (geology, iter) cell and each kind that competed in
    that cell, plot ``cell_best_overall − kind_best_in_cell`` (≥ 0). A kind
    consistently near zero is reliable even when not winning outright.

    Tells you whether a kind that 'loses' is losing by a hair or by a mile.
    """
    if not rows:
        return
    bests = _per_cell_best_by_kind(rows)
    if not bests:
        return
    gaps: dict[str, list[float]] = {k: [] for k in KIND_COLORS}
    for cell, by_kind in bests.items():
        if not by_kind:
            continue
        cell_max = max(by_kind.values())
        for kind, val in by_kind.items():
            gaps[kind].append(float(cell_max - val))
    # Drop kinds with no samples to keep the violin layout clean.
    plotted_kinds = [k for k in KIND_COLORS if gaps[k]]
    if not plotted_kinds:
        return

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=MANIM_BG)
    _style_ax(ax)
    data = [gaps[k] for k in plotted_kinds]
    positions = list(range(1, len(plotted_kinds) + 1))
    parts = ax.violinplot(data, positions=positions, widths=0.85,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, k in zip(parts["bodies"], plotted_kinds):
        body.set_facecolor(KIND_COLORS[k])
        body.set_edgecolor(KIND_COLORS[k])
        body.set_alpha(0.55)
    # Overlay median + mean markers.
    for pos, k in zip(positions, plotted_kinds):
        vals = gaps[k]
        if not vals:
            continue
        ax.scatter([pos], [float(np.median(vals))], color=MANIM_WHITE,
                   marker="_", s=300, linewidths=2, zorder=4)
        ax.scatter([pos], [float(np.mean(vals))], color=MANIM_ORANGE,
                   marker="D", s=45, zorder=5,
                   label="Mean" if k == plotted_kinds[0] else None)
        ax.scatter([pos], [0.0], color=MANIM_GREEN, marker="*", s=60,
                   zorder=6,
                   label="Cell winner" if k == plotted_kinds[0] else None)
        # Show win count above each violin so the viewer can tell whether a
        # tight distribution means "always wins" or "rarely competes".
        win_count = sum(1 for v in vals if v == 0.0)
        ax.text(pos, max(vals) * 1.04 if max(vals) > 0 else 0.05,
                f"wins={win_count}/{len(vals)}",
                ha="center", color=MANIM_WHITE, fontsize=TICK_SIZE)

    ax.set_xticks(positions)
    ax.set_xticklabels([k.capitalize() for k in plotted_kinds])
    ax.set_ylabel("Gap to cell-best real revenue (lower = better)")
    ax.set_xlabel("Acquisition kind")
    ax.set_title("Per-cell gap distribution: how far behind the cell winner?")
    ax.legend(loc="upper right")
    _save(fig, out_dir / "per_kind_gap_distribution.png")


def plot_calibration_metrics(history: list[dict], out_dir: Path) -> None:
    """Per-iteration MAPE + signed bias, frontier vs adversarial breakdown."""
    iters = [r["iteration"] for r in history]
    mape = [r.get("batch_mape") for r in history]
    signed = [r.get("batch_signed_pct_bias") for r in history]
    front = [r.get("frontier_mape") for r in history]
    adv = [r.get("adversarial_mape") for r in history]
    exploit = [r.get("exploit_mape") for r in history]
    cma = [r.get("cma_mape") for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor=MANIM_BG)
    _style_ax(axes[0])
    _style_ax(axes[1])

    axes[0].plot(iters, [v * 100 if v is not None else np.nan for v in mape],
                 color=MANIM_BLUE, marker="o", lw=2, label="Batch MAPE")
    axes[0].plot(iters, [v * 100 if v is not None else np.nan for v in front],
                 color=MANIM_BLUE, marker="s", lw=2, label="Frontier MAPE")
    axes[0].plot(iters, [v * 100 if v is not None else np.nan for v in adv],
                 color=MANIM_RED, marker="^", lw=2, label="Adversarial MAPE")
    axes[0].plot(iters, [v * 100 if v is not None else np.nan for v in exploit],
                 color=MANIM_GREEN, marker="D", lw=2, label="Exploit MAPE")
    axes[0].plot(iters, [v * 100 if v is not None else np.nan for v in cma],
                 color=MANIM_PURPLE, marker="v", lw=2, label="CMA MAPE")
    axes[0].set_xlabel("AL iteration")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_title("Calibration: per-kind |pred − real| / |real|")
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
    """4×2 hexbin: rows={frontier,adversarial,exploit,cma} × cols={injector,producer}."""
    if not rows:
        return
    kinds = ("frontier", "adversarial", "exploit", "cma")
    types = (("injector", "Blues"), ("producer", "Oranges"))
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    extent = [min(xs), max(xs), min(ys), max(ys)]

    fig, axes = plt.subplots(4, 2, figsize=(15, 22), facecolor=MANIM_BG)
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
            ax.set_title(f"{kind} {wt} (n={len(pts)})", color=KIND_COLORS.get(kind, MANIM_WHITE))
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
# Ensemble-mode EMV plots
# -----------------------------------------------------------------------------


def _per_iter_emv_by_kind(rows: list[dict]) -> dict[int, dict[str, list[float]]]:
    """Group rows by (iteration, kind) and compute per-candidate EMV.

    Returns a nested dict {iter: {kind: [emv, ...]}}. EMV per candidate is the
    mean of real_revenue across all rows sharing the same snapshot_id within an
    iteration. Iterations without any multi-row snapshots are skipped (those
    are per-geology iterations and don't have an EMV interpretation).
    """
    by_iter_snap: dict[int, dict[str, list[dict]]] = {}
    for r in rows:
        it = int(r.get("iteration", -1))
        sid = r.get("snapshot_id")
        if it < 0 or not sid:
            continue
        by_iter_snap.setdefault(it, {}).setdefault(str(sid), []).append(r)
    out: dict[int, dict[str, list[float]]] = {}
    for it, snap_map in by_iter_snap.items():
        any_multi = any(len(rs) > 1 for rs in snap_map.values())
        if not any_multi:
            continue
        kind_to_emvs: dict[str, list[float]] = {}
        for sid, rs in snap_map.items():
            reals = [float(r.get("real_revenue", float("nan"))) for r in rs]
            reals = [v for v in reals if np.isfinite(v)]
            if not reals:
                continue
            emv = float(np.mean(reals))
            kind = str(rs[0].get("kind", "frontier"))
            kind_to_emvs.setdefault(kind, []).append(emv)
        if kind_to_emvs:
            out[it] = kind_to_emvs
    return out


def plot_emv_distribution(history: list[dict], rows: list[dict], out_dir: Path) -> None:
    """Per-iteration violin of per-candidate real EMV, split by kind."""
    grouped = _per_iter_emv_by_kind(rows)
    if not grouped:
        # No ensemble iterations yet; render a placeholder so the dashboard
        # doesn't silently miss the plot.
        fig, ax = plt.subplots(figsize=(8, 4))
        _style_ax(ax)
        ax.text(0.5, 0.5, "No ensemble-mode iterations yet", ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_dir / "emv_distribution.png")
        return

    iters_sorted = sorted(grouped.keys())
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(iters_sorted) + 4), 5))
    _style_ax(ax)
    for kind, color in KIND_COLORS.items():
        offset = KIND_OFFSETS.get(kind, 0.0)
        positions = []
        data = []
        for it in iters_sorted:
            vals = grouped[it].get(kind, [])
            if not vals:
                continue
            positions.append(it + offset)
            data.append(vals)
        if not data:
            continue
        # Scatter for visibility on small cohorts; violins when ≥4 points.
        for pos, vals in zip(positions, data):
            seed = abs(int(pos * 100)) % (2**32 - 1)
            jitter = (np.random.RandomState(seed).rand(len(vals)) - 0.5) * 0.08
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       s=18, color=color, alpha=0.7, edgecolor="none", label=None)
        violin_data = [d for d in data if len(d) >= 4]
        violin_pos = [p for p, d in zip(positions, data) if len(d) >= 4]
        if violin_data:
            parts = ax.violinplot(violin_data, positions=violin_pos, widths=0.18,
                                   showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.35)
            if "cmedians" in parts:
                parts["cmedians"].set_color(color)
        ax.plot([], [], color=color, marker="o", linestyle="None", label=kind)

    # Best-EMV-so-far overlay.
    best_emv_iters = [int(r["iteration"]) for r in history if r.get("best_emv_so_far") is not None]
    best_emv_vals = [float(r["best_emv_so_far"]) for r in history if r.get("best_emv_so_far") is not None]
    if best_emv_iters:
        ax.plot(best_emv_iters, best_emv_vals, color=MANIM_WHITE, linewidth=1.5,
                label="best EMV so far")

    ax.set_xticks(iters_sorted)
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Real EMV (mean discounted revenue across ensemble)")
    ax.set_title("Per-candidate real EMV distribution, by kind")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, out_dir / "emv_distribution.png")


def plot_best_emv_so_far(history: list[dict], out_dir: Path) -> None:
    """Single line: running best real EMV across iterations."""
    iters = [int(r["iteration"]) for r in history if r.get("best_emv_so_far") is not None]
    vals = [float(r["best_emv_so_far"]) for r in history if r.get("best_emv_so_far") is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_ax(ax)
    if not iters:
        ax.text(0.5, 0.5, "No ensemble-mode iterations yet", ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_dir / "best_emv_so_far.png")
        return
    ax.plot(iters, vals, color=MANIM_BLUE, marker="o", linewidth=2.0, markersize=6)
    # In-batch best per iter as a dimmer trace, for context on volatility.
    in_batch_iters = [int(r["iteration"]) for r in history if r.get("best_emv_in_batch") is not None]
    in_batch_vals = [float(r["best_emv_in_batch"]) for r in history if r.get("best_emv_in_batch") is not None]
    if in_batch_iters:
        ax.plot(in_batch_iters, in_batch_vals, color=MANIM_ORANGE, marker="x",
                linewidth=1.0, alpha=0.7, label="best EMV in batch")
        ax.legend(loc="lower right")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Real EMV")
    ax.set_title("Best ensemble EMV (running)")
    _save(fig, out_dir / "best_emv_so_far.png")


def _exploit_seed_rows(rows: list[dict]) -> list[dict]:
    """Collapse multi-geo rows of each exploit candidate into one row per snapshot.

    Returns dicts with keys: iteration, snapshot_id, seed_predicted_emv,
    terminal_predicted_emv, seed_real_emv, terminal_real_emv.
    """
    by_iter_snap: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        if str(r.get("kind", "")) != "exploit":
            continue
        sid = r.get("snapshot_id")
        if not sid:
            continue
        by_iter_snap.setdefault((int(r["iteration"]), str(sid)), []).append(r)

    out: list[dict] = []
    for (it, sid), rs in by_iter_snap.items():
        # All K rows of one snapshot share the same predicted/terminal EMV and
        # the same seed_source pointer.
        first = rs[0]
        seed_pred = first.get("seed_predicted_emv")
        term_pred = first.get("terminal_predicted_emv")
        seed_real = first.get("seed_real_emv")
        reals = [float(r.get("real_revenue", float("nan"))) for r in rs]
        reals = [v for v in reals if np.isfinite(v)]
        term_real = float(np.mean(reals)) if reals else None
        out.append({
            "iteration": it,
            "snapshot_id": sid,
            "seed_predicted_emv": seed_pred,
            "terminal_predicted_emv": term_pred,
            "seed_real_emv": seed_real,
            "terminal_real_emv": term_real,
            "seed_source_snapshot_id": first.get("seed_source_snapshot_id"),
        })
    return out


def _scatter_seed_vs_terminal(
    pts: list[tuple[int, float, float]],
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Helper for the two seed-vs-terminal scatters. ``pts`` is [(iter, x, y), ...]."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    _style_ax(ax)
    if not pts:
        ax.text(0.5, 0.5, "No exploit-with-seed candidates yet",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_path)
        return

    iters_arr = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    scat = ax.scatter(xs, ys, c=iters_arr, cmap="viridis", s=42, alpha=0.85,
                      edgecolor=MANIM_WHITE, linewidth=0.3)
    cbar = fig.colorbar(scat, ax=ax, label="iteration", fraction=0.04, pad=0.04)
    cbar.ax.yaxis.label.set_color(MANIM_WHITE)
    cbar.ax.tick_params(colors=MANIM_WHITE)

    lo = float(min(xs.min(), ys.min()))
    hi = float(max(xs.max(), ys.max()))
    span = hi - lo
    pad = 0.02 * span if span > 0 else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=MANIM_GREY, linestyle="--", linewidth=1.0, label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    above = int(np.sum(ys > xs))
    total = len(pts)
    frac = above / total if total else 0.0
    ax.text(0.02, 0.98,
            f"{above}/{total} ({100 * frac:.0f}%) above y=x",
            transform=ax.transAxes, va="top", ha="left", color=MANIM_WHITE,
            bbox=dict(facecolor="#111111", edgecolor=MANIM_GREY, alpha=0.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    _save(fig, out_path)


def plot_exploit_seed_vs_terminal_predicted(rows: list[dict], out_dir: Path) -> None:
    """Per-seed scatter: predicted EMV at step 0 vs at step k_safe."""
    pts: list[tuple[int, float, float]] = []
    for r in _exploit_seed_rows(rows):
        s = r["seed_predicted_emv"]
        t = r["terminal_predicted_emv"]
        if s is None or t is None or not (np.isfinite(s) and np.isfinite(t)):
            continue
        pts.append((int(r["iteration"]), float(s), float(t)))
    _scatter_seed_vs_terminal(
        pts,
        out_dir / "exploit_seed_vs_terminal_predicted.png",
        title="Exploit: predicted EMV at seed vs after Adam",
        xlabel="Seed predicted EMV (step 0)",
        ylabel="Terminal predicted EMV (step k_safe)",
    )


def plot_exploit_seed_vs_terminal_real(rows: list[dict], out_dir: Path) -> None:
    """Per-seed scatter: real EMV of the prior elite seed vs IX-evaluated terminal EMV."""
    pts: list[tuple[int, float, float]] = []
    for r in _exploit_seed_rows(rows):
        s = r["seed_real_emv"]
        t = r["terminal_real_emv"]
        if s is None or t is None or not (np.isfinite(s) and np.isfinite(t)):
            continue
        pts.append((int(r["iteration"]), float(s), float(t)))
    _scatter_seed_vs_terminal(
        pts,
        out_dir / "exploit_seed_vs_terminal_real.png",
        title="Exploit: real EMV of seed vs IX-evaluated terminal",
        xlabel="Seed real EMV (prior iter)",
        ylabel="Terminal real EMV (this iter, mean over K geos)",
    )


def plot_exploit_emv_progression(history: list[dict], out_dir: Path) -> None:
    """Per-iter exploit_best_emv vs running best_emv_so_far."""
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_ax(ax)
    iters_ex = [int(r["iteration"]) for r in history if r.get("exploit_best_emv") is not None]
    vals_ex = [float(r["exploit_best_emv"]) for r in history if r.get("exploit_best_emv") is not None]
    iters_bo = [int(r["iteration"]) for r in history if r.get("best_emv_so_far") is not None]
    vals_bo = [float(r["best_emv_so_far"]) for r in history if r.get("best_emv_so_far") is not None]
    if not iters_ex and not iters_bo:
        ax.text(0.5, 0.5, "No ensemble-mode iterations yet",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_dir / "exploit_emv_progression.png")
        return
    if iters_bo:
        ax.plot(iters_bo, vals_bo, color=MANIM_WHITE, linewidth=1.5,
                label="best EMV so far (any kind)")
    if iters_ex:
        ax.plot(iters_ex, vals_ex, color=MANIM_GREEN, marker="o", linewidth=1.5,
                label="best exploit EMV this iter")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Real EMV")
    ax.set_title("Exploit cohort EMV progression vs running best")
    ax.legend(loc="lower right")
    _save(fig, out_dir / "exploit_emv_progression.png")


def plot_exploit_per_geology_progression(history: list[dict], out_dir: Path) -> None:
    """Per-geology facets: exploit's best per-geo revenue vs running per-geo best."""
    # Collect the set of geology indices that appear in any history record.
    geo_set: set[str] = set()
    for r in history:
        pg = r.get("per_geology") or {}
        geo_set.update(pg.keys())
        ebpg = r.get("exploit_best_per_geology") or {}
        geo_set.update(ebpg.keys())
    if not geo_set:
        fig, ax = plt.subplots(figsize=(8, 4))
        _style_ax(ax)
        ax.text(0.5, 0.5, "No ensemble-mode iterations yet",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_dir / "exploit_per_geology_progression.png")
        return
    geos = sorted(geo_set, key=lambda s: int(s))

    n = len(geos)
    cols = int(np.ceil(np.sqrt(n)))
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols,
                             figsize=(3.2 * cols, 2.6 * rows_n),
                             squeeze=False)

    # Pre-compute running per-geo best (all kinds) for overlay.
    running_best: dict[str, list[tuple[int, float]]] = {g: [] for g in geos}
    cur_best: dict[str, float] = {}
    for rec in history:
        it = int(rec["iteration"])
        pg = rec.get("per_geology") or {}
        for g in geos:
            entry = pg.get(g)
            if not entry:
                continue
            mr = entry.get("max_real_revenue")
            if mr is None or not np.isfinite(mr):
                continue
            prev = cur_best.get(g)
            cur_best[g] = float(mr) if prev is None else max(prev, float(mr))
            running_best[g].append((it, cur_best[g]))

    any_exploit = False
    for ax, g in zip(axes.flat, geos):
        _style_ax(ax)
        # Running per-geo best (all kinds) — context.
        pts = running_best.get(g, [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=MANIM_WHITE, linewidth=1.2, alpha=0.7,
                    label="best (any kind)")
        # Exploit per-iter best for this geo — main signal.
        ex_xs: list[int] = []
        ex_ys: list[float] = []
        for rec in history:
            ebpg = rec.get("exploit_best_per_geology") or {}
            v = ebpg.get(g)
            if v is None or not np.isfinite(v):
                continue
            ex_xs.append(int(rec["iteration"]))
            ex_ys.append(float(v))
            any_exploit = True
        if ex_xs:
            ax.plot(ex_xs, ex_ys, color=MANIM_GREEN, marker="o",
                    linewidth=1.2, label="exploit")
        ax.set_title(f"geo {g}", fontsize=10)
        ax.tick_params(labelsize=8)
    # Hide any unused axes.
    for ax in axes.flat[len(geos):]:
        ax.set_axis_off()
    # Single legend at figure level.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Exploit per-geology progression", color=MANIM_WHITE, y=1.04)
    _save(fig, out_dir / "exploit_per_geology_progression.png")


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
    plot_kind_revenue_summary(rows, out_dir)
    # Per-geology kind-attribution suite (5 plots): cumulative wins bar,
    # winner heatmap, per-geology faceted best-so-far, win-share over iters,
    # gap distribution. Together they answer: "which kind finds the most
    # bests for each geology, and by how much does it beat the others?"
    plot_cumulative_wins_per_kind(rows, out_dir)
    plot_per_geology_winner_heatmap(rows, out_dir)
    plot_per_geology_faceted_best(rows, out_dir)
    plot_win_rate_over_iters(rows, out_dir)
    plot_per_kind_gap_distribution(rows, out_dir)
    plot_best_revenue_with_geo_updates(history, rows, out_dir)
    plot_per_geology_best_so_far_facets(rows, out_dir)
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

    # Ensemble-mode EMV plots (guarded: produce a placeholder PNG if no
    # ensemble iterations are present in this run).
    plot_emv_distribution(history, rows, out_dir)
    plot_best_emv_so_far(history, out_dir)
    plot_exploit_seed_vs_terminal_predicted(rows, out_dir)
    plot_exploit_seed_vs_terminal_real(rows, out_dir)
    plot_exploit_emv_progression(history, out_dir)
    plot_exploit_per_geology_progression(history, out_dir)

    print(f"\nDone. {len(history)} iteration(s), {len(rows)} candidate rows. Plots in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
