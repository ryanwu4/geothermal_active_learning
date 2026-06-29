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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make ``orchestrator`` importable when this script is run directly (sys.path[0] is scripts/).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from orchestrator import emv as _emv  # noqa: E402  (strict ensemble-EMV recomputation)

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
    # No-surrogate baseline ablation kind. AL runs never emit this label, so
    # adding it here is a no-op for the AL plots; for baseline runs it lets
    # the per-kind violin / scatter / win-rate / heatmap plots find rows.
    "baseline": MANIM_ORANGE,
}
KIND_OFFSETS = {
    "frontier": -0.30,
    "adversarial": -0.10,
    "exploit": +0.10,
    "cma": +0.30,
    # Center the baseline kind — in pure-baseline runs it's the only series,
    # so the dodge offset doesn't matter; in mixed runs (none today, but
    # leaving the door open) it sits between exploit and cma.
    "baseline": 0.0,
}


# Objective label, set once by main() after detecting the run's objective. Defaults to revenue;
# flipped to "NPV" for npv-objective runs so axis labels/titles track what is actually plotted.
# (The objective-value fields in the rows/history are swapped to NPV at load time; surrogate
# revenue is preserved in *_rev fields. The predicted-real GAP is identical in NPV space because
# the per-candidate costs cancel, so accuracy/gap plots remain meaningful.)
OBJECTIVE_MODE = "revenue"
OBJ_LABEL = "discounted revenue"
OBJ_LABEL_CAP = "Discounted revenue"
OBJ_EMV = "EMV (mean discounted revenue across ensemble)"


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


def _style_ax3d(ax) -> None:
    """Apply the Manim dark theme to a 3D axes — its panes/grid aren't covered by the global
    rcParams or _style_ax, so without this the 3D panel renders light-grey-on-white and clashes
    with the rest of the (black) dashboard."""
    ax.set_facecolor(MANIM_BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color((0.0, 0.0, 0.0, 1.0))      # black panes (MANIM_BG)
        except Exception:
            pass
        try:
            axis.pane.set_edgecolor(MANIM_GREY)
        except Exception:
            pass
        try:
            axis._axinfo["grid"].update(color=(0.27, 0.27, 0.27, 0.7), linewidth=0.6)
        except Exception:
            pass
        axis.label.set_color(MANIM_WHITE)
        axis.line.set_color(MANIM_WHITE)
    ax.tick_params(colors=MANIM_WHITE)
    ax.title.set_color(MANIM_WHITE)


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


def _load_npv_context(run_root: Path) -> dict | None:
    """Read npv_context.json (written by the local driver in npv mode) — the paths/params needed
    to rebuild the deviated-well geometry for the 3D best-config rendering. None if absent."""
    p = run_root / "npv_context.json"
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None


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


def _detect_objective(run_root: Path, history: list[dict]) -> str:
    """Return "npv" if any iteration's per_candidate_metrics is tagged objective=npv, else "revenue"."""
    for rec in history:
        payload = _load_per_candidate(run_root, rec["iteration"])
        if payload and str(payload.get("objective", "revenue")) == "npv":
            return "npv"
    return "revenue"


def _apply_npv_objective(run_root: Path, history: list[dict], rows: list[dict]) -> None:
    """Swap objective-value fields to NPV for npv-objective runs (in place).

    The local drivers (via orchestrator.npv_metrics) write real_npv / predicted_npv /
    terminal_real_npv per row and best_real_npv_in_batch per iter into per_candidate_metrics.json.
    Here we move those into the field names the plots already consume so every objective /
    convergence / ensemble plot shows NPV with no per-plot edits. Surrogate revenue stays available
    in real_revenue_rev / predicted_revenue_rev. History best fields are rebuilt as the running max
    of each iter's best_real_npv_in_batch.
    """
    for r in rows:
        # Use the NPV value; if it couldn't be computed for this row (missing snapshot coords,
        # dead-rock, well-count mismatch), set None — NOT NaN. Every plot filters these fields with
        # `is not None`, which a NaN slips past (breaking gaussian_kde / asarray_chkfinite); None is
        # the value the plots already handle (baseline predicted_revenue is always None), so missing
        # rows are dropped from the NPV axis instead of mixing units or crashing the dashboard.
        r["real_revenue"] = r["real_npv"] if r.get("real_npv") is not None else None
        r["predicted_revenue"] = r["predicted_npv"] if r.get("predicted_npv") is not None else None
        # Always bind the ensemble "real EMV" to the NPV value (None if augmentation couldn't
        # compute it). Leaving the revenue-valued terminal_real_emv would silently plot
        # predicted-NPV (y) against real-REVENUE (x) on the ensemble pred-vs-real panels; binding
        # None instead makes those panels' finite-guards drop the row rather than mix units.
        r["terminal_real_emv"] = r.get("terminal_real_npv")
    run_max = None
    for rec in history:
        payload = _load_per_candidate(run_root, rec["iteration"])
        b = payload.get("best_real_npv_in_batch") if payload else None
        if b is not None and np.isfinite(b):
            run_max = b if run_max is None else max(run_max, b)
        if run_max is not None:
            rec["best_real_revenue"] = run_max
            rec["best_emv_in_batch"] = b
            rec["best_emv_so_far"] = run_max


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
    ax.set_ylabel(OBJ_LABEL_CAP)
    ax.set_title(f"Best real {OBJ_LABEL} per iter (per (snapshot, geology) rows; orange ◇ = per-kind mean)")
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
    ax.set_ylabel(f"Best real {OBJ_LABEL} so far (per kind)")
    ax.set_title(f"Per-kind best-so-far real {OBJ_LABEL} (max over (snapshot, geology) rows)")
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
    ax.set_title(f"Per-cell winner: which kind produced the best real {OBJ_LABEL}?")
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
    fig.suptitle(f"Per-geology best real {OBJ_LABEL} per kind (running max)",
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
    fig.suptitle(f"Per-geology best real {OBJ_LABEL} so far "
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
    # per-geo running max stepped up (a new best-so-far). The *first* observed
    # iter establishes the baseline, not a discovery — without this guard the
    # bar at iter 0 always equals len(geos) by construction, swamping any real
    # discovery signal in later iters.
    geo_lines: dict[int, tuple[list[int], list[float]]] = {}
    step_up_iters: list[int] = []
    for geo in geos:
        xs: list[int] = []
        ys: list[float] = []
        running: float | None = None
        for it in iters:
            v = per_geo_iter.get((geo, it))
            if v is None:
                continue
            if running is None:
                running = v  # baseline; not a step-up
            elif v > running:
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
    ax_line.set_ylabel(f"Best real {OBJ_LABEL} so far (per geology)",
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
    ax.set_ylabel(f"Gap to cell-best real {OBJ_LABEL} (lower = better)")
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


def plot_pred_real_gap(rows: list[dict], out_dir: Path) -> None:
    """Over-exploitation monitor: predicted − real revenue (M$) per AL iteration.

    NEGATIVE = surrogate UNDER-predicts the points it selected — the safe,
    conservative regime (real meets/exceeds prediction). A swing strongly
    POSITIVE = the surrogate over-predicts its own picks → over-exploitation
    (it is selecting configs IX doesn't reward). Mirrors the pred−real gap the
    cma_trajectory_long IX experiment used as the acquisition-termination cue:
    the cold/from-scratch run stayed negative the whole way, while the
    incumbent-warm-started run crossed to +52 M$ as real revenue degraded.

    The gap is computed PER CANDIDATE (ensemble EMV): for each (iteration,
    snapshot) we average pred and real over that candidate's per-geology rows,
    then aggregate over candidates. This matches the experiment's per-candidate
    definition and avoids over-weighting candidates that happened to run more
    geologies (e.g. when dead-rock drops reduce a candidate's geology count).
    """
    if not rows:
        return
    # STRICT gate: on ensemble iterations (K>1), only count a candidate whose ensemble is complete
    # (all K geologies finite); a config with any failed geology is thrown out rather than averaged
    # over survivors. Per-geology iterations (K<=1) are ungated. ``strict_emv_map`` lists the
    # surviving snapshots per iter; ``k_by_iter`` distinguishes ensemble vs per-geology iters.
    strict_emv_map = _strict_per_candidate_emv_by_iter(rows, "real_revenue")
    k_by_iter = {it: _emv.expected_k_for_run(irows) for it, irows in _rows_by_iter(rows).items()}
    # Collapse per-geology rows to one (pred_emv, real_emv) per (iter, snapshot).
    by_iter_snap: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for i, r in enumerate(rows):
        p, q = r.get("predicted_revenue"), r.get("real_revenue")
        if p is None or q is None or not (np.isfinite(p) and np.isfinite(q)):
            continue
        it = int(r["iteration"])
        raw_sid = r.get("snapshot_id")
        sid = raw_sid or f"_row{i}"  # fall back to per-row if no id
        if raw_sid and k_by_iter.get(it, 0) > 1 and str(raw_sid) not in strict_emv_map.get(it, {}):
            continue  # ensemble candidate thrown out (a geology failed)
        by_iter_snap.setdefault((it, sid), []).append((float(p), float(q)))
    if not by_iter_snap:
        return
    by_iter: dict[int, list[float]] = {}
    for (it, _sid), pairs in by_iter_snap.items():
        pred_emv = float(np.mean([p for p, _ in pairs]))
        real_emv = float(np.mean([q for _, q in pairs]))
        by_iter.setdefault(it, []).append(pred_emv - real_emv)
    iters = sorted(by_iter)
    gaps = [np.asarray(by_iter[i]) for i in iters]
    mean_g = [float(np.mean(g)) / 1e6 for g in gaps]
    med_g = [float(np.median(g)) / 1e6 for g in gaps]
    q25 = [float(np.percentile(g, 25)) / 1e6 for g in gaps]
    q75 = [float(np.percentile(g, 75)) / 1e6 for g in gaps]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=MANIM_BG)
    _style_ax(ax)
    # ylim must span the mean line too: right-skewed gaps (a few low-value
    # geologies) push the mean above Q75, and clipping it would hide the
    # headline over-exploitation signal.
    lo = min(q25 + med_g + mean_g + [0.0])
    hi = max(q75 + med_g + mean_g + [0.0])
    pad = 0.10 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    # shade the over-exploitation (pred > real) half-plane
    ax.axhspan(0.0, hi + pad, color=MANIM_RED, alpha=0.06)
    ax.axhline(0.0, color=MANIM_GREY, lw=1.5, ls="--")
    ax.fill_between(iters, q25, q75, color=MANIM_BLUE, alpha=0.18,
                    label="per-candidate IQR")
    ax.plot(iters, med_g, color=MANIM_BLUE, marker="s", lw=2,
            label="median(pred − real)")
    ax.plot(iters, mean_g, color=MANIM_ORANGE, marker="o", lw=2,
            label="mean(pred − real)")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel(f"Predicted − real {OBJ_LABEL} (M$), per candidate")
    ax.set_title("Over-exploitation monitor: pred − real per iteration\n"
                 "(< 0 = under-predict / safe; > 0 = over-predict / over-exploitation risk)")
    ax.legend(loc="best")
    _save(fig, out_dir / "pred_real_gap_over_iterations.png")


def plot_predicted_vs_real_scatter(rows: list[dict], out_dir: Path) -> None:
    """Predicted vs real revenue across all iterations, colored by iter."""
    if not rows:
        return

    def _finite_pair(r):
        p, q = r.get("predicted_revenue"), r.get("real_revenue")
        if p is None or q is None:
            return None
        try:
            pf, qf = float(p), float(q)
        except (TypeError, ValueError):
            return None
        return (pf, qf) if np.isfinite(pf) and np.isfinite(qf) else None

    iters = sorted({r["iteration"] for r in rows})
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=MANIM_BG)
    _style_ax(ax)

    all_pred: list[float] = []
    all_real: list[float] = []
    for it in iters:
        pairs = [pq for pq in (_finite_pair(r) for r in rows if r["iteration"] == it) if pq is not None]
        if not pairs:
            continue
        pred = np.array([p for p, _ in pairs])
        real = np.array([q for _, q in pairs])
        all_pred.extend(pred.tolist())
        all_real.extend(real.tolist())
        if len(iters) > 1:
            color = cmap((it - iters[0]) / max(1, iters[-1] - iters[0]))
        else:
            color = MANIM_BLUE
        ax.scatter(real, pred, s=40, color=color, alpha=0.85,
                   edgecolors="white", linewidth=0.4,
                   label=f"iter {it}" if it in (iters[0], iters[-1]) or len(iters) <= 4 else None)

    if not all_pred:
        ax.text(0.5, 0.5, "No finite (pred, real) pairs yet",
                ha="center", va="center", color=MANIM_GREY, transform=ax.transAxes)
        ax.set_axis_off()
        _save(fig, out_dir / "scatter_pred_vs_real.png")
        return
    lo = float(min(min(all_pred), min(all_real)))
    hi = float(max(max(all_pred), max(all_real)))
    ax.plot([lo, hi], [lo, hi], color=MANIM_WHITE, lw=1, ls="--", label="y = x")

    ax.set_xlabel(f"Intersect-true {OBJ_LABEL}")
    ax.set_ylabel(f"Surrogate-predicted {OBJ_LABEL}")
    ax.set_title("Predicted vs real per (snapshot, geology) row, colored by AL iteration")
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


# ---------------------------------------------------------------------------
# Panel-gating diagnostics (orchestrator.gating). Rendered every AL iteration via
# _render_local_plots, so they update in real time. Both no-op for ungated runs
# (no IterationRecord carries gate_enabled), so they cost nothing when gating is off.
# ---------------------------------------------------------------------------
_GATE_BOX = "#39ff14"   # bright lime, reads on the dark heatmap


def plot_geology_selection(history: list[dict], out_dir: Path) -> None:
    """Which geologies were simulated in INTERSECT each iteration (panel gating).

    Per-geology MAPE heatmap with a lime box around every (geology, iteration) cell that was
    actually queried this iteration — i.e. the live view of the gate self-shrinking onto the
    hard cluster. A warmup/full iteration (panel == full ensemble) boxes every geology.
    """
    from matplotlib.patches import Rectangle  # local import; only this plot needs it
    if not any(r.get("gate_enabled") for r in history):
        return  # ungated run — every geology is always queried; nothing to show
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
                matrix[i, j] = v * 100

    fig, ax = plt.subplots(figsize=(max(8, len(iters) * 0.6), max(4, len(geo_keys) * 0.4)),
                           facecolor=MANIM_BG)
    _style_ax(ax)
    # Clip the colour scale so the iter-0 geo-8 spike doesn't wash out the structure.
    finite = matrix[np.isfinite(matrix)]
    vmax = max(float(np.percentile(finite, 95)), 8.0) if finite.size else 20.0
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")

    geo_to_row = {int(g): i for i, g in enumerate(geo_keys) if g.lstrip("-").isdigit()}
    for j, r in enumerate(rows):
        panel = r.get("panel_geology_indices")
        # panel is None on a full/warmup iter of a gated run (the whole ensemble ran).
        queried = set(geo_to_row.values()) if panel is None else {
            geo_to_row[g] for g in panel if g in geo_to_row}
        for i in queried:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=_GATE_BOX, linewidth=1.6, zorder=5))

    ax.set_xticks(range(len(iters))); ax.set_xticklabels(iters)
    ax.set_yticks(range(len(geo_keys))); ax.set_yticklabels([f"geo {g}" for g in geo_keys])
    ax.set_xlabel("AL iteration")
    ax.set_title("Geologies simulated in INTERSECT (lime box = queried)")
    cbar = fig.colorbar(im, ax=ax, extend="max")
    cbar.ax.tick_params(colors=MANIM_WHITE)
    cbar.set_label("surrogate MAPE (%)", color=MANIM_WHITE)
    _save(fig, out_dir / "geology_selection_over_iters.png")


def plot_ix_query_cost(history: list[dict], out_dir: Path) -> None:
    """INTERSECT query cost per iteration: panel + completion vs the full-budget baseline.

    The real-time 'how much did the gate save' diagnostic. Stacked bars show panel IX (all
    configs × panel geologies) and completion IX (top-M × fill geologies); the dashed line is
    the full-budget cost (n_configs × K) the iteration would have spent ungated. Title carries
    the cumulative savings. No-op for ungated runs.
    """
    if not any(r.get("gate_enabled") for r in history):
        return
    recs = [r for r in history if r.get("per_geology")]
    if not recs:
        return
    K = max((len(r["per_geology"]) for r in recs), default=0)
    if K == 0:
        return
    # n_configs is constant per run; recover it from a full/warmup iter (submitted / K).
    n_configs = None
    for r in history:
        if r.get("panel_geology_indices") is None and r.get("submitted"):
            n_configs = round(r["submitted"] / K)
            break

    iters, panel_ix, compl_ix, full_base = [], [], [], []
    for r in history:
        if r.get("submitted") is None:
            continue
        iters.append(r["iteration"])
        p = r.get("ix_runs_panel")
        if p is None:                       # full/warmup iter: all IX is "panel" (full ensemble)
            panel_ix.append(int(r["submitted"])); compl_ix.append(0)
        else:
            panel_ix.append(int(p)); compl_ix.append(int(r.get("ix_runs_completion") or 0))
        nc = n_configs or round(r["submitted"] / K)
        full_base.append(int(nc * K))
    if not iters:
        return

    x = list(range(len(iters)))
    fig, ax = plt.subplots(figsize=(max(8, len(iters) * 0.7), 5), facecolor=MANIM_BG)
    _style_ax(ax)
    ax.bar(x, panel_ix, color=MANIM_GREEN, label="panel IX (all configs × panel geos)")
    ax.bar(x, compl_ix, bottom=panel_ix, color=MANIM_BLUE,
           label="completion IX (top-M × fill geos)")
    ax.plot(x, full_base, color=MANIM_GREY, marker="o", ls="--", lw=2,
            label="full-budget baseline (n_configs × K)")
    total_actual = sum(panel_ix) + sum(compl_ix)
    total_full = sum(full_base)
    saved = 100 * (1 - total_actual / total_full) if total_full else 0.0
    ax.set_xticks(x); ax.set_xticklabels(iters)
    ax.set_xlabel("AL iteration"); ax.set_ylabel("INTERSECT runs")
    ax.set_title(f"INTERSECT query cost per iteration — cumulative {saved:.0f}% fewer "
                 f"({total_actual:,} vs {total_full:,})")
    ax.legend(loc="best")
    _save(fig, out_dir / "ix_query_cost_over_iters.png")


def plot_real_revenue_distribution(rows: list[dict], out_dir: Path) -> None:
    """Box+strip plot of real revenue per iteration: track frontier movement."""
    if not rows:
        return

    def _finite_real(r):
        v = r.get("real_revenue")
        if v is None:
            return None
        try:
            vf = float(v)
        except (TypeError, ValueError):
            return None
        return vf if np.isfinite(vf) else None

    iters = sorted({r["iteration"] for r in rows})
    data: list[list[float]] = []
    plotted_iters: list[int] = []
    for it in iters:
        vals = [v for v in (_finite_real(r) for r in rows if r["iteration"] == it) if v is not None]
        if vals:
            data.append(vals)
            plotted_iters.append(it)
    if not data:
        return
    iters = plotted_iters

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
    ax.set_ylabel(f"Real {OBJ_LABEL} (per (snapshot, geology) row)")
    ax.set_title(f"Distribution of real {OBJ_LABEL} per (snapshot, geology) row, across iterations")
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
    ax.set_ylabel(f"{OBJ_LABEL_CAP} gap")
    ax.set_title(f"Top-1 vs mean of per-(snapshot, geology) real {OBJ_LABEL}, per iter")
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
                if r.get(k) is not None and np.isfinite(r[k])]
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
                     if r["iteration"] == it and r.get("real_revenue") is not None
                     and np.isfinite(r["real_revenue"])]
        pred_vals = [r["predicted_revenue"] for r in rows
                     if r["iteration"] == it and r.get("predicted_revenue") is not None
                     and np.isfinite(r["predicted_revenue"])]
        if len(real_vals) >= 2 and len(set(real_vals)) > 1:
            ax.plot(xs, gaussian_kde(real_vals)(xs), color=color, lw=2,
                    label=f"iter {it} real")
        if len(pred_vals) >= 2 and len(set(pred_vals)) > 1:
            ax.plot(xs, gaussian_kde(pred_vals)(xs), color=color, lw=2, ls="--",
                    label=f"iter {it} pred")
    ax.set_xlabel(OBJ_LABEL_CAP)
    ax.set_ylabel("Density")
    ax.set_title(f"Predicted (--) vs real (—) per-geology {OBJ_LABEL} KDEs, last {n} iters")
    ax.legend(loc="best", ncol=2)
    _save(fig, out_dir / "pred_real_kde_shift.png")


def plot_pred_real_kde_shift_ensemble(rows: list[dict], out_dir: Path,
                                       n_recent: int = 5) -> None:
    """ENSEMBLE-LEVEL KDE: one point per snapshot per iter.

    ``plot_pred_real_kde_shift`` uses per-geology rows (M_snapshots × K_geos
    points per iter); this one collapses to one point per snapshot using
    ``terminal_predicted_emv`` vs mean-over-K ``real_revenue`` (the EMV that
    selection/Adam actually optimize against). Skips with a placeholder if no
    ensemble rows are present (per-geology runs).
    """
    out_path = out_dir / "pred_real_kde_shift_ensemble.png"
    if not _is_ensemble_run(rows):
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=MANIM_BG)
        _style_ax(ax)
        ax.text(0.5, 0.5, "No ensemble-mode iterations — skipping ensemble KDE",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_path)
        return
    try:
        from scipy.stats import gaussian_kde
    except Exception as e:
        print(f"  skipping pred_real_kde_shift_ensemble — scipy unavailable: {e}")
        return

    snap_rows = _ensemble_snapshot_rows(rows)
    if not snap_rows:
        return
    iters = sorted({r["iteration"] for r in snap_rows})
    recent = iters[-n_recent:]
    if not recent:
        return

    # Build x grid from the union of finite values.
    all_vals: list[float] = []
    for r in snap_rows:
        if r["iteration"] not in recent:
            continue
        for k in ("terminal_predicted_emv", "terminal_real_emv"):
            v = r.get(k)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(vf):
                all_vals.append(vf)
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
        real_vals = [float(r["terminal_real_emv"]) for r in snap_rows
                     if r["iteration"] == it
                     and r["terminal_real_emv"] is not None
                     and np.isfinite(float(r["terminal_real_emv"]))]
        pred_vals = [float(r["terminal_predicted_emv"]) for r in snap_rows
                     if r["iteration"] == it
                     and r["terminal_predicted_emv"] is not None
                     and np.isfinite(float(r["terminal_predicted_emv"]))]
        if len(real_vals) >= 2 and len(set(real_vals)) > 1:
            ax.plot(xs, gaussian_kde(real_vals)(xs), color=color, lw=2,
                    label=f"iter {it} real (n={len(real_vals)})")
        if len(pred_vals) >= 2 and len(set(pred_vals)) > 1:
            ax.plot(xs, gaussian_kde(pred_vals)(xs), color=color, lw=2, ls="--",
                    label=f"iter {it} pred (n={len(pred_vals)})")
    ax.set_xlabel(f"Ensemble EMV (mean {OBJ_LABEL} over K geologies)")
    ax.set_ylabel("Density")
    ax.set_title(f"Predicted (--) vs real (—) ENSEMBLE EMV KDEs, last {n} iters")
    ax.legend(loc="best", ncol=2)
    _save(fig, out_path)


def plot_predicted_vs_real_scatter_ensemble(rows: list[dict], out_dir: Path) -> None:
    """ENSEMBLE-LEVEL pred-vs-real: one point per snapshot.

    ``plot_predicted_vs_real_scatter`` plots one point per (snapshot, geology);
    this collapses to one point per snapshot (``terminal_predicted_emv`` on the
    y-axis, mean-over-K ``real_revenue`` on the x-axis). Tells you whether the
    surrogate's EMV — the quantity selection/Adam actually maximize — tracks
    the simulator's average. Skips with a placeholder for per-geology runs.
    """
    out_path = out_dir / "scatter_pred_vs_real_ensemble.png"
    if not _is_ensemble_run(rows):
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=MANIM_BG)
        _style_ax(ax)
        ax.text(0.5, 0.5, "No ensemble-mode iterations — skipping ensemble scatter",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_path)
        return

    snap_rows = _ensemble_snapshot_rows(rows)
    pairs: list[tuple[int, str, float, float]] = []
    for r in snap_rows:
        if r["terminal_real_emv"] is None:
            continue
        try:
            x = float(r["terminal_real_emv"])
            y = float(r["terminal_predicted_emv"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pairs.append((int(r["iteration"]), str(r.get("kind", "?")), x, y))

    fig, ax = plt.subplots(figsize=(8, 8), facecolor=MANIM_BG)
    _style_ax(ax)
    if not pairs:
        ax.text(0.5, 0.5, "No finite ensemble snapshots with real EMV yet",
                ha="center", va="center", color=MANIM_GREY,
                transform=ax.transAxes)
        ax.set_axis_off()
        _save(fig, out_path)
        return

    iters = sorted({p[0] for p in pairs})
    cmap = plt.get_cmap("plasma")
    # Distinguish exploit vs frontier (vs cma/adversarial) by marker shape so
    # iter-color stays available for the colormap.
    kind_marker = {"exploit": "o", "frontier": "^", "cma": "D", "adversarial": "s"}
    seen_iter_labels: set[int] = set()
    seen_kind_labels: set[str] = set()
    for it, kind, x, y in pairs:
        if len(iters) > 1:
            color = cmap((it - iters[0]) / max(1, iters[-1] - iters[0]))
        else:
            color = MANIM_BLUE
        marker = kind_marker.get(kind, "o")
        # One legend entry per iter (color) and per kind (marker).
        iter_label = f"iter {it}" if it not in seen_iter_labels and (
            it in (iters[0], iters[-1]) or len(iters) <= 4) else None
        kind_label = (f"kind: {kind}" if kind not in seen_kind_labels else None)
        if iter_label:
            seen_iter_labels.add(it)
        if kind_label:
            seen_kind_labels.add(kind)
        ax.scatter([x], [y], s=55, color=color, alpha=0.85, marker=marker,
                   edgecolors="white", linewidth=0.4,
                   label=iter_label or kind_label)

    xs = [p[2] for p in pairs]
    ys = [p[3] for p in pairs]
    lo = float(min(min(xs), min(ys)))
    hi = float(max(max(xs), max(ys)))
    ax.plot([lo, hi], [lo, hi], color=MANIM_WHITE, lw=1, ls="--", label="y = x")

    ax.set_xlabel("Real EMV (IX mean over K geologies)")
    ax.set_ylabel("Predicted EMV (surrogate terminal)")
    ax.set_title(f"Ensemble pred vs real EMV — {len(pairs)} snapshots over {len(iters)} iters")
    handles, labels = ax.get_legend_handles_labels()
    # Drop duplicate labels (matplotlib doesn't dedupe by default).
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=9)
    _save(fig, out_path)


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


# Revenue values in this project are denominated in USD; a "real" candidate's
# discounted revenue is at minimum tens of millions. A floor of $1M for the
# MAPE denominator excludes degenerate near-zero ground-truths (which would
# otherwise dominate the average) without filtering out genuinely small valid
# revenues. Fixed (not data-dependent) so the metric is comparable across iters.
_MAPE_DENOM_FLOOR_USD = 1.0e6


def _mape_clipped_p99(abs_err: np.ndarray, y_true: np.ndarray) -> float:
    """Clipped MAPE in percent.

    - Denominator floor: $1M (constant), so each iter's MAPE is a comparable
      number rather than a function of that iter's distribution.
    - Cap: 100% per-row (cap *the ratio*, not the post-mean number). A bad
      single row contributes at most 100% before averaging. We deliberately
      avoid a percentile-of-the-current-distribution cap because that is
      self-referential (for small N the p99 ≈ max, so no clipping happens).
    """
    yt = np.abs(np.asarray(y_true, dtype=np.float64))
    ae = np.asarray(abs_err, dtype=np.float64)
    if yt.size == 0:
        return float("nan")
    denom = np.maximum(yt, _MAPE_DENOM_FLOOR_USD)
    raw = ae / denom * 100.0
    return float(np.mean(np.minimum(raw, 100.0)))


def plot_holdout_mape_over_iters(holdout_rows: list[dict], out_dir: Path) -> None:
    """Aggregate train/val/test MAPE (per-row capped at 100%) per iter, FULL dataset.

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
        ns: list[int] = []
        for it in iters:
            sub = [r for r in holdout_rows
                   if r["iteration"] == it and r["split"] == split]
            if len(sub) < 5:
                continue
            ae = np.array([r["abs_err"] for r in sub])
            yt = np.array([r["y_true"] for r in sub])
            ys.append(_mape_clipped_p99(ae, yt))
            xs.append(it)
            ns.append(len(sub))
        if xs:
            # Train set grows across iters, so a single "n≈X" misleads.
            # Show the range when it varies, the exact n when stable.
            n_lo, n_hi = min(ns), max(ns)
            n_str = f"n={n_lo}" if n_lo == n_hi else f"n={n_lo}–{n_hi}"
            ax.plot(xs, ys, color=color, marker="o", lw=2,
                    label=f"{split} ({n_str})")
    ax.axhline(4.03, color=MANIM_RED, ls="--", lw=1,
               label="FINDINGS benchmark geo-8 test MAPE (4.03%)")
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Holdout MAPE (per-row capped at 100%) (%)")
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
    ax.set_title("Per-geology TEST MAPE (full dataset, per-row capped at 100%)")
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
    ax.set_title(f"Holdout pred vs real per test case, iter {latest} (geo-8 highlighted with red edge)")
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


def _render_deviated_3d(ax3, wells: list[dict], npv_ctx: dict) -> None:
    """Draw the 3-segment deviated well shape (1 km lead + diagonal to reservoir top + vertical
    through reservoir) for ``wells`` into a 3D axes, rebuilding the geometry from npv_context
    (cube + one geology's reservoir top + facilities). Mirrors the analysis render prototype."""
    import sys as _sys
    sr = npv_ctx.get("surrogate_repo")
    if sr and sr not in _sys.path:
        _sys.path.insert(0, sr)
    import h5py
    from geothermal.well_geometry import (  # type: ignore
        load_geo_coord_cube, reservoir_top_k_map, facilities_surface_xy, angled_well_segments,
    )
    cube = load_geo_coord_cube(npv_ctx["geo_cube_path"])
    with h5py.File(npv_ctx["reservoir_geology_h5"], "r") as h:
        poro = h["Input/Porosity"][:]
    ksurf = int(npv_ctx.get("ksurf", 2))
    lead = float(npv_ctx.get("vertical_lead_m", 1000.0))
    rtop = reservoir_top_k_map(poro, poro_thresh=float(npv_ctx.get("poro_thresh", 0.01)))
    fac_surf = facilities_surface_xy(cube, npv_ctx["facilities"], ksurf=ksurf)
    _style_ax3d(ax3)  # match the dashboard's black Manim theme (panes/grid/ticks/labels)
    coords = np.array([[w["x"], w["y"], w["z"]] for w in wells], dtype=float)
    is_inj = [w.get("well_type") == "injector" for w in wells]
    segs = angled_well_segments(coords, cube=cube, fac_surf_xy=fac_surf,
                                reservoir_top_k_map=rtop, vertical_lead_m=lead, ksurf=ksurf)
    zmax = lead
    for fx, fy in fac_surf:  # facility vertical leads, drawn once each
        ax3.plot([fx, fx], [fy, fy], [0.0, lead], color=MANIM_GREY, lw=3, zorder=4)
        # White-filled square so the facility is visible on the black pane (a black marker
        # would vanish); mirrors the white-edged facility marker in the 2D map.
        ax3.scatter([fx], [fy], [0.0], marker="s", s=70, color=MANIM_WHITE,
                    edgecolor=MANIM_WHITE, zorder=6)
    seen = {"injector": False, "producer": False}
    for s, inj in zip(segs, is_inj):
        c = MANIM_BLUE if inj else MANIM_ORANGE
        fx, fy = s["fac_xy"]
        rx, ry, rt = s["reservoir_top"]
        bx, by, bt = s["well_bottom"]
        lbl = ("Injector" if inj else "Producer")
        key = "injector" if inj else "producer"
        ax3.plot([fx, rx], [fy, ry], [lead, rt], color=c, lw=1.8, zorder=5,
                 label=(lbl if not seen[key] else None))
        seen[key] = True
        ax3.plot([rx, bx], [ry, by], [rt, bt], color=MANIM_PURPLE, lw=2.8, zorder=6)
        ax3.scatter([bx], [by], [bt], marker="^" if inj else "v", color=c, s=45,
                    edgecolor=MANIM_WHITE, linewidths=0.5, zorder=7)
        zmax = max(zmax, rt, bt)
    ax3.set_xlabel("X east (m)")
    ax3.set_ylabel("Y north (m)")
    ax3.set_zlabel("TVD (m)")
    ax3.set_zlim(zmax + 150, -100)  # depth increases downward
    try:
        ax3.set_box_aspect((4, 4, 3))
    except Exception:
        pass
    ax3.view_init(elev=18, azim=-60)
    ax3.set_title("Deviated well shape — 1 km lead (grey) + diagonal to reservoir top\n"
                  "+ vertical through reservoir (purple)")


def plot_best_well_config(run_root: Path, well_rows: list[dict], out_dir: Path,
                          rows: list[dict] | None = None, npv_ctx: dict | None = None,
                          z_slice: int = 30) -> None:
    """Top-1 well config across all iters, over a single-Z slice of log PermX. In npv mode (npv_ctx
    set) the "best" is the ensemble-mean-NPV winner and a 3D deviated-well shape is added."""
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
    # Best selection: ensemble-mean NPV when available (npv mode), else IX revenue.
    npv_by_snap: dict[tuple[int, str], list[float]] = {}
    for r in (rows or []):
        v = r.get("real_npv")
        if v is None or not np.isfinite(float(v)):
            continue
        npv_by_snap.setdefault((int(r["iteration"]), str(r["snapshot_id"])), []).append(float(v))
    best = None
    score_label = None
    if npv_by_snap:
        cands = [(k, float(np.mean(v))) for k, v in npv_by_snap.items() if k in by_snap]
        if cands:
            best_key, best_npv = max(cands, key=lambda kv: kv[1])
            best = by_snap[best_key]
            score_label = f"ensemble-mean NPV = {best_npv / 1e6:.1f} M$"
    if best is None:
        _, best = max(by_snap.items(), key=lambda kv: kv[1]["real_revenue"])
        score_label = f"INTERSECT discounted revenue = {best['real_revenue'] / 1e6:.1f} M$"
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
                slab = permx[k, :, :].astype(float)   # (axis1 = Julia j = well x, axis2 = Julia i = well y)
                mask = active[k, :, :] > 0
            with np.errstate(invalid="ignore"):
                slab_masked = np.where(mask & (slab > 0), slab, np.nan)
                # Transpose to (y, x) display frame so a well scattered at (x, y) sits on
                # PermX[k, x, y]. Without .T the background is BOTH transposed and stretched
                # (the two horizontal axes differ in size, e.g. 70 vs 76). Mirrors the
                # reference impl in inference/run_ensemble_active_learning.py.
                background = np.log10(slab_masked).T
            nx, ny = slab.shape   # nx = axis1 (well x), ny = axis2 (well y)
            z_slice = k
        except Exception as e:
            print(f"  warning: could not load geology background: {e}")

    # Two-panel (map + 3D deviated shape) in npv mode; single map otherwise.
    ax3 = None
    if npv_ctx is not None:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
        fig = plt.figure(figsize=(19, 8), facecolor=MANIM_BG)
        ax = fig.add_subplot(1, 2, 1)
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(9, 8), facecolor=MANIM_BG)
    _style_ax(ax)
    if background is not None:
        im = ax.imshow(background, origin="lower", cmap="viridis", alpha=0.7,
                       extent=[0, nx, 0, ny])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"log10(PermX) at z = {z_slice}", color=MANIM_WHITE)
        cbar.ax.tick_params(colors=MANIM_WHITE)

    # Facilities (from npv_ctx) in the same grid convention as the wells: a facility (i, j)
    # plots at x = j-1, y = i-1 (wells use x = j_idx-1, y = i_idx-1).
    fac_grid = [(int(fj) - 1, int(fi) - 1) for (fi, fj) in (npv_ctx or {}).get("facilities", [])]

    xs_all = [w["x"] for w in wells] + [fx for fx, _ in fac_grid]
    ys_all = [w["y"] for w in wells] + [fy for _, fy in fac_grid]
    if background is None and xs_all:
        ax.set_xlim(min(xs_all) - 2, max(xs_all) + 2)
        ax.set_ylim(min(ys_all) - 2, max(ys_all) + 2)
    seen_fac = False
    for (fx, fy), (fi, fj) in zip(fac_grid, (npv_ctx or {}).get("facilities", [])):
        ax.scatter(fx, fy, marker="s", s=170, color="#000000", edgecolors=MANIM_WHITE,
                   linewidths=1.6, zorder=7, label=("Facility" if not seen_fac else None))
        seen_fac = True
        ax.annotate(f"facility ({int(fi)},{int(fj)})", (fx, fy),
                    xytext=(6, 6), textcoords="offset points", fontsize=9,
                    color=MANIM_WHITE, zorder=8)
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
        f"geo {best['geology_index']} ({geology_name})\n{score_label}"
    )
    ax.legend(loc="upper right")
    if ax3 is not None:
        try:
            _render_deviated_3d(ax3, wells, npv_ctx)
            ax3.legend(loc="upper right", fontsize=9)
        except Exception as e:  # never let the 3D panel kill the (working) 2D map
            ax3.set_axis_off()
            ax3.text2D(0.5, 0.5, f"3D deviated render skipped:\n{type(e).__name__}: {e}",
                       ha="center", va="center", color=MANIM_GREY, transform=ax3.transAxes)
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
# Run-health: simulation failures
# -----------------------------------------------------------------------------


def _is_finite_num(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and np.isfinite(x)


def _failure_stats_per_iter(run_root: Path, history: list[dict]) -> list[dict]:
    """Per-iteration IX-failure stats, read FRESH from each per_candidate_metrics.json.

    Deliberately re-reads the file (rather than using the shared ``rows``) so the count keys off the
    raw ``real_revenue`` sim output: ``_apply_npv_objective`` overwrites ``rows[*]["real_revenue"]``
    with NPV (None when the proxy-NPV couldn't be built even though the sim succeeded), which would
    misclassify good sims as failures. The on-disk file always keeps the true revenue.

    A "failed individual run" = a submitted (config, geology) IX task that did NOT yield a finite
    real_revenue — whether it produced no output at all or produced an unusable one. We split that
    into ``no_output`` (submitted minus completed=files-produced) and ``bad_output`` (produced but
    non-finite revenue) using the file's ``n_completed``.

    A "config with a failing geology" = a snapshot whose finite-revenue geology count is below the
    ensemble size K (= distinct geology_index this iter). Configs that lost *every* geology leave no
    rows in the surrogate-path file, so we recover their count from submitted/K and fold them in.
    """
    stats: list[dict] = []
    for rec in history:
        it = rec["iteration"]
        payload = _load_per_candidate(run_root, it)
        if payload is None:
            continue
        cands = payload.get("candidates", []) or []
        submitted = payload.get("n_submitted")
        if submitted is None:  # fall back to state's per-iter count
            submitted = rec.get("submitted")
        completed = payload.get("n_completed")
        if completed is None:
            completed = rec.get("completed")

        geo_ids = {c.get("geology_index") for c in cands if c.get("geology_index") is not None}
        K = len(geo_ids)
        by_snap: dict[str, int] = {}          # snapshot_id -> finite-revenue geology count
        for c in cands:
            sid = c.get("snapshot_id")
            if not sid:
                continue
            by_snap.setdefault(sid, 0)
            if _is_finite_num(c.get("real_revenue")):
                by_snap[sid] += 1
        n_ok = sum(by_snap.values())

        # Configs present in the file plus any that vanished entirely (no rows).
        configs_present = len(by_snap)
        expected_configs = round(submitted / K) if (submitted and K) else configs_present
        missing_configs = max(0, expected_configs - configs_present)
        configs_total = max(configs_present, expected_configs)
        configs_with_failure = sum(1 for n in by_snap.values() if n < K) + missing_configs

        sub = int(submitted) if submitted else n_ok
        comp = int(completed) if completed is not None else n_ok
        failed_total = max(0, sub - n_ok)
        no_output = max(0, sub - comp)
        bad_output = max(0, failed_total - no_output)  # produced a file but revenue unusable
        stats.append({
            "iteration": it,
            "submitted": sub,
            "n_ok": n_ok,
            "no_output": no_output,
            "bad_output": bad_output,
            "failed_total": failed_total,
            "failure_rate": (100.0 * failed_total / sub) if sub else 0.0,
            "K": K,
            "configs_total": configs_total,
            "configs_with_failure": configs_with_failure,
            "configs_clean": max(0, configs_total - configs_with_failure),
            "frac_configs_failed": (100.0 * configs_with_failure / configs_total) if configs_total else 0.0,
        })
    return stats


def plot_run_failure_diagnostics(run_root: Path, history: list[dict], out_dir: Path) -> None:
    """Two-panel run-health diagnostic over iterations:

    (L) individual IX runs split into succeeded / produced-but-unusable / no-output, with the
        per-iteration failure rate overlaid.
    (R) well configs split into clean vs has-≥1-failing-geology, with the affected-fraction overlaid.
    """
    stats = _failure_stats_per_iter(run_root, history)
    if not stats:
        return
    iters = [s["iteration"] for s in stats]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor=MANIM_BG)
    width = 0.7

    # ---- Left: individual run failures ----
    ax = axes[0]
    _style_ax(ax)
    ok = [s["n_ok"] for s in stats]
    bad = [s["bad_output"] for s in stats]
    no_out = [s["no_output"] for s in stats]
    ax.bar(iters, ok, width, color=MANIM_GREEN, label="Succeeded (finite revenue)")
    ax.bar(iters, bad, width, bottom=ok, color=MANIM_ORANGE,
           label="Produced but unusable")
    ax.bar(iters, no_out, width, bottom=[o + b for o, b in zip(ok, bad)],
           color=MANIM_RED, label="No output")
    for s in stats:  # annotate total failed above the stack
        if s["failed_total"] > 0:
            ax.text(s["iteration"], s["submitted"], f" {s['failed_total']}",
                    ha="center", va="bottom", color=MANIM_WHITE, fontsize=TICK_SIZE)
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Individual IX runs")
    ax.set_xticks(iters)
    ax.set_xlim(min(iters) - 0.7, max(iters) + 0.7)  # margin so a 1-iter run isn't a full-width block
    ax.set_title("Failed individual IX runs per iteration")

    ax_r = ax.twinx()
    _style_ax(ax_r)
    ax_r.set_facecolor(MANIM_BG)
    rates = [s["failure_rate"] for s in stats]
    ax_r.plot(iters, rates, color=MANIM_WHITE, marker="o", lw=2, label="Failure rate")
    ax_r.set_ylabel("Failure rate (%)", color=MANIM_WHITE)
    ax_r.set_ylim(0, max(5.0, min(100.0, max(rates) * 1.4)) if rates else 5.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.9)

    # ---- Right: configs with any failing geology ----
    ax2 = axes[1]
    _style_ax(ax2)
    clean = [s["configs_clean"] for s in stats]
    cfail = [s["configs_with_failure"] for s in stats]
    ax2.bar(iters, clean, width, color=MANIM_GREEN, label="All geologies ok")
    ax2.bar(iters, cfail, width, bottom=clean, color=MANIM_RED,
            label="≥1 failing geology")
    for s in stats:
        if s["configs_with_failure"] > 0:
            ax2.text(s["iteration"], s["configs_total"],
                     f" {s['configs_with_failure']}/{s['configs_total']}",
                     ha="center", va="bottom", color=MANIM_WHITE, fontsize=TICK_SIZE)
    ax2.set_xlabel("AL iteration")
    ax2.set_ylabel("Well configs (snapshots)")
    ax2.set_xticks(iters)
    ax2.set_xlim(min(iters) - 0.7, max(iters) + 0.7)
    ax2.set_title("Well configs with ≥1 failing geology per iteration")

    ax2_r = ax2.twinx()
    _style_ax(ax2_r)
    ax2_r.set_facecolor(MANIM_BG)
    fracs = [s["frac_configs_failed"] for s in stats]
    ax2_r.plot(iters, fracs, color=MANIM_WHITE, marker="o", lw=2, label="% configs affected")
    ax2_r.set_ylabel("Configs affected (%)", color=MANIM_WHITE)
    ax2_r.set_ylim(0, max(5.0, min(100.0, max(fracs) * 1.4)) if fracs else 5.0)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.9)

    fig.suptitle("Run health: simulation failures", color=MANIM_WHITE, fontsize=TITLE_SIZE)
    _save(fig, out_dir / "run_failure_diagnostics.png")


# -----------------------------------------------------------------------------
# Ensemble-mode EMV plots
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Strict ensemble-EMV recomputation (plot time)
#
# These recompute EMV from the per-geology ``real_revenue`` rows under the STRICT rule
# (orchestrator.emv): a config's EMV is defined only if ALL K geologies in its ensemble produced a
# finite value; any failed/missing geology throws out the whole config. They deliberately do NOT
# trust the stored ``terminal_real_emv`` or the state-history ``best_emv_*`` (those were written with
# the older, biased survivor-mean rule), so re-running this script regenerates strict figures from
# the existing per_candidate_metrics.json WITHOUT mutating any run artifact. In NPV-objective runs
# ``real_revenue`` has already been rebound to the NPV value by ``_apply_npv_objective``, so the same
# helpers yield strict NPV EMV with no NPV-specific branching.
# -----------------------------------------------------------------------------


def _rows_by_iter(rows: list[dict]) -> dict[int, list[dict]]:
    by_iter: dict[int, list[dict]] = {}
    for r in rows:
        it = r.get("iteration")
        if it is None:
            continue
        by_iter.setdefault(int(it), []).append(r)
    return by_iter


def _strict_per_candidate_emv_by_iter(
    rows: list[dict], value_key: str = "real_revenue"
) -> dict[int, dict[str, float]]:
    """{iter: {snapshot_id: strict_emv}} computed per iteration.

    K (the ensemble size) is inferred PER ITERATION so per-geology seed iters (K=1) and full-ensemble
    iters (K=15) are each judged against their own size. Snapshots that fail the strict check
    (any failed/missing geology) are omitted from the inner dict.
    """
    out: dict[int, dict[str, float]] = {}
    for it, irows in _rows_by_iter(rows).items():
        k = _emv.expected_k_for_run(irows)
        if k <= 0:
            out[it] = {}
            continue
        by_snap: dict[str, list[dict]] = {}
        for r in irows:
            sid = r.get("snapshot_id")
            if not sid:
                continue
            by_snap.setdefault(str(sid), []).append(r)
        emv_map: dict[str, float] = {}
        for sid, srows in by_snap.items():
            v = _emv.strict_per_snapshot_emv(srows, value_key, expected_k=k)
            if v is not None:
                emv_map[sid] = float(v)
        out[it] = emv_map
    return out


def _strict_best_emv_trajectory(
    rows: list[dict], value_key: str = "real_revenue"
) -> tuple[list[int], list[float | None], list[float | None]]:
    """(iters, best_in_batch, best_so_far) from strict per-candidate EMVs, ensemble iters only.

    ``best_in_batch[i]`` is the max strict EMV among iter i's surviving snapshots, or ``None`` if
    every config that iteration was thrown out. ``best_so_far`` carries the running max forward
    across ``None`` gaps (a wiped-out iteration holds the line, it never drops to 0). Only ensemble
    iterations (K > 1) are emitted, so per-geology seed iters don't masquerade as EMV points.
    """
    per_iter = _strict_per_candidate_emv_by_iter(rows, value_key)
    by_iter = _rows_by_iter(rows)
    iters_out: list[int] = []
    in_batch: list[float | None] = []
    so_far: list[float | None] = []
    run_max: float | None = None
    for it in sorted(per_iter.keys()):
        if _emv.expected_k_for_run(by_iter.get(it, [])) <= 1:
            continue  # per-geology iteration — no ensemble-EMV interpretation
        emv_map = per_iter.get(it, {})
        b = max(emv_map.values()) if emv_map else None
        if b is not None:
            run_max = b if run_max is None else max(run_max, b)
        iters_out.append(it)
        in_batch.append(b)
        so_far.append(run_max)
    return iters_out, in_batch, so_far


def _per_iter_emv_by_kind(rows: list[dict]) -> dict[int, dict[str, list[float]]]:
    """Group rows by (iteration, kind) and compute per-candidate STRICT EMV.

    Returns a nested dict {iter: {kind: [emv, ...]}}. EMV per candidate is the strict ensemble mean
    of real_revenue across that snapshot's K geology rows (orchestrator.emv): a config with any
    failed/missing geology is dropped entirely rather than averaged over its survivors. Iterations
    without any multi-row snapshots are skipped (those are per-geology iterations with no EMV
    interpretation).
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
        k = _emv.expected_k_for_run([r for rs in snap_map.values() for r in rs])
        kind_to_emvs: dict[str, list[float]] = {}
        for sid, rs in snap_map.items():
            v = _emv.strict_per_snapshot_emv(rs, "real_revenue", expected_k=k)
            if v is None:
                continue  # config thrown out (a geology failed)
            kind = str(rs[0].get("kind", "frontier"))
            kind_to_emvs.setdefault(kind, []).append(float(v))
        if kind_to_emvs:
            out[it] = kind_to_emvs
    return out


def plot_emv_distribution(history: list[dict], rows: list[dict], out_dir: Path) -> None:
    """ENSEMBLE-ONLY: per-iteration violin of per-snapshot real EMV, split by kind."""
    if not _is_ensemble_run(rows):
        fig, ax = plt.subplots(figsize=(8, 4))
        _style_ax(ax)
        ax.text(0.5, 0.5, "No ensemble-mode iterations — skipping per-snapshot EMV distribution",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_dir / "emv_distribution.png")
        return
    grouped = _per_iter_emv_by_kind(rows)
    if not grouped:
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

    # Best-EMV-so-far overlay (STRICT, recomputed from per-geology rows — not the stored
    # history, which used the older survivor-mean rule). ``traj_iters`` spans EVERY ensemble
    # iteration, including any where all configs were thrown out (no violin/scatter is drawn
    # there, leaving a visible blank slot on the axis).
    traj_iters, _ib, traj_so_far = _strict_best_emv_trajectory(rows)
    best_emv_iters = [it for it, v in zip(traj_iters, traj_so_far) if v is not None]
    best_emv_vals = [float(v) for v in traj_so_far if v is not None]
    if best_emv_iters:
        ax.plot(best_emv_iters, best_emv_vals, color=MANIM_WHITE, linewidth=1.5,
                label="best EMV so far")

    # Pin x-ticks to the full ensemble-iteration range so iterations with no surviving config
    # (every geology failed) keep a blank x-slot rather than collapsing out of the axis.
    xticks = traj_iters or iters_sorted
    ax.set_xticks(xticks)
    if xticks:
        ax.set_xlim(min(xticks) - 0.5, max(xticks) + 0.5)
    ax.set_xlabel("AL iteration")
    ax.set_ylabel(f"Real EMV (mean {OBJ_LABEL} across ensemble)")
    ax.set_title("Per-snapshot real EMV distribution by kind (mean over K geologies)")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, out_dir / "emv_distribution.png")


def _per_iter_pes_by_kind(rows: list[dict]) -> dict[int, dict[str, list[float]]]:
    """Group rows by (iteration, kind) and compute per-candidate Probability of Economic Success.

    PES per candidate = percent of its K geology runs with NPV > 0 (computed from per-(snapshot,
    geology) ``real_npv``). Returns {iter: {kind: [pes_pct, ...]}}. NPV-mode only: rows without
    ``real_npv`` contribute nothing, so revenue runs yield an empty dict (placeholder plot).
    Iterations with no multi-geology snapshots are skipped (no ensemble PES interpretation).
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
        if not any(len(rs) > 1 for rs in snap_map.values()):
            continue
        kind_to_pes: dict[str, list[float]] = {}
        for sid, rs in snap_map.items():
            npvs = [float(v) for r in rs
                    if (v := r.get("real_npv")) is not None and np.isfinite(float(v))]
            if not npvs:
                continue
            pes = 100.0 * float(np.mean([1.0 if v > 0 else 0.0 for v in npvs]))
            kind = str(rs[0].get("kind", "frontier"))
            kind_to_pes.setdefault(kind, []).append(pes)
        if kind_to_pes:
            out[it] = kind_to_pes
    return out


def plot_pes_distribution(history: list[dict], rows: list[dict], out_dir: Path) -> None:
    """Per-iteration distribution of Probability of Economic Success (PES = % of geologies with
    NPV > 0) per candidate, split by kind. Same style as plot_emv_distribution. NPV-mode only."""
    grouped = _per_iter_pes_by_kind(rows)
    if not grouped:
        fig, ax = plt.subplots(figsize=(8, 4))
        _style_ax(ax)
        ax.text(0.5, 0.5, "No per-geology NPV — PES needs an NPV-objective ensemble run",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_dir / "pes_distribution.png")
        return

    iters_sorted = sorted(grouped.keys())
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(iters_sorted) + 4), 5))
    _style_ax(ax)
    for kind, color in KIND_COLORS.items():
        offset = KIND_OFFSETS.get(kind, 0.0)
        positions, data = [], []
        for it in iters_sorted:
            vals = grouped[it].get(kind, [])
            if not vals:
                continue
            positions.append(it + offset)
            data.append(vals)
        if not data:
            continue
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

    # Mean-PES-per-iter trend overlay (all kinds pooled).
    mean_iters, mean_vals = [], []
    for it in iters_sorted:
        allv = [v for vals in grouped[it].values() for v in vals]
        if allv:
            mean_iters.append(it)
            mean_vals.append(float(np.mean(allv)))
    if mean_iters:
        ax.plot(mean_iters, mean_vals, color=MANIM_WHITE, linewidth=1.5, label="mean PES")

    ax.set_ylim(-5, 105)
    ax.set_xticks(iters_sorted)
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Probability of economic success  (% of geologies with NPV > 0)")
    ax.set_title("Per-snapshot PES distribution by kind (fraction of K geologies with positive NPV)")
    ax.legend(loc="best", ncol=2)
    _save(fig, out_dir / "pes_distribution.png")


def plot_best_emv_so_far(history: list[dict], rows: list[dict], out_dir: Path) -> None:
    """Single line: running best real EMV across iterations.

    STRICT: the trajectory is recomputed from the per-geology rows (orchestrator.emv), so iterations
    whose batch-best was a config with a failed geology no longer count — the curve reflects only
    configs that completed all K geologies. Does not trust the stored survivor-mean history.
    """
    traj_iters, in_batch_all, so_far_all = _strict_best_emv_trajectory(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_ax(ax)
    if not traj_iters:
        ax.text(0.5, 0.5, "No ensemble-mode iterations yet", ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_dir / "best_emv_so_far.png")
        return
    # Keep EVERY ensemble iteration's x-slot, even iterations where every config was thrown out
    # (best_in_batch is None there because all configs had a failed geology). Plot None as NaN so
    # the line breaks into a visible GAP at the wiped iteration rather than connecting across it,
    # and pin the x-ticks/limits to the full iteration range so the blank slot stays on the axis.
    # best_so_far carries forward across such gaps (a flat hold), so only leading all-wiped iters
    # leave a gap in the running-best line.
    so_far_y = [np.nan if v is None else float(v) for v in so_far_all]
    in_batch_y = [np.nan if v is None else float(v) for v in in_batch_all]
    if not any(np.isfinite(so_far_y)):
        ax.text(0.5, 0.5, "No complete-ensemble configs yet (all iterations had a failed geology)",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_dir / "best_emv_so_far.png")
        return
    ax.plot(traj_iters, so_far_y, color=MANIM_BLUE, marker="o", linewidth=2.0, markersize=6,
            label="best EMV so far")
    # In-batch best per iter as a dimmer trace, for context on volatility (gaps at wiped iters).
    if any(np.isfinite(in_batch_y)):
        ax.plot(traj_iters, in_batch_y, color=MANIM_ORANGE, marker="x",
                linewidth=1.0, alpha=0.7, label="best EMV in batch")
    ax.legend(loc="lower right")
    ax.set_xticks(traj_iters)
    ax.set_xlim(min(traj_iters) - 0.5, max(traj_iters) + 0.5)
    ax.set_xlabel("AL iteration")
    ax.set_ylabel("Real EMV")
    ax.set_title("Best ensemble EMV (running)")
    _save(fig, out_dir / "best_emv_so_far.png")


def _is_ensemble_run(rows: list[dict]) -> bool:
    """An ensemble row carries a finite ``terminal_predicted_emv``; per-geology
    rows leave it as None. Used to gate ensemble-only plots."""
    for r in rows:
        v = r.get("terminal_predicted_emv")
        if v is None:
            continue
        try:
            if np.isfinite(float(v)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _ensemble_snapshot_rows(rows: list[dict]) -> list[dict]:
    """Collapse multi-geo rows of EVERY ensemble candidate (any kind) into one
    row per (iter, snapshot_id). Mirrors ``_exploit_seed_rows`` but doesn't
    filter to exploit-only.

    Returns dicts with: iteration, kind, snapshot_id, terminal_predicted_emv,
    terminal_real_emv (= mean over K real_revenues for that snapshot).
    """
    k_by_iter = {it: _emv.expected_k_for_run(irows) for it, irows in _rows_by_iter(rows).items()}
    by_iter_snap: dict[tuple[int, str], list[dict]] = {}
    for r in rows:
        sid = r.get("snapshot_id")
        if not sid:
            continue
        by_iter_snap.setdefault((int(r["iteration"]), str(sid)), []).append(r)

    out: list[dict] = []
    for (it, sid), rs in by_iter_snap.items():
        first = rs[0]
        term_pred = first.get("terminal_predicted_emv")
        try:
            tp = float(term_pred) if term_pred is not None else float("nan")
        except (TypeError, ValueError):
            tp = float("nan")
        if not np.isfinite(tp):
            continue  # not an ensemble snapshot
        # STRICT: None unless ALL K geologies are finite — a config with any failed geology is
        # thrown out of the ensemble pred-vs-real panels (downstream finite-guards drop the None).
        term_real = _emv.strict_per_snapshot_emv(rs, "real_revenue", expected_k=k_by_iter.get(it, 0))
        n_geos = sum(1 for r in rs if _emv._is_finite(r.get("real_revenue")))
        out.append({
            "iteration": it,
            "kind": first.get("kind", "?"),
            "snapshot_id": sid,
            "terminal_predicted_emv": tp,
            "terminal_real_emv": term_real,
            "n_geos": n_geos,
        })
    return out


def _exploit_seed_rows(rows: list[dict]) -> list[dict]:
    """Collapse multi-geo rows of each exploit candidate into one row per snapshot.

    Returns dicts with keys: iteration, snapshot_id, seed_predicted_emv,
    terminal_predicted_emv, seed_real_emv, terminal_real_emv.
    """
    # Per-iter K from ALL rows (the full ensemble), computed before filtering to exploit-only.
    k_by_iter = {it: _emv.expected_k_for_run(irows) for it, irows in _rows_by_iter(rows).items()}
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
        # STRICT: None unless ALL K geologies are finite (config with any failed geology is dropped).
        term_real = _emv.strict_per_snapshot_emv(rs, "real_revenue", expected_k=k_by_iter.get(it, 0))
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
    """ENSEMBLE-ONLY per-seed scatter: predicted EMV at step 0 vs at step k_safe."""
    out_path = out_dir / "exploit_seed_vs_terminal_predicted.png"
    if not _is_ensemble_run(rows):
        fig, ax = plt.subplots(figsize=(8, 4))
        _style_ax(ax)
        ax.text(0.5, 0.5, "No ensemble-mode iterations — skipping seed-vs-terminal predicted",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_path)
        return
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
        title="Exploit: predicted EMV at seed vs after Adam (one point per snapshot)",
        xlabel="Seed predicted EMV (step 0)",
        ylabel="Terminal predicted EMV (step k_safe)",
    )


def plot_exploit_seed_vs_terminal_real(rows: list[dict], out_dir: Path) -> None:
    """ENSEMBLE-ONLY per-seed scatter: real EMV of the prior elite seed vs IX-evaluated terminal EMV."""
    out_path = out_dir / "exploit_seed_vs_terminal_real.png"
    if not _is_ensemble_run(rows):
        fig, ax = plt.subplots(figsize=(8, 4))
        _style_ax(ax)
        ax.text(0.5, 0.5, "No ensemble-mode iterations — skipping seed-vs-terminal real",
                ha="center", va="center", color=MANIM_GREY)
        ax.set_axis_off()
        _save(fig, out_path)
        return
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
        title="Exploit: seed real EMV vs IX-evaluated terminal EMV (one point per snapshot)",
        xlabel="Seed real EMV (prior iter)",
        ylabel="Terminal real EMV (this iter, mean over K geos)",
    )


def plot_exploit_emv_progression(history: list[dict], rows: list[dict], out_dir: Path) -> None:
    """Per-iter exploit_best_emv vs running best_emv_so_far.

    The "best so far (any kind)" line is recomputed STRICTly from per-geology rows (orchestrator.emv)
    to stay consistent with best_emv_so_far.png; the exploit cohort line is left from history (it
    only exists for surrogate runs, which have no failed geologies, so strict == survivor there).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_ax(ax)
    iters_ex = [int(r["iteration"]) for r in history if r.get("exploit_best_emv") is not None]
    vals_ex = [float(r["exploit_best_emv"]) for r in history if r.get("exploit_best_emv") is not None]
    traj_iters, _ib, traj_so_far = _strict_best_emv_trajectory(rows)
    iters_bo = [it for it, v in zip(traj_iters, traj_so_far) if v is not None]
    vals_bo = [float(v) for v in traj_so_far if v is not None]
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

    # Objective-aware: in npv-objective runs swap the objective-value fields to NPV so the
    # convergence / objective / ensemble plots show what was actually optimized, and relabel axes.
    objective = _detect_objective(args.run_root, history)
    if objective == "npv":
        _apply_npv_objective(args.run_root, history, rows)
        global OBJECTIVE_MODE, OBJ_LABEL, OBJ_LABEL_CAP, OBJ_EMV
        OBJECTIVE_MODE = "npv"
        OBJ_LABEL = "NPV"
        OBJ_LABEL_CAP = "NPV"
        OBJ_EMV = "EMV (mean NPV across ensemble)"
        print("Objective: NPV (proxy). Objective/convergence/ensemble plots show NPV; surrogate "
              "revenue preserved in *_rev. Predicted-real gap is identical in NPV space (costs cancel).")
    # npv_context.json (written by the local driver in npv mode) drives the 3D deviated-well shape
    # on the best-config panel; None in revenue mode -> that panel stays 2D-only.
    npv_ctx = _load_npv_context(args.run_root)
    # Data prep for the holdout / well-coord plot families. Guarded so a load failure can't
    # block the rest of the dashboard.
    try:
        holdout_rows = _all_holdout_rows(args.run_root, history)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] holdout-row load failed: {type(e).__name__}: {e}", file=sys.stderr)
        holdout_rows = []
    try:
        well_rows = _load_well_coord_rows(args.run_root, history)
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] well-coord load failed: {type(e).__name__}: {e}", file=sys.stderr)
        well_rows = []

    # Each plot is dispatched independently: a single plot raising (e.g. gaussian_kde on a
    # degenerate cohort) must NOT abort the whole dashboard subprocess and silently drop every
    # subsequent plot. Render what we can; log and skip what we can't.
    plot_calls = [
        ("best_revenue", lambda: plot_best_revenue(history, rows, out_dir)),
        ("kind_revenue_summary", lambda: plot_kind_revenue_summary(rows, out_dir)),
        ("cumulative_wins_per_kind", lambda: plot_cumulative_wins_per_kind(rows, out_dir)),
        ("per_geology_winner_heatmap", lambda: plot_per_geology_winner_heatmap(rows, out_dir)),
        ("per_geology_faceted_best", lambda: plot_per_geology_faceted_best(rows, out_dir)),
        ("win_rate_over_iters", lambda: plot_win_rate_over_iters(rows, out_dir)),
        ("per_kind_gap_distribution", lambda: plot_per_kind_gap_distribution(rows, out_dir)),
        ("best_revenue_with_geo_updates", lambda: plot_best_revenue_with_geo_updates(history, rows, out_dir)),
        ("per_geology_best_so_far_facets", lambda: plot_per_geology_best_so_far_facets(rows, out_dir)),
        ("calibration_metrics", lambda: plot_calibration_metrics(history, out_dir)),
        ("pred_real_gap", lambda: plot_pred_real_gap(rows, out_dir)),
        ("per_geology_mape_heatmap", lambda: plot_per_geology_mape_heatmap(history, out_dir)),
        ("geology_selection_over_iters", lambda: plot_geology_selection(history, out_dir)),
        ("ix_query_cost_over_iters", lambda: plot_ix_query_cost(history, out_dir)),
        ("training_growth", lambda: plot_training_growth(history, out_dir)),
        ("wallclock_breakdown", lambda: plot_wallclock_breakdown(history, out_dir)),
        ("run_failure_diagnostics", lambda: plot_run_failure_diagnostics(args.run_root, history, out_dir)),
        ("predicted_vs_real_scatter", lambda: plot_predicted_vs_real_scatter(rows, out_dir)),
        ("real_revenue_distribution", lambda: plot_real_revenue_distribution(rows, out_dir)),
        ("topk_mean_gap", lambda: plot_topk_mean_gap(rows, out_dir)),
        ("pred_real_kde_shift", lambda: plot_pred_real_kde_shift(rows, out_dir)),
        ("pred_real_kde_shift_ensemble", lambda: plot_pred_real_kde_shift_ensemble(rows, out_dir)),
        ("predicted_vs_real_scatter_ensemble", lambda: plot_predicted_vs_real_scatter_ensemble(rows, out_dir)),
        ("holdout_mape_over_iters", lambda: plot_holdout_mape_over_iters(holdout_rows, out_dir)),
        ("holdout_per_geology_heatmap", lambda: plot_holdout_per_geology_heatmap(holdout_rows, out_dir)),
        ("holdout_pred_vs_real_scatter", lambda: plot_holdout_pred_vs_real_scatter(holdout_rows, out_dir)),
        ("well_position_heatmaps", lambda: plot_well_position_heatmaps(well_rows, out_dir)),
        ("well_position_heatmaps_by_kind", lambda: plot_well_position_heatmaps_by_kind(well_rows, out_dir)),
        ("well_position_heatmaps_over_iters", lambda: plot_well_position_heatmaps_over_iters(well_rows, out_dir)),
        ("best_well_config", lambda: plot_best_well_config(args.run_root, well_rows, out_dir,
                                                           rows=rows, npv_ctx=npv_ctx)),
        ("emv_distribution", lambda: plot_emv_distribution(history, rows, out_dir)),
        ("pes_distribution", lambda: plot_pes_distribution(history, rows, out_dir)),
        ("best_emv_so_far", lambda: plot_best_emv_so_far(history, rows, out_dir)),
        ("exploit_seed_vs_terminal_predicted", lambda: plot_exploit_seed_vs_terminal_predicted(rows, out_dir)),
        ("exploit_seed_vs_terminal_real", lambda: plot_exploit_seed_vs_terminal_real(rows, out_dir)),
        ("exploit_emv_progression", lambda: plot_exploit_emv_progression(history, rows, out_dir)),
        ("exploit_per_geology_progression", lambda: plot_exploit_per_geology_progression(history, out_dir)),
    ]
    n_ok = 0
    for label, fn in plot_calls:
        try:
            fn()
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — one bad plot must not drop the rest
            print(f"  [skip] {label} failed to render: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\nDone. {len(history)} iteration(s), {len(rows)} candidate rows. "
          f"{n_ok}/{len(plot_calls)} plots rendered in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
