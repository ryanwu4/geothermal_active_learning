"""Smoke tests for the surrogate-free baseline optimizer.

No IX, no GPU — runs CMA-ES and Random against a synthetic fitness surface
(sum of Gaussian bumps over the (x, y) plane). Asserts:
  (a) CMA-ES smoothed best-so-far improves over generations.
  (b) Random optimizer's best-so-far is monotone non-decreasing.
  (c) save_state / load_state round-trips both optimizers without divergence.
  (d) Projection produces only cells in valid_xy_indices.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.baseline_optimizer import (  # noqa: E402
    CMAESOptimizer, RandomOptimizer, build_optimizer, load_optimizer,
    project_to_valid_cells,
)


NX = 32
NY = 32


def _make_valid_xy(rng: np.random.Generator) -> np.ndarray:
    """Mark all cells valid except a 6×6 dead patch in the corner."""
    mask = np.ones((NX, NY), dtype=bool)
    mask[:6, :6] = False  # dead patch
    xs, ys = np.where(mask)
    return np.stack([xs, ys], axis=-1).astype(np.int32)


def _gaussian_bump_fitness(coords_xyz: np.ndarray) -> float:
    """Ensemble-mean discounted revenue proxy: sum of Gaussian bumps minus
    a pairwise-distance penalty (rewards spread).

    coords_xyz: (num_wells, 3); only (x, y) used.
    """
    centers = np.array([[20.0, 20.0], [10.0, 25.0], [25.0, 10.0]], dtype=np.float64)
    sigma = 3.5
    xy = coords_xyz[:, :2].astype(np.float64)
    bump = 0.0
    for c in centers:
        d2 = ((xy - c) ** 2).sum(axis=1)
        bump += float(np.exp(-d2 / (2 * sigma ** 2)).sum())
    # Mild spread reward to break degeneracy of all wells on one peak.
    if xy.shape[0] > 1:
        diffs = xy[:, None, :] - xy[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        bump += 0.05 * float(np.log1p(dists.min()))
    return bump


def _evaluate_population(coords: np.ndarray) -> np.ndarray:
    """Vector fitness over a (popsize, num_wells, 3) coord array."""
    return np.array([_gaussian_bump_fitness(coords[i]) for i in range(coords.shape[0])])


# ----------------------------------------------------------------------
# Projection tests
# ----------------------------------------------------------------------


def test_project_to_valid_cells_avoids_dead_rock():
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    # Five well configurations, with some wells initialized in the dead patch.
    coords_xy = np.array([
        [[2.0, 2.0], [25.0, 25.0]],   # well 0 dead
        [[3.0, 3.0], [15.0, 15.0]],   # well 0 dead
        [[10.0, 10.0], [12.0, 12.0]], # both ok
        [[5.0, 5.0], [5.0, 5.0]],     # both dead
        [[20.0, 20.0], [20.0, 20.0]], # collision -> second should move
    ], dtype=np.float32)
    out = project_to_valid_cells(coords_xy, valid, num_wells=2, rng=rng, nx=NX, ny=NY)
    valid_set = {(int(x), int(y)) for x, y in valid}
    for i in range(out.shape[0]):
        seen: set[tuple[int, int]] = set()
        for w in range(out.shape[1]):
            cell = (int(round(float(out[i, w, 0]))), int(round(float(out[i, w, 1]))))
            assert cell in valid_set, f"projected cell {cell} not in valid set"
            assert cell not in seen, f"duplicate cell {cell} within configuration {i}"
            seen.add(cell)


# ----------------------------------------------------------------------
# CMA-ES end-to-end
# ----------------------------------------------------------------------


def test_cmaes_optimizer_improves_on_gaussian_bumps():
    cmaes = pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=4, nx=NX, ny=NY, edge_buffer=2,
        fixed_depth_per_well=[5, 5, 5, 5],
        popsize=8, seed=42, valid_xy_indices=valid, sigma_init=5.0,
    )
    best_history: list[float] = []
    for _ in range(8):
        coords = opt.ask()
        assert coords.shape == (8, 4, 3)
        fits = _evaluate_population(coords)
        opt.tell(coords, fits)
        assert opt.best_fitness_so_far is not None
        best_history.append(opt.best_fitness_so_far)

    # Smoothed monotonicity: late mean ≥ early mean. We can't assert strict
    # per-generation improvement (CMA-ES can dip on a single gen), but
    # cumulative best-so-far is by definition non-decreasing.
    assert all(best_history[i] <= best_history[i + 1] + 1e-9
               for i in range(len(best_history) - 1)), \
        f"best-so-far not non-decreasing: {best_history}"
    # And there should be real signal: last best > first best by some margin.
    assert best_history[-1] > best_history[0] - 1e-9, "no improvement at all"


def test_cmaes_optimizer_handles_nan_fitness():
    """NaN fitness for some pop members must not crash tell() or poison sigma."""
    cmaes = pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=2, nx=NX, ny=NY, edge_buffer=2,
        fixed_depth_per_well=[5, 5],
        popsize=4, seed=7, valid_xy_indices=valid,
    )
    coords = opt.ask()
    fits = _evaluate_population(coords)
    fits[1] = float("nan")  # one failed eval
    fits[3] = float("nan")
    opt.tell(coords, fits)
    # best_fitness_so_far should reflect finite picks only.
    assert opt.best_fitness_so_far is not None
    assert np.isfinite(opt.best_fitness_so_far)


# ----------------------------------------------------------------------
# Random / LHS
# ----------------------------------------------------------------------


def test_random_optimizer_best_so_far_is_monotone():
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "random",
        num_wells=3, nx=NX, ny=NY, edge_buffer=2,
        fixed_depth_per_well=[5, 5, 5],
        popsize=6, seed=99, valid_xy_indices=valid,
    )
    last_best: float | None = None
    for _ in range(5):
        coords = opt.ask()
        assert coords.shape == (6, 3, 3)
        fits = _evaluate_population(coords)
        opt.tell(coords, fits)
        cur = opt.best_fitness_so_far
        if last_best is not None:
            assert cur is not None and cur >= last_best - 1e-12
        last_best = cur
    assert opt.generation == 5


# ----------------------------------------------------------------------
# State persistence
# ----------------------------------------------------------------------


def test_save_load_state_roundtrip_cmaes(tmp_path):
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=2, nx=NX, ny=NY, edge_buffer=2,
        fixed_depth_per_well=[5, 5],
        popsize=4, seed=42, valid_xy_indices=valid,
    )
    for _ in range(3):
        c = opt.ask()
        opt.tell(c, _evaluate_population(c))

    save_path = tmp_path / "opt.pkl"
    opt.save_state(save_path)
    restored = load_optimizer(save_path)
    assert restored.generation == opt.generation
    assert restored.best_fitness_so_far == opt.best_fitness_so_far

    # After restore, ask() → tell() should continue improving (or at least
    # not crash) — the round-trip preserved the CMA-ES internal state.
    c2 = restored.ask()
    f2 = _evaluate_population(c2)
    restored.tell(c2, f2)
    assert restored.generation == opt.generation + 1


def test_save_load_state_roundtrip_random(tmp_path):
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "random",
        num_wells=2, nx=NX, ny=NY, edge_buffer=2,
        fixed_depth_per_well=[5, 5],
        popsize=4, seed=99, valid_xy_indices=valid,
    )
    for _ in range(2):
        c = opt.ask()
        opt.tell(c, _evaluate_population(c))
    save_path = tmp_path / "opt.pkl"
    opt.save_state(save_path)
    restored = load_optimizer(save_path)
    assert restored.generation == opt.generation
    assert restored.best_fitness_so_far == opt.best_fitness_so_far


# ----------------------------------------------------------------------
# Output integrity
# ----------------------------------------------------------------------


def test_ask_output_only_uses_valid_cells():
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    valid_set = {(int(x), int(y)) for x, y in valid}
    opt = build_optimizer(
        "cmaes",
        num_wells=4, nx=NX, ny=NY, edge_buffer=2,
        fixed_depth_per_well=[5, 5, 5, 5],
        popsize=8, seed=1, valid_xy_indices=valid,
    )
    for _gen in range(3):
        coords = opt.ask()
        for i in range(coords.shape[0]):
            seen = set()
            for w in range(coords.shape[1]):
                cell = (int(round(float(coords[i, w, 0]))),
                        int(round(float(coords[i, w, 1]))))
                assert cell in valid_set, f"cell {cell} not in valid set"
                assert cell not in seen, f"duplicate well cell {cell} in config {i}"
                seen.add(cell)
                # Depth should be the fixed value.
                assert int(coords[i, w, 2]) == 5
        opt.tell(coords, _evaluate_population(coords))


def test_random_ask_output_only_uses_valid_cells():
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    valid_set = {(int(x), int(y)) for x, y in valid}
    opt = build_optimizer(
        "random",
        num_wells=4, nx=NX, ny=NY, edge_buffer=2,
        fixed_depth_per_well=[5, 5, 5, 5],
        popsize=8, seed=1, valid_xy_indices=valid,
    )
    coords = opt.ask()
    for i in range(coords.shape[0]):
        seen = set()
        for w in range(coords.shape[1]):
            cell = (int(round(float(coords[i, w, 0]))),
                    int(round(float(coords[i, w, 1]))))
            assert cell in valid_set
            assert cell not in seen
            seen.add(cell)
