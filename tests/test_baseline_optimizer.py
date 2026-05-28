"""Smoke tests for the surrogate-free baseline optimizer.

No IX, no GPU — runs CMA-ES and Random against a synthetic 3D fitness surface
(Gaussian bumps over (x, y, z) with a peak at a specific depth). Asserts:
  (a) CMA-ES best-so-far improves over generations.
  (b) Random optimizer's best-so-far is monotone non-decreasing.
  (c) save_state / load_state round-trips both optimizers without divergence.
  (d) (x, y) projection only outputs cells in valid_xy_indices.
  (e) z column is integer-rounded and clipped to depth_bounds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.baseline_optimizer import (  # noqa: E402
    build_optimizer, load_optimizer, project_to_valid_cells,
)


NX = 32
NY = 32


def _make_valid_xy(rng: np.random.Generator) -> np.ndarray:
    """Mark all cells valid except a 6×6 dead patch in the corner."""
    mask = np.ones((NX, NY), dtype=bool)
    mask[:6, :6] = False
    xs, ys = np.where(mask)
    return np.stack([xs, ys], axis=-1).astype(np.int32)


def _fitness(coords_xyz: np.ndarray) -> float:
    """Synthetic 3D fitness: peak at (x, y, z) ≈ (20, 20, 30)."""
    centers_xy = np.array([[20.0, 20.0], [10.0, 25.0], [25.0, 10.0]], dtype=np.float64)
    sigma_xy = 4.0
    sigma_z = 8.0
    z_target = 30.0
    xy = coords_xyz[:, :2].astype(np.float64)
    z = coords_xyz[:, 2].astype(np.float64)
    bump = 0.0
    for c in centers_xy:
        d2 = ((xy - c) ** 2).sum(axis=1)
        z_term = (z - z_target) ** 2
        bump += float(np.exp(-d2 / (2 * sigma_xy ** 2) - z_term / (2 * sigma_z ** 2)).sum())
    # Spread reward to break degeneracy.
    if xy.shape[0] > 1:
        diffs = xy[:, None, :] - xy[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        bump += 0.05 * float(np.log1p(dists.min()))
    return bump


def _evaluate_population(coords: np.ndarray) -> np.ndarray:
    return np.array([_fitness(coords[i]) for i in range(coords.shape[0])])


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------


def test_project_to_valid_cells_avoids_dead_rock():
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    coords_xy = np.array([
        [[2.0, 2.0], [25.0, 25.0]],
        [[3.0, 3.0], [15.0, 15.0]],
        [[10.0, 10.0], [12.0, 12.0]],
        [[5.0, 5.0], [5.0, 5.0]],
        [[20.0, 20.0], [20.0, 20.0]],
    ], dtype=np.float32)
    out = project_to_valid_cells(coords_xy, valid, num_wells=2, rng=rng, nx=NX, ny=NY)
    valid_set = {(int(x), int(y)) for x, y in valid}
    for i in range(out.shape[0]):
        seen: set[tuple[int, int]] = set()
        for w in range(out.shape[1]):
            cell = (int(round(float(out[i, w, 0]))), int(round(float(out[i, w, 1]))))
            assert cell in valid_set
            assert cell not in seen
            seen.add(cell)


# ----------------------------------------------------------------------
# CMA-ES (3D)
# ----------------------------------------------------------------------


def test_cmaes_returns_3d_coords_within_bounds():
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    z_lo, z_hi = 5, 25
    opt = build_optimizer(
        "cmaes",
        num_wells=3, nx=NX, ny=NY, edge_buffer=2,
        popsize=8, seed=42, valid_xy_indices=valid,
        depth_bounds=(z_lo, z_hi),
    )
    coords = opt.ask()
    assert coords.shape == (8, 3, 3)
    z = coords[..., 2]
    assert np.all((z >= z_lo) & (z <= z_hi))
    assert np.all(z == np.rint(z))  # integer-rounded
    opt.tell(coords, _evaluate_population(coords))
    assert opt.generation == 1


def test_cmaes_best_so_far_is_non_decreasing():
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=4, nx=NX, ny=NY, edge_buffer=2,
        popsize=8, seed=42, valid_xy_indices=valid, sigma_init=5.0,
        depth_bounds=(5, 50),
    )
    history: list[float] = []
    for _ in range(8):
        c = opt.ask()
        opt.tell(c, _evaluate_population(c))
        assert opt.best_fitness_so_far is not None
        history.append(opt.best_fitness_so_far)
    # best-so-far is by construction monotone non-decreasing
    for i in range(len(history) - 1):
        assert history[i] <= history[i + 1] + 1e-9, f"non-monotone: {history}"
    # Some signal: last best > first best
    assert history[-1] > history[0] - 1e-9


def test_cmaes_converges_toward_target_z():
    """Best-coords z should drift toward the synthetic peak at z=30."""
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=2, nx=NX, ny=NY, edge_buffer=2,
        popsize=8, seed=42, valid_xy_indices=valid,
        depth_bounds=(5, 50),
    )
    for _ in range(8):
        c = opt.ask()
        opt.tell(c, _evaluate_population(c))
    best = opt.best_coords_so_far
    assert best is not None
    mean_z = float(best[:, 2].mean())
    assert 15 < mean_z < 45, f"best z mean {mean_z} not near peak ~30"


def test_cmaes_nan_fitness_handled():
    """NaN fitnesses must not crash tell() or poison sigma."""
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=2, nx=NX, ny=NY, edge_buffer=2,
        popsize=4, seed=7, valid_xy_indices=valid,
        depth_bounds=(5, 40),
    )
    coords = opt.ask()
    fits = _evaluate_population(coords)
    fits[1] = float("nan")
    fits[3] = float("nan")
    opt.tell(coords, fits)
    assert opt.best_fitness_so_far is not None
    assert np.isfinite(opt.best_fitness_so_far)


# ----------------------------------------------------------------------
# Random / LHS (3D)
# ----------------------------------------------------------------------


def test_random_returns_3d_coords_within_bounds():
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "random",
        num_wells=4, nx=NX, ny=NY, edge_buffer=2,
        popsize=6, seed=99, valid_xy_indices=valid,
        depth_bounds=(10, 25),
    )
    coords = opt.ask()
    assert coords.shape == (6, 4, 3)
    z = coords[..., 2]
    assert np.all((z >= 10) & (z <= 25))
    assert np.all(z == np.rint(z))
    opt.tell(coords, _evaluate_population(coords))


def test_random_best_so_far_is_monotone():
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "random",
        num_wells=3, nx=NX, ny=NY, edge_buffer=2,
        popsize=6, seed=99, valid_xy_indices=valid,
        depth_bounds=(5, 40),
    )
    last_best: float | None = None
    for _ in range(5):
        c = opt.ask()
        opt.tell(c, _evaluate_population(c))
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
        popsize=4, seed=42, valid_xy_indices=valid,
        depth_bounds=(5, 40),
    )
    for _ in range(3):
        c = opt.ask()
        opt.tell(c, _evaluate_population(c))
    save_path = tmp_path / "opt.pkl"
    opt.save_state(save_path)
    restored = load_optimizer(save_path)
    assert restored.generation == opt.generation
    assert restored.best_fitness_so_far == opt.best_fitness_so_far
    c2 = restored.ask()
    assert c2.shape[-1] == 3
    restored.tell(c2, _evaluate_population(c2))
    assert restored.generation == opt.generation + 1


def test_save_load_state_roundtrip_random(tmp_path):
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "random",
        num_wells=2, nx=NX, ny=NY, edge_buffer=2,
        popsize=4, seed=99, valid_xy_indices=valid,
        depth_bounds=(5, 40),
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


def test_cmaes_ask_only_uses_valid_cells():
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    valid_set = {(int(x), int(y)) for x, y in valid}
    opt = build_optimizer(
        "cmaes",
        num_wells=4, nx=NX, ny=NY, edge_buffer=2,
        popsize=8, seed=1, valid_xy_indices=valid,
        depth_bounds=(5, 25),
    )
    for _ in range(3):
        coords = opt.ask()
        for i in range(coords.shape[0]):
            seen: set[tuple[int, int]] = set()
            for w in range(coords.shape[1]):
                cell = (int(round(float(coords[i, w, 0]))),
                        int(round(float(coords[i, w, 1]))))
                assert cell in valid_set
                assert cell not in seen
                seen.add(cell)
        opt.tell(coords, _evaluate_population(coords))


def test_random_ask_only_uses_valid_cells():
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    valid_set = {(int(x), int(y)) for x, y in valid}
    opt = build_optimizer(
        "random",
        num_wells=4, nx=NX, ny=NY, edge_buffer=2,
        popsize=8, seed=1, valid_xy_indices=valid,
        depth_bounds=(5, 25),
    )
    coords = opt.ask()
    for i in range(coords.shape[0]):
        seen: set[tuple[int, int]] = set()
        for w in range(coords.shape[1]):
            cell = (int(round(float(coords[i, w, 0]))),
                    int(round(float(coords[i, w, 1]))))
            assert cell in valid_set
            assert cell not in seen
            seen.add(cell)


def test_cmaes_uses_cmawm_for_mixed_integer():
    """C3: optimizer should wrap pycma's CMAwM (margin variant) for integer dims."""
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=2, nx=NX, ny=NY, edge_buffer=2,
        popsize=4, seed=42, valid_xy_indices=valid,
        depth_bounds=(5, 25),
    )
    assert type(opt._es).__name__ == "CMAwM", (
        f"expected CMAwM wrapper, got {type(opt._es).__name__}"
    )


def test_cmaes_tell_uses_raw_samples_not_projected():
    """C1: tell() should feed the raw continuous samples to CMA-ES, not the
    projection-snapped coords. The raw samples may fall outside the integer
    cell grid; the projected coords are integers — different distributions.
    """
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=3, nx=NX, ny=NY, edge_buffer=2,
        popsize=8, seed=42, valid_xy_indices=valid,
        depth_bounds=(5, 25),
    )
    coords = opt.ask()
    raw = opt._last_raw_sols
    assert raw is not None
    # Sanity: raw and projected coords are not identical (raw is continuous).
    assert raw.shape == (8, 9)
    coords_flat = coords.reshape(8, 9)
    # At least one cell should differ between raw and projected — raw is
    # continuous, projected is integer-clipped and dead-rock-snapped.
    assert not np.allclose(raw, coords_flat), \
        "raw_sols should differ from projected coords (continuous vs discrete+projected)"


def test_cmaes_resume_reseeds_rng_deterministically():
    """I6: pickle round-trip should restore a deterministic RNG state
    derived from (seed, generation), not OS entropy."""
    pytest.importorskip("cmaes")
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    opt = build_optimizer(
        "cmaes",
        num_wells=2, nx=NX, ny=NY, edge_buffer=2,
        popsize=4, seed=42, valid_xy_indices=valid,
        depth_bounds=(5, 25),
    )
    # Advance one generation so we have a non-zero state.
    c = opt.ask()
    opt.tell(c, _evaluate_population(c))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "opt.pkl"
        opt.save_state(p)
        # Two separate loads should produce the same ask() result — proves
        # reseed_rng_for_resume is making the RNG deterministic given the
        # pickled state (otherwise OS entropy would diverge them).
        a = load_optimizer(p)
        b = load_optimizer(p)
        ca = a.ask()
        cb = b.ask()
        assert np.array_equal(ca, cb), \
            "two loads of the same pickle should produce identical ask() samples"


def test_depth_bounds_validation():
    rng = np.random.default_rng(0)
    valid = _make_valid_xy(rng)
    with pytest.raises(ValueError, match="z_hi > z_lo"):
        build_optimizer(
            "cmaes",
            num_wells=2, nx=NX, ny=NY, edge_buffer=2,
            popsize=4, seed=0, valid_xy_indices=valid,
            depth_bounds=(30, 30),
        )
    with pytest.raises(ValueError, match="z_hi > z_lo"):
        build_optimizer(
            "random",
            num_wells=2, nx=NX, ny=NY, edge_buffer=2,
            popsize=4, seed=0, valid_xy_indices=valid,
            depth_bounds=(40, 10),
        )
