"""Modular global optimizers for the surrogate-free AL ablation.

Each optimizer proposes a population of candidate well placements per generation,
receives back the IX-truth fitness (ensemble-mean discounted revenue across
geologies), and adapts internal state for the next generation. Unlike
``orchestrator/acquire.py``, no surrogate model is involved — fitness comes
exclusively from IX simulator runs dispatched via the existing Sherlock
infrastructure.

Two optimizers are provided:

* :class:`CMAESOptimizer` — wraps ``cmaes.CMA`` (pycma). Primary baseline.
* :class:`RandomOptimizer` — stratified LHS on each generation; ``tell`` is a
  no-op except for tracking running best. Lower-bound floor.

A shared :func:`project_to_valid_cells` helper enforces the same edge-buffered,
dead-rock-avoiding, uniqueness-respecting projection that the surrogate
acquisition pathway uses (see ``orchestrator/acquire.py:_cma_seed_starts``).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


# Sentinel for the minimizer when an IX evaluation failed (NaN fitness). cmaes
# minimizes; we hand it a huge finite penalty so the rank-based update simply
# rejects that sample without poisoning sigma.
WORST_SENTINEL = 1e30


# ----------------------------------------------------------------------
# Shared projection helper
# ----------------------------------------------------------------------


def project_to_valid_cells(
    coords_xy: np.ndarray,
    valid_xy_indices: np.ndarray,
    num_wells: int,
    rng: np.random.Generator,
    *,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Project a (popsize, num_wells, 2) float array onto valid integer cells.

    For each well in each population member: round to nearest cell; if that
    cell is not in ``valid_xy_indices`` (dead rock) OR is already used by a
    prior well in the same configuration, pick a random valid cell from
    ``valid_xy_indices``. Returns float coords (still in [0, nx-1] × [0, ny-1])
    so downstream code can treat them uniformly.

    Mirrors the projection block at acquire.py:587-603.
    """
    out = coords_xy.astype(np.float32).copy()
    valid_set: set[tuple[int, int]] = {(int(x), int(y)) for x, y in valid_xy_indices.astype(int)}
    M = out.shape[0]
    for i in range(M):
        used: set[tuple[int, int]] = set()
        for w in range(num_wells):
            rx, ry = float(out[i, w, 0]), float(out[i, w, 1])
            cell = (
                int(np.clip(round(rx), 0, nx - 1)),
                int(np.clip(round(ry), 0, ny - 1)),
            )
            if (cell not in valid_set) or (cell in used):
                pick = valid_xy_indices[rng.integers(0, len(valid_xy_indices))]
                cell = (int(pick[0]), int(pick[1]))
            used.add(cell)
            out[i, w, 0] = float(cell[0])
            out[i, w, 1] = float(cell[1])
    return out


def intersect_valid_xy_indices(geology_h5_paths: list[Path]) -> np.ndarray:
    """Intersection of valid (x, y) reservoir cells across all geologies.

    Reads ``Input/Temperature0`` from each H5; a cell is "valid" if its column
    has any layer with temperature > -900 (the dead-rock sentinel used in
    ``geothermal/active_learning_utils.py:to_julia_wells_text``). The optimizer
    proposes one (x, y) shared across all geologies, so we project onto the
    intersection.

    Returns an (N, 2) int array of valid (x, y) cell indices.
    """
    import h5py

    mask_intersect: np.ndarray | None = None
    for h5_path in geology_h5_paths:
        with h5py.File(h5_path, "r") as f:
            temp0 = f["Input/Temperature0"][:]  # (z, x, y)
        per_col = (temp0 > -900).any(axis=0)  # (x, y)
        mask_intersect = per_col if mask_intersect is None else (mask_intersect & per_col)
    if mask_intersect is None:
        raise RuntimeError("No geology files supplied to intersect_valid_xy_indices.")
    xs, ys = np.where(mask_intersect)
    return np.stack([xs, ys], axis=-1).astype(np.int32)


# ----------------------------------------------------------------------
# Optimizer protocol
# ----------------------------------------------------------------------


@runtime_checkable
class BaselineOptimizer(Protocol):
    """Common surface for surrogate-free global optimizers.

    Conventions:
    - ``ask()`` returns coords of shape ``(popsize, num_wells, 3)`` with the
      x and y dims already projected onto valid cells. The z column is filled
      with the optimizer's ``fixed_depth_per_well`` (one int per well).
    - ``tell(coords, fitnesses)`` takes the same coords array back together
      with a 1-D ``(popsize,)`` array of fitnesses (HIGHER = BETTER, i.e.
      ensemble-mean discounted revenue). Use NaN for failed evaluations; the
      optimizer is responsible for handling NaN safely.
    - Saving/loading state is by ``pickle`` of the whole instance to ``path``.
    """

    def ask(self) -> np.ndarray: ...
    def tell(self, coords: np.ndarray, fitnesses: np.ndarray) -> None: ...
    def save_state(self, path: Path) -> None: ...
    @property
    def generation(self) -> int: ...
    @property
    def best_fitness_so_far(self) -> float | None: ...


def load_optimizer(path: Path) -> BaselineOptimizer:
    """Generic loader — returns whichever optimizer subclass was pickled."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, BaselineOptimizer):
        raise TypeError(f"Loaded object from {path} is not a BaselineOptimizer ({type(obj)}).")
    return obj


# ----------------------------------------------------------------------
# CMA-ES
# ----------------------------------------------------------------------


@dataclass
class CMAESOptimizer:
    """CMA-ES over per-well (x, y, z).

    Maintains a ``cmaes.CMA`` instance over a ``3 * num_wells``-dim real-valued
    space, bounded per well by ``[edge_buffer, nx-1-edge_buffer]``,
    ``[edge_buffer, ny-1-edge_buffer]``, and ``[z_lo, z_hi]``. ``ask()`` projects
    (x, y) onto ``valid_xy_indices`` with uniqueness enforcement and rounds z to
    the nearest integer cell clipped to ``[z_lo, z_hi]``. ``tell()`` converts
    max-fitness to min-cost (NaN→WORST_SENTINEL) and feeds CMA-ES the
    post-projection coords so the covariance reflects what the simulator
    actually saw.
    """

    num_wells: int
    nx: int
    ny: int
    edge_buffer: int
    depth_bounds: tuple[int, int]   # (z_lo, z_hi) — required
    popsize: int
    sigma_init: float
    seed: int
    valid_xy_indices: np.ndarray = field(repr=False)

    # Runtime state (populated by __post_init__ / restored by pickle).
    _es: object = field(default=None, repr=False)
    _generation: int = 0
    _best_fitness: float | None = None
    _best_coords: np.ndarray | None = field(default=None, repr=False)
    _last_proposal: np.ndarray | None = field(default=None, repr=False)  # (popsize, num_wells, 3) projected
    _last_raw_sols: np.ndarray | None = field(default=None, repr=False)  # (popsize, dim) raw CMA-ES samples

    def __post_init__(self) -> None:
        if not (self.depth_bounds[1] > self.depth_bounds[0]):
            raise ValueError(f"depth_bounds must have z_hi > z_lo; got {self.depth_bounds}")
        if self._es is None:
            self._initialize_es()

    def _initialize_es(self) -> None:
        try:
            from cmaes import CMA  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "CMAESOptimizer requires the `cmaes` package. Install via `pip install cmaes`."
            ) from e

        dim = self.num_wells * 3
        x_lo, x_hi = float(self.edge_buffer), float(self.nx - 1 - self.edge_buffer)
        y_lo, y_hi = float(self.edge_buffer), float(self.ny - 1 - self.edge_buffer)
        z_lo, z_hi = float(self.depth_bounds[0]), float(self.depth_bounds[1])
        bounds = np.array(
            [[x_lo, x_hi], [y_lo, y_hi], [z_lo, z_hi]] * self.num_wells,
            dtype=np.float64,
        )
        mean_init = np.tile(
            [0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), 0.5 * (z_lo + z_hi)],
            self.num_wells,
        ).astype(np.float64)
        mean_init = np.clip(mean_init, bounds[:, 0] + 1e-3, bounds[:, 1] - 1e-3)

        self._es = CMA(
            mean=mean_init,
            sigma=float(self.sigma_init),
            bounds=bounds,
            seed=int(self.seed & 0xFFFFFFFF),
            population_size=int(self.popsize),
        )

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best_fitness_so_far(self) -> float | None:
        return self._best_fitness

    @property
    def best_coords_so_far(self) -> np.ndarray | None:
        return None if self._best_coords is None else self._best_coords.copy()

    def ask(self) -> np.ndarray:
        """Sample popsize candidates, project to valid cells, return (popsize, num_wells, 3)."""
        es = self._es
        if es is None:
            raise RuntimeError("CMA-ES instance not initialized.")
        sols = np.stack([es.ask() for _ in range(self.popsize)], axis=0)  # (popsize, dim)
        coords = sols.reshape(self.popsize, self.num_wells, 3)

        # rng for projection — bind to (seed, generation) so a resumed run reproduces.
        rng = np.random.default_rng((int(self.seed) & 0xFFFFFFFF) ^ (self._generation * 7919))
        projected_xy = project_to_valid_cells(
            coords[:, :, :2], self.valid_xy_indices, self.num_wells, rng,
            nx=self.nx, ny=self.ny,
        )

        z_lo, z_hi = self.depth_bounds
        # Round z to integer cell index, then clip. CMA-ES's box constraint
        # already keeps z_raw in [z_lo, z_hi] but we re-clip after rounding
        # to handle the half-integer edge case.
        z_int = np.clip(
            np.rint(coords[:, :, 2]).astype(np.int32), int(z_lo), int(z_hi),
        ).astype(np.float32)
        coords_xyz = np.concatenate(
            [projected_xy, z_int[..., None]], axis=-1
        ).astype(np.float32)

        self._last_proposal = coords_xyz
        self._last_raw_sols = sols
        return coords_xyz

    def tell(self, coords: np.ndarray, fitnesses: np.ndarray) -> None:
        """Feed back fitnesses for the most recently proposed population."""
        if self._last_proposal is None or self._last_raw_sols is None:
            raise RuntimeError("tell() called before ask() — nothing to update.")
        if coords.shape[0] != self.popsize:
            raise ValueError(f"coords popsize={coords.shape[0]} != expected {self.popsize}")
        if fitnesses.shape != (self.popsize,):
            raise ValueError(f"fitnesses shape {fitnesses.shape} != ({self.popsize},)")

        fits = np.asarray(fitnesses, dtype=np.float64)
        finite_mask = np.isfinite(fits)
        # Max-fitness → min-cost; failed evals get WORST_SENTINEL so they're
        # ranked last without injecting NaN into the covariance update.
        costs = np.where(finite_mask, -fits, WORST_SENTINEL)

        # Use post-projection coords for the CMA update so the covariance
        # estimate reflects what the simulator actually saw. Flatten to
        # (popsize, 3*num_wells) matching what _es.ask() returned.
        flat_proj = coords.reshape(self.popsize, -1).astype(np.float64)
        es = self._es
        assert es is not None
        es.tell(list(zip([s for s in flat_proj], costs.tolist())))  # type: ignore[attr-defined]

        # Update best-so-far over finite evaluations.
        for i in range(self.popsize):
            if not finite_mask[i]:
                continue
            v = float(fits[i])
            if (self._best_fitness is None) or (v > self._best_fitness):
                self._best_fitness = v
                self._best_coords = coords[i].copy()

        self._generation += 1
        self._last_proposal = None
        self._last_raw_sols = None

    def save_state(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


# ----------------------------------------------------------------------
# Random / LHS baseline
# ----------------------------------------------------------------------


@dataclass
class RandomOptimizer:
    """LHS-stratified random sampler over the edge-buffered (x, y, z) window.

    Stateless except for the running best-so-far. ``tell()`` only updates the
    best; it does not adapt sampling. Uses ``scipy.stats.qmc.LatinHypercube``
    when available; falls back to uniform if not.
    """

    num_wells: int
    nx: int
    ny: int
    edge_buffer: int
    depth_bounds: tuple[int, int]   # (z_lo, z_hi) — required
    popsize: int
    seed: int
    valid_xy_indices: np.ndarray = field(repr=False)

    _generation: int = 0
    _best_fitness: float | None = None
    _best_coords: np.ndarray | None = field(default=None, repr=False)
    _last_proposal: np.ndarray | None = field(default=None, repr=False)
    _rng: np.random.Generator | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not (self.depth_bounds[1] > self.depth_bounds[0]):
            raise ValueError(f"depth_bounds must have z_hi > z_lo; got {self.depth_bounds}")
        if self._rng is None:
            self._rng = np.random.default_rng(int(self.seed) & 0xFFFFFFFF)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best_fitness_so_far(self) -> float | None:
        return self._best_fitness

    @property
    def best_coords_so_far(self) -> np.ndarray | None:
        return None if self._best_coords is None else self._best_coords.copy()

    def _sample_lhs(self) -> np.ndarray:
        """LHS over (popsize, num_wells, 3) in the edge-buffered window."""
        dim = self.num_wells * 3
        try:
            from scipy.stats import qmc  # type: ignore
            sampler = qmc.LatinHypercube(d=dim, seed=int(self._rng.integers(0, 2**31 - 1)))
            unit = sampler.random(self.popsize)
        except ImportError:
            unit = self._rng.random((self.popsize, dim))

        x_lo, x_hi = float(self.edge_buffer), float(self.nx - 1 - self.edge_buffer)
        y_lo, y_hi = float(self.edge_buffer), float(self.ny - 1 - self.edge_buffer)
        z_lo, z_hi = float(self.depth_bounds[0]), float(self.depth_bounds[1])
        coords = np.empty((self.popsize, self.num_wells, 3), dtype=np.float32)
        for w in range(self.num_wells):
            base = 3 * w
            coords[:, w, 0] = x_lo + unit[:, base + 0] * (x_hi - x_lo)
            coords[:, w, 1] = y_lo + unit[:, base + 1] * (y_hi - y_lo)
            coords[:, w, 2] = z_lo + unit[:, base + 2] * (z_hi - z_lo)
        return coords

    def ask(self) -> np.ndarray:
        coords = self._sample_lhs()
        rng_proj = np.random.default_rng((int(self.seed) & 0xFFFFFFFF) ^ (self._generation * 1009))
        projected_xy = project_to_valid_cells(
            coords[:, :, :2], self.valid_xy_indices, self.num_wells, rng_proj,
            nx=self.nx, ny=self.ny,
        )
        z_lo, z_hi = self.depth_bounds
        z_int = np.clip(
            np.rint(coords[:, :, 2]).astype(np.int32), int(z_lo), int(z_hi),
        ).astype(np.float32)
        coords_xyz = np.concatenate(
            [projected_xy, z_int[..., None]], axis=-1
        ).astype(np.float32)
        self._last_proposal = coords_xyz
        return coords_xyz

    def tell(self, coords: np.ndarray, fitnesses: np.ndarray) -> None:
        if coords.shape[0] != self.popsize:
            raise ValueError(f"coords popsize={coords.shape[0]} != expected {self.popsize}")
        if fitnesses.shape != (self.popsize,):
            raise ValueError(f"fitnesses shape {fitnesses.shape} != ({self.popsize},)")
        for i in range(self.popsize):
            v = float(fitnesses[i])
            if not np.isfinite(v):
                continue
            if (self._best_fitness is None) or (v > self._best_fitness):
                self._best_fitness = v
                self._best_coords = coords[i].copy()
        self._generation += 1
        self._last_proposal = None

    def save_state(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_optimizer(
    kind: str,
    *,
    num_wells: int,
    nx: int,
    ny: int,
    edge_buffer: int,
    popsize: int,
    seed: int,
    valid_xy_indices: np.ndarray,
    depth_bounds: tuple[int, int],
    sigma_init: float = 5.0,
) -> BaselineOptimizer:
    """Construct an optimizer by string name. Search dim = 3 × num_wells."""
    kind = kind.lower()
    if kind == "cmaes":
        return CMAESOptimizer(
            num_wells=num_wells, nx=nx, ny=ny, edge_buffer=edge_buffer,
            depth_bounds=depth_bounds, popsize=popsize,
            sigma_init=sigma_init, seed=seed, valid_xy_indices=valid_xy_indices,
        )
    if kind in ("random", "lhs"):
        return RandomOptimizer(
            num_wells=num_wells, nx=nx, ny=ny, edge_buffer=edge_buffer,
            depth_bounds=depth_bounds, popsize=popsize,
            seed=seed, valid_xy_indices=valid_xy_indices,
        )
    raise ValueError(f"Unknown optimizer kind {kind!r}. Choices: cmaes, random/lhs.")
