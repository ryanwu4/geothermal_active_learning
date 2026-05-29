"""Tests for the CMA-ES-over-surrogate acquisition mode.

The model-heavy entry point ``_run_acquisition_cma_surrogate`` itself needs
CUDA, a real surrogate checkpoint, and per-geology HDF5 files (same constraint
the ensemble suite documents), so it is exercised by the local smoke run rather
than here. These tests cover the genuinely-new, model-free pieces:

  * ``baseline_optimizer.CMAESOptimizer`` ``mean_init`` seeding + validation.
  * The CMA ask/tell/pool loop pattern the acquisition uses, driven by an
    analytic fitness: it must (a) improve the (x, y) objective and (b) keep
    exploring depth (z) broadly even when fitness is flat in z — which mirrors
    the depth-blind surrogate and is the whole point of optimizing depth here.
  * The ``run_acquisition`` dispatch routing ``mode == "cma_surrogate"`` to the
    new function.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cmaes = pytest.importorskip("cmaes")  # CMAESOptimizer requires the `cmaes` package

from orchestrator import acquire as acq
from orchestrator.acquire import (
    AcquisitionConfig,
    GeologySpec,
    WellSpec,
    _dedup_indices_by_xy,
)
from orchestrator.baseline_optimizer import CMAESOptimizer, build_optimizer


def _full_grid_indices(nx: int, ny: int, edge_buffer: int) -> np.ndarray:
    """All (x, y) cells inside the edge-buffered window — the valid set."""
    xs = np.arange(edge_buffer, nx - edge_buffer)
    ys = np.arange(edge_buffer, ny - edge_buffer)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.int32)


# ---------------------------------------------------------------------------
# 1. mean_init seeding
# ---------------------------------------------------------------------------


class TestMeanInit:
    def test_wrong_length_raises(self) -> None:
        valid = _full_grid_indices(40, 40, 5)
        with pytest.raises(ValueError):
            CMAESOptimizer(
                num_wells=4, nx=40, ny=40, edge_buffer=5,
                depth_bounds=(5, 25), popsize=8, sigma_init=4.0, seed=0,
                valid_xy_indices=valid,
                mean_init=np.zeros(4 * 3 - 1),  # one short
            )

    def test_mean_init_shifts_initial_depth(self) -> None:
        """With a tight sigma, the first ask population's mean depth should track
        the mean_init z (z is clipped to depth bounds, not snapped to valid xy)."""
        valid = _full_grid_indices(40, 40, 5)
        num_wells = 4
        # Two seeds identical except z: shallow vs deep.
        shallow = np.zeros((num_wells, 3))
        deep = np.zeros((num_wells, 3))
        for w in range(num_wells):
            shallow[w] = [12.0, 12.0, 8.0]
            deep[w] = [12.0, 12.0, 22.0]
        opt_lo = build_optimizer(
            "cmaes", num_wells=num_wells, nx=40, ny=40, edge_buffer=5,
            popsize=16, seed=1, valid_xy_indices=valid,
            depth_bounds=(5, 25), sigma_init=2.0, mean_init=shallow.reshape(-1),
        )
        opt_hi = build_optimizer(
            "cmaes", num_wells=num_wells, nx=40, ny=40, edge_buffer=5,
            popsize=16, seed=1, valid_xy_indices=valid,
            depth_bounds=(5, 25), sigma_init=2.0, mean_init=deep.reshape(-1),
        )
        z_lo = opt_lo.ask()[:, :, 2].mean()
        z_hi = opt_hi.ask()[:, :, 2].mean()
        assert z_lo < z_hi, f"deep-seeded mean z ({z_hi}) should exceed shallow ({z_lo})"

    def test_default_mean_when_none(self) -> None:
        """mean_init=None must still build and ask a valid population."""
        valid = _full_grid_indices(40, 40, 5)
        opt = build_optimizer(
            "cmaes", num_wells=4, nx=40, ny=40, edge_buffer=5,
            popsize=8, seed=0, valid_xy_indices=valid,
            depth_bounds=(5, 25), sigma_init=4.0,
        )
        coords = opt.ask()
        assert coords.shape == (8, 4, 3)
        # x, y inside the edge window; z inside depth bounds.
        assert (coords[:, :, 0] >= 5).all() and (coords[:, :, 0] <= 34).all()
        assert (coords[:, :, 2] >= 5).all() and (coords[:, :, 2] <= 25).all()


# ---------------------------------------------------------------------------
# 2. The CMA ask/tell/pool loop the acquisition uses
# ---------------------------------------------------------------------------


def test_cma_loop_improves_xy_and_explores_depth() -> None:
    """Drive CMAESOptimizer with an analytic fitness that depends only on (x, y)
    (flat in z) — exactly the regime the depth-blind surrogate puts CMA in.

    Asserts the two properties the acquisition relies on:
      (1) the (x, y) objective improves over generations (CMA works), and
      (2) depth (z) keeps being sampled broadly across its bounds because the
          fitness exerts no selection pressure on z (so the queried batch will
          carry depth-varying IX labels).
    """
    nx = ny = 40
    edge_buffer = 5
    num_wells = 4
    z_lo, z_hi = 5, 25
    valid = _full_grid_indices(nx, ny, edge_buffer)
    target = np.array([20.0, 20.0], dtype=np.float64)  # common xy target

    opt = build_optimizer(
        "cmaes", num_wells=num_wells, nx=nx, ny=ny, edge_buffer=edge_buffer,
        popsize=16, seed=7, valid_xy_indices=valid,
        depth_bounds=(z_lo, z_hi), sigma_init=6.0,
    )

    def fitness(coords: np.ndarray) -> np.ndarray:
        # Higher is better. -mean squared distance of each well's (x, y) to target.
        d2 = ((coords[:, :, :2] - target[None, None, :]) ** 2).sum(axis=-1)  # (pop, wells)
        return -d2.mean(axis=1)

    pool_z: list[float] = []
    best_by_gen: list[float] = []
    for _gen in range(40):
        coords = opt.ask()
        fits = fitness(coords)
        opt.tell(coords, fits)
        best_by_gen.append(float(fits.max()))
        pool_z.extend(coords[:, :, 2].ravel().tolist())

    # (1) Objective improves: late-gen best beats early-gen best.
    early = max(best_by_gen[:5])
    late = max(best_by_gen[-5:])
    assert late > early, f"CMA did not improve xy fitness (early={early}, late={late})"

    # (2) Depth explored broadly across [z_lo, z_hi] (fitness is z-flat).
    pz = np.asarray(pool_z)
    assert pz.min() <= z_lo + 4, f"shallow depths never sampled (min z={pz.min()})"
    assert pz.max() >= z_hi - 4, f"deep depths never sampled (max z={pz.max()})"
    assert pz.std() >= 3.0, f"depth barely varied (std={pz.std():.2f})"


def test_cma_tell_is_nan_safe() -> None:
    """A generation where some fitnesses are NaN must not crash tell()."""
    valid = _full_grid_indices(40, 40, 5)
    opt = build_optimizer(
        "cmaes", num_wells=3, nx=40, ny=40, edge_buffer=5,
        popsize=8, seed=3, valid_xy_indices=valid,
        depth_bounds=(5, 25), sigma_init=5.0,
    )
    coords = opt.ask()
    fits = np.full(8, 1.0e8, dtype=np.float64)
    fits[::2] = np.nan  # half the batch failed
    opt.tell(coords, fits)  # must not raise
    assert opt.generation == 1
    assert opt.best_fitness_so_far is not None


# ---------------------------------------------------------------------------
# 3. exploit shortlist dedup (_dedup_indices_by_xy)
# ---------------------------------------------------------------------------


def _cfg(xy_list, z_list):
    """Build a (num_wells,3) coord array from per-well xy + z."""
    n = len(xy_list)
    c = np.zeros((n, 3), dtype=np.float32)
    for w, ((x, y), z) in enumerate(zip(xy_list, z_list)):
        c[w] = [x, y, z]
    return c


class TestDedupByXY:
    def test_excludes_depth_from_distance(self) -> None:
        """Two configs identical in xy but very different in z are near-duplicate
        *placements* — only the higher-ranked one should survive."""
        a = _cfg([(10, 10), (30, 30)], [10, 10])
        b = _cfg([(10, 10), (30, 30)], [60, 60])  # same xy, deep z
        far = _cfg([(50, 50), (20, 40)], [10, 10])
        coords = [a, b, far]
        order = np.array([0, 1, 2])  # a, then b (dup of a), then far
        # k=2 so the backfill (which would re-add the filtered dup to reach k)
        # is not triggered — this isolates the dedup filter itself.
        picked = _dedup_indices_by_xy(coords, order, num_wells=2, k=2, min_xy_dist=4.0)
        # b must be filtered (same xy as a); a and far survive.
        assert picked == [0, 2], f"expected [0,2], got {picked}"

    def test_keeps_distinct_xy(self) -> None:
        a = _cfg([(10, 10), (30, 30)], [10, 10])
        b = _cfg([(40, 40), (12, 50)], [10, 10])
        picked = _dedup_indices_by_xy([a, b], np.array([0, 1]), num_wells=2, k=2, min_xy_dist=4.0)
        assert picked == [0, 1]

    def test_backfills_when_too_few_distinct(self) -> None:
        """If fewer than k distinct-xy layouts exist, backfill from order so the
        shortlist is never silently short."""
        a = _cfg([(10, 10)], [10])
        b = _cfg([(10, 10)], [11])  # dup xy
        c = _cfg([(10, 10)], [12])  # dup xy
        picked = _dedup_indices_by_xy([a, b, c], np.array([0, 1, 2]), num_wells=1, k=3, min_xy_dist=4.0)
        # Only 'a' is distinct; backfill adds b, c (in order) to reach k=3.
        assert len(picked) == 3
        assert picked[0] == 0
        assert set(picked) == {0, 1, 2}

    def test_respects_order(self) -> None:
        # order puts index 2 first; it should be picked first.
        a = _cfg([(10, 10)], [10])
        b = _cfg([(40, 40)], [10])
        c = _cfg([(70, 70)], [10])
        picked = _dedup_indices_by_xy([a, b, c], np.array([2, 0, 1]), num_wells=1, k=2, min_xy_dist=4.0)
        assert picked == [2, 0]


# ---------------------------------------------------------------------------
# 4. run_acquisition dispatch
# ---------------------------------------------------------------------------


def test_run_acquisition_dispatches_cma_surrogate(tmp_path: Path, monkeypatch) -> None:
    """mode == 'cma_surrogate' must route to _run_acquisition_cma_surrogate
    (before any model load), and NOT to the ensemble or per-geology paths."""
    called = {}

    def _sentinel(cfg, *, out_dir, iteration, run_id_prefix="al"):
        called["cma"] = True
        return {"manifest": {}, "manifest_path": "x", "candidates": []}

    def _boom_ensemble(*a, **k):
        raise AssertionError("ensemble path must not be reached for cma_surrogate")

    monkeypatch.setattr(acq, "_run_acquisition_cma_surrogate", _sentinel)
    monkeypatch.setattr(acq, "_run_acquisition_ensemble", _boom_ensemble)

    cfg = acq.AcquisitionConfig(
        surrogate_repo=Path("/tmp/repo"),
        checkpoint_path=Path("/tmp/ckpt.pt"),
        scaler_path=Path("/tmp/scaler.pkl"),
        norm_config_path=Path("/tmp/norm.json"),
        geologies=[],
        wells=[],
        n_starts_per_geology=0,
        k_safe=45,
        k_adv=45,
        adv_fraction=0.0,
        edge_buffer=10,
        learning_rate=0.5,
        log_every_n_steps=25,
        revenue_target="graph_discounted_net_revenue",
        mode="cma_surrogate",
    )
    out = acq.run_acquisition(cfg, out_dir=tmp_path, iteration=0)
    assert called.get("cma") is True
    assert out["candidates"] == []


# ---------------------------------------------------------------------------
# 5. Full per-query-rebuild flow with a fake surrogate (no CUDA / checkpoint / H5)
# ---------------------------------------------------------------------------
#
# These drive the REAL _run_acquisition_cma_surrogate (via run_acquisition) end
# to end, stubbing only the heavy/model-dependent boundaries: the worker context
# + model, the geology load, the graph builder, and torch_geometric's Batch. The
# real CMA-ES engine, dedup, FPS, projection, snapshot/Candidate emission and
# manifest writer all run. This locks in the refactor's load-bearing invariants:
# reported predicted_emv == the accurate prediction at the candidate's coords,
# each candidate's graph is built exactly once, depth varies in-bounds, etc.

import torch  # available wherever orchestrator.acquire imports

_FAKE_TARGET = (20.0, 20.0)  # xy the fake surrogate "likes" (higher revenue)


class _FakeWell:
    def __init__(self):
        self.pos_xyz = None


class _FakeBatch:
    """Stands in for a torch_geometric Batch: supports .to() and bd['well']."""
    def __init__(self, m):
        self.m = m
        self._w = _FakeWell()

    def to(self, _device):
        return self

    def __getitem__(self, _key):
        return self._w


class _FakeBatchType:
    @staticmethod
    def from_data_list(graphs):
        return _FakeBatch(len(graphs))


def _fake_model_factory(num_wells):
    """Deterministic surrogate: revenue = -mean_w ||xy - target||^2 (depends on
    xy; geology-independent so mean-over-K == this value)."""
    def _model(bd):
        pos = bd["well"].pos_xyz  # (M*num_wells, 3) torch
        m = pos.shape[0] // num_wells
        xyz = pos.reshape(m, num_wells, 3)
        tx = torch.tensor(_FAKE_TARGET, dtype=pos.dtype)
        d = ((xyz[:, :, :2] - tx) ** 2).sum(dim=-1).mean(dim=1)  # (m,)
        return -d
    return _model


def _expected_revenue(coords_xyz: np.ndarray) -> float:
    xy = np.asarray(coords_xyz)[:, :2]
    return float(-((xy - np.array(_FAKE_TARGET)) ** 2).sum(axis=1).mean())


def _install_fake_surrogate(monkeypatch, tmp_path, *, K, num_wells, nx=40, ny=40, z_dim=30):
    """Patch the model/geology/graph boundaries of acquire for a fake run.

    Returns (geologies, build_calls). ``build_calls`` records every
    _build_static_batch_for_starts invocation so tests can assert build-once.
    """
    import h5py

    # Tiny real H5 files so the `with h5py.File(gp)` in the function succeeds;
    # contents are ignored (metadata + load are patched below).
    geos = []
    for i in range(K):
        p = tmp_path / f"geo_{i}.h5"
        h5py.File(p, "w").close()
        geos.append(GeologySpec(geology_index=i, geology_h5_file=str(p), geology_name=f"geo_{i}"))

    ctx = SimpleNamespace(
        device=torch.device("cpu"),
        model=_fake_model_factory(num_wells),
        scaler=None,
        target_mean=0.0,
        target_scale=1.0,
        norm_config={},
        is_injector_list=[i % 2 == 0 for i in range(num_wells)],
        num_wells=num_wells,
        base_seed=123,
        build_wells_table=None, extract_vertical_profiles=None, extract_well_data=None,
        read_geology_metadata=(lambda *a, **k: {}),
        to_julia_wells_text=(lambda **k: "# fake jl\n"),
        build_single_hetero_data=None,
        PROPERTIES=[], PERM_PROPS=[], find_z_cutoff=None, get_valid_mask=None,
        node_encoder="cnn", enrich_global_attr=False,
    )

    fake_temp0 = np.ones((z_dim, nx, ny), dtype=np.float32)  # every column live rock

    def _fake_load_geology(gp, *a, **k):
        return {
            "physics_dict": {}, "full_shape": (z_dim, nx, ny), "z_cutoff": z_dim,
            "nx": nx, "ny": ny, "z_max": z_dim, "temp0_full": fake_temp0,
        }

    build_calls: list[int] = []

    def _fake_build_static(cfgs, *a, **k):
        build_calls.append(len(cfgs))
        return [0] * len(cfgs)  # length must equal #candidates; contents unused

    monkeypatch.setattr(acq, "_build_worker_context", lambda cfg, it, dev: ctx)
    monkeypatch.setattr(acq, "_read_geology_metadata_safe",
                        lambda rgm, src, geology_name: {"geology_config_id": 1,
                                                        "scenario_name": "s", "sample_num": 0})
    monkeypatch.setattr(acq, "_load_geology", _fake_load_geology)
    monkeypatch.setattr(acq, "_build_static_batch_for_starts", _fake_build_static)
    monkeypatch.setattr(acq, "Batch", _FakeBatchType)
    return geos, build_calls


def _fake_cfg(geos, *, num_wells, n_exploit, n_frontier, gens, popsize,
              depth_min=2, depth_max=20, edge_buffer=5,
              cma_warm_start=False, prior_metrics=None):
    wells = [WellSpec(type=("injector" if i % 2 == 0 else "producer"), depth=10)
             for i in range(num_wells)]
    return AcquisitionConfig(
        surrogate_repo=Path("/tmp/repo"), checkpoint_path=Path("/tmp/c.ckpt"),
        scaler_path=Path("/tmp/s.pkl"), norm_config_path=Path("/tmp/n.json"),
        geologies=geos, wells=wells,
        n_starts_per_geology=0, k_safe=45, k_adv=45, adv_fraction=0.0,
        edge_buffer=edge_buffer, learning_rate=0.5, log_every_n_steps=25,
        revenue_target="graph_discounted_net_revenue", seed=42,
        device="cpu", devices=["cpu"], mode="cma_surrogate",
        cma_popsize=popsize, cma_generations=gens, cma_sigma_init=6.0,
        n_exploit=n_exploit, n_frontier=n_frontier,
        depth_min=depth_min, depth_max=depth_max,
        cma_warm_start=cma_warm_start, prior_metrics=prior_metrics,
    )


def _capture_mean_init(monkeypatch):
    """Spy on baseline_optimizer.build_optimizer (imported locally inside
    _run_acquisition_cma_surrogate) to capture the mean_init it receives."""
    import orchestrator.baseline_optimizer as bopt
    captured = {}
    real = bopt.build_optimizer

    def _cap(*a, **k):
        captured["mean_init"] = k.get("mean_init")
        return real(*a, **k)

    monkeypatch.setattr(bopt, "build_optimizer", _cap)
    return captured


class TestRunCmaSurrogateFake:
    def test_reported_emv_equals_accurate_prediction(self, tmp_path, monkeypatch) -> None:
        """The reported predicted_emv for every emitted candidate must equal the
        surrogate's prediction at that candidate's ACTUAL coords — i.e. the value
        it was ranked/selected on (the central invariant of the refactor)."""
        K, num_wells = 3, 4
        geos, _ = _install_fake_surrogate(monkeypatch, tmp_path, K=K, num_wells=num_wells)
        cfg = _fake_cfg(geos, num_wells=num_wells, n_exploit=3, n_frontier=1, gens=5, popsize=8)
        res = acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=0)
        cands = res["candidates"]
        assert len(cands) == 4
        for c in cands:
            assert np.isclose(c.predicted_revenue, _expected_revenue(c.coords_xyz), rtol=1e-4, atol=1.0), (
                f"{c.snapshot_id}: reported {c.predicted_revenue} != accurate "
                f"{_expected_revenue(c.coords_xyz)} at its coords"
            )
            # per-geology predictions are the same accurate value (geology-indep fake)
            pbg = c.extras["well_config_paths_by_geology"]
            assert len(pbg) == K

    def test_graph_built_exactly_once_per_candidate(self, tmp_path, monkeypatch) -> None:
        """No rebuild at emit: _build_static_batch_for_starts is called only in
        the CMA loop (gens x K) plus the single frontier rebuild (1 x K)."""
        K, num_wells, gens = 3, 4, 5
        geos, build_calls = _install_fake_surrogate(monkeypatch, tmp_path, K=K, num_wells=num_wells)
        cfg = _fake_cfg(geos, num_wells=num_wells, n_exploit=3, n_frontier=1, gens=gens, popsize=8)
        acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=0)
        assert len(build_calls) == (gens + 1) * K  # +1 for the frontier forward

    def test_depth_varies_within_bounds(self, tmp_path, monkeypatch) -> None:
        geos, _ = _install_fake_surrogate(monkeypatch, tmp_path, K=2, num_wells=4)
        cfg = _fake_cfg(geos, num_wells=4, n_exploit=4, n_frontier=2, gens=6, popsize=10,
                        depth_min=2, depth_max=20)
        res = acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=0)
        zs = np.concatenate([c.coords_xyz[:, 2] for c in res["candidates"]])
        # z_hi capped at min(depth_max, z_max-1) = min(20, 29) = 20.
        assert zs.min() >= 2 and zs.max() <= 20
        assert zs.std() > 0.0, "depth should vary across wells/candidates"

    def test_n_frontier_zero_is_exploit_only(self, tmp_path, monkeypatch) -> None:
        geos, build_calls = _install_fake_surrogate(monkeypatch, tmp_path, K=2, num_wells=3)
        cfg = _fake_cfg(geos, num_wells=3, n_exploit=4, n_frontier=0, gens=4, popsize=8)
        res = acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=0)
        kinds = {c.kind for c in res["candidates"]}
        assert kinds == {"exploit"}
        assert len(res["candidates"]) == 4
        # No frontier rebuild → exactly gens x K builds.
        assert len(build_calls) == 4 * 2

    def test_pool_smaller_than_n_exploit_warns_and_degrades(self, tmp_path, monkeypatch, capsys) -> None:
        """Tiny budget: pool < n_exploit. Must warn and emit fewer, not crash."""
        geos, _ = _install_fake_surrogate(monkeypatch, tmp_path, K=2, num_wells=3)
        cfg = _fake_cfg(geos, num_wells=3, n_exploit=20, n_frontier=0, gens=1, popsize=4)
        res = acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=0)
        assert 0 < len(res["candidates"]) <= 4  # pool capped at popsize=4
        assert "WARNING" in capsys.readouterr().out

    def test_n_exploit_plus_frontier_zero_raises(self, tmp_path, monkeypatch) -> None:
        geos, _ = _install_fake_surrogate(monkeypatch, tmp_path, K=2, num_wells=3)
        cfg = _fake_cfg(geos, num_wells=3, n_exploit=0, n_frontier=0, gens=2, popsize=4)
        with pytest.raises(RuntimeError, match="n_exploit \\+ n_frontier"):
            acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=0)

    def test_manifest_and_files_written(self, tmp_path, monkeypatch) -> None:
        geos, _ = _install_fake_surrogate(monkeypatch, tmp_path, K=3, num_wells=4)
        cfg = _fake_cfg(geos, num_wells=4, n_exploit=3, n_frontier=1, gens=4, popsize=8)
        out = tmp_path / "acq"
        res = acq.run_acquisition(cfg, out_dir=out, iteration=0)
        manifest = res["manifest"]
        assert manifest["acquisition_optimizer"] == "cma_surrogate"
        assert manifest["snapshot_count"] == len(res["candidates"])
        # one .jl + one snapshot json per candidate
        assert len(list((out / "well_configs").glob("*.jl"))) == len(res["candidates"])
        assert len(list((out / "snapshots_json").glob("*.json"))) == len(res["candidates"])
        # snapshot ids unique across exploit + frontier
        ids = [c.snapshot_id for c in res["candidates"]]
        assert len(ids) == len(set(ids))


class TestDeadRockDrop:
    """A candidate whose graph fails to build in some geology (a well on dead
    rock at its CMA-chosen depth → extract_well_data drops it → RuntimeError)
    must be scored NaN and excluded, NOT crash the run (the bug the user hit)."""

    def test_build_failure_scored_nan_not_crashed(self, tmp_path, monkeypatch) -> None:
        geos, _ = _install_fake_surrogate(monkeypatch, tmp_path, K=2, num_wells=4)
        # Poison the first candidate the builder ever sees (by its well-0
        # signature) and raise whenever that candidate is in the batch — both the
        # full-batch build AND its per-candidate probe — exactly like a real
        # dead-rock drop. Every other candidate builds fine.
        state: dict = {"poison": None}

        def _sig(c):
            return (round(c[0]["x"]), round(c[0]["y"]), int(c[0]["depth"]))

        def _poison_build(cfgs, *a, **k):
            if state["poison"] is None:
                state["poison"] = _sig(cfgs[0])
            if any(_sig(c) == state["poison"] for c in cfgs):
                raise RuntimeError("simulated dead-rock drop (test)")
            return [0] * len(cfgs)

        monkeypatch.setattr(acq, "_build_static_batch_for_starts", _poison_build)
        cfg = _fake_cfg(geos, num_wells=4, n_exploit=3, n_frontier=1, gens=3, popsize=8)
        res = acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=0)
        # Completed without crashing; ≥1 drop recorded; all emitted preds finite.
        assert res["manifest"]["dead_rock_drops"] >= 1
        assert res["candidates"], "should still emit candidates after dropping the bad one"
        for c in res["candidates"]:
            assert np.isfinite(c.predicted_revenue)


class TestColdRestartGate:
    """The cma_warm_start gate — the IX-validated cold-restart default."""

    def test_cold_default_ignores_incumbent_at_iter1(self, tmp_path, monkeypatch) -> None:
        """cma_warm_start=False (default): even at iteration>=1 with prior_metrics
        present, CMA stays cold (mean_init=None) and candidates carry no seed source."""
        geos, _ = _install_fake_surrogate(monkeypatch, tmp_path, K=2, num_wells=4)
        captured = _capture_mean_init(monkeypatch)
        # If anything other than the flag were gating, a non-empty prior_metrics
        # would trigger seeding — so this isolates the gate.
        cfg = _fake_cfg(geos, num_wells=4, n_exploit=3, n_frontier=1, gens=4, popsize=8,
                        cma_warm_start=False,
                        prior_metrics=[tmp_path / "iter_0000" / "per_candidate_metrics.json"])
        res = acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=1)
        assert captured["mean_init"] is None, "cold restart must pass mean_init=None"
        for c in res["candidates"]:
            assert c.extras.get("seed_source_snapshot_id") is None

    def test_warm_start_seeds_from_incumbent_at_iter1(self, tmp_path, monkeypatch) -> None:
        """cma_warm_start=True: CMA mean is seeded from the incumbent and exploit
        candidates record the seed source."""
        geos, _ = _install_fake_surrogate(monkeypatch, tmp_path, K=2, num_wells=4)
        captured = _capture_mean_init(monkeypatch)
        coords = np.array([[10.0 + w, 12.0 + w, 15.0] for w in range(4)], dtype=np.float32)
        monkeypatch.setattr(acq, "_load_elite_seeds_ensemble",
                            lambda prior_metrics, k, rng: [(coords, "snapXYZ", 0)])
        cfg = _fake_cfg(geos, num_wells=4, n_exploit=3, n_frontier=1, gens=4, popsize=8,
                        cma_warm_start=True,
                        prior_metrics=[tmp_path / "iter_0000" / "per_candidate_metrics.json"])
        res = acq.run_acquisition(cfg, out_dir=tmp_path / "acq", iteration=1)
        mi = captured["mean_init"]
        assert mi is not None and np.asarray(mi).shape == (4 * 3,), "warm start must pass mean_init"
        exploit = [c for c in res["candidates"] if c.kind == "exploit"]
        assert exploit and all(c.extras.get("seed_source_snapshot_id") == "snapXYZ" for c in exploit)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
