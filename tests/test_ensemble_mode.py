"""Synthetic test suite for the new ensemble acquisition mode.

This suite exercises the four units that make up the ensemble path:

  * ``orchestrator/acquire.py``: ``_load_elite_seeds_ensemble``,
    ``_emit_snapshot_ensemble``, ``_snapshot_to_candidate_ensemble``,
    ``write_selected_manifest_ensemble``.
  * ``orchestrator/select.py``: ``select_batch_ensemble``.
  * ``orchestrator/ingest.py``: ``ingest_iteration`` (with the heavy I/O
    primitives — ``_run_preprocess_h5``, ``_merge_compiled_h5s``,
    ``_count_cases``, ``_read_real_revenue`` — monkeypatched to no-ops/dict
    lookups so we don't need IX outputs or H5 files).
  * ``orchestrator/state.py``: ``IterationRecord`` round-trip + the new
    ``best_emv_so_far_value`` helper.

The Adam-loop entry point ``_run_acquisition_ensemble`` itself is out of scope
here — it needs CUDA, a real surrogate checkpoint, and per-geology HDF5 files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from orchestrator import acquire as acq
from orchestrator import ingest as ing
from orchestrator.acquire import (
    GeologySpec,
    _emit_snapshot_ensemble,
    _load_elite_seeds_ensemble,
    _snapshot_to_candidate_ensemble,
    write_selected_manifest_ensemble,
)
from orchestrator.ingest import ingest_iteration
from orchestrator.select import Candidate, select_batch_ensemble
from orchestrator.state import IterationRecord, RunState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_snapshot_json(
    path: Path,
    snapshot_id: str,
    wells: list[dict] | None = None,
) -> None:
    """Write a minimal snapshots_json/<id>.json shaped like the acquisition writes."""
    if wells is None:
        wells = [
            {"well_id": 0, "type": "injector", "x": 10.0, "y": 20.0, "z": 5.0,
             "i_idx": 21, "j_idx": 11, "k_idx": 6, "rate": 8000.0},
            {"well_id": 1, "type": "producer", "x": 30.0, "y": 40.0, "z": 5.0,
             "i_idx": 41, "j_idx": 31, "k_idx": 6, "rate": -8000.0},
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"snapshot_id": snapshot_id, "wells": wells}, f)


def _write_metrics_json(
    metrics_path: Path,
    iteration: int,
    *,
    per_candidate_emv: dict[str, float] | None = None,
    candidates: list[dict] | None = None,
) -> None:
    payload: dict[str, Any] = {"iteration": iteration}
    if per_candidate_emv is not None:
        payload["per_candidate_emv"] = per_candidate_emv
    if candidates is not None:
        payload["candidates"] = candidates
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(payload, f)


def _make_geologies(k: int = 3) -> list[GeologySpec]:
    return [
        GeologySpec(
            geology_index=i,
            geology_h5_file=f"/tmp/geo_{i:02d}.h5",
            geology_name=f"geo_{i:02d}",
        )
        for i in range(k)
    ]


def _make_candidate(
    *,
    snapshot_id: str = "run000000_step0045_frontier",
    kind: str = "frontier",
    predicted_revenue: float = 1.0e8,
    geology_index: int = 0,
    extras: dict | None = None,
) -> Candidate:
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 3.0]], dtype=np.float32)
    return Candidate(
        geology_index=geology_index,
        geology_file=f"/tmp/geo_{geology_index:02d}.h5",
        geology_name=f"geo_{geology_index:02d}",
        geology_config_id=geology_index + 1,
        geology_scenario_name=f"scen_{geology_index}",
        geology_sample_num=geology_index,
        snapshot_id=snapshot_id,
        run_id=0,
        iteration=1,
        kind=kind,
        predicted_revenue=predicted_revenue,
        coords_xyz=coords,
        is_injector=[True, False],
        well_config_path=f"/tmp/wells/{snapshot_id}.jl",
        snapshot_json_path=f"/tmp/snap/{snapshot_id}.json",
        extras=extras or {},
    )


# ===========================================================================
# 1. _load_elite_seeds_ensemble — edge cases 1–11
# ===========================================================================


class TestLoadEliteSeedsEnsemble:
    def test_01_empty_prior_metrics(self) -> None:
        rng = np.random.default_rng(0)
        assert _load_elite_seeds_ensemble([], k=5, rng=rng) == []

    def test_02_nonexistent_metrics_file(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        bogus = tmp_path / "iter_0001" / "per_candidate_metrics.json"
        # do NOT create it
        assert _load_elite_seeds_ensemble([bogus], k=5, rng=rng) == []

    def test_03_malformed_json(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        p = tmp_path / "iter_0001" / "per_candidate_metrics.json"
        p.parent.mkdir(parents=True)
        p.write_text("{not valid json")
        # Should silently skip and return [] (no crash).
        assert _load_elite_seeds_ensemble([p], k=5, rng=rng) == []

    def test_04_uses_per_candidate_emv_when_present(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        iter_dir = tmp_path / "iter_0001"
        # snapshot JSON dir uses the local-hybrid layout (run_root/acquire/iter/snapshots_json).
        snap_dir = tmp_path / "acquire" / "iter_0001" / "snapshots_json"
        for sid, wells in [
            ("snap_a", [{"x": 0.0, "y": 0.0, "z": 0.0}]),
            ("snap_b", [{"x": 50.0, "y": 50.0, "z": 0.0}]),
            ("snap_c", [{"x": 100.0, "y": 100.0, "z": 0.0}]),
        ]:
            _write_snapshot_json(snap_dir / f"{sid}.json", sid, wells)
        _write_metrics_json(
            iter_dir / "per_candidate_metrics.json",
            iteration=1,
            per_candidate_emv={"snap_a": 1.0e8, "snap_b": 3.0e8, "snap_c": 2.0e8},
        )
        picks = _load_elite_seeds_ensemble(
            [iter_dir / "per_candidate_metrics.json"], k=3, rng=rng,
        )
        # 3 picks, ranked descending by EMV: snap_b, snap_c, snap_a — but the
        # function shuffles before truncation. So we just verify the *set* of
        # picked snapshot ids and the source iteration is correct.
        assert len(picks) == 3
        sids = {p[1] for p in picks}
        assert sids == {"snap_a", "snap_b", "snap_c"}
        assert all(p[2] == 1 for p in picks)
        # Coords have shape (n_wells, 3).
        for coords, _sid, _it in picks:
            assert coords.shape[1] == 3

    def test_05_derives_emv_from_candidates_rows_when_no_per_candidate_emv(
        self, tmp_path: Path,
    ) -> None:
        """Legacy per-geology metrics file: EMV must be derived by averaging
        real_revenue rows grouped by snapshot_id.
        """
        rng = np.random.default_rng(0)
        iter_dir = tmp_path / "iter_0002"
        snap_dir = tmp_path / "acquire" / "iter_0002" / "snapshots_json"
        for sid in ("snap_x", "snap_y"):
            _write_snapshot_json(
                snap_dir / f"{sid}.json", sid,
                wells=[{"x": (5.0 if sid == "snap_x" else 60.0), "y": 5.0, "z": 0.0}],
            )
        # 3 rows for snap_x (avg=2.0e8), 2 rows for snap_y (avg=5.0e8). snap_y wins.
        candidates = [
            {"snapshot_id": "snap_x", "real_revenue": 1.0e8, "geology_index": 0},
            {"snapshot_id": "snap_x", "real_revenue": 2.0e8, "geology_index": 1},
            {"snapshot_id": "snap_x", "real_revenue": 3.0e8, "geology_index": 2},
            {"snapshot_id": "snap_y", "real_revenue": 4.0e8, "geology_index": 0},
            {"snapshot_id": "snap_y", "real_revenue": 6.0e8, "geology_index": 1},
        ]
        _write_metrics_json(
            iter_dir / "per_candidate_metrics.json", iteration=2, candidates=candidates,
        )
        picks = _load_elite_seeds_ensemble(
            [iter_dir / "per_candidate_metrics.json"], k=2, rng=rng,
        )
        assert len(picks) == 2
        assert {p[1] for p in picks} == {"snap_x", "snap_y"}

    def test_06_snapshot_json_missing(self, tmp_path: Path) -> None:
        """A metrics entry whose snapshot JSON is missing from BOTH candidate
        directories must be silently skipped.
        """
        rng = np.random.default_rng(0)
        iter_dir = tmp_path / "iter_0003"
        # Intentionally do NOT write any snapshot json — but DO mention it in metrics.
        _write_metrics_json(
            iter_dir / "per_candidate_metrics.json",
            iteration=3,
            per_candidate_emv={"missing_snap": 9.9e8},
        )
        picks = _load_elite_seeds_ensemble(
            [iter_dir / "per_candidate_metrics.json"], k=5, rng=rng,
        )
        assert picks == []

    def test_07_snapshot_json_missing_wells(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        iter_dir = tmp_path / "iter_0004"
        snap_dir = tmp_path / "acquire" / "iter_0004" / "snapshots_json"
        snap_dir.mkdir(parents=True)
        # Write a snapshot JSON with no `wells` key.
        (snap_dir / "snap_nowells.json").write_text(json.dumps({"snapshot_id": "snap_nowells"}))
        _write_metrics_json(
            iter_dir / "per_candidate_metrics.json",
            iteration=4,
            per_candidate_emv={"snap_nowells": 1.0e8},
        )
        picks = _load_elite_seeds_ensemble(
            [iter_dir / "per_candidate_metrics.json"], k=3, rng=rng,
        )
        assert picks == []

    def test_08_min_distance_filter(self, tmp_path: Path) -> None:
        """Two near-identical snapshots → only the higher-EMV one survives."""
        rng = np.random.default_rng(0)
        iter_dir = tmp_path / "iter_0005"
        snap_dir = tmp_path / "acquire" / "iter_0005" / "snapshots_json"
        # snap_close_a and snap_close_b are within < 3 grid cells (L2 in flattened
        # coords). snap_far is well outside the radius.
        _write_snapshot_json(
            snap_dir / "snap_close_a.json", "snap_close_a",
            wells=[{"x": 10.0, "y": 10.0, "z": 5.0}, {"x": 30.0, "y": 30.0, "z": 5.0}],
        )
        _write_snapshot_json(
            snap_dir / "snap_close_b.json", "snap_close_b",
            wells=[{"x": 10.5, "y": 10.5, "z": 5.0}, {"x": 30.0, "y": 30.0, "z": 5.0}],
        )
        _write_snapshot_json(
            snap_dir / "snap_far.json", "snap_far",
            wells=[{"x": 80.0, "y": 80.0, "z": 5.0}, {"x": 90.0, "y": 90.0, "z": 5.0}],
        )
        # Highest EMV first → snap_close_a, then snap_close_b should be filtered,
        # then snap_far survives.
        _write_metrics_json(
            iter_dir / "per_candidate_metrics.json",
            iteration=5,
            per_candidate_emv={"snap_close_a": 3.0e8, "snap_close_b": 2.5e8, "snap_far": 1.0e8},
        )
        picks = _load_elite_seeds_ensemble(
            [iter_dir / "per_candidate_metrics.json"], k=5, rng=rng,
        )
        sids = {p[1] for p in picks}
        # snap_close_b must be filtered out.
        assert "snap_close_b" not in sids
        assert "snap_close_a" in sids
        assert "snap_far" in sids
        assert len(picks) == 2

    def test_09_k_larger_than_pool(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        iter_dir = tmp_path / "iter_0006"
        snap_dir = tmp_path / "acquire" / "iter_0006" / "snapshots_json"
        _write_snapshot_json(snap_dir / "only.json", "only", wells=[{"x": 1.0, "y": 1.0, "z": 1.0}])
        _write_metrics_json(
            iter_dir / "per_candidate_metrics.json", iteration=6,
            per_candidate_emv={"only": 1.0e8},
        )
        picks = _load_elite_seeds_ensemble(
            [iter_dir / "per_candidate_metrics.json"], k=50, rng=rng,
        )
        assert len(picks) == 1
        assert picks[0][1] == "only"

    def test_10_multi_iter_global_ranking(self, tmp_path: Path) -> None:
        """Picks must rank globally across iters; source_iteration is correct."""
        rng = np.random.default_rng(0)
        prior_metrics: list[Path] = []
        for it in (1, 2, 3):
            iter_dir = tmp_path / f"iter_{it:04d}"
            snap_dir = tmp_path / "acquire" / f"iter_{it:04d}" / "snapshots_json"
            sid = f"snap_it{it}"
            # Place all three snaps far apart so min-distance filter doesn't drop anything.
            _write_snapshot_json(
                snap_dir / f"{sid}.json", sid,
                wells=[{"x": float(10 + 30 * it), "y": float(10 + 30 * it), "z": 5.0}],
            )
            # Make iter 2 the top scorer, iter 1 second, iter 3 last.
            emv = {1: 2.0e8, 2: 5.0e8, 3: 1.0e8}[it]
            _write_metrics_json(
                iter_dir / "per_candidate_metrics.json", iteration=it,
                per_candidate_emv={sid: emv},
            )
            prior_metrics.append(iter_dir / "per_candidate_metrics.json")
        picks = _load_elite_seeds_ensemble(prior_metrics, k=3, rng=rng)
        assert len(picks) == 3
        sid_to_iter = {p[1]: p[2] for p in picks}
        assert sid_to_iter == {"snap_it1": 1, "snap_it2": 2, "snap_it3": 3}

    def test_11_non_finite_emv_skipped(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        iter_dir = tmp_path / "iter_0007"
        snap_dir = tmp_path / "acquire" / "iter_0007" / "snapshots_json"
        _write_snapshot_json(snap_dir / "good.json", "good",
                             wells=[{"x": 5.0, "y": 5.0, "z": 5.0}])
        _write_snapshot_json(snap_dir / "bad.json", "bad",
                             wells=[{"x": 50.0, "y": 50.0, "z": 5.0}])
        # Non-finite EMVs (NaN/inf) must be skipped.
        _write_metrics_json(
            iter_dir / "per_candidate_metrics.json", iteration=7,
            per_candidate_emv={"good": 1.0e8, "bad": float("nan")},
        )
        picks = _load_elite_seeds_ensemble(
            [iter_dir / "per_candidate_metrics.json"], k=5, rng=rng,
        )
        assert {p[1] for p in picks} == {"good"}


# ===========================================================================
# 2. select_batch_ensemble — edge cases 12–17
# ===========================================================================


class TestSelectBatchEnsemble:
    def test_12_empty_input(self) -> None:
        assert select_batch_ensemble([]) == []
        assert select_batch_ensemble([], batch_size=10) == []

    def test_13_exploit_only(self) -> None:
        cands = [
            _make_candidate(snapshot_id="e1", kind="exploit", predicted_revenue=3.0),
            _make_candidate(snapshot_id="e2", kind="exploit", predicted_revenue=1.0),
            _make_candidate(snapshot_id="e3", kind="exploit", predicted_revenue=5.0),
        ]
        out = select_batch_ensemble(cands)
        assert [c.snapshot_id for c in out] == ["e3", "e1", "e2"]

    def test_14_frontier_only(self) -> None:
        cands = [
            _make_candidate(snapshot_id="f1", kind="frontier", predicted_revenue=3.0),
            _make_candidate(snapshot_id="f2", kind="frontier", predicted_revenue=5.0),
            _make_candidate(snapshot_id="f3", kind="frontier", predicted_revenue=1.0),
        ]
        out = select_batch_ensemble(cands)
        assert [c.snapshot_id for c in out] == ["f2", "f1", "f3"]

    def test_15_mixed_pool_no_cap(self) -> None:
        cands = [
            _make_candidate(snapshot_id="f_low", kind="frontier", predicted_revenue=1.0),
            _make_candidate(snapshot_id="e_high", kind="exploit", predicted_revenue=2.0),
            _make_candidate(snapshot_id="f_high", kind="frontier", predicted_revenue=10.0),
            _make_candidate(snapshot_id="e_low", kind="exploit", predicted_revenue=1.5),
        ]
        out = select_batch_ensemble(cands)
        # exploit first (by EMV desc), then frontier (by EMV desc).
        assert [c.snapshot_id for c in out] == ["e_high", "e_low", "f_high", "f_low"]

    def test_16_mixed_pool_with_cap(self) -> None:
        cands = [
            _make_candidate(snapshot_id="e_top", kind="exploit", predicted_revenue=100.0),
            _make_candidate(snapshot_id="e_mid", kind="exploit", predicted_revenue=50.0),
            _make_candidate(snapshot_id="e_low", kind="exploit", predicted_revenue=10.0),
            _make_candidate(snapshot_id="f_top", kind="frontier", predicted_revenue=80.0),
            _make_candidate(snapshot_id="f_mid", kind="frontier", predicted_revenue=40.0),
            _make_candidate(snapshot_id="f_low", kind="frontier", predicted_revenue=5.0),
        ]
        out = select_batch_ensemble(cands, batch_size=3)
        # Top-3 by EMV across kinds → e_top(100), f_top(80), e_mid(50)
        # Re-sorted by kind: exploit first (e_top, e_mid), then frontier (f_top).
        assert [c.snapshot_id for c in out] == ["e_top", "e_mid", "f_top"]

    def test_17_batch_larger_than_pool(self) -> None:
        cands = [
            _make_candidate(snapshot_id="a", kind="exploit", predicted_revenue=5.0),
            _make_candidate(snapshot_id="b", kind="frontier", predicted_revenue=3.0),
        ]
        out = select_batch_ensemble(cands, batch_size=100)
        assert len(out) == 2
        assert [c.snapshot_id for c in out] == ["a", "b"]


# ===========================================================================
# 3. _emit_snapshot_ensemble + _snapshot_to_candidate_ensemble — 18–21
# ===========================================================================


def _make_predictions_by_geology(geologies: list[GeologySpec]) -> list[dict]:
    return [
        {
            "geology_index": g.geology_index,
            "geology_name": g.geology_name,
            "geology_file": g.geology_h5_file,
            "geology_config_id": g.geology_index + 1,
            "geology_scenario_name": f"scen_{g.geology_index}",
            "geology_sample_num": g.geology_index,
            "discounted_total_revenue": 1.0e8 * (g.geology_index + 1),
            "total_energy_production": 0.0,
        }
        for g in geologies
    ]


def _fake_to_julia_wells_text(**kwargs) -> str:
    """Stand-in for geothermal.active_learning_utils.to_julia_wells_text.

    The real implementation produces a Julia-syntax dictionary literal; for
    tests we just stringify the kwargs so we can recover them later.
    """
    return f"# fake jl text\n# kwargs keys: {sorted(kwargs.keys())}\n"


class TestEmitSnapshotEnsemble:
    def test_18_snapshot_record_has_required_fields(self, tmp_path: Path) -> None:
        geos = _make_geologies(3)
        geo_metas = {
            g.geology_index: {
                "geology_config_id": g.geology_index + 1,
                "scenario_name": f"scen_{g.geology_index}",
                "sample_num": g.geology_index,
            }
            for g in geos
        }
        coords = np.array([[10.0, 20.0, 5.0], [30.0, 40.0, 5.0]], dtype=np.float32)
        wcdir = tmp_path / "well_configs"; wcdir.mkdir()
        sjdir = tmp_path / "snapshots_json"; sjdir.mkdir()
        rec = _emit_snapshot_ensemble(
            run_id=7, iteration_step=45, kind="frontier",
            coords_xyz=coords, is_injector_list=[True, False],
            predictions_by_geology=_make_predictions_by_geology(geos),
            predicted_emv=2.0e8,
            seed_source_snapshot_id=None,
            seed_source_iteration=None,
            seed_predicted_emv=None,
            geologies=geos, geology_metas=geo_metas,
            well_configs_dir=wcdir, snapshots_json_dir=sjdir,
            to_julia_wells_text=_fake_to_julia_wells_text,
        )
        assert "snapshot_id" in rec
        assert rec["run_id"] == 7
        assert rec["iteration"] == 45
        assert rec["kind"] == "frontier"
        assert rec["mode"] == "ensemble"
        # well_config_path must point to a single .jl file.
        assert rec["well_config_path"].endswith(".jl")
        assert Path(rec["well_config_path"]).exists()
        # And well_config_paths_by_geology has K entries all pointing to that same .jl.
        wcpg = rec["well_config_paths_by_geology"]
        assert len(wcpg) == 3
        assert all(e["well_config_path"] == rec["well_config_path"] for e in wcpg)
        # Each entry carries the Julia consumer's required keys.
        for e in wcpg:
            assert {"geology_index", "geology_config_id", "well_config_path"} <= e.keys()

    def test_19_snapshot_json_has_predictions_by_geology(self, tmp_path: Path) -> None:
        geos = _make_geologies(3)
        geo_metas = {g.geology_index: {"geology_config_id": g.geology_index + 1,
                                       "scenario_name": "s", "sample_num": 0}
                     for g in geos}
        coords = np.array([[10.0, 20.0, 5.0], [30.0, 40.0, 5.0]], dtype=np.float32)
        wcdir = tmp_path / "well_configs"; wcdir.mkdir()
        sjdir = tmp_path / "snapshots_json"; sjdir.mkdir()
        pbg = _make_predictions_by_geology(geos)
        rec = _emit_snapshot_ensemble(
            run_id=0, iteration_step=10, kind="exploit",
            coords_xyz=coords, is_injector_list=[True, False],
            predictions_by_geology=pbg,
            predicted_emv=3.0e8,
            seed_source_snapshot_id="prev_a",
            seed_source_iteration=2,
            seed_predicted_emv=1.0e8,
            geologies=geos, geology_metas=geo_metas,
            well_configs_dir=wcdir, snapshots_json_dir=sjdir,
            to_julia_wells_text=_fake_to_julia_wells_text,
        )
        with open(rec["json_path"], "r") as f:
            on_disk = json.load(f)
        assert "predictions_by_geology" in on_disk
        assert len(on_disk["predictions_by_geology"]) == 3
        for entry in on_disk["predictions_by_geology"]:
            # Julia consumer expects these two keys at minimum.
            assert "geology_index" in entry
            assert "discounted_total_revenue" in entry
        assert on_disk["mode"] == "ensemble"

    def test_20_seed_metadata_roundtrip(self, tmp_path: Path) -> None:
        geos = _make_geologies(2)
        geo_metas = {g.geology_index: {"geology_config_id": g.geology_index + 1,
                                       "scenario_name": "s", "sample_num": 0}
                     for g in geos}
        coords = np.array([[10.0, 20.0, 5.0]], dtype=np.float32)
        wcdir = tmp_path / "well_configs"; wcdir.mkdir()
        sjdir = tmp_path / "snapshots_json"; sjdir.mkdir()
        rec = _emit_snapshot_ensemble(
            run_id=3, iteration_step=12, kind="exploit",
            coords_xyz=coords, is_injector_list=[True],
            predictions_by_geology=_make_predictions_by_geology(geos),
            predicted_emv=4.2e8,
            seed_source_snapshot_id="seed_xyz",
            seed_source_iteration=5,
            seed_predicted_emv=2.1e8,
            geologies=geos, geology_metas=geo_metas,
            well_configs_dir=wcdir, snapshots_json_dir=sjdir,
            to_julia_wells_text=_fake_to_julia_wells_text,
        )
        # In the record
        assert rec["seed_source_snapshot_id"] == "seed_xyz"
        assert rec["seed_source_iteration"] == 5
        assert math.isclose(rec["seed_predicted_emv"], 2.1e8)
        # And on disk
        with open(rec["json_path"], "r") as f:
            on_disk = json.load(f)
        assert on_disk["seed_source_snapshot_id"] == "seed_xyz"
        assert on_disk["seed_source_iteration"] == 5
        assert math.isclose(on_disk["seed_predicted_emv"], 2.1e8)
        # terminal_predicted_emv = predicted_emv
        assert math.isclose(on_disk["terminal_predicted_emv"], 4.2e8)

    def test_21_to_julia_wells_text_receives_correct_kwargs(self, tmp_path: Path) -> None:
        geos = _make_geologies(2)
        geo_metas = {
            0: {"geology_config_id": 1, "scenario_name": "scenA", "sample_num": 7},
            1: {"geology_config_id": 2, "scenario_name": "scenB", "sample_num": 8},
        }
        coords = np.array([[10.0, 20.0, 5.0]], dtype=np.float32)
        wcdir = tmp_path / "well_configs"; wcdir.mkdir()
        sjdir = tmp_path / "snapshots_json"; sjdir.mkdir()
        captured: dict[str, Any] = {}

        def recorder(**kwargs) -> str:
            captured.update(kwargs)
            return "# fake\n"

        _emit_snapshot_ensemble(
            run_id=0, iteration_step=1, kind="frontier",
            coords_xyz=coords, is_injector_list=[True],
            predictions_by_geology=_make_predictions_by_geology(geos),
            predicted_emv=9.9e8,
            seed_source_snapshot_id=None,
            seed_source_iteration=None,
            seed_predicted_emv=None,
            geologies=geos, geology_metas=geo_metas,
            well_configs_dir=wcdir, snapshots_json_dir=sjdir,
            to_julia_wells_text=recorder,
        )
        assert "coords_xyz" in captured
        assert "is_injector_list" in captured
        assert "score" in captured and math.isclose(captured["score"], 9.9e8)
        assert "predicted_discounted_revenue" in captured
        # Uses the FIRST geology's metadata for the .jl header (by design).
        assert captured["geology_config_id"] == 1
        assert captured["geology_scenario_name"] == "scenA"
        assert captured["geology_sample_num"] == 7


# ===========================================================================
# 4. write_selected_manifest_ensemble — 22–24
# ===========================================================================


class TestWriteSelectedManifestEnsemble:
    def test_22_top_level_schema(self, tmp_path: Path) -> None:
        geos = _make_geologies(2)
        wcpg = [
            {"geology_index": 0, "geology_config_id": 1, "well_config_path": "/tmp/a.jl"},
            {"geology_index": 1, "geology_config_id": 2, "well_config_path": "/tmp/a.jl"},
        ]
        c = _make_candidate(
            snapshot_id="cand_a", kind="exploit", predicted_revenue=3.0e8,
            extras={"well_config_paths_by_geology": wcpg, "predicted_emv": 3.0e8},
        )
        out = tmp_path / "manifest.json"
        write_selected_manifest_ensemble(
            [c], out_path=out, iteration=4, geologies=geos,
            extras={"selection": {"batch_size": 1}},
        )
        with open(out, "r") as f:
            m = json.load(f)
        assert m["mode"] == "ensemble"
        assert isinstance(m["snapshots"], list)
        assert isinstance(m["geology_metadata"], list)
        assert m["selection"] == {"batch_size": 1}
        assert m["iteration"] == 4
        assert m["snapshot_count"] == 1

    def test_23_per_snapshot_paths_by_geology(self, tmp_path: Path) -> None:
        geos = _make_geologies(3)
        wcpg = [
            {"geology_index": i, "geology_config_id": i + 1,
             "well_config_path": "/tmp/shared.jl"}
            for i in range(3)
        ]
        c = _make_candidate(
            snapshot_id="cand_b", extras={"well_config_paths_by_geology": wcpg,
                                          "predicted_emv": 4.0e8},
        )
        out = tmp_path / "manifest.json"
        write_selected_manifest_ensemble([c], out_path=out, iteration=1, geologies=geos)
        with open(out, "r") as f:
            m = json.load(f)
        snap = m["snapshots"][0]
        assert len(snap["well_config_paths_by_geology"]) == 3
        for e in snap["well_config_paths_by_geology"]:
            assert {"geology_index", "geology_config_id", "well_config_path"} <= e.keys()

    def test_24_fallback_when_extras_missing(self, tmp_path: Path) -> None:
        """A candidate without extras['well_config_paths_by_geology'] should
        fall back to a 1-element array using the candidate's recorded geology.
        """
        geos = _make_geologies(3)
        # extras is empty — simulate the "extras stripped after IPC" failure mode.
        c = _make_candidate(snapshot_id="cand_c", geology_index=1, extras={})
        out = tmp_path / "manifest.json"
        write_selected_manifest_ensemble([c], out_path=out, iteration=2, geologies=geos)
        with open(out, "r") as f:
            m = json.load(f)
        snap = m["snapshots"][0]
        wcpg = snap["well_config_paths_by_geology"]
        # Exactly one fallback entry, pointing at the candidate's geology.
        assert len(wcpg) == 1
        assert wcpg[0]["geology_index"] == 1
        assert wcpg[0]["well_config_path"] == c.well_config_path


# Also worth checking _snapshot_to_candidate_ensemble keeps the K-array in extras.
def test_snapshot_to_candidate_ensemble_preserves_extras() -> None:
    snap = {
        "snapshot_id": "s",
        "run_id": 1,
        "iteration": 2,
        "kind": "exploit",
        "mode": "ensemble",
        "well_config_path": "/tmp/s.jl",
        "json_path": "/tmp/s.json",
        "well_config_paths_by_geology": [
            {"geology_index": 0, "geology_name": "g0", "geology_file": "/tmp/g0.h5",
             "geology_config_id": 1, "well_config_path": "/tmp/s.jl"},
            {"geology_index": 1, "geology_name": "g1", "geology_file": "/tmp/g1.h5",
             "geology_config_id": 2, "well_config_path": "/tmp/s.jl"},
        ],
        "predicted_emv": 7.0e8,
        "predicted_discounted_total_revenue": 7.0e8,
        "seed_source_snapshot_id": "prev",
        "seed_source_iteration": 1,
        "seed_predicted_emv": 5.0e8,
        "terminal_predicted_emv": 7.0e8,
    }
    cand = _snapshot_to_candidate_ensemble(
        snap, coords_xyz=np.zeros((1, 3), dtype=np.float32), is_injector_list=[True],
    )
    assert cand.extras["mode"] == "ensemble"
    assert cand.extras["well_config_paths_by_geology"] == snap["well_config_paths_by_geology"]
    assert math.isclose(cand.extras["predicted_emv"], 7.0e8)
    assert cand.extras["seed_source_snapshot_id"] == "prev"
    assert cand.extras["seed_source_iteration"] == 1
    assert math.isclose(cand.extras["seed_predicted_emv"], 5.0e8)


# ===========================================================================
# 5. ingest_iteration (monkeypatched I/O) — 25–29
# ===========================================================================


def _make_tasks_payload(
    tasks: list[dict],
) -> dict:
    return {"tasks": tasks}


def _setup_ingest_environment(
    tmp_path: Path,
    *,
    tasks: list[dict],
    snapshot_jsons: dict[str, dict],
    real_revs: dict[str, float],
    monkeypatch,
) -> dict[str, Path]:
    """Wire up the directory layout ingest_iteration expects and monkeypatch
    the heavy I/O primitives. Returns the relevant paths.
    """
    # Tasks JSON: ingest expects it at <stage_run_dir>/array/tasks.json, and
    # derives the status dir as <array_tasks_json>.parent.parent / "status".
    stage_dir = tmp_path / "stage_run"
    array_dir = stage_dir / "array"
    array_dir.mkdir(parents=True)
    status_dir = stage_dir / "status"
    status_dir.mkdir()
    tasks_json = array_dir / "tasks.json"
    with open(tasks_json, "w") as f:
        json.dump(_make_tasks_payload(tasks), f)

    # Snapshot JSONs: place them at wherever each task's snapshot_json_path points.
    for task in tasks:
        sj = task.get("snapshot_json_path", "")
        if sj and sj in snapshot_jsons:
            Path(sj).parent.mkdir(parents=True, exist_ok=True)
            with open(sj, "w") as f:
                json.dump(snapshot_jsons[sj], f)

    # IX output dir: drop empty placeholder files so ``produced.exists()`` is True
    # for every task whose output we want to mark as completed. Tests can override
    # by removing specific files to simulate missing-H5 modes.
    ix_output_dir = tmp_path / "ix_outputs"
    ix_output_dir.mkdir()
    for task in tasks:
        out_name = task.get("output_file_name", "")
        if out_name and task.get("_h5_present", True):
            (ix_output_dir / out_name).write_bytes(b"")

    # Write status JSONs so missing-h5 cases don't get charged to MISSING_SILENTLY.
    for task in tasks:
        tid = task.get("task_id")
        if tid is None:
            continue
        status_payload = task.get("_status", {"success": True})
        with open(status_dir / f"task_{int(tid):06d}.json", "w") as f:
            json.dump(status_payload, f)

    raw_ix_archive = tmp_path / "raw_ix_archive"
    delta_h5 = tmp_path / "delta.h5"
    prior_h5 = tmp_path / "prior.h5"
    next_h5 = tmp_path / "next.h5"

    # Monkeypatch the heavy I/O. Use the exact patterns the user used previously.
    monkeypatch.setattr(ing, "_run_preprocess_h5", lambda **kw: None)
    monkeypatch.setattr(ing, "_merge_compiled_h5s", lambda *a, **kw: None)
    monkeypatch.setattr(ing, "_count_cases", lambda h5: 12345)
    monkeypatch.setattr(ing, "_read_real_revenue", lambda h5, case_id: real_revs.get(case_id))
    # And the symlink helper that touches the real filesystem (we don't care
    # about archive results in the test).
    monkeypatch.setattr(ing, "_link_ix_outputs_to_archive", lambda src, dst: [])

    return {
        "array_tasks_json": tasks_json,
        "ix_output_dir": ix_output_dir,
        "raw_ix_archive": raw_ix_archive,
        "delta_h5": delta_h5,
        "prior_compiled_h5": prior_h5,
        "next_compiled_h5": next_h5,
    }


class TestIngestIteration:
    def test_25_per_geology_back_compat(self, tmp_path: Path, monkeypatch) -> None:
        """One row per snapshot → ensemble aggregates are None."""
        snap_dir = tmp_path / "snaps"; snap_dir.mkdir()
        tasks = []
        snap_jsons: dict[str, dict] = {}
        real_revs: dict[str, float] = {}
        # 3 snapshots × 1 geo each.
        for i in range(3):
            sid = f"geo{i:02d}_run{i:06d}_step0045_frontier"
            sj_path = str(snap_dir / f"{sid}.json")
            out_name = f"v2.5_al_{i+1}_run{i:06d}_iter01.h5"
            case_id = Path(out_name).stem
            tasks.append({
                "task_id": i, "snapshot_id": sid, "output_file_name": out_name,
                "snapshot_json_path": sj_path, "geology_index": i,
                "scenario": i + 1, "predicted_discounted_total_revenue": 1.0e8 + i,
            })
            snap_jsons[sj_path] = {"snapshot_id": sid, "kind": "frontier"}
            real_revs[case_id] = 0.9e8 + i

        paths = _setup_ingest_environment(
            tmp_path, tasks=tasks, snapshot_jsons=snap_jsons, real_revs=real_revs,
            monkeypatch=monkeypatch,
        )
        m = ingest_iteration(
            iteration=1, surrogate_repo=tmp_path, norm_config_path=tmp_path / "nc.json",
            economics_config=None, log_path=None, **paths,
        )
        assert m.n_submitted == 3
        assert m.n_completed == 3
        assert m.per_candidate_emv is None
        assert m.best_emv_in_batch is None
        assert m.exploit_best_emv is None

    def test_26_ensemble_happy_path(self, tmp_path: Path, monkeypatch) -> None:
        """2 snapshots × 3 geos = 6 tasks. per_candidate_emv = mean of 3 reals."""
        snap_dir = tmp_path / "snaps"; snap_dir.mkdir()
        snap_jsons: dict[str, dict] = {}
        real_revs: dict[str, float] = {}
        tasks: list[dict] = []
        # snap A: reals 1, 2, 3 → mean 2.0; snap B: reals 4, 5, 6 → mean 5.0.
        snap_specs = [
            ("snap_A", "exploit", [1.0e8, 2.0e8, 3.0e8]),
            ("snap_B", "frontier", [4.0e8, 5.0e8, 6.0e8]),
        ]
        tid = 0
        for sid, kind, reals in snap_specs:
            sj_path = str(snap_dir / f"{sid}.json")
            snap_jsons[sj_path] = {"snapshot_id": sid, "kind": kind,
                                   "predicted_emv": 5.5e8, "terminal_predicted_emv": 5.5e8}
            for gi, real in enumerate(reals):
                out_name = f"v2.5_al_{gi+1}_run{tid:06d}_iter02.h5"
                case_id = Path(out_name).stem
                tasks.append({
                    "task_id": tid, "snapshot_id": sid, "output_file_name": out_name,
                    "snapshot_json_path": sj_path, "geology_index": gi, "scenario": gi + 1,
                    "predicted_discounted_total_revenue": 5.5e8,
                })
                real_revs[case_id] = real
                tid += 1
        paths = _setup_ingest_environment(
            tmp_path, tasks=tasks, snapshot_jsons=snap_jsons, real_revs=real_revs,
            monkeypatch=monkeypatch,
        )
        m = ingest_iteration(
            iteration=2, surrogate_repo=tmp_path, norm_config_path=tmp_path / "nc.json",
            economics_config=None, log_path=None,
            prior_best_emv=1.0e8,
            **paths,
        )
        assert m.n_submitted == 6
        assert m.n_completed == 6
        assert m.per_candidate_emv is not None
        assert math.isclose(m.per_candidate_emv["snap_A"], 2.0e8)
        assert math.isclose(m.per_candidate_emv["snap_B"], 5.0e8)
        assert math.isclose(m.best_emv_in_batch, 5.0e8)
        # best_emv_so_far = max(prior_best_emv=1e8, best_in_batch=5e8) = 5e8.
        assert math.isclose(m.best_emv_so_far, 5.0e8)

    def test_27_partial_ix_failures(self, tmp_path: Path, monkeypatch) -> None:
        """Snap has 3 IX runs submitted, but only 2 H5 files present → EMV
        is computed over the 2 available reals.
        """
        snap_dir = tmp_path / "snaps"; snap_dir.mkdir()
        snap_jsons: dict[str, dict] = {}
        real_revs: dict[str, float] = {}
        tasks: list[dict] = []
        sid = "snap_partial"
        sj_path = str(snap_dir / f"{sid}.json")
        snap_jsons[sj_path] = {"snapshot_id": sid, "kind": "exploit",
                               "predicted_emv": 1.0e8, "terminal_predicted_emv": 1.0e8}
        reals = [2.0e8, 4.0e8, None]  # third one missing
        for gi, real in enumerate(reals):
            out_name = f"v2.5_al_{gi+1}_run000000_iter03.h5"
            case_id = Path(out_name).stem
            task: dict = {
                "task_id": gi, "snapshot_id": sid, "output_file_name": out_name,
                "snapshot_json_path": sj_path, "geology_index": gi, "scenario": gi + 1,
                "predicted_discounted_total_revenue": 1.0e8,
            }
            if real is None:
                task["_h5_present"] = False
                # Mark its status as a real worker failure so the ingest pipeline
                # routes it through n_failed_per_status (not MISSING_SILENTLY).
                task["_status"] = {"success": False, "phase": "ix", "error": "ix crashed"}
            else:
                real_revs[case_id] = real
            tasks.append(task)
        paths = _setup_ingest_environment(
            tmp_path, tasks=tasks, snapshot_jsons=snap_jsons, real_revs=real_revs,
            monkeypatch=monkeypatch,
        )
        m = ingest_iteration(
            iteration=3, surrogate_repo=tmp_path, norm_config_path=tmp_path / "nc.json",
            economics_config=None, log_path=None, **paths,
        )
        assert m.n_submitted == 3
        assert m.n_completed == 2
        assert m.n_failed_per_status == 1
        assert m.per_candidate_emv is not None
        # Mean of the 2 available reals = (2e8 + 4e8) / 2 = 3e8.
        assert math.isclose(m.per_candidate_emv[sid], 3.0e8)

    def test_28_seed_real_emv_lookup(self, tmp_path: Path, monkeypatch) -> None:
        """A candidate with seed_source_snapshot_id="X" and source_iter=2
        gets seed_real_emv = prior_per_candidate_emv_by_iter[2]["X"]."""
        snap_dir = tmp_path / "snaps"; snap_dir.mkdir()
        snap_jsons: dict[str, dict] = {}
        real_revs: dict[str, float] = {}
        tasks: list[dict] = []
        # snap_now seeded from snap_X (iter=2). prior map has it.
        # snap_orphan seeded from snap_Z (iter=99). prior map doesn't have iter 99.
        for sid, seed_id, seed_iter in [("snap_now", "snap_X", 2), ("snap_orphan", "snap_Z", 99)]:
            sj_path = str(snap_dir / f"{sid}.json")
            snap_jsons[sj_path] = {
                "snapshot_id": sid, "kind": "exploit",
                "seed_source_snapshot_id": seed_id,
                "seed_source_iteration": seed_iter,
                "seed_predicted_emv": 1.0e8,
                "predicted_emv": 2.0e8,
                "terminal_predicted_emv": 2.0e8,
            }
        tid = 0
        for sid in ("snap_now", "snap_orphan"):
            sj_path = str(snap_dir / f"{sid}.json")
            for gi in range(2):
                out_name = f"v2.5_al_{gi+1}_run{tid:06d}_iter04.h5"
                case_id = Path(out_name).stem
                tasks.append({
                    "task_id": tid, "snapshot_id": sid, "output_file_name": out_name,
                    "snapshot_json_path": sj_path, "geology_index": gi, "scenario": gi + 1,
                    "predicted_discounted_total_revenue": 2.0e8,
                })
                real_revs[case_id] = 2.0e8
                tid += 1
        paths = _setup_ingest_environment(
            tmp_path, tasks=tasks, snapshot_jsons=snap_jsons, real_revs=real_revs,
            monkeypatch=monkeypatch,
        )
        m = ingest_iteration(
            iteration=4, surrogate_repo=tmp_path, norm_config_path=tmp_path / "nc.json",
            economics_config=None, log_path=None,
            prior_per_candidate_emv_by_iter={2: {"snap_X": 7.7e8}},
            **paths,
        )
        # Find the rows we care about.
        for_now = [c for c in m.candidates or [] if c.snapshot_id == "snap_now"]
        for_orphan = [c for c in m.candidates or [] if c.snapshot_id == "snap_orphan"]
        assert for_now and all(math.isclose(c.seed_real_emv, 7.7e8) for c in for_now)
        # Missing iter in the prior map → seed_real_emv stays None.
        assert for_orphan and all(c.seed_real_emv is None for c in for_orphan)

    def test_29_exploit_rollups(self, tmp_path: Path, monkeypatch) -> None:
        """exploit_best_emv and exploit_best_per_geology populated correctly
        even when frontier candidates exist alongside.
        """
        snap_dir = tmp_path / "snaps"; snap_dir.mkdir()
        snap_jsons: dict[str, dict] = {}
        real_revs: dict[str, float] = {}
        tasks: list[dict] = []
        # 2 exploit snapshots, 1 frontier — all run on K=2 geos.
        snap_specs = [
            ("snap_E1", "exploit",   [1.0e8, 2.0e8]),  # EMV = 1.5e8
            ("snap_E2", "exploit",   [4.0e8, 6.0e8]),  # EMV = 5.0e8 ← top exploit
            ("snap_F1", "frontier",  [8.0e8, 9.0e8]),  # EMV = 8.5e8 (top overall)
        ]
        tid = 0
        for sid, kind, reals in snap_specs:
            sj_path = str(snap_dir / f"{sid}.json")
            snap_jsons[sj_path] = {"snapshot_id": sid, "kind": kind,
                                   "predicted_emv": 5.0e8, "terminal_predicted_emv": 5.0e8}
            for gi, real in enumerate(reals):
                out_name = f"v2.5_al_{gi+1}_run{tid:06d}_iter05.h5"
                case_id = Path(out_name).stem
                tasks.append({
                    "task_id": tid, "snapshot_id": sid, "output_file_name": out_name,
                    "snapshot_json_path": sj_path, "geology_index": gi, "scenario": gi + 1,
                    "predicted_discounted_total_revenue": 5.0e8,
                })
                real_revs[case_id] = real
                tid += 1
        paths = _setup_ingest_environment(
            tmp_path, tasks=tasks, snapshot_jsons=snap_jsons, real_revs=real_revs,
            monkeypatch=monkeypatch,
        )
        m = ingest_iteration(
            iteration=5, surrogate_repo=tmp_path, norm_config_path=tmp_path / "nc.json",
            economics_config=None, log_path=None, **paths,
        )
        # Best across ALL kinds is snap_F1; best EXPLOIT is snap_E2.
        assert m.per_candidate_emv is not None
        assert math.isclose(m.best_emv_in_batch, 8.5e8)
        assert math.isclose(m.exploit_best_emv, 5.0e8)
        # Per-geology best across the exploit cohort: geo 0 → max(1e8, 4e8) = 4e8;
        # geo 1 → max(2e8, 6e8) = 6e8.
        assert m.exploit_best_per_geology is not None
        assert math.isclose(m.exploit_best_per_geology["0"], 4.0e8)
        assert math.isclose(m.exploit_best_per_geology["1"], 6.0e8)


# ===========================================================================
# 6. State — 30, 31
# ===========================================================================


class TestState:
    def test_30_load_existing_state_mirror(self) -> None:
        path = Path("/home/rwu4/omv_geothermal/geothermal_active_learning/local_workspace_5_18/state_mirror.json")
        assert path.exists(), f"prerequisite file not found: {path}"
        state = RunState.load(path)
        # Should load without errors. New fields default to None for legacy rows.
        assert state.history, "expected non-empty history"
        for rec in state.history:
            assert hasattr(rec, "best_emv_in_batch")
            assert hasattr(rec, "best_emv_so_far")
            assert hasattr(rec, "per_candidate_emv")
            assert hasattr(rec, "exploit_best_emv")
            assert hasattr(rec, "exploit_best_per_geology")

    def test_31_best_emv_so_far_over_mixed_history(self) -> None:
        s = RunState(run_id="r", config_path="c", wandb_run_id="w")
        # Mix of None and finite values.
        s.upsert_iter(IterationRecord(iteration=0, best_emv_in_batch=None))
        s.upsert_iter(IterationRecord(iteration=1, best_emv_in_batch=2.0e8))
        s.upsert_iter(IterationRecord(iteration=2, best_emv_in_batch=1.0e8))
        s.upsert_iter(IterationRecord(iteration=3, best_emv_in_batch=None))
        s.upsert_iter(IterationRecord(iteration=4, best_emv_in_batch=5.0e8))
        assert math.isclose(s.best_emv_so_far_value(), 5.0e8)

    def test_31b_best_emv_all_none(self) -> None:
        s = RunState(run_id="r", config_path="c", wandb_run_id="w")
        s.upsert_iter(IterationRecord(iteration=0, best_emv_in_batch=None))
        s.upsert_iter(IterationRecord(iteration=1, best_emv_in_batch=None))
        assert s.best_emv_so_far_value() is None

    def test_31c_best_emv_empty_history(self) -> None:
        s = RunState(run_id="r", config_path="c", wandb_run_id="w")
        assert s.best_emv_so_far_value() is None


# Make this file runnable directly too, for the plain-assertion fallback path.
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
