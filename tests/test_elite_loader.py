"""Tests for ``orchestrator.acquire._load_elite_seeds``.

Builds a synthetic ``per_candidate_metrics.json`` + matching
``acquire/iter_NNNN/snapshots_json/*.json`` layout under ``tmp_path`` and
exercises the pure-Python coord-extraction + min-distance filter.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orchestrator.acquire import _load_elite_seeds


def _write_metrics_and_snapshots(
    run_root: Path,
    iter_idx: int,
    candidates: list[dict],
    snapshots: dict[str, dict] | None = None,
) -> Path:
    """Write a per-iteration metrics JSON and matching snapshots dir.

    ``candidates`` is a list of dicts with keys at least:
        geology_index, real_revenue, snapshot_id
    ``snapshots`` maps snapshot_id → dict written under
    ``run_root/acquire/iter_NNNN/snapshots_json/<id>.json``. Each snapshot
    dict needs at least a ``wells`` key with ``[{x,y,z}, ...]``.
    Returns the metrics path.
    """
    iter_name = f"iter_{iter_idx:04d}"
    iter_dir = run_root / iter_name
    iter_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir = run_root / "acquire" / iter_name / "snapshots_json"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = iter_dir / "per_candidate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"candidates": candidates}, f)

    for snap_id, payload in (snapshots or {}).items():
        with open(snaps_dir / f"{snap_id}.json", "w") as f:
            json.dump(payload, f)

    return metrics_path


def _well(x: float, y: float, z: float) -> dict:
    return {"well_id": 0, "type": "injector", "x": x, "y": y, "z": z}


def _make_snapshot(wells_xyz: list[tuple[float, float, float]]) -> dict:
    return {"wells": [_well(x, y, z) for (x, y, z) in wells_xyz]}


def _write_metrics_and_snapshots_sherlock(
    run_root: Path,
    iter_idx: int,
    candidates: list[dict],
    snapshots: dict[str, dict] | None = None,
) -> Path:
    """Sherlock-style layout used by orchestrator/paths.py:

    ``<run_root>/iter_NNNN/per_candidate_metrics.json``
    ``<run_root>/iter_NNNN/acquire/snapshots_json/<id>.json``

    Distinct from the local-hybrid layout above (acquire/iter_NNNN/...) so the
    loader has to probe both. Regression coverage for the B1 audit finding.
    """
    iter_dir = run_root / f"iter_{iter_idx:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    snaps_dir = iter_dir / "acquire" / "snapshots_json"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = iter_dir / "per_candidate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"candidates": candidates}, f)

    for snap_id, payload in (snapshots or {}).items():
        with open(snaps_dir / f"{snap_id}.json", "w") as f:
            json.dump(payload, f)

    return metrics_path


def test_elite_loader_handles_sherlock_layout(tmp_path: Path) -> None:
    """Regression: the loader must probe both candidate snapshot dirs.

    On Sherlock the snapshots_json dir is nested under each iter
    (``iter_NNNN/acquire/snapshots_json``), not under a top-level
    ``acquire/iter_NNNN/`` directory like the local-hybrid driver uses.
    """
    candidates = [
        {"geology_index": 0, "real_revenue": 1.0e14, "snapshot_id": "snapSh"},
    ]
    snapshots = {
        "snapSh": _make_snapshot([(10.0, 10.0, 5.0), (20.0, 20.0, 5.0),
                                  (30.0, 30.0, 5.0), (40.0, 40.0, 5.0)]),
    }
    metrics_path = _write_metrics_and_snapshots_sherlock(
        tmp_path, 0, candidates, snapshots
    )
    rng = np.random.default_rng(0)
    out = _load_elite_seeds(
        geology_index=0,
        prior_metrics=[metrics_path],
        k=4,
        rng=rng,
    )
    assert len(out) == 1, "Sherlock-layout snapshots_json was not discovered"
    assert out[0].shape == (4, 3)


def test_elite_loader_returns_correct_shape(tmp_path: Path) -> None:
    candidates = [
        {"geology_index": 0, "real_revenue": 1.0e14, "snapshot_id": "snapA"},
    ]
    snapshots = {
        "snapA": _make_snapshot([(10.0, 10.0, 5.0), (20.0, 20.0, 5.0),
                                  (30.0, 30.0, 5.0), (40.0, 40.0, 5.0)]),
    }
    metrics_path = _write_metrics_and_snapshots(tmp_path, 0, candidates, snapshots)

    rng = np.random.default_rng(0)
    out = _load_elite_seeds(
        geology_index=0,
        prior_metrics=[metrics_path],
        k=4,
        rng=rng,
    )
    assert len(out) == 1
    coords = out[0]
    assert coords.shape == (4, 3)
    assert coords.dtype == np.float32


def test_elite_loader_ranks_by_real_revenue(tmp_path: Path) -> None:
    # 5 candidates, all geo 0, with revenues [100, 500, 200, 800, 300] and
    # distinct coords so the min-distance filter does NOT drop any of them.
    revenues = [100.0, 500.0, 200.0, 800.0, 300.0]
    candidates = []
    snapshots = {}
    for i, rev in enumerate(revenues):
        snap_id = f"snap{i}"
        candidates.append({"geology_index": 0, "real_revenue": rev, "snapshot_id": snap_id})
        # Coords are widely separated so distance filter is a no-op.
        offset = float(i) * 50.0
        snapshots[snap_id] = _make_snapshot([
            (offset + 0.0, offset + 0.0, 5.0),
            (offset + 5.0, offset + 0.0, 5.0),
            (offset + 0.0, offset + 5.0, 5.0),
            (offset + 5.0, offset + 5.0, 5.0),
        ])
    metrics_path = _write_metrics_and_snapshots(tmp_path, 0, candidates, snapshots)

    # Recover revenues by matching coords back via the first well's x value.
    # Order is shuffled, so just look at the set of top-3.
    rng = np.random.default_rng(123)
    out = _load_elite_seeds(
        geology_index=0,
        prior_metrics=[metrics_path],
        k=3,
        rng=rng,
    )
    assert len(out) == 3
    # First-well x → revenue lookup.
    x_to_rev = {}
    for i, rev in enumerate(revenues):
        x_to_rev[float(i) * 50.0] = rev
    chosen_revs = {x_to_rev[float(c[0, 0])] for c in out}
    assert chosen_revs == {800.0, 500.0, 300.0}


def test_elite_loader_min_distance_filter(tmp_path: Path) -> None:
    # Build 4 candidates: two pairs of near-duplicates (within 3 cells) and one
    # well-separated unique. Expect the distance filter to drop at least one
    # of the duplicates, so the result is <= 3 even though k=4.
    base = [(10.0, 10.0, 5.0), (20.0, 20.0, 5.0), (30.0, 30.0, 5.0), (40.0, 40.0, 5.0)]

    def _shift(coords, dx):
        return [(x + dx, y, z) for (x, y, z) in coords]

    candidates = [
        # Two near-identical (≈0 distance).
        {"geology_index": 0, "real_revenue": 1000.0, "snapshot_id": "dup_a1"},
        {"geology_index": 0, "real_revenue": 990.0, "snapshot_id": "dup_a2"},
        # Two more near-identical, but far from a.
        {"geology_index": 0, "real_revenue": 980.0, "snapshot_id": "dup_b1"},
        {"geology_index": 0, "real_revenue": 970.0, "snapshot_id": "dup_b2"},
    ]
    snapshots = {
        "dup_a1": _make_snapshot(base),
        "dup_a2": _make_snapshot(_shift(base, 0.1)),  # ~0.2 total L2 → < 3
        "dup_b1": _make_snapshot(_shift(base, 80.0)),  # far away
        "dup_b2": _make_snapshot(_shift(base, 80.1)),  # near b1
    }
    metrics_path = _write_metrics_and_snapshots(tmp_path, 0, candidates, snapshots)

    rng = np.random.default_rng(7)
    out = _load_elite_seeds(
        geology_index=0,
        prior_metrics=[metrics_path],
        k=4,
        rng=rng,
    )
    assert len(out) <= 3, f"expected min-dist filter to keep <=3, got {len(out)}"


def test_elite_loader_empty_priors(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    out = _load_elite_seeds(
        geology_index=0,
        prior_metrics=[],
        k=4,
        rng=rng,
    )
    assert out == []


def test_elite_loader_no_priors_for_geo(tmp_path: Path) -> None:
    # All candidates are for geo 1, but we request geo 0.
    candidates = [
        {"geology_index": 1, "real_revenue": 100.0, "snapshot_id": "snapA"},
    ]
    snapshots = {"snapA": _make_snapshot([(1.0, 1.0, 1.0)])}
    metrics_path = _write_metrics_and_snapshots(tmp_path, 0, candidates, snapshots)

    rng = np.random.default_rng(0)
    out = _load_elite_seeds(
        geology_index=0,
        prior_metrics=[metrics_path],
        k=4,
        rng=rng,
    )
    assert out == []


def test_elite_loader_missing_snapshot_json(tmp_path: Path) -> None:
    # Metrics references three snapshots; only one JSON file exists.
    candidates = [
        {"geology_index": 0, "real_revenue": 300.0, "snapshot_id": "exists"},
        {"geology_index": 0, "real_revenue": 200.0, "snapshot_id": "missing_1"},
        {"geology_index": 0, "real_revenue": 100.0, "snapshot_id": "missing_2"},
    ]
    snapshots = {
        "exists": _make_snapshot([(10.0, 10.0, 5.0), (20.0, 20.0, 5.0)]),
    }
    metrics_path = _write_metrics_and_snapshots(tmp_path, 0, candidates, snapshots)

    rng = np.random.default_rng(0)
    out = _load_elite_seeds(
        geology_index=0,
        prior_metrics=[metrics_path],
        k=4,
        rng=rng,
    )
    # Two missing snapshots silently skipped; one survives.
    assert len(out) == 1


def test_elite_loader_filters_non_finite_real(tmp_path: Path) -> None:
    candidates = [
        {"geology_index": 0, "real_revenue": None, "snapshot_id": "null_rev"},
        {"geology_index": 0, "real_revenue": 100.0, "snapshot_id": "ok"},
    ]
    snapshots = {
        "null_rev": _make_snapshot([(1.0, 1.0, 1.0)]),
        "ok": _make_snapshot([(2.0, 2.0, 2.0)]),
    }
    metrics_path = _write_metrics_and_snapshots(tmp_path, 0, candidates, snapshots)

    rng = np.random.default_rng(0)
    out = _load_elite_seeds(
        geology_index=0,
        prior_metrics=[metrics_path],
        k=4,
        rng=rng,
    )
    # The null-revenue entry must be skipped; only "ok" returns.
    assert len(out) == 1
    # First-well coord identifies it.
    assert float(out[0][0, 0]) == 2.0
