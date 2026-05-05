"""Determinism and balance tests for diversity selection."""
from __future__ import annotations

import numpy as np
import pytest

from orchestrator.select import Candidate, select_batch


def _make_candidate(
    geology_index: int,
    coords: list[list[float]],
    is_injector: list[bool],
    predicted: float,
    kind: str = "frontier",
    snapshot_id: str | None = None,
) -> Candidate:
    return Candidate(
        geology_index=geology_index,
        geology_file=f"geo_{geology_index}.h5",
        geology_name=f"geo_{geology_index}",
        geology_config_id=geology_index,
        geology_scenario_name=None,
        geology_sample_num=None,
        snapshot_id=snapshot_id or f"g{geology_index}_p{predicted:.2f}",
        run_id=geology_index * 100 + int(predicted * 10),
        iteration=45,
        kind=kind,
        predicted_revenue=predicted,
        coords_xyz=np.asarray(coords, dtype=np.float32),
        is_injector=is_injector,
        well_config_path=f"/tmp/wc_{geology_index}_{predicted}.jl",
    )


def _grid_candidates(n_per_geo: int, n_geos: int, kind: str = "frontier") -> list[Candidate]:
    rng = np.random.default_rng(0)
    cands = []
    for g in range(n_geos):
        for i in range(n_per_geo):
            coords = rng.uniform(0, 100, size=(4, 3)).tolist()
            inj = [True, False, True, False]
            cands.append(_make_candidate(g, coords, inj, predicted=float(g * 10 + i), kind=kind))
    return cands


def test_select_batch_respects_size() -> None:
    cands = _grid_candidates(20, 3, kind="frontier") + _grid_candidates(5, 3, kind="adversarial")
    selected = select_batch(cands, batch_size=15, frontier_fraction=0.8)
    assert len(selected) == 15


def test_select_batch_frontier_adversarial_split() -> None:
    cands = _grid_candidates(20, 3, kind="frontier") + _grid_candidates(20, 3, kind="adversarial")
    selected = select_batch(cands, batch_size=20, frontier_fraction=0.85)
    n_frontier = sum(1 for c in selected if c.kind == "frontier")
    n_adv = sum(1 for c in selected if c.kind == "adversarial")
    # Allow ±1 due to rounding in the proportional split.
    assert abs(n_frontier - 17) <= 1
    assert abs(n_adv - 3) <= 1
    assert n_frontier + n_adv == 20


def test_select_batch_deterministic_across_calls() -> None:
    cands = _grid_candidates(10, 3, kind="frontier") + _grid_candidates(5, 3, kind="adversarial")
    a = select_batch(cands, batch_size=12, frontier_fraction=0.8)
    b = select_batch(cands, batch_size=12, frontier_fraction=0.8)
    assert [c.snapshot_id for c in a] == [c.snapshot_id for c in b]


def test_select_batch_distributes_across_geologies() -> None:
    cands = _grid_candidates(20, 4, kind="frontier")
    selected = select_batch(cands, batch_size=12, frontier_fraction=1.0)
    geo_counts = {}
    for c in selected:
        geo_counts[c.geology_index] = geo_counts.get(c.geology_index, 0) + 1
    # Each geology should get at least 1 slot when total >= n_geologies.
    assert len(geo_counts) == 4
    for v in geo_counts.values():
        assert v >= 1


def test_select_batch_handles_empty_pool() -> None:
    selected = select_batch([], batch_size=5, frontier_fraction=0.8)
    assert selected == []


def test_select_batch_handles_short_pool() -> None:
    cands = _grid_candidates(2, 2, kind="frontier")  # 4 frontier candidates
    selected = select_batch(cands, batch_size=10, frontier_fraction=0.85)
    # Can't exceed pool size; should be at most what's available.
    assert len(selected) <= 4


def test_flatten_coords_permutation_invariant() -> None:
    """Two configs that differ only by within-type well reordering should map
    to the same feature vector under _flatten_coords.
    """
    from orchestrator.select import _flatten_coords

    coords_a = np.array([
        [10.0, 10.0, 5.0],   # injector
        [20.0, 20.0, 5.0],   # producer
        [30.0, 30.0, 5.0],   # injector
        [40.0, 40.0, 5.0],   # producer
    ], dtype=np.float32)
    inj_a = [True, False, True, False]

    # Same physical placement, but well 0 ↔ well 2 (both injectors) and
    # well 1 ↔ well 3 (both producers) are swapped.
    coords_b = np.array([
        [30.0, 30.0, 5.0],   # injector (was well 2)
        [40.0, 40.0, 5.0],   # producer (was well 3)
        [10.0, 10.0, 5.0],   # injector (was well 0)
        [20.0, 20.0, 5.0],   # producer (was well 1)
    ], dtype=np.float32)
    inj_b = [True, False, True, False]

    a = _make_candidate(0, coords_a.tolist(), inj_a, predicted=1.0)
    b = _make_candidate(0, coords_b.tolist(), inj_b, predicted=1.0)

    fa = _flatten_coords(a)
    fb = _flatten_coords(b)
    np.testing.assert_array_equal(fa, fb)


def test_flatten_coords_distinguishes_truly_different_configs() -> None:
    """Sanity: configs with genuinely different placements must produce
    different feature vectors.
    """
    from orchestrator.select import _flatten_coords

    coords_a = [[10.0, 10.0, 5.0], [20.0, 20.0, 5.0]]
    coords_b = [[10.0, 10.0, 5.0], [20.0, 21.0, 5.0]]  # one well shifted by 1
    inj = [True, False]
    a = _make_candidate(0, coords_a, inj, predicted=1.0)
    b = _make_candidate(0, coords_b, inj, predicted=1.0)
    assert not np.array_equal(_flatten_coords(a), _flatten_coords(b))


def test_diversity_collapses_permuted_duplicates() -> None:
    """The diversity FPS must not pick a permuted duplicate as 'diverse'."""
    # 3 candidates: A, A-permuted (physically identical), B (truly different).
    # With 2 slots, FPS should pick {A, B}, never {A, A-permuted}.
    a_coords = [[10.0, 10.0, 5.0], [20.0, 20.0, 5.0], [30.0, 30.0, 5.0], [40.0, 40.0, 5.0]]
    inj = [True, False, True, False]

    # A-permuted: same physical placement, swap injectors and swap producers.
    ap_coords = [[30.0, 30.0, 5.0], [40.0, 40.0, 5.0], [10.0, 10.0, 5.0], [20.0, 20.0, 5.0]]

    # B: genuinely different placement.
    b_coords = [[80.0, 80.0, 5.0], [90.0, 90.0, 5.0], [60.0, 60.0, 5.0], [70.0, 70.0, 5.0]]

    cands = [
        _make_candidate(0, a_coords, inj, predicted=10.0, snapshot_id="A"),
        _make_candidate(0, ap_coords, inj, predicted=9.0, snapshot_id="A_permuted"),
        _make_candidate(0, b_coords, inj, predicted=8.0, snapshot_id="B"),
    ]
    selected = select_batch(cands, batch_size=2, frontier_fraction=1.0)
    chosen = {c.snapshot_id for c in selected}
    assert "A" in chosen
    assert "B" in chosen
    assert "A_permuted" not in chosen


def test_select_batch_top_revenue_seed() -> None:
    """The first frontier pick per geology should be the highest-revenue one."""
    cands = [
        _make_candidate(0, [[0, 0, 0]] * 4, [True, False, True, False], 100.0, snapshot_id="top0"),
        _make_candidate(0, [[10, 10, 10]] * 4, [True, False, True, False], 50.0, snapshot_id="mid0"),
        _make_candidate(0, [[20, 20, 20]] * 4, [True, False, True, False], 25.0, snapshot_id="low0"),
    ]
    selected = select_batch(cands, batch_size=1, frontier_fraction=1.0)
    assert selected[0].snapshot_id == "top0"
