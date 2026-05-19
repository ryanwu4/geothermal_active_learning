"""Tests for the ``_select_top_per_geology`` helper.

Used by adversarial/exploit/cma kinds — picks the highest-predicted-revenue
candidates per geology in round-robin order, no FPS diversification.
"""
from __future__ import annotations

import numpy as np

from orchestrator.select import Candidate, _select_top_per_geology


def _make_candidate(
    geology_index: int,
    predicted: float,
    kind: str = "exploit",
    snapshot_id: str | None = None,
) -> Candidate:
    return Candidate(
        geology_index=geology_index,
        geology_file=f"geo_{geology_index}.h5",
        geology_name=f"geo_{geology_index}",
        geology_config_id=geology_index,
        geology_scenario_name=None,
        geology_sample_num=None,
        snapshot_id=snapshot_id or f"g{geology_index}_p{predicted:.4f}",
        run_id=geology_index * 1000 + int(predicted * 10),
        iteration=0,
        kind=kind,
        predicted_revenue=predicted,
        coords_xyz=np.zeros((4, 3), dtype=np.float32),
        is_injector=[True, False, True, False],
        well_config_path=f"/tmp/{geology_index}_{predicted}.jl",
    )


def test_top_per_geology_round_robin() -> None:
    cands = []
    for g in range(3):
        for i in range(5):
            cands.append(_make_candidate(g, predicted=float(g * 100 + i)))
    selected = _select_top_per_geology(cands, target=9)
    assert len(selected) == 9
    per_geo: dict[int, list[float]] = {}
    for c in selected:
        per_geo.setdefault(c.geology_index, []).append(c.predicted_revenue)
    assert set(per_geo.keys()) == {0, 1, 2}
    for g, revs in per_geo.items():
        assert len(revs) == 3
        # Within each geology, the picks should be ordered by descending revenue.
        assert revs == sorted(revs, reverse=True)
        # And the top-3 should be the top-3 of that geology's candidates.
        expected_top3 = sorted([float(g * 100 + i) for i in range(5)], reverse=True)[:3]
        assert revs == expected_top3


def test_top_per_geology_uneven_geos() -> None:
    cands = []
    # Geo 0: 10 candidates, predicted 0..9
    for i in range(10):
        cands.append(_make_candidate(0, predicted=float(i), snapshot_id=f"g0_{i}"))
    # Geo 1: 2 candidates
    for i in range(2):
        cands.append(_make_candidate(1, predicted=float(100 + i), snapshot_id=f"g1_{i}"))
    # Geo 2: 5 candidates
    for i in range(5):
        cands.append(_make_candidate(2, predicted=float(200 + i), snapshot_id=f"g2_{i}"))

    selected = _select_top_per_geology(cands, target=10)
    counts: dict[int, int] = {}
    for c in selected:
        counts[c.geology_index] = counts.get(c.geology_index, 0) + 1
    # Round-robin: g0,g1,g2,g0,g1,g2,g0,g2,g0,g2 → g0:4, g1:2, g2:4.
    assert counts[0] == 4
    assert counts[1] == 2
    assert counts[2] == 4
    assert sum(counts.values()) == 10


def test_top_per_geology_zero_target() -> None:
    cands = [_make_candidate(0, predicted=1.0)]
    assert _select_top_per_geology(cands, target=0) == []


def test_top_per_geology_empty_input() -> None:
    assert _select_top_per_geology([], target=5) == []
