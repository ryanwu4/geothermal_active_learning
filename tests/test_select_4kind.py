"""Tests for the 4-kind selection path in ``orchestrator.select.select_batch``.

Covers the new ``kind_fractions`` API (frontier/adversarial/exploit/cma), the
rounding-residual behavior, backfill semantics when a pool is short, and the
legacy ``frontier_fraction`` regression path.
"""
from __future__ import annotations

import numpy as np

from orchestrator.select import _VALID_KINDS, Candidate, select_batch


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
        snapshot_id=snapshot_id or f"g{geology_index}_p{predicted:.2f}_{kind}",
        run_id=geology_index * 1000 + int(predicted * 10),
        iteration=1,
        kind=kind,
        predicted_revenue=predicted,
        coords_xyz=np.asarray(coords, dtype=np.float32),
        is_injector=is_injector,
        well_config_path=f"/tmp/wc_{geology_index}_{kind}_{predicted}.jl",
    )


def _pool(n_per_geo: int, n_geos: int, kind: str, seed: int) -> list[Candidate]:
    """Build ``n_per_geo`` candidates per geology for ``kind``.

    Each kind uses its own seed so coords don't collide across kinds (the FPS
    diversity filter would otherwise drop perfect duplicates).
    """
    rng = np.random.default_rng(seed)
    cands: list[Candidate] = []
    for g in range(n_geos):
        for i in range(n_per_geo):
            coords = rng.uniform(0, 100, size=(4, 3)).tolist()
            inj = [True, False, True, False]
            cands.append(
                _make_candidate(
                    g, coords, inj,
                    predicted=float(g * 10 + i + 0.01 * hash(kind) % 1),
                    kind=kind,
                    snapshot_id=f"{kind}_g{g}_i{i}",
                )
            )
    return cands


def test_valid_kinds_constant() -> None:
    assert _VALID_KINDS == ("frontier", "adversarial", "exploit", "cma")


def test_select_4way_exact_split() -> None:
    # 60 cands * 4 geos * 4 kinds — but use 15 per geo to keep total fast.
    # Spec says "60 candidates per kind (240 total), equal 0.25 fractions,
    # batch=200 → assert 50 of each kind". We supply 60 per kind by using 4
    # geos × 15 = 60.
    cands = []
    cands += _pool(15, 4, "frontier", seed=1)
    cands += _pool(15, 4, "adversarial", seed=2)
    cands += _pool(15, 4, "exploit", seed=3)
    cands += _pool(15, 4, "cma", seed=4)
    selected = select_batch(
        cands,
        batch_size=200,
        kind_fractions={"frontier": 0.25, "adversarial": 0.25, "exploit": 0.25, "cma": 0.25},
    )
    counts = {k: sum(1 for c in selected if c.kind == k) for k in _VALID_KINDS}
    assert counts["frontier"] == 50
    assert counts["adversarial"] == 50
    assert counts["exploit"] == 50
    assert counts["cma"] == 50
    assert len(selected) == 200


def test_select_4way_unequal_fractions() -> None:
    cands = []
    cands += _pool(40, 4, "frontier", seed=11)        # 160 frontier
    cands += _pool(20, 4, "adversarial", seed=12)     # 80 adversarial
    cands += _pool(20, 4, "exploit", seed=13)         # 80 exploit
    cands += _pool(20, 4, "cma", seed=14)             # 80 cma
    selected = select_batch(
        cands,
        batch_size=200,
        kind_fractions={"frontier": 0.5, "adversarial": 0.25, "exploit": 0.125, "cma": 0.125},
    )
    counts = {k: sum(1 for c in selected if c.kind == k) for k in _VALID_KINDS}
    assert counts["frontier"] == 100
    assert counts["adversarial"] == 50
    assert counts["exploit"] == 25
    assert counts["cma"] == 25
    assert len(selected) == 200


def test_select_4way_rounding_residual() -> None:
    cands = []
    cands += _pool(80, 4, "frontier", seed=21)
    cands += _pool(80, 4, "adversarial", seed=22)
    cands += _pool(80, 4, "exploit", seed=23)
    cands += _pool(80, 4, "cma", seed=24)
    selected = select_batch(
        cands,
        batch_size=201,
        kind_fractions={"frontier": 0.25, "adversarial": 0.25, "exploit": 0.25, "cma": 0.25},
    )
    counts = {k: sum(1 for c in selected if c.kind == k) for k in _VALID_KINDS}
    assert sum(counts.values()) == 201
    assert len(selected) == 201
    # No kind off by more than 1 from 50.25 → counts in {50, 51}.
    for k, v in counts.items():
        assert v in (50, 51), f"kind={k} count={v} out of {{50,51}}"


def test_select_4way_short_one_pool_backfill() -> None:
    cands = []
    cands += _pool(80, 4, "frontier", seed=31)
    cands += _pool(80, 4, "adversarial", seed=32)
    cands += _pool(80, 4, "exploit", seed=33)
    # Only 5 cma candidates total (in geo 0).
    cma_short = []
    rng = np.random.default_rng(34)
    for i in range(5):
        coords = rng.uniform(0, 100, size=(4, 3)).tolist()
        cma_short.append(_make_candidate(
            0, coords, [True, False, True, False],
            predicted=float(100 + i), kind="cma",
            snapshot_id=f"cma_only_{i}",
        ))
    cands += cma_short
    selected = select_batch(
        cands,
        batch_size=200,
        kind_fractions={"frontier": 0.25, "adversarial": 0.25, "exploit": 0.25, "cma": 0.25},
    )
    assert len(selected) == 200
    counts = {k: sum(1 for c in selected if c.kind == k) for k in _VALID_KINDS}
    # cma cannot exceed 5 (the only ones available).
    assert counts["cma"] <= 5
    # The 45 (or so) missing cma slots came from another kind; tags preserved
    # (no relabeling) — total kind tags still sum to batch_size.
    assert sum(counts.values()) == 200
    # Each selected cma candidate should be one of the 5 we put in the pool.
    cma_chosen_ids = {c.snapshot_id for c in selected if c.kind == "cma"}
    assert cma_chosen_ids.issubset({f"cma_only_{i}" for i in range(5)})


def test_select_4way_empty_pool_for_one_kind() -> None:
    cands = []
    cands += _pool(80, 4, "frontier", seed=41)
    cands += _pool(80, 4, "adversarial", seed=42)
    # zero exploit candidates
    cands += _pool(80, 4, "cma", seed=44)
    selected = select_batch(
        cands,
        batch_size=200,
        kind_fractions={"frontier": 0.25, "adversarial": 0.25, "exploit": 0.25, "cma": 0.25},
    )
    counts = {k: sum(1 for c in selected if c.kind == k) for k in _VALID_KINDS}
    assert counts["exploit"] == 0
    # Backfill compensates from other kinds; total must be 200.
    assert sum(counts.values()) == 200


def test_select_legacy_mode_still_works() -> None:
    cands = []
    cands += _pool(80, 4, "frontier", seed=51)
    cands += _pool(80, 4, "adversarial", seed=52)
    selected = select_batch(cands, batch_size=200, frontier_fraction=0.85)
    counts = {k: sum(1 for c in selected if c.kind == k) for k in _VALID_KINDS}
    assert counts["frontier"] == 170
    assert counts["adversarial"] == 30
    assert counts["exploit"] == 0
    assert counts["cma"] == 0
    assert sum(counts.values()) == 200


def test_select_legacy_default_fraction() -> None:
    """``frontier_fraction=None`` (default) should fall back to 0.85."""
    cands = []
    cands += _pool(80, 4, "frontier", seed=61)
    cands += _pool(80, 4, "adversarial", seed=62)
    selected_default = select_batch(cands, batch_size=200)
    selected_explicit = select_batch(cands, batch_size=200, frontier_fraction=0.85)
    counts_default = {k: sum(1 for c in selected_default if c.kind == k) for k in _VALID_KINDS}
    counts_explicit = {k: sum(1 for c in selected_explicit if c.kind == k) for k in _VALID_KINDS}
    assert counts_default == counts_explicit
    # And specifically: 170 frontier, 30 adversarial.
    assert counts_default["frontier"] == 170
    assert counts_default["adversarial"] == 30


def test_select_partial_kind_fractions() -> None:
    """When only one kind is listed, all slots go to that kind."""
    cands = []
    cands += _pool(80, 4, "frontier", seed=71)
    cands += _pool(80, 4, "adversarial", seed=72)
    cands += _pool(80, 4, "exploit", seed=73)
    cands += _pool(80, 4, "cma", seed=74)
    selected = select_batch(
        cands,
        batch_size=200,
        kind_fractions={"exploit": 1.0},
    )
    counts = {k: sum(1 for c in selected if c.kind == k) for k in _VALID_KINDS}
    assert counts["exploit"] == 200
    assert counts["frontier"] == 0
    assert counts["adversarial"] == 0
    assert counts["cma"] == 0
