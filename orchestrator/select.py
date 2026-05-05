"""Candidate selection — pick a diverse, frontier-heavy batch for INTERSECT.

Inputs are flat lists of candidate snapshots; each snapshot describes a single
well configuration in a single geology and carries a surrogate-predicted
revenue along with its kind ("frontier" or "adversarial").

The selection rule (matching the plan):
  * Frontier subset = ``round(B * frontier_fraction)`` candidates, chosen by
    taking the highest-revenue points per geology and then enforcing diversity
    across geologies via greedy farthest-point selection in flattened
    well-config space.
  * Adversarial subset = ``B - frontier_count`` candidates, chosen as the
    highest-revenue adversarial points per geology with no further diversity
    pass (small set, intent is to probe specific exploits).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class Candidate:
    geology_index: int
    geology_file: str
    geology_name: str
    geology_config_id: str | int | None
    geology_scenario_name: str | None
    geology_sample_num: int | str | None
    snapshot_id: str
    run_id: int
    iteration: int
    kind: str  # "frontier" or "adversarial"
    predicted_revenue: float
    coords_xyz: np.ndarray  # (n_wells, 3) float32
    is_injector: list[bool]
    well_config_path: str
    snapshot_json_path: str = ""
    extras: dict = field(default_factory=dict)


def _flatten_coords(c: Candidate) -> np.ndarray:
    """Permutation-invariant feature vector for diversity comparison.

    All injectors are interchangeable and all producers are interchangeable, so
    two physically identical configurations with shuffled well indices should
    have identical feature vectors. We group wells by type and sort each group
    in lex order on ``(x, y, z)`` before flattening.

    The injector / producer counts are constant across all candidates in a
    selection (they come from the shared ``wells`` config), so we don't need
    to encode the type flags themselves — the segment boundaries are fixed.
    """
    coords = c.coords_xyz.astype(np.float32)  # (n_wells, 3)
    inj = np.asarray(c.is_injector, dtype=bool)

    inj_coords = coords[inj]
    prod_coords = coords[~inj]

    if inj_coords.shape[0] > 0:
        order = np.lexsort((inj_coords[:, 2], inj_coords[:, 1], inj_coords[:, 0]))
        inj_coords = inj_coords[order]
    if prod_coords.shape[0] > 0:
        order = np.lexsort((prod_coords[:, 2], prod_coords[:, 1], prod_coords[:, 0]))
        prod_coords = prod_coords[order]

    return np.concatenate([inj_coords.reshape(-1), prod_coords.reshape(-1)])


def _farthest_point_select(features: np.ndarray, k: int, seed_idx: int = 0) -> list[int]:
    """Greedy farthest-point sampling in feature space.

    Deterministic given ``seed_idx`` (default 0 picks the first item, which the
    caller arranges to be the highest-scoring candidate). Output preserves the
    selection order; first item is the farthest-from-seed pick.
    """
    n = features.shape[0]
    if k >= n:
        return list(range(n))

    selected = [seed_idx]
    # min distance from each point to the current selection set.
    diffs = features - features[seed_idx]
    min_d = np.linalg.norm(diffs, axis=1)
    while len(selected) < k:
        next_idx = int(np.argmax(min_d))
        if next_idx in selected:
            # All remaining have zero distance — break to avoid infinite loop.
            break
        selected.append(next_idx)
        diffs = features - features[next_idx]
        d = np.linalg.norm(diffs, axis=1)
        min_d = np.minimum(min_d, d)
    return selected


def _select_frontier_per_geology(
    candidates: Sequence[Candidate], target: int
) -> list[Candidate]:
    """Distribute ``target`` slots across geologies proportional to candidate counts,
    then greedily diversify within each geology.
    """
    if target <= 0 or not candidates:
        return []

    by_geo: dict[int, list[Candidate]] = {}
    for c in candidates:
        by_geo.setdefault(c.geology_index, []).append(c)

    # Initial proportional allocation, rounded down.
    geo_keys = sorted(by_geo.keys())
    counts = np.array([len(by_geo[g]) for g in geo_keys], dtype=np.float64)
    weights = counts / counts.sum()
    raw = weights * target
    base = np.floor(raw).astype(int)
    remainder = target - int(base.sum())
    # Distribute leftover slots by largest fractional remainder.
    fractional = raw - base
    order = np.argsort(-fractional)
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    # Clamp so we don't ask for more than each geology has.
    for i, g in enumerate(geo_keys):
        base[i] = min(int(base[i]), len(by_geo[g]))

    # If clamping reduced the total below target, redistribute leftover to
    # geologies that still have headroom, again by fractional remainder.
    short = target - int(base.sum())
    if short > 0:
        for i in order:
            if short <= 0:
                break
            geo = geo_keys[i]
            headroom = len(by_geo[geo]) - int(base[i])
            if headroom > 0:
                take = min(short, headroom)
                base[i] += take
                short -= take

    selected: list[Candidate] = []
    for i, g in enumerate(geo_keys):
        slot = int(base[i])
        if slot <= 0:
            continue
        # Within this geology: sort by revenue desc, seed FPS at the top, then
        # diversify across the rest.
        cands = sorted(by_geo[g], key=lambda c: -c.predicted_revenue)
        if slot >= len(cands):
            selected.extend(cands)
            continue
        feats = np.stack([_flatten_coords(c) for c in cands], axis=0)
        chosen = _farthest_point_select(feats, slot, seed_idx=0)
        selected.extend(cands[i] for i in chosen)

    return selected


def _select_adversarial(
    candidates: Sequence[Candidate], target: int
) -> list[Candidate]:
    """Top-revenue adversarial picks, distributed across geologies."""
    if target <= 0 or not candidates:
        return []

    by_geo: dict[int, list[Candidate]] = {}
    for c in candidates:
        by_geo.setdefault(c.geology_index, []).append(c)

    geo_keys = sorted(by_geo.keys())
    selected: list[Candidate] = []
    # Round-robin picking across geologies, taking the top-remaining-revenue
    # candidate at each turn. This spreads adversarial probes across geologies
    # rather than concentrating them on a single high-disagreement field.
    iters = {g: iter(sorted(by_geo[g], key=lambda c: -c.predicted_revenue)) for g in geo_keys}
    while len(selected) < target:
        progress = False
        for g in geo_keys:
            if len(selected) >= target:
                break
            try:
                selected.append(next(iters[g]))
                progress = True
            except StopIteration:
                continue
        if not progress:
            break
    return selected


def select_batch(
    candidates: Sequence[Candidate],
    *,
    batch_size: int,
    frontier_fraction: float,
) -> list[Candidate]:
    """Pick ``batch_size`` candidates with the configured frontier/adversarial mix.

    Order in the returned list is ``frontier first, adversarial last`` so
    downstream code can identify the boundary by ``kind`` if needed.
    """
    if batch_size <= 0:
        return []
    frontier_target = max(0, int(round(batch_size * frontier_fraction)))
    adversarial_target = batch_size - frontier_target

    frontier_pool = [c for c in candidates if c.kind == "frontier"]
    adversarial_pool = [c for c in candidates if c.kind == "adversarial"]

    front_selected = _select_frontier_per_geology(frontier_pool, frontier_target)
    adv_selected = _select_adversarial(adversarial_pool, adversarial_target)

    # If one pool was short, top up from the other so we hit batch_size when
    # possible.
    short = batch_size - (len(front_selected) + len(adv_selected))
    if short > 0:
        if len(front_selected) < frontier_target:
            extras = _select_adversarial(
                [c for c in adversarial_pool if c not in adv_selected], short
            )
            adv_selected.extend(extras)
        elif len(adv_selected) < adversarial_target:
            extras = _select_frontier_per_geology(
                [c for c in frontier_pool if c not in front_selected], short
            )
            front_selected.extend(extras)

    return list(front_selected) + list(adv_selected)
