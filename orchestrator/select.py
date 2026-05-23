"""Candidate selection — pick a diverse, multi-kind batch for INTERSECT.

Inputs are flat lists of candidate snapshots; each snapshot describes a single
well configuration in a single geology and carries a surrogate-predicted
revenue along with its kind:
  * ``frontier``     — LHS-seeded Adam at k_safe; diversity-selected (FPS).
  * ``adversarial``  — LHS-seeded Adam at k_adv; round-robin top-revenue.
  * ``exploit``      — elite-seeded Adam at k_safe; round-robin top-revenue.
  * ``cma``          — CMA-ES-seeded Adam at k_safe; round-robin top-revenue.

Selection supports two modes:
  * Back-compat (legacy 2-kind): ``frontier_fraction`` knob splits the batch
    between frontier and adversarial, matching the original AL design.
  * 4-kind: ``kind_fractions`` dict allocates per-kind targets; each pool is
    selected by its own rule (FPS for frontier, top-revenue for the others).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# Valid kind tags; mirrors orchestrator/acquire.py:_KIND_ORDER.
_VALID_KINDS = ("frontier", "adversarial", "exploit", "cma")


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
    kind: str  # one of "frontier", "adversarial", "exploit", "cma"
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
    """Distribute ``target`` slots across geologies *equally* (not proportionally),
    then greedily diversify within each geology.

    Equal allocation matches the `n_starts_per_geology` guarantee at acquisition
    time: every geology is meant to get the same number of seed configurations,
    so the selection step shouldn't undo that by penalising geologies whose
    Adam runs produced fewer finite candidates. The previous proportional
    allocation drove the iter-5 geo-8 acquisition bias when geo-8 Adam runs
    went non-finite more often than the cohort.
    """
    if target <= 0 or not candidates:
        return []

    by_geo: dict[int, list[Candidate]] = {}
    for c in candidates:
        by_geo.setdefault(c.geology_index, []).append(c)

    geo_keys = sorted(by_geo.keys())
    n_geos = len(geo_keys)
    # Equal allocation: floor(target / n_geos) per geology, remainder distributed
    # deterministically by geology id (round-robin).
    base = np.array([target // n_geos] * n_geos, dtype=int)
    remainder = target - int(base.sum())
    for i in range(remainder):
        base[i % n_geos] += 1
    # Clamp so we don't ask for more than each geology has, then redistribute
    # any leftover to geologies that still have headroom.
    headroom = np.array([len(by_geo[g]) for g in geo_keys], dtype=int)
    for i in range(n_geos):
        base[i] = min(int(base[i]), int(headroom[i]))
    short = target - int(base.sum())
    while short > 0:
        # Find geologies with remaining headroom; prefer those with the most.
        slack = headroom - base
        slack[slack <= 0] = -1  # ignore exhausted geologies
        if int(slack.max()) <= 0:
            break
        i = int(np.argmax(slack))
        take = min(short, int(slack[i]))
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


def _select_top_per_geology(
    candidates: Sequence[Candidate], target: int
) -> list[Candidate]:
    """Top-revenue picks, distributed round-robin across geologies. No FPS.

    Shared selection rule for ``adversarial``, ``exploit``, and ``cma`` kinds —
    all three should *concentrate* on high-predicted-revenue picks rather than
    diversify, because their job is exploitation (exploit/cma) or probing
    specific high-revenue surrogate predictions (adversarial).
    """
    if target <= 0 or not candidates:
        return []

    by_geo: dict[int, list[Candidate]] = {}
    for c in candidates:
        by_geo.setdefault(c.geology_index, []).append(c)

    geo_keys = sorted(by_geo.keys())
    selected: list[Candidate] = []
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


# Back-compat alias for the original 2-kind selection path.
_select_adversarial = _select_top_per_geology


def select_batch(
    candidates: Sequence[Candidate],
    *,
    batch_size: int,
    frontier_fraction: float | None = None,
    kind_fractions: dict[str, float] | None = None,
) -> list[Candidate]:
    """Pick ``batch_size`` candidates with the configured kind mix.

    Two calling conventions:
      * Legacy (2-kind): pass ``frontier_fraction`` only. Adversarial gets the
        remainder. Matches the original AL design.
      * 4-kind: pass ``kind_fractions`` with any subset of
        ``{"frontier","adversarial","exploit","cma"}``. Fractions need not sum
        to 1.0 — they're normalized to integer per-kind targets that sum to
        ``batch_size``. Any kind not listed gets target=0.

    Returned order: frontier → adversarial → exploit → cma (stable for
    downstream consumers that may key off position).
    """
    if batch_size <= 0:
        return []

    # Resolve per-kind targets.
    if kind_fractions is not None:
        fractions = {k: max(0.0, float(v)) for k, v in kind_fractions.items() if k in _VALID_KINDS}
        total = sum(fractions.values())
        if total <= 0:
            return []
        # Allocate by rounding, then fix sum by adjusting the largest residual.
        raw = {k: batch_size * (v / total) for k, v in fractions.items()}
        targets = {k: int(round(v)) for k, v in raw.items()}
        diff = batch_size - sum(targets.values())
        if diff != 0:
            # Distribute the rounding remainder to the kinds with largest
            # fractional residuals (positive diff) or smallest (negative diff).
            residuals = sorted(
                ((k, raw[k] - targets[k]) for k in targets),
                key=lambda t: -t[1] if diff > 0 else t[1],
            )
            for i in range(abs(diff)):
                k = residuals[i % len(residuals)][0]
                targets[k] += 1 if diff > 0 else -1
        for k in _VALID_KINDS:
            targets.setdefault(k, 0)
    else:
        ff = 0.85 if frontier_fraction is None else float(frontier_fraction)
        frontier_target = max(0, int(round(batch_size * ff)))
        adversarial_target = batch_size - frontier_target
        targets = {
            "frontier": frontier_target,
            "adversarial": adversarial_target,
            "exploit": 0,
            "cma": 0,
        }

    pools = {k: [c for c in candidates if c.kind == k] for k in _VALID_KINDS}

    front_selected = _select_frontier_per_geology(pools["frontier"], targets["frontier"])
    adv_selected = _select_top_per_geology(pools["adversarial"], targets["adversarial"])
    exploit_selected = _select_top_per_geology(pools["exploit"], targets["exploit"])
    cma_selected = _select_top_per_geology(pools["cma"], targets["cma"])

    chosen: dict[str, list[Candidate]] = {
        "frontier": front_selected,
        "adversarial": adv_selected,
        "exploit": exploit_selected,
        "cma": cma_selected,
    }

    # Backfill any short pool from the largest-surplus pool. Tags are preserved
    # — a frontier candidate filling in for cma still reads as ``kind=frontier``.
    def _surplus(kind: str) -> int:
        return len(pools[kind]) - len(chosen[kind])

    def _short() -> int:
        return batch_size - sum(len(v) for v in chosen.values())

    while _short() > 0:
        # Pick the surplus-largest source kind.
        source_kinds = sorted(_VALID_KINDS, key=lambda k: -_surplus(k))
        if _surplus(source_kinds[0]) <= 0:
            break  # nothing left anywhere
        src = source_kinds[0]
        # Find the most-short destination to direct the backfill into.
        dst = max(_VALID_KINDS, key=lambda k: targets[k] - len(chosen[k]))
        existing_ids = {id(c) for c in chosen[src]}
        available = [c for c in pools[src] if id(c) not in existing_ids]
        if not available:
            break
        # Use the appropriate selector for the source kind.
        selector = _select_frontier_per_geology if src == "frontier" else _select_top_per_geology
        take = min(_short(), len(available))
        extras = selector(available, take)
        if not extras:
            break
        chosen[dst].extend(extras)

    return (
        list(chosen["frontier"])
        + list(chosen["adversarial"])
        + list(chosen["exploit"])
        + list(chosen["cma"])
    )


def select_batch_ensemble(
    candidates: Sequence[Candidate],
    *,
    batch_size: int | None = None,
) -> list[Candidate]:
    """Flat-pool selector for ensemble mode.

    Each candidate already represents a full well configuration evaluated across
    the entire geological ensemble, so per-geology slot allocation no longer
    applies. We just rank by predicted EMV (the same number stored in
    ``predicted_revenue`` for ensemble candidates) and optionally cap to
    ``batch_size``. Order: exploit first, then frontier, then anything else —
    matches the conventional "highest-value cohort first" ordering.
    """
    if not candidates:
        return []
    pool = list(candidates)
    kind_priority = {"exploit": 0, "frontier": 1}
    pool.sort(key=lambda c: (kind_priority.get(c.kind, 99), -float(c.predicted_revenue)))
    if batch_size is not None and batch_size >= 0 and len(pool) > batch_size:
        # Cap by predicted EMV across the whole pool (kind-agnostic) so we
        # don't drop a frontier candidate that scored higher than the worst
        # exploit. Then re-apply the kind-then-EMV ordering.
        pool = sorted(pool, key=lambda c: -float(c.predicted_revenue))[:batch_size]
        pool.sort(key=lambda c: (kind_priority.get(c.kind, 99), -float(c.predicted_revenue)))
    return pool
