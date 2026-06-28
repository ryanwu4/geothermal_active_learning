"""Strict ensemble-EMV helpers — the single source of truth for the throw-out rule.

EMV (Expected Monetary Value) for a well config = the mean of a per-geology metric
(``real_revenue``, or NPV) across a FIXED ensemble of K geologies. The **strict** rule
this module enforces: a config's EMV is defined ONLY if every one of its K expected
geologies produced a finite value. If ANY geology is missing or non-finite, the whole
config is thrown out (EMV = ``None``) — excluded from best-in-batch, distributions, and
all EMV figures. This replaces the older "average over the survivors" behaviour, which
biased EMV (a config that failed on its worst geology looked artificially better) and
made configs non-comparable (each scored on a different geology subset).

Shared by the live ingest writers (``orchestrator.ingest``, ``scripts.phase_ingest_baseline``,
``orchestrator.npv_metrics``) and by the plot-time recomputation (``scripts.plot_convergence``
and the analysis overlays) so the rule is identical everywhere.

Two failure encodings exist in the persisted ``per_candidate_metrics.json`` and BOTH must be
caught by a COUNT-BASED check (never by counting nulls):

  - baseline writer (``phase_ingest_baseline``): emits a row with the metric = ``null`` for each
    failed geology (the snapshot keeps all K rows).
  - surrogate writer (``ingest``): omits the failed geology's row entirely (the snapshot ends
    with < K rows).

Both reduce the number of *finite distinct-geology* values below K, which is exactly what
:func:`strict_emv` / :func:`strict_per_snapshot_emv` test.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def _is_finite(v: Any) -> bool:
    """True iff ``v`` is non-None and a finite real number (handles numpy/None/str)."""
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def strict_emv(values: Iterable[Any], *, expected_k: int) -> float | None:
    """Mean of ``values`` iff EXACTLY ``expected_k`` of them are finite, else ``None``.

    ``values`` must already be deduplicated to one entry per geology (callers holding raw
    per-geology rows should use :func:`strict_per_snapshot_emv`, which dedupes first). A config
    with any missing/non-finite geology has fewer than ``expected_k`` finite values and is thrown
    out. A spurious extra value (more than ``expected_k``) also returns ``None`` — that signals a
    malformed cohort the caller should not silently average.
    """
    if not expected_k or expected_k <= 0:
        return None
    finite = [float(v) for v in values if _is_finite(v)]
    if len(finite) != expected_k:
        return None
    return sum(finite) / len(finite)


def _row_get(row: Any, key: str) -> Any:
    """Read ``key`` from a dict row or an object with attribute access."""
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def strict_per_snapshot_emv(
    rows: Sequence[Any],
    value_key: str,
    *,
    expected_k: int,
    geo_key: str = "geology_index",
) -> float | None:
    """Strict EMV for ONE snapshot's per-geology rows.

    Dedupes ``rows`` by ``geo_key`` (keeping the last finite value seen for a geology) and then
    applies :func:`strict_emv` with ``expected_k``. Accepts dict rows or objects with attribute
    access. Returns ``None`` if fewer than ``expected_k`` distinct geologies carry a finite
    ``value_key`` — i.e. the config had any failed/missing geology. This uniformly handles both
    failure encodings (a ``null`` row is non-finite → not counted; an omitted row → short count).
    """
    by_geo: dict[Any, float] = {}
    for r in rows:
        v = _row_get(r, value_key)
        if not _is_finite(v):
            continue
        by_geo[_row_get(r, geo_key)] = float(v)
    return strict_emv(by_geo.values(), expected_k=expected_k)


def expected_k_for_run(
    rows: Sequence[Any],
    *,
    geo_key: str = "geology_index",
    snap_key: str = "snapshot_id",
    iter_key: str = "iteration",
) -> int:
    """Infer the ensemble size K = max distinct-geology count over all (iter, snapshot) groups.

    Counts distinct ``geo_key`` values per group — a failed geology that was still emitted as a
    (null-valued) row still counts toward the intended K, so the strict check uses the full
    ensemble size even when some geologies failed. Where failures are omitted instead, complete
    snapshots still pin K, so the run-wide max recovers the true ensemble size as long as at least
    one config completed fully.

    Yields 15 for a full-ensemble run and 1 for a per-geology run. Call with a SINGLE iteration's
    rows to get that iteration's K (so mixed runs judge per-geology seed iters at K=1 and ensemble
    iters at K=15 separately).
    """
    groups: dict[Any, set] = {}
    for r in rows:
        sid = _row_get(r, snap_key)
        if sid is None:
            continue
        key = (_row_get(r, iter_key), sid)
        groups.setdefault(key, set()).add(_row_get(r, geo_key))
    if not groups:
        return 0
    return max(len(s) for s in groups.values())
