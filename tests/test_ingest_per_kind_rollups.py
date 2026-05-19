"""Tests for the per-kind MAPE/MAE rollups in IngestMetrics + CandidateMetric.

The full ``ingest_iteration`` pipeline is too heavy to exercise here (subprocess
preprocess + h5 I/O), so we instead validate the schema and replay the small
``_mean_finite``-based rollup math directly on synthetic ``CandidateMetric``
rows.
"""
from __future__ import annotations

import math

import numpy as np

from orchestrator.ingest import CandidateMetric, IngestMetrics


def _mean_finite(vals: list[float]) -> float | None:
    """Reimplement the per-iteration helper inline.

    The real ``_mean_finite`` is a local closure inside ``ingest_iteration``
    (not importable). We reproduce its behavior verbatim here so the rollup
    math under test matches the production code path.
    """
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def test_ingest_metrics_has_new_fields() -> None:
    m = IngestMetrics(
        n_submitted=0, n_completed=0, completion_rate=0.0,
        batch_mape=None, batch_signed_pct_bias=None,
        frontier_mape=None, adversarial_mape=None,
        best_real_revenue_in_batch=None, best_real_revenue_so_far=None,
        n_train_samples=0,
    )
    # New fields default to None when not supplied.
    assert m.exploit_mape is None
    assert m.exploit_mae is None
    assert m.cma_mape is None
    assert m.cma_mae is None

    # And they accept explicit values.
    m2 = IngestMetrics(
        n_submitted=0, n_completed=0, completion_rate=0.0,
        batch_mape=None, batch_signed_pct_bias=None,
        frontier_mape=None, adversarial_mape=None,
        best_real_revenue_in_batch=None, best_real_revenue_so_far=None,
        n_train_samples=0,
        exploit_mape=0.11, exploit_mae=12.3,
        cma_mape=0.22, cma_mae=45.6,
    )
    assert m2.exploit_mape == 0.11
    assert m2.exploit_mae == 12.3
    assert m2.cma_mape == 0.22
    assert m2.cma_mae == 45.6


def _make_cm(kind: str, abs_pct: float, abs_err: float = 0.0) -> CandidateMetric:
    return CandidateMetric(
        snapshot_id=f"s_{kind}_{abs_pct}",
        case_id="case",
        output_file_name="out.h5",
        geology_index=0,
        geology_config_id=0,
        kind=kind,
        predicted_revenue=1.0,
        real_revenue=1.0,
        abs_pct_error=abs_pct,
        signed_pct_error=abs_pct,
        abs_error=abs_err,
    )


def test_per_kind_mape_aggregation() -> None:
    candidates = [
        _make_cm("frontier", 0.1, 10.0),
        _make_cm("frontier", 0.2, 20.0),
        _make_cm("adversarial", 0.5, 50.0),
        _make_cm("exploit", 0.05, 5.0),
        _make_cm("exploit", 0.15, 15.0),
        _make_cm("exploit", 0.25, 25.0),
        _make_cm("cma", 0.3, 30.0),
        _make_cm("cma", 0.4, 40.0),
        # One non-finite candidate that should be ignored.
        _make_cm("exploit", float("nan"), float("nan")),
    ]

    frontier_mape = _mean_finite([c.abs_pct_error for c in candidates if c.kind == "frontier"])
    adversarial_mape = _mean_finite([c.abs_pct_error for c in candidates if c.kind == "adversarial"])
    exploit_mape = _mean_finite([c.abs_pct_error for c in candidates if c.kind == "exploit"])
    cma_mape = _mean_finite([c.abs_pct_error for c in candidates if c.kind == "cma"])

    assert math.isclose(frontier_mape, 0.15, rel_tol=1e-9)
    assert math.isclose(adversarial_mape, 0.5, rel_tol=1e-9)
    # Exploit nan is dropped; mean over the three finite values 0.05/0.15/0.25.
    assert math.isclose(exploit_mape, 0.15, rel_tol=1e-9)
    assert math.isclose(cma_mape, 0.35, rel_tol=1e-9)

    # MAE rollups too.
    exploit_mae = _mean_finite([c.abs_error for c in candidates if c.kind == "exploit"])
    cma_mae = _mean_finite([c.abs_error for c in candidates if c.kind == "cma"])
    assert math.isclose(exploit_mae, 15.0, rel_tol=1e-9)
    assert math.isclose(cma_mae, 35.0, rel_tol=1e-9)


def test_mean_finite_handles_all_nan() -> None:
    assert _mean_finite([float("nan"), float("nan")]) is None


def test_mean_finite_handles_empty() -> None:
    assert _mean_finite([]) is None


def test_candidate_metric_has_kind_field_with_new_kinds() -> None:
    # Schema spot-check: CandidateMetric.kind must accept the new "exploit"/"cma"
    # tags without complaint (it's a free-form str, so we just verify the round
    # trip preserves the value).
    for k in ("frontier", "adversarial", "exploit", "cma"):
        c = _make_cm(k, 0.1)
        assert c.kind == k
