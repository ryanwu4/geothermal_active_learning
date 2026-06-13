"""Regression tests for NPV-objective guards & objective-aware control flow.

Locks in the audit fixes: (1) run_acquisition refuses objective=npv with a non-cma_surrogate
mode; (2) the revenue-plateau stop is skipped in npv mode and its relative-threshold math is
sign-safe; (3) the workspace objective marker refuses a flipped resume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orchestrator.acquire import AcquisitionConfig, run_acquisition
from orchestrator.stop import StoppingConfig, evaluate_stopping, _revenue_plateau
from orchestrator.state import RunState, IterationRecord
from orchestrator.npv_metrics import assert_objective_marker, _row_npv


def _fake_npv_state():
    """Minimal npv_state with stubbed callables (no cube/geology needed) for _row_npv tests."""
    return dict(
        rtop_by_geo={0: 11}, is_injector_list=[True, False], cube=None, fac_surf=None,
        flowline=0.0, vertical_lead_m=1000.0, ksurf=2, terms={},
        compute_angled_well_length=lambda coords, **k: np.ones(np.asarray(coords).shape[0]),
        compute_npv=lambda rev, wl, inj, **k: {"npv": float(rev) - float(np.sum(wl))},
    )


def test_row_npv_well_count_mismatch_returns_none():
    """A snapshot whose well count differs from is_injector_list must yield None, not a wrong NPV."""
    st = _fake_npv_state()  # is_injector_list has 2 wells
    assert _row_npv(st, np.zeros((2, 3)), 0, 4.0e8) is not None      # matching count -> computed
    assert _row_npv(st, np.zeros((3, 3)), 0, 4.0e8) is None          # 3 wells vs 2 -> bail
    assert _row_npv(st, np.zeros((2, 3)), 0, None) is None           # missing revenue
    assert _row_npv(st, np.zeros((2, 3)), 0, float("nan")) is None   # non-finite revenue
    assert _row_npv(st, np.zeros((2, 3)), 99, 4.0e8) is None         # geology not in rtop map


def _cfg(**ov) -> AcquisitionConfig:
    d = dict(
        surrogate_repo=Path("/tmp/sr"), checkpoint_path=Path("/tmp/c.pt"),
        scaler_path=Path("/tmp/s.pkl"), norm_config_path=Path("/tmp/n.json"),
        geologies=[], wells=[], n_starts_per_geology=4, k_safe=20, k_adv=40,
        adv_fraction=0.1, edge_buffer=4, learning_rate=0.5, log_every_n_steps=5,
        revenue_target="graph_discounted_net_revenue",
    )
    d.update(ov)
    return AcquisitionConfig(**d)


def _state(best_revs: list[float]) -> RunState:
    st = RunState(run_id="r", config_path="/tmp/cfg.json", wandb_run_id="w")
    for i, b in enumerate(best_revs):
        st.history.append(IterationRecord(iteration=i, best_real_revenue=b))
    st.iteration = len(best_revs) - 1
    return st


@pytest.mark.parametrize("mode", ["per_geology", "ensemble"])
def test_npv_requires_cma_surrogate(mode, tmp_path):
    """objective=npv with a non-cma_surrogate mode must fail loud (else it'd optimize revenue)."""
    cfg = _cfg(objective="npv", mode=mode)
    with pytest.raises(ValueError, match="cma_surrogate"):
        run_acquisition(cfg, out_dir=tmp_path, iteration=0)


def test_npv_cma_surrogate_passes_guard(tmp_path, monkeypatch):
    """cma_surrogate + npv passes the guard (dispatches into the cma path, which we stub)."""
    import orchestrator.acquire as acq
    called = {}
    monkeypatch.setattr(acq, "_run_acquisition_cma_surrogate",
                        lambda cfg, **k: called.setdefault("ok", True) or {"candidates": []})
    run_acquisition(_cfg(objective="npv", mode="cma_surrogate"), out_dir=tmp_path, iteration=0)
    assert called.get("ok")


def test_revenue_plateau_skipped_in_npv_mode():
    """A flat revenue history triggers the plateau stop in revenue mode but NOT in npv mode."""
    st = _state([1.0e8] * 8)  # fully plateaued
    cfg = StoppingConfig(max_iterations=999, plateau_window=3, plateau_threshold_relative=0.005)
    assert evaluate_stopping(st, cfg, objective="revenue").should_stop is True
    dec = evaluate_stopping(st, cfg, objective="npv")
    assert dec.should_stop is False  # revenue plateau must not fire when optimizing NPV


def test_revenue_plateau_relative_threshold_sign_safe():
    """The plateau math must not crash or invert for negative / zero-crossing best values."""
    # Negative, flat -> plateaued (no crash, no sign inversion).
    assert _revenue_plateau(_state([-2.0e7] * 6), window=3, threshold_rel=0.005) is True
    # Crossing zero with a real improvement -> not a plateau.
    assert _revenue_plateau(_state([-1e7, -5e6, 0.0, 5e6, 1e7, 2e7]), window=3, threshold_rel=0.005) is False
    # All zeros -> safe (no divide-by-zero), treated as non-plateau.
    assert _revenue_plateau(_state([0.0] * 6), window=3, threshold_rel=0.005) is False


def test_objective_marker_blocks_flipped_resume(tmp_path):
    assert_objective_marker(tmp_path, "npv")          # first call pins it
    assert_objective_marker(tmp_path, "npv")          # same -> ok
    with pytest.raises(RuntimeError, match="objective"):
        assert_objective_marker(tmp_path, "revenue")  # flipped -> refuse
