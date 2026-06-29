"""Unit + golden-replay tests for the panel-gating module.

The golden-replay test reconstructs the validated causal loop from
``analysis/ix_savings_panel/make_panel_heatmap.py`` on the real
``local_workspace_cma_npv_multistart_seed42`` data and asserts the module reproduces
~55% IX savings at 0% EMV-optimum shortfall. It is skipped when that workspace is absent.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest

from orchestrator.gating import (
    GeologyGatingConfig,
    compute_emv_hat,
    compute_panel,
    rank_top_m,
    slice_manifest,
)

ALL15 = list(range(15))


# --------------------------------------------------------------------------- config
def test_config_defaults_disabled():
    g = GeologyGatingConfig()
    assert g.enabled is False
    assert g.mape_cutoff_pct == 5.0 and g.top_m == 3 and g.warmup_iters == 1
    assert g.audit_configs == 0 and g.mape_key == "mape"


def test_config_audit_not_implemented_when_enabled():
    with pytest.raises(NotImplementedError):
        GeologyGatingConfig.from_config(
            {"geology_gating": {"enabled": True, "audit_configs": 1}})


def test_config_from_block_and_unknown_keys_ignored():
    cfg = {"geology_gating": {"enabled": True, "mape_cutoff_pct": 6.0, "top_m": 5, "bogus": 1}}
    g = GeologyGatingConfig.from_config(cfg)
    assert g.enabled and g.mape_cutoff_pct == 6.0 and g.top_m == 5


def test_config_absent_block_is_disabled_noop():
    assert GeologyGatingConfig.from_config({}).enabled is False
    assert GeologyGatingConfig.from_config(None).enabled is False


def test_config_validation_rejects_bad_when_enabled():
    with pytest.raises(ValueError):
        GeologyGatingConfig.from_config({"geology_gating": {"enabled": True, "top_m": 0}})
    # disabled config is never validated
    GeologyGatingConfig.from_config({"geology_gating": {"enabled": False, "top_m": 0}})


# --------------------------------------------------------------------------- compute_panel
def _hist(*recs):
    """Build a per_geology_history (oldest->newest) from {geo_idx: mape_fraction} dicts.

    Writes both "mape" and "mape_floored" (equal here) so the test is robust to the gate's
    configured mape_key; the production rollup persists both keys too.
    """
    return [{str(g): {"mape": v, "mape_floored": v} for g, v in rec.items()} for rec in recs]


def test_panel_disabled_is_full():
    g = GeologyGatingConfig(enabled=False)
    d = compute_panel(per_geology_history=[], all_geology_indices=ALL15, iteration=5, gcfg=g)
    assert d.is_full and d.panel == set(ALL15) and d.reason == "disabled"


def test_panel_warmup_forces_full():
    g = GeologyGatingConfig(enabled=True, warmup_iters=2)
    d = compute_panel(per_geology_history=_hist({i: 0.01 for i in ALL15}),
                      all_geology_indices=ALL15, iteration=1, gcfg=g)
    assert d.is_full and d.reason == "warmup"


def test_panel_cold_start_no_history_full():
    g = GeologyGatingConfig(enabled=True, warmup_iters=0)
    d = compute_panel(per_geology_history=[None, {}], all_geology_indices=ALL15,
                      iteration=3, gcfg=g)
    assert d.is_full and d.reason == "cold_start"


def test_panel_gates_on_last_mape_with_pct_units():
    # tau = 5%; geo at 4% (0.04) -> filled, geo at 6% (0.06) -> panel.
    g = GeologyGatingConfig(enabled=True, warmup_iters=0, mape_cutoff_pct=5.0)
    hist = _hist({0: 0.04, 1: 0.06, 2: 0.05})  # 2 is exactly at cutoff -> NOT > tau -> filled
    d = compute_panel(per_geology_history=hist, all_geology_indices=[0, 1, 2],
                      iteration=1, gcfg=g)
    assert d.panel == {1} and d.reason == "gated" and not d.is_full


def test_panel_uses_most_recent_measurement():
    g = GeologyGatingConfig(enabled=True, warmup_iters=0, mape_cutoff_pct=5.0)
    # geo 0 was hot (0.20) then calibrated (0.02) most recently -> filled.
    hist = _hist({0: 0.20}, {0: 0.02})
    d = compute_panel(per_geology_history=hist, all_geology_indices=[0], iteration=2, gcfg=g)
    assert d.panel == set()


def test_panel_never_measured_geo_is_kept():
    g = GeologyGatingConfig(enabled=True, warmup_iters=0, mape_cutoff_pct=5.0)
    hist = _hist({0: 0.02})  # geo 7 never measured
    d = compute_panel(per_geology_history=hist, all_geology_indices=[0, 7], iteration=2, gcfg=g)
    assert d.panel == {7}


def test_panel_non_contiguous_indices():
    g = GeologyGatingConfig(enabled=True, warmup_iters=0, mape_cutoff_pct=5.0)
    idx = [71, 73, 88]
    hist = _hist({71: 0.10, 73: 0.01, 88: 0.07})
    d = compute_panel(per_geology_history=hist, all_geology_indices=idx, iteration=1, gcfg=g)
    assert d.panel == {71, 88}


# --------------------------------------------------------------------------- compute_emv_hat
def test_emv_hat_formula_mixes_real_and_pred():
    panel = {0, 1}
    real = {"a": {0: 100.0, 1: 200.0}}              # panel reals
    pred = {"a": {0: 999.0, 1: 999.0, 2: 30.0}}     # full-K preds (panel preds ignored)
    out = compute_emv_hat(panel=panel, all_geology_indices=[0, 1, 2],
                          real_by_config=real, pred_by_config=pred)
    assert out["a"] == pytest.approx((100.0 + 200.0 + 30.0) / 3)


def test_emv_hat_empty_panel_is_pred_mean():
    out = compute_emv_hat(panel=set(), all_geology_indices=[0, 1, 2],
                          real_by_config={}, pred_by_config={"a": {0: 3.0, 1: 6.0, 2: 9.0}})
    assert out["a"] == pytest.approx(6.0)


def test_emv_hat_falls_back_to_pred_when_panel_real_missing():
    # geo 1 in panel but its real failed/absent -> use its pred.
    out = compute_emv_hat(panel={0, 1}, all_geology_indices=[0, 1],
                          real_by_config={"a": {0: 10.0}},
                          pred_by_config={"a": {0: 99.0, 1: 20.0}})
    assert out["a"] == pytest.approx((10.0 + 20.0) / 2)


def test_rank_top_m_orders_and_truncates():
    assert rank_top_m({"a": 1.0, "b": 3.0, "c": 2.0}, 2) == ["b", "c"]
    assert rank_top_m({"a": 1.0}, 0) == []


# --------------------------------------------------------------------------- manifest slicing
def _full_manifest():
    return {
        "iteration": 4,
        "snapshots": [
            {"snapshot_id": "s1", "well_config_paths_by_geology": [
                {"geology_index": g} for g in range(5)]},
            {"snapshot_id": "s2", "well_config_paths_by_geology": [
                {"geology_index": g} for g in range(5)]},
        ],
        "snapshot_count": 2,
    }


def test_slice_manifest_panel_keeps_all_snaps_restricts_geos():
    out = slice_manifest(_full_manifest(), geos=[0, 1], gate_phase="panel")
    assert out["snapshot_count"] == 2 and out["gate_phase"] == "panel"
    for snap in out["snapshots"]:
        assert {e["geology_index"] for e in snap["well_config_paths_by_geology"]} == {0, 1}


def test_slice_manifest_completion_keeps_topm_and_fill_geos(tmp_path):
    out_path = tmp_path / "manifest_iter_0004.completion.json"
    out = slice_manifest(_full_manifest(), geos=[2, 3, 4], sids=["s1"],
                         out_path=out_path, gate_phase="completion")
    assert out["snapshot_count"] == 1 and out["gate_phase"] == "completion"
    snap = out["snapshots"][0]
    assert snap["snapshot_id"] == "s1"
    assert {e["geology_index"] for e in snap["well_config_paths_by_geology"]} == {2, 3, 4}
    assert json.loads(out_path.read_text())["snapshot_count"] == 1


# --------------------------------------------------------------------------- golden replay
SEED42 = Path(__file__).resolve().parents[1] / "local_workspace_cma_npv_multistart_seed42"


def _load_iters(root: Path):
    iters = {}
    for f in sorted(glob.glob(str(root / "iter_*/per_candidate_metrics.json"))):
        d = json.load(open(f))
        by: dict[str, dict[int, dict]] = {}
        for c in d["candidates"]:
            by.setdefault(c["snapshot_id"], {})[int(c["geology_index"])] = c
        cfgs = [s for s, m in by.items() if len(m) == 15]
        if not cfgs:
            continue
        real = {s: {g: float(by[s][g]["real_revenue"]) for g in range(15)} for s in cfgs}
        pred = {s: {g: float(by[s][g]["predicted_revenue"]) for g in range(15)} for s in cfgs}
        iters[int(d["iteration"])] = (cfgs, real, pred)
    return iters


@pytest.mark.skipif(not SEED42.exists(), reason="seed42 workspace not present")
def test_golden_replay_matches_validated_savings():
    """Drive compute_panel/compute_emv_hat through the validated causal loop on real data.

    Reproduces analysis/ix_savings_panel/make_panel_heatmap.py: ~55% IX saved, 0% shortfall
    at tau=5%, M=3, with iteration 0 forced full (warmup_iters=1).
    """
    iters = _load_iters(SEED42)
    assert len(iters) >= 10, "expected the full seed42 run"
    ts = sorted(iters)
    gcfg = GeologyGatingConfig(enabled=True, mape_cutoff_pct=5.0, top_m=3, warmup_iters=1)

    history: list[dict] = []
    ix_used = 0
    full_ix = 0
    inc = float("-inf")
    full_best = max(
        sum(real[s].values()) / 15 for _, real, _ in iters.values() for s in real
    )

    def mape_over(sids, geos, real, pred):
        out = {}
        for g in geos:
            errs = [abs(real[s][g] - pred[s][g]) / abs(real[s][g]) for s in sids
                    if real[s][g] != 0]
            if errs:
                m = sum(errs) / len(errs)
                out[str(g)] = {"mape": m, "mape_floored": m}
        return out

    for t in ts:
        cfgs, real, pred = iters[t]
        n = len(cfgs)
        full_ix += n * 15
        d = compute_panel(per_geology_history=history, all_geology_indices=ALL15,
                          iteration=t, gcfg=gcfg)
        panel = d.panel
        fill = [g for g in ALL15 if g not in panel]
        ix_used += n * len(panel)

        real_panel = {s: {g: real[s][g] for g in panel} for s in cfgs}
        emv_hat = compute_emv_hat(panel=panel, all_geology_indices=ALL15,
                                  real_by_config=real_panel, pred_by_config=pred)
        if fill:
            top = rank_top_m(emv_hat, gcfg.top_m)
            ix_used += len(top) * len(fill)
            completed = top
        else:
            completed = cfgs  # full panel: every config is ground-truth

        for s in completed:
            inc = max(inc, sum(real[s].values()) / 15)

        # update last-measured MAPE: panel geos from all configs, fill geos from completed
        measured = mape_over(cfgs, panel, real, pred)
        measured.update(mape_over(completed, fill, real, pred))
        history.append(measured)

    saved = 100 * (1 - ix_used / full_ix)
    shortfall = 100 * (full_best - inc) / full_best
    assert shortfall == pytest.approx(0.0, abs=1e-6), f"shortfall={shortfall}"
    assert 53.0 <= saved <= 57.0, f"saved={saved:.1f}% (expected ~55%)"
