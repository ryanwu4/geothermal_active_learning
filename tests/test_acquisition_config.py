"""Schema-only tests for the new 4-kind knobs on ``AcquisitionConfig`` and the
canonical ordering constants on ``orchestrator.acquire``.
"""
from __future__ import annotations

from pathlib import Path

from orchestrator.acquire import (
    AcquisitionConfig,
    _KIND_ORDER,
    _START_KIND_TO_SAFE_KIND,
)


def _minimal_cfg(**overrides) -> AcquisitionConfig:
    defaults = dict(
        surrogate_repo=Path("/tmp/surrogate_repo"),
        checkpoint_path=Path("/tmp/ckpt.pt"),
        scaler_path=Path("/tmp/scaler.pkl"),
        norm_config_path=Path("/tmp/norm.json"),
        geologies=[],
        wells=[],
        n_starts_per_geology=4,
        k_safe=20,
        k_adv=40,
        adv_fraction=0.15,
        edge_buffer=4,
        learning_rate=0.5,
        log_every_n_steps=5,
        revenue_target="graph_discounted_net_revenue",
    )
    defaults.update(overrides)
    return AcquisitionConfig(**defaults)


def test_acquisition_config_defaults() -> None:
    cfg = _minimal_cfg()
    assert cfg.n_elite_per_geology == 0
    assert cfg.n_cma_per_geology == 0
    assert cfg.elite_top_k == 10
    assert cfg.elite_seed_noise == 2.0
    assert cfg.cma_popsize == 16
    assert cfg.cma_generations == 10
    assert cfg.cma_sigma_init == 5.0
    assert cfg.prior_metrics is None


def test_acquisition_config_accepts_4kind_overrides() -> None:
    cfg = _minimal_cfg(
        n_elite_per_geology=3,
        n_cma_per_geology=2,
        elite_top_k=5,
        elite_seed_noise=1.5,
        cma_popsize=8,
        cma_generations=4,
        cma_sigma_init=2.5,
        prior_metrics=[Path("/tmp/iter_0000/per_candidate_metrics.json")],
    )
    assert cfg.n_elite_per_geology == 3
    assert cfg.n_cma_per_geology == 2
    assert cfg.elite_top_k == 5
    assert cfg.elite_seed_noise == 1.5
    assert cfg.cma_popsize == 8
    assert cfg.cma_generations == 4
    assert cfg.cma_sigma_init == 2.5
    assert cfg.prior_metrics is not None
    assert len(cfg.prior_metrics) == 1


def test_start_kind_to_safe_kind_map() -> None:
    assert _START_KIND_TO_SAFE_KIND == {
        "lhs": "frontier",
        "elite": "exploit",
        "cma": "cma",
    }


def test_kind_order_canonical() -> None:
    assert _KIND_ORDER == {
        "frontier": 0,
        "adversarial": 1,
        "exploit": 2,
        "cma": 3,
    }
