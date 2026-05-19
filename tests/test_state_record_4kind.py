"""Tests for the new exploit/cma fields on ``IterationRecord``."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from orchestrator.state import IterationRecord, RunState


def test_iteration_record_new_fields(tmp_path: Path) -> None:
    rec = IterationRecord(
        iteration=3,
        submitted=10,
        completed=10,
        best_real_revenue=1.5e14,
        batch_mape=0.04,
        exploit_mape=0.07,
        exploit_mae=2.3e13,
        cma_mape=0.11,
        cma_mae=3.4e13,
    )
    # asdict should include the new fields.
    d = asdict(rec)
    assert d["exploit_mape"] == 0.07
    assert d["exploit_mae"] == 2.3e13
    assert d["cma_mape"] == 0.11
    assert d["cma_mae"] == 3.4e13

    # Full save → load round-trip should preserve them.
    state = RunState(
        run_id="r", config_path="c", wandb_run_id="w", iteration=3,
        history=[rec],
    )
    path = tmp_path / "state.json"
    state.save(path)

    loaded = RunState.load(path)
    assert len(loaded.history) == 1
    loaded_rec = loaded.history[0]
    assert loaded_rec.exploit_mape == 0.07
    assert loaded_rec.exploit_mae == 2.3e13
    assert loaded_rec.cma_mape == 0.11
    assert loaded_rec.cma_mae == 3.4e13


def test_iteration_record_backward_compat_load(tmp_path: Path) -> None:
    """Loading a state.json written before the 4-kind fields were added must
    not crash; the new fields should come back as ``None``.
    """
    legacy_payload = {
        "run_id": "r",
        "config_path": "c",
        "wandb_run_id": "w",
        "iteration": 1,
        "target": "graph_discounted_net_revenue",
        "current_checkpoint": None,
        "current_scaler": None,
        "current_compiled_h5": None,
        "norm_config_path": None,
        "schema_version": 1,
        "history": [
            {
                "iteration": 0,
                "submitted": 5,
                "completed": 5,
                "best_real_revenue": 1.0,
                # Deliberately omit exploit_mape/exploit_mae/cma_mape/cma_mae.
            },
        ],
    }
    path = tmp_path / "legacy_state.json"
    path.write_text(json.dumps(legacy_payload))

    loaded = RunState.load(path)
    assert len(loaded.history) == 1
    rec = loaded.history[0]
    assert rec.exploit_mape is None
    assert rec.exploit_mae is None
    assert rec.cma_mape is None
    assert rec.cma_mae is None
    # Other fields should also load correctly.
    assert rec.iteration == 0
    assert rec.submitted == 5
    assert rec.best_real_revenue == 1.0


def test_iteration_record_defaults_are_none() -> None:
    rec = IterationRecord(iteration=0)
    assert rec.exploit_mape is None
    assert rec.exploit_mae is None
    assert rec.cma_mape is None
    assert rec.cma_mae is None
