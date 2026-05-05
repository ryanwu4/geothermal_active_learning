"""Round-trip and history-helper tests for RunState."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.state import IterationRecord, RunState, new_run_id, new_wandb_run_id


def test_round_trip(tmp_path: Path) -> None:
    state = RunState(
        run_id="al_test",
        config_path="/tmp/cfg.json",
        wandb_run_id="abc123",
        iteration=2,
        target="graph_discounted_net_revenue",
        history=[
            IterationRecord(iteration=0, submitted=10, completed=9, best_real_revenue=1.0e14, batch_mape=0.03),
            IterationRecord(iteration=1, submitted=12, completed=12, best_real_revenue=1.2e14, batch_mape=0.025),
        ],
    )
    path = tmp_path / "state.json"
    state.save(path)
    loaded = RunState.load(path)
    assert loaded.run_id == "al_test"
    assert loaded.iteration == 2
    assert len(loaded.history) == 2
    assert loaded.history[1].best_real_revenue == pytest.approx(1.2e14)


def test_atomic_save_no_partial_file(tmp_path: Path) -> None:
    """Saving must replace atomically; tempfile artifacts shouldn't linger."""
    state = RunState(run_id="al_x", config_path="cfg", wandb_run_id="w")
    path = tmp_path / "state.json"
    state.save(path)
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []
    # File is valid JSON.
    json.loads(path.read_text())


def test_upsert_iter_replaces_existing() -> None:
    state = RunState(run_id="r", config_path="c", wandb_run_id="w")
    state.upsert_iter(IterationRecord(iteration=0, submitted=5))
    state.upsert_iter(IterationRecord(iteration=1, submitted=8))
    state.upsert_iter(IterationRecord(iteration=0, submitted=99))
    assert len(state.history) == 2
    rec0 = state.get_iter(0)
    assert rec0 is not None and rec0.submitted == 99


def test_best_real_revenue_so_far_handles_missing() -> None:
    state = RunState(run_id="r", config_path="c", wandb_run_id="w")
    state.upsert_iter(IterationRecord(iteration=0, best_real_revenue=None))
    state.upsert_iter(IterationRecord(iteration=1, best_real_revenue=1.0))
    state.upsert_iter(IterationRecord(iteration=2, best_real_revenue=0.5))
    assert state.best_real_revenue_so_far() == 1.0


def test_new_run_id_is_unique() -> None:
    a = new_run_id("test")
    b = new_run_id("test")
    assert a != b
    assert a.startswith("test_") and b.startswith("test_")


def test_new_wandb_run_id_is_hex_only() -> None:
    rid = new_wandb_run_id()
    assert all(c in "0123456789abcdef" for c in rid)
    assert len(rid) == 12
