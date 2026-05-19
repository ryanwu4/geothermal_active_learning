"""Persistent state for an active learning run.

State is a single JSON file at ``<run_root>/state.json``. Every phase reads at
start, mutates locally, and writes atomically (write-temp-then-rename) at end.
The chain of SLURM jobs depends on this file outliving any single job; nothing
in process memory is load-bearing across job boundaries.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class IterationRecord:
    iteration: int
    submitted: int = 0
    completed: int = 0
    best_real_revenue: float | None = None
    batch_mape: float | None = None
    batch_signed_pct_bias: float | None = None
    frontier_mape: float | None = None
    adversarial_mape: float | None = None
    # Floored-MAPE and MAE counterparts. The floored MAPE divides by
    # `max(|real|, 0.1 * cohort_median_real)` so geo-8-style small-denominator
    # cases don't artificially balloon the metric. MAE is reported alongside to
    # disambiguate "absolute error worsened" from "denominator shrank."
    batch_mape_floored: float | None = None
    batch_mae: float | None = None
    frontier_mae: float | None = None
    adversarial_mae: float | None = None
    # 4-kind acquisition rollups (added alongside elite/cma kinds). All None on
    # older runs that pre-date the change; that's intentional — plot code uses
    # ``.get(...)`` so missing fields stay invisible rather than rendering as 0.
    exploit_mape: float | None = None
    exploit_mae: float | None = None
    cma_mape: float | None = None
    cma_mae: float | None = None
    n_train_samples: int | None = None
    train_val_loss: float | None = None
    wallclock_acquire_min: float | None = None
    wallclock_train_min: float | None = None
    wallclock_ix_min: float | None = None
    wallclock_ingest_min: float | None = None
    timestamp: str | None = None
    train_acquire_job_id: str | None = None
    ix_array_job_id: str | None = None
    ingest_job_id: str | None = None
    notes: str = ""
    # Per-geology aggregate calibration for this iteration's batch. Keyed by
    # `str(geology_index)` so the JSON round-trips cleanly. Each value carries
    # at least: count, mape, signed_bias, max_real_revenue. Optional — older
    # iterations may have None.
    per_geology: dict[str, dict[str, object]] | None = None


@dataclass
class RunState:
    run_id: str
    config_path: str
    wandb_run_id: str
    iteration: int = 0
    target: str = "graph_discounted_net_revenue"
    current_checkpoint: str | None = None
    current_scaler: str | None = None
    current_compiled_h5: str | None = None
    norm_config_path: str | None = None
    history: list[IterationRecord] = field(default_factory=list)
    schema_version: int = 1

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["history"] = [asdict(h) for h in self.history]
        return d

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunState":
        history = [IterationRecord(**h) for h in payload.get("history", [])]
        kwargs = {k: v for k, v in payload.items() if k != "history"}
        return cls(history=history, **kwargs)

    @classmethod
    def load(cls, path: Path | str) -> "RunState":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a killed job mid-write doesn't corrupt the state.
        with tempfile.NamedTemporaryFile(
            "w", dir=str(path.parent), delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(self.to_dict(), tmp, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, path)

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def get_iter(self, iteration: int) -> IterationRecord | None:
        for rec in self.history:
            if rec.iteration == iteration:
                return rec
        return None

    def upsert_iter(self, rec: IterationRecord) -> None:
        for i, existing in enumerate(self.history):
            if existing.iteration == rec.iteration:
                self.history[i] = rec
                return
        self.history.append(rec)
        self.history.sort(key=lambda r: r.iteration)

    def best_real_revenue_so_far(self) -> float | None:
        best = None
        for rec in self.history:
            if rec.best_real_revenue is None:
                continue
            best = rec.best_real_revenue if best is None else max(best, rec.best_real_revenue)
        return best


def new_run_id(prefix: str = "al") -> str:
    """Generate a sortable, unique run id like ``al_20260503_143012_a1b2c3``."""
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}_{stamp}_{suffix}"


def new_wandb_run_id() -> str:
    return uuid.uuid4().hex[:12]
