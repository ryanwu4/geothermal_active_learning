"""Ingest INTERSECT outputs back into the AL training set.

Per iteration:
1. Symlink the new IX output H5s into the run-wide ``raw_ix_archive`` dir.
2. Invoke ``preprocess_h5.py`` over the iteration's IX outputs alone — gives us
   a "delta" compiled H5 with only the new cases.
3. Merge the delta groups into the previous compiled H5 to form the new
   training set. (h5py group copy is much faster than re-running preprocess
   over the full archive every iteration.)
4. Match each completed IX run to its manifest snapshot by ``case_id`` (which
   ``preprocess_h5.py`` derives as the source filename stem) so we can compute
   ``predicted_revenue`` vs. ``real_revenue`` calibration metrics.
5. Return a dict of metrics for state + wandb logging.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


class IngestError(RuntimeError):
    pass


@dataclass
class IngestMetrics:
    n_submitted: int
    n_completed: int
    completion_rate: float
    batch_mape: float | None
    batch_signed_pct_bias: float | None
    frontier_mape: float | None
    adversarial_mape: float | None
    best_real_revenue_in_batch: float | None
    best_real_revenue_so_far: float | None
    n_train_samples: int


def _link_ix_outputs_to_archive(ix_output_dir: Path, archive_dir: Path) -> list[Path]:
    """Symlink each new IX output H5 into the run-wide archive. Returns archive paths."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    linked: list[Path] = []
    for src in sorted(ix_output_dir.glob("*.h5")):
        dest = archive_dir / src.name
        if not dest.exists():
            os.symlink(src.resolve(), dest)
        linked.append(dest)
    return linked


def _run_preprocess_h5(
    surrogate_repo: Path,
    input_dir: Path,
    output_h5: Path,
    norm_config: Path,
    economics_config: Path | None = None,
    workers: int = 4,
    log_path: Path | None = None,
) -> None:
    """Run preprocess_h5.py over ``input_dir`` and emit ``output_h5``."""
    cmd = [
        sys.executable,
        str(Path(surrogate_repo) / "preprocess_h5.py"),
        "--input-dir", str(input_dir),
        "--output-h5", str(output_h5),
        "--norm-config", str(norm_config),
        "--workers", str(workers),
    ]
    if economics_config is not None:
        cmd.extend(["--economics-config", str(economics_config)])

    log_handle = open(log_path, "w") if log_path else None
    try:
        proc = subprocess.run(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT if log_handle else None,
            check=False,
            cwd=str(surrogate_repo),
        )
    finally:
        if log_handle:
            log_handle.close()
    if proc.returncode != 0:
        raise IngestError(f"preprocess_h5.py failed with code {proc.returncode}; see {log_path}")


def _merge_compiled_h5s(prior_h5: Path, delta_h5: Path, out_h5: Path) -> None:
    """Concatenate two compiled H5s into ``out_h5`` by copying groups.

    Top-level attrs (norm_*, target_*) come from ``prior_h5``. Groups present
    in ``delta_h5`` overwrite same-name groups in ``prior_h5`` (newer wins,
    relevant only if a case_id collision occurred — shouldn't happen given the
    iter token in IX output names).
    """
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()

    with h5py.File(prior_h5, "r") as prior, \
         h5py.File(delta_h5, "r") as delta, \
         h5py.File(out_h5, "w") as out:

        # Carry forward top-level attributes from prior (norm bounds + economics).
        for k in prior.attrs:
            out.attrs[k] = prior.attrs[k]

        delta_keys = set(delta.keys())
        for key in prior.keys():
            if key in delta_keys:
                # Defer to delta (newer recompute may differ slightly).
                continue
            prior.copy(key, out)
        for key in delta.keys():
            delta.copy(key, out)


def _read_real_revenue(compiled_h5: Path, case_id: str) -> float | None:
    """Pull the IX-true discounted revenue for one case from a compiled H5."""
    with h5py.File(compiled_h5, "r") as f:
        if case_id not in f:
            return None
        ds = f[case_id].get("field_discounted_net_revenue")
        if ds is None:
            return None
        return float(np.asarray(ds))


def _count_cases(compiled_h5: Path) -> int:
    with h5py.File(compiled_h5, "r") as f:
        return len(list(f.keys()))


def ingest_iteration(
    *,
    iteration: int,
    surrogate_repo: Path,
    norm_config_path: Path,
    economics_config: Path | None,
    array_tasks_json: Path,
    ix_output_dir: Path,
    raw_ix_archive: Path,
    delta_h5: Path,
    prior_compiled_h5: Path,
    next_compiled_h5: Path,
    log_path: Path | None = None,
    prior_best_revenue: float | None = None,
    workers: int = 4,
) -> IngestMetrics:
    """Run the full ingest pipeline for one iteration; return metrics."""
    if not array_tasks_json.exists():
        raise IngestError(f"Missing tasks JSON: {array_tasks_json}")

    with open(array_tasks_json, "r") as f:
        tasks_payload = json.load(f)
    tasks = tasks_payload.get("tasks", [])
    n_submitted = len(tasks)

    # Step 1: archive new outputs.
    linked = _link_ix_outputs_to_archive(ix_output_dir, raw_ix_archive)
    # We'll consult ``ix_output_dir`` directly for the iteration's delta (since
    # everything new lands there before the symlink-archive step).

    # Step 2: preprocess only the iteration's outputs into a delta H5.
    _run_preprocess_h5(
        surrogate_repo=surrogate_repo,
        input_dir=ix_output_dir,
        output_h5=delta_h5,
        norm_config=norm_config_path,
        economics_config=economics_config,
        workers=workers,
        log_path=log_path,
    )

    # Step 3: merge into the new compiled H5.
    _merge_compiled_h5s(prior_compiled_h5, delta_h5, next_compiled_h5)
    n_train_samples = _count_cases(next_compiled_h5)

    # Step 4: match outputs back to manifest snapshots and compute metrics.
    pred_real_pairs: list[tuple[float, float, str]] = []  # (predicted, real, kind)
    n_completed = 0
    for task in tasks:
        out_name = task.get("output_file_name", "")
        case_id = Path(out_name).stem
        if not out_name:
            continue
        produced = ix_output_dir / out_name
        if not produced.exists():
            continue
        n_completed += 1
        real = _read_real_revenue(delta_h5, case_id)
        if real is None:
            continue
        predicted = float(task.get("predicted_discounted_total_revenue", float("nan")))
        # task doesn't carry "kind" — read from snapshot json if available.
        kind = "frontier"
        snap_json = task.get("snapshot_json_path", "")
        if snap_json:
            try:
                with open(snap_json, "r") as f:
                    payload = json.load(f)
                kind = payload.get("kind", kind)
            except Exception:
                pass
        pred_real_pairs.append((predicted, real, kind))

    completion_rate = (n_completed / n_submitted) if n_submitted > 0 else 0.0

    def _safe_mape(rows: list[tuple[float, float, str]]) -> float | None:
        if not rows:
            return None
        diffs = []
        for p, r, _ in rows:
            if r == 0 or not np.isfinite(p) or not np.isfinite(r):
                continue
            diffs.append(abs(p - r) / abs(r))
        return float(np.mean(diffs)) if diffs else None

    def _safe_signed_bias(rows: list[tuple[float, float, str]]) -> float | None:
        if not rows:
            return None
        diffs = []
        for p, r, _ in rows:
            if r == 0 or not np.isfinite(p) or not np.isfinite(r):
                continue
            diffs.append((p - r) / abs(r))
        return float(np.mean(diffs)) if diffs else None

    batch_mape = _safe_mape(pred_real_pairs)
    batch_signed_pct_bias = _safe_signed_bias(pred_real_pairs)
    frontier_mape = _safe_mape([row for row in pred_real_pairs if row[2] == "frontier"])
    adversarial_mape = _safe_mape([row for row in pred_real_pairs if row[2] == "adversarial"])

    real_values = [r for _, r, _ in pred_real_pairs if np.isfinite(r)]
    best_in_batch = float(max(real_values)) if real_values else None
    best_so_far: float | None
    if best_in_batch is None:
        best_so_far = prior_best_revenue
    elif prior_best_revenue is None:
        best_so_far = best_in_batch
    else:
        best_so_far = max(prior_best_revenue, best_in_batch)

    return IngestMetrics(
        n_submitted=n_submitted,
        n_completed=n_completed,
        completion_rate=completion_rate,
        batch_mape=batch_mape,
        batch_signed_pct_bias=batch_signed_pct_bias,
        frontier_mape=frontier_mape,
        adversarial_mape=adversarial_mape,
        best_real_revenue_in_batch=best_in_batch,
        best_real_revenue_so_far=best_so_far,
        n_train_samples=n_train_samples,
    )
