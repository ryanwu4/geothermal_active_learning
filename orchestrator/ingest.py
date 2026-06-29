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

from .emv import strict_emv


class IngestError(RuntimeError):
    pass


@dataclass
class CandidateMetric:
    """Per-candidate calibration row, persisted for post-run analysis."""
    snapshot_id: str
    case_id: str
    output_file_name: str
    geology_index: int
    geology_config_id: int | str | None
    kind: str  # "frontier" | "adversarial" | "exploit" | "cma"
    predicted_revenue: float
    real_revenue: float
    abs_pct_error: float
    signed_pct_error: float
    abs_error: float = float("nan")
    # Floored APE: abs(pred-real) / max(abs(real), denom_floor). The denom floor
    # prevents geo-8-style "MAPE blows up because true revenue is small" artifacts.
    abs_pct_error_floored: float = float("nan")
    # Gating provenance: "full" (ungated / warmup full ensemble), "panel" (phase-1 panel
    # IX), or "completion" (phase-2 top-M completion). Lets the completion phase union rows
    # by source and lets post-hoc analysis separate panel from completion IX.
    provenance: str = "full"
    # Ensemble-mode seed/terminal tracking. None on per-geology rows. The
    # acquisition writes the predicted half; ingest populates the real half by
    # looking up the prior iter's per_candidate_emv for the seed snapshot.
    seed_source_snapshot_id: str | None = None
    seed_source_iteration: int | None = None
    seed_predicted_emv: float | None = None
    terminal_predicted_emv: float | None = None
    seed_real_emv: float | None = None
    terminal_real_emv: float | None = None


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
    # Absolute-error and floored-MAPE counterparts (added to expose whether a
    # MAPE blow-up reflects real model regression or just a small denominator).
    batch_mae: float | None = None
    batch_mape_floored: float | None = None
    frontier_mae: float | None = None
    adversarial_mae: float | None = None
    # 4-kind per-kind rollups (added with the elite/cma acquisition paths).
    exploit_mape: float | None = None
    exploit_mae: float | None = None
    cma_mape: float | None = None
    cma_mae: float | None = None
    # New: per-candidate rows + per-geology rollups for richer post-run plots.
    candidates: list[CandidateMetric] | None = None
    per_geology: dict[str, dict[str, float | int | None]] | None = None
    # Outcome counters from the status-JSON pass (added 2026-05-11). Sum of these
    # plus n_completed equals n_submitted. Tracking them separately keeps
    # silent-failure modes distinguishable from a stuck SLURM queue.
    n_failed_per_status: int = 0
    n_succeeded_but_missing_h5: int = 0
    n_missing_silently: int = 0
    # Ensemble-mode aggregates (None on per-geology iterations).
    per_candidate_emv: dict[str, float] | None = None
    best_emv_in_batch: float | None = None
    best_emv_so_far: float | None = None
    exploit_best_emv: float | None = None
    exploit_best_per_geology: dict[str, float] | None = None


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


def _mean_finite(vals: list[float]) -> float | None:
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def rollup_metrics(
    candidates: list[CandidateMetric],
    *,
    n_submitted: int,
    n_completed: int,
    n_train_samples: int,
    prior_best_revenue: float | None = None,
    prior_best_emv: float | None = None,
    prior_per_candidate_emv_by_iter: dict[int, dict[str, float]] | None = None,
    expected_k: int | None = None,
    n_failed_per_status: int = 0,
    n_succeeded_but_missing_h5: int = 0,
    n_missing_silently: int = 0,
) -> IngestMetrics:
    """Compute batch / per-geology / ensemble metrics from already-built candidate rows.

    Factored out of :func:`ingest_iteration` so the gated completion phase can recompute
    over a UNION of panel + completion rows without re-reading any H5. Mutates each
    candidate's ``abs_pct_error_floored`` (recomputed against the cohort median).

    ``expected_k``: when given, overrides the inferred ensemble size. Gating passes the
    configured K (e.g. 15) so strict EMV admits ONLY fully-completed configs — panel-only
    configs (with < K finite-geology rows) are correctly excluded from the incumbent, even
    on a phase where every config is partial (which would otherwise infer a too-small K).
    """
    completion_rate = (n_completed / n_submitted) if n_submitted > 0 else 0.0

    batch_mape = _mean_finite([c.abs_pct_error for c in candidates])
    batch_signed_pct_bias = _mean_finite([c.signed_pct_error for c in candidates])
    frontier_mape = _mean_finite([c.abs_pct_error for c in candidates if c.kind == "frontier"])
    adversarial_mape = _mean_finite([c.abs_pct_error for c in candidates if c.kind == "adversarial"])
    exploit_mape = _mean_finite([c.abs_pct_error for c in candidates if c.kind == "exploit"])
    cma_mape = _mean_finite([c.abs_pct_error for c in candidates if c.kind == "cma"])

    # Floored MAPE: divide by max(|real|, 0.1 * cohort_median_real). Lets MAPE remain
    # interpretable on the geo-8-like cohort with small true values.
    finite_reals = [abs(c.real_revenue) for c in candidates if np.isfinite(c.real_revenue) and c.real_revenue != 0]
    cohort_median_real = float(np.median(finite_reals)) if finite_reals else 0.0
    floor_denom = 0.1 * cohort_median_real
    for c in candidates:
        if not np.isfinite(c.real_revenue) or not np.isfinite(c.predicted_revenue):
            c.abs_pct_error_floored = float("nan")
            continue
        denom = max(abs(c.real_revenue), floor_denom) if floor_denom > 0 else abs(c.real_revenue)
        c.abs_pct_error_floored = float(abs(c.predicted_revenue - c.real_revenue) / denom) if denom > 0 else float("nan")
    batch_mape_floored = _mean_finite([c.abs_pct_error_floored for c in candidates])
    batch_mae = _mean_finite([c.abs_error for c in candidates])
    frontier_mae = _mean_finite([c.abs_error for c in candidates if c.kind == "frontier"])
    adversarial_mae = _mean_finite([c.abs_error for c in candidates if c.kind == "adversarial"])
    exploit_mae = _mean_finite([c.abs_error for c in candidates if c.kind == "exploit"])
    cma_mae = _mean_finite([c.abs_error for c in candidates if c.kind == "cma"])

    # Per-geology rollup — useful for spotting geologies the surrogate
    # systematically struggles on, and for the convergence plot script.
    per_geology: dict[str, dict[str, float | int | None]] = {}
    for c in candidates:
        bucket = per_geology.setdefault(str(c.geology_index), {
            "count": 0,
            "mape": [],
            "mape_floored": [],
            "mae": [],
            "signed_bias": [],
            "max_real_revenue": None,
        })
        bucket["count"] += 1  # type: ignore[operator]
        if np.isfinite(c.abs_pct_error):
            bucket["mape"].append(c.abs_pct_error)  # type: ignore[union-attr]
        if np.isfinite(c.abs_pct_error_floored):
            bucket["mape_floored"].append(c.abs_pct_error_floored)  # type: ignore[union-attr]
        if np.isfinite(c.abs_error):
            bucket["mae"].append(c.abs_error)  # type: ignore[union-attr]
        if np.isfinite(c.signed_pct_error):
            bucket["signed_bias"].append(c.signed_pct_error)  # type: ignore[union-attr]
        if np.isfinite(c.real_revenue):
            cur = bucket["max_real_revenue"]
            bucket["max_real_revenue"] = (
                c.real_revenue if cur is None else max(float(cur), c.real_revenue)  # type: ignore[arg-type]
            )
    # Collapse list-valued aggregates to scalars.
    for g, bucket in per_geology.items():
        bucket["mape"] = _mean_finite(bucket["mape"])  # type: ignore[arg-type]
        bucket["mape_floored"] = _mean_finite(bucket["mape_floored"])  # type: ignore[arg-type]
        bucket["mae"] = _mean_finite(bucket["mae"])  # type: ignore[arg-type]
        bucket["signed_bias"] = _mean_finite(bucket["signed_bias"])  # type: ignore[arg-type]

    real_values = [c.real_revenue for c in candidates if np.isfinite(c.real_revenue)]
    best_in_batch = float(max(real_values)) if real_values else None
    best_so_far: float | None
    if best_in_batch is None:
        best_so_far = prior_best_revenue
    elif prior_best_revenue is None:
        best_so_far = best_in_batch
    else:
        best_so_far = max(prior_best_revenue, best_in_batch)

    # ----- Ensemble-mode aggregates -----
    grouped_by_snap: dict[str, list[CandidateMetric]] = {}
    for c in candidates:
        if c.snapshot_id:
            grouped_by_snap.setdefault(c.snapshot_id, []).append(c)
    # STRICT EMV (see orchestrator.emv): a config's EMV is defined only if ALL K geologies
    # produced finite revenue. K defaults to the largest per-snapshot geology count this
    # iteration, but a caller may pass ``expected_k`` to pin it (gating: the configured
    # ensemble size, so partial panel-only configs cannot masquerade as complete).
    if expected_k is None:
        expected_k = max((len(rows) for rows in grouped_by_snap.values()), default=0)
    per_candidate_emv: dict[str, float] = {}
    for sid, rows in grouped_by_snap.items():
        by_geo = {r.geology_index: r.real_revenue for r in rows if np.isfinite(r.real_revenue)}
        emv_val = strict_emv(by_geo.values(), expected_k=expected_k)
        if emv_val is not None:
            per_candidate_emv[sid] = float(emv_val)

    # Stamp terminal_real_emv onto every row whose snapshot has a populated EMV.
    for c in candidates:
        if c.snapshot_id and c.snapshot_id in per_candidate_emv:
            c.terminal_real_emv = per_candidate_emv[c.snapshot_id]

    # Look up seed real EMV from prior iters' per_candidate_emv.
    if prior_per_candidate_emv_by_iter:
        for c in candidates:
            if not c.seed_source_snapshot_id or c.seed_source_iteration is None:
                continue
            src_emv_map = prior_per_candidate_emv_by_iter.get(int(c.seed_source_iteration))
            if not src_emv_map:
                continue
            v = src_emv_map.get(str(c.seed_source_snapshot_id))
            if v is not None and np.isfinite(v):
                c.seed_real_emv = float(v)

    is_ensemble_iter = any(len(rows) > 1 for rows in grouped_by_snap.values())
    best_emv_in_batch: float | None = None
    best_emv_so_far_val: float | None = None
    exploit_best_emv: float | None = None
    exploit_best_per_geology: dict[str, float] | None = None
    if is_ensemble_iter:
        emv_values = list(per_candidate_emv.values())
        if emv_values:
            best_emv_in_batch = float(max(emv_values))
        if best_emv_in_batch is None:
            best_emv_so_far_val = prior_best_emv
        elif prior_best_emv is None:
            best_emv_so_far_val = best_emv_in_batch
        else:
            best_emv_so_far_val = max(float(prior_best_emv), best_emv_in_batch)

        # Exploit-cohort restriction.
        exploit_sids = {c.snapshot_id for c in candidates if c.kind == "exploit" and c.snapshot_id}
        exploit_emvs = [per_candidate_emv[s] for s in exploit_sids if s in per_candidate_emv]
        if exploit_emvs:
            exploit_best_emv = float(max(exploit_emvs))
        exploit_per_geo: dict[str, float] = {}
        for c in candidates:
            if c.kind != "exploit" or not np.isfinite(c.real_revenue):
                continue
            key = str(c.geology_index)
            cur = exploit_per_geo.get(key)
            exploit_per_geo[key] = c.real_revenue if cur is None else max(cur, c.real_revenue)
        exploit_best_per_geology = exploit_per_geo or None

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
        batch_mae=batch_mae,
        batch_mape_floored=batch_mape_floored,
        frontier_mae=frontier_mae,
        adversarial_mae=adversarial_mae,
        exploit_mape=exploit_mape,
        exploit_mae=exploit_mae,
        cma_mape=cma_mape,
        cma_mae=cma_mae,
        candidates=candidates,
        per_geology=per_geology,
        n_failed_per_status=n_failed_per_status,
        n_succeeded_but_missing_h5=n_succeeded_but_missing_h5,
        n_missing_silently=n_missing_silently,
        per_candidate_emv=(per_candidate_emv if is_ensemble_iter else None),
        best_emv_in_batch=best_emv_in_batch,
        best_emv_so_far=best_emv_so_far_val,
        exploit_best_emv=exploit_best_emv,
        exploit_best_per_geology=exploit_best_per_geology,
    )


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
    prior_best_emv: float | None = None,
    prior_per_candidate_emv_by_iter: dict[int, dict[str, float]] | None = None,
    workers: int = 4,
    provenance: str = "full",
    expected_k: int | None = None,
) -> IngestMetrics:
    """Run the full ingest pipeline for one iteration; return metrics.

    ``provenance`` stamps each candidate row ("full"/"panel"/"completion") for the gated
    two-phase loop. ``expected_k`` pins the strict-EMV ensemble size (gating passes the
    configured K so panel-only configs are excluded from the incumbent); ``None`` infers it.
    """
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
    # Status JSONs live at <stage_run_dir>/status/task_NNNNNN.json; the worker
    # writes one per task. We use them to distinguish:
    #   - completed_with_h5:        success=true + h5 file present (normal)
    #   - failed_per_status:        success=false (worker raised; surface it)
    #   - succeeded_but_missing_h5: success=true but h5 absent (loud alarm)
    #   - missing_silently:         no status JSON and no h5 (worker never ran)
    status_dir = array_tasks_json.parent.parent / "status"
    n_failed_per_status = 0
    n_succeeded_but_missing_h5 = 0
    n_missing_silently = 0
    candidates: list[CandidateMetric] = []
    n_completed = 0
    for task in tasks:
        out_name = task.get("output_file_name", "")
        case_id = Path(out_name).stem
        if not out_name:
            continue
        produced = ix_output_dir / out_name
        task_id = task.get("task_id")
        status_path = status_dir / f"task_{int(task_id):06d}.json" if task_id is not None else None
        status_payload: dict | None = None
        if status_path is not None and status_path.exists():
            try:
                with open(status_path, "r") as f:
                    status_payload = json.load(f)
            except Exception as e:
                print(f"[ingest] WARN: failed to parse status JSON {status_path}: {e}")
        if not produced.exists():
            if status_payload is None:
                n_missing_silently += 1
                print(f"[ingest] MISSING_SILENTLY case_id={case_id} (no status JSON, no h5; worker never ran or output never staged)")
            elif status_payload.get("success") is False:
                n_failed_per_status += 1
                phase = status_payload.get("phase", "?")
                err = (status_payload.get("error") or "")[:300]
                print(f"[ingest] FAILED_PER_STATUS case_id={case_id} phase={phase} error={err}")
            else:
                n_succeeded_but_missing_h5 += 1
                print(f"[ingest] SUCCEEDED_BUT_MISSING_H5 case_id={case_id} status_path={status_path} — worker claimed success but output is absent; this is the IsWell-class symptom and should be investigated")
            continue
        n_completed += 1
        real = _read_real_revenue(delta_h5, case_id)
        if real is None:
            continue
        predicted = float(task.get("predicted_discounted_total_revenue", float("nan")))

        # Pull `kind` from the snapshot JSON the orchestrator wrote during
        # acquisition (the task JSON itself doesn't carry it). Ensemble mode
        # also stores seed metadata inside the snapshot JSON which we surface
        # onto each CandidateMetric row.
        kind = "frontier"
        seed_source_snapshot_id: str | None = None
        seed_source_iteration: int | None = None
        seed_predicted_emv: float | None = None
        terminal_predicted_emv: float | None = None
        snap_json = task.get("snapshot_json_path", "")
        if snap_json:
            try:
                with open(snap_json, "r") as f:
                    payload = json.load(f)
                kind = payload.get("kind", kind)
                seed_source_snapshot_id = payload.get("seed_source_snapshot_id")
                _ssi = payload.get("seed_source_iteration")
                seed_source_iteration = int(_ssi) if _ssi is not None else None
                _spe = payload.get("seed_predicted_emv")
                seed_predicted_emv = float(_spe) if _spe is not None and np.isfinite(_spe) else None
                _tpe = payload.get("terminal_predicted_emv", payload.get("predicted_emv"))
                terminal_predicted_emv = float(_tpe) if _tpe is not None and np.isfinite(_tpe) else None
            except Exception as e:
                # In ensemble mode this is the only place seed/terminal metadata
                # lands on each row; silently swallowing the parse failure means
                # the seed-vs-terminal scatters lose data with no audit trail.
                print(f"[ingest] WARN: failed to parse snapshot JSON {snap_json}: {e}",
                      flush=True)

        if real == 0 or not np.isfinite(predicted) or not np.isfinite(real):
            abs_pct = float("nan")
            signed_pct = float("nan")
            abs_err = float("nan")
        else:
            abs_pct = abs(predicted - real) / abs(real)
            signed_pct = (predicted - real) / abs(real)
            abs_err = abs(predicted - real)

        candidates.append(CandidateMetric(
            snapshot_id=str(task.get("snapshot_id", "")),
            case_id=case_id,
            output_file_name=out_name,
            geology_index=int(task.get("geology_index", -1)),
            geology_config_id=task.get("scenario") or task.get("geology_config_id"),
            kind=kind,
            predicted_revenue=predicted,
            real_revenue=float(real),
            abs_pct_error=float(abs_pct),
            signed_pct_error=float(signed_pct),
            abs_error=float(abs_err),
            provenance=provenance,
            seed_source_snapshot_id=seed_source_snapshot_id,
            seed_source_iteration=seed_source_iteration,
            seed_predicted_emv=seed_predicted_emv,
            terminal_predicted_emv=terminal_predicted_emv,
        ))

    return rollup_metrics(
        candidates,
        n_submitted=n_submitted,
        n_completed=n_completed,
        n_train_samples=n_train_samples,
        prior_best_revenue=prior_best_revenue,
        prior_best_emv=prior_best_emv,
        prior_per_candidate_emv_by_iter=prior_per_candidate_emv_by_iter,
        expected_k=expected_k,
        n_failed_per_status=n_failed_per_status,
        n_succeeded_but_missing_h5=n_succeeded_but_missing_h5,
        n_missing_silently=n_missing_silently,
    )
