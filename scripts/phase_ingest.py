#!/usr/bin/env python3
"""Phase C: ingest IX outputs, update state, chain the next iteration.

Invoked from ``ingest.sbatch`` after the IX array's ``afterany`` dependency
fires. Reads the run state, runs preprocess+merge to extend the training set,
computes calibration metrics, logs to wandb, evaluates stopping criteria, and
either submits the next ``train_acquire`` or writes a ``done.json`` marker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataclasses import asdict, fields

from orchestrator.ingest import CandidateMetric, ingest_iteration, rollup_metrics
from orchestrator.log import init_run
from orchestrator.paths import resolve_run_paths
from orchestrator.slurm import submit_sbatch, write_rendered
from orchestrator.state import IterationRecord, RunState
from orchestrator.stop import StoppingConfig, evaluate_stopping


_CAND_FIELDS = {f.name for f in fields(CandidateMetric)}


def _reconstruct_candidates(per_cand_json: Path) -> list[CandidateMetric]:
    """Rebuild CandidateMetric rows from a written per_candidate_metrics.json."""
    if not per_cand_json.exists():
        return []
    with open(per_cand_json, "r") as f:
        payload = json.load(f)
    out: list[CandidateMetric] = []
    for row in payload.get("candidates", []):
        out.append(CandidateMetric(**{k: v for k, v in row.items() if k in _CAND_FIELDS}))
    return out


def _union_candidates(
    panel_rows: list[CandidateMetric], completion_rows: list[CandidateMetric]
) -> list[CandidateMetric]:
    """Union by (snapshot_id, geology_index); completion rows win on collision.

    Panel and completion geologies are disjoint per config, so collisions are not
    expected — but completion-wins makes a re-run idempotent and safe regardless.
    """
    by_key: dict[tuple[str, int], CandidateMetric] = {}
    for c in panel_rows:
        by_key[(c.snapshot_id, int(c.geology_index))] = c
    for c in completion_rows:
        by_key[(c.snapshot_id, int(c.geology_index))] = c
    return list(by_key.values())


def _write_per_candidate(per_cand_path: Path, iteration: int, metrics) -> None:
    payload = {
        "iteration": iteration,
        "n_submitted": metrics.n_submitted,
        "n_completed": metrics.n_completed,
        "candidates": [asdict(c) for c in (metrics.candidates or [])],
        "per_candidate_emv": metrics.per_candidate_emv,
        "best_emv_in_batch": metrics.best_emv_in_batch,
        "exploit_best_emv": metrics.exploit_best_emv,
        "exploit_best_per_geology": metrics.exploit_best_per_geology,
    }
    with open(per_cand_path, "w") as f:
        json.dump(payload, f, indent=2)


def _record_metrics(rec: IterationRecord, metrics, ingest_elapsed_min: float) -> None:
    rec.completed = metrics.n_completed
    rec.batch_mape = metrics.batch_mape
    rec.batch_signed_pct_bias = metrics.batch_signed_pct_bias
    rec.frontier_mape = metrics.frontier_mape
    rec.adversarial_mape = metrics.adversarial_mape
    rec.exploit_mape = metrics.exploit_mape
    rec.cma_mape = metrics.cma_mape
    rec.batch_mape_floored = metrics.batch_mape_floored
    rec.batch_mae = metrics.batch_mae
    rec.frontier_mae = metrics.frontier_mae
    rec.adversarial_mae = metrics.adversarial_mae
    rec.exploit_mae = metrics.exploit_mae
    rec.cma_mae = metrics.cma_mae
    rec.best_real_revenue = metrics.best_real_revenue_so_far
    rec.n_train_samples = metrics.n_train_samples
    rec.wallclock_ingest_min = ingest_elapsed_min
    rec.per_geology = metrics.per_geology
    rec.best_emv_in_batch = metrics.best_emv_in_batch
    rec.best_emv_so_far = metrics.best_emv_so_far
    rec.per_candidate_emv = metrics.per_candidate_emv
    rec.exploit_best_emv = metrics.exploit_best_emv
    rec.exploit_best_per_geology = metrics.exploit_best_per_geology


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _stage_iteration_outputs(
    *,
    ix_output_root: Path,
    array_tasks_json: Path,
    iter_scratch_dir: Path,
) -> Path:
    """Symlink only this iteration's IX outputs into a per-iter scratch dir.

    `ix_output_root` is the shared `FILEPATHS.h5s_dir_out` from the Julia
    config — every IX run anyone has ever submitted lands there. We narrow to
    just this iteration's expected `output_file_name`s (from `array_tasks.json`)
    by symlinking them into a scoped dir, so downstream `preprocess_h5.py` only
    ingests the new outputs.

    Returns the path to the per-iteration scratch dir.
    """
    iter_scratch_dir.mkdir(parents=True, exist_ok=True)

    with open(array_tasks_json, "r") as f:
        tasks_payload = json.load(f)
    tasks = tasks_payload.get("tasks", [])

    n_linked = 0
    n_missing = 0
    for task in tasks:
        out_name = task.get("output_file_name", "")
        if not out_name:
            continue
        src = ix_output_root / out_name
        if not src.exists():
            n_missing += 1
            continue
        dest = iter_scratch_dir / out_name
        if dest.exists() or dest.is_symlink():
            continue
        try:
            dest.symlink_to(src.resolve())
            n_linked += 1
        except OSError as e:
            print(f"WARN: failed to symlink {src} -> {dest}: {e}")
    print(
        f"[ingest] scoped {n_linked} IX outputs into {iter_scratch_dir} "
        f"(missing: {n_missing})"
    )
    return iter_scratch_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--array-tasks-json", type=Path, required=True)
    parser.add_argument("--ix-output-dir", type=Path, required=True)
    # Panel-gating two-phase support (orchestrator.gating). "full" = ungated legacy
    # behavior. "panel" = phase-1 partial ingest that records metrics but does NOT run the
    # stopping check or chain the next iteration. "completion" = phase-2 ingest of the top-M
    # completions, merged+unioned with the panel rows, which DOES advance the loop.
    parser.add_argument("--gate-phase", choices=["full", "panel", "completion"], default="full")
    parser.add_argument("--expected-k", type=int, default=None,
                        help="ensemble size K to pin strict EMV (gated phases); None infers it")
    args = parser.parse_args()
    gate_phase = args.gate_phase
    expected_k = args.expected_k

    state = RunState.load(args.run_root / "state.json")
    config = _load_config(Path(state.config_path))

    paths = resolve_run_paths(scratch_root=args.run_root.parent, run_id=args.run_root.name)
    paths.ensure_dirs()

    surrogate_repo = Path(config["paths"]["surrogate_repo"]).resolve()
    norm_config_path = Path(state.norm_config_path) if state.norm_config_path else (
        surrogate_repo / "norm_config.json"
    )
    economics_config = surrogate_repo / "configs" / "economics.json"
    # Acquisition objective (revenue|npv). The ingest itself stays revenue-based (no cube on
    # Sherlock); we only need it to tag reporting and to skip the revenue-plateau stop in npv mode.
    objective = str(config.get("acquisition", {}).get("objective", "revenue"))
    bootstrap_h5 = Path(config["paths"]["bootstrap_compiled_h5"]).resolve()

    # Build prior_per_candidate_emv_by_iter so the ingest can look up real EMV
    # for each ensemble-mode candidate's seed source. Only iterations whose
    # IterationRecord populated per_candidate_emv contribute; per-geology
    # iterations are skipped (their candidates have no seed_source_*).
    prior_per_candidate_emv_by_iter: dict[int, dict[str, float]] = {}
    for rec_hist in state.history:
        if rec_hist.per_candidate_emv:
            prior_per_candidate_emv_by_iter[int(rec_hist.iteration)] = dict(rec_hist.per_candidate_emv)

    per_cand_path = paths.iter_dir(state.iteration) / "per_candidate_metrics.json"
    ingest_started = time.time()

    if gate_phase == "completion":
        # Phase 2: ingest the top-M completions, MERGE into this iteration's PARTIAL compiled
        # H5 (written by the panel phase), then recompute the iteration's metrics over the
        # UNION of panel + completion rows. Now the top-M configs have all K geologies, so
        # strict EMV (expected_k=K) admits them and the incumbent becomes ground-truth.
        partial_compiled = paths.iter_compiled_h5(state.iteration)
        tmp_compiled = partial_compiled.with_name(partial_compiled.stem + "_tmp" + partial_compiled.suffix)
        delta = paths.iter_preprocessed_h5(state.iteration)
        completion_delta = delta.with_name(delta.stem + "_completion" + delta.suffix)
        scratch = paths.iter_ix_output_dir(state.iteration)
        completion_scratch = scratch.with_name(scratch.name + "_completion")
        ix_output_dir = _stage_iteration_outputs(
            ix_output_root=args.ix_output_dir,
            array_tasks_json=args.array_tasks_json,
            iter_scratch_dir=completion_scratch,
        )
        completion_metrics = ingest_iteration(
            iteration=state.iteration,
            surrogate_repo=surrogate_repo,
            norm_config_path=norm_config_path,
            economics_config=economics_config,
            array_tasks_json=args.array_tasks_json,
            ix_output_dir=ix_output_dir,
            raw_ix_archive=paths.raw_ix_archive,
            delta_h5=completion_delta,
            prior_compiled_h5=partial_compiled,
            next_compiled_h5=tmp_compiled,
            log_path=paths.logs_dir / f"ingest_iter_{state.iteration:04d}.completion.preprocess.log",
            prior_best_revenue=state.best_real_revenue_so_far(),
            prior_best_emv=state.best_emv_so_far_value(),
            prior_per_candidate_emv_by_iter=prior_per_candidate_emv_by_iter or None,
            workers=4,
            provenance="completion",
            expected_k=expected_k,
        )
        # Atomically promote the merged H5 over the partial one (separate paths so the
        # _merge_compiled_h5s read-prior/write-next never aliases the same file).
        os.replace(tmp_compiled, partial_compiled)
        next_compiled_h5 = partial_compiled

        # Union with the panel rows from phase 1 and recompute the iteration's metrics.
        panel_payload: dict = {}
        if per_cand_path.exists():
            with open(per_cand_path, "r") as f:
                panel_payload = json.load(f)
        union = _union_candidates(_reconstruct_candidates(per_cand_path),
                                  completion_metrics.candidates or [])
        metrics = rollup_metrics(
            union,
            n_submitted=int(panel_payload.get("n_submitted", 0)) + completion_metrics.n_submitted,
            n_completed=int(panel_payload.get("n_completed", 0)) + completion_metrics.n_completed,
            n_train_samples=completion_metrics.n_train_samples,
            prior_best_revenue=state.best_real_revenue_so_far(),
            prior_best_emv=state.best_emv_so_far_value(),
            prior_per_candidate_emv_by_iter=prior_per_candidate_emv_by_iter or None,
            expected_k=expected_k,
            n_failed_per_status=completion_metrics.n_failed_per_status,
            n_succeeded_but_missing_h5=completion_metrics.n_succeeded_but_missing_h5,
            n_missing_silently=completion_metrics.n_missing_silently,
        )
    else:
        # full (legacy ungated) or panel (gated phase 1): standard single-batch ingest.
        prior_iter = state.iteration - 1
        prior_compiled_h5 = (
            paths.iter_compiled_h5(prior_iter) if prior_iter >= 0 else bootstrap_h5
        )
        if not prior_compiled_h5.exists() and state.iteration == 0:
            prior_compiled_h5 = bootstrap_h5
        next_compiled_h5 = paths.iter_compiled_h5(state.iteration)
        # `args.ix_output_dir` is the shared h5s_dir_out from the Julia config; scope to
        # just this iter's outputs so preprocess_h5 doesn't sweep up unrelated runs.
        ix_output_dir = _stage_iteration_outputs(
            ix_output_root=args.ix_output_dir,
            array_tasks_json=args.array_tasks_json,
            iter_scratch_dir=paths.iter_ix_output_dir(state.iteration),
        )
        metrics = ingest_iteration(
            iteration=state.iteration,
            surrogate_repo=surrogate_repo,
            norm_config_path=norm_config_path,
            economics_config=economics_config,
            array_tasks_json=args.array_tasks_json,
            ix_output_dir=ix_output_dir,
            raw_ix_archive=paths.raw_ix_archive,
            delta_h5=paths.iter_preprocessed_h5(state.iteration),
            prior_compiled_h5=prior_compiled_h5,
            next_compiled_h5=next_compiled_h5,
            log_path=paths.logs_dir / f"ingest_iter_{state.iteration:04d}.preprocess.log",
            prior_best_revenue=state.best_real_revenue_so_far(),
            prior_best_emv=state.best_emv_so_far_value(),
            prior_per_candidate_emv_by_iter=prior_per_candidate_emv_by_iter or None,
            workers=4,
            provenance=("panel" if gate_phase == "panel" else "full"),
            expected_k=expected_k,
        )

    ingest_elapsed_min = (time.time() - ingest_started) / 60.0

    print(
        f"[ingest] phase={gate_phase} iter={state.iteration} submitted={metrics.n_submitted} "
        f"completed={metrics.n_completed} mape={metrics.batch_mape} "
        f"signed_bias={metrics.batch_signed_pct_bias} "
        f"best_real={metrics.best_real_revenue_so_far} "
        f"n_train_samples={metrics.n_train_samples}"
    )

    rec = state.get_iter(state.iteration) or IterationRecord(iteration=state.iteration)
    _record_metrics(rec, metrics, ingest_elapsed_min)
    state.upsert_iter(rec)
    _write_per_candidate(per_cand_path, state.iteration, metrics)
    state.current_compiled_h5 = str(next_compiled_h5)
    state.save(paths.state_file)

    # Phase-1 (panel) stops here: it must NOT run the stopping check or chain the next
    # iteration. The local driver computes EMV_hat, picks the top-M, and submits the
    # completion phase (which re-enters this script with --gate-phase completion).
    if gate_phase == "panel":
        print(f"[ingest] panel phase complete for iter={state.iteration}; awaiting completion.")
        return 0

    wandb_handle = init_run(
        run_id=state.wandb_run_id,
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        config={"al_run_id": state.run_id},
        tags=config["wandb"].get("tags"),
        name=state.run_id,
        resume="allow",
    )
    if wandb_handle.is_active:
        wandb_handle.log(
            {
                "iteration": state.iteration,
                # NOTE: in npv mode the OPTIMIZED objective is NPV; these revenue series are the
                # surrogate's ground-truth revenue (still useful), NOT the objective. The NPV
                # best-so-far is logged by the local driver (run_al_local.py) which has the cube.
                "objective": objective,
                "best_real_revenue_so_far": metrics.best_real_revenue_so_far,
                "best_real_revenue_in_batch": metrics.best_real_revenue_in_batch,
                "batch_mape_real_vs_pred": metrics.batch_mape,
                "batch_signed_pct_bias": metrics.batch_signed_pct_bias,
                "frontier_mape": metrics.frontier_mape,
                "adversarial_mape": metrics.adversarial_mape,
                "exploit_mape": metrics.exploit_mape,
                "cma_mape": metrics.cma_mape,
                "batch_mae": metrics.batch_mae,
                "frontier_mae": metrics.frontier_mae,
                "adversarial_mae": metrics.adversarial_mae,
                "exploit_mae": metrics.exploit_mae,
                "cma_mae": metrics.cma_mae,
                "n_completed": metrics.n_completed,
                "n_submitted": metrics.n_submitted,
                "completion_rate": metrics.completion_rate,
                "n_train_samples": metrics.n_train_samples,
                "wallclock_ingest_min": ingest_elapsed_min,
            },
            step=state.iteration,
        )

    # ----- Stopping check -----
    stop_cfg = StoppingConfig(
        max_iterations=int(config["stopping"]["max_iterations"]),
        plateau_window=int(config["stopping"].get("plateau_window", 5)),
        plateau_threshold_relative=float(config["stopping"].get("plateau_threshold_relative", 0.005)),
        target_mape=config["stopping"].get("target_mape"),
        target_mape_window=int(config["stopping"].get("target_mape_window", 3)),
        consecutive_zero_completion_limit=int(
            config["stopping"].get("consecutive_zero_completion_limit", 2)
        ),
    )
    decision = evaluate_stopping(state, stop_cfg, objective=objective)
    if decision.should_stop:
        marker = paths.done_marker
        marker.write_text(json.dumps({
            "stopped_at_iteration": state.iteration,
            "reason": decision.reason,
            # best_real_revenue is REVENUE even on npv runs (the ingest is revenue-based); the
            # objective tag disambiguates so post-processors don't read it as the optimized value.
            "objective": objective,
            "history": [
                {"iteration": r.iteration, "best_real_revenue": r.best_real_revenue, "batch_mape": r.batch_mape}
                for r in state.history
            ],
        }, indent=2))
        print(f"[ingest] Stopping: {decision.reason}")
        if wandb_handle.is_active:
            wandb_handle.log({"stopped": True, "stop_reason": decision.reason}, step=state.iteration)
            wandb_handle.finish()
        return 0

    # ----- Chain next iteration -----
    state.iteration += 1
    state.save(paths.state_file)

    # In hybrid mode the local driver on bend takes over here — it polls this
    # ingest job's terminal state and runs the next train_acquire on its own
    # GPU. Skip submitting on Sherlock so we don't burn a GPU node we won't use.
    compute_loc = (config.get("compute") or {}).get("train_acquire_location", "sherlock")
    if compute_loc == "local":
        print(
            f"[ingest] hybrid mode (compute.train_acquire_location=local): "
            f"not submitting next train_acquire on Sherlock. Local driver "
            f"will pick up iter={state.iteration}."
        )
        if wandb_handle.is_active:
            wandb_handle.finish()
        return 0

    next_template = REPO_ROOT / "sbatch" / "train_acquire.sbatch.template"
    next_sbatch = paths.sbatch_dir / f"train_acquire_iter_{state.iteration:04d}.sbatch"
    write_rendered(
        next_template,
        next_sbatch,
        {
            "RUN_ROOT": str(args.run_root),
            "REPO_ROOT": str(REPO_ROOT),
            "ITER": str(state.iteration),
            "LOG_OUT": str(paths.logs_dir / f"train_acquire_iter_{state.iteration:04d}.out"),
            "LOG_ERR": str(paths.logs_dir / f"train_acquire_iter_{state.iteration:04d}.err"),
            "JOB_NAME": f"AL_TRAINACQ_{state.run_id}_{state.iteration:04d}",
        },
    )
    next_job = submit_sbatch(next_sbatch)
    rec = state.get_iter(state.iteration) or IterationRecord(iteration=state.iteration)
    rec.train_acquire_job_id = next_job
    state.upsert_iter(rec)
    state.save(paths.state_file)
    print(f"[ingest] Submitted next iter (iter={state.iteration}) as job {next_job}")

    if wandb_handle.is_active:
        wandb_handle.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
