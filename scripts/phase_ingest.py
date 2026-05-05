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
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.ingest import ingest_iteration
from orchestrator.log import init_run
from orchestrator.paths import resolve_run_paths
from orchestrator.slurm import submit_sbatch, write_rendered
from orchestrator.state import IterationRecord, RunState
from orchestrator.stop import StoppingConfig, evaluate_stopping


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
    args = parser.parse_args()

    state = RunState.load(args.run_root / "state.json")
    config = _load_config(Path(state.config_path))

    paths = resolve_run_paths(scratch_root=args.run_root.parent, run_id=args.run_root.name)
    paths.ensure_dirs()

    surrogate_repo = Path(config["paths"]["surrogate_repo"]).resolve()
    norm_config_path = Path(state.norm_config_path) if state.norm_config_path else (
        surrogate_repo / "norm_config.json"
    )
    economics_config = surrogate_repo / "configs" / "economics.json"
    bootstrap_h5 = Path(config["paths"]["bootstrap_compiled_h5"]).resolve()

    prior_iter = state.iteration - 1
    prior_compiled_h5 = (
        paths.iter_compiled_h5(prior_iter) if prior_iter >= 0 else bootstrap_h5
    )
    if not prior_compiled_h5.exists() and state.iteration == 0:
        prior_compiled_h5 = bootstrap_h5
    next_compiled_h5 = paths.iter_compiled_h5(state.iteration)

    # `args.ix_output_dir` is the shared h5s_dir_out from the Julia config.
    # Scope to just this iter's outputs by symlinking the ones the array tasks
    # promised, so preprocess_h5 doesn't sweep up unrelated runs.
    ix_output_dir = _stage_iteration_outputs(
        ix_output_root=args.ix_output_dir,
        array_tasks_json=args.array_tasks_json,
        iter_scratch_dir=paths.iter_ix_output_dir(state.iteration),
    )
    ingest_started = time.time()
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
        workers=4,
    )
    ingest_elapsed_min = (time.time() - ingest_started) / 60.0

    print(
        f"[ingest] iter={state.iteration} submitted={metrics.n_submitted} "
        f"completed={metrics.n_completed} mape={metrics.batch_mape} "
        f"signed_bias={metrics.batch_signed_pct_bias} "
        f"best_real={metrics.best_real_revenue_so_far} "
        f"n_train_samples={metrics.n_train_samples}"
    )

    rec = state.get_iter(state.iteration) or IterationRecord(iteration=state.iteration)
    rec.completed = metrics.n_completed
    rec.batch_mape = metrics.batch_mape
    rec.batch_signed_pct_bias = metrics.batch_signed_pct_bias
    rec.frontier_mape = metrics.frontier_mape
    rec.adversarial_mape = metrics.adversarial_mape
    rec.best_real_revenue = metrics.best_real_revenue_so_far
    rec.n_train_samples = metrics.n_train_samples
    rec.wallclock_ingest_min = ingest_elapsed_min
    state.upsert_iter(rec)
    # Make the new compiled H5 the active training set for the next iteration.
    state.current_compiled_h5 = str(next_compiled_h5)
    state.save(paths.state_file)

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
                "best_real_revenue_so_far": metrics.best_real_revenue_so_far,
                "best_real_revenue_in_batch": metrics.best_real_revenue_in_batch,
                "batch_mape_real_vs_pred": metrics.batch_mape,
                "batch_signed_pct_bias": metrics.batch_signed_pct_bias,
                "frontier_mape": metrics.frontier_mape,
                "adversarial_mape": metrics.adversarial_mape,
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
        consecutive_zero_completion_limit=int(
            config["stopping"].get("consecutive_zero_completion_limit", 2)
        ),
    )
    decision = evaluate_stopping(state, stop_cfg)
    if decision.should_stop:
        marker = paths.done_marker
        marker.write_text(json.dumps({
            "stopped_at_iteration": state.iteration,
            "reason": decision.reason,
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
