#!/usr/bin/env python3
"""Bootstrap a new AL run: build initial state, render the first sbatch, submit it.

After this returns, a chain of dependent SLURM jobs runs the loop autonomously
until either ``stopping.max_iterations`` is reached or another stopping
criterion fires. Inspect ``state.json`` under the run dir to monitor progress.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.paths import resolve_run_paths
from orchestrator.slurm import submit_sbatch, write_rendered
from orchestrator.state import IterationRecord, RunState, new_run_id, new_wandb_run_id


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="AL pipeline JSON config")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run id (default: auto-generated like al_<ts>_<hex>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the sbatch and write state but do not submit.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    scratch_root = Path(config["paths"]["scratch_root"])
    run_id = args.run_id or new_run_id(prefix=config.get("run_id_prefix", "al"))
    paths = resolve_run_paths(scratch_root, run_id)
    paths.ensure_dirs()
    paths.ensure_iter_dirs(0)

    # Resolve and persist the norm config once at bootstrap. Future iterations
    # reuse the same file so normalization stays consistent across the run.
    #
    # Resolution priority:
    #   1. Explicit `paths.norm_config_path` in the AL config (preferred — set
    #      this when the bootstrap was produced by `preprocess_h5.py
    #      --compute-only` over a non-default archive).
    #   2. <surrogate_repo>/norm_config.json
    #   3. <surrogate_repo>/configs/norm_config.json
    #   4. <surrogate_repo>/trained/norm_config.json (legacy training layout)
    surrogate_repo = Path(config["paths"]["surrogate_repo"]).resolve()
    explicit = config["paths"].get("norm_config_path")
    if explicit:
        norm_config_path = Path(explicit).expanduser().resolve()
        if not norm_config_path.exists():
            print(
                f"ERROR: paths.norm_config_path does not exist: {norm_config_path}",
                file=sys.stderr,
            )
            return 1
    else:
        candidates = [
            surrogate_repo / "norm_config.json",
            surrogate_repo / "configs" / "norm_config.json",
            surrogate_repo / "trained" / "norm_config.json",
        ]
        norm_config_path = next((c for c in candidates if c.exists()), candidates[0])
        if not norm_config_path.exists():
            print(
                "ERROR: could not auto-locate norm_config.json under "
                f"{surrogate_repo}. Set paths.norm_config_path in the AL config.",
                file=sys.stderr,
            )
            return 1
    print(f"Using norm_config: {norm_config_path}")

    state = RunState(
        run_id=run_id,
        config_path=str(args.config.resolve()),
        wandb_run_id=new_wandb_run_id(),
        iteration=0,
        target=config.get("target", "graph_discounted_net_revenue"),
        current_compiled_h5=str(Path(config["paths"]["bootstrap_compiled_h5"]).resolve()),
        norm_config_path=str(norm_config_path),
        history=[IterationRecord(iteration=0)],
    )
    state.save(paths.state_file)
    print(f"Initialized run state at {paths.state_file}")

    template = REPO_ROOT / "sbatch" / "train_acquire.sbatch.template"
    sbatch_path = paths.sbatch_dir / "train_acquire_iter_0000.sbatch"
    write_rendered(
        template,
        sbatch_path,
        {
            "RUN_ROOT": str(paths.run_root),
            "REPO_ROOT": str(REPO_ROOT),
            "ITER": "0",
            "LOG_OUT": str(paths.logs_dir / "train_acquire_iter_0000.out"),
            "LOG_ERR": str(paths.logs_dir / "train_acquire_iter_0000.err"),
            "JOB_NAME": f"AL_TRAINACQ_{run_id}_0000",
        },
    )
    print(f"Rendered initial sbatch at {sbatch_path}")

    if args.dry_run:
        print("--dry-run: skipping submission.")
        return 0

    job_id = submit_sbatch(sbatch_path)
    rec = state.get_iter(0) or IterationRecord(iteration=0)
    rec.train_acquire_job_id = job_id
    state.upsert_iter(rec)
    state.save(paths.state_file)

    print(f"Submitted iter 0 train_acquire as job {job_id}")
    print(f"Run root: {paths.run_root}")
    print(f"Tail logs: tail -F {paths.logs_dir}/train_acquire_iter_0000.out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
