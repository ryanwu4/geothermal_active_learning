#!/usr/bin/env python3
"""Re-submit the next pending phase for an AL run that died mid-chain.

The state file records the last successfully completed phase's outputs. We
infer where to pick up:

* If ``state.iteration`` has no completed history record (no
  ``best_real_revenue``), re-submit ``train_acquire`` for that iteration.
* If a completed record exists for ``state.iteration``, the next iteration
  is already populated — re-submit its ``train_acquire``.

Intentionally conservative: never tries to recover a partial Phase A; you
re-run the whole phase. Phase B (IX array) handles its own internal failures
via ``afterany`` already.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.paths import resolve_run_paths
from orchestrator.slurm import submit_sbatch, write_rendered
from orchestrator.state import IterationRecord, RunState


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path, help="Path to the AL run root (parent of state.json)")
    parser.add_argument(
        "--force-iteration",
        type=int,
        default=None,
        help="Override the iteration to resume at (rare; for surgical recovery).",
    )
    args = parser.parse_args()

    state = RunState.load(args.run_root / "state.json")
    paths = resolve_run_paths(scratch_root=args.run_root.parent, run_id=args.run_root.name)
    paths.ensure_dirs()

    if (paths.done_marker).exists():
        print(f"Run already completed (done.json present). Reason: ", end="")
        print((paths.done_marker).read_text())
        return 0

    target_iter = args.force_iteration if args.force_iteration is not None else state.iteration
    rec = state.get_iter(target_iter)
    if rec is not None and rec.best_real_revenue is not None:
        # That iteration finished ingest already — bump forward.
        target_iter = target_iter + 1
        state.iteration = target_iter
        state.upsert_iter(IterationRecord(iteration=target_iter))
        state.save(paths.state_file)

    paths.ensure_iter_dirs(target_iter)
    template = REPO_ROOT / "sbatch" / "train_acquire.sbatch.template"
    sbatch_path = paths.sbatch_dir / f"train_acquire_iter_{target_iter:04d}.sbatch"
    write_rendered(
        template,
        sbatch_path,
        {
            "RUN_ROOT": str(paths.run_root),
            "REPO_ROOT": str(REPO_ROOT),
            "ITER": str(target_iter),
            "LOG_OUT": str(paths.logs_dir / f"train_acquire_iter_{target_iter:04d}.out"),
            "LOG_ERR": str(paths.logs_dir / f"train_acquire_iter_{target_iter:04d}.err"),
            "JOB_NAME": f"AL_TRAINACQ_{state.run_id}_{target_iter:04d}",
        },
    )
    job_id = submit_sbatch(sbatch_path)
    rec = state.get_iter(target_iter) or IterationRecord(iteration=target_iter)
    rec.train_acquire_job_id = job_id
    state.upsert_iter(rec)
    state.save(paths.state_file)
    print(f"Resumed run {state.run_id} at iteration {target_iter} as job {job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
