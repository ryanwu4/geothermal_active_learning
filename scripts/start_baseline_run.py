#!/usr/bin/env python3
"""Bootstrap a fresh baseline (no-surrogate) global-optimizer run.

Writes a fresh ``state.json`` under ``<scratch_root>/<run_id>/`` so the local
driver (``scripts/run_baseline_global.py``) has somewhere to land. Unlike
``start_al_run.py``, no SLURM job is submitted here — the local driver owns
the loop body and sbatches jobs per-iteration as it goes.

Typically invoked over SSH from the local driver as a one-shot. Can also be
run directly on Sherlock to pre-stage a run id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.paths import resolve_run_paths  # noqa: E402
from orchestrator.state import IterationRecord, RunState, new_run_id, new_wandb_run_id  # noqa: E402


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Baseline pipeline JSON config (e.g. configs/baseline_global.json)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional explicit run id (default: auto-generated like baseline_<ts>_<hex>)")
    args = parser.parse_args()

    config = _load_config(args.config)
    scratch_root = Path(config["paths"]["scratch_root"])
    run_id = args.run_id or new_run_id(prefix=config.get("run_id_prefix", "baseline"))
    paths = resolve_run_paths(scratch_root, run_id)
    paths.ensure_dirs()
    paths.ensure_iter_dirs(0)

    norm_path = config["paths"].get("norm_config_path")
    if norm_path:
        norm_p = Path(norm_path).expanduser().resolve()
        if not norm_p.exists():
            print(f"ERROR: paths.norm_config_path does not exist: {norm_p}", file=sys.stderr)
            return 1
    else:
        # Without a norm_config we can't run preprocess_h5 in the ingest stage.
        print("ERROR: baseline config must set paths.norm_config_path "
              "(preprocess_h5 needs it).", file=sys.stderr)
        return 1

    state = RunState(
        run_id=run_id,
        config_path=str(args.config.resolve()),
        wandb_run_id=new_wandb_run_id(),
        iteration=0,
        target=config.get("target", "baseline_ensemble_mean_discounted_revenue"),
        current_compiled_h5=None,
        current_checkpoint=None,
        current_scaler=None,
        norm_config_path=str(norm_p),
        history=[IterationRecord(iteration=0)],
    )
    state.save(paths.state_file)
    print(f"Initialized baseline run state at {paths.state_file}")
    print(f"Run id: {run_id}")
    print(f"Run root: {paths.run_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
