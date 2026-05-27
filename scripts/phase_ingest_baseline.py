#!/usr/bin/env python3
"""Surrogate-free ingest: read IX outputs, aggregate per-candidate ensemble fitness.

Counterpart to ``phase_ingest.py`` for the global-optimizer ablation. Skips all
MAPE/calibration bookkeeping (there is no surrogate prediction to compare
against) and only computes ground-truth ensemble-mean discounted revenue per
candidate. Output is a single ``fitness.json`` file that the local driver pulls
and feeds back to the optimizer's ``tell()``.

Pipeline:
1. Symlink this iteration's IX outputs into a per-iter scratch dir
   (reuses ``_stage_iteration_outputs`` from ``phase_ingest.py``).
2. Run ``preprocess_h5.py`` over those outputs → ``delta.h5``. The delta H5
   carries the ``field_discounted_net_revenue`` dataset per case.
3. For each manifest snapshot, look up its K IX cases (one per geology),
   read each case's real revenue, take the mean across geologies.
4. Write ``fitness.json`` with per-candidate per-geology revenues and
   ensemble-mean fitness; update ``state.json``.

This script does NOT merge the delta into a running compiled archive, because
the baseline never trains a surrogate. (The local driver could opt to merge
later for downstream re-analysis; out of scope here.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from orchestrator.ingest import _run_preprocess_h5, _read_real_revenue  # type: ignore  # noqa: E402
from orchestrator.paths import resolve_run_paths  # noqa: E402
from orchestrator.state import IterationRecord, RunState  # noqa: E402

# Reuse the symlink helper from phase_ingest. Importing it directly so we don't
# duplicate the IX-output scoping logic.
from phase_ingest import _stage_iteration_outputs  # type: ignore  # noqa: E402


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _read_manifest_snapshots(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r") as f:
        return json.load(f).get("snapshots", [])


def _read_array_tasks(array_tasks_json: Path) -> list[dict]:
    with open(array_tasks_json, "r") as f:
        return json.load(f).get("tasks", [])


def _group_tasks_by_snapshot(tasks: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for t in tasks:
        sid = str(t.get("snapshot_id", ""))
        if not sid:
            continue
        out.setdefault(sid, []).append(t)
    return out


def _read_snapshot_wells(snapshot_json_path: Path) -> tuple[list[list[float]], list[bool]]:
    with open(snapshot_json_path, "r") as f:
        snap = json.load(f)
    wells = snap.get("wells", [])
    coords_xyz = [[float(w["x"]), float(w["y"]), float(w["z"])] for w in wells]
    is_injector = [str(w.get("type", "")).lower() == "injector" for w in wells]
    return coords_xyz, is_injector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--array-tasks-json", type=Path, required=True)
    parser.add_argument("--ix-output-dir", type=Path, required=True,
                        help="Shared IX output dir (FILEPATHS.h5s_dir_out).")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fitness-out", type=Path, required=True,
                        help="Output JSON path for fitness aggregation.")
    args = parser.parse_args()

    started = time.time()
    state = RunState.load(args.run_root / "state.json")
    config = _load_config(Path(state.config_path))

    surrogate_repo = Path(config["paths"]["surrogate_repo"]).resolve()
    economics_config = surrogate_repo / "configs" / "economics.json"
    norm_config_path = (
        Path(state.norm_config_path).resolve() if state.norm_config_path
        else (surrogate_repo / "norm_config.json")
    )
    if not norm_config_path.exists():
        print(f"ERROR: norm_config not found at {norm_config_path}", file=sys.stderr)
        return 1

    paths = resolve_run_paths(scratch_root=args.run_root.parent, run_id=args.run_root.name)
    paths.ensure_dirs()
    paths.ensure_iter_dirs(args.iteration)

    # Scope outputs.
    iter_ix_dir = _stage_iteration_outputs(
        ix_output_root=args.ix_output_dir,
        array_tasks_json=args.array_tasks_json,
        iter_scratch_dir=paths.iter_ix_output_dir(args.iteration),
    )

    # Preprocess only this iteration's outputs.
    delta_h5 = paths.iter_preprocessed_h5(args.iteration)
    print(f"[ingest-baseline] preprocess_h5 → {delta_h5}")
    _run_preprocess_h5(
        surrogate_repo=surrogate_repo,
        input_dir=iter_ix_dir,
        output_h5=delta_h5,
        norm_config=norm_config_path,
        economics_config=economics_config,
        workers=4,
        log_path=paths.logs_dir / f"ingest_baseline_iter_{args.iteration:04d}.preprocess.log",
    )

    # Aggregate fitness per snapshot.
    tasks = _read_array_tasks(args.array_tasks_json)
    snapshots = _read_manifest_snapshots(args.manifest)
    tasks_by_snap = _group_tasks_by_snapshot(tasks)

    n_submitted = len(tasks)
    n_completed = 0
    candidates_payload: list[dict] = []
    best_in_batch: float | None = None

    for snap in snapshots:
        sid = str(snap.get("snapshot_id", ""))
        snap_tasks = tasks_by_snap.get(sid, [])
        per_geo: dict[str, float] = {}
        n_failed_geos = 0
        for t in snap_tasks:
            geo_idx = str(int(t.get("geology_index", -1)))
            case_id = Path(str(t.get("output_file_name", ""))).stem
            produced = iter_ix_dir / t.get("output_file_name", "")
            if not produced.exists() or not case_id:
                n_failed_geos += 1
                continue
            n_completed += 1
            real = _read_real_revenue(delta_h5, case_id)
            if real is None or not np.isfinite(real):
                n_failed_geos += 1
                continue
            per_geo[geo_idx] = float(real)

        if per_geo:
            ensemble_mean = float(np.mean(list(per_geo.values())))
        else:
            ensemble_mean = float("nan")

        # Pull coords from snapshot JSON if available so the local driver can
        # confirm what the optimizer actually proposed (and recover state from
        # fitness.json alone if needed).
        coords_xyz: list[list[float]] = []
        is_injector: list[bool] = []
        snap_json_path = snap.get("json_path") or ""
        if snap_json_path:
            sp = Path(snap_json_path)
            if sp.exists():
                try:
                    coords_xyz, is_injector = _read_snapshot_wells(sp)
                except Exception as e:
                    print(f"[ingest-baseline] WARN: failed to read {sp}: {e}")

        candidates_payload.append({
            "snapshot_id": sid,
            "kind": snap.get("kind", ""),
            "coords_xyz": coords_xyz,
            "is_injector": is_injector,
            "per_geology_revenue": per_geo,
            "ensemble_mean_revenue": ensemble_mean,
            "n_completed_geos": len(per_geo),
            "n_failed_geos": n_failed_geos,
        })
        if np.isfinite(ensemble_mean):
            if (best_in_batch is None) or (ensemble_mean > best_in_batch):
                best_in_batch = ensemble_mean

    prior_best = state.best_real_revenue_so_far()
    best_so_far = best_in_batch if prior_best is None else (
        max(prior_best, best_in_batch) if best_in_batch is not None else prior_best
    )

    fitness_payload = {
        "iteration": args.iteration,
        "generation": args.iteration,  # for baseline they are identical
        "n_submitted_tasks": n_submitted,
        "n_completed_tasks": n_completed,
        "n_candidates": len(candidates_payload),
        "best_ensemble_revenue_in_batch": best_in_batch,
        "best_ensemble_revenue_so_far": best_so_far,
        "candidates": candidates_payload,
    }
    args.fitness_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.fitness_out, "w") as f:
        json.dump(fitness_payload, f, indent=2)
    print(f"[ingest-baseline] wrote fitness JSON → {args.fitness_out} "
          f"({len(candidates_payload)} candidates, best_in_batch={best_in_batch})")

    # Update state. We advance ``state.iteration`` here (mirroring
    # ``phase_ingest.py:266-267``) so the local driver doesn't need a separate
    # increment step — that previously left a resume race window where the
    # driver crashed between local tell() and remote state push, leaving
    # optimizer.generation ahead of state.iteration. The driver now just
    # pulls a fresh, advanced state.
    elapsed_min = (time.time() - started) / 60.0
    rec = state.get_iter(args.iteration) or IterationRecord(iteration=args.iteration)
    rec.submitted = n_submitted
    rec.completed = n_completed
    rec.best_real_revenue = best_so_far
    rec.best_emv_in_batch = best_in_batch
    rec.best_emv_so_far = best_so_far
    rec.wallclock_ingest_min = elapsed_min
    state.upsert_iter(rec)
    state.iteration = args.iteration + 1
    state.save(paths.state_file)
    print(f"[ingest-baseline] iter={args.iteration} completed={n_completed}/{n_submitted} "
          f"best_in_batch={best_in_batch} best_so_far={best_so_far} ({elapsed_min:.2f} min); "
          f"advanced state.iteration → {state.iteration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
