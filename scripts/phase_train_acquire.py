#!/usr/bin/env python3
"""Phase A: train surrogate (warm or fresh), acquire candidates, stage IX submission.

Invoked from ``train_acquire.sbatch``. Reads the run's state, decides
warm-start vs. from-scratch, runs gradient acquisition over all configured
geologies, picks a frontier+adversarial batch, runs the Julia staging tool to
generate the IX array sbatch, submits it, then chains the ingest job with an
``afterany`` dependency.
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

from orchestrator.acquire import (
    AcquisitionConfig,
    GeologySpec,
    WellSpec,
    run_acquisition,
    write_selected_manifest,
)
from orchestrator.log import init_run
from orchestrator.paths import resolve_run_paths
from orchestrator.retrain import _find_best_checkpoint, run_train, should_train_from_scratch
from orchestrator.select import select_batch
from orchestrator.slurm import submit_sbatch, write_rendered
from orchestrator.stage import stage_iteration
from orchestrator.state import IterationRecord, RunState


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_geology_list(path: Path) -> list[dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload.get("geologies", [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    state = RunState.load(args.run_root / "state.json")
    config = _load_config(Path(state.config_path))

    paths = resolve_run_paths(scratch_root=args.run_root.parent, run_id=args.run_root.name)
    paths.ensure_dirs()
    paths.ensure_iter_dirs(state.iteration)

    surrogate_repo = Path(config["paths"]["surrogate_repo"]).resolve()
    julia_repo = Path(config["paths"]["julia_repo"]).resolve()
    bootstrap_h5 = Path(config["paths"]["bootstrap_compiled_h5"]).resolve()

    wandb_handle = init_run(
        run_id=state.wandb_run_id,
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        config={"al_run_id": state.run_id, "iteration": state.iteration},
        tags=config["wandb"].get("tags"),
        name=state.run_id,
        resume="allow",
    )

    train_cfg = config["training"]
    acq_cfg = config["acquisition"]
    sel_cfg = config["selection"]
    ix_cfg = config["intersect"]

    # ----- Step 1: train (warm-start or from-scratch) -----
    iter_dir = paths.iter_dir(state.iteration)
    train_log = paths.logs_dir / f"train_iter_{state.iteration:04d}.log"

    if state.iteration == 0:
        # Bootstrap: use existing checkpoint + scaler from config; skip retraining
        # at iter 0 since that data is already represented.
        ckpt_dir = Path(config["paths"]["bootstrap_checkpoint_dir"]).resolve()
        try:
            new_ckpt = _find_best_checkpoint(ckpt_dir)
        except Exception as e:
            print(f"ERROR: no bootstrap checkpoint found in {ckpt_dir}: {e}", file=sys.stderr)
            return 1
        new_scaler = new_ckpt.parent / "scaler.pkl"
        if not new_scaler.exists():
            print(f"ERROR: bootstrap scaler missing: {new_scaler}", file=sys.stderr)
            return 1
        if not bootstrap_h5.exists():
            print(f"ERROR: bootstrap_compiled_h5 not found: {bootstrap_h5}", file=sys.stderr)
            return 1
        compiled_h5_for_acq = bootstrap_h5
        was_from_scratch = False
        print(f"Iteration 0 bootstrap: ckpt={new_ckpt}, scaler={new_scaler}")
    else:
        # Pick the compiled H5 produced by the previous ingest.
        prior_iter = state.iteration - 1
        compiled_h5_for_acq = paths.iter_compiled_h5(prior_iter)
        if not compiled_h5_for_acq.exists():
            # Fall back to whatever state has tracked.
            if state.current_compiled_h5 and Path(state.current_compiled_h5).exists():
                compiled_h5_for_acq = Path(state.current_compiled_h5)
            else:
                print(f"ERROR: no compiled H5 found for iter {prior_iter}", file=sys.stderr)
                return 1

        from_scratch = should_train_from_scratch(
            state.iteration, int(train_cfg.get("from_scratch_every_k", 5))
        )
        was_from_scratch = from_scratch
        warm_ckpt = None if from_scratch else (
            Path(state.current_checkpoint) if state.current_checkpoint else None
        )
        warm_scaler = None if from_scratch else (
            Path(state.current_scaler) if state.current_scaler else None
        )

        train_started = time.time()
        result = run_train(
            surrogate_repo=surrogate_repo,
            h5_path=compiled_h5_for_acq,
            output_root=paths.iter_model_dir(state.iteration),
            target=state.target,
            seed=int(train_cfg.get("seed", 42)),
            run_id=0,
            cache_to_gpu=True,
            gpu="0",
            edge_encoder=train_cfg.get("edge_encoder", "cnn"),
            batch_size=int(train_cfg.get("batch_size", 16)),
            learning_rate=float(train_cfg.get("lr", 3e-4)),
            max_epochs=int(train_cfg.get("max_epochs_full", 180)),
            early_stop_patience=int(train_cfg.get("early_stop_patience", 30)),
            warm_start_checkpoint=warm_ckpt,
            warm_start_scaler=warm_scaler,
            max_epochs_finetune=int(train_cfg["max_epochs_finetune"]) if not from_scratch else None,
            log_path=train_log,
        )
        new_ckpt = result.checkpoint_path
        new_scaler = result.scaler_path
        elapsed_train_min = (time.time() - train_started) / 60.0
        rec = state.get_iter(state.iteration) or IterationRecord(iteration=state.iteration)
        rec.wallclock_train_min = elapsed_train_min
        rec.train_val_loss = result.final_val_loss
        state.upsert_iter(rec)

    state.current_checkpoint = str(new_ckpt)
    state.current_scaler = str(new_scaler)
    state.current_compiled_h5 = str(compiled_h5_for_acq)
    state.save(paths.state_file)

    # ----- Step 2: acquire candidates over all geologies -----
    # Prefer state (set during start_al_run via paths.norm_config_path), then
    # config override, then legacy fallbacks. Hard error if none exists — this
    # has to be the same file the surrogate was trained on or normalization
    # diverges silently.
    if state.norm_config_path and Path(state.norm_config_path).exists():
        norm_config_path = Path(state.norm_config_path)
    elif config["paths"].get("norm_config_path"):
        norm_config_path = Path(config["paths"]["norm_config_path"]).expanduser().resolve()
    else:
        for cand in (
            surrogate_repo / "norm_config.json",
            surrogate_repo / "configs" / "norm_config.json",
            surrogate_repo / "trained" / "norm_config.json",
        ):
            if cand.exists():
                norm_config_path = cand
                break
        else:
            print(
                f"ERROR: norm_config.json not found. Set paths.norm_config_path "
                f"in {state.config_path} or place the file under {surrogate_repo}.",
                file=sys.stderr,
            )
            return 1
    if not norm_config_path.exists():
        print(f"ERROR: resolved norm_config_path does not exist: {norm_config_path}",
              file=sys.stderr)
        return 1
    geology_entries = _load_geology_list(
        Path(config["paths"].get("geologies_config")) if config["paths"].get("geologies_config")
        else REPO_ROOT / "configs" / "geologies_default.json"
    )
    geologies = [
        GeologySpec(
            geology_index=int(e["geology_index"]),
            geology_h5_file=str(Path(e["geology_h5_file"]).resolve()),
            geology_name=e.get("geology_name"),
        )
        for e in geology_entries
    ]
    wells = [WellSpec(type=w["type"], depth=int(w["depth"])) for w in config["wells"]]

    # Collect prior-iter per_candidate_metrics.json paths for the elite path to
    # rank by real INTERSECT revenue. Cold start (iter 0): list is empty,
    # acquire falls back to LHS for the elite kind.
    prior_metrics: list[Path] = []
    for it_prior in range(state.iteration):
        p = paths.iter_dir(it_prior) / "per_candidate_metrics.json"
        if p.exists():
            prior_metrics.append(p)

    acq = AcquisitionConfig(
        surrogate_repo=surrogate_repo,
        checkpoint_path=Path(state.current_checkpoint),
        scaler_path=Path(state.current_scaler),
        norm_config_path=norm_config_path,
        geologies=geologies,
        wells=wells,
        n_starts_per_geology=int(acq_cfg["n_starts_per_geology"]),
        k_safe=int(acq_cfg["k_safe"]),
        k_adv=int(acq_cfg["k_adv"]),
        adv_fraction=float(acq_cfg["adv_fraction"]),
        edge_buffer=int(acq_cfg.get("edge_buffer", 10)),
        learning_rate=float(acq_cfg.get("lr", 0.5)),
        log_every_n_steps=int(acq_cfg.get("log_every_n_steps", 25)),
        revenue_target=state.target,
        seed=int(acq_cfg.get("seed", 42)),
        device=str(acq_cfg.get("devices", ["cuda:0"])[0]),
        devices=[str(d) for d in acq_cfg.get("devices", ["cuda:0"])],
        n_elite_per_geology=int(acq_cfg.get("n_elite_per_geology", 0)),
        n_cma_per_geology=int(acq_cfg.get("n_cma_per_geology", 0)),
        elite_top_k=int(acq_cfg.get("elite_top_k", 10)),
        elite_seed_noise=float(acq_cfg.get("elite_seed_noise", 2.0)),
        cma_popsize=int(acq_cfg.get("cma_popsize", 16)),
        cma_generations=int(acq_cfg.get("cma_generations", 10)),
        cma_sigma_init=float(acq_cfg.get("cma_sigma_init", 5.0)),
        prior_metrics=prior_metrics,
    )
    acq_started = time.time()
    acq_result = run_acquisition(
        acq, out_dir=paths.iter_acquire_dir(state.iteration), iteration=state.iteration
    )
    acq_elapsed_min = (time.time() - acq_started) / 60.0
    print(f"Acquired {len(acq_result['candidates'])} raw candidates in {acq_elapsed_min:.2f} min")

    # ----- Step 3: select batch -----
    # 4-kind mode if any of the new fraction keys are present; otherwise
    # fall back to the legacy frontier/adversarial split.
    kind_fraction_keys = ("frontier_fraction", "adversarial_fraction", "exploit_fraction", "cma_fraction")
    has_kind_fractions = any(
        k in sel_cfg for k in kind_fraction_keys if k != "frontier_fraction"
    )
    if has_kind_fractions:
        kind_fractions = {
            k.replace("_fraction", ""): float(sel_cfg[k])
            for k in kind_fraction_keys
            if k in sel_cfg
        }
        selected = select_batch(
            acq_result["candidates"],
            batch_size=int(sel_cfg["batch_size"]),
            kind_fractions=kind_fractions,
        )
    else:
        selected = select_batch(
            acq_result["candidates"],
            batch_size=int(sel_cfg["batch_size"]),
            frontier_fraction=float(sel_cfg.get("frontier_fraction", 0.85)),
        )
    print(f"Selected {len(selected)} candidates for IX submission")

    selected_manifest = paths.iter_manifest(state.iteration)
    write_selected_manifest(
        selected,
        out_path=selected_manifest,
        iteration=state.iteration,
        geologies=geologies,
        extras={"selection": sel_cfg},
    )

    # ----- Step 4: stage IX array via Julia -----
    stage_result = stage_iteration(
        julia_repo=julia_repo,
        julia_config=config["paths"]["julia_config"],
        surrogate_repo=surrogate_repo,
        manifest_path=selected_manifest,
        stage_root=paths.iter_ix_stage_dir(state.iteration),
        output_prefix=f"al_{state.run_id}_iter{state.iteration:04d}",
        cpus=int(ix_cfg.get("cpus_per_task", 2)),
        mem=str(ix_cfg.get("mem_per_task", "8GB")),
        time_limit=str(ix_cfg.get("time_per_run", "00:30:00")),
        np_procs=int(ix_cfg.get("np", 2)),
        max_concurrent=int(ix_cfg.get("max_concurrent", 70)),
        job_name=f"AL_IX_{state.run_id}_{state.iteration:04d}",
    )
    print(f"Staged IX array at {stage_result.stage_run_dir}, sbatch={stage_result.sbatch_path}")

    # ----- Step 5: submit IX array, then ingest with dependency -----
    ix_job_id = submit_sbatch(stage_result.sbatch_path)
    print(f"Submitted IX array as job {ix_job_id}")

    # IX worker writes outputs to `FILEPATHS.h5s_dir_out` from the Julia config
    # (a shared dir like /scratch/.../intersect_data/h5s_out, NOT the staging
    # dir). Pull that path here so ingest knows where to find this iteration's
    # files. Filenames are unique per iteration via the al_<run>_iter<NN>
    # output prefix, so ingest can scope to just this iter's outputs.
    julia_config_path = config["paths"]["julia_config"]
    if not Path(julia_config_path).is_absolute():
        julia_config_path = str(julia_repo / julia_config_path)
    with open(julia_config_path, "r") as f:
        julia_config_payload = json.load(f)
    ix_output_root = julia_config_payload.get("FILEPATHS", {}).get("h5s_dir_out")
    if not ix_output_root:
        print(f"ERROR: julia config {julia_config_path} missing FILEPATHS.h5s_dir_out",
              file=sys.stderr)
        return 1

    ingest_template = REPO_ROOT / "sbatch" / "ingest.sbatch.template"
    ingest_sbatch = paths.sbatch_dir / f"ingest_iter_{state.iteration:04d}.sbatch"
    write_rendered(
        ingest_template,
        ingest_sbatch,
        {
            "RUN_ROOT": str(args.run_root),
            "REPO_ROOT": str(REPO_ROOT),
            "ITER": str(state.iteration),
            "ARRAY_TASKS_JSON": str(stage_result.tasks_json),
            "IX_OUTPUT_DIR": ix_output_root,
            "LOG_OUT": str(paths.logs_dir / f"ingest_iter_{state.iteration:04d}.out"),
            "LOG_ERR": str(paths.logs_dir / f"ingest_iter_{state.iteration:04d}.err"),
            "JOB_NAME": f"AL_INGEST_{state.run_id}_{state.iteration:04d}",
        },
    )
    ingest_job_id = submit_sbatch(ingest_sbatch, dependency=f"afterany:{ix_job_id}")
    print(f"Submitted ingest as job {ingest_job_id} (afterany:{ix_job_id})")

    # ----- Step 6: persist state -----
    rec = state.get_iter(state.iteration) or IterationRecord(iteration=state.iteration)
    rec.submitted = stage_result.tasks_count
    rec.wallclock_acquire_min = acq_elapsed_min
    rec.ix_array_job_id = ix_job_id
    rec.ingest_job_id = ingest_job_id
    rec.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    state.upsert_iter(rec)
    state.save(paths.state_file)

    if wandb_handle.is_active:
        wandb_handle.log(
            {
                "iteration": state.iteration,
                "n_candidates_submitted": stage_result.tasks_count,
                "wallclock_acquire_min": acq_elapsed_min,
                "from_scratch": bool(was_from_scratch),
            },
            step=state.iteration,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
