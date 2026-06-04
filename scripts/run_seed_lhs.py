#!/usr/bin/env python3
"""Clean-start LHS seed pre-step — sample on bend, run IX + compile on Sherlock.

Produces the initial training set for a clean-start AL run that begins with NO
pretrained surrogate:
  1. LHS-sample ~``seed.n_seed_samples`` well configs per geology (depth
     randomized), locally on bend (no surrogate, no IX).
  2. Push the manifest + per-snapshot artifacts to Sherlock and stage an IX
     array via the same Julia stager the AL loop uses (remote runner).
  3. Submit the IX array, then a seed-ingest job (afterany) that preprocesses
     the outputs into ``seed_compiled.h5`` + a FRESH ``norm_config.json`` — at
     exactly the paths the AL config references (auto-wired).
  4. Poll the ingest job to terminal state.

Run BEFORE ``run_al_local.py`` with the SAME clean-start config:
    python scripts/run_seed_lhs.py --config configs/al_cma_surrogate_clean.json

This mirrors ``run_al_local.py``'s iteration body MINUS train/acquire, reusing
its hybrid SSH/stage/poll/path-rewrite helpers verbatim. Requires an
authenticated Sherlock ControlMaster socket (``ssh sherlock`` once), same as the
AL driver.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.acquire import GeologySpec, WellSpec
from orchestrator.remote import RemoteSession, SshUnavailable
from orchestrator.seed import build_seed_manifest
from orchestrator.slurm import render_template
from orchestrator.stage import stage_iteration

# Reuse the AL driver's hybrid helpers (sibling script — scripts/ is on sys.path
# when this runs, same pattern as phase_ingest_baseline importing phase_ingest).
from run_al_local import (  # type: ignore  # noqa: E402
    _build_geology_remappings,
    _load_config,
    _load_geology_list,
    _make_sherlock_runner,
    _poll_until_terminal,
    _read_julia_config,
    _rewrite_manifest_paths,
)


def _resolve_seed_id(cfg: dict, cli_seed_id: str | None) -> str:
    """Stable id used in the IX output prefix. Order: --seed-id, seed.seed_id,
    else the parent-dir name of the configured seed compiled H5.
    """
    if cli_seed_id:
        return cli_seed_id
    sid = (cfg.get("seed") or {}).get("seed_id")
    if sid:
        return str(sid)
    return Path(cfg["paths"]["bootstrap_compiled_h5"]).parent.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Clean-start AL config (must contain a 'seed' block).")
    parser.add_argument("--seed-id", type=str, default=None,
                        help="Override the seed id used in the IX output prefix.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    compute = cfg.get("compute") or {}
    for required in ("ssh_host", "ssh_control_path", "remote_repo_root",
                     "local_surrogate_repo", "local_geologies_config", "local_workspace"):
        if not compute.get(required):
            print(f"ERROR: compute.{required} is required", file=sys.stderr)
            return 1
    seed_cfg = cfg.get("seed")
    if not seed_cfg or not seed_cfg.get("n_seed_samples"):
        print("ERROR: config is missing a 'seed' block with n_seed_samples", file=sys.stderr)
        return 1

    seed_id = _resolve_seed_id(cfg, args.seed_id)
    acq_cfg = cfg.get("acquisition") or {}
    n_seed_samples = int(seed_cfg["n_seed_samples"])
    depth_min = int(seed_cfg.get("depth_min", acq_cfg.get("depth_min", 5)))
    depth_max = int(seed_cfg.get("depth_max", acq_cfg.get("depth_max", 70)))
    edge_buffer = int(seed_cfg.get("edge_buffer", acq_cfg.get("edge_buffer", 10)))
    seed_val = int(seed_cfg.get("seed", acq_cfg.get("seed", 42)))

    # ----- Local sampling on bend (uses LOCAL geology paths + LOCAL surrogate) --
    local_surrogate_repo = Path(compute["local_surrogate_repo"]).resolve()
    local_geo_path = Path(compute["local_geologies_config"]).expanduser().resolve()
    geo_entries = _load_geology_list(local_geo_path)
    geologies = [
        GeologySpec(
            geology_index=int(e["geology_index"]),
            geology_h5_file=str(Path(e["geology_h5_file"]).resolve()),
            geology_name=e.get("geology_name"),
        )
        for e in geo_entries
    ]
    missing = [g.geology_h5_file for g in geologies if not Path(g.geology_h5_file).exists()]
    if missing:
        print(f"ERROR: {len(missing)} local geology H5 files missing. First few: {missing[:3]}",
              file=sys.stderr)
        return 1
    wells = [WellSpec(type=w["type"], depth=int(w["depth"])) for w in cfg["wells"]]

    local_seed_dir = Path(compute["local_workspace"]).expanduser().resolve() / "seed"
    local_acquire_dir = local_seed_dir / "acquire"
    local_manifest = local_seed_dir / "manifests" / "seed_manifest.json"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    (local_seed_dir / "rendered_sbatch").mkdir(parents=True, exist_ok=True)

    print(f"[seed-driver] sampling {n_seed_samples} configs across {len(geologies)} geologies "
          f"(depth {depth_min}-{depth_max}, seed {seed_val}) → {local_manifest}")
    manifest_path, n_emitted = build_seed_manifest(
        surrogate_repo=local_surrogate_repo,
        geologies=geologies,
        wells=wells,
        n_seed_samples=n_seed_samples,
        acquire_dir=local_acquire_dir,
        manifest_path=local_manifest,
        depth_min=depth_min,
        depth_max=depth_max,
        edge_buffer=edge_buffer,
        seed=seed_val,
        iteration=0,
    )
    print(f"[seed-driver] emitted {n_emitted} seed candidates")

    # ----- Remote staging on Sherlock --------------------------------------
    remote = RemoteSession(host=compute["ssh_host"], control_path=compute["ssh_control_path"])
    poll_every = int(compute.get("poll_interval_sec", 600))
    remote_repo_root = compute["remote_repo_root"]
    remote_surrogate_repo = cfg["paths"]["surrogate_repo"]
    # Everything seed-related lives alongside the configured seed outputs.
    seed_compiled_h5 = cfg["paths"]["bootstrap_compiled_h5"]
    seed_norm_config = cfg["paths"]["norm_config_path"]
    remote_seed_root = str(Path(seed_compiled_h5).parent)
    economics_config = f"{remote_surrogate_repo.rstrip('/')}/configs/economics.json"

    try:
        remote.check_alive()
        ix_output_root = (
            _read_julia_config(remote, cfg).get("FILEPATHS", {}).get("h5s_dir_out")
        )
        if not ix_output_root:
            raise RuntimeError("julia config missing FILEPATHS.h5s_dir_out")

        remote_acquire_dir = f"{remote_seed_root}/acquire"
        for d in (remote_seed_root, remote_acquire_dir, f"{remote_seed_root}/manifests",
                  f"{remote_seed_root}/logs", f"{remote_seed_root}/sbatch_rendered",
                  f"{remote_seed_root}/ix_stage"):
            remote.run(["mkdir", "-p", d], check=True)

        # Push acquire artifacts; rewrite local paths → Sherlock paths in manifest.
        remote.push(str(local_acquire_dir) + "/", remote_acquire_dir)
        remappings = [
            (str(local_acquire_dir), remote_acquire_dir),
            *_build_geology_remappings(cfg),
        ]
        rewritten = _rewrite_manifest_paths(manifest_path, remappings)
        rewritten_local = local_seed_dir / "manifests" / "seed_manifest.remote.json"
        rewritten_local.write_text(json.dumps(rewritten, indent=2))
        remote_manifest = f"{remote_seed_root}/manifests/seed_manifest.json"
        remote.push(rewritten_local, remote_manifest)
        print(f"[seed-driver] pushed acquire dir + manifest to {remote_seed_root}")

        # Stage IX. output_prefix MUST carry an _iter token (geology-resolution
        # regex requirement); seed snapshots use iteration=0 → trailing _iter0000.
        ix_cfg = cfg["intersect"]
        stage = stage_iteration(
            julia_repo=Path(cfg["paths"]["julia_repo"]),
            julia_config=cfg["paths"]["julia_config"],
            surrogate_repo=Path(remote_surrogate_repo),
            manifest_path=Path(remote_manifest),
            stage_root=Path(f"{remote_seed_root}/ix_stage"),
            output_prefix=f"seed_{seed_id}_iter0000",
            cpus=int(ix_cfg.get("cpus_per_task", 2)),
            mem=str(ix_cfg.get("mem_per_task", "8GB")),
            time_limit=str(ix_cfg.get("time_per_run", "00:40:00")),
            np_procs=int(ix_cfg.get("np", 2)),
            max_concurrent=int(ix_cfg.get("max_concurrent", 70)),
            job_name=f"SEED_IX_{seed_id}",
            runner=_make_sherlock_runner(remote),
            skip_script_existence_check=True,
        )
        print(f"[seed-driver] staged {stage.tasks_count} IX tasks (sbatch={stage.sbatch_path})")
        if stage.tasks_count != n_emitted:
            print(f"[seed-driver] WARNING: staged tasks ({stage.tasks_count}) != emitted "
                  f"candidates ({n_emitted}) — expected 1:1 in per-geology mode.",
                  file=sys.stderr)

        ix_job = remote.submit_sbatch(str(stage.sbatch_path))
        print(f"[seed-driver] submitted IX array as job {ix_job}")

        # Render + push the seed-ingest sbatch; submit afterany.
        template = REPO_ROOT / "sbatch" / "seed_ingest.sbatch.template"
        rendered = render_template(template, {
            "JOB_NAME": f"SEED_INGEST_{seed_id}",
            "LOG_OUT": f"{remote_seed_root}/logs/seed_ingest.out",
            "LOG_ERR": f"{remote_seed_root}/logs/seed_ingest.err",
            "RUN_ROOT": remote_seed_root,
            "REPO_ROOT": remote_repo_root,
            "ARRAY_TASKS_JSON": str(stage.tasks_json),
            "IX_OUTPUT_DIR": ix_output_root,
            "SURROGATE_REPO": remote_surrogate_repo,
            "SEED_COMPILED_H5": seed_compiled_h5,
            "SEED_NORM_CONFIG": seed_norm_config,
            "ECONOMICS_CONFIG": economics_config,
        })
        local_ingest_sbatch = local_seed_dir / "rendered_sbatch" / "seed_ingest.sbatch"
        local_ingest_sbatch.write_text(rendered)
        local_ingest_sbatch.chmod(0o755)
        remote_ingest_sbatch = f"{remote_seed_root}/sbatch_rendered/seed_ingest.sbatch"
        remote.push(local_ingest_sbatch, remote_ingest_sbatch)
        ingest_job = remote.submit_sbatch(remote_ingest_sbatch, dependency=f"afterany:{ix_job}")
        print(f"[seed-driver] submitted seed-ingest as job {ingest_job} (afterany:{ix_job})")

        final = _poll_until_terminal(remote, ingest_job, every=poll_every, label="seed-ingest")
        if final != "COMPLETED":
            print(f"[seed-driver] seed-ingest finished non-COMPLETED ({final}). Inspect "
                  f"{remote_seed_root}/logs/seed_ingest.err on Sherlock.", file=sys.stderr)
            return 1
        print(f"[seed-driver] seed complete. Compiled H5: {seed_compiled_h5}\n"
              f"             Fresh norm config: {seed_norm_config}\n"
              f"             Now run: python scripts/run_al_local.py --config {args.config}")
        return 0

    except SshUnavailable as e:
        print(f"[seed-driver] Sherlock SSH unavailable: {e}\n"
              f"Re-establish the ControlMaster socket and re-run:\n"
              f"    ssh {compute['ssh_host']}   # accept Duo\n"
              f"    python scripts/run_seed_lhs.py --config {shlex.quote(str(args.config))}",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
