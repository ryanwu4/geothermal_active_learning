#!/usr/bin/env python3
"""Local hybrid driver for the surrogate-free baseline (CMA-ES / random over raw IX).

Mirrors ``scripts/run_al_local.py`` but with the train+acquire half replaced by
a thin ``BaselineOptimizer.ask() → tell()`` loop. The Sherlock-side pipeline
(Julia stage → SLURM array → preprocess_h5 + fitness aggregation) is identical
in shape, just routed through ``phase_ingest_baseline.py`` instead of
``phase_ingest.py``.

Usage:
    python scripts/run_baseline_global.py --config configs/baseline_global.json
    # resume an existing run:
    python scripts/run_baseline_global.py --config configs/baseline_global.json --run-id <id>

Per iteration:
  1. Pull state.json (+ optimizer_state.pkl if it exists) from Sherlock.
  2. optimizer.ask() → (popsize, num_wells, 3) candidate coords.
  3. Emit per-candidate .jl + snapshot JSONs locally; write ensemble manifest.
  4. Push acquire dir + manifest (with paths remapped to Sherlock) to Sherlock.
  5. Julia stage → IX array sbatch → submit; submit baseline ingest with
     afterany dependency on the IX array.
  6. Poll the ingest job until terminal.
  7. Pull fitness.json from Sherlock; feed to optimizer.tell().
  8. Save optimizer state and push back to Sherlock.

SSH-authentication failures write ``NEEDS_AUTH.md`` and exit 2 — re-run with
``--run-id <id>`` after re-establishing the ControlMaster socket.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import pickle
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from orchestrator.baseline_optimizer import (  # noqa: E402
    BaselineOptimizer, build_optimizer, intersect_valid_xy_indices, load_optimizer,
)
from orchestrator.log import init_run  # noqa: E402
from orchestrator.remote import RemoteSession, SshUnavailable, TERMINAL_JOB_STATES  # noqa: E402
from orchestrator.slurm import render_template  # noqa: E402
from orchestrator.stage import stage_iteration  # noqa: E402
from orchestrator.state import IterationRecord, RunState, new_run_id, new_wandb_run_id  # noqa: E402

# Reuse the SSH plumbing & helpers from the AL driver verbatim — keeping a
# single source of truth for ControlMaster handling, manifest path-rewriting,
# Sherlock env wrapping, and polling.
from run_al_local import (  # type: ignore  # noqa: E402
    _build_geology_remappings,
    _make_sherlock_runner,
    _poll_until_terminal,
    _rewrite_manifest_paths,
    _send_email_notice,
    _wrap_with_sherlock_env,
    _validate_run_id,
)


# ----------------------------------------------------------------------
# Workspace layout
# ----------------------------------------------------------------------


def _ensure_workspace(ws: Path) -> None:
    for sub in ("optimizer_state", "manifests", "logs", "acquire", "rendered_sbatch"):
        (ws / sub).mkdir(parents=True, exist_ok=True)


def _state_mirror_path(ws: Path) -> Path:
    return ws / "state_mirror.json"


def _optimizer_state_local(ws: Path) -> Path:
    return ws / "optimizer_state.pkl"


def _fitness_local(ws: Path, iteration: int) -> Path:
    return ws / f"iter_{iteration:04d}" / "fitness.json"


# ----------------------------------------------------------------------
# Config / IO
# ----------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_geology_list(path: Path) -> list[dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload.get("geologies", [])


# ----------------------------------------------------------------------
# Auth-failure scaffolding (mirrors run_al_local._write_needs_auth shape)
# ----------------------------------------------------------------------


def _write_needs_auth(
    ws: Path,
    *,
    config_path: Path,
    run_id: str | None,
    iteration: int | None,
    last_step: str,
    error: BaseException,
    ssh_host: str,
    notify_email: str | None = None,
    notify_msmtp_account: str = "gmail",
) -> None:
    resume_args = f"--config {shlex.quote(str(config_path))}"
    if run_id:
        resume_args += f" --run-id {shlex.quote(run_id)}"
    body = f"""# Baseline run paused — Sherlock SSH unavailable

Driver could not reach Sherlock through the ControlMaster socket. To resume:

- Run id: `{run_id or "(not bootstrapped yet)"}`
- Iteration in progress: `{iteration if iteration is not None else "(unknown)"}`
- Last successful step: `{last_step}`
- Underlying error:

      {error}

## Resume

1. In a tmux pane, re-open the ControlMaster:

       ssh {ssh_host}

2. Verify:

       ssh -O check {ssh_host}

3. Re-launch:

       python scripts/run_baseline_global.py {resume_args}
"""
    (ws / "NEEDS_AUTH.md").write_text(body)
    print(f"[baseline-driver] wrote {ws / 'NEEDS_AUTH.md'} — exiting.", file=sys.stderr)
    if notify_email:
        host = socket.gethostname()
        subject = f"[baseline] SSH to Sherlock unavailable on {host} (run={run_id or 'unbootstrapped'})"
        _send_email_notice(to_addr=notify_email, msmtp_account=notify_msmtp_account,
                           subject=subject, body=body)


# ----------------------------------------------------------------------
# Bootstrap remote run
# ----------------------------------------------------------------------


def _bootstrap_remote_run(
    remote: RemoteSession, cfg: dict, local_config_path: Path,
    explicit_run_id: str | None,
) -> str:
    """Push the config to Sherlock and ssh-run start_baseline_run.py to create state.json."""
    run_id = explicit_run_id or new_run_id(prefix=cfg.get("run_id_prefix", "baseline"))
    scratch_root = cfg["paths"]["scratch_root"]
    remote_repo_root = cfg["compute"]["remote_repo_root"]
    staging_dir = f"{scratch_root}/baseline_config_staging"
    remote_config_path = f"{staging_dir}/{run_id}.json"

    remote.run(["mkdir", "-p", staging_dir])
    remote.push(local_config_path, remote_config_path)

    print(f"[baseline-driver] bootstrapping remote run {run_id} (config={remote_config_path})")
    proc = remote.run(
        ["bash", "-lc", _wrap_with_sherlock_env(
            f"python scripts/start_baseline_run.py "
            f"--config {shlex.quote(remote_config_path)} "
            f"--run-id {shlex.quote(run_id)}"
        )],
        cwd=remote_repo_root, check=True,
    )
    print(proc.stdout.strip())
    return run_id


# ----------------------------------------------------------------------
# State helpers
# ----------------------------------------------------------------------


def _pull_state(remote: RemoteSession, remote_run_root: str, ws: Path) -> RunState:
    dst = _state_mirror_path(ws)
    remote.pull(f"{remote_run_root}/state.json", dst)
    return RunState.load(dst)


def _push_state(remote: RemoteSession, remote_run_root: str, state: RunState, ws: Path) -> None:
    dst = _state_mirror_path(ws)
    state.save(dst)
    remote.push(dst, f"{remote_run_root}/state.json")


def _check_done(remote: RemoteSession, remote_run_root: str) -> bool:
    return remote.run(["test", "-f", f"{remote_run_root}/done.json"], check=False).returncode == 0


def _pull_optimizer_state(remote: RemoteSession, remote_run_root: str, ws: Path) -> bool:
    """Pull optimizer_state.pkl if it exists on Sherlock. Returns True on success."""
    remote_path = f"{remote_run_root}/optimizer_state.pkl"
    if remote.run(["test", "-f", remote_path], check=False).returncode != 0:
        return False
    remote.pull(remote_path, _optimizer_state_local(ws))
    return True


def _push_optimizer_state(remote: RemoteSession, remote_run_root: str, ws: Path) -> None:
    remote.push(_optimizer_state_local(ws), f"{remote_run_root}/optimizer_state.pkl")


# ----------------------------------------------------------------------
# Geology + bounds resolution
# ----------------------------------------------------------------------


def _load_geologies_resolved(cfg: dict) -> tuple[list[dict], np.ndarray, int, int]:
    """Load the local geology list, validate H5 files, and compute shared bounds.

    Returns (geology_entries, valid_xy_indices, nx, ny).
    """
    geology_path = Path(cfg["compute"]["local_geologies_config"]).expanduser().resolve()
    entries = _load_geology_list(geology_path)
    missing = [e for e in entries if not Path(e["geology_h5_file"]).exists()]
    if missing:
        raise RuntimeError(
            f"Local geology H5 files missing ({len(missing)}). First: "
            f"{missing[0]['geology_h5_file']}. Either rsync from Sherlock or fix paths."
        )
    h5_paths = [Path(e["geology_h5_file"]) for e in entries]
    valid_xy = intersect_valid_xy_indices(h5_paths)

    # nx, ny from the first geology's temperature grid.
    import h5py
    with h5py.File(h5_paths[0], "r") as f:
        temp0_shape = f["Input/Temperature0"].shape  # (z, x, y)
    _, nx, ny = temp0_shape
    print(f"[baseline-driver] geologies={len(entries)} nx={nx} ny={ny} "
          f"valid_intersection_cells={valid_xy.shape[0]}")
    return entries, valid_xy, int(nx), int(ny)


# ----------------------------------------------------------------------
# Snapshot emission (baseline: no surrogate prediction)
# ----------------------------------------------------------------------


def _jl_quote(value) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _emit_baseline_snapshot(
    *,
    run_token: int,
    generation: int,
    pop_idx: int,
    coords_xyz: np.ndarray,        # (num_wells, 3)
    is_injector_list: list[bool],
    geology_entries: list[dict],
    well_configs_dir: Path,
    snapshots_json_dir: Path,
) -> dict:
    """Write a .jl well config + snapshot JSON for one baseline candidate.

    Returns the manifest snapshot record (with the full
    ``well_config_paths_by_geology`` array so the Julia stager fans this single
    candidate out into K IX tasks).
    """
    snapshot_id = f"run{run_token:06d}_step{generation:04d}_baseline_p{pop_idx:03d}"
    jl_path = well_configs_dir / f"{snapshot_id}.jl"
    json_path = snapshots_json_dir / f"{snapshot_id}.json"

    # Inline .jl writer — baseline has no surrogate prediction to embed.
    lines = []
    lines.append("# Auto-generated by run_baseline_global.py (no-surrogate ablation)")
    lines.append(f"# Generation: {generation}, population index: {pop_idx}")
    lines.append(f"# Evaluated on {len(geology_entries)} geologies (ensemble mean fitness)")
    # First geology's metadata (geology-agnostic .jl; Julia stager rewrites
    # config_id per geology from the manifest entries).
    first = geology_entries[0]
    geo_file = str(Path(first["geology_h5_file"]).resolve())
    geo_name = first.get("geology_name") or Path(geo_file).stem
    geo_config_id = first.get("scenario") or first.get("geology_config_id")
    lines.append(f"# Geology file: {geo_file}")
    lines.append(f"# Geology name: {geo_name}")
    if geo_config_id is not None:
        lines.append(f"geology_config_id = {_jl_quote(geo_config_id)}")
    lines.append(f"geology_source_file = {_jl_quote(geo_file)}")
    lines.append(f"geology_source_name = {_jl_quote(geo_name)}")
    lines.append("wells = [")
    for w, (x, y, z) in enumerate(coords_xyz):
        j_idx = int(round(float(x))) + 1
        i_idx = int(round(float(y))) + 1
        k_idx = int(round(float(z))) + 1
        is_inj = is_injector_list[w]
        well_type = '"INJECTOR"' if is_inj else '"PRODUCER"'
        rate = 8000.0 if is_inj else -8000.0
        lines.append(f"    ({i_idx}, {j_idx}, {k_idx}, {well_type}, {rate}),")
    lines.append("]\n")
    jl_path.write_text("\n".join(lines))

    wells_json = []
    for w, (x, y, z) in enumerate(coords_xyz):
        is_inj = is_injector_list[w]
        j_idx = int(round(float(x))) + 1
        i_idx = int(round(float(y))) + 1
        k_idx = int(round(float(z))) + 1
        wells_json.append({
            "well_id": int(w),
            "type": "injector" if is_inj else "producer",
            "x": float(x), "y": float(y), "z": float(z),
            "i_idx": int(i_idx), "j_idx": int(j_idx), "k_idx": int(k_idx),
            "rate": float(8000.0 if is_inj else -8000.0),
        })
    snap_payload = {
        "snapshot_id": snapshot_id,
        "run_id": int(run_token),
        "iteration": int(generation),
        "kind": "baseline",
        "mode": "ensemble",
        "wells": wells_json,
    }
    json_path.write_text(json.dumps(snap_payload, indent=2))

    well_config_paths_by_geology = []
    for g in geology_entries:
        well_config_paths_by_geology.append({
            "geology_index": int(g["geology_index"]),
            "geology_name": g.get("geology_name") or Path(g["geology_h5_file"]).stem,
            "geology_file": str(Path(g["geology_h5_file"]).resolve()),
            "geology_config_id": g.get("scenario") or g.get("geology_config_id"),
            "geology_scenario_name": g.get("geology_scenario_name", ""),
            "geology_sample_num": g.get("geology_sample_num"),
            "well_config_path": str(jl_path),
            "predicted_discounted_total_revenue": float("nan"),
        })

    return {
        "snapshot_id": snapshot_id,
        "run_id": int(run_token),
        "iteration": int(generation),
        "kind": "baseline",
        "mode": "ensemble",
        "json_path": str(json_path),
        "well_config_path": str(jl_path),
        "well_config_paths_by_geology": well_config_paths_by_geology,
        # NaN-safe: keep these fields present so the Julia stager (which casts
        # to Float64) doesn't choke. NaN → Julia parses, then writes NaN into
        # tasks JSON. Downstream baseline ingest ignores these fields.
        "predicted_discounted_total_revenue": 0.0,
        "predicted_emv": 0.0,
    }


def _write_manifest(
    *,
    snapshot_records: list[dict],
    out_path: Path,
    iteration: int,
    geology_entries: list[dict],
    extras: dict | None = None,
) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "mode": "ensemble",
        "snapshot_count": len(snapshot_records),
        "snapshots": snapshot_records,
        "geology_metadata": [
            {
                "geology_index": int(g["geology_index"]),
                "geology_name": g.get("geology_name") or Path(g["geology_h5_file"]).stem,
                "geology_file": str(Path(g["geology_h5_file"]).resolve()),
            }
            for g in geology_entries
        ],
    }
    if extras:
        manifest.update(extras)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path


# ----------------------------------------------------------------------
# Sherlock-side ingest sbatch rendering
# ----------------------------------------------------------------------


def _render_and_push_ingest_sbatch(
    *,
    remote: RemoteSession,
    remote_run_root: str,
    remote_repo_root: str,
    iteration: int,
    run_id: str,
    array_tasks_json_remote: str,
    ix_output_dir_remote: str,
    manifest_remote: str,
    fitness_out_remote: str,
    ws: Path,
) -> str:
    template = REPO_ROOT / "sbatch" / "ingest_baseline.sbatch.template"
    rendered = render_template(template, {
        "RUN_ROOT": remote_run_root,
        "REPO_ROOT": remote_repo_root,
        "ITER": str(iteration),
        "ARRAY_TASKS_JSON": array_tasks_json_remote,
        "IX_OUTPUT_DIR": ix_output_dir_remote,
        "MANIFEST_PATH": manifest_remote,
        "FITNESS_OUT": fitness_out_remote,
        "LOG_OUT": f"{remote_run_root}/logs/ingest_baseline_iter_{iteration:04d}.out",
        "LOG_ERR": f"{remote_run_root}/logs/ingest_baseline_iter_{iteration:04d}.err",
        "JOB_NAME": f"BASE_INGEST_{run_id}_{iteration:04d}",
    })
    local_sbatch = ws / "rendered_sbatch" / f"ingest_baseline_iter_{iteration:04d}.sbatch"
    local_sbatch.write_text(rendered)
    local_sbatch.chmod(0o755)
    remote_path = f"{remote_run_root}/sbatch_rendered/ingest_baseline_iter_{iteration:04d}.sbatch"
    remote.push(local_sbatch, remote_path)
    return remote_path


# ----------------------------------------------------------------------
# Iteration body
# ----------------------------------------------------------------------


def _build_or_load_optimizer(
    *, cfg: dict, ws: Path, geology_entries: list[dict],
    valid_xy: np.ndarray, nx: int, ny: int, remote: RemoteSession,
    remote_run_root: str,
) -> BaselineOptimizer:
    """Pull optimizer_state.pkl from Sherlock if it exists, else construct fresh."""
    if _pull_optimizer_state(remote, remote_run_root, ws):
        print(f"[baseline-driver] resuming optimizer state from Sherlock "
              f"({_optimizer_state_local(ws)})")
        return load_optimizer(_optimizer_state_local(ws))

    opt_cfg = cfg["optimizer"]
    wells = cfg["wells"]
    fixed_depth = int(opt_cfg.get("fixed_depth", 50))
    fixed_depth_per_well = [
        int(w.get("depth", fixed_depth)) for w in wells
    ]
    num_wells = len(wells)
    return build_optimizer(
        kind=str(opt_cfg.get("type", "cmaes")),
        num_wells=num_wells,
        nx=nx, ny=ny,
        edge_buffer=int(opt_cfg.get("edge_buffer", 10)),
        fixed_depth_per_well=fixed_depth_per_well,
        popsize=int(opt_cfg["popsize"]),
        seed=int(opt_cfg.get("seed", 42)),
        valid_xy_indices=valid_xy,
        sigma_init=float(opt_cfg.get("sigma_init", 5.0)),
    )


def _wells_is_injector(cfg: dict) -> list[bool]:
    return [str(w["type"]).lower() == "injector" for w in cfg["wells"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Baseline AL config (e.g. configs/baseline_global.json)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Resume an existing run. Omit to bootstrap.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    compute = cfg.get("compute") or {}
    for required in ("ssh_host", "ssh_control_path", "remote_repo_root",
                     "local_workspace", "local_geologies_config"):
        if not compute.get(required):
            print(f"ERROR: compute.{required} is required", file=sys.stderr)
            return 1

    ws = Path(compute["local_workspace"]).expanduser().resolve()
    _ensure_workspace(ws)

    # Lock so two drivers don't clobber the same workspace.
    lock_path = ws / ".driver.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"ERROR: another driver already holds {lock_path}.", file=sys.stderr)
        return 1
    os.write(lock_fd, f"{os.getpid()}\n".encode())

    remote = RemoteSession(host=compute["ssh_host"], control_path=compute["ssh_control_path"])
    poll_every = int(compute.get("poll_interval_sec", 600))

    # wandb on bend (online); Sherlock ingest is offline and sync'd later.
    os.environ["WANDB_MODE"] = compute.get("wandb_mode", "online")
    os.environ["WANDB_DIR"] = str(ws)
    (ws / "wandb").mkdir(parents=True, exist_ok=True)
    wandb_handle = None

    # Geology + bounds resolution (one-shot, all iters share).
    geology_entries, valid_xy, nx, ny = _load_geologies_resolved(cfg)
    is_injector_list = _wells_is_injector(cfg)

    last_step = "start"
    run_id = args.run_id
    if run_id is not None:
        try:
            _validate_run_id(run_id)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    iteration: int | None = None

    try:
        remote.check_alive()
        last_step = "ssh_alive"
        scratch_root = cfg["paths"]["scratch_root"]
        if run_id is None:
            run_id = _bootstrap_remote_run(remote, cfg, args.config, explicit_run_id=None)
            last_step = f"bootstrapped run_id={run_id}"
        else:
            probe = remote.run(["test", "-f", f"{scratch_root}/{run_id}/state.json"], check=False)
            if probe.returncode != 0:
                print(f"[baseline-driver] --run-id {run_id} given but no remote state — bootstrapping.")
                run_id = _bootstrap_remote_run(remote, cfg, args.config, explicit_run_id=run_id)
                last_step = f"bootstrapped run_id={run_id} (resumed-id)"
        print(f"[baseline-driver] driving run_id={run_id}")
        remote_run_root = f"{scratch_root}/{run_id}"
        remote_repo_root = compute["remote_repo_root"]

        # Read Sherlock's julia config to locate the shared IX output dir.
        julia_repo = cfg["paths"]["julia_repo"]
        julia_config = cfg["paths"]["julia_config"]
        if not julia_config.startswith("/"):
            julia_config = f"{julia_repo}/{julia_config}"
        proc = remote.run(["cat", julia_config], check=True)
        julia_payload = json.loads(proc.stdout)
        ix_output_root = julia_payload.get("FILEPATHS", {}).get("h5s_dir_out")
        if not ix_output_root:
            raise RuntimeError("julia config missing FILEPATHS.h5s_dir_out")
        last_step = "loaded julia_config"

        max_iters = int(cfg["stopping"].get("max_iterations", 30))

        # Build optimizer once; persisted across iters via pickle.
        optimizer = _build_or_load_optimizer(
            cfg=cfg, ws=ws, geology_entries=geology_entries,
            valid_xy=valid_xy, nx=nx, ny=ny, remote=remote,
            remote_run_root=remote_run_root,
        )

        while True:
            if _check_done(remote, remote_run_root):
                print(f"[baseline-driver] done.json present — run complete.")
                (ws / "NEEDS_AUTH.md").unlink(missing_ok=True)
                if wandb_handle is not None:
                    wandb_handle.finish()
                return 0

            state = _pull_state(remote, remote_run_root, ws)
            iteration = state.iteration
            last_step = f"pulled state iter={iteration}"
            print(f"[baseline-driver] iter {iteration} starting "
                  f"(optimizer.generation={optimizer.generation})")

            if iteration >= max_iters:
                print(f"[baseline-driver] reached max_iterations={max_iters} — writing done.json.")
                done = {"stopped_at_iteration": iteration, "reason": "max_iterations"}
                done_local = ws / "done.json"
                done_local.write_text(json.dumps(done, indent=2))
                remote.push(done_local, f"{remote_run_root}/done.json")
                if wandb_handle is not None:
                    wandb_handle.finish()
                return 0

            if wandb_handle is None:
                wandb_handle = init_run(
                    run_id=state.wandb_run_id,
                    project=cfg["wandb"]["project"],
                    entity=cfg["wandb"].get("entity"),
                    config={"baseline_run_id": state.run_id, "compute": "local",
                            "optimizer": cfg["optimizer"]},
                    tags=cfg["wandb"].get("tags"),
                    name=state.run_id,
                    resume="allow",
                )

            # ---- 1. ask() ----
            t0 = time.time()
            coords = optimizer.ask()  # (popsize, num_wells, 3)
            popsize = coords.shape[0]
            print(f"[baseline-driver] proposed {popsize} candidates "
                  f"(generation={optimizer.generation})")
            last_step = f"ask iter {iteration}"

            # ---- 2. Emit per-candidate artifacts + manifest locally ----
            local_acquire_dir = ws / "acquire" / f"iter_{iteration:04d}"
            well_configs_dir = local_acquire_dir / "well_configs"
            snapshots_json_dir = local_acquire_dir / "snapshots_json"
            well_configs_dir.mkdir(parents=True, exist_ok=True)
            snapshots_json_dir.mkdir(parents=True, exist_ok=True)

            snapshot_records: list[dict] = []
            for p in range(popsize):
                rec = _emit_baseline_snapshot(
                    run_token=0,
                    generation=iteration,
                    pop_idx=p,
                    coords_xyz=coords[p],
                    is_injector_list=is_injector_list,
                    geology_entries=geology_entries,
                    well_configs_dir=well_configs_dir,
                    snapshots_json_dir=snapshots_json_dir,
                )
                snapshot_records.append(rec)

            manifest_path = ws / "manifests" / f"manifest_iter_{iteration:04d}.json"
            _write_manifest(
                snapshot_records=snapshot_records,
                out_path=manifest_path,
                iteration=iteration,
                geology_entries=geology_entries,
                extras={"optimizer": cfg["optimizer"]},
            )
            ask_min = (time.time() - t0) / 60.0

            # ---- 3. Push acquire dir + manifest (path-rewritten) to Sherlock ----
            remote_acquire_dir = f"{remote_run_root}/iter_{iteration:04d}/acquire"
            remote.run(["mkdir", "-p", remote_acquire_dir])
            remote.push(str(local_acquire_dir) + "/", remote_acquire_dir)

            remappings = [
                (str(local_acquire_dir), remote_acquire_dir),
                *_build_geology_remappings(cfg),
            ]
            rewritten = _rewrite_manifest_paths(manifest_path, remappings)
            rewritten_local = ws / "manifests" / f"manifest_iter_{iteration:04d}.remote.json"
            rewritten_local.write_text(json.dumps(rewritten, indent=2))
            remote_manifest = f"{remote_run_root}/manifests/manifest_iter_{iteration:04d}.json"
            remote.run(["mkdir", "-p", f"{remote_run_root}/manifests"])
            remote.push(rewritten_local, remote_manifest)
            last_step = "pushed acquire dir + manifest"

            # ---- 4. Stage IX on Sherlock ----
            stage_root_remote = f"{remote_run_root}/iter_{iteration:04d}/ix_stage"
            remote.run(["mkdir", "-p", stage_root_remote])
            ix_cfg = cfg["intersect"]
            stage = stage_iteration(
                julia_repo=Path(cfg["paths"]["julia_repo"]),
                julia_config=cfg["paths"]["julia_config"],
                surrogate_repo=Path(cfg["paths"]["surrogate_repo"]),
                manifest_path=Path(remote_manifest),
                stage_root=Path(stage_root_remote),
                output_prefix=f"baseline_{run_id}_iter{iteration:04d}",
                cpus=int(ix_cfg.get("cpus_per_task", 2)),
                mem=str(ix_cfg.get("mem_per_task", "8GB")),
                time_limit=str(ix_cfg.get("time_per_run", "00:40:00")),
                np_procs=int(ix_cfg.get("np", 2)),
                max_concurrent=int(ix_cfg.get("max_concurrent", 50)),
                job_name=f"BASE_IX_{run_id}_{iteration:04d}",
                runner=_make_sherlock_runner(remote),
                skip_script_existence_check=True,
            )
            last_step = f"staged IX (sbatch={stage.sbatch_path})"

            ix_job = remote.submit_sbatch(str(stage.sbatch_path))
            print(f"[baseline-driver] submitted IX array as job {ix_job} "
                  f"({stage.tasks_count} tasks)")
            last_step = f"submitted IX job {ix_job}"

            # ---- 5. Submit baseline ingest with afterany dep ----
            fitness_out_remote = (
                f"{remote_run_root}/iter_{iteration:04d}/fitness.json"
            )
            ingest_sbatch_remote = _render_and_push_ingest_sbatch(
                remote=remote,
                remote_run_root=remote_run_root,
                remote_repo_root=remote_repo_root,
                iteration=iteration,
                run_id=run_id,
                array_tasks_json_remote=str(stage.tasks_json),
                ix_output_dir_remote=ix_output_root,
                manifest_remote=remote_manifest,
                fitness_out_remote=fitness_out_remote,
                ws=ws,
            )
            ingest_job = remote.submit_sbatch(
                ingest_sbatch_remote, dependency=f"afterany:{ix_job}",
            )
            print(f"[baseline-driver] submitted baseline ingest job {ingest_job} (afterany:{ix_job})")
            last_step = f"submitted ingest job {ingest_job}"

            # Record IX + ingest job ids on the iteration record.
            rec = state.get_iter(iteration) or IterationRecord(iteration=iteration)
            rec.submitted = stage.tasks_count
            rec.ix_array_job_id = ix_job
            rec.ingest_job_id = ingest_job
            rec.wallclock_acquire_min = ask_min
            rec.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            state.upsert_iter(rec)
            _push_state(remote, remote_run_root, state, ws)

            (ws / "NEEDS_AUTH.md").unlink(missing_ok=True)

            # ---- 6. Poll until ingest terminal ----
            final = _poll_until_terminal(remote, ingest_job, every=poll_every, label="baseline-ingest")
            last_step = f"ingest {ingest_job} → {final}"
            if final != "COMPLETED":
                print(f"[baseline-driver] ingest finished non-COMPLETED ({final}). "
                      f"Logs: {remote_run_root}/logs/ingest_baseline_iter_{iteration:04d}.err "
                      f"on Sherlock. Re-run with --run-id {run_id} after fixing.")
                if wandb_handle is not None:
                    wandb_handle.log({"ingest_terminal_state": final}, step=iteration)
                    wandb_handle.finish()
                return 1

            # ---- 7. Pull fitness.json + tell() ----
            local_fitness = _fitness_local(ws, iteration)
            local_fitness.parent.mkdir(parents=True, exist_ok=True)
            remote.pull(fitness_out_remote, local_fitness)
            with open(local_fitness, "r") as f:
                fitness_payload = json.load(f)
            print(f"[baseline-driver] pulled fitness for iter {iteration}: "
                  f"best_in_batch={fitness_payload.get('best_ensemble_revenue_in_batch')} "
                  f"best_so_far={fitness_payload.get('best_ensemble_revenue_so_far')}")

            # Build the (popsize,) fitness array aligned to the optimizer's
            # ask() return. Snapshot ids are deterministic
            # ("..._p000", "..._p001", ...) so we sort by pop_idx.
            cands_by_pop: dict[int, float] = {}
            for c in fitness_payload.get("candidates", []):
                sid = str(c.get("snapshot_id", ""))
                # snapshot_id ends in "_p###"
                p_token = sid.rsplit("_p", 1)
                if len(p_token) != 2:
                    continue
                try:
                    p_idx = int(p_token[1])
                except ValueError:
                    continue
                v = c.get("ensemble_mean_revenue", float("nan"))
                cands_by_pop[p_idx] = float(v) if v is not None else float("nan")
            fitnesses = np.array(
                [cands_by_pop.get(p, float("nan")) for p in range(popsize)],
                dtype=np.float64,
            )
            n_finite = int(np.isfinite(fitnesses).sum())
            print(f"[baseline-driver] tell(): {n_finite}/{popsize} finite fitnesses, "
                  f"max={np.nanmax(fitnesses) if n_finite else 'nan'}")

            optimizer.tell(coords, fitnesses)
            optimizer.save_state(_optimizer_state_local(ws))
            _push_optimizer_state(remote, remote_run_root, ws)
            last_step = f"tell(): generation now {optimizer.generation}"

            # wandb log
            if wandb_handle is not None and wandb_handle.is_active:
                wandb_handle.log({
                    "iteration": iteration,
                    "generation": optimizer.generation,
                    "popsize": popsize,
                    "n_finite_in_batch": n_finite,
                    "best_ensemble_revenue_in_batch": fitness_payload.get("best_ensemble_revenue_in_batch"),
                    "best_ensemble_revenue_so_far": fitness_payload.get("best_ensemble_revenue_so_far"),
                    "n_submitted_tasks": stage.tasks_count,
                    "n_completed_tasks": fitness_payload.get("n_completed_tasks"),
                    "wallclock_ask_min": ask_min,
                }, step=iteration)

            # ---- 8. Advance state.iteration on Sherlock ----
            fresh = _pull_state(remote, remote_run_root, ws)  # ingest updated it
            fresh.iteration += 1
            _push_state(remote, remote_run_root, fresh, ws)
            last_step = f"advanced state to iter {fresh.iteration}"

    except SshUnavailable as e:
        _write_needs_auth(
            ws,
            config_path=args.config.resolve(),
            run_id=run_id,
            iteration=iteration,
            last_step=last_step,
            error=e,
            ssh_host=compute["ssh_host"],
            notify_email=compute.get("notify_email"),
            notify_msmtp_account=compute.get("notify_msmtp_account", "gmail"),
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
