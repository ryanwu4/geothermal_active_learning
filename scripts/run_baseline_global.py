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
    _notify_run_event,
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


def _pull_per_candidate_metrics(
    remote: RemoteSession, remote_run_root: str, iteration: int, ws: Path,
) -> bool:
    """Mirror Sherlock's per_candidate_metrics.json for one iter to bend.

    The local plot suite (``plot_convergence.py``) reads
    ``<ws>/iter_NNNN/per_candidate_metrics.json``; mirror it so the
    revenue / well-position / wallclock plots can render off the
    baseline ingest's output. Surrogate-dependent plots (MAPE,
    pred-vs-real scatter, calibration, holdout) will silently skip
    because predicted_revenue is NaN in the baseline schema.
    """
    remote_path = f"{remote_run_root}/iter_{iteration:04d}/per_candidate_metrics.json"
    if remote.run(["test", "-f", remote_path], check=False).returncode != 0:
        return False
    local = ws / f"iter_{iteration:04d}" / "per_candidate_metrics.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    remote.pull(remote_path, local)
    return True


def _render_plots(ws: Path) -> None:
    """Invoke plot_convergence.py against the local workspace.

    Runs in a subprocess so a plot-side import (matplotlib backend, font
    cache, etc.) can't contaminate the driver. Plotting is diagnostic; a
    plot failure should never crash the AL loop.
    """
    state_path = _state_mirror_path(ws)
    if not state_path.exists():
        return
    try:
        with open(state_path) as f:
            payload = json.load(f)
    except Exception:
        return
    if not payload.get("history"):
        return
    out_dir = ws / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_script = REPO_ROOT / "scripts" / "plot_convergence.py"
    cmd = [sys.executable, str(plot_script), str(ws), "--out-dir", str(out_dir)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        if result.returncode != 0:
            print(f"[baseline-driver] plot rendering failed (rc={result.returncode}):\n"
                  f"{result.stderr[-1500:]}", file=sys.stderr)
        else:
            print(f"[baseline-driver] rendered plots → {out_dir}")
    except Exception as e:
        print(f"[baseline-driver] plot rendering bailed: {e}", file=sys.stderr)


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


def _decode_h5_scalar(v):
    """Decode an h5py scalar return into a plain Python type (bytes → str, np → py)."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.bytes_):
        return v.astype(str)
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _decode_h5_scalar(v.item())
        if v.size == 1:
            return _decode_h5_scalar(v.reshape(-1)[0])
        return [_decode_h5_scalar(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _read_geology_metadata(h5_path: Path) -> dict:
    """Read Metadata/{RepNum, ScenarioName, SampleNum} from a geology H5.

    Mirrors geothermal/active_learning_utils.py:read_geology_metadata so we
    don't need a runtime dep on the surrogate repo on the bend side. Missing
    keys come back as empty strings — Julia's CSV writer accepts those but
    chokes on ``None``/``nothing``.

    The derived ``geology_config_id`` follows the same convention as
    ``active_learning_utils.py:41`` — RepNum if present, else ScenarioName.
    """
    import h5py
    out: dict = {"rep_num": "", "scenario_name": "", "sample_num": "",
                 "geology_config_id": ""}
    with h5py.File(h5_path, "r") as f:
        meta_group = None
        if "Metadata" in f:
            meta_group = f["Metadata"]
        elif "metadata" in f:
            meta_group = f["metadata"]
        if meta_group is None:
            return out
        for src_key, dst_key in (("RepNum", "rep_num"),
                                 ("ScenarioName", "scenario_name"),
                                 ("SampleNum", "sample_num")):
            if src_key in meta_group:
                v = _decode_h5_scalar(meta_group[src_key][()])
                out[dst_key] = v if v is not None else ""
    rep_num = out["rep_num"]
    if rep_num not in (None, "", b""):
        out["geology_config_id"] = rep_num
    elif out["scenario_name"] not in (None, "", b""):
        out["geology_config_id"] = out["scenario_name"]
    return out


def _load_geologies_resolved(cfg: dict) -> tuple[list[dict], np.ndarray, int, int, int]:
    """Load the local geology list, validate H5 files, and compute shared bounds.

    Each returned entry is augmented with H5-metadata fields:
    ``rep_num``, ``scenario_name``, ``sample_num`` (empty string if absent).
    These feed into the manifest so the Julia stager's CSV writer sees plain
    strings/ints instead of ``None``/``nothing``.

    Returns (geology_entries, valid_xy_indices, nx, ny, nz_min) — ``nz_min`` is
    the minimum z-extent across geologies, used as an upper bound on free-depth
    optimization so we never propose a perforation deeper than the shallowest
    reservoir grid.
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

    # Augment each entry with H5 metadata for the manifest writer. AL's
    # ensemble path uses RepNum from the H5 as ``geology_config_id`` (see
    # active_learning_utils.py:41) — preferring it here keeps the IX output
    # scenario token aligned with AL even when the local JSON's "scenario"
    # field happens to disagree with the H5's recorded RepNum.
    for e, p in zip(entries, h5_paths):
        meta = _read_geology_metadata(p)
        e["geology_scenario_name"] = e.get("geology_scenario_name") or meta["scenario_name"]
        e["geology_sample_num"] = e.get("geology_sample_num") if e.get("geology_sample_num") is not None else meta["sample_num"]
        # Prefer H5 RepNum; fall back to the local JSON's ``scenario`` int.
        h5_config_id = meta["geology_config_id"]
        if h5_config_id not in (None, "", b""):
            e["geology_config_id"] = h5_config_id
        else:
            e["geology_config_id"] = e.get("scenario") or e.get("geology_config_id")
        if not e.get("geology_name"):
            e["geology_name"] = Path(e["geology_h5_file"]).stem

    # nx, ny from the first geology's temperature grid; nz_min = the smallest
    # *active* z-extent across geologies. "Active" here means layers that
    # contain at least one non-dead-rock cell (Temperature0 > -900). The raw
    # H5 z-dim is often much larger than the actual reservoir (e.g. 326 vs
    # ~70 productive layers), so using the raw dim as a depth-bound ceiling
    # would silently allow nonsense placements like z=200 below the basement.
    import h5py
    active_nz: list[int] = []
    shapes = []
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            temp0 = f["Input/Temperature0"][:]  # (z, x, y)
        shapes.append(temp0.shape)
        per_layer_any = (temp0 > -900).any(axis=(1, 2))  # bool, (z,)
        if not per_layer_any.any():
            raise RuntimeError(f"No active layers in geology {p}; cannot bound depth.")
        # Last active layer index + 1 = active z-extent.
        active_nz.append(int(np.where(per_layer_any)[0][-1]) + 1)
    nz_min = min(active_nz)
    nxs = {s[1] for s in shapes}
    nys = {s[2] for s in shapes}
    if len(nxs) != 1 or len(nys) != 1:
        raise RuntimeError(
            f"Geology grids disagree on (x, y) extent: nx={nxs}, ny={nys}. "
            f"Cannot share a single optimizer bound across geologies."
        )
    nx, ny = nxs.pop(), nys.pop()
    print(f"[baseline-driver] geologies={len(entries)} nx={nx} ny={ny} "
          f"active_nz_per_geology={active_nz} (using nz_min={nz_min}) "
          f"valid_intersection_cells={valid_xy.shape[0]}")
    return entries, valid_xy, int(nx), int(ny), int(nz_min)


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

    def _safe(v, default=""):
        # Julia CSV.write refuses to render Python None (Julia ``nothing``).
        # Coerce any unset / None to a printable empty string. Ints / floats
        # pass through unchanged so the Julia stager's as_int / Float64 casts
        # still work.
        return default if v is None else v

    well_config_paths_by_geology = []
    for g in geology_entries:
        well_config_paths_by_geology.append({
            "geology_index": int(g["geology_index"]),
            "geology_name": _safe(g.get("geology_name"), Path(g["geology_h5_file"]).stem),
            "geology_file": str(Path(g["geology_h5_file"]).resolve()),
            "geology_config_id": g.get("geology_config_id") if g.get("geology_config_id") not in (None, "") else g.get("scenario"),
            "geology_scenario_name": _safe(g.get("geology_scenario_name"), ""),
            "geology_sample_num": _safe(g.get("geology_sample_num"), ""),
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
        # NaN, not 0.0 — the baseline has no surrogate prediction, and writing
        # a literal zero is a lie that could pollute downstream plots that
        # treat predicted_* as a real number. Julia parses NaN floats fine.
        "predicted_discounted_total_revenue": float("nan"),
        "predicted_emv": float("nan"),
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


def _resolve_depth_bounds(opt_cfg: dict, nz_min: int) -> tuple[int, int]:
    """Resolve (z_lo, z_hi) from config.

    Defaults: ``z_lo=5``, ``z_hi=min(70, nz_min-1)`` — biased toward the
    productive zone the prior depth campaign surfaced
    (see ``move_b_depth_results`` memory) while clipping to the shallowest
    geology grid.
    """
    z_lo = int(opt_cfg.get("depth_min", 5))
    default_hi = min(70, nz_min - 1)
    z_hi = min(int(opt_cfg.get("depth_max", default_hi)), nz_min - 1)
    if z_hi <= z_lo:
        raise ValueError(
            f"optimizer.depth_max ({z_hi}) must be > depth_min ({z_lo}) and ≤ nz_min-1 ({nz_min - 1})."
        )
    return (z_lo, z_hi)


def _build_or_load_optimizer(
    *, cfg: dict, ws: Path, geology_entries: list[dict],
    valid_xy: np.ndarray, nx: int, ny: int, nz_min: int,
    remote: RemoteSession, remote_run_root: str,
) -> BaselineOptimizer:
    """Pull optimizer_state.pkl from Sherlock if it exists, else construct fresh."""
    if _pull_optimizer_state(remote, remote_run_root, ws):
        print(f"[baseline-driver] resuming optimizer state from Sherlock "
              f"({_optimizer_state_local(ws)})")
        return load_optimizer(_optimizer_state_local(ws))

    opt_cfg = cfg["optimizer"]
    num_wells = len(cfg["wells"])
    depth_bounds = _resolve_depth_bounds(opt_cfg, nz_min)
    print(f"[baseline-driver] building optimizer "
          f"(type={opt_cfg.get('type', 'cmaes')}, popsize={opt_cfg['popsize']}, "
          f"num_wells={num_wells}, z ∈ [{depth_bounds[0]}, {depth_bounds[1]}], "
          f"nz_min={nz_min})")
    return build_optimizer(
        kind=str(opt_cfg.get("type", "cmaes")),
        num_wells=num_wells,
        nx=nx, ny=ny,
        edge_buffer=int(opt_cfg.get("edge_buffer", 10)),
        popsize=int(opt_cfg["popsize"]),
        seed=int(opt_cfg.get("seed", 42)),
        valid_xy_indices=valid_xy,
        depth_bounds=depth_bounds,
        sigma_init=float(opt_cfg.get("sigma_init", 5.0)),
    )


def _wells_is_injector(cfg: dict) -> list[bool]:
    return [str(w["type"]).lower() == "injector" for w in cfg["wells"]]


def _build_baseline_npv_state(cfg: dict, geology_entries: list[dict],
                              is_injector_list: list[bool], ws: Path | None = None) -> dict | None:
    """Local proxy-NPV state for the direct-IX-CMA ablation, or None in revenue mode.

    Gated on cfg["optimizer"].get("objective")=="npv" (mirrors the surrogate path's
    acquisition.objective). The driver runs locally with the cube + local geology H5s, so it
    wraps each candidate's real-IX per-geology revenue with the deviated-well CAPEX + surface
    flowline + surface-facility CAPEX + discounted OPEX from economics.json. Reused from
    geothermal.well_geometry so the math is identical to the surrogate path.
    """
    opt_cfg = cfg.get("optimizer", {})
    if str(opt_cfg.get("objective", "revenue")) != "npv":
        return None
    import h5py  # local-only import; baseline ingest stays revenue-based on Sherlock
    surrogate_repo = Path(
        cfg.get("compute", {}).get("local_surrogate_repo") or cfg["paths"]["surrogate_repo"]
    ).expanduser().resolve()
    if str(surrogate_repo) not in sys.path:
        sys.path.insert(0, str(surrogate_repo))
    from geothermal.well_geometry import (  # type: ignore
        load_geo_coord_cube, reservoir_top_k_map, facilities_surface_xy,
        surface_flowline_length, compute_angled_well_length, compute_npv, load_npv_terms,
    )
    geo_cube_path = opt_cfg.get("geo_cube_path")
    if not geo_cube_path or not Path(geo_cube_path).expanduser().exists():
        raise FileNotFoundError(
            f"objective='npv' requires optimizer.geo_cube_path (CX/CY/CZ h5); got {geo_cube_path}"
        )
    econ_path = opt_cfg.get("economics_config_path") or (surrogate_repo / "configs" / "economics.json")
    ksurf = int(opt_cfg.get("ksurf", 2))
    poro_thresh = float(opt_cfg.get("poro_thresh", 0.01))
    facilities = opt_cfg.get("facilities", [[20, 30], [40, 40]])
    cube = load_geo_coord_cube(Path(geo_cube_path).expanduser())
    terms = load_npv_terms(str(econ_path))
    fac_surf = facilities_surface_xy(cube, facilities, ksurf=ksurf)
    flowline = surface_flowline_length(fac_surf)
    rtop_by_geo: dict[int, np.ndarray] = {}
    for e in geology_entries:
        with h5py.File(e["geology_h5_file"], "r") as h:
            poro = h["Input/Porosity"][:]
        rtop_by_geo[int(e["geology_index"])] = reservoir_top_k_map(poro, poro_thresh=poro_thresh)
    print(f"[baseline-driver] objective=npv: flowline_between={flowline:.1f} m, "
          f"{len(rtop_by_geo)} geologies, surface_capex={terms.get('CAPEX_SURFACE_FACILITIES', 0.0):.3e}")
    if ws is not None:
        from orchestrator.npv_metrics import write_npv_context
        write_npv_context(
            ws, surrogate_repo=surrogate_repo, geo_cube_path=geo_cube_path, facilities=facilities,
            vertical_lead_m=float(opt_cfg.get("vertical_lead_m", 1000.0)), ksurf=ksurf,
            poro_thresh=poro_thresh,
            reservoir_geology_h5=geology_entries[0]["geology_h5_file"] if geology_entries else "",
        )
    return {
        "compute_angled_well_length": compute_angled_well_length,
        "compute_npv": compute_npv,
        "cube": cube, "terms": terms, "fac_surf": fac_surf, "flowline": flowline,
        "rtop_by_geo": rtop_by_geo, "vertical_lead_m": float(opt_cfg.get("vertical_lead_m", 1000.0)),
        "ksurf": ksurf, "is_injector_list": is_injector_list,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Baseline AL config (e.g. configs/baseline_global.json)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Resume an existing run. Omit to bootstrap.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    # Foot-gun guard: the baseline reads objective under cfg["optimizer"] (the surrogate
    # path reads it under cfg["acquisition"]). If a user put objective:"npv" under
    # "acquisition" here it would be silently ignored and run revenue — fail loudly instead.
    if str((cfg.get("acquisition") or {}).get("objective", "revenue")) == "npv":
        print("ERROR: baseline reads objective from cfg['optimizer'], not cfg['acquisition']. "
              "Move objective/economics_config_path/geo_cube_path/facilities/vertical_lead_m "
              "into the 'optimizer' block.", file=sys.stderr)
        return 1
    compute = cfg.get("compute") or {}
    for required in ("ssh_host", "ssh_control_path", "remote_repo_root",
                     "local_workspace", "local_geologies_config"):
        if not compute.get(required):
            print(f"ERROR: compute.{required} is required", file=sys.stderr)
            return 1

    ws = Path(compute["local_workspace"]).expanduser().resolve()
    _ensure_workspace(ws)
    # Pin the objective to this workspace so a resume with a flipped config fails loud.
    from orchestrator.npv_metrics import assert_objective_marker
    assert_objective_marker(ws, str((cfg.get("optimizer") or {}).get("objective", "revenue")))

    # Email notification target (shared by completion / failure / auth notices).
    notify_email = compute.get("notify_email")
    notify_msmtp_account = compute.get("notify_msmtp_account", "gmail")

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
    geology_entries, valid_xy, nx, ny, nz_min = _load_geologies_resolved(cfg)
    is_injector_list = _wells_is_injector(cfg)
    # Proxy-NPV ablation state (None unless optimizer.objective == "npv").
    npv_state = _build_baseline_npv_state(cfg, geology_entries, is_injector_list, ws=ws)

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
            valid_xy=valid_xy, nx=nx, ny=ny, nz_min=nz_min,
            remote=remote, remote_run_root=remote_run_root,
        )

        # Invariant: optimizer.generation must equal state.iteration on
        # entering the loop. Any mismatch means a previous run crashed
        # between save_state and the next iter's ingest finish (or the
        # other way), and silently continuing would corrupt either the
        # CMA covariance (re-feeding the same fitness) or the IX run
        # sequence (skipping a generation). Refuse and let the user
        # inspect the run dir manually.
        _initial_state = _pull_state(remote, remote_run_root, ws)
        if optimizer.generation != _initial_state.iteration:
            raise RuntimeError(
                f"State/optimizer mismatch on resume: "
                f"state.iteration={_initial_state.iteration}, "
                f"optimizer.generation={optimizer.generation}. "
                f"A previous run likely crashed mid-handoff. Inspect "
                f"{remote_run_root}/state.json and {remote_run_root}/optimizer_state.pkl "
                f"on Sherlock and decide which to align to the other before resuming."
            )

        while True:
            if _check_done(remote, remote_run_root):
                print(f"[baseline-driver] done.json present — run complete.")
                (ws / "NEEDS_AUTH.md").unlink(missing_ok=True)
                if wandb_handle is not None:
                    wandb_handle.finish()
                _notify_run_event(
                    event="completed", ws=ws, driver_label="baseline",
                    run_id=run_id, iteration=iteration, last_step=last_step,
                    notify_email=notify_email, notify_msmtp_account=notify_msmtp_account,
                    detail="done.json present on Sherlock — the run finished cleanly.",
                )
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
                _notify_run_event(
                    event="completed", ws=ws, driver_label="baseline",
                    run_id=run_id, iteration=iteration, last_step=last_step,
                    notify_email=notify_email, notify_msmtp_account=notify_msmtp_account,
                    detail=f"Reached max_iterations={max_iters} — wrote done.json.",
                )
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

            # NB: ``run_token`` must vary across population members within a
            # single (iteration, scenario). The Julia stager builds its IX
            # output filename as ``v2.5_{prefix}_{scenario}_run{run_id:04d}_iter{iter:04d}.h5``
            # — snapshot_id is not part of it — so two snapshots sharing
            # ``(run_id, iteration, scenario)`` collide and the stager aborts.
            # AL ensemble mode sidesteps this by having one snapshot per kind
            # per iter; we instead encode the pop index into run_token.
            snapshot_records: list[dict] = []
            for p in range(popsize):
                rec = _emit_baseline_snapshot(
                    run_token=p,
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
                _notify_run_event(
                    event="failed", ws=ws, driver_label="baseline",
                    run_id=run_id, iteration=iteration, last_step=last_step,
                    notify_email=notify_email, notify_msmtp_account=notify_msmtp_account,
                    detail=(f"Ingest finished non-COMPLETED ({final}). Logs: "
                            f"{remote_run_root}/logs/ingest_baseline_iter_{iteration:04d}.err "
                            f"on Sherlock."),
                )
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
                if npv_state is not None and p_idx < coords.shape[0]:
                    # Wrap real-IX per-geology revenue with the proxy-NPV cost terms.
                    # NPV_k = revenue_k - CAPEX(coords, geo_k) - flowline - surface - discounted OPEX;
                    # fitness = ensemble-mean NPV (higher = better, same sign as revenue).
                    per_geo = c.get("per_geology_revenue", {}) or {}
                    npvs: list[float] = []
                    for gk, gr in per_geo.items():
                        rtop = npv_state["rtop_by_geo"].get(int(gk))
                        if rtop is None or gr is None or not np.isfinite(gr):
                            continue
                        wl = npv_state["compute_angled_well_length"](
                            coords[p_idx], cube=npv_state["cube"], fac_surf_xy=npv_state["fac_surf"],
                            reservoir_top_k_map=rtop, vertical_lead_m=npv_state["vertical_lead_m"],
                            ksurf=npv_state["ksurf"],
                        )
                        npvs.append(npv_state["compute_npv"](
                            float(gr), wl, npv_state["is_injector_list"],
                            flowline_between_m=npv_state["flowline"], npv_terms=npv_state["terms"],
                        )["npv"])
                    v = float(np.mean(npvs)) if npvs else float("nan")
                    c["ensemble_mean_npv"] = v  # recorded in the re-saved local fitness copy
                else:
                    v = c.get("ensemble_mean_revenue", float("nan"))
                cands_by_pop[p_idx] = float(v) if v is not None else float("nan")
            if npv_state is not None:
                # Persist the NPV-augmented fitness locally (remote ingest stays revenue-based).
                with open(local_fitness, "w") as f:
                    json.dump(fitness_payload, f, indent=2)
            fitnesses = np.array(
                [cands_by_pop.get(p, float("nan")) for p in range(popsize)],
                dtype=np.float64,
            )
            n_finite = int(np.isfinite(fitnesses).sum())
            fit_label = "NPV" if npv_state is not None else "revenue"
            best_npv_in_batch = float(np.nanmax(fitnesses)) if (npv_state is not None and n_finite) else None
            print(f"[baseline-driver] tell(): {n_finite}/{popsize} finite {fit_label} fitnesses, "
                  f"max={np.nanmax(fitnesses) if n_finite else 'nan'}")

            # I4: all-NaN batch guard. If every candidate's IX evaluation
            # failed, ``fitnesses`` is all NaN and a tell() would corrupt
            # the CMA-ES covariance/sigma (every entry rank-tied at
            # WORST_SENTINEL). This is virtually always an infrastructure
            # problem on Sherlock (full SLURM array failure, IX licensing,
            # corrupt manifest) rather than a fitness landscape feature, so
            # we bail loudly rather than silently mis-train. State on
            # Sherlock has already advanced (ingest's iteration++ ran), so
            # the user investigates and resumes with --run-id <id>; the
            # startup invariant will then catch the state vs optimizer
            # mismatch and report it explicitly.
            if n_finite == 0:
                if wandb_handle is not None and wandb_handle.is_active:
                    wandb_handle.log({
                        "iteration": iteration,
                        "all_nan_batch": True,
                        "n_failed_tasks": stage.tasks_count,
                    }, step=iteration)
                    wandb_handle.finish()
                raise RuntimeError(
                    f"All {popsize} candidates returned non-finite ensemble revenue "
                    f"for iter {iteration} — refusing to call optimizer.tell() with "
                    f"all-WORST_SENTINEL fitness (that would corrupt CMA-ES sigma).\n"
                    f"  Inspect: {remote_run_root}/iter_{iteration:04d}/fitness.json on Sherlock\n"
                    f"  SLURM logs: {remote_run_root}/logs/ingest_baseline_iter_{iteration:04d}.err\n"
                    f"  IX array logs under the per-iter stage dir.\n"
                    f"Resume with: --run-id {run_id} after fixing the underlying failure."
                )

            optimizer.tell(coords, fitnesses)
            optimizer.save_state(_optimizer_state_local(ws))
            _push_optimizer_state(remote, remote_run_root, ws)
            last_step = f"tell(): generation now {optimizer.generation}"
            # NB: there is a short window between save_state above and the
            # next iteration's pull where a crash could leave
            # optimizer.generation == iteration+1 on bend but
            # state.iteration == iteration+1 on Sherlock (ingest advanced it).
            # The startup invariant check below catches mismatches on resume.

            # wandb log
            if wandb_handle is not None and wandb_handle.is_active:
                log_payload = {
                    "iteration": iteration,
                    "generation": optimizer.generation,
                    "popsize": popsize,
                    "n_finite_in_batch": n_finite,
                    "objective": ("npv" if npv_state is not None else "revenue"),
                    # NB: best_ensemble_revenue_* always track REVENUE (from the remote
                    # ingest). In npv mode the optimizer climbs NPV, so the NPV series
                    # below is the one that matches what tell() optimized.
                    "best_ensemble_revenue_in_batch": fitness_payload.get("best_ensemble_revenue_in_batch"),
                    "best_ensemble_revenue_so_far": fitness_payload.get("best_ensemble_revenue_so_far"),
                    "n_submitted_tasks": stage.tasks_count,
                    "n_completed_tasks": fitness_payload.get("n_completed_tasks"),
                    "wallclock_ask_min": ask_min,
                }
                if npv_state is not None:
                    # Match the hybrid/dashboard key names so npv runs overlay on one wandb panel.
                    log_payload["best_real_npv_in_batch"] = best_npv_in_batch
                    bsf = getattr(optimizer, "best_fitness_so_far", None)
                    log_payload["best_real_npv_so_far"] = float(bsf if bsf is not None else best_npv_in_batch)
                wandb_handle.log(log_payload, step=iteration)

            # ---- 8. Refresh state from Sherlock for the next iter loop ----
            # The Sherlock-side ingest already advanced state.iteration; we
            # just pull so the next ``while True`` head sees the new value.
            _pull_state(remote, remote_run_root, ws)
            last_step = f"refreshed state after ingest of iter {iteration}"

            # Pull per_candidate_metrics.json + render the diagnostic plot
            # dashboard. Surrogate-dependent panels (MAPE / pred-vs-real
            # scatter / calibration / holdout) silently skip on the baseline
            # because predicted_revenue is NaN. Revenue / well-position /
            # wallclock / EMV plots render off the per-(snapshot, geology)
            # rows and the iter history.
            try:
                if _pull_per_candidate_metrics(remote, remote_run_root, iteration, ws):
                    # In npv mode, augment the pulled metrics with locally-computed real NPV
                    # (remote ingest is revenue-only) so the dashboard shows the NPV objective.
                    if npv_state is not None:
                        from orchestrator.npv_metrics import augment_metrics_file
                        for it_done in range(iteration + 1):
                            augment_metrics_file(
                                ws / f"iter_{it_done:04d}" / "per_candidate_metrics.json",
                                ws / "acquire" / f"iter_{it_done:04d}" / "snapshots_json",
                                npv_state,
                            )
                    _render_plots(ws)
            except SshUnavailable:
                raise
            except Exception as e:
                print(f"[baseline-driver] post-ingest plot refresh failed: {e}",
                      file=sys.stderr)

    except SshUnavailable as e:
        _write_needs_auth(
            ws,
            config_path=args.config.resolve(),
            run_id=run_id,
            iteration=iteration,
            last_step=last_step,
            error=e,
            ssh_host=compute["ssh_host"],
            notify_email=notify_email,
            notify_msmtp_account=notify_msmtp_account,
        )
        return 2
    except Exception as e:
        # Any other failure during the unattended loop (incl. the all-NaN-batch
        # and state/optimizer-mismatch RuntimeErrors): email, then re-raise so
        # the traceback still surfaces and the exit code stays non-zero.
        # KeyboardInterrupt / SystemExit propagate untouched.
        try:
            _notify_run_event(
                event="failed", ws=ws, driver_label="baseline",
                run_id=run_id, iteration=iteration, last_step=last_step,
                notify_email=notify_email, notify_msmtp_account=notify_msmtp_account,
                detail=f"Driver crashed with {type(e).__name__}: {e}",
            )
        except Exception as notify_err:
            print(f"[baseline-driver] failure-notify itself failed: {notify_err}",
                  file=sys.stderr)
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
