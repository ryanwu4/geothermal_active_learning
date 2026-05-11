#!/usr/bin/env python3
"""Local hybrid AL driver — runs train+acquire+select on bend, IX+ingest on Sherlock.

Usage (with a ControlMaster socket already authenticated in another tmux pane):
    python scripts/run_al_local.py --config configs/al_hybrid.json
    # resume an existing run:
    python scripts/run_al_local.py --config configs/al_hybrid.json --run-id <id>

Each iteration:
  1. Pull state.json from Sherlock (canonical source of truth).
  2. Pull the prior-iteration compiled H5, overwriting one local file.
  3. Train + acquire + select on bend's GPU.
  4. Push selection manifest to Sherlock.
  5. ssh-run Julia stage script, submit IX array + ingest sbatches on Sherlock
     (ingest depends afterany on the IX array).
  6. Poll Sherlock for the ingest job's terminal state.
  7. Loop. Stop when the remote run writes ``done.json``.

If ssh ever fails (socket dead, Duo expired), write ``NEEDS_AUTH.md`` in the
local workspace and exit 2 — the user re-establishes the socket and re-runs.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import socket
import subprocess
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
from orchestrator.remote import RemoteSession, SshUnavailable, TERMINAL_JOB_STATES
from orchestrator.retrain import run_train, should_train_from_scratch
from orchestrator.select import select_batch
from orchestrator.slurm import render_template
from orchestrator.stage import stage_iteration
from orchestrator.state import IterationRecord, RunState, new_run_id


# Allow only the characters used by ``new_run_id`` (timestamp + hex suffix +
# user-supplied prefix). Reject path traversal and shell/email metachars.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"invalid run_id {run_id!r} — must match {_RUN_ID_RE.pattern}"
        )


def _sanitize_header_value(s: str) -> str:
    """Strip CR/LF so an attacker can't inject extra mail headers via run_id/email."""
    return s.replace("\r", " ").replace("\n", " ").strip()


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
# Workspace layout
# ----------------------------------------------------------------------


def _ensure_workspace(ws: Path) -> None:
    for sub in ("models", "manifests", "logs", "acquire", "rendered_sbatch"):
        (ws / sub).mkdir(parents=True, exist_ok=True)


def _local_compiled_h5(ws: Path) -> Path:
    return ws / "current_compiled.h5"


def _local_norm_config(ws: Path) -> Path:
    return ws / "norm_config.json"


def _local_extra_path(ws: Path) -> Path:
    return ws / "local_state_extra.json"


def _read_local_extra(ws: Path) -> dict:
    p = _local_extra_path(ws)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _write_local_extra(ws: Path, extra: dict) -> None:
    _local_extra_path(ws).write_text(json.dumps(extra, indent=2))


def _state_mirror_path(ws: Path) -> Path:
    return ws / "state_mirror.json"


# ----------------------------------------------------------------------
# Auth-failure notice
# ----------------------------------------------------------------------


def _send_email_notice(
    *,
    to_addr: str,
    msmtp_account: str,
    subject: str,
    body: str,
) -> None:
    """Pipe a plain-text email through msmtp. Best-effort — never raises."""
    safe_to = _sanitize_header_value(to_addr)
    safe_subject = _sanitize_header_value(subject)
    msg = f"To: {safe_to}\nSubject: {safe_subject}\n\n{body}\n"
    try:
        proc = subprocess.run(
            ["msmtp", "-a", msmtp_account, safe_to],
            input=msg, text=True, capture_output=True, check=False, timeout=30,
        )
        if proc.returncode != 0:
            print(
                f"[local-driver] msmtp failed (rc={proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()}",
                file=sys.stderr,
            )
        else:
            print(f"[local-driver] sent auth-failure notice to {to_addr}", file=sys.stderr)
    except Exception as e:
        print(f"[local-driver] could not send auth-failure email: {e}", file=sys.stderr)


def _write_needs_auth(
    ws: Path,
    *,
    config_path: Path,
    run_id: str | None,
    iteration: int | None,
    last_step: str,
    error: BaseException,
    ssh_host: str,
    control_path: str,
    notify_email: str | None = None,
    notify_msmtp_account: str = "gmail",
) -> None:
    # With ControlMaster=auto + ControlPath set in ~/.ssh/config, a plain
    # ``ssh <host>`` both creates the master socket and authenticates (Duo).
    # ControlPersist keeps the master alive after the interactive shell exits.
    cmd_open = f"ssh {ssh_host}   # accept Duo; ControlPersist keeps the socket"
    cmd_check = f"ssh -O check {ssh_host}"
    resume_args = f"--config {shlex.quote(str(config_path))}"
    if run_id:
        resume_args += f" --run-id {shlex.quote(run_id)}"
    body = f"""# AL run paused — Sherlock SSH unavailable

The local driver could not reach Sherlock through the ControlMaster socket. No
state was lost on either side; ingest jobs already submitted on Sherlock will
finish on their own. To resume:

- Run id: `{run_id or "(not bootstrapped yet)"}`
- Iteration in progress: `{iteration if iteration is not None else "(unknown)"}`
- Last successful step: `{last_step}`
- Underlying error:

      {error}

## Resume

1. In your dedicated tmux pane, re-establish the ControlMaster socket and
   re-authenticate (Duo) once:

       {cmd_open}

2. Verify it's live:

       {cmd_check}

   (should print `Master running`)

3. Re-launch the driver with the same run id:

       python scripts/run_al_local.py {resume_args}

The driver re-reads state.json from Sherlock and continues from where it left
off.
"""
    (ws / "NEEDS_AUTH.md").write_text(body)
    print(f"[local-driver] wrote {ws / 'NEEDS_AUTH.md'} — exiting.", file=sys.stderr)

    if notify_email:
        host = socket.gethostname()
        subject = (
            f"[AL hybrid] SSH to Sherlock unavailable on {host} "
            f"(run={run_id or 'unbootstrapped'}, iter={iteration})"
        )
        _send_email_notice(
            to_addr=notify_email,
            msmtp_account=notify_msmtp_account,
            subject=subject,
            body=body,
        )


# ----------------------------------------------------------------------
# Bootstrap a new remote run
# ----------------------------------------------------------------------


def _bootstrap_remote_run(
    remote: RemoteSession,
    cfg: dict,
    local_config_path: Path,
    explicit_run_id: str | None,
) -> str:
    """Create the run dir + state.json on Sherlock without submitting anything.

    Generates a run_id locally so the local driver knows it, pushes the local
    AL config to ``<scratch_root>/al_config_staging/<run_id>.json``, then
    ssh-runs ``scripts/start_al_run.py --dry-run`` on Sherlock with that config.
    """
    run_id = explicit_run_id or new_run_id(prefix=cfg.get("run_id_prefix", "al"))
    scratch_root = cfg["paths"]["scratch_root"]
    remote_repo_root = cfg["compute"]["remote_repo_root"]
    staging_dir = f"{scratch_root}/al_config_staging"
    remote_config_path = f"{staging_dir}/{run_id}.json"

    remote.run(["mkdir", "-p", staging_dir])
    remote.push(local_config_path, remote_config_path)

    print(f"[local-driver] bootstrapping remote run {run_id} (config={remote_config_path})")
    proc = remote.run(
        ["bash", "-lc", _wrap_with_sherlock_env(
            f"python scripts/start_al_run.py "
            f"--config {shlex.quote(remote_config_path)} "
            f"--run-id {shlex.quote(run_id)} "
            f"--dry-run"
        )],
        cwd=remote_repo_root,
        check=True,
    )
    print(proc.stdout.strip())
    return run_id


# Sherlock's login-node default ``python`` is Python 2 and ``julia`` isn't on
# PATH at all. Every interpreter invocation we ssh-run has to load the same
# modules + venv that the sbatch templates do — mirrors
# sbatch/train_acquire.sbatch.template:14-18. Wrapped in ``bash -lc`` so ``ml``
# is in scope.
_SHERLOCK_ENV_PREAMBLE = (
    "ml python/3.12.1 uv hdf5/1.14.4 openblas/0.3.20 && "
    "module use /home/groups/sh_s-dss/share/sdss/modules/modulefiles/ && "
    "module load julia && "
    "export UV_PYTHON=$(which python3) && "
    "source ~/geothermal-pomdp/bin/activate"
)


def _wrap_with_sherlock_env(cmd: str) -> str:
    return f"{_SHERLOCK_ENV_PREAMBLE} && {cmd}"


def _make_sherlock_runner(remote: RemoteSession, *, cwd: str | None = None):
    """Build a stage_iteration runner that wraps argv in the Sherlock env."""
    def _runner(argv: list[str]):
        cmd = " ".join(shlex.quote(a) for a in argv)
        return remote.run(
            ["bash", "-lc", _wrap_with_sherlock_env(cmd)],
            cwd=cwd, check=False,
        )
    return _runner


def _rewrite_manifest_paths(local_manifest: Path, remappings: list[tuple[str, str]]) -> dict:
    """Load the manifest, apply each ``(local_prefix, remote_prefix)`` remap to
    every string value, and return the rewritten payload.

    Used so the Julia stage script (running on Sherlock) can resolve snapshot
    JSON / well_config / geology h5 files written or referenced by acquire on
    bend. ``remappings`` is order-sensitive: longer/more-specific prefixes
    should come first so they take precedence over generic ones.
    """
    payload = json.loads(local_manifest.read_text())

    def _rewrite_str(s: str) -> str:
        for local_prefix, remote_prefix in remappings:
            if s.startswith(local_prefix):
                return remote_prefix + s[len(local_prefix):]
        return s

    def _walk(o):
        if isinstance(o, dict):
            return {k: _walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_walk(v) for v in o]
        if isinstance(o, str):
            return _rewrite_str(o)
        return o

    return _walk(payload)


def _render_local_plots(state: RunState, ws: Path) -> None:
    """Render the state-derived diagnostic plots into ``<ws>/plots/``.

    Uses the same plotting functions as ``scripts/plot_convergence.py`` but
    only the subset that derives from ``state.history`` — we don't have the
    Sherlock-side per-iteration ``per_candidate_metrics.json`` files locally,
    so the scatter / distribution plots are skipped here.
    """
    try:
        # Defer import: matplotlib pulls in a lot, no need at module load.
        from dataclasses import asdict
        import matplotlib
        matplotlib.use("Agg")  # headless on bend
        from scripts import plot_convergence as pc  # type: ignore
    except Exception as e:
        print(f"[local-driver] plotting deferred — could not import plot_convergence: {e}",
              file=sys.stderr)
        return

    out_dir = ws / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = [asdict(r) for r in state.history]
    if not history:
        return
    pc._set_style()
    try:
        pc.plot_best_revenue(history, out_dir)
        pc.plot_calibration_metrics(history, out_dir)
        pc.plot_per_geology_mape_heatmap(history, out_dir)
        pc.plot_training_growth(history, out_dir)
        pc.plot_wallclock_breakdown(history, out_dir)
    except Exception as e:
        # Plotting is diagnostic — never let a plot bug kill the AL loop.
        print(f"[local-driver] plot rendering failed: {e}", file=sys.stderr)


def _build_geology_remappings(cfg: dict) -> list[tuple[str, str]]:
    """Build (local_geology_path → remote_geology_path) remappings by matching
    filenames between the bend-side and Sherlock-side geology configs.

    Both configs live in the repo at the same relative path on both sides, so
    we read them locally — no ssh round-trip needed.
    """
    local_cfg = Path(cfg["compute"]["local_geologies_config"]).expanduser().resolve()
    remote_cfg_rel = cfg["paths"]["geologies_config"]
    # `paths.geologies_config` is a relative path resolved from the repo root
    # on whichever side is reading it. We have the same repo locally on bend.
    if Path(remote_cfg_rel).is_absolute():
        remote_cfg = Path(remote_cfg_rel)
    else:
        remote_cfg = (REPO_ROOT / remote_cfg_rel).resolve()
    if not remote_cfg.exists():
        raise RuntimeError(
            f"Cannot read Sherlock-side geology config locally: {remote_cfg} "
            f"does not exist on bend. Add it to the repo or fix paths.geologies_config."
        )

    local_entries = _load_geology_list(local_cfg)
    remote_entries = _load_geology_list(remote_cfg)
    local_by_fn = {e["filename"]: e["geology_h5_file"] for e in local_entries}
    remote_by_fn = {e["filename"]: e["geology_h5_file"] for e in remote_entries}
    pairs = []
    for fn, lp in local_by_fn.items():
        rp = remote_by_fn.get(fn)
        if rp and lp != rp:
            pairs.append((lp, rp))
    return pairs


# ----------------------------------------------------------------------
# State synchronization
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
    proc = remote.run(["test", "-f", f"{remote_run_root}/done.json"], check=False)
    return proc.returncode == 0


# ----------------------------------------------------------------------
# Per-iteration data fetch
# ----------------------------------------------------------------------


def _ensure_norm_config(remote: RemoteSession, state: RunState, ws: Path) -> Path:
    """Pull norm_config.json once. Returns the local path."""
    local = _local_norm_config(ws)
    if local.exists():
        return local
    if not state.norm_config_path:
        raise RuntimeError(
            "state.norm_config_path is unset on Sherlock — bootstrap may have failed."
        )
    remote.pull(state.norm_config_path, local)
    return local


def _ensure_bootstrap_artifacts(
    remote: RemoteSession, cfg: dict, ws: Path
) -> tuple[Path, Path]:
    """For iter 0: pull bootstrap checkpoint + scaler from Sherlock."""
    boot_dir_remote = cfg["paths"]["bootstrap_checkpoint_dir"]
    boot_dir_local = ws / "models" / "bootstrap"
    boot_dir_local.mkdir(parents=True, exist_ok=True)
    # Sync the whole checkpoint dir (small — a few MB).
    remote.pull(boot_dir_remote.rstrip("/") + "/", boot_dir_local)
    best = sorted(boot_dir_local.glob("best-*.ckpt"))
    if not best:
        raise RuntimeError(f"No best-*.ckpt found locally after pulling {boot_dir_remote}")
    ckpt = best[0]
    scaler = ckpt.parent / "scaler.pkl"
    if not scaler.exists():
        raise RuntimeError(f"Expected scaler.pkl alongside {ckpt}")
    return ckpt, scaler


def _pull_compiled_h5(remote: RemoteSession, remote_run_root: str, prior_iter: int, dest: Path) -> None:
    """Pull ``compiled/compiled_iter_NNNN.h5`` from Sherlock, overwriting ``dest``.

    This is the only large transfer per iter (~11–15 GB). rsync ``--inplace``
    is set in RemoteSession.pull so a partial transfer can resume in place.
    """
    remote_path = f"{remote_run_root}/compiled/compiled_iter_{prior_iter:04d}.h5"
    print(f"[local-driver] pulling {remote_path} → {dest}")
    remote.pull(remote_path, dest)


# ----------------------------------------------------------------------
# Sbatch rendering for ingest (Sherlock-side)
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
    ws: Path,
) -> str:
    """Render the ingest sbatch from the canonical template and push to Sherlock.

    Returns the remote path the sbatch was placed at.
    """
    template = REPO_ROOT / "sbatch" / "ingest.sbatch.template"
    rendered = render_template(template, {
        "RUN_ROOT": remote_run_root,
        "REPO_ROOT": remote_repo_root,
        "ITER": str(iteration),
        "ARRAY_TASKS_JSON": array_tasks_json_remote,
        "IX_OUTPUT_DIR": ix_output_dir_remote,
        "LOG_OUT": f"{remote_run_root}/logs/ingest_iter_{iteration:04d}.out",
        "LOG_ERR": f"{remote_run_root}/logs/ingest_iter_{iteration:04d}.err",
        "JOB_NAME": f"AL_INGEST_{run_id}_{iteration:04d}",
    })
    local_sbatch = ws / "rendered_sbatch" / f"ingest_iter_{iteration:04d}.sbatch"
    local_sbatch.write_text(rendered)
    local_sbatch.chmod(0o755)
    remote_path = f"{remote_run_root}/sbatch_rendered/ingest_iter_{iteration:04d}.sbatch"
    remote.push(local_sbatch, remote_path)
    return remote_path


# ----------------------------------------------------------------------
# Polling
# ----------------------------------------------------------------------


_MAX_CONSECUTIVE_UNKNOWN = 5


def _poll_until_terminal(
    remote: RemoteSession, job_id: str, *, every: int, label: str = "ingest"
) -> str:
    """Poll ``sacct`` until job is in a terminal state. Returns final state.

    Aborts (via RuntimeError) after ``_MAX_CONSECUTIVE_UNKNOWN`` consecutive
    ``UNKNOWN`` lookups so a purged or never-scheduled job doesn't leave the
    driver spinning indefinitely.
    """
    every = max(int(every), 1)
    print(f"[local-driver] polling {label} job {job_id} every {every}s")
    consecutive_unknown = 0
    while True:
        state = remote.job_state(job_id)
        if state in TERMINAL_JOB_STATES:
            print(f"[local-driver] {label} job {job_id} → {state}")
            return state
        if state == "UNKNOWN":
            consecutive_unknown += 1
            if consecutive_unknown >= _MAX_CONSECUTIVE_UNKNOWN:
                raise RuntimeError(
                    f"{label} job {job_id} returned UNKNOWN "
                    f"{consecutive_unknown} times in a row — neither sacct nor "
                    f"squeue can find it. Check Sherlock manually."
                )
        else:
            consecutive_unknown = 0
        print(f"[local-driver] {label} job {job_id} state={state}; sleeping {every}s")
        time.sleep(every)


# ----------------------------------------------------------------------
# Iteration body
# ----------------------------------------------------------------------


def _train_locally(
    *,
    state: RunState,
    cfg: dict,
    ws: Path,
    iter_idx: int,
    local_extra: dict,
) -> tuple[Path, Path, dict]:
    """Run train.py on bend's GPU. Returns (ckpt, scaler, metrics_dict)."""
    train_cfg = cfg["training"]
    surrogate_repo = Path(cfg["compute"]["local_surrogate_repo"]).resolve()
    h5_path = _local_compiled_h5(ws)
    if not h5_path.exists():
        raise RuntimeError(
            f"compiled H5 missing at {h5_path}; pull from Sherlock before training."
        )

    from_scratch = should_train_from_scratch(
        iter_idx, int(train_cfg.get("from_scratch_every_k", 5))
    )
    warm_ckpt = None
    warm_scaler = None
    if not from_scratch:
        prev_ckpt = local_extra.get("current_checkpoint")
        prev_scaler = local_extra.get("current_scaler")
        if prev_ckpt and Path(prev_ckpt).exists():
            warm_ckpt = Path(prev_ckpt)
            warm_scaler = Path(prev_scaler) if prev_scaler and Path(prev_scaler).exists() else None

    out_root = ws / "models" / f"iter_{iter_idx:04d}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = ws / "logs" / f"train_iter_{iter_idx:04d}.log"
    started = time.time()
    result = run_train(
        surrogate_repo=surrogate_repo,
        h5_path=h5_path,
        output_root=out_root,
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
        max_epochs_finetune=(
            int(train_cfg["max_epochs_finetune"]) if not from_scratch else None
        ),
        log_path=log_path,
    )
    elapsed_min = (time.time() - started) / 60.0
    return result.checkpoint_path, result.scaler_path, {
        "wallclock_train_min": elapsed_min,
        "train_val_loss": result.final_val_loss,
        "was_from_scratch": from_scratch,
    }


def _acquire_and_select_locally(
    *,
    cfg: dict,
    state: RunState,
    ws: Path,
    iter_idx: int,
    ckpt: Path,
    scaler: Path,
) -> tuple[Path, int, float]:
    """Run acquisition, select batch, write manifest. Returns (manifest_path, n_selected, elapsed_min)."""
    acq_cfg = cfg["acquisition"]
    sel_cfg = cfg["selection"]
    surrogate_repo = Path(cfg["compute"]["local_surrogate_repo"]).resolve()
    geology_path = Path(cfg["compute"]["local_geologies_config"]).expanduser().resolve()
    geology_entries = _load_geology_list(geology_path)
    geologies = [
        GeologySpec(
            geology_index=int(e["geology_index"]),
            geology_h5_file=str(Path(e["geology_h5_file"]).resolve()),
            geology_name=e.get("geology_name"),
        )
        for e in geology_entries
    ]
    missing = [g.geology_h5_file for g in geologies if not Path(g.geology_h5_file).exists()]
    if missing:
        raise RuntimeError(
            f"Local geology H5 files missing ({len(missing)}). First few: {missing[:3]}. "
            f"Either rsync them from Sherlock or fix paths in {geology_path}."
        )
    wells = [WellSpec(type=w["type"], depth=int(w["depth"])) for w in cfg["wells"]]
    norm_config = _local_norm_config(ws)

    acq = AcquisitionConfig(
        surrogate_repo=surrogate_repo,
        checkpoint_path=ckpt,
        scaler_path=scaler,
        norm_config_path=norm_config,
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
        device="cuda:0",
    )
    started = time.time()
    out_dir = ws / "acquire" / f"iter_{iter_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    acq_result = run_acquisition(acq, out_dir=out_dir, iteration=iter_idx)
    elapsed_min = (time.time() - started) / 60.0
    print(f"[local-driver] acquired {len(acq_result['candidates'])} candidates in {elapsed_min:.2f} min")

    selected = select_batch(
        acq_result["candidates"],
        batch_size=int(sel_cfg["batch_size"]),
        frontier_fraction=float(sel_cfg.get("frontier_fraction", 0.85)),
    )
    manifest_path = ws / "manifests" / f"manifest_iter_{iter_idx:04d}.json"
    write_selected_manifest(
        selected,
        out_path=manifest_path,
        iteration=iter_idx,
        geologies=geologies,
        extras={"selection": sel_cfg},
    )
    return manifest_path, len(selected), elapsed_min


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="AL config (hybrid mode)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Resume an existing run. Omit to bootstrap a fresh one on Sherlock.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    compute = cfg.get("compute") or {}
    if compute.get("train_acquire_location") != "local":
        print("ERROR: compute.train_acquire_location must be 'local' for this driver",
              file=sys.stderr)
        return 1
    for required in ("ssh_host", "ssh_control_path", "remote_repo_root",
                     "local_workspace", "local_surrogate_repo", "local_geologies_config"):
        if not compute.get(required):
            print(f"ERROR: compute.{required} is required", file=sys.stderr)
            return 1

    ws = Path(compute["local_workspace"]).expanduser().resolve()
    _ensure_workspace(ws)

    # Prevent two drivers from clobbering each other on the same workspace.
    # The lock auto-releases when ``lock_fd`` is GC'd / process exits.
    lock_path = ws / ".driver.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"ERROR: another driver already holds {lock_path}. "
            f"If you're sure none is running: rm {lock_path}",
            file=sys.stderr,
        )
        return 1
    os.write(lock_fd, f"{os.getpid()}\n".encode())

    remote = RemoteSession(host=compute["ssh_host"], control_path=compute["ssh_control_path"])
    poll_every = int(compute.get("poll_interval_sec", 600))

    # Bend has outbound internet, so log wandb online here. (Sherlock-side
    # train_acquire/ingest sbatches still set WANDB_MODE=offline because the
    # GPU partition may not have egress; those runs sync up via
    # scripts/wandb_sync.sh after the fact. Both contexts use the same
    # state.wandb_run_id so the dashboards merge cleanly.)
    #
    # Note on WANDB_DIR semantics: wandb auto-creates a ``wandb/`` subdir
    # under whatever path ``WANDB_DIR`` points at. Setting it to the
    # workspace root makes runs land at ``<ws>/wandb/run-<ts>-<id>/``.
    os.environ["WANDB_MODE"] = compute.get("wandb_mode", "online")
    os.environ["WANDB_DIR"] = str(ws)
    (ws / "wandb").mkdir(parents=True, exist_ok=True)
    wandb_handle = None  # initialized once we know run_id + state.wandb_run_id

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
            # ``--run-id`` was supplied. If the corresponding state.json is
            # missing on Sherlock (e.g. a prior bootstrap died before writing
            # it), run bootstrap with this explicit id instead of failing on
            # the upcoming pull.
            probe = remote.run(
                ["test", "-f", f"{scratch_root}/{run_id}/state.json"], check=False,
            )
            if probe.returncode != 0:
                print(
                    f"[local-driver] --run-id {run_id} given but no "
                    f"{scratch_root}/{run_id}/state.json on Sherlock — "
                    f"running bootstrap with this id."
                )
                run_id = _bootstrap_remote_run(
                    remote, cfg, args.config, explicit_run_id=run_id,
                )
                last_step = f"bootstrapped run_id={run_id} (resumed-id)"
        print(f"[local-driver] driving run_id={run_id}")
        remote_run_root = f"{cfg['paths']['scratch_root']}/{run_id}"
        remote_repo_root = compute["remote_repo_root"]

        # Read julia_config from Sherlock once to discover ix_output_dir.
        julia_config_payload = _read_julia_config(remote, cfg)
        ix_output_root = (
            julia_config_payload.get("FILEPATHS", {}).get("h5s_dir_out")
        )
        if not ix_output_root:
            raise RuntimeError(
                f"julia config missing FILEPATHS.h5s_dir_out — cannot locate IX outputs"
            )
        last_step = "loaded julia_config"

        while True:
            if _check_done(remote, remote_run_root):
                print(f"[local-driver] Sherlock has done.json — run complete.")
                (ws / "NEEDS_AUTH.md").unlink(missing_ok=True)
                if wandb_handle is not None:
                    wandb_handle.finish()
                return 0

            state = _pull_state(remote, remote_run_root, ws)
            iteration = state.iteration
            last_step = f"pulled state iter={iteration}"
            print(f"[local-driver] iter {iteration} starting")

            # Refresh the diagnostic plots up-front so the user sees a current
            # picture of the previous iterations even while the new one is
            # still training/staging.
            _render_local_plots(state, ws)

            if wandb_handle is None:
                print(f"[wandb] initializing run_id={state.wandb_run_id} "
                      f"project={cfg['wandb']['project']} entity={cfg['wandb'].get('entity')}")
                wandb_handle = init_run(
                    run_id=state.wandb_run_id,
                    project=cfg["wandb"]["project"],
                    entity=cfg["wandb"].get("entity"),
                    config={"al_run_id": state.run_id, "compute": "local"},
                    tags=cfg["wandb"].get("tags"),
                    name=state.run_id,
                    resume="allow",
                )
                if wandb_handle.is_active:
                    print(f"[wandb] run dir: {wandb_handle._run.dir}")
                else:
                    print("[wandb] init returned no-op handle — see [wandb] errors above.",
                          file=sys.stderr)

            _ensure_norm_config(remote, state, ws)
            local_extra = _read_local_extra(ws)

            if iteration == 0:
                ckpt, scaler = _ensure_bootstrap_artifacts(remote, cfg, ws)
                train_metrics = {"wallclock_train_min": 0.0,
                                 "train_val_loss": None,
                                 "was_from_scratch": False}
                last_step = "iter0 bootstrap artifacts ready"
            else:
                _pull_compiled_h5(remote, remote_run_root,
                                  prior_iter=iteration - 1,
                                  dest=_local_compiled_h5(ws))
                last_step = f"pulled compiled_iter_{iteration-1:04d}.h5"
                ckpt, scaler, train_metrics = _train_locally(
                    state=state, cfg=cfg, ws=ws,
                    iter_idx=iteration, local_extra=local_extra,
                )
                last_step = f"trained iter {iteration}"

            local_extra["current_checkpoint"] = str(ckpt)
            local_extra["current_scaler"] = str(scaler)
            _write_local_extra(ws, local_extra)

            manifest_path = ws / "manifests" / f"manifest_iter_{iteration:04d}.json"
            local_acquire_dir = ws / "acquire" / f"iter_{iteration:04d}"
            # Short-circuit: if a previous run produced the manifest and its
            # snapshot/well_config artifacts, reuse them instead of re-running
            # acquisition (which would overwrite identical files).
            reused = (
                manifest_path.exists()
                and (local_acquire_dir / "snapshots_json").is_dir()
                and (local_acquire_dir / "well_configs").is_dir()
            )
            if reused:
                existing = json.loads(manifest_path.read_text())
                n_selected = len(existing.get("snapshots", []))
                acq_min = 0.0
                print(f"[local-driver] reusing existing manifest at {manifest_path} "
                      f"({n_selected} snapshots) — skipping acquire+select")
            else:
                manifest_path, n_selected, acq_min = _acquire_and_select_locally(
                    cfg=cfg, state=state, ws=ws, iter_idx=iteration,
                    ckpt=ckpt, scaler=scaler,
                )
            last_step = f"acquired+selected iter {iteration}" + (" (reused)" if reused else "")

            # The manifest references absolute paths to per-snapshot JSONs and
            # well_config .jl files that acquire wrote on bend, plus geology
            # H5 files at bend-local paths. Julia stage runs on Sherlock so we
            # (a) rsync the whole acquire/iter_NNNN/ tree over, and (b) rewrite
            # bend paths in the manifest to their Sherlock counterparts before
            # pushing it. Geology paths are mapped by matching filenames in the
            # local vs remote geology configs.
            remote_acquire_dir = f"{remote_run_root}/iter_{iteration:04d}/acquire"
            remote.run(["mkdir", "-p", remote_acquire_dir])
            # Trailing slash on src so rsync mirrors contents (snapshots_json/,
            # well_configs/) directly into remote_acquire_dir.
            remote.push(str(local_acquire_dir) + "/", remote_acquire_dir)

            remappings = [
                # acquire dir prefix first — most specific
                (str(local_acquire_dir), remote_acquire_dir),
                # geology h5 files — independent location
                *_build_geology_remappings(cfg),
            ]
            rewritten = _rewrite_manifest_paths(manifest_path, remappings)
            rewritten_local = ws / "manifests" / f"manifest_iter_{iteration:04d}.remote.json"
            rewritten_local.write_text(json.dumps(rewritten, indent=2))
            remote_manifest = f"{remote_run_root}/manifests/manifest_iter_{iteration:04d}.json"
            remote.run(["mkdir", "-p", f"{remote_run_root}/manifests"])
            remote.push(rewritten_local, remote_manifest)
            last_step = "pushed acquire dir + manifest"

            # Stage IX on Sherlock via remote runner.
            stage_root_remote = f"{remote_run_root}/iter_{iteration:04d}/ix_stage"
            remote.run(["mkdir", "-p", stage_root_remote])
            ix_cfg = cfg["intersect"]
            stage = stage_iteration(
                julia_repo=Path(cfg["paths"]["julia_repo"]),
                julia_config=cfg["paths"]["julia_config"],
                surrogate_repo=Path(cfg["paths"]["surrogate_repo"]),
                manifest_path=Path(remote_manifest),
                stage_root=Path(stage_root_remote),
                output_prefix=f"al_{run_id}_iter{iteration:04d}",
                cpus=int(ix_cfg.get("cpus_per_task", 2)),
                mem=str(ix_cfg.get("mem_per_task", "8GB")),
                time_limit=str(ix_cfg.get("time_per_run", "00:30:00")),
                np_procs=int(ix_cfg.get("np", 2)),
                max_concurrent=int(ix_cfg.get("max_concurrent", 70)),
                job_name=f"AL_IX_{run_id}_{iteration:04d}",
                runner=_make_sherlock_runner(remote),
                skip_script_existence_check=True,
            )
            last_step = f"staged IX (sbatch={stage.sbatch_path})"

            ix_job = remote.submit_sbatch(str(stage.sbatch_path))
            print(f"[local-driver] submitted IX array as job {ix_job}")
            last_step = f"submitted IX job {ix_job}"

            ingest_sbatch_remote = _render_and_push_ingest_sbatch(
                remote=remote,
                remote_run_root=remote_run_root,
                remote_repo_root=remote_repo_root,
                iteration=iteration,
                run_id=run_id,
                array_tasks_json_remote=str(stage.tasks_json),
                ix_output_dir_remote=ix_output_root,
                ws=ws,
            )
            ingest_job = remote.submit_sbatch(
                ingest_sbatch_remote, dependency=f"afterany:{ix_job}",
            )
            print(f"[local-driver] submitted ingest as job {ingest_job} (afterany:{ix_job})")
            last_step = f"submitted ingest job {ingest_job}"

            # Update state.json on Sherlock with this iteration's record.
            rec = state.get_iter(iteration) or IterationRecord(iteration=iteration)
            rec.submitted = stage.tasks_count
            rec.wallclock_acquire_min = acq_min
            rec.wallclock_train_min = train_metrics["wallclock_train_min"]
            rec.train_val_loss = train_metrics["train_val_loss"]
            rec.ix_array_job_id = ix_job
            rec.ingest_job_id = ingest_job
            rec.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            state.upsert_iter(rec)
            # Note: state.current_checkpoint stays bound to local paths in
            # hybrid mode. Ingest doesn't read it — it derives compiled H5
            # from paths.iter_compiled_h5(prior_iter) directly.
            state.current_checkpoint = str(ckpt)
            state.current_scaler = str(scaler)
            _push_state(remote, remote_run_root, state, ws)
            last_step = "pushed state"

            # Mirrors the per-iter log emitted by phase_train_acquire on Sherlock-only
            # runs, so dashboards look the same regardless of where train+acquire ran.
            if wandb_handle is not None:
                wandb_handle.log({
                    "iteration": iteration,
                    "n_candidates_submitted": stage.tasks_count,
                    "wallclock_acquire_min": acq_min,
                    "wallclock_train_min": train_metrics["wallclock_train_min"],
                    "train_val_loss": train_metrics["train_val_loss"],
                    "from_scratch": bool(train_metrics["was_from_scratch"]),
                }, step=iteration)

            # Clear any stale auth notice on a successful iter pass-through.
            (ws / "NEEDS_AUTH.md").unlink(missing_ok=True)

            final = _poll_until_terminal(remote, ingest_job, every=poll_every, label="ingest")
            last_step = f"ingest job {ingest_job} → {final}"
            # Pull the freshly-ingested state and refresh plots so the
            # just-completed iteration's metrics show up immediately.
            try:
                fresh_state = _pull_state(remote, remote_run_root, ws)
                _render_local_plots(fresh_state, ws)
            except SshUnavailable:
                raise
            except Exception as e:
                print(f"[local-driver] post-ingest plot refresh failed: {e}",
                      file=sys.stderr)
            if final != "COMPLETED":
                print(
                    f"[local-driver] ingest finished non-COMPLETED ({final}). "
                    f"Inspect logs at {remote_run_root}/logs/ingest_iter_{iteration:04d}.err "
                    f"on Sherlock. Stopping driver — re-run with --run-id {run_id} after "
                    f"resolving."
                )
                if wandb_handle is not None:
                    wandb_handle.log({"ingest_terminal_state": final}, step=iteration)
                    wandb_handle.finish()
                return 1

    except SshUnavailable as e:
        _write_needs_auth(
            ws,
            config_path=args.config.resolve(),
            run_id=run_id,
            iteration=iteration,
            last_step=last_step,
            error=e,
            ssh_host=compute["ssh_host"],
            control_path=compute["ssh_control_path"],
            notify_email=compute.get("notify_email"),
            notify_msmtp_account=compute.get("notify_msmtp_account", "gmail"),
        )
        return 2


def _read_julia_config(remote: RemoteSession, cfg: dict) -> dict:
    """Read Sherlock-side julia_config JSON over ssh."""
    julia_repo = cfg["paths"]["julia_repo"]
    julia_config = cfg["paths"]["julia_config"]
    if not julia_config.startswith("/"):
        julia_config = f"{julia_repo}/{julia_config}"
    proc = remote.run(["cat", julia_config], check=True)
    return json.loads(proc.stdout)


if __name__ == "__main__":
    sys.exit(main())
