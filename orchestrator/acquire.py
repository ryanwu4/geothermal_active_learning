"""Multi-geology gradient-based acquisition for the AL loop.

Per geology, run multi-start Adam on the surrogate's predicted revenue,
snapshot ``frontier`` candidates at K_safe steps and ``adversarial`` candidates
at K_adv steps. Reuses the surrogate-side primitives directly so we don't
duplicate physics-grid normalization or graph construction logic.

The loop here is intentionally small and *not* an ensemble — the empirical
findings (see project memory) showed deep ensembles do not provide useful
calibration in this domain, so we run a single checkpoint.

The output is a manifest JSON in the schema consumed by
``GeologicalSimulationWrapper.jl/scripts/cli_surrogate_array_prepare.jl``,
plus per-snapshot ``.jl`` files written via ``to_julia_wells_text`` from
``geothermal/active_learning_utils.py``.
"""
from __future__ import annotations

import json
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.optim as optim
from scipy.stats import qmc
from torch_geometric.data import Batch

from .select import Candidate


@dataclass
class WellSpec:
    type: str  # "injector" or "producer"
    depth: int


@dataclass
class GeologySpec:
    geology_index: int
    geology_h5_file: str
    geology_name: str | None = None  # Optional; defaults to file stem.


@dataclass
class AcquisitionConfig:
    surrogate_repo: Path
    checkpoint_path: Path
    scaler_path: Path
    norm_config_path: Path
    geologies: list[GeologySpec]
    wells: list[WellSpec]
    n_starts_per_geology: int
    k_safe: int
    k_adv: int
    adv_fraction: float
    edge_buffer: int
    learning_rate: float
    log_every_n_steps: int
    revenue_target: str
    seed: int = 42
    device: str = "cuda:0"
    # Optional list of CUDA devices for cross-geology parallelism. When None or
    # of length 1, runs in-process on a single device (back-compat). When > 1,
    # spawns one worker subprocess per device and dispatches geologies across
    # them via a shared queue. See ``run_acquisition`` for details.
    devices: list[str] | None = None


# Canonical ordering for snapshots in the aggregate manifest so parallel runs
# produce byte-identical output to serial runs.
_KIND_ORDER = {"frontier": 0, "adversarial": 1}


def _snapshot_sort_key(snap: dict) -> tuple[int, int, int]:
    return (
        int(snap.get("well_config_paths_by_geology", [{}])[0].get("geology_index", 0)),
        _KIND_ORDER.get(snap.get("kind", ""), 99),
        int(snap.get("run_id", 0)),
    )


def _ensure_surrogate_imports(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _load_geology(
    geology_h5_file: Path,
    norm_config: dict,
    PROPERTIES,
    PERM_PROPS,
    get_valid_mask,
    find_z_cutoff,
) -> dict[str, Any]:
    """Load one geology file and produce the static physics tensors + bounds."""
    with h5py.File(geology_h5_file, "r") as src:
        valid_mask = get_valid_mask(src)
        z_cutoff = find_z_cutoff(valid_mask, invalid_threshold=0.95)
        valid_mask_cropped = valid_mask[:z_cutoff]

        physics_dict: dict[str, torch.Tensor] = {}
        for prop in PROPERTIES:
            data = src[f"Input/{prop}"][:z_cutoff].astype(np.float32)
            if prop in PERM_PROPS:
                data = np.log10(np.maximum(data, 1e-15))
            p_min = norm_config[prop]["min"]
            p_max = norm_config[prop]["max"]
            normalized = (
                (data - p_min) / (p_max - p_min)
                if p_max > p_min
                else np.zeros_like(data)
            )
            normalized = np.clip(normalized, 0.0, 1.0)
            normalized[~valid_mask_cropped] = 0.0
            physics_dict[prop] = torch.tensor(normalized, dtype=torch.float32)

        temp0_full = src["Input/Temperature0"][:]

    physics_dict["valid_mask"] = torch.tensor(valid_mask_cropped, dtype=torch.float32)
    full_shape = (z_cutoff, valid_mask.shape[1], valid_mask.shape[2])
    return {
        "physics_dict": physics_dict,
        "full_shape": full_shape,
        "z_cutoff": z_cutoff,
        "nx": valid_mask.shape[1],
        "ny": valid_mask.shape[2],
        "z_max": z_cutoff,
        "temp0_full": temp0_full,
    }


def _build_static_batch_for_starts(
    cfgs: list[list[dict]],
    geology_h5_file: Path,
    physics_dict: dict[str, torch.Tensor],
    full_shape: tuple[int, int, int],
    z_cutoff: int,
    nx: int,
    ny: int,
    revenue_target: str,
    scaler,
    extract_well_data,
    build_wells_table,
    extract_vertical_profiles,
    build_single_hetero_data,
    temp0_full: np.ndarray,
    node_encoder: str = "profile",
    enrich_global_attr: bool = False,
):
    """Build a list of normalized HeteroData objects, one per start."""
    static_graphs = []
    with h5py.File(geology_h5_file, "r") as src:
        for m_idx, w_cfg in enumerate(cfgs):
            is_well = np.zeros((z_cutoff, nx, ny), dtype=np.int32)
            inj_rate = np.zeros((z_cutoff, nx, ny), dtype=np.float32)
            for w in w_cfg:
                ix, iy = int(round(w["x"])), int(round(w["y"]))
                ix = int(np.clip(ix, 0, nx - 1))
                iy = int(np.clip(iy, 0, ny - 1))
                for z in range(int(w["depth"])):
                    if temp0_full[z, ix, iy] <= -900:
                        is_well[z, ix, iy] = -999
                        inj_rate[z, ix, iy] = -999
                    else:
                        is_well[z, ix, iy] = 1
                        inj_rate[z, ix, iy] = (
                            8000.0 if w["type"] == "injector" else -8000.0
                        )

            (
                x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z,
                porosity, temp0, press0, depth_centroid,
            ) = extract_well_data(is_well, inj_rate, src)
            wells = build_wells_table(
                x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z,
                porosity, temp0, press0,
            )
            vertical_profiles = extract_vertical_profiles(is_well, x_idx, y_idx, src)
            raw_graph = build_single_hetero_data(
                wells=wells,
                physics_dict=physics_dict,
                full_shape=full_shape,
                target=revenue_target,
                target_val=0.0,
                vertical_profile=vertical_profiles,
                case_id=f"start{m_idx}",
                node_encoder=node_encoder,
                enrich_global_attr=enrich_global_attr,
            )
            static_graphs.append(scaler.transform_graph(raw_graph))
    return static_graphs


def _scaler_to_torch(scaler, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(scaler.target_scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.target_scaler.scale_, dtype=torch.float32, device=device)
    return mean, scale


def _read_geology_metadata_safe(read_geology_metadata, src, geology_name: str) -> dict:
    meta = read_geology_metadata(src, geology_name=geology_name)
    if meta.get("geology_config_id") in (None, ""):
        raise RuntimeError(
            f"Geology {geology_name!r} is missing geology_config_id (Metadata/RepNum or ScenarioName). "
            "Refusing to emit a manifest snapshot without it — Julia staging would silently default "
            "geology_config_id to 1 and collide all such snapshots under scenario 1."
        )
    return meta


@dataclass
class _WorkerContext:
    """Per-process state for acquisition. Built once per worker."""

    device: torch.device
    model: Any
    scaler: Any
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    norm_config: dict
    is_injector_list: list[bool]
    num_wells: int
    base_seed: int
    # Resolved late-import callables (cached so we don't re-resolve per geology).
    build_wells_table: Any
    extract_vertical_profiles: Any
    extract_well_data: Any
    read_geology_metadata: Any
    to_julia_wells_text: Any
    build_single_hetero_data: Any
    PROPERTIES: Any
    PERM_PROPS: Any
    find_z_cutoff: Any
    get_valid_mask: Any
    # Data-pipeline knobs read from the loaded checkpoint's hparams so the graphs
    # we build here match the scaler the model was trained with.
    node_encoder: str = "profile"
    enrich_global_attr: bool = False


def _build_worker_context(
    cfg: AcquisitionConfig, iteration: int, device_str: str
) -> _WorkerContext:
    """Resolve surrogate imports and load model+scaler+normalization onto a
    given CUDA device. Safe to call from a freshly-spawned subprocess.
    """
    _ensure_surrogate_imports(cfg.surrogate_repo)

    # Late imports so the surrogate repo's modules resolve from sys.path.
    from compile_minimal_geothermal_h5 import (  # type: ignore
        build_wells_table,
        extract_vertical_profiles,
        extract_well_data,
    )
    from geothermal.active_learning_utils import (  # type: ignore
        read_geology_metadata,
        to_julia_wells_text,
    )
    from geothermal.data import build_single_hetero_data  # type: ignore
    from geothermal.model import HeteroGNNRegressor  # type: ignore
    from preprocess_h5 import (  # type: ignore
        PROPERTIES,
        PERM_PROPS,
        find_z_cutoff,
        get_valid_mask,
    )

    with open(cfg.norm_config_path, "r") as f:
        norm_config = json.load(f)
    with open(cfg.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    # PyTorch 2.6+ defaults weights_only=True; allow PosixPath.
    import pathlib
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
    model = HeteroGNNRegressor.load_from_checkpoint(
        str(cfg.checkpoint_path), map_location=device
    ).to(device)
    model.eval()
    target_mean, target_scale = _scaler_to_torch(scaler, device)

    # Recover the data-pipeline config from the checkpoint's saved_hyperparameters.
    # node_encoder is stored directly. enrich_global_attr is inferred from global_dim:
    # legacy (n_wells only) = 1; enriched (n_wells + 7 reservoir means/anisotropy) = 8.
    hp = model.hparams
    node_encoder = getattr(hp, "node_encoder", "profile")
    global_dim = int(getattr(hp, "global_dim", 1))
    enrich_global_attr = global_dim != 1

    is_injector_list = [w.type == "injector" for w in cfg.wells]
    num_wells = len(cfg.wells)
    base_seed = cfg.seed + iteration

    return _WorkerContext(
        device=device,
        model=model,
        scaler=scaler,
        target_mean=target_mean,
        target_scale=target_scale,
        norm_config=norm_config,
        is_injector_list=is_injector_list,
        num_wells=num_wells,
        base_seed=base_seed,
        build_wells_table=build_wells_table,
        extract_vertical_profiles=extract_vertical_profiles,
        extract_well_data=extract_well_data,
        read_geology_metadata=read_geology_metadata,
        to_julia_wells_text=to_julia_wells_text,
        build_single_hetero_data=build_single_hetero_data,
        PROPERTIES=PROPERTIES,
        PERM_PROPS=PERM_PROPS,
        find_z_cutoff=find_z_cutoff,
        get_valid_mask=get_valid_mask,
        node_encoder=node_encoder,
        enrich_global_attr=enrich_global_attr,
    )


def _per_geology_seed(base_seed: int, geology_index: int) -> int:
    """Deterministic per-geology seed. Decoupling each geology from the global
    sampler state lets us partition geologies across worker processes without
    changing numerical results.
    """
    # Mask to 64 bits so numpy/scipy seed APIs accept it on all platforms.
    return ((base_seed * 1_000_003) ^ (geology_index * 1009)) & ((1 << 63) - 1)


def _run_one_geology(
    *,
    cfg: AcquisitionConfig,
    geo: GeologySpec,
    iteration: int,
    ctx: _WorkerContext,
    well_configs_dir: Path,
    snapshots_json_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run gradient-descent acquisition for a single geology on ``ctx.device``.

    Returns (snapshot_records, candidate_payloads). Candidate payloads are
    plain dicts so they survive the IPC boundary; the parent converts them
    back to ``Candidate`` instances via ``_payload_to_candidate``.
    """
    device = ctx.device
    model = ctx.model
    target_mean = ctx.target_mean
    target_scale = ctx.target_scale
    is_injector_list = ctx.is_injector_list
    num_wells = ctx.num_wells

    geo_path = Path(geo.geology_h5_file)
    geo_name = geo.geology_name or geo_path.stem
    with h5py.File(geo_path, "r") as src:
        geo_meta = _read_geology_metadata_safe(ctx.read_geology_metadata, src, geo_name)

    loaded = _load_geology(
        geo_path, ctx.norm_config,
        ctx.PROPERTIES, ctx.PERM_PROPS, ctx.get_valid_mask, ctx.find_z_cutoff,
    )
    nx, ny, z_max = loaded["nx"], loaded["ny"], loaded["z_max"]
    physics_dict, full_shape = loaded["physics_dict"], loaded["full_shape"]
    z_cutoff = loaded["z_cutoff"]
    temp0_full = loaded["temp0_full"]

    # Per-geology RNG so geology results don't depend on iteration order across workers.
    seed_geo = _per_geology_seed(ctx.base_seed, geo.geology_index)
    rng = np.random.default_rng(seed_geo)
    sampler = qmc.LatinHypercube(d=2 * num_wells, seed=seed_geo)

    x_lo, x_hi = float(cfg.edge_buffer), float(nx - 1 - cfg.edge_buffer)
    y_lo, y_hi = float(cfg.edge_buffer), float(ny - 1 - cfg.edge_buffer)
    lhs = sampler.random(n=cfg.n_starts_per_geology)
    cfgs: list[list[dict]] = []
    for n in range(cfg.n_starts_per_geology):
        c = []
        for w in range(num_wells):
            rx = x_lo + lhs[n, 2 * w] * (x_hi - x_lo)
            ry = y_lo + lhs[n, 2 * w + 1] * (y_hi - y_lo)
            depth = min(int(cfg.wells[w].depth), int(z_max))
            c.append({
                "x": float(rx), "y": float(ry),
                "depth": int(depth), "type": cfg.wells[w].type,
            })
        cfgs.append(c)

    static_graphs = _build_static_batch_for_starts(
        cfgs, geo_path, physics_dict, full_shape, z_cutoff, nx, ny,
        cfg.revenue_target, ctx.scaler,
        ctx.extract_well_data, ctx.build_wells_table, ctx.extract_vertical_profiles,
        ctx.build_single_hetero_data, temp0_full,
        node_encoder=ctx.node_encoder,
        enrich_global_attr=ctx.enrich_global_attr,
    )
    batch_data = Batch.from_data_list(static_graphs).to(device)

    starts = [
        [[w["x"], w["y"], float(w["depth"])] for w in c] for c in cfgs
    ]
    coords = torch.tensor(starts, dtype=torch.float32, device=device)
    coords.requires_grad = True
    optimizer = optim.Adam([coords], lr=cfg.learning_rate)
    M = cfg.n_starts_per_geology
    last_valid_coords = coords.detach().clone()

    def _predict_unscaled(c: torch.Tensor) -> torch.Tensor:
        batch_data["well"].pos_xyz = c.view(-1, 3)
        pred_scaled = model(batch_data).view(M)
        return pred_scaled * target_scale + target_mean

    n_adv = max(0, int(round(M * cfg.adv_fraction)))
    adv_idx = set(int(i) for i in rng.choice(M, size=n_adv, replace=False)) if n_adv > 0 else set()

    snapshots: list[dict[str, Any]] = []
    cand_payloads: list[dict[str, Any]] = []

    K_total = max(cfg.k_safe, cfg.k_adv)
    for step in range(1, K_total + 1):
        optimizer.zero_grad()
        preds = _predict_unscaled(coords)
        loss = -preds.sum()
        loss.backward()

        with torch.no_grad():
            grads = coords.grad
            if not torch.isfinite(grads).all():
                torch.nan_to_num_(grads, nan=0.0, posinf=0.0, neginf=0.0)
            for d, max_val in enumerate([nx - 1, ny - 1, z_max - 1]):
                mask_lo = (coords[:, :, d] <= 1e-4) & (grads[:, :, d] > 0)
                mask_hi = (coords[:, :, d] >= max_val - 1e-4) & (grads[:, :, d] < 0)
                mask = mask_lo | mask_hi
                if mask.any():
                    grads[..., d][mask] = 0.0
                    st = optimizer.state.get(coords, {})
                    if "exp_avg" in st:
                        st["exp_avg"][..., d][mask] = 0.0
        optimizer.step()
        with torch.no_grad():
            if not torch.isfinite(coords).all():
                bad = ~torch.isfinite(coords)
                coords[bad] = last_valid_coords[bad]
                st = optimizer.state.get(coords, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st and not torch.isfinite(st[key]).all():
                        torch.nan_to_num_(st[key], nan=0.0, posinf=0.0, neginf=0.0)
            coords[:, :, 0].clamp_(0, nx - 1)
            coords[:, :, 1].clamp_(0, ny - 1)
            coords[:, :, 2].clamp_(0, z_max - 1)
            last_valid_coords.copy_(coords)

        if step == cfg.k_safe:
            with torch.no_grad():
                preds_now = _predict_unscaled(coords).detach().cpu().numpy()
            for m in range(M):
                cxyz = coords[m].detach().cpu().numpy()
                pred_val = float(preds_now[m])
                if not (np.isfinite(cxyz).all() and np.isfinite(pred_val)):
                    print(f"[acquire] skip non-finite frontier snapshot geo={geo.geology_index} m={m}", flush=True)
                    continue
                snap = _emit_snapshot(
                    run_id=geo.geology_index * 10000 + m,
                    iteration_step=step,
                    kind="frontier",
                    coords_xyz=cxyz,
                    is_injector_list=is_injector_list,
                    predicted_revenue=pred_val,
                    geo=geo,
                    geo_path=geo_path,
                    geo_name=geo_name,
                    geo_meta=geo_meta,
                    well_configs_dir=well_configs_dir,
                    snapshots_json_dir=snapshots_json_dir,
                    to_julia_wells_text=ctx.to_julia_wells_text,
                )
                snapshots.append(snap)
                cand_payloads.append({
                    "snap": snap,
                    "coords_xyz": cxyz.tolist(),
                    "is_injector": list(is_injector_list),
                })

        if step == cfg.k_adv and adv_idx:
            with torch.no_grad():
                preds_now = _predict_unscaled(coords).detach().cpu().numpy()
            for m in sorted(adv_idx):
                cxyz = coords[m].detach().cpu().numpy()
                pred_val = float(preds_now[m])
                if not (np.isfinite(cxyz).all() and np.isfinite(pred_val)):
                    print(f"[acquire] skip non-finite adversarial snapshot geo={geo.geology_index} m={m}", flush=True)
                    continue
                snap = _emit_snapshot(
                    run_id=geo.geology_index * 10000 + m,
                    iteration_step=step,
                    kind="adversarial",
                    coords_xyz=cxyz,
                    is_injector_list=is_injector_list,
                    predicted_revenue=pred_val,
                    geo=geo,
                    geo_path=geo_path,
                    geo_name=geo_name,
                    geo_meta=geo_meta,
                    well_configs_dir=well_configs_dir,
                    snapshots_json_dir=snapshots_json_dir,
                    to_julia_wells_text=ctx.to_julia_wells_text,
                )
                snapshots.append(snap)
                cand_payloads.append({
                    "snap": snap,
                    "coords_xyz": cxyz.tolist(),
                    "is_injector": list(is_injector_list),
                })

    return snapshots, cand_payloads


def _payload_to_candidate(payload: dict[str, Any]) -> Candidate:
    coords_xyz = np.asarray(payload["coords_xyz"], dtype=np.float32)
    return _snapshot_to_candidate(payload["snap"], coords_xyz, payload["is_injector"])


def _install_parent_death_signal() -> None:
    """On Linux, request SIGTERM when our parent process dies.

    Without this, killing the orchestrator (Ctrl-C, SIGKILL, kernel OOM killer,
    SLURM job timeout, etc.) leaves the multi-GPU worker subprocesses orphaned —
    they get re-parented to PID 1 and continue running, holding GPU memory and
    file handles until manually `pkill`-ed.

    The PR_SET_PDEATHSIG prctl wires the kernel to send SIGTERM to this process
    the moment its parent exits. Linux-only; silently no-op elsewhere.
    """
    try:
        import sys as _sys
        if not _sys.platform.startswith("linux"):
            return
        import ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # Send SIGTERM (15) when parent dies. SIGTERM lets Python's atexit run.
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        # Best-effort; failure here is not a reason to crash the worker.
        pass


def _multi_gpu_worker(
    device_str: str,
    cfg: AcquisitionConfig,
    iteration: int,
    well_configs_dir: str,
    snapshots_json_dir: str,
    in_queue,
    out_queue,
) -> None:
    """Subprocess entry. Owns one CUDA device for the lifetime of the
    acquisition call. Pulls GeologySpec instances off ``in_queue`` and pushes
    (snapshots, candidate_payloads) onto ``out_queue``. ``None`` is the
    poison-pill stop signal.
    """
    # Auto-terminate if the parent dies — prevents orphan workers from holding
    # the GPU when the orchestrator is killed.
    _install_parent_death_signal()
    # Also handle SIGTERM gracefully so the worker exits cleanly (no zombie).
    def _sigterm(_sig, _frame):
        # Drain queues just enough to avoid mp_ctx deadlocks; then exit.
        try:
            out_queue.put(("__init_error__", "worker received SIGTERM"))
        except Exception:
            pass
        os._exit(0)
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    try:
        # Pin this process to the requested device so any incidental CUDA
        # ops (e.g. inside torch_geometric) land on the right GPU.
        if device_str.startswith("cuda:") and torch.cuda.is_available():
            torch.cuda.set_device(int(device_str.split(":", 1)[1]))
        ctx = _build_worker_context(cfg, iteration, device_str)
        wc_dir = Path(well_configs_dir)
        sj_dir = Path(snapshots_json_dir)
    except BaseException as e:  # propagate setup errors to parent
        out_queue.put(("__init_error__", repr(e)))
        return

    while True:
        geo = in_queue.get()
        if geo is None:
            return
        try:
            snaps, cand_payloads = _run_one_geology(
                cfg=cfg,
                geo=geo,
                iteration=iteration,
                ctx=ctx,
                well_configs_dir=wc_dir,
                snapshots_json_dir=sj_dir,
            )
            out_queue.put((device_str, geo.geology_index, snaps, cand_payloads))
        except BaseException as e:
            out_queue.put(("__error__", geo.geology_index, repr(e)))


def _terminate_procs(procs: list, join_timeout: float = 5.0) -> None:
    """Terminate (then kill, if needed) every still-alive worker.

    Used both in the normal `finally` cleanup and in the signal-handler path.
    Idempotent: safe to call multiple times.
    """
    for p in procs:
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass
    for p in procs:
        try:
            p.join(timeout=join_timeout)
        except Exception:
            pass
    # Anything still alive gets SIGKILL.
    for p in procs:
        if p.is_alive():
            try:
                p.kill()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                p.join(timeout=2)
            except Exception:
                pass


def _run_acquisition_multi_gpu(
    *,
    cfg: AcquisitionConfig,
    iteration: int,
    devices: list[str],
    well_configs_dir: Path,
    snapshots_json_dir: Path,
) -> tuple[list[dict[str, Any]], list[Candidate]]:
    import torch.multiprocessing as tmp

    mp_ctx = tmp.get_context("spawn")
    in_queue = mp_ctx.Queue()
    out_queue = mp_ctx.Queue()

    for geo in cfg.geologies:
        in_queue.put(geo)
    for _ in devices:
        in_queue.put(None)  # one poison pill per worker

    procs = []
    for dev in devices:
        p = mp_ctx.Process(
            target=_multi_gpu_worker,
            args=(
                dev, cfg, iteration,
                str(well_configs_dir), str(snapshots_json_dir),
                in_queue, out_queue,
            ),
        )
        p.start()
        procs.append(p)

    # Install a signal handler in the PARENT that tears down workers cleanly
    # on Ctrl-C / SIGTERM (e.g. when the orchestrator is killed). Combined with
    # PR_SET_PDEATHSIG in the worker, this gives us defense-in-depth: workers
    # die from either the parent's signal handler OR the kernel notification
    # when the parent vanishes abruptly.
    _prev_sigint = signal.getsignal(signal.SIGINT)
    _prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _parent_signal_cleanup(sig, frame):
        _terminate_procs(procs)
        # Re-raise via KeyboardInterrupt so the surrounding `try` cleanup runs
        # and the calling code sees a clean exception.
        raise KeyboardInterrupt(f"acquisition aborted by signal {sig}")

    try:
        signal.signal(signal.SIGINT, _parent_signal_cleanup)
        signal.signal(signal.SIGTERM, _parent_signal_cleanup)
    except (ValueError, OSError):
        # signal.signal only works on the main thread; if we're being invoked
        # from a worker thread (e.g. AL orchestrator background thread), skip.
        pass

    snapshots: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    expected = len(cfg.geologies)
    received = 0
    errors: list[str] = []

    try:
        while received < expected:
            msg = out_queue.get()
            if msg[0] == "__init_error__":
                errors.append(f"worker init failure: {msg[1]}")
                # If one worker died at init, we still need to drain the rest
                # via the poison pills already queued. Mark all of that worker's
                # share as failed so we don't deadlock.
                expected -= 1
                continue
            if msg[0] == "__error__":
                errors.append(f"worker error on geo={msg[1]}: {msg[2]}")
                received += 1
                continue
            _dev, _geo_idx, snaps, cand_payloads = msg
            snapshots.extend(snaps)
            for payload in cand_payloads:
                candidates.append(_payload_to_candidate(payload))
            received += 1
    finally:
        _terminate_procs(procs, join_timeout=30.0)
        # Restore prior signal handlers so we don't poison the orchestrator's
        # behavior in later phases.
        try:
            signal.signal(signal.SIGINT, _prev_sigint)
            signal.signal(signal.SIGTERM, _prev_sigterm)
        except (ValueError, OSError):
            pass

    if errors:
        raise RuntimeError("Multi-GPU acquisition failures:\n  " + "\n  ".join(errors))

    return snapshots, candidates


def run_acquisition(
    cfg: AcquisitionConfig,
    *,
    out_dir: Path,
    iteration: int,
    run_id_prefix: str = "al",
) -> dict[str, Any]:
    """Run multi-geology acquisition and emit a manifest at ``out_dir``.

    Returns the parsed manifest dict; also writes:
      - ``out_dir/well_configs/<snapshot_id>.jl`` per snapshot
      - ``out_dir/snapshots_json/<snapshot_id>.json`` per snapshot
      - ``out_dir/manifest.json`` aggregate
    """
    # Surrogate imports + heavy state (model, scaler, normalization) are
    # loaded inside ``_build_worker_context`` so the same path works for
    # in-process and spawned-worker execution.
    out_dir = Path(out_dir)
    well_configs_dir = out_dir / "well_configs"
    snapshots_json_dir = out_dir / "snapshots_json"
    well_configs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_json_dir.mkdir(parents=True, exist_ok=True)

    devices = list(cfg.devices) if cfg.devices else [cfg.device]

    snapshots: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    started = time.time()

    if len(devices) <= 1:
        # In-process path. Load model once, iterate geologies serially.
        device_str = devices[0]
        ctx = _build_worker_context(cfg, iteration, device_str)
        for geo in cfg.geologies:
            snaps, cand_payloads = _run_one_geology(
                cfg=cfg,
                geo=geo,
                iteration=iteration,
                ctx=ctx,
                well_configs_dir=well_configs_dir,
                snapshots_json_dir=snapshots_json_dir,
            )
            snapshots.extend(snaps)
            for payload in cand_payloads:
                candidates.append(_payload_to_candidate(payload))
    else:
        # Multi-GPU path. Spawn one worker per device; geologies are pulled off
        # a shared queue. Each worker loads model+scaler once per process.
        # Order of completion is not deterministic, so we sort the merged
        # snapshot list at the end to keep the manifest reproducible.
        snapshots, candidates = _run_acquisition_multi_gpu(
            cfg=cfg,
            iteration=iteration,
            devices=devices,
            well_configs_dir=well_configs_dir,
            snapshots_json_dir=snapshots_json_dir,
        )
        snapshots.sort(key=_snapshot_sort_key)
        candidates.sort(key=lambda c: (c.geology_index, _KIND_ORDER.get(c.kind, 99), c.run_id))

    # Write aggregate manifest matching cli_surrogate_array_prepare.jl schema.
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "run_id_prefix": run_id_prefix,
        "k_safe": cfg.k_safe,
        "k_adv": cfg.k_adv,
        "adv_fraction": cfg.adv_fraction,
        "n_starts_per_geology": cfg.n_starts_per_geology,
        "geology_metadata": [
            {
                "geology_index": g.geology_index,
                "geology_name": g.geology_name or Path(g.geology_h5_file).stem,
                "geology_file": str(Path(g.geology_h5_file).resolve()),
            }
            for g in cfg.geologies
        ],
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "wallclock_seconds": time.time() - started,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {"manifest": manifest, "manifest_path": str(manifest_path), "candidates": candidates}


def _emit_snapshot(
    *,
    run_id: int,
    iteration_step: int,
    kind: str,
    coords_xyz: np.ndarray,
    is_injector_list: list[bool],
    predicted_revenue: float,
    geo: GeologySpec,
    geo_path: Path,
    geo_name: str,
    geo_meta: dict,
    well_configs_dir: Path,
    snapshots_json_dir: Path,
    to_julia_wells_text,
) -> dict[str, Any]:
    snapshot_id = f"geo{geo.geology_index:02d}_run{run_id:06d}_step{iteration_step:04d}_{kind}"
    jl_path = well_configs_dir / f"{snapshot_id}.jl"
    json_path = snapshots_json_dir / f"{snapshot_id}.json"

    jl_text = to_julia_wells_text(
        coords_xyz=coords_xyz,
        is_injector_list=is_injector_list,
        score=predicted_revenue,
        score_label="Predicted Discounted Revenue",
        geology_file=str(geo_path),
        geology_name=geo_name,
        geology_config_id=geo_meta.get("geology_config_id"),
        geology_scenario_name=geo_meta.get("scenario_name"),
        geology_sample_num=geo_meta.get("sample_num"),
        predicted_discounted_revenue=predicted_revenue,
    )
    jl_path.write_text(jl_text)

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
        "run_id": int(run_id),
        "iteration": int(iteration_step),
        "kind": kind,
        "predicted_discounted_total_revenue": float(predicted_revenue),
        "geology_index": geo.geology_index,
        "geology_name": geo_name,
        "geology_file": str(geo_path),
        "geology_config_id": geo_meta.get("geology_config_id"),
        "geology_scenario_name": geo_meta.get("scenario_name"),
        "geology_sample_num": geo_meta.get("sample_num"),
        "wells": wells_json,
        "predictions_by_geology": [
            {
                "geology_index": geo.geology_index,
                "geology_name": geo_name,
                "geology_file": str(geo_path),
                "geology_config_id": geo_meta.get("geology_config_id"),
                "geology_scenario_name": geo_meta.get("scenario_name"),
                "geology_sample_num": geo_meta.get("sample_num"),
                "discounted_total_revenue": float(predicted_revenue),
                "total_energy_production": 0.0,
            }
        ],
    }
    with open(json_path, "w") as f:
        json.dump(snap_payload, f, indent=2)

    # Record in the format expected by cli_surrogate_array_prepare.jl.
    snapshot_record = {
        "snapshot_id": snapshot_id,
        "run_id": int(run_id),
        "iteration": int(iteration_step),
        "kind": kind,
        "json_path": str(json_path),
        "well_config_path": str(jl_path),
        "well_config_paths_by_geology": [
            {
                "geology_index": geo.geology_index,
                "geology_name": geo_name,
                "geology_file": str(geo_path),
                "geology_config_id": geo_meta.get("geology_config_id"),
                "geology_scenario_name": geo_meta.get("scenario_name"),
                "geology_sample_num": geo_meta.get("sample_num"),
                "well_config_path": str(jl_path),
            }
        ],
        "predicted_discounted_total_revenue": float(predicted_revenue),
    }
    return snapshot_record


def _snapshot_to_candidate(snap: dict, coords_xyz: np.ndarray, is_injector_list: list[bool]) -> Candidate:
    geo_entry = snap["well_config_paths_by_geology"][0]
    return Candidate(
        geology_index=int(geo_entry["geology_index"]),
        geology_file=str(geo_entry["geology_file"]),
        geology_name=str(geo_entry["geology_name"]),
        geology_config_id=geo_entry.get("geology_config_id"),
        geology_scenario_name=geo_entry.get("geology_scenario_name"),
        geology_sample_num=geo_entry.get("geology_sample_num"),
        snapshot_id=snap["snapshot_id"],
        run_id=int(snap["run_id"]),
        iteration=int(snap["iteration"]),
        kind=snap["kind"],
        predicted_revenue=float(snap["predicted_discounted_total_revenue"]),
        coords_xyz=np.asarray(coords_xyz, dtype=np.float32),
        is_injector=list(is_injector_list),
        well_config_path=str(snap["well_config_path"]),
        snapshot_json_path=str(snap.get("json_path", "")),
    )


def write_selected_manifest(
    selected: list[Candidate],
    *,
    out_path: Path,
    iteration: int,
    geologies: list[GeologySpec],
    extras: dict | None = None,
) -> Path:
    """Persist the *selected* batch as a manifest in the
    ``cli_surrogate_array_prepare.jl`` schema.
    """
    snapshots = []
    for c in selected:
        snapshots.append({
            "snapshot_id": c.snapshot_id,
            "run_id": int(c.run_id),
            "iteration": int(c.iteration),
            "kind": c.kind,
            "json_path": c.snapshot_json_path,
            "well_config_path": c.well_config_path,
            "well_config_paths_by_geology": [
                {
                    "geology_index": c.geology_index,
                    "geology_name": c.geology_name,
                    "geology_file": c.geology_file,
                    "geology_config_id": c.geology_config_id,
                    "geology_scenario_name": c.geology_scenario_name,
                    "geology_sample_num": c.geology_sample_num,
                    "well_config_path": c.well_config_path,
                }
            ],
            "predicted_discounted_total_revenue": float(c.predicted_revenue),
        })

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "geology_metadata": [
            {
                "geology_index": g.geology_index,
                "geology_name": g.geology_name or Path(g.geology_h5_file).stem,
                "geology_file": str(Path(g.geology_h5_file).resolve()),
            }
            for g in geologies
        ],
    }
    if extras:
        manifest.update(extras)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path
