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
import pickle
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
            )
            static_graphs.append(scaler.transform_graph(raw_graph))
    return static_graphs


def _scaler_to_torch(scaler, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(scaler.target_scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.target_scaler.scale_, dtype=torch.float32, device=device)
    return mean, scale


def _read_geology_metadata_safe(read_geology_metadata, src, geology_name: str) -> dict:
    try:
        return read_geology_metadata(src, geology_name=geology_name)
    except Exception:
        return {"geology_name": geology_name}


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

    out_dir = Path(out_dir)
    well_configs_dir = out_dir / "well_configs"
    snapshots_json_dir = out_dir / "snapshots_json"
    well_configs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_json_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.norm_config_path, "r") as f:
        norm_config = json.load(f)
    with open(cfg.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    # PyTorch 2.6+ defaults weights_only=True; allow PosixPath.
    import pathlib
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([pathlib.PosixPath, pathlib.WindowsPath])
    model = HeteroGNNRegressor.load_from_checkpoint(
        str(cfg.checkpoint_path), map_location=device
    ).to(device)
    model.eval()
    target_mean, target_scale = _scaler_to_torch(scaler, device)

    is_injector_list = [w.type == "injector" for w in cfg.wells]
    num_wells = len(cfg.wells)

    # Use a deterministic seed offset per iteration so AL runs are reproducible.
    base_seed = cfg.seed + iteration
    rng = np.random.default_rng(base_seed)
    sampler = qmc.LatinHypercube(d=2 * num_wells, seed=base_seed)

    snapshots: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    started = time.time()

    for geo in cfg.geologies:
        geo_path = Path(geo.geology_h5_file)
        geo_name = geo.geology_name or geo_path.stem
        with h5py.File(geo_path, "r") as src:
            geo_meta = _read_geology_metadata_safe(read_geology_metadata, src, geo_name)

        loaded = _load_geology(
            geo_path, norm_config,
            PROPERTIES, PERM_PROPS, get_valid_mask, find_z_cutoff,
        )
        nx, ny, z_max = loaded["nx"], loaded["ny"], loaded["z_max"]
        physics_dict, full_shape = loaded["physics_dict"], loaded["full_shape"]
        z_cutoff = loaded["z_cutoff"]
        temp0_full = loaded["temp0_full"]

        # LHS init for this geology.
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
            cfg.revenue_target, scaler,
            extract_well_data, build_wells_table, extract_vertical_profiles,
            build_single_hetero_data, temp0_full,
        )
        batch_data = Batch.from_data_list(static_graphs).to(device)

        # Continuous coords initialized from LHS.
        starts = [
            [[w["x"], w["y"], float(w["depth"])] for w in c] for c in cfgs
        ]
        coords = torch.tensor(starts, dtype=torch.float32, device=device)
        coords.requires_grad = True
        optimizer = optim.Adam([coords], lr=cfg.learning_rate)
        M = cfg.n_starts_per_geology

        def _predict_unscaled(c: torch.Tensor) -> torch.Tensor:
            batch_data["well"].pos_xyz = c.view(-1, 3)
            pred_scaled = model(batch_data).view(M)
            return pred_scaled * target_scale + target_mean

        # Pre-pick which starts will be tagged as adversarial when we hit K_adv.
        n_adv = max(0, int(round(M * cfg.adv_fraction)))
        adv_idx = set(int(i) for i in rng.choice(M, size=n_adv, replace=False)) if n_adv > 0 else set()

        # Run gradient descent up to K_adv. Snapshot at K_safe (frontier) and
        # K_adv (adversarial subset). Suppress per-step intermediate snapshots.
        K_total = max(cfg.k_safe, cfg.k_adv)
        for step in range(1, K_total + 1):
            optimizer.zero_grad()
            preds = _predict_unscaled(coords)
            loss = -preds.sum()
            loss.backward()

            # Boundary projection (matches run_ensemble_active_learning.py logic).
            with torch.no_grad():
                grads = coords.grad
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
                coords[:, :, 0].clamp_(0, nx - 1)
                coords[:, :, 1].clamp_(0, ny - 1)
                coords[:, :, 2].clamp_(0, z_max - 1)

            if step == cfg.k_safe:
                with torch.no_grad():
                    preds_now = _predict_unscaled(coords).detach().cpu().numpy()
                for m in range(M):
                    snap = _emit_snapshot(
                        run_id=geo.geology_index * 10000 + m,
                        iteration_step=step,
                        kind="frontier",
                        coords_xyz=coords[m].detach().cpu().numpy(),
                        is_injector_list=is_injector_list,
                        predicted_revenue=float(preds_now[m]),
                        geo=geo,
                        geo_path=geo_path,
                        geo_name=geo_name,
                        geo_meta=geo_meta,
                        well_configs_dir=well_configs_dir,
                        snapshots_json_dir=snapshots_json_dir,
                        to_julia_wells_text=to_julia_wells_text,
                    )
                    snapshots.append(snap)
                    candidates.append(_snapshot_to_candidate(snap, coords[m].detach().cpu().numpy(), is_injector_list))

            if step == cfg.k_adv and adv_idx:
                with torch.no_grad():
                    preds_now = _predict_unscaled(coords).detach().cpu().numpy()
                for m in sorted(adv_idx):
                    snap = _emit_snapshot(
                        run_id=geo.geology_index * 10000 + m,
                        iteration_step=step,
                        kind="adversarial",
                        coords_xyz=coords[m].detach().cpu().numpy(),
                        is_injector_list=is_injector_list,
                        predicted_revenue=float(preds_now[m]),
                        geo=geo,
                        geo_path=geo_path,
                        geo_name=geo_name,
                        geo_meta=geo_meta,
                        well_configs_dir=well_configs_dir,
                        snapshots_json_dir=snapshots_json_dir,
                        to_julia_wells_text=to_julia_wells_text,
                    )
                    snapshots.append(snap)
                    candidates.append(_snapshot_to_candidate(snap, coords[m].detach().cpu().numpy(), is_injector_list))

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
