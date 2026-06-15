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
import threading
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
    n_starts_per_geology: int  # LHS starts (frontier+adversarial provenance)
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
    # 4-kind acquisition knobs. Defaults of 0 keep old behavior (LHS-only).
    n_elite_per_geology: int = 0           # exploit kind: Adam seeded from top-K real-revenue priors
    n_cma_per_geology: int = 0             # cma kind: Adam seeded from CMA-ES warm-up
    elite_top_k: int = 10                  # pool size before sampling n_elite seeds
    elite_seed_noise: float = 2.0          # σ (grid cells) added to each elite seed
    cma_popsize: int = 16
    cma_generations: int = 10
    cma_sigma_init: float = 5.0
    # cma_surrogate only: if True, seed the CMA mean from the prior real-revenue
    # incumbent on iter>=1 (warm start). Default False = cold LHS restart every AL
    # iteration. The cma_trajectory_long IX experiment showed warm-starting drops
    # the search into the surrogate's over-confident exploit regime and
    # over-exploits (real revenue degrades while predicted climbs; pred-real gap
    # swings to +52 M$), whereas a cold restart stayed calibrated and reached
    # ~+55 M$ higher real revenue at gen 100.
    cma_warm_start: bool = False
    # List of prior-iter per_candidate_metrics.json paths, in iter order.
    # Phase script populates this; acquire derives snapshots_json dirs from it
    # to recover well coords for the elite path.
    prior_metrics: list[Path] | None = None

    # Mode switch. "per_geology" (default) runs the legacy per-geology Adam loop;
    # "ensemble" runs one Adam loop per candidate against ALL K geologies, with
    # loss = mean over the ensemble. Ensemble mode ignores the cma/adversarial
    # knobs and uses n_exploit / n_frontier instead.
    mode: str = "per_geology"
    # Ensemble-mode knobs. Each candidate spawns K IX runs (one per geology) so
    # n_candidates × K is the simulator budget per iteration.
    n_exploit: int = 0     # ensemble: Adam-from-elite-seed starts (kind="exploit")
    n_frontier: int = 0    # ensemble: Adam-from-LHS starts (kind="frontier")

    # Depth (well-length) bounds for the "cma_surrogate" mode, which optimizes
    # per-well (x, y, z). Ignored by the Adam paths (they pin depth per WellSpec).
    # z_hi is additionally capped at the reservoir grid extent (z_max-1) at
    # runtime so emitted depths stay inside every geology's active grid.
    depth_min: int = 5
    depth_max: int = 70

    # ----- proxy-NPV objective (gated; default "revenue" preserves current behavior) -----
    # When objective=="npv", the cma_surrogate fitness wraps the surrogate's predicted
    # per-geology discounted revenue with the deviated-well CAPEX (geothermal.well_geometry),
    # a surface-flowline cost between the fixed facilities, surface-facility CAPEX, and
    # discounted OPEX. All economic numbers come from economics_config_path.
    objective: str = "revenue"                 # "revenue" | "npv"
    # PRE-0 distinct-K guard (cma_surrogate only): the hardcoded min_xy_dist=4 dedup
    # SILENTLY backfills near-duplicate layouts when the (converged) pool can't supply
    # n_exploit distinct picks. distinct_k_realized is ALWAYS logged; when this flag is
    # True the acquisition HARD-FAILS (pre-IX) if distinct < ceil(0.9*n_exploit) so
    # duplicate layouts never ship as wasted IX. Default False preserves prior behavior.
    assert_distinct_k: bool = False
    economics_config_path: Path | None = None
    geo_cube_path: Path | None = None          # h5 with CX/CY/CZ structural coord cube (k,j,i)
    facilities: tuple = ((20, 30), (40, 40))   # fixed surface facility (i, j) locations
    vertical_lead_m: float = 1000.0
    ksurf: int = 2
    poro_thresh: float = 0.01


# Canonical ordering for snapshots in the aggregate manifest so parallel runs
# produce byte-identical output to serial runs.
_KIND_ORDER = {"frontier": 0, "adversarial": 1, "exploit": 2, "cma": 3}

# Maps the provenance of a start ("lhs"/"elite"/"cma") to the kind tag used at
# the k_safe snapshot step. Adversarial snapshots at k_adv are emitted ONLY for
# LHS-provenance starts (exploit/cma deep-Adam endpoints would muddy semantics).
_START_KIND_TO_SAFE_KIND = {"lhs": "frontier", "elite": "exploit", "cma": "cma"}


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


# Raw (un-normalized) full grids consumed per-well by extract_well_data /
# extract_vertical_profiles. Preloading these once per geology (fix #2) lets the
# CMA-over-surrogate build loop avoid re-reading them from the H5 for every
# candidate; the dict is keyed by the dataset names those functions index.
_WELL_DATA_GRID_NAMES = (
    "Input/PermX",
    "Input/PermY",
    "Input/PermZ",
    "Input/Porosity",
    "Input/Temperature0",
    "Input/Pressure0",
)


def _load_geology(
    geology_h5_file: Path,
    norm_config: dict,
    PROPERTIES,
    PERM_PROPS,
    get_valid_mask,
    find_z_cutoff,
    *,
    preload_well_grids: bool = False,
) -> dict[str, Any]:
    """Load one geology file and produce the static physics tensors + bounds.

    When ``preload_well_grids`` is True, also reads the full raw ``Input/*``
    grids that ``extract_well_data`` / ``extract_vertical_profiles`` consume,
    returned under ``"well_grids"`` (fix #2: read once per geology, reuse across
    every candidate). Default False keeps the legacy return shape unchanged.
    """
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

        well_grids: dict[str, np.ndarray] | None = None
        if preload_well_grids:
            well_grids = {name: src[name][:] for name in _WELL_DATA_GRID_NAMES}

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
        "well_grids": well_grids,
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
    device: torch.device | str | None = None,
    preloaded_grids: dict[str, np.ndarray] | None = None,
):
    """Build a list of normalized HeteroData objects, one per start.

    Speedups (both numerically identical to the default path):

    * ``device`` (fix #1, physics-on-GPU): when given, the geology's shared
      ``physics_dict`` tensors are moved to ``device`` ONCE here (they are the
      same object reference across every candidate's ``physics_context``, see
      ``data.build_single_hetero_data``). The model's per-graph
      ``volume.to(device)`` in ``PhysicsSlabExtractor.forward`` then becomes a
      no-op view instead of re-copying the CPU physics volume to the GPU on
      every forward of every generation. Caller is responsible for ``.to(device)``
      on the assembled ``Batch`` as before; this only relocates the physics
      wrappers PyG does not track.
    * ``preloaded_grids`` (fix #2): passed straight through to
      ``extract_well_data`` / ``extract_vertical_profiles`` so the full raw H5
      grids are read once per geology by the caller instead of re-read from
      ``src`` for every candidate. ``None`` reproduces the legacy per-candidate
      H5 reads exactly.
    """
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
            ) = extract_well_data(is_well, inj_rate, src, preloaded_grids=preloaded_grids)
            wells = build_wells_table(
                x_idx, y_idx, depth, inj, perm_x, perm_y, perm_z,
                porosity, temp0, press0,
            )
            vertical_profiles = extract_vertical_profiles(
                is_well, x_idx, y_idx, src, preloaded_grids=preloaded_grids
            )
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
            n_wells_actual = int(raw_graph["well"].x.shape[0])
            n_wells_expected = len(w_cfg)
            if n_wells_actual != n_wells_expected:
                raise RuntimeError(
                    f"Static batch graph {m_idx}: built {n_wells_actual} wells, "
                    f"expected {n_wells_expected}. Likely cause: a well was "
                    f"placed on a fully-invalid temp column and extract_well_data "
                    f"dropped it. The LHS sampler should already resample "
                    f"dead-rock placements; this should not happen."
                )
            static_graphs.append(scaler.transform_graph(raw_graph))

    # Fix #1 (physics-on-GPU): relocate the physics volumes to ``device`` ONCE,
    # AFTER graph construction. build_single_hetero_data reads physics tensors on
    # the host (e.g. valid_mask.numpy() for enrich_global_attr), so the move must
    # happen post-build. Every candidate's PhysicsContext wraps the SAME physics
    # dict object (by reference, preserved through scaler.transform_graph), so a
    # single moved dict reassigned onto each context covers the whole geology
    # batch. The model's per-graph volume.to(device) in PhysicsSlabExtractor then
    # no-ops on an already-on-device tensor instead of re-copying CPU->GPU on
    # every forward. Numerically identical: a device move does not alter values.
    if device is not None and static_graphs:
        moved: dict[int, dict] = {}
        for g in static_graphs:
            pc = g.physics_context
            key = id(pc.d)
            if key not in moved:
                moved[key] = {k: v.to(device) for k, v in pc.d.items()}
            pc.d = moved[key]
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
    # ----- proxy-NPV state (only populated when objective=="npv") -----
    objective: str = "revenue"
    npv_terms: dict | None = None
    geo_cube: dict | None = None
    fac_surf_xy: Any = None              # (F, 2) facility surface coords (m)
    flowline_between_m: float = 0.0      # constant surface flowline length between facilities
    vertical_lead_m: float = 1000.0
    ksurf: int = 2
    poro_thresh: float = 0.01
    # well_geometry callables (late-resolved from the surrogate repo)
    compute_angled_well_length: Any = None
    compute_npv: Any = None
    reservoir_top_k_map: Any = None


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

    # ----- proxy-NPV state (geology-independent parts loaded once) -----
    objective = str(getattr(cfg, "objective", "revenue"))
    npv_terms = None
    geo_cube = None
    fac_surf_xy = None
    flowline_between_m = 0.0
    compute_angled_well_length = compute_npv = reservoir_top_k_map = None
    if objective == "npv":
        from geothermal.well_geometry import (  # type: ignore
            load_geo_coord_cube,
            reservoir_top_k_map,
            facilities_surface_xy,
            surface_flowline_length,
            compute_angled_well_length,
            compute_npv,
            load_npv_terms,
        )
        if not cfg.economics_config_path or not Path(cfg.economics_config_path).exists():
            raise FileNotFoundError(
                f"objective='npv' requires economics_config_path; got {cfg.economics_config_path}"
            )
        if not cfg.geo_cube_path or not Path(cfg.geo_cube_path).exists():
            raise FileNotFoundError(
                f"objective='npv' requires geo_cube_path (CX/CY/CZ h5); got {cfg.geo_cube_path}"
            )
        npv_terms = load_npv_terms(str(cfg.economics_config_path))
        geo_cube = load_geo_coord_cube(cfg.geo_cube_path)
        fac_surf_xy = facilities_surface_xy(geo_cube, cfg.facilities, ksurf=int(cfg.ksurf))
        flowline_between_m = surface_flowline_length(fac_surf_xy)

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
        objective=objective,
        npv_terms=npv_terms,
        geo_cube=geo_cube,
        fac_surf_xy=fac_surf_xy,
        flowline_between_m=flowline_between_m,
        vertical_lead_m=float(cfg.vertical_lead_m),
        ksurf=int(cfg.ksurf),
        poro_thresh=float(cfg.poro_thresh),
        compute_angled_well_length=compute_angled_well_length,
        compute_npv=compute_npv,
        reservoir_top_k_map=reservoir_top_k_map,
    )


def _per_geology_seed(base_seed: int, geology_index: int) -> int:
    """Deterministic per-geology seed. Decoupling each geology from the global
    sampler state lets us partition geologies across worker processes without
    changing numerical results.
    """
    # Mask to 64 bits so numpy/scipy seed APIs accept it on all platforms.
    return ((base_seed * 1_000_003) ^ (geology_index * 1009)) & ((1 << 63) - 1)


def _load_elite_seeds(
    geology_index: int,
    prior_metrics: list[Path],
    k: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Pull top-K real-revenue configs for ``geology_index`` across prior iters.

    Returns at most ``k`` coord arrays of shape ``(num_wells, 3)``, ranked by
    descending real INTERSECT revenue. Cross-references each candidate's
    ``snapshot_id`` to the corresponding ``snapshots_json/<id>.json`` to recover
    the (x, y, z) well coordinates. Silently skips entries with non-finite real
    revenue or missing snapshot JSONs. Enforces a minimum pairwise L2 distance
    (≥3 grid cells in the flattened coord vector) to avoid collapsed picks.
    """
    if not prior_metrics:
        return []

    entries: list[tuple[float, str, list[Path]]] = []  # (real_rev, snapshot_id, candidate_dirs)
    for metrics_path in prior_metrics:
        metrics_path = Path(metrics_path)
        if not metrics_path.exists():
            continue
        # The snapshots_json dir lives in one of two layouts depending on the
        # driver. We try both and use whichever exists at lookup time:
        #   * Sherlock (orchestrator/paths.py): <run_root>/iter_NNNN/acquire/snapshots_json/
        #   * Local hybrid (scripts/run_al_local.py): <run_root>/acquire/iter_NNNN/snapshots_json/
        iter_name = metrics_path.parent.name  # "iter_0004"
        run_root = metrics_path.parent.parent
        candidate_dirs = [
            metrics_path.parent / "acquire" / "snapshots_json",          # Sherlock layout
            run_root / "acquire" / iter_name / "snapshots_json",         # Local-hybrid layout
        ]
        try:
            with open(metrics_path, "r") as f:
                payload = json.load(f)
        except Exception:
            continue
        for c in payload.get("candidates", []):
            if int(c.get("geology_index", -1)) != geology_index:
                continue
            real = c.get("real_revenue")
            if real is None or not np.isfinite(real):
                continue
            snap_id = c.get("snapshot_id")
            if not snap_id:
                continue
            entries.append((float(real), str(snap_id), candidate_dirs))

    if not entries:
        return []

    # Rank descending by real_revenue, then walk in order applying a min-distance filter.
    entries.sort(key=lambda t: -t[0])
    picked: list[np.ndarray] = []
    picked_flat: list[np.ndarray] = []
    min_dist = 3.0  # grid cells (in flattened coord space)
    for real_rev, snap_id, candidate_dirs in entries:
        if len(picked) >= k:
            break
        snap_path = None
        for sdir in candidate_dirs:
            cand = sdir / f"{snap_id}.json"
            if cand.exists():
                snap_path = cand
                break
        if snap_path is None:
            continue
        try:
            with open(snap_path, "r") as f:
                snap = json.load(f)
        except Exception:
            continue
        wells = snap.get("wells", [])
        if not wells:
            continue
        coords = np.asarray(
            [[float(w["x"]), float(w["y"]), float(w["z"])] for w in wells],
            dtype=np.float32,
        )
        flat = coords.reshape(-1)
        if picked_flat:
            dists = np.linalg.norm(np.stack(picked_flat) - flat[None, :], axis=1)
            if float(dists.min()) < min_dist:
                continue
        picked.append(coords)
        picked_flat.append(flat)

    # Light shuffle to avoid pulling the same elite to start 0 every iter.
    rng.shuffle(picked)
    return picked[:k]


def _cma_seed_starts(
    n_seeds: int,
    cfg: AcquisitionConfig,
    geo,  # GeologySpec
    num_wells: int,
    wells_spec,  # list[WellSpec]
    nx: int,
    ny: int,
    z_max: int,
    static_graphs_factory,  # callable(cfgs) -> static_graphs, lazy because we don't want to rebuild from scratch
    model,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    device: torch.device,
    seed_geo: int,
    elite_seeds: list[np.ndarray],
    valid_xy_indices: np.ndarray,
    well_xy_valid_fn,
) -> list[np.ndarray]:
    """Run a short CMA-ES warm-up using the surrogate as fitness (forward-only)
    and return the top-``n_seeds`` coordinate arrays from the final population.

    These coords subsequently feed into the main Adam loop as initial starts
    (tagged kind="cma"), so CMA-ES is acting as a smarter sampler than LHS — it
    naturally concentrates around high-fitness regions of the surrogate while
    maintaining covariance-controlled exploration.

    Bounds: same edge-buffered window as LHS. If a CMA sample falls on dead rock
    or claims a duplicate cell, it's projected to the nearest valid cell from
    ``valid_xy_indices`` rather than rejected — keeps the population the
    requested size without inflating compute.
    """
    if n_seeds <= 0:
        return []
    try:
        from cmaes import CMA  # type: ignore
    except ImportError:
        print("[acquire] cmaes not installed; falling back to LHS for cma kind. "
              "Install via `pip install cmaes`.", flush=True)
        return []

    dim = num_wells * 2  # (x, y) per well; depth held fixed per WellSpec
    x_lo, x_hi = float(cfg.edge_buffer), float(nx - 1 - cfg.edge_buffer)
    y_lo, y_hi = float(cfg.edge_buffer), float(ny - 1 - cfg.edge_buffer)
    bounds = np.array(
        [[x_lo, x_hi], [y_lo, y_hi]] * num_wells, dtype=np.float64
    )

    # Initial mean: centroid of elites if available, else LHS-window center.
    if elite_seeds:
        elite_stack = np.stack([e[:, :2].reshape(-1) for e in elite_seeds], axis=0)
        mean_init = elite_stack.mean(axis=0).astype(np.float64)
    else:
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        mean_init = np.tile([cx, cy], num_wells).astype(np.float64)
    mean_init = np.clip(mean_init, bounds[:, 0] + 1e-3, bounds[:, 1] - 1e-3)

    es = CMA(
        mean=mean_init,
        sigma=float(cfg.cma_sigma_init),
        bounds=bounds,
        seed=int(seed_geo & 0xFFFFFFFF),
        population_size=int(cfg.cma_popsize),
    )

    # We need to evaluate popsize candidates per generation through the surrogate.
    # Build a static graph batch once for ``popsize`` slots with placeholder coords;
    # then reuse, mutating only well positions on each forward.
    #
    # CRITICAL: every well in every placeholder must land on a DISTINCT valid
    # (x, y) cell. ``extract_well_data`` dedups by (x_idx, y_idx), so if any
    # two wells share a cell the resulting graph drops one and the batched
    # forward expects fewer wells than we supply. Picking unique cells from
    # ``valid_xy_indices`` guarantees a 12-well topology; the exact positions
    # don't matter because each CMA-ES generation overwrites
    # ``batch_data["well"].pos_xyz`` with the sampled coords.
    if valid_xy_indices.shape[0] < num_wells:
        return []
    stride = max(1, valid_xy_indices.shape[0] // num_wells)
    placeholder_cells = valid_xy_indices[::stride][:num_wells]
    if placeholder_cells.shape[0] < num_wells:
        placeholder_cells = valid_xy_indices[:num_wells]
    placeholder_cfgs: list[list[dict]] = []
    for _ in range(int(cfg.cma_popsize)):
        c = []
        for w in range(num_wells):
            c.append({
                "x": float(placeholder_cells[w, 0]),
                "y": float(placeholder_cells[w, 1]),
                "depth": int(min(wells_spec[w].depth, z_max)),
                "type": wells_spec[w].type,
            })
        placeholder_cfgs.append(c)
    try:
        static_graphs = static_graphs_factory(placeholder_cfgs)
    except Exception as e:
        print(f"[acquire] CMA-ES warm-up bailout (graph build failed): {e}", flush=True)
        return []
    batch_data = Batch.from_data_list(static_graphs).to(device)
    M = int(cfg.cma_popsize)

    best_population: list[tuple[np.ndarray, float]] = []  # (coords (num_wells,3), pred)
    z_per_well = np.array(
        [float(min(wells_spec[w].depth, z_max)) for w in range(num_wells)],
        dtype=np.float32,
    )
    for _gen in range(int(cfg.cma_generations)):
        sols = np.stack([es.ask() for _ in range(M)], axis=0)  # (M, dim)
        # Project to valid cells (dead-rock + edge buffer already enforced via bounds).
        coords_xy = sols.reshape(M, num_wells, 2)
        # Build (M, num_wells, 3) by attaching fixed z per well.
        coords_xyz = np.concatenate(
            [coords_xy, np.broadcast_to(z_per_well[None, :, None], (M, num_wells, 1))],
            axis=-1,
        ).astype(np.float32)
        # Snap to nearest valid (x,y) cell if any well lands on dead rock.
        for i in range(M):
            used: set[tuple[int, int]] = set()
            for w in range(num_wells):
                rx, ry = float(coords_xyz[i, w, 0]), float(coords_xyz[i, w, 1])
                cell = (int(np.clip(round(rx), 0, nx - 1)), int(np.clip(round(ry), 0, ny - 1)))
                if (not well_xy_valid_fn(rx, ry, int(z_per_well[w]))) or cell in used:
                    pick = valid_xy_indices[
                        np.random.default_rng(seed_geo + _gen * 7919 + i * 31 + w).integers(
                            0, len(valid_xy_indices)
                        )
                    ]
                    rx, ry = float(pick[0]), float(pick[1])
                    cell = (int(round(rx)), int(round(ry)))
                used.add(cell)
                coords_xyz[i, w, 0] = rx
                coords_xyz[i, w, 1] = ry
        c_t = torch.from_numpy(coords_xyz).to(device)
        with torch.no_grad():
            batch_data["well"].pos_xyz = c_t.view(-1, 3)
            pred_scaled = model(batch_data).view(M)
            preds = (pred_scaled * target_scale + target_mean).detach().cpu().numpy()
        # Sanitize non-finite predictions before handing to CMA-ES and before
        # ranking. cmaes minimizes; we give NaN samples the worst possible
        # finite fitness (+inf maps to a huge sentinel) so they're rejected
        # by the rank-based update without contaminating sigma. Without this,
        # Python's `sort(key=lambda t: -t[1])` puts NaN at the *top* of
        # best_population (NaN comparisons return False → sort treats it as
        # not-less-than anything), corrupting the elite pool fed back into
        # the main Adam loop.
        finite_mask = np.isfinite(preds)
        WORST_SENTINEL = 1e30  # large finite penalty for the minimizer
        cma_fitness = np.where(finite_mask, -preds, WORST_SENTINEL).astype(np.float64)
        es.tell(list(zip([s for s in sols], cma_fitness.tolist())))
        for i in range(M):
            if not finite_mask[i]:
                continue
            best_population.append((coords_xyz[i].copy(), float(preds[i])))

    # Return top n_seeds across all generations by predicted revenue (high → low).
    best_population.sort(key=lambda t: -t[1])
    return [c for c, _p in best_population[:n_seeds]]


def _run_one_geology(
    *,
    cfg: AcquisitionConfig,
    geo: GeologySpec,
    iteration: int,
    ctx: _WorkerContext,
    well_configs_dir: Path,
    snapshots_json_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Run gradient-descent acquisition for a single geology on ``ctx.device``.

    Returns (snapshot_records, candidate_payloads, timings). Candidate payloads
    are plain dicts so they survive the IPC boundary; the parent converts them
    back to ``Candidate`` instances via ``_payload_to_candidate``. ``timings``
    is a dict of {stage_name: seconds} for the GPU/CPU profiling CSV.
    """
    timings: dict[str, float] = {}
    t_total_start = time.perf_counter()

    device = ctx.device
    model = ctx.model
    target_mean = ctx.target_mean
    target_scale = ctx.target_scale
    is_injector_list = ctx.is_injector_list
    num_wells = ctx.num_wells

    geo_path = Path(geo.geology_h5_file)
    geo_name = geo.geology_name or geo_path.stem

    t0 = time.perf_counter()
    with h5py.File(geo_path, "r") as src:
        geo_meta = _read_geology_metadata_safe(ctx.read_geology_metadata, src, geo_name)
    loaded = _load_geology(
        geo_path, ctx.norm_config,
        ctx.PROPERTIES, ctx.PERM_PROPS, ctx.get_valid_mask, ctx.find_z_cutoff,
    )
    timings["geo_load_s"] = time.perf_counter() - t0

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

    # Precompute valid-XY columns within the edge buffer for resampling wells
    # whose LHS placement happens to land on dead rock (entire Z column has
    # temp0 <= -900). Without this filter, extract_well_data would drop the
    # well entirely (well_mask = is_well == 1 is empty for that column), the
    # static batch ends up with < M*num_wells wells, and the CNN node encoder
    # crashes downstream with a stale well_batch / pos_xyz shape mismatch.
    has_valid_z = np.any(temp0_full > -900, axis=0)  # (X, Y) bool
    ix_lo, ix_hi = int(np.ceil(x_lo)), int(np.floor(x_hi))
    iy_lo, iy_hi = int(np.ceil(y_lo)), int(np.floor(y_hi))
    edge_window = np.zeros_like(has_valid_z, dtype=bool)
    edge_window[ix_lo : ix_hi + 1, iy_lo : iy_hi + 1] = True
    valid_xy = has_valid_z & edge_window
    valid_xy_indices = np.argwhere(valid_xy)
    if valid_xy_indices.shape[0] < num_wells:
        raise RuntimeError(
            f"Geology {geo.geology_index}: only {valid_xy_indices.shape[0]} valid "
            f"(x,y) columns inside the edge buffer {cfg.edge_buffer}; need at "
            f"least {num_wells} to place all wells."
        )

    def _well_xy_valid(rx_: float, ry_: float, depth_: int) -> bool:
        ix_ = int(np.clip(int(round(rx_)), 0, nx - 1))
        iy_ = int(np.clip(int(round(ry_)), 0, ny - 1))
        return bool(np.any(temp0_full[: max(1, depth_), ix_, iy_] > -900))

    def _build_cfg_from_coords(coords_arr: np.ndarray) -> list[dict]:
        """coords_arr: (num_wells, 3). Returns the list-of-dicts the static
        graph builder expects."""
        return [
            {
                "x": float(coords_arr[w, 0]),
                "y": float(coords_arr[w, 1]),
                "depth": int(min(cfg.wells[w].depth, z_max)),
                "type": cfg.wells[w].type,
            }
            for w in range(num_wells)
        ]

    def _project_to_valid(coords_arr: np.ndarray, rng_local: np.random.Generator) -> np.ndarray:
        """Snap any invalid or duplicate-cell well to a valid cell drawn from
        ``valid_xy_indices``. Operates in-place but also returns the array.
        """
        used_cells: set[tuple[int, int]] = set()
        for w in range(num_wells):
            rx, ry = float(coords_arr[w, 0]), float(coords_arr[w, 1])
            depth = int(min(cfg.wells[w].depth, z_max))
            tries = 0
            while True:
                cell = (
                    int(np.clip(int(round(rx)), 0, nx - 1)),
                    int(np.clip(int(round(ry)), 0, ny - 1)),
                )
                if _well_xy_valid(rx, ry, depth) and cell not in used_cells:
                    break
                if tries >= 200:
                    raise RuntimeError(
                        f"Geology {geo.geology_index}: failed to project well {w} "
                        f"to a valid unused cell after {tries} tries"
                    )
                pick = valid_xy_indices[int(rng_local.integers(0, len(valid_xy_indices)))]
                rx = float(pick[0])
                ry = float(pick[1])
                tries += 1
            used_cells.add(cell)
            coords_arr[w, 0] = rx
            coords_arr[w, 1] = ry
        return coords_arr

    def _lhs_one_start() -> np.ndarray:
        """Single LHS-validated start, shape (num_wells, 3)."""
        s = sampler.random(n=1)[0]
        coords_arr = np.zeros((num_wells, 3), dtype=np.float32)
        for w in range(num_wells):
            coords_arr[w, 0] = x_lo + s[2 * w] * (x_hi - x_lo)
            coords_arr[w, 1] = y_lo + s[2 * w + 1] * (y_hi - y_lo)
            coords_arr[w, 2] = int(min(cfg.wells[w].depth, int(z_max)))
        return _project_to_valid(coords_arr, rng)

    # ------------------------------------------------------------------
    # Generate starts (3 provenances: lhs, elite, cma)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    cfgs: list[list[dict]] = []
    start_kinds: list[str] = []

    # 1. LHS starts (always present; produce frontier+adversarial snapshots)
    for _ in range(int(cfg.n_starts_per_geology)):
        coords_arr = _lhs_one_start()
        cfgs.append(_build_cfg_from_coords(coords_arr))
        start_kinds.append("lhs")

    # 2. Elite starts (only if requested + priors available)
    elite_seeds_coords: list[np.ndarray] = []
    if int(cfg.n_elite_per_geology) > 0 and cfg.prior_metrics:
        elite_seeds_coords = _load_elite_seeds(
            geology_index=geo.geology_index,
            prior_metrics=list(cfg.prior_metrics or []),
            k=int(cfg.elite_top_k),
            rng=rng,
        )
    for i in range(int(cfg.n_elite_per_geology)):
        if elite_seeds_coords:
            base = elite_seeds_coords[i % len(elite_seeds_coords)].copy()
            noise = rng.normal(0.0, float(cfg.elite_seed_noise), size=base.shape).astype(np.float32)
            noise[:, 2] = 0.0  # never perturb z
            coords_arr = (base + noise).astype(np.float32)
            coords_arr[:, 0] = np.clip(coords_arr[:, 0], x_lo, x_hi)
            coords_arr[:, 1] = np.clip(coords_arr[:, 1], y_lo, y_hi)
            for w in range(num_wells):
                coords_arr[w, 2] = int(min(cfg.wells[w].depth, int(z_max)))
            coords_arr = _project_to_valid(coords_arr, rng)
        else:
            # Cold start (iter 0): fall back to LHS, still tagged "elite" so the
            # schema is uniform from iter 0.
            coords_arr = _lhs_one_start()
        cfgs.append(_build_cfg_from_coords(coords_arr))
        start_kinds.append("elite")

    # 3. CMA starts (only if requested)
    cma_seeds_coords: list[np.ndarray] = []
    if int(cfg.n_cma_per_geology) > 0:
        def _cma_factory(cfgs_inner: list[list[dict]]):
            return _build_static_batch_for_starts(
                cfgs_inner, geo_path, physics_dict, full_shape, z_cutoff, nx, ny,
                cfg.revenue_target, ctx.scaler,
                ctx.extract_well_data, ctx.build_wells_table, ctx.extract_vertical_profiles,
                ctx.build_single_hetero_data, temp0_full,
                node_encoder=ctx.node_encoder, enrich_global_attr=ctx.enrich_global_attr,
            )
        try:
            cma_seeds_coords = _cma_seed_starts(
                n_seeds=int(cfg.n_cma_per_geology), cfg=cfg, geo=geo,
                num_wells=num_wells, wells_spec=cfg.wells,
                nx=nx, ny=ny, z_max=z_max,
                static_graphs_factory=_cma_factory,
                model=model, target_mean=target_mean, target_scale=target_scale,
                device=device, seed_geo=seed_geo,
                elite_seeds=elite_seeds_coords,
                valid_xy_indices=valid_xy_indices, well_xy_valid_fn=_well_xy_valid,
            )
        except Exception as e:
            print(f"[acquire] CMA-ES warm-up failed for geo={geo.geology_index}: {e}; "
                  f"falling back to LHS for cma kind", flush=True)
            cma_seeds_coords = []
    # Top up with LHS if CMA returned fewer than requested (or wasn't installed).
    while len(cma_seeds_coords) < int(cfg.n_cma_per_geology):
        cma_seeds_coords.append(_lhs_one_start())
    for coords_arr in cma_seeds_coords[: int(cfg.n_cma_per_geology)]:
        cfgs.append(_build_cfg_from_coords(_project_to_valid(coords_arr.copy(), rng)))
        start_kinds.append("cma")

    timings["seed_gen_s"] = time.perf_counter() - t0
    timings["n_lhs"] = int(start_kinds.count("lhs"))
    timings["n_elite"] = int(start_kinds.count("elite"))
    timings["n_cma"] = int(start_kinds.count("cma"))

    # ------------------------------------------------------------------
    # Build static graph batch for ALL starts at once
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    static_graphs = _build_static_batch_for_starts(
        cfgs, geo_path, physics_dict, full_shape, z_cutoff, nx, ny,
        cfg.revenue_target, ctx.scaler,
        ctx.extract_well_data, ctx.build_wells_table, ctx.extract_vertical_profiles,
        ctx.build_single_hetero_data, temp0_full,
        node_encoder=ctx.node_encoder,
        enrich_global_attr=ctx.enrich_global_attr,
    )
    batch_data = Batch.from_data_list(static_graphs).to(device)
    timings["graph_build_s"] = time.perf_counter() - t0

    starts = [
        [[w["x"], w["y"], float(w["depth"])] for w in c] for c in cfgs
    ]
    coords = torch.tensor(starts, dtype=torch.float32, device=device)
    coords.requires_grad = True
    optimizer = optim.Adam([coords], lr=cfg.learning_rate)
    M = len(cfgs)
    last_valid_coords = coords.detach().clone()

    def _predict_unscaled(c: torch.Tensor) -> torch.Tensor:
        batch_data["well"].pos_xyz = c.view(-1, 3)
        pred_scaled = model(batch_data).view(M)
        return pred_scaled * target_scale + target_mean

    # Adversarial subset: random subset of LHS-provenance starts only.
    lhs_indices = [i for i, k in enumerate(start_kinds) if k == "lhs"]
    n_lhs = len(lhs_indices)
    n_adv = max(0, int(round(n_lhs * cfg.adv_fraction)))
    if n_adv > 0 and lhs_indices:
        adv_idx = set(
            int(i) for i in rng.choice(lhs_indices, size=min(n_adv, n_lhs), replace=False)
        )
    else:
        adv_idx = set()

    snapshots: list[dict[str, Any]] = []
    cand_payloads: list[dict[str, Any]] = []

    t0 = time.perf_counter()
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
                # Proactively scrub Adam moments when grads were non-finite —
                # otherwise exp_avg / exp_avg_sq can still carry NaN/Inf from
                # this step's bad backward and poison every subsequent
                # optimizer.step(). Mirrors the ensemble-mode guard.
                st = optimizer.state.get(coords, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st and not torch.isfinite(st[key]).all():
                        torch.nan_to_num_(st[key], nan=0.0, posinf=0.0, neginf=0.0)
            for d, max_val in enumerate([nx - 1, ny - 1, z_max - 1]):
                mask_lo = (coords[:, :, d] <= 1e-4) & (grads[:, :, d] > 0)
                mask_hi = (coords[:, :, d] >= max_val - 1e-4) & (grads[:, :, d] < 0)
                mask = mask_lo | mask_hi
                if mask.any():
                    grads[..., d][mask] = 0.0
                    st = optimizer.state.get(coords, {})
                    if "exp_avg" in st:
                        st["exp_avg"][..., d][mask] = 0.0
                    # Also zero exp_avg_sq at the boundary slice — leaving a
                    # stale variance there keeps the Adam denominator skewed
                    # for ~10 steps after the coord pins to the wall.
                    if "exp_avg_sq" in st:
                        st["exp_avg_sq"][..., d][mask] = 0.0
        optimizer.step()
        with torch.no_grad():
            if not torch.isfinite(coords).all():
                bad = ~torch.isfinite(coords)
                coords[bad] = last_valid_coords[bad]
                st = optimizer.state.get(coords, {})
                # Two scrub modes layered together:
                #   1. Zero Adam moments AT the recovered positions: huge-
                #      but-finite moments would pass isfinite() and then
                #      catapult the recovered coord straight back across the
                #      domain on the next step.
                #   2. Sanitize any wholly-NaN moments (legacy behavior).
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st:
                        st[key][bad] = 0.0
                        if not torch.isfinite(st[key]).all():
                            torch.nan_to_num_(st[key], nan=0.0, posinf=0.0, neginf=0.0)
            coords[:, :, 0].clamp_(0, nx - 1)
            coords[:, :, 1].clamp_(0, ny - 1)
            coords[:, :, 2].clamp_(0, z_max - 1)
            last_valid_coords.copy_(coords)

        if step == cfg.k_safe:
            t_snap = time.perf_counter()
            with torch.no_grad():
                preds_now = _predict_unscaled(coords).detach().cpu().numpy()
            for m in range(M):
                cxyz = coords[m].detach().cpu().numpy()
                pred_val = float(preds_now[m])
                if not (np.isfinite(cxyz).all() and np.isfinite(pred_val)):
                    print(f"[acquire] skip non-finite k_safe snapshot geo={geo.geology_index} m={m} kind={start_kinds[m]}", flush=True)
                    continue
                safe_kind = _START_KIND_TO_SAFE_KIND[start_kinds[m]]
                snap = _emit_snapshot(
                    run_id=geo.geology_index * 10000 + m,
                    iteration_step=step,
                    kind=safe_kind,
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
            timings["snapshot_io_safe_s"] = timings.get("snapshot_io_safe_s", 0.0) + (time.perf_counter() - t_snap)

        if step == cfg.k_adv and adv_idx:
            t_snap = time.perf_counter()
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
            timings["snapshot_io_adv_s"] = timings.get("snapshot_io_adv_s", 0.0) + (time.perf_counter() - t_snap)

    timings["adam_loop_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_total_start
    return snapshots, cand_payloads, timings


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
        # Cap intra-op thread parallelism so 2 workers × default(=ncores)
        # threads doesn't oversubscribe a single physical CPU. The parent set
        # OMP/MKL/OPENBLAS env vars before spawn (see _run_acquisition_multi_gpu)
        # so numpy/BLAS pick them up at import; this call belts-and-suspenders
        # PyTorch's intra-op pool, which is set independently from the env vars.
        omp = int(os.environ.get("OMP_NUM_THREADS", "1") or "1")
        torch.set_num_threads(max(1, omp))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already set elsewhere; ignore
        # Pin this process to the requested device so any incidental CUDA
        # ops (e.g. inside torch_geometric) land on the right GPU.
        if device_str.startswith("cuda:") and torch.cuda.is_available():
            torch.cuda.set_device(int(device_str.split(":", 1)[1]))
        t_ctx = time.perf_counter()
        ctx = _build_worker_context(cfg, iteration, device_str)
        ctx_load_s = time.perf_counter() - t_ctx
        wc_dir = Path(well_configs_dir)
        sj_dir = Path(snapshots_json_dir)
    except BaseException as e:  # propagate setup errors to parent
        out_queue.put(("__init_error__", repr(e)))
        return

    first_geo = True
    while True:
        geo = in_queue.get()
        if geo is None:
            return
        try:
            snaps, cand_payloads, timings = _run_one_geology(
                cfg=cfg,
                geo=geo,
                iteration=iteration,
                ctx=ctx,
                well_configs_dir=wc_dir,
                snapshots_json_dir=sj_dir,
            )
            # Attribute the one-time worker context load cost to the first
            # geology processed by this worker. Matches the serial path's
            # treatment so the CSV's worker_ctx_load_s column is symmetric.
            if first_geo:
                timings["worker_ctx_load_s"] = ctx_load_s
                first_geo = False
            out_queue.put((device_str, geo.geology_index, snaps, cand_payloads, timings))
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
) -> tuple[list[dict[str, Any]], list[Candidate], list[dict[str, Any]]]:
    import multiprocessing
    import torch.multiprocessing as tmp

    n_workers = len(devices)
    # Cap each worker's intra-op CPU thread pool so 2 workers × N-core default
    # doesn't oversubscribe physical cores. Without this, htop shows 200%-300%
    # CPU per worker on bend_gpu and wallclock plateaus regardless of GPU count.
    # Leave 1 core for the parent + OS; divide the rest evenly. Floor at 1.
    total_cores = multiprocessing.cpu_count()
    threads_per_worker = max(1, (total_cores - 1) // max(1, n_workers))
    # These env vars must be set BEFORE spawning so the child inherits them
    # at numpy/torch import time (the BLAS libraries snapshot the env once).
    _thread_env_keys = (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    _prior_env: dict[str, str | None] = {}
    for k in _thread_env_keys:
        _prior_env[k] = os.environ.get(k)
        # Don't override an explicit user-set value; users tuning by hand win.
        if _prior_env[k] is None:
            os.environ[k] = str(threads_per_worker)

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
    print(
        f"[acquire] multi-GPU dispatch: n_workers={n_workers}, "
        f"threads_per_worker={threads_per_worker} (total_cores={total_cores})",
        flush=True,
    )

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
    timing_rows: list[dict[str, Any]] = []
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
            _dev, _geo_idx, snaps, cand_payloads, timings = msg
            snapshots.extend(snaps)
            for payload in cand_payloads:
                candidates.append(_payload_to_candidate(payload))
            row = {"device": _dev, "geology_index": _geo_idx}
            row.update(timings)
            timing_rows.append(row)
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
        # Restore thread-count env vars to whatever they were before. This
        # matters because later phases (training, plotting) run in the same
        # process and may want the original default thread budget.
        for k, v in _prior_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    if errors:
        raise RuntimeError("Multi-GPU acquisition failures:\n  " + "\n  ".join(errors))

    return snapshots, candidates, timing_rows


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
    # Foot-gun guard: only the cma_surrogate path implements the NPV objective. The Adam
    # ensemble / per_geology / multi-GPU paths silently score+rank on revenue; pairing them with
    # objective="npv" would optimize revenue while the dashboard relabels everything NPV. Fail loud.
    if getattr(cfg, "objective", "revenue") == "npv" and getattr(cfg, "mode", "per_geology") != "cma_surrogate":
        raise ValueError(
            f"objective='npv' is only implemented for mode='cma_surrogate'; got mode="
            f"'{getattr(cfg, 'mode', 'per_geology')}'. Set mode='cma_surrogate' or objective='revenue'."
        )
    if getattr(cfg, "mode", "per_geology") == "cma_surrogate":
        return _run_acquisition_cma_surrogate(
            cfg, out_dir=out_dir, iteration=iteration, run_id_prefix=run_id_prefix,
        )
    if getattr(cfg, "mode", "per_geology") == "ensemble":
        return _run_acquisition_ensemble(
            cfg, out_dir=out_dir, iteration=iteration, run_id_prefix=run_id_prefix,
        )

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
    timing_rows: list[dict[str, Any]] = []
    started = time.time()

    if len(devices) <= 1:
        # In-process path. Load model once, iterate geologies serially.
        device_str = devices[0]
        t_ctx = time.perf_counter()
        ctx = _build_worker_context(cfg, iteration, device_str)
        ctx_load_s = time.perf_counter() - t_ctx
        for i, geo in enumerate(cfg.geologies):
            snaps, cand_payloads, timings = _run_one_geology(
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
            row = {"device": device_str, "geology_index": geo.geology_index}
            row.update(timings)
            if i == 0:
                row["worker_ctx_load_s"] = ctx_load_s
            timing_rows.append(row)
    else:
        # Multi-GPU path. Spawn one worker per device; geologies are pulled off
        # a shared queue. Each worker loads model+scaler once per process.
        # Order of completion is not deterministic, so we sort the merged
        # snapshot list at the end to keep the manifest reproducible.
        snapshots, candidates, timing_rows = _run_acquisition_multi_gpu(
            cfg=cfg,
            iteration=iteration,
            devices=devices,
            well_configs_dir=well_configs_dir,
            snapshots_json_dir=snapshots_json_dir,
        )
        snapshots.sort(key=_snapshot_sort_key)
        candidates.sort(key=lambda c: (c.geology_index, _KIND_ORDER.get(c.kind, 99), c.run_id))

    # Write per-stage profiling CSV for the GPU-bottleneck investigation.
    if timing_rows:
        profiling_dir = out_dir / "profiling"
        profiling_dir.mkdir(parents=True, exist_ok=True)
        csv_path = profiling_dir / f"gpu_timings_iter_{iteration:04d}.csv"
        all_keys = sorted({k for row in timing_rows for k in row.keys()})
        # Stable column order: device, geology_index first; then alphabetical.
        col_order = ["device", "geology_index"] + [k for k in all_keys if k not in ("device", "geology_index")]
        with open(csv_path, "w") as f:
            f.write(",".join(col_order) + "\n")
            for row in timing_rows:
                f.write(",".join(str(row.get(k, "")) for k in col_order) + "\n")

    # Write aggregate manifest matching cli_surrogate_array_prepare.jl schema.
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "run_id_prefix": run_id_prefix,
        "k_safe": cfg.k_safe,
        "k_adv": cfg.k_adv,
        "adv_fraction": cfg.adv_fraction,
        "n_starts_per_geology": cfg.n_starts_per_geology,
        "n_elite_per_geology": cfg.n_elite_per_geology,
        "n_cma_per_geology": cfg.n_cma_per_geology,
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


# ============================================================================
# Ensemble (EMV) acquisition: one candidate × all K geologies per Adam loop.
# ============================================================================

def _load_elite_seeds_ensemble(
    prior_metrics: list[Path],
    k: int,
    rng: np.random.Generator,
    objective: str = "revenue",
) -> list[tuple[np.ndarray, str, int]]:
    """Pull top-K real-EMV configs across prior iters (ensemble mode).

    Returns at most ``k`` triples of ``(coords, source_snapshot_id, source_iteration)``,
    ranked by descending per-candidate real EMV (mean over the candidate's K geology runs).
    In npv mode the ranking value is real NPV (per-row ``real_npv``, written into prior
    per_candidate_metrics by the local driver's npv augmentation) so the warm-start incumbent is
    the best-NPV config, not the best-revenue one. In revenue mode it's real_revenue, read from
    the pre-aggregated ``per_candidate_emv`` when present.

    Coordinates come from each candidate's ``snapshots_json/<id>.json`` file.

    Coordinates come from each candidate's ``snapshots_json/<id>.json`` file.
    Enforces a minimum pairwise L2 distance (>=3 grid cells in flattened coord
    space) to avoid collapsed picks.
    """
    if not prior_metrics:
        return []

    # entries: (emv, snapshot_id, source_iteration, candidate_dirs)
    entries: list[tuple[float, str, int, list[Path]]] = []
    for metrics_path in prior_metrics:
        metrics_path = Path(metrics_path)
        if not metrics_path.exists():
            continue
        # Same dual-layout dance as the per-geology elite reader.
        iter_name = metrics_path.parent.name  # "iter_0004"
        run_root = metrics_path.parent.parent
        candidate_dirs = [
            metrics_path.parent / "acquire" / "snapshots_json",
            run_root / "acquire" / iter_name / "snapshots_json",
        ]
        try:
            with open(metrics_path, "r") as f:
                payload = json.load(f)
        except Exception:
            continue
        iter_idx = int(payload.get("iteration", -1))
        use_npv = (objective == "npv")
        # Prefer the pre-aggregated per_candidate_emv (REVENUE) only in revenue mode. In npv mode
        # it is the wrong quantity, so always aggregate from per-row real_npv instead.
        per_cand_emv = {} if use_npv else (payload.get("per_candidate_emv") or {})
        if not isinstance(per_cand_emv, dict):
            per_cand_emv = {}
        # Otherwise, aggregate on the fly from the candidates list. Only
        # snapshots that fanned out to >1 IX run are real EMVs; per-geology
        # iterations have one row per snapshot whose "EMV" would just be that
        # single geology's value, which is meaningless for ensemble seeding.
        if not per_cand_emv:
            grouped: dict[str, list[float]] = {}
            for c in payload.get("candidates", []):
                sid = c.get("snapshot_id")
                val = c.get("real_npv") if use_npv else c.get("real_revenue")
                if not sid or val is None or not np.isfinite(val):
                    continue
                grouped.setdefault(str(sid), []).append(float(val))
            per_cand_emv = {
                sid: float(np.mean(vals))
                for sid, vals in grouped.items()
                if len(vals) > 1
            }
        for sid, emv in per_cand_emv.items():
            if emv is None or not np.isfinite(emv):
                continue
            entries.append((float(emv), str(sid), iter_idx, candidate_dirs))

    if not entries:
        return []

    entries.sort(key=lambda t: -t[0])
    picked: list[tuple[np.ndarray, str, int]] = []
    picked_flat: list[np.ndarray] = []
    min_dist = 3.0
    for emv, snap_id, src_iter, candidate_dirs in entries:
        if len(picked) >= k:
            break
        snap_path = None
        for sdir in candidate_dirs:
            cand = sdir / f"{snap_id}.json"
            if cand.exists():
                snap_path = cand
                break
        if snap_path is None:
            continue
        try:
            with open(snap_path, "r") as f:
                snap = json.load(f)
        except Exception:
            continue
        wells = snap.get("wells", [])
        if not wells:
            continue
        coords = np.asarray(
            [[float(w["x"]), float(w["y"]), float(w["z"])] for w in wells],
            dtype=np.float32,
        )
        flat = coords.reshape(-1)
        if picked_flat:
            dists = np.linalg.norm(np.stack(picked_flat) - flat[None, :], axis=1)
            if float(dists.min()) < min_dist:
                continue
        picked.append((coords, snap_id, src_iter))
        picked_flat.append(flat)

    # Light shuffle so the same elite doesn't always seed candidate 0.
    order = rng.permutation(len(picked))
    picked = [picked[int(i)] for i in order]
    return picked[:k]


def _emit_snapshot_ensemble(
    *,
    run_id: int,
    iteration_step: int,
    kind: str,
    coords_xyz: np.ndarray,
    is_injector_list: list[bool],
    predictions_by_geology: list[dict],
    predicted_emv: float,
    seed_source_snapshot_id: str | None,
    seed_source_iteration: int | None,
    seed_predicted_emv: float | None,
    geologies: list[GeologySpec],
    geology_metas: dict[int, dict],
    well_configs_dir: Path,
    snapshots_json_dir: Path,
    to_julia_wells_text,
    extra_fields: dict | None = None,
) -> dict[str, Any]:
    """Write one .jl + one snapshot JSON for an ensemble candidate.

    The .jl carries the predicted EMV as its score; per-geology predictions live
    inside the snapshot JSON's ``predictions_by_geology`` array. The manifest
    record's ``well_config_paths_by_geology`` lists K entries pointing to the
    same .jl, so Julia's array prepare expands one candidate into K IX tasks.
    """
    snapshot_id = f"run{run_id:06d}_step{iteration_step:04d}_{kind}"
    jl_path = well_configs_dir / f"{snapshot_id}.jl"
    json_path = snapshots_json_dir / f"{snapshot_id}.json"

    # The .jl file is geology-agnostic in ensemble mode; we still record one
    # geology's metadata in it so existing downstream tools that read the header
    # have something to log. Use the first geology by convention.
    first = geologies[0]
    first_meta = geology_metas.get(first.geology_index, {})
    jl_text = to_julia_wells_text(
        coords_xyz=coords_xyz,
        is_injector_list=is_injector_list,
        score=predicted_emv,
        score_label="Predicted EMV (mean discounted revenue across ensemble)",
        geology_file=str(first.geology_h5_file),
        geology_name=first.geology_name or Path(first.geology_h5_file).stem,
        geology_config_id=first_meta.get("geology_config_id"),
        geology_scenario_name=first_meta.get("scenario_name"),
        geology_sample_num=first_meta.get("sample_num"),
        predicted_discounted_revenue=predicted_emv,
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
        "mode": "ensemble",
        "predicted_emv": float(predicted_emv),
        "predicted_discounted_total_revenue": float(predicted_emv),  # back-compat alias
        "seed_source_snapshot_id": seed_source_snapshot_id,
        "seed_source_iteration": (None if seed_source_iteration is None else int(seed_source_iteration)),
        "seed_predicted_emv": (None if seed_predicted_emv is None else float(seed_predicted_emv)),
        "terminal_predicted_emv": float(predicted_emv),
        "wells": wells_json,
        "predictions_by_geology": predictions_by_geology,
    }
    if extra_fields:
        snap_payload.update(extra_fields)
    with open(json_path, "w") as f:
        json.dump(snap_payload, f, indent=2)

    well_config_paths_by_geology = []
    for g in geologies:
        meta = geology_metas.get(g.geology_index, {})
        # The corresponding predicted revenue for this geology (for the manifest).
        per_geo_pred = next(
            (p for p in predictions_by_geology if int(p.get("geology_index", -1)) == g.geology_index),
            None,
        )
        well_config_paths_by_geology.append({
            "geology_index": g.geology_index,
            "geology_name": g.geology_name or Path(g.geology_h5_file).stem,
            "geology_file": str(Path(g.geology_h5_file).resolve()),
            "geology_config_id": meta.get("geology_config_id"),
            "geology_scenario_name": meta.get("scenario_name"),
            "geology_sample_num": meta.get("sample_num"),
            "well_config_path": str(jl_path),
            "predicted_discounted_total_revenue": float(per_geo_pred["discounted_total_revenue"]) if per_geo_pred else float("nan"),
        })

    snapshot_record = {
        "snapshot_id": snapshot_id,
        "run_id": int(run_id),
        "iteration": int(iteration_step),
        "kind": kind,
        "mode": "ensemble",
        "json_path": str(json_path),
        "well_config_path": str(jl_path),
        "well_config_paths_by_geology": well_config_paths_by_geology,
        "predicted_emv": float(predicted_emv),
        "predicted_discounted_total_revenue": float(predicted_emv),
        "seed_source_snapshot_id": seed_source_snapshot_id,
        "seed_source_iteration": (None if seed_source_iteration is None else int(seed_source_iteration)),
        "seed_predicted_emv": (None if seed_predicted_emv is None else float(seed_predicted_emv)),
    }
    if extra_fields:
        snapshot_record.update(extra_fields)
    return snapshot_record


def _snapshot_to_candidate_ensemble(
    snap: dict, coords_xyz: np.ndarray, is_injector_list: list[bool]
) -> Candidate:
    """Build a Candidate from an ensemble snapshot record.

    The Candidate dataclass was designed for per-geology selection. In ensemble
    mode each snapshot covers all geologies, so we point the candidate's
    geology fields at the *first* entry in ``well_config_paths_by_geology`` to
    keep downstream code (stage.py, select.py back-compat) happy; the manifest
    writer for ensemble mode preserves the full K-element array.
    """
    geo_entry = snap["well_config_paths_by_geology"][0]
    extras = {
        "mode": "ensemble",
        "well_config_paths_by_geology": snap["well_config_paths_by_geology"],
        "predicted_emv": float(snap.get("predicted_emv", snap.get("predicted_discounted_total_revenue", 0.0))),
        "seed_source_snapshot_id": snap.get("seed_source_snapshot_id"),
        "seed_source_iteration": snap.get("seed_source_iteration"),
        "seed_predicted_emv": snap.get("seed_predicted_emv"),
        "terminal_predicted_emv": snap.get("terminal_predicted_emv", snap.get("predicted_emv")),
    }
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
        predicted_revenue=float(snap.get("predicted_emv", snap.get("predicted_discounted_total_revenue", 0.0))),
        coords_xyz=np.asarray(coords_xyz, dtype=np.float32),
        is_injector=list(is_injector_list),
        well_config_path=str(snap["well_config_path"]),
        snapshot_json_path=str(snap.get("json_path", "")),
        extras=extras,
    )


def write_selected_manifest_ensemble(
    selected: list[Candidate],
    *,
    out_path: Path,
    iteration: int,
    geologies: list[GeologySpec],
    extras: dict | None = None,
) -> Path:
    """Manifest writer for ensemble candidates: each candidate fans out to K
    IX tasks via the full ``well_config_paths_by_geology`` array stashed in
    ``c.extras``.
    """
    snapshots = []
    for c in selected:
        wcp_by_geo = c.extras.get("well_config_paths_by_geology") if c.extras else None
        if not wcp_by_geo:
            # Fall back to a single-entry per the candidate's recorded geology.
            wcp_by_geo = [
                {
                    "geology_index": c.geology_index,
                    "geology_name": c.geology_name,
                    "geology_file": c.geology_file,
                    "geology_config_id": c.geology_config_id,
                    "geology_scenario_name": c.geology_scenario_name,
                    "geology_sample_num": c.geology_sample_num,
                    "well_config_path": c.well_config_path,
                }
            ]
        snapshots.append({
            "snapshot_id": c.snapshot_id,
            "run_id": int(c.run_id),
            "iteration": int(c.iteration),
            "kind": c.kind,
            "mode": "ensemble",
            "json_path": c.snapshot_json_path,
            "well_config_path": c.well_config_path,
            "well_config_paths_by_geology": wcp_by_geo,
            "predicted_discounted_total_revenue": float(c.predicted_revenue),
            "predicted_emv": float(c.extras.get("predicted_emv", c.predicted_revenue)) if c.extras else float(c.predicted_revenue),
            "seed_source_snapshot_id": (c.extras or {}).get("seed_source_snapshot_id"),
            "seed_source_iteration": (c.extras or {}).get("seed_source_iteration"),
            "seed_predicted_emv": (c.extras or {}).get("seed_predicted_emv"),
        })

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "mode": "ensemble",
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


def _dedup_indices_by_xy(
    coords_list: list[np.ndarray],
    order: np.ndarray,
    num_wells: int,
    k: int,
    min_xy_dist: float,
) -> list[int]:
    """Greedily pick up to ``k`` indices from ``coords_list`` (walked in
    ``order``), skipping any candidate whose well (x, y) layout is within
    ``min_xy_dist`` (L2 over the flattened per-well xy vector) of an already
    picked one.

    Depth (z) is deliberately EXCLUDED from the distance: two configs that
    differ only in depth are the same well *placement*, and mixing z-layer units
    with xy-cell units in one L2 lets depth noise mask near-duplicate layouts.
    If fewer than ``k`` distinct layouts survive, backfills from ``order``
    (ignoring the filter) so the shortlist is never silently short.
    """
    picked: list[int] = []
    picked_xy: list[np.ndarray] = []
    for idx in order:
        if len(picked) >= k:
            break
        i = int(idx)
        xy = np.asarray(coords_list[i])[:, :2].reshape(-1)
        if picked_xy:
            d = np.linalg.norm(np.stack(picked_xy) - xy[None, :], axis=1)
            if float(d.min()) < min_xy_dist:
                continue
        picked.append(i)
        picked_xy.append(xy)
    if len(picked) < k:
        chosen = set(picked)
        for idx in order:
            if len(picked) >= k:
                break
            i = int(idx)
            if i in chosen:
                continue
            picked.append(i)
            chosen.add(i)
    return picked


def _count_distinct_xy(
    coords_list: list[np.ndarray],
    indices: list[int],
    num_wells: int,
    min_xy_dist: float,
) -> int:
    """Count how many of ``indices`` have mutually-distinct xy layouts (greedy, L2 over
    the flattened per-well xy vector, >= ``min_xy_dist`` apart) — i.e. the number of picks
    that are NOT near-duplicate backfill from :func:`_dedup_indices_by_xy`. Used by the
    PRE-0 distinct-K guard so silent backfill (which inflates the apparent batch with
    duplicate layouts = wasted IX) is observable and, optionally, fatal pre-IX.
    """
    kept_xy: list[np.ndarray] = []
    for i in indices:
        xy = np.asarray(coords_list[int(i)])[:, :2].reshape(-1)
        if kept_xy:
            d = np.linalg.norm(np.stack(kept_xy) - xy[None, :], axis=1)
            if float(d.min()) < min_xy_dist:
                continue
        kept_xy.append(xy)
    return len(kept_xy)


def _run_acquisition_cma_surrogate(
    cfg: AcquisitionConfig,
    *,
    out_dir: Path,
    iteration: int,
    run_id_prefix: str = "al",
) -> dict[str, Any]:
    """CMA-ES-over-surrogate acquisition (ensemble flavor).

    Runs CMA-ES locally over the surrogate to pick the batch of well
    configurations to query INTERSECT for — a gradient-free alternative to the
    Adam ensemble path that does not chase the surrogate's (anti-aligned, see
    analysis/diag1_adam_destruction) local gradient. Each candidate is one full
    well configuration scored as the mean predicted revenue across ALL K
    geologies, so it ships K IX tasks, exactly like ``_run_acquisition_ensemble``.
    Output shape is identical to that path; Julia staging keys off
    ``well_config_paths_by_geology``, not the manifest mode, so downstream is
    unchanged.

    Decision vector is per-well (x, y, z); depth z is searched for parity with
    the IX baseline. Unlike the Adam paths — which are forced to build the graph
    once and only overwrite ``pos_xyz`` (backprop needs a differentiable path) —
    this gradient-free search REBUILDS an accurate graph for every queried
    candidate, so the node-level features, perforation depth and KNN topology
    track the actual well positions. Node embeddings dominate the surrogate's
    accuracy, so keeping them correct at every query is worth the ~2x cost over
    a static-``pos_xyz``-overwrite proxy. Because every query is accurate, search
    ranking, selection and the reported ``predicted_emv`` are all the same
    number — there is no separate re-rank stage. (The surrogate's depth response
    is still weak — it was trained over a narrow band — so z is explored more
    than truly optimized; that's fine, it yields depth-varying IX labels.)

    The search engine is reused from the IX baseline
    (``baseline_optimizer.build_optimizer``), driven with a local surrogate
    fitness closure instead of real IX runs. Single-device, forward-only: this
    skips the multi-GPU replica machinery (and its peer-copy hazard) the Adam
    ensemble path needs — though per-query rebuild makes geology-parallelism
    across GPUs the obvious next speedup if iteration wall-clock matters.
    """
    from .baseline_optimizer import build_optimizer, project_to_valid_cells
    from .select import _farthest_point_select

    out_dir = Path(out_dir)
    well_configs_dir = out_dir / "well_configs"
    snapshots_json_dir = out_dir / "snapshots_json"
    well_configs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_json_dir.mkdir(parents=True, exist_ok=True)

    device_strs = list(cfg.devices) if cfg.devices else [cfg.device]
    started = time.time()
    timings: dict[str, float] = {}

    t_ctx = time.perf_counter()
    ctx = _build_worker_context(cfg, iteration, device_strs[0])
    timings["worker_ctx_load_s"] = time.perf_counter() - t_ctx
    device = ctx.device
    model = ctx.model
    target_mean = ctx.target_mean
    target_scale = ctx.target_scale
    is_injector_list = ctx.is_injector_list
    num_wells = ctx.num_wells
    # Defensive: fake worker contexts in tests may omit the npv fields.
    objective = getattr(ctx, "objective", "revenue")

    geologies = list(cfg.geologies)
    K = len(geologies)
    if K == 0:
        raise RuntimeError("cma_surrogate acquisition requires at least one geology.")

    # ----- Load every geology's static physics tensors + metadata -----
    t0 = time.perf_counter()
    geology_metas: dict[int, dict] = {}
    geology_loaded: list[dict[str, Any]] = []
    for g in geologies:
        gp = Path(g.geology_h5_file)
        with h5py.File(gp, "r") as src:
            geology_metas[g.geology_index] = _read_geology_metadata_safe(
                ctx.read_geology_metadata, src, g.geology_name or gp.stem
            )
        d = _load_geology(
            gp, ctx.norm_config,
            ctx.PROPERTIES, ctx.PERM_PROPS, ctx.get_valid_mask, ctx.find_z_cutoff,
            # Fix #2: read the raw well-data grids ONCE per geology here so the
            # per-generation per-candidate graph rebuild below reuses them
            # instead of re-reading the full H5 grids for every CMA candidate.
            preload_well_grids=True,
        )
        if objective == "npv":
            # Per-geology reservoir-top map from the already-preloaded raw porosity
            # (geothermal.well_geometry.reservoir_top_k_map). Indexed [j-1, i-1].
            d["reservoir_top_k"] = ctx.reservoir_top_k_map(
                d["well_grids"]["Input/Porosity"], poro_thresh=ctx.poro_thresh
            )
        geology_loaded.append(d)
    timings["geo_load_s"] = time.perf_counter() - t0

    # If every geology shares the same reservoir-top map (the common case — the
    # porous/overburden stratigraphy is structural, while geologies differ in
    # perm/temp), the deviated well length is geology-independent. Detect this once
    # so the NPV fitness computes each candidate's well length ONCE instead of K
    # times, and the emitted cost breakdown reconciles exactly with predicted NPV.
    shared_rtop = None
    if objective == "npv":
        rtops = [d["reservoir_top_k"] for d in geology_loaded]
        if all(np.array_equal(rtops[0], rt) for rt in rtops[1:]):
            shared_rtop = rtops[0]

    nx = min(d["nx"] for d in geology_loaded)
    ny = min(d["ny"] for d in geology_loaded)
    z_max = min(d["z_max"] for d in geology_loaded)
    x_lo, x_hi = float(cfg.edge_buffer), float(nx - 1 - cfg.edge_buffer)
    y_lo, y_hi = float(cfg.edge_buffer), float(ny - 1 - cfg.edge_buffer)

    # Columns valid (live rock + inside edge buffer) across ALL geologies.
    ix_lo, ix_hi = int(np.ceil(x_lo)), int(np.floor(x_hi))
    iy_lo, iy_hi = int(np.ceil(y_lo)), int(np.floor(y_hi))
    has_valid_z_all = None
    for d in geology_loaded:
        hv = np.any(d["temp0_full"][:z_max] > -900, axis=0)
        has_valid_z_all = hv if has_valid_z_all is None else (has_valid_z_all & hv)
    edge_window = np.zeros_like(has_valid_z_all, dtype=bool)
    edge_window[ix_lo:ix_hi + 1, iy_lo:iy_hi + 1] = True
    valid_xy = has_valid_z_all & edge_window
    valid_xy_indices = np.argwhere(valid_xy)
    if valid_xy_indices.shape[0] < num_wells:
        raise RuntimeError(
            f"cma_surrogate: only {valid_xy_indices.shape[0]} (x,y) columns are "
            f"valid across all {K} geologies inside edge buffer {cfg.edge_buffer}; "
            f"need >= {num_wells}."
        )

    # Depth bounds: cap z_hi at the reservoir grid extent so emitted/forwarded
    # depths stay inside every geology's active grid.
    z_hi_eff = int(min(int(cfg.depth_max), int(z_max) - 1))
    z_lo_eff = int(min(int(cfg.depth_min), z_hi_eff - 1))
    if z_hi_eff <= z_lo_eff:
        raise RuntimeError(
            f"cma_surrogate: degenerate depth bounds (z_lo={z_lo_eff}, z_hi={z_hi_eff}) "
            f"from depth_min={cfg.depth_min}, depth_max={cfg.depth_max}, z_max={z_max}."
        )

    rng = np.random.default_rng(_per_geology_seed(ctx.base_seed, 0))

    # ----- Build ACCURATE per-candidate graphs for one (x,y,z) batch: one
    # static-graph batch per geology. Called fresh for every CMA query so the
    # node-level features, perforation depth and KNN topology track the actual
    # well positions (the dominant accuracy driver), not just pos_xyz. -----
    def _build_static_graphs_one_geo(cfgs: list[list[dict]], g, d) -> list[Any]:
        return _build_static_batch_for_starts(
            cfgs, Path(g.geology_h5_file), d["physics_dict"], d["full_shape"],
            d["z_cutoff"], d["nx"], d["ny"],
            cfg.revenue_target, ctx.scaler,
            ctx.extract_well_data, ctx.build_wells_table, ctx.extract_vertical_profiles,
            ctx.build_single_hetero_data, d["temp0_full"],
            node_encoder=ctx.node_encoder, enrich_global_attr=ctx.enrich_global_attr,
            # Fix #1: physics volumes onto the GPU once per geology (the model's
            # per-graph volume.to(device) then no-ops). Fix #2: reuse the grids
            # preloaded above instead of re-reading the H5 per candidate.
            device=device, preloaded_grids=d.get("well_grids"),
        )

    # Count candidate-evals dropped because a well snapped onto a dead-rock column
    # at its CMA-chosen (possibly shallow) depth: valid_xy only guarantees live
    # rock somewhere in [0, z_max), but extract_well_data needs live rock in the
    # well's perforation interval [0, depth) and drops the well otherwise (→ wrong
    # well count → _build_static_batch_for_starts raises). Now that depth is
    # optimized, this can happen; such candidates are scored NaN (CMA ranks them
    # worst), NOT crashed on — the failed-eval contract used by the AL pipeline
    # and validated in analysis/cma_trajectory_long.
    drop_stats = {"n_dropped": 0, "n_gens_with_drops": 0}

    def _build_batches_with_mask(cfgs: list[list[dict]]) -> tuple[list[Any], np.ndarray]:
        """Build per-geology batches over candidates that build in ALL geologies,
        returning (batches_over_valid_subset, valid_mask). FAST PATH (common
        case): one batched build per geology, identical to before. Only if a
        batched build raises do we per-candidate probe THAT geology to find and
        mask the offender(s)."""
        M_local = len(cfgs)
        ok = np.ones(M_local, dtype=bool)
        per_geo_graphs: list[list[Any] | None] = []
        for g, d in zip(geologies, geology_loaded):
            try:
                per_geo_graphs.append(_build_static_graphs_one_geo(cfgs, g, d))
            except RuntimeError:
                per_geo_graphs.append(None)  # rebuild over the valid subset below
                for i in range(M_local):
                    if not ok[i]:
                        continue
                    try:
                        _build_static_graphs_one_geo([cfgs[i]], g, d)
                    except RuntimeError:
                        ok[i] = False
        valid_idx = np.where(ok)[0]
        cfgs_valid = [cfgs[i] for i in valid_idx]
        batches = []
        for (g, d), graphs in zip(zip(geologies, geology_loaded), per_geo_graphs):
            if graphs is not None and ok.all():
                static_graphs = graphs  # reuse the full-population build (fast path)
            else:
                static_graphs = _build_static_graphs_one_geo(cfgs_valid, g, d)
            batches.append(Batch.from_data_list(static_graphs).to(device))
        return batches, ok

    def _predict_all_geos(coords_xyz_np: np.ndarray, batches: list[Any]) -> np.ndarray:
        """Return (M, K) unscaled predictions; M must match the batch slot count.

        The surrogate-grid z is clamped to [0, z_max-1] for the forward (the
        emitted/queried depth keeps its full CMA value up to z_hi_eff)."""
        M_local = coords_xyz_np.shape[0]
        c = coords_xyz_np.astype(np.float32).copy()
        c[:, :, 0] = np.clip(c[:, :, 0], 0, nx - 1)
        c[:, :, 1] = np.clip(c[:, :, 1], 0, ny - 1)
        c[:, :, 2] = np.clip(c[:, :, 2], 0, z_max - 1)
        c_t = torch.from_numpy(c).to(device)
        cols: list[np.ndarray] = []
        with torch.no_grad():
            for bd in batches:
                bd["well"].pos_xyz = c_t.view(-1, 3)
                pred_scaled = model(bd).view(M_local)
                preds = (pred_scaled * target_scale + target_mean).detach().cpu().numpy()
                cols.append(preds)
        return np.stack(cols, axis=1)  # (M, K)

    # ----- CMA-ES over the surrogate, rebuilding an ACCURATE graph for EVERY
    # queried candidate. Per-query rebuild (vs the static pos_xyz overwrite the
    # Adam path is forced into) keeps the node-level embeddings correct — they
    # dominate the surrogate's accuracy — at ~2x the per-eval cost. Since every
    # query is accurate, search ranking == selection == reported prediction; no
    # separate re-rank stage is needed. -----
    popsize = int(cfg.cma_popsize)
    n_exploit = int(cfg.n_exploit)
    n_frontier = int(cfg.n_frontier)
    if n_exploit + n_frontier == 0:
        raise RuntimeError("cma_surrogate: n_exploit + n_frontier must be > 0.")

    def _coords_to_cfgs(coords_arr: np.ndarray) -> list[list[dict]]:
        return [
            [
                {"x": float(c[w, 0]), "y": float(c[w, 1]),
                 "depth": int(min(int(round(float(c[w, 2]))), z_max - 1)),
                 "type": cfg.wells[w].type}
                for w in range(num_wells)
            ]
            for c in coords_arr
        ]

    def _predict_rebuilt(coords_arr: np.ndarray) -> np.ndarray:
        """Build accurate per-candidate graphs; return (M, K) unscaled per-geology
        predictions. Candidates whose graph can't be built in some geology
        (dead-rock at their depth) get a full-NaN row, so CMA treats them as
        failed evals (np.nanmean -> NaN -> ranked worst, filtered from the pool)."""
        cfgs = _coords_to_cfgs(coords_arr)
        M_local = coords_arr.shape[0]
        out = np.full((M_local, K), np.nan, dtype=np.float64)
        batches, mask = _build_batches_with_mask(cfgs)
        n_drop = int((~mask).sum())
        if n_drop:
            drop_stats["n_dropped"] += n_drop
            drop_stats["n_gens_with_drops"] += 1
        if not mask.any():
            return out  # whole population invalid -> all NaN
        valid_idx = np.where(mask)[0]
        out[valid_idx] = _predict_all_geos(coords_arr[valid_idx], batches)
        return out

    # ----- Optional: seed CMA mean from the best real-revenue incumbent --------
    # Gated on cfg.cma_warm_start (default False): the cma_trajectory_long IX
    # experiment showed warm-starting from the incumbent over-exploits the
    # surrogate's exploit regime, so we default to a cold LHS restart each
    # iteration (mean_init stays None).
    mean_init = None
    seed_incumbent_id: str | None = None
    seed_incumbent_iter: int | None = None
    if cfg.cma_warm_start and iteration >= 1 and cfg.prior_metrics:
        elite_triples = _load_elite_seeds_ensemble(
            prior_metrics=list(cfg.prior_metrics or []), k=1, rng=rng, objective=objective,
        )
        # Make a silent degrade-to-cold visible: in npv mode the incumbent ranking needs the
        # prior metrics to carry real_npv (written by the local dashboard augmentation). If that
        # didn't happen (augmentation failed/skipped), elite_triples is empty and warm-start
        # quietly becomes a cold restart — log it rather than hide it.
        if not elite_triples and objective == "npv":
            print("[acquire-cma] WARNING: cma_warm_start requested but no NPV-ranked incumbent "
                  "found in prior metrics (real_npv missing — augmentation may have been skipped); "
                  "falling back to a COLD restart.", flush=True)
        # Guard: only seed if the incumbent has the same well count as this run.
        if elite_triples and np.asarray(elite_triples[0][0]).shape[0] == num_wells:
            base, seed_incumbent_id, seed_incumbent_iter = elite_triples[0]
            base = np.asarray(base, dtype=np.float64).copy()  # (num_wells, 3)
            # Anchor xy on the incumbent (projected onto a valid cell so the
            # anchor isn't silently resampled away at ask() time); start z at
            # mid-range so depth is broadly explored rather than pinned.
            base[:, :2] = project_to_valid_cells(
                base[None, :, :2].astype(np.float32), valid_xy_indices,
                num_wells, rng, nx=nx, ny=ny,
            )[0].astype(np.float64)
            base[:, 2] = 0.5 * (z_lo_eff + z_hi_eff)
            mean_init = base.reshape(-1)

    opt = build_optimizer(
        "cmaes",
        num_wells=num_wells, nx=nx, ny=ny, edge_buffer=int(cfg.edge_buffer),
        popsize=popsize, seed=int(cfg.seed) + int(iteration),
        valid_xy_indices=valid_xy_indices,
        depth_bounds=(z_lo_eff, z_hi_eff),
        sigma_init=float(cfg.cma_sigma_init),
        mean_init=mean_init,
    )

    def _candidate_scores(coords_np: np.ndarray, preds: np.ndarray):
        """Map (popsize, K) revenue preds -> (score (popsize,), npv_rows).

        revenue mode: score = ensemble-mean predicted revenue; npv_rows is None.
        npv mode: per-geology NPV_k = revenue_k - deviated-well CAPEX(coords, geo_k) -
        surface-flowline CAPEX - surface-facility CAPEX - discounted OPEX (all from
        economics.json); score = ensemble-mean NPV; npv_rows is (popsize, K). The cost
        terms are geology-dependent only via each geology's reservoir-top map. Rows with
        no finite revenue stay NaN (CMA ranks them worst), matching the revenue path.
        """
        M = preds.shape[0]
        score = np.full(M, np.nan, dtype=np.float64)
        finite = np.isfinite(preds).any(axis=1)
        if objective != "npv":
            if finite.any():
                score[finite] = np.nanmean(preds[finite], axis=1)
            return score, None
        npv_rows = np.full((M, K), np.nan, dtype=np.float64)
        for r in np.where(finite)[0]:
            wl_shared = None
            if shared_rtop is not None:
                wl_shared = ctx.compute_angled_well_length(
                    coords_np[r], cube=ctx.geo_cube, fac_surf_xy=ctx.fac_surf_xy,
                    reservoir_top_k_map=shared_rtop,
                    vertical_lead_m=ctx.vertical_lead_m, ksurf=ctx.ksurf,
                )
            for k_idx, d in enumerate(geology_loaded):
                rev = preds[r, k_idx]
                if not np.isfinite(rev):
                    continue
                wl = wl_shared if wl_shared is not None else ctx.compute_angled_well_length(
                    coords_np[r], cube=ctx.geo_cube, fac_surf_xy=ctx.fac_surf_xy,
                    reservoir_top_k_map=d["reservoir_top_k"],
                    vertical_lead_m=ctx.vertical_lead_m, ksurf=ctx.ksurf,
                )
                npv_rows[r, k_idx] = ctx.compute_npv(
                    float(rev), wl, is_injector_list,
                    flowline_between_m=ctx.flowline_between_m, npv_terms=ctx.npv_terms,
                )["npv"]
        nfinite = np.isfinite(npv_rows).any(axis=1)
        if nfinite.any():
            score[nfinite] = np.nanmean(npv_rows[nfinite], axis=1)
        return score, npv_rows

    # Pool keeps each finite candidate's coords AND its accurate per-geology
    # prediction row, so the emit step reuses them with no recompute. In npv mode
    # pool_npv carries the per-geology NPV row (None in revenue mode).
    pool_coords: list[np.ndarray] = []   # each (num_wells, 3)
    pool_preds: list[np.ndarray] = []    # each (K,) accurate per-geology revenue row
    pool_emv: list[float] = []           # the candidate SCORE (NPV in npv mode, else revenue)
    pool_npv: list[Any] = []             # each (K,) per-geology NPV row, or None
    t0 = time.perf_counter()
    for _gen in range(int(cfg.cma_generations)):
        coords = opt.ask()  # (popsize, num_wells, 3), projected to valid cells
        preds = _predict_rebuilt(coords)  # (popsize, K) — ACCURATE graphs
        emv, npv_rows = _candidate_scores(coords, preds)
        opt.tell(coords, emv)
        for i in range(popsize):
            if np.isfinite(emv[i]) and np.isfinite(preds[i]).all():
                pool_coords.append(coords[i].astype(np.float32).copy())
                pool_preds.append(preds[i].astype(np.float32).copy())
                pool_emv.append(float(emv[i]))
                pool_npv.append(npv_rows[i].astype(np.float32).copy() if npv_rows is not None else None)
    timings["cma_loop_s"] = time.perf_counter() - t0
    timings["cma_generations"] = float(cfg.cma_generations)
    timings["cma_popsize"] = float(popsize)
    timings["graph_builds"] = float(int(cfg.cma_generations) * popsize * K)
    timings["pool_size"] = float(len(pool_coords))
    if drop_stats["n_dropped"]:
        print(
            f"[acquire-cma] {drop_stats['n_dropped']} candidate-evals scored NaN "
            f"(well on dead-rock at its CMA-chosen depth) across "
            f"{drop_stats['n_gens_with_drops']} generations — ranked worst by CMA, "
            f"excluded from the pool (not a crash).",
            flush=True,
        )
    if not pool_coords:
        raise RuntimeError(
            "cma_surrogate: CMA-ES produced no finite-fitness candidates — the "
            "surrogate returned non-finite predictions for every sample."
        )

    # ----- Exploit picks: top accurate-EMV, de-duplicated by xy layout --------
    order = np.argsort(-np.asarray(pool_emv))
    exploit_idx = _dedup_indices_by_xy(pool_coords, order, num_wells, n_exploit, min_xy_dist=4.0)
    if len(exploit_idx) < n_exploit:
        print(
            f"[acquire-cma] WARNING: only {len(exploit_idx)} exploit candidates available "
            f"(requested {n_exploit}); pool too small (pool={len(pool_coords)}, "
            f"generations={cfg.cma_generations}, popsize={popsize}).",
            flush=True,
        )

    # ----- PRE-0 distinct-K guard ------------------------------------------------
    # The min_xy_dist=4 dedup above backfills near-duplicate layouts when the pool
    # can't supply n_exploit distinct picks; the len()<n_exploit warning never fires
    # (the pool is large), so duplicates would ship as wasted IX (15 redundant runs
    # each) AND corrupt the distinct-K x-axis of the sample-efficiency study. Always
    # log the truly-distinct count; HARD-FAIL pre-IX (no IX wasted) when the config
    # opts in via assert_distinct_k. Threshold: >= ceil(0.9 * n_exploit).
    distinct_k_realized = _count_distinct_xy(pool_coords, exploit_idx, num_wells, 4.0)
    distinct_k_floor = int(np.ceil(0.9 * n_exploit))
    timings["distinct_k_realized"] = float(distinct_k_realized)
    timings["n_exploit_requested"] = float(n_exploit)
    timings["distinct_k_floor"] = float(distinct_k_floor)
    print(
        f"[acquire-cma] distinct-K: {distinct_k_realized}/{n_exploit} emitted exploit "
        f"layouts are >=4-apart distinct (floor {distinct_k_floor}); "
        f"{n_exploit - distinct_k_realized} backfilled near-duplicate(s).",
        flush=True,
    )
    if getattr(cfg, "assert_distinct_k", False) and distinct_k_realized < distinct_k_floor:
        raise RuntimeError(
            f"cma_surrogate distinct-K guard: only {distinct_k_realized} of {n_exploit} "
            f"emitted exploit layouts are distinct (>= {distinct_k_floor} required); the "
            f"min_xy_dist=4 dedup backfilled {n_exploit - distinct_k_realized} near-duplicate(s) "
            f"(pool={len(pool_coords)}, popsize={popsize}, generations={cfg.cma_generations}). "
            f"Raise cma_popsize/cma_generations (more pre-convergence spread), split the batch "
            f"across more cold-restart iterations, or implement CMA multi-restart. Failing "
            f"pre-IX so no IX is wasted."
        )

    # ----- Frontier picks: LHS + FPS on xy, scored on accurate rebuilt graphs --
    frontier_coords: list[np.ndarray] = []
    frontier_preds: list[np.ndarray] = []
    frontier_npv: list[Any] = []
    if n_frontier > 0:
        pool_n = max(8, 4 * n_frontier)
        sampler = qmc.LatinHypercube(d=2 * num_wells, seed=int(rng.integers(0, 2**31 - 1)))
        unit = sampler.random(pool_n)
        lhs = np.zeros((pool_n, num_wells, 3), dtype=np.float32)
        for w in range(num_wells):
            lhs[:, w, 0] = x_lo + unit[:, 2 * w] * (x_hi - x_lo)
            lhs[:, w, 1] = y_lo + unit[:, 2 * w + 1] * (y_hi - y_lo)
        # Snap z to integer layers so the rebuilt graph's depth (int(round(z)))
        # and the pos_xyz z fed to the forward agree — CMA-loop coords are
        # already integer-z (CMAwM snaps them), so this keeps frontier consistent.
        lhs[:, :, 2] = np.round(rng.uniform(z_lo_eff, z_hi_eff, size=(pool_n, num_wells)))
        lhs[:, :, :2] = project_to_valid_cells(
            lhs[:, :, :2], valid_xy_indices, num_wells, rng, nx=nx, ny=ny,
        )
        # FPS on xy only — z scatter shouldn't inflate "spread" (z-layer units
        # aren't comparable to xy-cell units, and the surrogate is near-flat in z).
        xy_feats = lhs[:, :, :2].reshape(pool_n, -1)
        sel_idx = _farthest_point_select(xy_feats, min(n_frontier, pool_n), seed_idx=0)
        sel_coords = np.stack([lhs[i] for i in sel_idx], axis=0)
        t0 = time.perf_counter()
        sel_preds = _predict_rebuilt(sel_coords)  # (n_frontier, K) — ACCURATE
        timings["frontier_forward_s"] = time.perf_counter() - t0
        _, sel_npv = _candidate_scores(sel_coords, sel_preds)
        for j in range(sel_coords.shape[0]):
            frontier_coords.append(sel_coords[j].astype(np.float32))
            frontier_preds.append(sel_preds[j].astype(np.float32))
            frontier_npv.append(sel_npv[j].astype(np.float32) if sel_npv is not None else None)

    # ----- Assemble emit list (exploit first, then frontier), reusing the
    # accurate per-geology predictions computed during the search/frontier. -----
    emit_entries: list[tuple[np.ndarray, str, np.ndarray, Any]] = []
    for i in exploit_idx:
        emit_entries.append((pool_coords[i], "exploit", pool_preds[i], pool_npv[i]))
    for c, p, npvr in zip(frontier_coords, frontier_preds, frontier_npv):
        if np.isfinite(p).all():
            emit_entries.append((c, "frontier", p, npvr))
    if not emit_entries:
        raise RuntimeError("cma_surrogate: no finite candidates to emit.")
    npv_mode = (objective == "npv")

    # ----- Emit snapshots + candidates -----
    snapshots: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    t0 = time.perf_counter()
    for m, (cxyz, kind, pred_row, npv_row) in enumerate(emit_entries):
        if not np.isfinite(pred_row).all() or not np.isfinite(cxyz).all():
            print(f"[acquire-cma] skip non-finite candidate m={m} kind={kind}", flush=True)
            continue
        predictions_by_geology = []
        for k_idx, g in enumerate(geologies):
            meta = geology_metas.get(g.geology_index, {})
            entry = {
                "geology_index": g.geology_index,
                "geology_name": g.geology_name or Path(g.geology_h5_file).stem,
                "geology_file": str(Path(g.geology_h5_file).resolve()),
                "geology_config_id": meta.get("geology_config_id"),
                "geology_scenario_name": meta.get("scenario_name"),
                "geology_sample_num": meta.get("sample_num"),
                "discounted_total_revenue": float(pred_row[k_idx]),
                "total_energy_production": float("nan"),
            }
            if npv_mode and npv_row is not None:
                entry["npv"] = float(npv_row[k_idx])
            predictions_by_geology.append(entry)
        # The candidate SCORE that selection ranks on: ensemble NPV in npv mode
        # (so the manifest/Candidate.predicted_revenue carries NPV), else mean revenue.
        extra_fields = None
        if npv_mode and npv_row is not None:
            predicted_emv = float(np.nanmean(npv_row))
            # Cost breakdown uses the MEAN well length over geologies so it reconciles
            # EXACTLY with predicted_npv (NPV is linear in revenue and in Σwell_length, so
            # mean-over-geologies of NPV == NPV at the mean well length). When reservoir tops
            # are shared this is just the single per-candidate well length.
            if shared_rtop is not None:
                wl_bd = ctx.compute_angled_well_length(
                    cxyz, cube=ctx.geo_cube, fac_surf_xy=ctx.fac_surf_xy,
                    reservoir_top_k_map=shared_rtop,
                    vertical_lead_m=ctx.vertical_lead_m, ksurf=ctx.ksurf,
                )
            else:
                wl_bd = np.mean([
                    ctx.compute_angled_well_length(
                        cxyz, cube=ctx.geo_cube, fac_surf_xy=ctx.fac_surf_xy,
                        reservoir_top_k_map=d["reservoir_top_k"],
                        vertical_lead_m=ctx.vertical_lead_m, ksurf=ctx.ksurf,
                    ) for d in geology_loaded
                ], axis=0)
            extra_fields = {
                "objective": "npv",
                "predicted_npv": predicted_emv,
                "predicted_revenue_mean": float(np.nanmean(pred_row)),
                "npv_cost_breakdown": ctx.compute_npv(
                    float(np.nanmean(pred_row)), wl_bd, is_injector_list,
                    flowline_between_m=ctx.flowline_between_m, npv_terms=ctx.npv_terms,
                ),
            }
        else:
            predicted_emv = float(np.mean(pred_row))
        is_exploit = (kind == "exploit")
        # run_id is a per-candidate uniqueness counter only. Geology is carried
        # per IX task by the scenario (geology_config_id) token in the output
        # filename — an ensemble candidate spans all K geologies under one
        # run_id, so run_id MUST NOT be used to encode geology. The training
        # split resolves geology by scenario (geothermal.data.resolve_geology_indices).
        snap = _emit_snapshot_ensemble(
            run_id=m,
            iteration_step=int(cfg.cma_generations),
            kind=kind,
            coords_xyz=cxyz,
            is_injector_list=is_injector_list,
            predictions_by_geology=predictions_by_geology,
            predicted_emv=predicted_emv,
            seed_source_snapshot_id=(seed_incumbent_id if is_exploit else None),
            seed_source_iteration=(seed_incumbent_iter if is_exploit else None),
            seed_predicted_emv=None,
            geologies=geologies,
            geology_metas=geology_metas,
            well_configs_dir=well_configs_dir,
            snapshots_json_dir=snapshots_json_dir,
            to_julia_wells_text=ctx.to_julia_wells_text,
            extra_fields=extra_fields,
        )
        snapshots.append(snap)
        candidates.append(_snapshot_to_candidate_ensemble(snap, cxyz, is_injector_list))
    timings["snapshot_io_s"] = time.perf_counter() - t0

    # ----- Profiling CSV (single row) -----
    profiling_dir = out_dir / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)
    csv_path = profiling_dir / f"gpu_timings_iter_{iteration:04d}.csv"
    csv_row = {"device": device_strs[0], "geology_index": "ALL", **timings}
    col_order = ["device", "geology_index"] + [
        k for k in sorted(csv_row.keys()) if k not in ("device", "geology_index")
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(col_order) + "\n")
        f.write(",".join(str(csv_row.get(k, "")) for k in col_order) + "\n")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "run_id_prefix": run_id_prefix,
        # This acquire-stage manifest is informational only (the IX manifest is
        # (re)written by write_selected_manifest_ensemble, which stamps
        # "ensemble"). Tag it honestly so cma vs Adam-ensemble iterations are
        # distinguishable from the persisted acquire artifact.
        "mode": "cma_surrogate",
        "acquisition_optimizer": "cma_surrogate",
        "cma_popsize": int(cfg.cma_popsize),
        "cma_generations": int(cfg.cma_generations),
        "cma_sigma_init": float(cfg.cma_sigma_init),
        "depth_bounds": [int(z_lo_eff), int(z_hi_eff)],
        "n_exploit": cfg.n_exploit,
        "n_frontier": cfg.n_frontier,
        "distinct_k_realized": int(distinct_k_realized),
        "distinct_k_floor": int(distinct_k_floor),
        "dead_rock_drops": int(drop_stats["n_dropped"]),
        "geology_metadata": [
            {
                "geology_index": g.geology_index,
                "geology_name": g.geology_name or Path(g.geology_h5_file).stem,
                "geology_file": str(Path(g.geology_h5_file).resolve()),
            }
            for g in geologies
        ],
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "wallclock_seconds": time.time() - started,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {"manifest": manifest, "manifest_path": str(manifest_path), "candidates": candidates}


def _run_acquisition_ensemble(
    cfg: AcquisitionConfig,
    *,
    out_dir: Path,
    iteration: int,
    run_id_prefix: str = "al",
) -> dict[str, Any]:
    """Ensemble acquisition: each Adam multistart produces one well configuration
    evaluated against all K geologies (loss = mean over K predicted revenues).

    Two kinds: ``exploit`` (elite-seeded) and ``frontier`` (LHS-seeded). No
    adversarial / CMA. Each candidate ships K IX tasks. Returns the same dict
    shape as ``run_acquisition`` for the per-geology path.
    """
    out_dir = Path(out_dir)
    well_configs_dir = out_dir / "well_configs"
    snapshots_json_dir = out_dir / "snapshots_json"
    well_configs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_json_dir.mkdir(parents=True, exist_ok=True)

    device_strs = list(cfg.devices) if cfg.devices else [cfg.device]
    started = time.time()
    timings: dict[str, float] = {}

    t_ctx = time.perf_counter()
    ctx = _build_worker_context(cfg, iteration, device_strs[0])
    timings["worker_ctx_load_s"] = time.perf_counter() - t_ctx
    device = ctx.device                       # master device
    model = ctx.model                          # master model (cuda:0)
    target_mean = ctx.target_mean              # on master
    target_scale = ctx.target_scale            # on master
    is_injector_list = ctx.is_injector_list
    num_wells = ctx.num_wells

    # ----- Replicate model + scaler buffers to every requested device -----
    # IMPORTANT: do NOT deepcopy a LightningModule and then ``.to(d)`` it.
    # PyTorch Lightning's lifecycle hooks (state-dict pre-hooks, lazy
    # initialization, internal trainer/checkpoint references) interfere with
    # both ``__deepcopy__`` and the subsequent ``.to(d)`` move in subtle
    # version-dependent ways — the symptom in practice was a replica with
    # all-zero parameters (parameter-sum 0 vs the master's ~30000) which
    # silently produced NaN predictions for every candidate and wasted 16+
    # minutes of Adam.
    #
    # The reliable pattern (used by ``_build_worker_context`` for the
    # per-geology multi-GPU path too) is to call ``load_from_checkpoint`` per
    # device with ``map_location=d``. The checkpoint is small; loading it N
    # times costs ~1 s × N, amortized over the ~80-minute Adam loop.
    _ensure_surrogate_imports(cfg.surrogate_repo)
    from geothermal.model import HeteroGNNRegressor  # type: ignore

    device_objs: list[torch.device] = [device]
    models_per_dev: list[Any] = [model]
    target_scales_per_dev: list[torch.Tensor] = [target_scale]
    target_means_per_dev: list[torch.Tensor] = [target_mean]
    for d_str in device_strs[1:]:
        d = torch.device(d_str)
        if d_str.startswith("cuda:") and torch.cuda.is_available():
            torch.cuda.set_device(int(d_str.split(":", 1)[1]))
        m_replica = HeteroGNNRegressor.load_from_checkpoint(
            str(cfg.checkpoint_path), map_location=d
        ).to(d)
        m_replica.eval()
        device_objs.append(d)
        models_per_dev.append(m_replica)
        # Route through CPU rather than direct GPU->GPU copy: on PCIe-only
        # multi-GPU boxes (no NVLink) tensor.to(other_cuda) silently fills with
        # zeros under some driver/PyTorch combos even when can_device_access_peer
        # reports True. The model replica path above is safe because
        # load_from_checkpoint(map_location=d) loads from disk (CPU->GPU); only
        # these scaler tensors live on the master GPU and need cross-device move.
        target_scales_per_dev.append(target_scale.cpu().to(d))
        target_means_per_dev.append(target_mean.cpu().to(d))
    n_dev = len(device_objs)

    # ----- Sanity-check replicas match master at startup -----
    # Defense in depth: the load-from-checkpoint replica path is reliable, but
    # the master and replicas live on different devices and use different
    # PyTorch loaders. A drift would silently produce NaN predictions later;
    # this assertion catches it in <100 ms before we burn ~80 minutes of Adam.
    if n_dev > 1:
        with torch.no_grad():
            master_params = sum(
                float(p.detach().abs().sum().cpu()) for p in model.parameters()
            )
            for i in range(1, n_dev):
                replica_params = sum(
                    float(p.detach().abs().sum().cpu()) for p in models_per_dev[i].parameters()
                )
                if not (abs(master_params - replica_params) < max(1.0, 1e-5 * abs(master_params))):
                    raise RuntimeError(
                        f"Ensemble acquisition: model replica on {device_objs[i]} "
                        f"has parameter-sum {replica_params:.6g} != master {master_params:.6g} "
                        f"(after load_from_checkpoint to {device_objs[i]}). This indicates a "
                        f"broken replication and will produce NaN predictions."
                    )
                # Catch the GPU->GPU peer-copy bug (broken on this box's PCIe
                # multi-GPU + driver/PyTorch combo): target_scale.to(other_cuda)
                # silently returns a zero tensor even though
                # can_device_access_peer reports True. The replica path now
                # routes scaler tensors through CPU, but a regression here
                # would silently corrupt every prediction on the replica.
                rm = float(target_means_per_dev[i].detach().abs().sum().cpu())
                rs = float(target_scales_per_dev[i].detach().abs().sum().cpu())
                mm = float(target_mean.detach().abs().sum().cpu())
                ms = float(target_scale.detach().abs().sum().cpu())
                if not (abs(rm - mm) < max(1e-3, 1e-5 * abs(mm)) and abs(rs - ms) < max(1e-3, 1e-5 * abs(ms))):
                    raise RuntimeError(
                        f"Ensemble acquisition: scaler tensors on {device_objs[i]} "
                        f"(mean abs-sum {rm:.6g}, scale abs-sum {rs:.6g}) differ from "
                        f"master (mean {mm:.6g}, scale {ms:.6g}). This is the GPU->GPU "
                        f"peer-copy silently-zeros bug; route the .to(d) through CPU."
                    )
        if torch.cuda.is_available():
            for d in device_objs:
                if d.type == "cuda":
                    torch.cuda.synchronize(d)

    geologies = list(cfg.geologies)
    K = len(geologies)
    if K == 0:
        raise RuntimeError("Ensemble acquisition requires at least one geology.")

    # ----- Load every geology's static physics tensors + metadata -----
    t0 = time.perf_counter()
    geology_metas: dict[int, dict] = {}
    geology_loaded: list[dict[str, Any]] = []
    geology_paths: list[Path] = []
    for g in geologies:
        gp = Path(g.geology_h5_file)
        geology_paths.append(gp)
        with h5py.File(gp, "r") as src:
            geology_metas[g.geology_index] = _read_geology_metadata_safe(
                ctx.read_geology_metadata, src, g.geology_name or gp.stem
            )
        geology_loaded.append(_load_geology(
            gp, ctx.norm_config,
            ctx.PROPERTIES, ctx.PERM_PROPS, ctx.get_valid_mask, ctx.find_z_cutoff,
        ))
    timings["geo_load_s"] = time.perf_counter() - t0

    # Intersection bounds: a coord must fit in every geology's grid. nx/ny are
    # typically identical across geologies, but z_cutoff can vary — use the min.
    nx = min(d["nx"] for d in geology_loaded)
    ny = min(d["ny"] for d in geology_loaded)
    z_max = min(d["z_max"] for d in geology_loaded)
    x_lo, x_hi = float(cfg.edge_buffer), float(nx - 1 - cfg.edge_buffer)
    y_lo, y_hi = float(cfg.edge_buffer), float(ny - 1 - cfg.edge_buffer)

    # A column is "valid" only if it has live rock in every geology AND is
    # inside the edge buffer. This guarantees the static-graph build can place
    # all wells in every geology without dropping any.
    ix_lo, ix_hi = int(np.ceil(x_lo)), int(np.floor(x_hi))
    iy_lo, iy_hi = int(np.ceil(y_lo)), int(np.floor(y_hi))
    has_valid_z_all = None
    for d in geology_loaded:
        hv = np.any(d["temp0_full"][:z_max] > -900, axis=0)  # (nx, ny) bool
        has_valid_z_all = hv if has_valid_z_all is None else (has_valid_z_all & hv)
    edge_window = np.zeros_like(has_valid_z_all, dtype=bool)
    edge_window[ix_lo : ix_hi + 1, iy_lo : iy_hi + 1] = True
    valid_xy = has_valid_z_all & edge_window
    valid_xy_indices = np.argwhere(valid_xy)
    if valid_xy_indices.shape[0] < num_wells:
        raise RuntimeError(
            f"Ensemble acquisition: only {valid_xy_indices.shape[0]} (x,y) columns "
            f"are simultaneously valid across all {K} geologies inside the edge "
            f"buffer {cfg.edge_buffer}; need at least {num_wells}."
        )

    def _well_xy_valid(rx_: float, ry_: float, depth_: int) -> bool:
        ix_ = int(np.clip(int(round(rx_)), 0, nx - 1))
        iy_ = int(np.clip(int(round(ry_)), 0, ny - 1))
        for d in geology_loaded:
            if not bool(np.any(d["temp0_full"][: max(1, depth_), ix_, iy_] > -900)):
                return False
        return True

    seed_global = (cfg.seed + iteration) & ((1 << 63) - 1)
    rng = np.random.default_rng(seed_global)
    sampler = qmc.LatinHypercube(d=2 * num_wells, seed=seed_global)

    def _project_to_valid(coords_arr: np.ndarray) -> np.ndarray:
        used_cells: set[tuple[int, int]] = set()
        for w in range(num_wells):
            rx, ry = float(coords_arr[w, 0]), float(coords_arr[w, 1])
            depth = int(min(cfg.wells[w].depth, z_max))
            tries = 0
            while True:
                cell = (
                    int(np.clip(int(round(rx)), 0, nx - 1)),
                    int(np.clip(int(round(ry)), 0, ny - 1)),
                )
                if _well_xy_valid(rx, ry, depth) and cell not in used_cells:
                    break
                if tries >= 200:
                    raise RuntimeError(
                        f"Ensemble acquisition: failed to project well {w} to "
                        f"a valid unused cell after {tries} tries"
                    )
                pick = valid_xy_indices[int(rng.integers(0, len(valid_xy_indices)))]
                rx = float(pick[0])
                ry = float(pick[1])
                tries += 1
            used_cells.add(cell)
            coords_arr[w, 0] = rx
            coords_arr[w, 1] = ry
        return coords_arr

    def _lhs_one_start() -> np.ndarray:
        s = sampler.random(n=1)[0]
        c = np.zeros((num_wells, 3), dtype=np.float32)
        for w in range(num_wells):
            c[w, 0] = x_lo + s[2 * w] * (x_hi - x_lo)
            c[w, 1] = y_lo + s[2 * w + 1] * (y_hi - y_lo)
            c[w, 2] = int(min(cfg.wells[w].depth, int(z_max)))
        return _project_to_valid(c)

    def _build_cfg_from_coords(coords_arr: np.ndarray) -> list[dict]:
        return [
            {
                "x": float(coords_arr[w, 0]),
                "y": float(coords_arr[w, 1]),
                "depth": int(min(cfg.wells[w].depth, z_max)),
                "type": cfg.wells[w].type,
            }
            for w in range(num_wells)
        ]

    # ----- Generate M starts (exploit first, then frontier) -----
    t0 = time.perf_counter()
    n_exploit = int(cfg.n_exploit)
    n_frontier = int(cfg.n_frontier)
    if n_exploit + n_frontier == 0:
        raise RuntimeError(
            "Ensemble acquisition: n_exploit + n_frontier must be > 0."
        )

    elite_triples: list[tuple[np.ndarray, str, int]] = []
    if n_exploit > 0 and cfg.prior_metrics:
        elite_triples = _load_elite_seeds_ensemble(
            prior_metrics=list(cfg.prior_metrics or []),
            k=int(cfg.elite_top_k),
            rng=rng,
        )

    cfgs: list[list[dict]] = []
    start_kinds: list[str] = []  # "exploit" or "frontier"
    seed_source_ids: list[str | None] = []
    seed_source_iters: list[int | None] = []

    for i in range(n_exploit):
        if elite_triples:
            base, src_id, src_iter = elite_triples[i % len(elite_triples)]
            base = base.copy()
            noise = rng.normal(0.0, float(cfg.elite_seed_noise), size=base.shape).astype(np.float32)
            noise[:, 2] = 0.0
            coords_arr = (base + noise).astype(np.float32)
            coords_arr[:, 0] = np.clip(coords_arr[:, 0], x_lo, x_hi)
            coords_arr[:, 1] = np.clip(coords_arr[:, 1], y_lo, y_hi)
            for w in range(num_wells):
                coords_arr[w, 2] = int(min(cfg.wells[w].depth, int(z_max)))
            coords_arr = _project_to_valid(coords_arr)
            seed_source_ids.append(src_id)
            seed_source_iters.append(src_iter)
        else:
            # Cold start: fall back to LHS but still mark this as kind=exploit
            # so the cohort comparison stays uniform. seed source stays None.
            coords_arr = _lhs_one_start()
            seed_source_ids.append(None)
            seed_source_iters.append(None)
        cfgs.append(_build_cfg_from_coords(coords_arr))
        start_kinds.append("exploit")

    for _ in range(n_frontier):
        coords_arr = _lhs_one_start()
        cfgs.append(_build_cfg_from_coords(coords_arr))
        start_kinds.append("frontier")
        seed_source_ids.append(None)
        seed_source_iters.append(None)

    M = len(cfgs)
    timings["seed_gen_s"] = time.perf_counter() - t0
    timings["n_exploit"] = int(n_exploit)
    timings["n_frontier"] = int(n_frontier)

    # =========================================================================
    # MULTI-DEVICE DESIGN: candidate partition (not geology partition).
    #
    # Background — why candidate-partition. We previously tried splitting
    # the K geologies across N GPUs (each GPU owns ~K/N batches, the Adam
    # loop sums gradients across devices via per-step coords replicas, the
    # no-grad forwards interleave cuda:0 and cuda:1 from the main thread).
    # That design consistently failed: the no-grad terminal forward
    # produced NaN predictions for every cuda:N>0 geology, on the order of
    # 75% of candidates, after the Adam loop ran. We verified via MVP
    # repros that:
    #   - Single device (cuda:0 only): all candidates pass.
    #   - Multi-device with geology partition: cuda:N>0 geologies NaN.
    #   - Multi-device with candidate partition (each GPU runs an
    #     independent single-device acquisition on its M/N share of
    #     candidates, with its own full set of K batches): all candidates
    #     pass, ~1.6× wallclock speedup over single-device.
    #
    # The root cause of the geology-split failure was not fully isolated
    # but appears to be an interaction between (a) PyG's `Batch.to(d)`
    # NOT moving the custom `PhysicsContext` attribute (data.py:52, plain
    # Python class with no `.to()`), (b) the model's `volumes_dict[ch]
    # .to(device, non_blocking=True)` lazy device transfer per forward,
    # and (c) PyTorch's caching allocator stream-event tracking across
    # threads with mixed per-thread default streams. Candidate partition
    # sidesteps all three: every batch lives on exactly one device for
    # its entire lifetime, and each device's forwards happen only in that
    # device's own thread/context.
    #
    # Memory: each GPU has K=15 batches (vs K/N in geology-split), but each
    # batch is sized M/N candidates instead of M, so total per-GPU storage
    # is comparable. The activation memory per forward is also lower
    # (M/N candidates × edges instead of M × edges in the geology-split
    # case), making the gradient-accumulation regime more comfortable.
    # =========================================================================

    # ----- Partition the M candidates across the N devices -----
    M_per_dev: list[int] = []
    base = M // n_dev
    rem = M - base * n_dev
    for i in range(n_dev):
        M_per_dev.append(base + (1 if i < rem else 0))
    # Build a master->device mapping so the global m index can recover its
    # device + local index after the threaded acquisition.
    slice_starts: list[int] = []
    offset = 0
    for i in range(n_dev):
        slice_starts.append(offset)
        offset += M_per_dev[i]
    assert offset == M

    print(
        f"[acquire-ensemble] candidate partition: M={M} across {n_dev} devices "
        f"-> per-device M = {M_per_dev}",
        flush=True,
    )

    # ----- Build per-device contexts: each device gets its own K batches -----
    # Each device's batches are size M_local (its slice of candidates).
    # Geology physics is the same across devices (CPU tensors, shared); only
    # the per-candidate well configurations differ.
    t0 = time.perf_counter()
    per_dev_batches: list[list[Any]] = [[] for _ in range(n_dev)]
    per_dev_cfgs: list[list[list[dict]]] = []
    for i in range(n_dev):
        s = slice_starts[i]
        e = s + M_per_dev[i]
        per_dev_cfgs.append(cfgs[s:e])

    for i in range(n_dev):
        if device_objs[i].type == "cuda":
            torch.cuda.set_device(device_objs[i])
        slice_cfgs = per_dev_cfgs[i]
        if len(slice_cfgs) == 0:
            continue
        for g, d in zip(geologies, geology_loaded):
            static_graphs = _build_static_batch_for_starts(
                slice_cfgs, Path(g.geology_h5_file), d["physics_dict"], d["full_shape"],
                d["z_cutoff"], d["nx"], d["ny"],
                cfg.revenue_target, ctx.scaler,
                ctx.extract_well_data, ctx.build_wells_table, ctx.extract_vertical_profiles,
                ctx.build_single_hetero_data, d["temp0_full"],
                node_encoder=ctx.node_encoder, enrich_global_attr=ctx.enrich_global_attr,
            )
            per_dev_batches[i].append(Batch.from_data_list(static_graphs).to(device_objs[i]))
    timings["graph_build_s"] = time.perf_counter() - t0
    timings["n_devices"] = float(n_dev)
    for i in range(n_dev):
        timings[f"M_dev{i}"] = float(M_per_dev[i])

    K_float = float(K)

    def _predict_one_geo(c: torch.Tensor, bd, dev_idx: int, M_local: int) -> torch.Tensor:
        """Forward M_local coords through one geology batch on its owning device.

        ``c`` must already live on the same device as ``bd`` / the model
        replica. Returns an ``(M_local,)`` tensor of unscaled predictions
        on that device.
        """
        bd["well"].pos_xyz = c.view(-1, 3)
        pred_scaled = models_per_dev[dev_idx](bd).view(M_local)
        return pred_scaled * target_scales_per_dev[dev_idx] + target_means_per_dev[dev_idx]

    def _run_one_device_slice(dev_idx: int) -> dict[str, Any]:
        """Run the FULL acquisition (seed fwd, Adam loop, post-Adam projection,
        terminal fwd) for this device's slice of M_local candidates.

        Returns a dict with arrays/lists indexed 0..M_local-1 within the slice.
        The caller stitches per-device results back together at the global
        m=0..M-1 indexing for snapshot emission.

        Runs entirely on ``device_objs[dev_idx]`` — no cross-device tensors,
        no cross-device kernels. This is the key property that distinguishes
        candidate partition from the previous (broken) geology-split design.
        """
        d = device_objs[dev_idx]
        if d.type == "cuda":
            torch.cuda.set_device(d)
        M_local = M_per_dev[dev_idx]
        slice_batches = per_dev_batches[dev_idx]
        slice_cfgs = per_dev_cfgs[dev_idx]
        slice_off = slice_starts[dev_idx]

        if M_local == 0:
            return {
                "M_local": 0,
                "final_coords": np.zeros((0, num_wells, 3), dtype=np.float32),
                "preds_seed": np.zeros((0, K), dtype=np.float32),
                "preds_final": np.zeros((0, K), dtype=np.float32),
                "n_moved": 0,
                "step_diag": [],
            }

        # Build coords for this slice on this device.
        starts_local = [
            [[w["x"], w["y"], float(w["depth"])] for w in c] for c in slice_cfgs
        ]
        coords_local = torch.tensor(starts_local, dtype=torch.float32, device=d)
        coords_local.requires_grad = True
        optimizer_local = optim.Adam([coords_local], lr=cfg.learning_rate)
        last_valid_local = coords_local.detach().clone()

        def _all_geos_no_grad(c_in: torch.Tensor) -> np.ndarray:
            cols = [None] * K
            with torch.no_grad():
                for k_idx, bd in enumerate(slice_batches):
                    p = _predict_one_geo(c_in, bd, dev_idx, M_local).detach()
                    cols[k_idx] = p.cpu().numpy()
            return np.stack(cols, axis=1)  # (M_local, K)

        # Seed forward.
        preds_seed_local = _all_geos_no_grad(coords_local)

        # Adam loop with gradient accumulation across geologies.
        step_diag: list[dict] = []
        for step in range(1, int(cfg.k_safe) + 1):
            optimizer_local.zero_grad(set_to_none=True)
            for bd in slice_batches:
                preds_k = _predict_one_geo(coords_local, bd, dev_idx, M_local)
                loss_k = -(preds_k.sum() / K_float)
                loss_k.backward()

            with torch.no_grad():
                grads = coords_local.grad
                grad_was_nan = not torch.isfinite(grads).all().item()
                if grad_was_nan:
                    torch.nan_to_num_(grads, nan=0.0, posinf=0.0, neginf=0.0)
                    st = optimizer_local.state.get(coords_local, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st and not torch.isfinite(st[key]).all():
                            torch.nan_to_num_(st[key], nan=0.0, posinf=0.0, neginf=0.0)
                for d_idx, max_val in enumerate([nx - 1, ny - 1, z_max - 1]):
                    mask_lo = (coords_local[:, :, d_idx] <= 1e-4) & (grads[:, :, d_idx] > 0)
                    mask_hi = (coords_local[:, :, d_idx] >= max_val - 1e-4) & (grads[:, :, d_idx] < 0)
                    mask = mask_lo | mask_hi
                    if mask.any():
                        grads[..., d_idx][mask] = 0.0
                        st = optimizer_local.state.get(coords_local, {})
                        if "exp_avg" in st:
                            st["exp_avg"][..., d_idx][mask] = 0.0
                        if "exp_avg_sq" in st:
                            st["exp_avg_sq"][..., d_idx][mask] = 0.0
            optimizer_local.step()
            with torch.no_grad():
                if not torch.isfinite(coords_local).all():
                    bad = ~torch.isfinite(coords_local)
                    coords_local[bad] = last_valid_local[bad]
                    st = optimizer_local.state.get(coords_local, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st:
                            st[key][bad] = 0.0
                            if not torch.isfinite(st[key]).all():
                                torch.nan_to_num_(st[key], nan=0.0, posinf=0.0, neginf=0.0)
                coords_local[:, :, 0].clamp_(0, nx - 1)
                coords_local[:, :, 1].clamp_(0, ny - 1)
                coords_local[:, :, 2].clamp_(0, z_max - 1)
                last_valid_local.copy_(coords_local)

            # Per-device step diagnostic (rare).
            if step == 1 or step == int(cfg.k_safe) or step % max(1, int(cfg.log_every_n_steps)) == 0:
                with torch.no_grad():
                    grad_norm = float(coords_local.grad.abs().max().item()) if coords_local.grad is not None else float("nan")
                    coord_finite = bool(torch.isfinite(coords_local).all().item())
                step_diag.append({
                    "step": step, "dev_idx": dev_idx,
                    "max_grad": grad_norm, "coords_finite": coord_finite,
                    "grad_was_nan": grad_was_nan,
                })

        # Post-Adam validity projection: move any wells that drifted into
        # dead-rock for any geology back to valid cells. No-op for wells
        # already in valid rock.
        n_moved = 0
        with torch.no_grad():
            coords_np_local = coords_local.detach().cpu().numpy()
            for m_local in range(M_local):
                cand_xyz = coords_np_local[m_local].copy()
                projected = _project_to_valid(cand_xyz.copy())
                if not np.allclose(projected, cand_xyz, atol=1e-3):
                    n_moved += 1
                    coords_np_local[m_local] = projected
            if n_moved > 0:
                coords_local.data.copy_(torch.from_numpy(coords_np_local).to(d))
                st = optimizer_local.state.get(coords_local, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st:
                        torch.nan_to_num_(st[key], nan=0.0, posinf=0.0, neginf=0.0)

        # Terminal forward.
        preds_final_local = _all_geos_no_grad(coords_local)

        if d.type == "cuda":
            torch.cuda.synchronize(d)

        return {
            "M_local": M_local,
            "final_coords": coords_local.detach().cpu().numpy(),
            "preds_seed": preds_seed_local,
            "preds_final": preds_final_local,
            "n_moved": n_moved,
            "step_diag": step_diag,
        }

    # ----- Run per-device slices in parallel -----
    # Each device runs its own complete acquisition. No cross-device sync
    # during the loop; only the final result stitching happens on the main
    # thread after all device-threads join.
    t0 = time.perf_counter()
    per_dev_results: list[dict[str, Any] | None] = [None] * n_dev
    per_dev_errors: list[BaseException | None] = [None] * n_dev

    def _worker(dev_idx: int) -> None:
        try:
            per_dev_results[dev_idx] = _run_one_device_slice(dev_idx)
        except BaseException as e:  # pragma: no cover
            per_dev_errors[dev_idx] = e

    if n_dev == 1:
        _worker(0)
    else:
        threads = [
            threading.Thread(target=_worker, args=(i,), name=f"acquire-ensemble-slice-dev{i}")
            for i in range(n_dev)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    for err in per_dev_errors:
        if err is not None:
            raise err

    timings["adam_loop_s"] = time.perf_counter() - t0

    # Print step diagnostics in deterministic dev_idx-then-step order.
    for r in per_dev_results:
        if r is None:
            continue
        for diag in r["step_diag"]:
            print(
                f"[acquire-ensemble] dev{diag['dev_idx']} step={diag['step']}/{cfg.k_safe} "
                f"max|grad|={diag['max_grad']:.4g} coords_finite={diag['coords_finite']} "
                f"grad_sanitized_this_step={diag['grad_was_nan']}",
                flush=True,
            )
    total_moved = sum((r["n_moved"] if r is not None else 0) for r in per_dev_results)
    print(
        f"[acquire-ensemble] post-Adam validity projection: moved {total_moved}/{M} "
        f"candidates back to valid rock",
        flush=True,
    )

    # ----- Stitch per-device results back into global (M, ...) arrays -----
    final_coords_np = np.concatenate(
        [r["final_coords"] for r in per_dev_results if r is not None and r["M_local"] > 0],
        axis=0,
    )  # (M, num_wells, 3)
    preds_seed = np.concatenate(
        [r["preds_seed"] for r in per_dev_results if r is not None and r["M_local"] > 0],
        axis=0,
    )  # (M, K)
    preds_final = np.concatenate(
        [r["preds_final"] for r in per_dev_results if r is not None and r["M_local"] > 0],
        axis=0,
    )  # (M, K)
    seed_predicted_emv = preds_seed.mean(axis=1).astype(float)
    terminal_predicted_emv = preds_final.mean(axis=1).astype(float)

    # Diagnostic: how many candidates ended up non-finite, by their owning device.
    if n_dev > 1:
        bad_by_dev: dict[int, list[int]] = {}
        for m in range(M):
            if not np.isfinite(preds_final[m]).all():
                # Which device owned this candidate?
                for i in range(n_dev):
                    if slice_starts[i] <= m < slice_starts[i] + M_per_dev[i]:
                        bad_by_dev.setdefault(i, []).append(m)
                        break
        if bad_by_dev:
            parts = []
            for dev_idx, m_list in sorted(bad_by_dev.items()):
                parts.append(f"{device_objs[dev_idx]}:{m_list}")
            print(
                f"[acquire-ensemble] post-Adam terminal: non-finite preds by device: "
                f"{'; '.join(parts)}",
                flush=True,
            )

    # ----- Snapshot emit (uses global-indexed final_coords_np and preds_final) -----
    t0 = time.perf_counter()
    snapshots: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    for m in range(M):
        cxyz = final_coords_np[m]
        coords_ok = bool(np.isfinite(cxyz).all())
        preds_ok = bool(np.isfinite(preds_final[m]).all())
        if not coords_ok or not preds_ok:
            # Be explicit about which check failed so future debugging doesn't
            # require re-instrumenting the function.
            reason = []
            if not coords_ok:
                n_bad = int((~np.isfinite(cxyz)).sum())
                reason.append(f"coords has {n_bad}/{cxyz.size} non-finite entries; sample={cxyz.flatten()[:6]}")
            if not preds_ok:
                bad_geos = [k for k in range(K) if not np.isfinite(preds_final[m, k])]
                reason.append(f"preds non-finite on geologies {bad_geos[:5]} (of {len(bad_geos)} bad)")
            print(
                f"[acquire-ensemble] skip non-finite snapshot m={m} kind={start_kinds[m]}: "
                f"{'; '.join(reason)}",
                flush=True,
            )
            continue
        predictions_by_geology = []
        for k_idx, g in enumerate(geologies):
            meta = geology_metas.get(g.geology_index, {})
            predictions_by_geology.append({
                "geology_index": g.geology_index,
                "geology_name": g.geology_name or Path(g.geology_h5_file).stem,
                "geology_file": str(Path(g.geology_h5_file).resolve()),
                "geology_config_id": meta.get("geology_config_id"),
                "geology_scenario_name": meta.get("scenario_name"),
                "geology_sample_num": meta.get("sample_num"),
                "discounted_total_revenue": float(preds_final[m, k_idx]),
                # The surrogate's current head predicts revenue only. Use NaN
                # rather than a hard-coded 0 so any downstream consumer (Julia
                # task CSV, plotting) doesn't average a placeholder zero into a
                # real metric.
                "total_energy_production": float("nan"),
            })
        # run_id is a per-candidate uniqueness counter only; geology is carried
        # per IX task by the scenario (geology_config_id) token, not run_id (an
        # ensemble candidate spans all K geologies under one run_id). See
        # geothermal.data.resolve_geology_indices.
        snap = _emit_snapshot_ensemble(
            run_id=m,
            iteration_step=int(cfg.k_safe),
            kind=start_kinds[m],
            coords_xyz=cxyz,
            is_injector_list=is_injector_list,
            predictions_by_geology=predictions_by_geology,
            predicted_emv=float(terminal_predicted_emv[m]),
            seed_source_snapshot_id=seed_source_ids[m],
            seed_source_iteration=seed_source_iters[m],
            seed_predicted_emv=float(seed_predicted_emv[m]),
            geologies=geologies,
            geology_metas=geology_metas,
            well_configs_dir=well_configs_dir,
            snapshots_json_dir=snapshots_json_dir,
            to_julia_wells_text=ctx.to_julia_wells_text,
        )
        snapshots.append(snap)
        candidates.append(
            _snapshot_to_candidate_ensemble(snap, cxyz, is_injector_list)
        )
    timings["snapshot_io_s"] = time.perf_counter() - t0

    # ----- Profiling CSV (single row in ensemble mode) -----
    profiling_dir = out_dir / "profiling"
    profiling_dir.mkdir(parents=True, exist_ok=True)
    csv_path = profiling_dir / f"gpu_timings_iter_{iteration:04d}.csv"
    row = {
        # Pipe-separated so the CSV parses cleanly with the default delimiter
        # (commas would inject extra columns).
        "device": "|".join(device_strs),
        "geology_index": "ALL",
        **timings,
    }
    col_order = ["device", "geology_index"] + [k for k in sorted(row.keys()) if k not in ("device", "geology_index")]
    with open(csv_path, "w") as f:
        f.write(",".join(col_order) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in col_order) + "\n")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "run_id_prefix": run_id_prefix,
        "mode": "ensemble",
        "k_safe": cfg.k_safe,
        "n_exploit": cfg.n_exploit,
        "n_frontier": cfg.n_frontier,
        "geology_metadata": [
            {
                "geology_index": g.geology_index,
                "geology_name": g.geology_name or Path(g.geology_h5_file).stem,
                "geology_file": str(Path(g.geology_h5_file).resolve()),
            }
            for g in geologies
        ],
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "wallclock_seconds": time.time() - started,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {"manifest": manifest, "manifest_path": str(manifest_path), "candidates": candidates}
