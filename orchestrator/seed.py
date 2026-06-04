"""Per-geology LHS seed sampler for a clean-start AL run.

Generates a manifest of well configurations sampled by Latin Hypercube
**independently for each geology** (with depth/z randomization), for a
clean-start active-learning run that begins with NO pretrained surrogate. Each
emitted snapshot is tied to a single geology and expands to exactly one
INTERSECT (IX) task, so a total budget of ``n_seed_samples`` configs ==
``n_seed_samples`` IX runs, split ~evenly across the geologies.

No surrogate is involved: ``predicted_revenue`` is NaN and the geology grid
metadata (valid xy cells, z-cutoff) is read directly from each geology H5 — we
do NOT call ``acquire._load_geology`` because that requires a normalization
config, which does not exist yet at seed time (it is computed downstream from
the seed IX outputs by ``preprocess_h5.py``).

Reuses, from ``orchestrator.acquire`` / ``orchestrator.baseline_optimizer``:
* ``_read_geology_metadata_safe`` — geology_config_id / scenario / sample_num
* ``_emit_snapshot`` / ``_snapshot_to_candidate`` / ``write_selected_manifest``
* ``project_to_valid_cells`` — dead-rock-avoiding, unique xy projection
* the late-imported ``read_geology_metadata`` / ``to_julia_wells_text`` from
  ``geothermal.active_learning_utils``, and ``find_z_cutoff`` / ``get_valid_mask``
  from ``preprocess_h5`` (both resolve from the surrogate repo on sys.path).

CRITICAL conventions (so the training-time geology-aware split resolves
correctly and IX output filenames never collide):

* Geology is recovered by ``data.resolve_geology_indices`` from the
  ``<scenario>`` (geology_config_id) token in the IX output stem
  ``v2.5_{output_prefix}_{scenario}_run{run_id:04d}_iter{iteration:04d}`` — the
  single source of truth, stamped per task from each geology's config_id. So
  geology labeling does NOT depend on run_id.
* ``run_id = geology_index * 10000 + local_idx`` is a per-snapshot uniqueness
  counter (and keeps parity with the per-geology acquisition at
  ``acquire._emit_snapshot`` 1031/1064). Its ``//10000`` geology encoding is now
  only a LEGACY FALLBACK used by the resolver when the geologies config is
  unavailable — hence local_idx must stay below the stride (see guard below).
* ``iteration = 0`` drives the trailing ``_iter0000`` token; the driver must
  pass ``output_prefix`` containing an ``_iter\\d+`` token (e.g.
  ``"seed_<id>_iter0000"``) so the stem matches ``AL_RE``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

from orchestrator.acquire import (
    GeologySpec,
    WellSpec,
    _emit_snapshot,
    _ensure_surrogate_imports,
    _per_geology_seed,
    _read_geology_metadata_safe,
    _snapshot_to_candidate,
    write_selected_manifest,
)
from orchestrator.baseline_optimizer import project_to_valid_cells

# Stride encoding geology_index into run_id, matching the production per-geology
# acquisition (acquire.py:1031,1064). Geology is now resolved from the scenario
# token (data.resolve_geology_indices), so this encoding is a uniqueness counter
# plus the resolver's legacy //10000 fallback; local_idx must stay strictly
# below the stride so that fallback still round-trips.
_RUN_ID_GEO_STRIDE = 10000


def _split_counts(n_total: int, n_groups: int) -> list[int]:
    """Distribute ``n_total`` samples across ``n_groups`` geologies as evenly as
    possible, handing the remainder to the lowest-index groups. Sums to exactly
    ``n_total``.
    """
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    base, rem = divmod(int(n_total), int(n_groups))
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def build_seed_manifest(
    *,
    surrogate_repo: Path,
    geologies: list[GeologySpec],
    wells: list[WellSpec],
    n_seed_samples: int,
    acquire_dir: Path,
    manifest_path: Path,
    depth_min: int = 5,
    depth_max: int = 70,
    edge_buffer: int = 10,
    seed: int = 42,
    iteration: int = 0,
) -> tuple[Path, int]:
    """Sample per-geology LHS well configs and write a stage-ready manifest.

    Writes ``acquire_dir/well_configs/*.jl`` + ``acquire_dir/snapshots_json/*.json``
    and the manifest at ``manifest_path``. Returns ``(manifest_path, n_emitted)``.
    """
    _ensure_surrogate_imports(surrogate_repo)
    # Late imports (resolve from the surrogate repo on sys.path). Mirrors
    # acquire._build_worker_context, but only the surrogate-free helpers.
    from geothermal.active_learning_utils import (  # type: ignore
        read_geology_metadata,
        to_julia_wells_text,
    )
    from preprocess_h5 import find_z_cutoff, get_valid_mask  # type: ignore

    if not geologies:
        raise ValueError("build_seed_manifest: no geologies supplied")

    num_wells = len(wells)
    is_injector_list = [w.type == "injector" for w in wells]

    well_configs_dir = acquire_dir / "well_configs"
    snapshots_json_dir = acquire_dir / "snapshots_json"
    well_configs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_json_dir.mkdir(parents=True, exist_ok=True)

    per_geo_counts = _split_counts(n_seed_samples, len(geologies))
    if n_seed_samples < len(geologies):
        print(
            f"[seed] WARNING: n_seed_samples={n_seed_samples} < n_geologies="
            f"{len(geologies)}; some geologies will get 0 seed samples and be "
            f"ABSENT from the seed training set (the surrogate gets no signal for "
            f"them until cma_surrogate acquisition queries them at iter >= 1)."
        )

    candidates = []
    counts_by_geo: dict[int, int] = {}
    for gi, geo in enumerate(geologies):
        n_geo = per_geo_counts[gi]
        if n_geo <= 0:
            # Record the zero so the gap is visible in the manifest's per-geology
            # counts rather than silently omitted.
            counts_by_geo[int(geo.geology_index)] = 0
            print(f"[seed] WARNING: geology {geo.geology_index} got 0 seed samples — "
                  f"absent from the seed set.")
            continue
        if n_geo >= _RUN_ID_GEO_STRIDE:
            raise ValueError(
                f"geology {geo.geology_index}: {n_geo} samples >= run_id stride "
                f"{_RUN_ID_GEO_STRIDE}; would collide geology encoding in run_id."
            )
        geo_path = Path(geo.geology_h5_file)
        geo_name = geo.geology_name or geo_path.stem

        with h5py.File(geo_path, "r") as src:
            geo_meta = _read_geology_metadata_safe(read_geology_metadata, src, geo_name)
            valid_mask = get_valid_mask(src)
            z_cutoff = int(find_z_cutoff(valid_mask, invalid_threshold=0.95))
            temp0_full = src["Input/Temperature0"][:]

        nx = int(valid_mask.shape[1])
        ny = int(valid_mask.shape[2])
        z_max = z_cutoff

        # Edge-buffered window ∩ live-rock columns (mirrors acquire.py:761-776).
        x_lo, x_hi = float(edge_buffer), float(nx - 1 - edge_buffer)
        y_lo, y_hi = float(edge_buffer), float(ny - 1 - edge_buffer)
        has_valid_z = np.any(temp0_full > -900, axis=0)  # (nx, ny) bool
        ix_lo, ix_hi = int(np.ceil(x_lo)), int(np.floor(x_hi))
        iy_lo, iy_hi = int(np.ceil(y_lo)), int(np.floor(y_hi))
        edge_window = np.zeros_like(has_valid_z, dtype=bool)
        edge_window[ix_lo : ix_hi + 1, iy_lo : iy_hi + 1] = True
        valid_xy = has_valid_z & edge_window
        valid_xy_indices = np.argwhere(valid_xy)
        if valid_xy_indices.shape[0] < num_wells:
            raise RuntimeError(
                f"Geology {geo.geology_index}: only {valid_xy_indices.shape[0]} valid "
                f"(x,y) columns inside edge buffer {edge_buffer}; need >= {num_wells}."
            )

        # Depth bounds capped at grid extent (mirrors acquire.py:2158-2166).
        z_hi_eff = int(min(int(depth_max), int(z_max) - 1))
        z_lo_eff = int(min(int(depth_min), z_hi_eff - 1))
        if z_hi_eff <= z_lo_eff:
            raise RuntimeError(
                f"Geology {geo.geology_index}: degenerate depth bounds "
                f"(z_lo={z_lo_eff}, z_hi={z_hi_eff}) from depth_min={depth_min}, "
                f"depth_max={depth_max}, z_max={z_max}."
            )

        # Per-geology RNG so results don't depend on geology ordering.
        seed_geo = _per_geology_seed(seed, geo.geology_index)
        rng = np.random.default_rng(seed_geo)
        sampler = qmc.LatinHypercube(d=2 * num_wells, seed=seed_geo)
        unit = sampler.random(n_geo)

        coords = np.zeros((n_geo, num_wells, 3), dtype=np.float32)
        for w in range(num_wells):
            coords[:, w, 0] = x_lo + unit[:, 2 * w] * (x_hi - x_lo)
            coords[:, w, 1] = y_lo + unit[:, 2 * w + 1] * (y_hi - y_lo)
        # Independent per-well, per-config integer depth (matches the cma
        # frontier sampler at acquire.py:2396) → depth randomization.
        coords[:, :, 2] = np.round(
            rng.uniform(z_lo_eff, z_hi_eff, size=(n_geo, num_wells))
        )
        coords[:, :, :2] = project_to_valid_cells(
            coords[:, :, :2], valid_xy_indices, num_wells, rng, nx=nx, ny=ny,
        )

        for local_idx in range(n_geo):
            coords_xyz = coords[local_idx]
            run_id = geo.geology_index * _RUN_ID_GEO_STRIDE + local_idx
            snap = _emit_snapshot(
                run_id=run_id,
                iteration_step=iteration,
                kind="frontier",
                coords_xyz=coords_xyz,
                is_injector_list=is_injector_list,
                predicted_revenue=float("nan"),
                geo=geo,
                geo_path=geo_path,
                geo_name=geo_name,
                geo_meta=geo_meta,
                well_configs_dir=well_configs_dir,
                snapshots_json_dir=snapshots_json_dir,
                to_julia_wells_text=to_julia_wells_text,
            )
            candidates.append(_snapshot_to_candidate(snap, coords_xyz, is_injector_list))
        counts_by_geo[int(geo.geology_index)] = n_geo

    if not candidates:
        raise RuntimeError("build_seed_manifest produced 0 candidates")

    write_selected_manifest(
        candidates,
        out_path=manifest_path,
        iteration=iteration,
        geologies=geologies,
        extras={
            "seed": {
                "n_seed_samples": int(n_seed_samples),
                "n_emitted": len(candidates),
                "depth_min": int(depth_min),
                "depth_max": int(depth_max),
                "edge_buffer": int(edge_buffer),
                "seed": int(seed),
                "per_geology_counts": counts_by_geo,
            }
        },
    )
    return manifest_path, len(candidates)
