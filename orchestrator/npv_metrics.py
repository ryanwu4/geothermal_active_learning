"""Local proxy-NPV augmentation of per_candidate_metrics for the AL dashboard.

Both local drivers run on the GPU host (bend) where the geo coordinate cube lives; the remote
Sherlock ingest computes only REVENUE (it has no cube). When the run optimizes NPV, this module
recomputes the proxy NPV LOCALLY from each candidate's per-geology real/predicted revenue + the
snapshot's coords, and writes it back into the local per_candidate_metrics.json so the diagnostic
dashboard reflects the optimized objective. Single source of truth for the math is
``geothermal.well_geometry`` (the same module the acquisition fitness uses).

Used by scripts/run_al_local.py (surrogate hybrid) and scripts/run_baseline_global.py (baseline).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def assert_objective_marker(ws: Path, objective: str) -> None:
    """Persist the run's objective in the local workspace and refuse to resume with a flipped one.

    Guards the realistic footgun: restarting a long-lived driver (``--run-id``) against a config
    whose ``objective`` differs from the one the workspace was started with would silently optimize
    a different quantity from the resume point on (and the optimizer-state pickle would carry a
    fit for the other objective). The objective is otherwise only in the config, never persisted.
    """
    ws = Path(ws)
    marker = ws / ".objective"
    if marker.exists():
        prev = marker.read_text().strip()
        if prev != str(objective):
            raise RuntimeError(
                f"workspace {ws} was started with objective='{prev}' but the config now says "
                f"objective='{objective}'. Refusing to resume with a different objective — use a "
                f"fresh local_workspace or restore the original objective in the config."
            )
    else:
        ws.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(objective))


def write_npv_context(
    ws: Path,
    *,
    surrogate_repo: str | Path,
    geo_cube_path: str | Path,
    facilities: Any,
    vertical_lead_m: float,
    ksurf: int,
    poro_thresh: float,
    reservoir_geology_h5: str | Path,
) -> None:
    """Write ws/npv_context.json so the standalone plotting subprocess can rebuild the deviated-well
    geometry (cube + facilities + one geology's reservoir top) to render the 3D best-config shape.
    Best-effort: never raises into the driver loop."""
    try:
        ctx = {
            "surrogate_repo": str(surrogate_repo),
            "geo_cube_path": str(geo_cube_path),
            "facilities": [list(map(int, f)) for f in facilities],
            "vertical_lead_m": float(vertical_lead_m),
            "ksurf": int(ksurf),
            "poro_thresh": float(poro_thresh),
            "reservoir_geology_h5": str(reservoir_geology_h5),
        }
        Path(ws).mkdir(parents=True, exist_ok=True)
        with open(Path(ws) / "npv_context.json", "w") as f:
            json.dump(ctx, f, indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"[npv] could not write npv_context.json: {e}")


def build_npv_state(
    *,
    surrogate_repo: str | Path,
    economics_config_path: str | Path | None,
    geo_cube_path: str | Path,
    facilities: Any,
    vertical_lead_m: float,
    ksurf: int,
    poro_thresh: float,
    geology_entries: list[dict],
    is_injector_list: list[bool],
) -> dict:
    """Load the cube, economics terms, facility surface coords, and per-geology reservoir-top maps.

    geology_entries: list of {"geology_index": int, "geology_h5_file": str}. Returns a dict of the
    loaded state + the well_geometry callables. Raises if the cube/economics are missing.
    """
    import h5py  # local-only

    surrogate_repo = Path(surrogate_repo).expanduser().resolve()
    if str(surrogate_repo) not in sys.path:
        sys.path.insert(0, str(surrogate_repo))
    from geothermal.well_geometry import (  # type: ignore
        load_geo_coord_cube, reservoir_top_k_map, facilities_surface_xy,
        surface_flowline_length, compute_angled_well_length, compute_npv, load_npv_terms,
    )

    geo_cube_path = Path(geo_cube_path).expanduser()
    if not geo_cube_path.exists():
        raise FileNotFoundError(f"npv: geo_cube_path not found: {geo_cube_path}")
    econ = economics_config_path or (surrogate_repo / "configs" / "economics.json")

    cube = load_geo_coord_cube(geo_cube_path)
    terms = load_npv_terms(str(econ))
    fac_surf = facilities_surface_xy(cube, facilities, ksurf=int(ksurf))
    flowline = surface_flowline_length(fac_surf)
    rtop_by_geo: dict[int, np.ndarray] = {}
    for e in geology_entries:
        with h5py.File(e["geology_h5_file"], "r") as h:
            poro = h["Input/Porosity"][:]
        rtop_by_geo[int(e["geology_index"])] = reservoir_top_k_map(poro, poro_thresh=float(poro_thresh))
    return {
        "cube": cube, "terms": terms, "fac_surf": fac_surf, "flowline": flowline,
        "rtop_by_geo": rtop_by_geo, "vertical_lead_m": float(vertical_lead_m), "ksurf": int(ksurf),
        "is_injector_list": is_injector_list,
        "compute_angled_well_length": compute_angled_well_length, "compute_npv": compute_npv,
    }


def load_coords_by_snapshot(snapshots_dir: Path) -> dict[str, np.ndarray]:
    """Map snapshot_id -> (num_wells, 3) coords from the snapshot JSONs in a dir."""
    out: dict[str, np.ndarray] = {}
    snapshots_dir = Path(snapshots_dir)
    if not snapshots_dir.is_dir():
        return out
    for p in snapshots_dir.glob("*.json"):
        try:
            snap = json.load(open(p))
        except Exception:
            continue
        sid = snap.get("snapshot_id")
        wells = snap.get("wells")
        if not sid or not wells:
            continue
        out[str(sid)] = np.array([[w["x"], w["y"], w["z"]] for w in wells], dtype=np.float64)
    return out


def _row_npv(state: dict, coords_xyz: np.ndarray, geo_idx: Any, revenue: Any) -> float | None:
    rtop = state["rtop_by_geo"].get(int(geo_idx)) if geo_idx is not None else None
    if rtop is None or revenue is None or not np.isfinite(float(revenue)):
        return None
    coords_xyz = np.asarray(coords_xyz, dtype=np.float64)
    # Well-count guard: CAPEX comes from these coords' well count while OPEX uses the fixed
    # is_injector_list (from cfg["wells"]). If a snapshot's well count drifts from the config
    # (changed wells on resume, legacy/corrupt snapshot), the two disagree and compute_npv would
    # return a plausible-but-wrong finite NPV with no error. Bail to None instead (mirrors the
    # warm-start guard in acquire.py). Returns None -> row's npv is null, dropped from objective plots.
    if coords_xyz.shape[0] != len(state["is_injector_list"]):
        return None
    wl = state["compute_angled_well_length"](
        coords_xyz, cube=state["cube"], fac_surf_xy=state["fac_surf"],
        reservoir_top_k_map=rtop, vertical_lead_m=state["vertical_lead_m"], ksurf=state["ksurf"],
    )
    return float(state["compute_npv"](
        float(revenue), wl, state["is_injector_list"],
        flowline_between_m=state["flowline"], npv_terms=state["terms"],
    )["npv"])


def augment_metrics_with_npv(
    payload: dict, coords_by_snapshot: dict[str, np.ndarray], state: dict
) -> dict:
    """In-place: add real_npv/predicted_npv per row (preserving *_rev), per-snapshot
    terminal_real_npv, objective tag, and best_real_npv_in_batch. Defensive: rows whose snapshot
    coords are unavailable keep their revenue fields and get null npv (never raises)."""
    rows = payload.get("candidates", [])
    by_snap: dict[str, list[float]] = {}
    for c in rows:
        sid = str(c.get("snapshot_id", ""))
        coords = coords_by_snapshot.get(sid)
        c["real_revenue_rev"] = c.get("real_revenue")
        c["predicted_revenue_rev"] = c.get("predicted_revenue")
        if coords is None:
            c["real_npv"] = None
            c["predicted_npv"] = None
            continue
        gi = c.get("geology_index")
        rn = _row_npv(state, coords, gi, c.get("real_revenue"))
        pn = _row_npv(state, coords, gi, c.get("predicted_revenue"))
        c["real_npv"] = rn
        c["predicted_npv"] = pn
        if rn is not None:
            by_snap.setdefault(sid, []).append(rn)
    term: dict[str, float | None] = {}
    for sid, vals in by_snap.items():
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        term[sid] = float(np.mean(finite)) if finite else None
    for c in rows:
        c["terminal_real_npv"] = term.get(str(c.get("snapshot_id", "")))
    payload["objective"] = "npv"
    finite_term = [v for v in term.values() if v is not None and np.isfinite(v)]
    payload["best_real_npv_in_batch"] = max(finite_term) if finite_term else None
    return payload


def augment_metrics_file(metrics_path: Path, snapshots_dir: Path, state: dict) -> bool:
    """Load a per_candidate_metrics.json, augment with NPV, write back. Returns True on success."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return False
    payload = json.load(open(metrics_path))
    coords = load_coords_by_snapshot(snapshots_dir)
    augment_metrics_with_npv(payload, coords, state)
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    return True
