"""Tests for the clean-start per-geology LHS seed sampler (orchestrator/seed.py).

The sampler needs only geology-H5 metadata (no surrogate model/checkpoint), so
these run end-to-end against the bend-local smoke geologies when present, and
skip cleanly otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.acquire import GeologySpec, WellSpec
from orchestrator.seed import _RUN_ID_GEO_STRIDE, _split_counts, build_seed_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_GEO_CFG = REPO_ROOT / "configs" / "geologies_smoke_local.json"
LOCAL_SURROGATE_REPO = Path("/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate")

_WELLS = [
    WellSpec(type=("injector" if i % 2 == 0 else "producer"), depth=50)
    for i in range(12)
]
_DEPTH_MIN, _DEPTH_MAX, _EDGE_BUFFER = 5, 70, 10


# ---------------------------------------------------------------------------
# Pure helper — no env needed
# ---------------------------------------------------------------------------


def test_split_counts_even_and_remainder():
    # 256 across 15 → first one gets the remainder, sums to 256.
    counts = _split_counts(256, 15)
    assert counts == [18] + [17] * 14
    assert sum(counts) == 256
    # Clean division.
    assert _split_counts(30, 15) == [2] * 15
    # Small case used by the e2e test below.
    assert _split_counts(6, 2) == [3, 3]
    assert _split_counts(7, 2) == [4, 3]


def test_split_counts_fewer_samples_than_geologies():
    # n_total < n_groups → low-index groups get 1, the rest get 0; sum preserved.
    counts = _split_counts(10, 15)
    assert counts == [1] * 10 + [0] * 5
    assert sum(counts) == 10
    assert _split_counts(1, 2) == [1, 0]


# ---------------------------------------------------------------------------
# End-to-end sampler over the smoke geologies
# ---------------------------------------------------------------------------


def _load_smoke_geologies() -> list[GeologySpec]:
    payload = json.loads(SMOKE_GEO_CFG.read_text())
    entries = payload if isinstance(payload, list) else payload.get("geologies", [])
    return [
        GeologySpec(
            geology_index=int(e["geology_index"]),
            geology_h5_file=str(Path(e["geology_h5_file"]).resolve()),
            geology_name=e.get("geology_name"),
        )
        for e in entries
    ]


def _require_smoke_env() -> list[GeologySpec]:
    if not SMOKE_GEO_CFG.exists():
        pytest.skip(f"smoke geology config missing: {SMOKE_GEO_CFG}")
    if not LOCAL_SURROGATE_REPO.exists():
        pytest.skip(f"local surrogate repo missing: {LOCAL_SURROGATE_REPO}")
    geologies = _load_smoke_geologies()
    missing = [g.geology_h5_file for g in geologies if not Path(g.geology_h5_file).exists()]
    if missing:
        pytest.skip(f"smoke geology H5s missing: {missing[:2]}")
    # The late imports inside build_seed_manifest (active_learning_utils,
    # preprocess_h5) require the surrogate repo to be importable.
    import sys
    if str(LOCAL_SURROGATE_REPO) not in sys.path:
        sys.path.insert(0, str(LOCAL_SURROGATE_REPO))
    pytest.importorskip("geothermal.active_learning_utils")
    pytest.importorskip("preprocess_h5")
    return geologies


def _z_bounds_for(geo: GeologySpec) -> tuple[int, int]:
    """Recompute the sampler's effective z bounds for a geology (mirrors
    build_seed_manifest) so we can assert emitted depths respect the grid cap."""
    import h5py
    from preprocess_h5 import find_z_cutoff, get_valid_mask  # type: ignore

    with h5py.File(geo.geology_h5_file, "r") as src:
        z_cutoff = int(find_z_cutoff(get_valid_mask(src), invalid_threshold=0.95))
    z_hi = int(min(_DEPTH_MAX, z_cutoff - 1))
    z_lo = int(min(_DEPTH_MIN, z_hi - 1))
    return z_lo, z_hi


def test_build_seed_manifest_e2e(tmp_path):
    geologies = _require_smoke_env()
    n_seed = 6  # 2 geologies → 3 + 3

    manifest_path, n_emitted = build_seed_manifest(
        surrogate_repo=LOCAL_SURROGATE_REPO,
        geologies=geologies,
        wells=_WELLS,
        n_seed_samples=n_seed,
        acquire_dir=tmp_path / "acquire",
        manifest_path=tmp_path / "manifests" / "seed_manifest.json",
        depth_min=_DEPTH_MIN,
        depth_max=_DEPTH_MAX,
        edge_buffer=_EDGE_BUFFER,
        seed=42,
        iteration=0,
    )

    assert n_emitted == n_seed
    manifest = json.loads(Path(manifest_path).read_text())
    snaps = manifest["snapshots"]
    assert manifest["snapshot_count"] == n_seed
    assert len(snaps) == n_seed

    # Per-geology counts sum to the budget and match the even split.
    per_geo = manifest["seed"]["per_geology_counts"]
    assert sum(per_geo.values()) == n_seed
    assert set(per_geo.values()) == {3}

    z_bounds = {g.geology_index: _z_bounds_for(g) for g in geologies}
    all_z: list[float] = []

    for snap in snaps:
        geo_entry = snap["well_config_paths_by_geology"][0]
        geo_idx = int(geo_entry["geology_index"])
        # CRITICAL: geology must round-trip through run_id // stride so the
        # training-time geology-aware split labels every case correctly.
        assert int(snap["run_id"]) // _RUN_ID_GEO_STRIDE == geo_idx
        # geology_config_id (scenario) must be present — Julia staging requires it.
        assert geo_entry["geology_config_id"] not in (None, "")
        # Predicted revenue is NaN at seed time (no surrogate).
        import math
        assert math.isnan(float(snap["predicted_discounted_total_revenue"]))

        # Inspect the per-snapshot JSON: depths integer, in-bounds, and varied.
        wells = json.loads(Path(snap["json_path"]).read_text())["wells"]
        assert len(wells) == len(_WELLS)
        z_lo, z_hi = z_bounds[geo_idx]
        zs = [w["z"] for w in wells]
        for z in zs:
            assert z == round(z), "depth must be snapped to an integer grid layer"
            assert z_lo <= z <= z_hi, f"depth {z} outside [{z_lo}, {z_hi}]"
        all_z.extend(zs)

    # Depth randomization: not every well sits at the same layer.
    assert len(set(all_z)) > 1, "expected depth variation across the seed batch"

    # Artifacts written where the driver expects them.
    assert (tmp_path / "acquire" / "well_configs").is_dir()
    assert (tmp_path / "acquire" / "snapshots_json").is_dir()
    assert len(list((tmp_path / "acquire" / "well_configs").glob("*.jl"))) == n_seed


def test_zero_count_geology_recorded(tmp_path):
    """n_seed_samples < n_geologies → trailing geologies get 0, recorded (not
    silently dropped) in the manifest's per-geology counts."""
    geologies = _require_smoke_env()
    manifest_path, n_emitted = build_seed_manifest(
        surrogate_repo=LOCAL_SURROGATE_REPO,
        geologies=geologies,
        wells=_WELLS,
        n_seed_samples=1,  # 2 geologies → [1, 0]
        acquire_dir=tmp_path / "acquire",
        manifest_path=tmp_path / "manifests" / "seed_manifest.json",
        depth_min=_DEPTH_MIN,
        depth_max=_DEPTH_MAX,
        edge_buffer=_EDGE_BUFFER,
        seed=42,
    )
    assert n_emitted == 1
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["snapshot_count"] == 1
    per_geo = manifest["seed"]["per_geology_counts"]
    # Both geologies appear; the second is recorded as 0 rather than omitted.
    assert per_geo[str(geologies[0].geology_index)] == 1
    assert per_geo[str(geologies[1].geology_index)] == 0
    assert sum(per_geo.values()) == 1


def test_run_id_stride_guard(tmp_path):
    """A per-geology count >= the run_id stride would collide the geology
    encoding and must raise before any sampling."""
    geologies = _require_smoke_env()
    with pytest.raises(ValueError, match="run_id stride"):
        build_seed_manifest(
            surrogate_repo=LOCAL_SURROGATE_REPO,
            geologies=geologies[:1],
            wells=_WELLS,
            n_seed_samples=_RUN_ID_GEO_STRIDE,  # single geology → n_geo == stride
            acquire_dir=tmp_path / "acquire",
            manifest_path=tmp_path / "manifests" / "seed_manifest.json",
            depth_min=_DEPTH_MIN,
            depth_max=_DEPTH_MAX,
            edge_buffer=_EDGE_BUFFER,
            seed=42,
        )
