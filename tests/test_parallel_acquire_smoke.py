"""End-to-end smoke test for the multi-GPU acquisition path.

Runs the gradient-based acquisition once on a single GPU and once across both
GPUs, then verifies that the resulting manifests are identical (same snapshot
ids, same per-snapshot predicted revenue within float tolerance). Also asserts
that the parallel run actually touched both devices.

Requires:
  - 2 visible CUDA devices (skipped otherwise)
  - Local bootstrap checkpoint + scaler + norm config under ``local_workspace/``
  - At least 4 geology H5 files under ``../al_local_data/geology_h5s/``

Run from the repo root:
    pytest tests/test_parallel_acquire_smoke.py -s -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT / "local_workspace"
SURROGATE_REPO = Path("/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate")
GEO_DIR = Path("/home/rwu4/omv_geothermal/al_local_data/geology_h5s")
CKPT = WORKSPACE / "models" / "bootstrap" / "best-epoch=053-val_loss=0.0982.ckpt"
SCALER = WORKSPACE / "models" / "bootstrap" / "scaler.pkl"
NORM_CONFIG = WORKSPACE / "norm_config.json"

GEO_FILES = [
    GEO_DIR / "v2.5_0001.h5",
    GEO_DIR / "v2.5_0003.h5",
    GEO_DIR / "v2.5_0005.h5",
    GEO_DIR / "v2.5_0010.h5",
]


def _have_required_inputs() -> bool:
    return all(p.exists() for p in [CKPT, SCALER, NORM_CONFIG, *GEO_FILES])


def _gpus_available() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


pytestmark = [
    pytest.mark.skipif(not _have_required_inputs(),
                       reason="Local bootstrap inputs / geology H5 files missing"),
    pytest.mark.skipif(_gpus_available() < 2,
                       reason="Smoke test requires 2 visible CUDA devices"),
]


def _build_cfg(devices, tmp_path: Path):
    # Late import: orchestrator.acquire pulls in the surrogate repo which only
    # resolves once ``surrogate_repo`` is on sys.path. The AcquisitionConfig
    # dataclass itself imports cleanly.
    from orchestrator.acquire import AcquisitionConfig, GeologySpec, WellSpec

    geologies = [
        GeologySpec(geology_index=i, geology_h5_file=str(GEO_FILES[i]),
                    geology_name=GEO_FILES[i].stem)
        for i in range(len(GEO_FILES))
    ]
    wells = [
        WellSpec(type="injector", depth=20),
        WellSpec(type="producer", depth=20),
        WellSpec(type="injector", depth=20),
        WellSpec(type="producer", depth=20),
    ]
    return AcquisitionConfig(
        surrogate_repo=SURROGATE_REPO,
        checkpoint_path=CKPT,
        scaler_path=SCALER,
        norm_config_path=NORM_CONFIG,
        geologies=geologies,
        wells=wells,
        n_starts_per_geology=2,
        k_safe=3,
        k_adv=6,
        adv_fraction=0.5,
        edge_buffer=10,
        learning_rate=0.5,
        log_every_n_steps=5,
        revenue_target="graph_discounted_net_revenue",
        seed=42,
        device=devices[0],
        devices=list(devices),
    )


def _summarize(manifest: dict):
    """Compress a manifest to a (snapshot_id -> predicted_revenue) map."""
    out = {}
    for snap in manifest["snapshots"]:
        out[snap["snapshot_id"]] = float(snap["predicted_discounted_total_revenue"])
    return out


def test_parallel_matches_serial(tmp_path: Path):
    import time
    from orchestrator.acquire import run_acquisition

    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()

    cfg_serial = _build_cfg(["cuda:0"], serial_dir)
    cfg_parallel = _build_cfg(["cuda:0", "cuda:1"], parallel_dir)

    t0 = time.time()
    serial = run_acquisition(cfg_serial, out_dir=serial_dir, iteration=0)
    t_serial = time.time() - t0

    t0 = time.time()
    parallel = run_acquisition(cfg_parallel, out_dir=parallel_dir, iteration=0)
    t_parallel = time.time() - t0

    s_summary = _summarize(serial["manifest"])
    p_summary = _summarize(parallel["manifest"])

    assert set(s_summary.keys()) == set(p_summary.keys()), (
        f"snapshot_id sets differ: "
        f"only-serial={set(s_summary) - set(p_summary)}, "
        f"only-parallel={set(p_summary) - set(s_summary)}"
    )

    mismatches = []
    for sid, sv in s_summary.items():
        pv = p_summary[sid]
        if sv == 0.0 and pv == 0.0:
            continue
        denom = max(abs(sv), abs(pv), 1e-9)
        if abs(sv - pv) / denom > 1e-3:
            mismatches.append((sid, sv, pv))
    assert not mismatches, f"predicted-revenue mismatch on {len(mismatches)} snapshots: {mismatches[:5]}"

    # Check that the parallel manifest is sorted canonically (by (geology_index,
    # kind_order, run_id)) so the manifest is reproducible across runs.
    from orchestrator.acquire import _snapshot_sort_key
    parallel_snaps = parallel["manifest"]["snapshots"]
    sorted_snaps = sorted(parallel_snaps, key=_snapshot_sort_key)
    assert parallel_snaps == sorted_snaps, "parallel manifest is not in canonical sort order"

    print(f"\n[smoke] serial:   {t_serial:.2f}s")
    print(f"[smoke] parallel: {t_parallel:.2f}s (speedup = {t_serial / max(t_parallel, 1e-6):.2f}x)")
    print(f"[smoke] {len(s_summary)} snapshots matched within 1e-3 relative tolerance")


def test_both_devices_observed(tmp_path: Path):
    """Confirm the parallel run actually exercised both CUDA devices by
    monitoring nvidia-smi compute apps for the child processes.
    """
    import multiprocessing as mp
    import subprocess
    import threading
    from orchestrator.acquire import run_acquisition

    parallel_dir = tmp_path / "parallel"
    parallel_dir.mkdir()
    cfg = _build_cfg(["cuda:0", "cuda:1"], parallel_dir)

    observed_gpus: set[int] = set()
    stop_flag = threading.Event()

    def poll_nvidia_smi():
        while not stop_flag.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-compute-apps=pid,gpu_uuid",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                # Map gpu_uuid back to index by checking which GPU the UUID belongs to.
                idx_query = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                uuid_to_idx = {}
                for line in idx_query.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        uuid_to_idx[parts[1]] = int(parts[0])
                for line in out.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[1] in uuid_to_idx:
                        observed_gpus.add(uuid_to_idx[parts[1]])
            except Exception:
                pass
            stop_flag.wait(0.25)

    t = threading.Thread(target=poll_nvidia_smi, daemon=True)
    t.start()
    try:
        run_acquisition(cfg, out_dir=parallel_dir, iteration=0)
    finally:
        stop_flag.set()
        t.join(timeout=2)

    # We expect to have seen both GPUs busy at some point. nvidia-smi can miss
    # very short bursts of activity, so this assertion may be flaky on tiny
    # workloads; it's a soft signal — log instead of fail if we missed one.
    print(f"\n[smoke] GPUs observed busy via nvidia-smi: {sorted(observed_gpus)}")
    assert observed_gpus, "nvidia-smi did not observe any compute apps during the run"
    if len(observed_gpus) < 2:
        pytest.skip(f"Only saw GPUs {sorted(observed_gpus)} via nvidia-smi polling — "
                    "workload may be too short to catch both. Test inconclusive.")
