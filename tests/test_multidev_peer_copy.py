"""Multi-GPU peer-copy bug regression tests.

Background
----------
On a 2x A100 PCIe machine (driver 595.71.05, PyTorch 2.11.0+cu128),
``tensor.to("cuda:1")`` issued from a ``cuda:0`` source silently returns a
zero-filled tensor on this host. ``torch.cuda.can_device_access_peer`` reports
True, so the copy goes through the (broken) peer path rather than staging
through CPU. This corrupted the ``target_mean`` / ``target_scale`` replicas in
``orchestrator/acquire.py:_run_acquisition_ensemble`` and made every prediction
on cuda:1 collapse to ~0, ruining the latest 4 iterations of an AL run.

The fix routes those scaler transfers through CPU
(``tensor.cpu().to(device)``) and adds an inline runtime check comparing the
replica's abs-sum against the master's at acquire.py ~line 2013.

The tests below come in two layers:

1. ``test_direct_peer_copy_zeros_bug`` — a direct guard against the broken
   environment. If ``cuda:0 -> cuda:1`` silently zero-fills on this box, the
   test xfails with a clear message rather than silently miscomputing.
2. ``test_scaler_replication_via_cpu_preserves_values`` — locks in the
   workaround used by ``acquire.py`` (route scaler tensors through CPU before
   landing on the replica device).

Both tests skip when fewer than 2 CUDA devices are visible.
"""
from __future__ import annotations

import pytest
import torch


def _cuda_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


pytestmark = pytest.mark.skipif(
    _cuda_device_count() < 2,
    reason="Requires >=2 CUDA devices to exercise cross-device peer copy",
)


# ---------------------------------------------------------------------------
# 1. Direct-bug test: detect the cuda:0 -> cuda:1 zero-fill at the environment
#    level. Not a guard against our code; a guard against PyTorch/driver.
# ---------------------------------------------------------------------------


def test_direct_peer_copy_zeros_bug():
    """Detect the underlying ``cuda:0 -> cuda:1`` silent zero-fill.

    Procedure:
      * place a known non-zero tensor on cuda:0
      * try ``.to("cuda:1")`` directly (the broken path) and capture whether
        the value survives
      * try ``.cpu().to("cuda:1")`` (the workaround) and assert *that* works
        so we can distinguish "PyTorch is broken" from "transfer is broken"

    If the direct GPU->GPU copy is broken on this host, the test xfails
    with a clear message — the point is to *surface* the issue, not fail CI
    on a known-bad environment.
    """
    src_val = 3.7e8  # arbitrary distinctive non-zero scalar, like real scaler
    src = torch.tensor([src_val], dtype=torch.float64, device="cuda:0")

    # Sanity: the source has the value we set.
    src_abs_sum = float(src.detach().abs().sum().cpu())
    assert src_abs_sum == pytest.approx(src_val, rel=1e-12), (
        f"source tensor on cuda:0 is wrong before any copy: {src_abs_sum}"
    )

    # IMPORTANT: the direct GPU->GPU copy MUST be probed BEFORE any
    # CPU-staged copy in the same process. On this box, executing a
    # ``.cpu().to(other_cuda)`` first appears to lazily initialise peer
    # access and "fix" subsequent direct copies — masking the bug. So we
    # check the suspect path first.
    direct = src.to("cuda:1")
    torch.cuda.synchronize("cuda:1")
    direct_abs_sum = float(direct.detach().abs().sum().cpu())

    # Use a fresh source tensor so the workaround test is independent of
    # any state the direct path may have mutated.
    src2 = torch.tensor([src_val], dtype=torch.float64, device="cuda:0")
    via_cpu = src2.cpu().to("cuda:1")
    torch.cuda.synchronize("cuda:1")
    via_cpu_abs_sum = float(via_cpu.detach().abs().sum().cpu())
    assert via_cpu_abs_sum == pytest.approx(src_val, rel=1e-12), (
        f"CPU-staged copy from cuda:0 to cuda:1 lost the value: "
        f"got {via_cpu_abs_sum}, expected {src_val}. This means even the "
        f"workaround is broken — investigate PyTorch/CUDA install."
    )

    if direct_abs_sum == pytest.approx(src_val, rel=1e-12):
        # Direct peer copy works on this box — great. The .cpu() workaround
        # in acquire.py is then defensive but harmless.
        return

    if direct_abs_sum == 0.0:
        pytest.xfail(
            f"GPU peer copy is broken on this box: cuda:0 -> cuda:1 silently "
            f"returned zeros (expected {src_val}). Apply the .cpu() "
            f"workaround for every cross-device tensor transfer (see "
            f"acquire.py:1988-1989). The .cpu().to(cuda:1) path "
            f"(abs-sum {via_cpu_abs_sum}) still works."
        )

    # Some other corruption mode — surface it loudly.
    pytest.fail(
        f"cuda:0 -> cuda:1 direct copy produced unexpected value "
        f"{direct_abs_sum} (expected {src_val}, or 0.0 for the known "
        f"zero-fill bug). Investigate before trusting any cross-device "
        f"tensor move on this host."
    )


# ---------------------------------------------------------------------------
# 2. Integration / regression: the acquire.py scaler-replication workaround.
#    Locks in that ``target_mean``/``target_scale`` survive the move to each
#    replica device when we route through CPU. Doesn't load a real model or
#    a real scaler — the regression is purely "does the tensor survive the
#    documented transfer pattern".
# ---------------------------------------------------------------------------


def test_scaler_replication_via_cpu_preserves_values():
    """Replicate the acquire.py scaler-move logic and assert value survival.

    Mirrors the production snippet at ``orchestrator/acquire.py:1988-1989``::

        target_scales_per_dev.append(target_scale.cpu().to(d))
        target_means_per_dev.append(target_mean.cpu().to(d))

    plus the inline abs-sum drift check at ``acquire.py:~2013``. The point is
    to catch a regression that re-introduces a direct ``target_mean.to(d)``
    call without staging through CPU.
    """
    n_dev = _cuda_device_count()
    assert n_dev >= 2

    # Master scaler tensors live on cuda:0, with realistic magnitudes drawn
    # from the actual surrogate (energy in J/year, ~1e8). Using float64 to
    # match the saved scaler dtype.
    target_mean = torch.tensor([3.7e8], dtype=torch.float64, device="cuda:0")
    target_scale = torch.tensor([1.25e8], dtype=torch.float64, device="cuda:0")

    device_objs = [torch.device("cuda:0")]
    target_means_per_dev = [target_mean]
    target_scales_per_dev = [target_scale]

    for i in range(1, n_dev):
        d = torch.device(f"cuda:{i}")
        device_objs.append(d)
        # EXACT line from acquire.py — do not change without updating both.
        target_scales_per_dev.append(target_scale.cpu().to(d))
        target_means_per_dev.append(target_mean.cpu().to(d))
        torch.cuda.synchronize(d)

    # Inline drift check, mirroring acquire.py:~2019-2029.
    mm = float(target_mean.detach().abs().sum().cpu())
    ms = float(target_scale.detach().abs().sum().cpu())
    for i in range(1, n_dev):
        rm = float(target_means_per_dev[i].detach().abs().sum().cpu())
        rs = float(target_scales_per_dev[i].detach().abs().sum().cpu())

        assert target_means_per_dev[i].device == device_objs[i], (
            f"replica target_mean landed on {target_means_per_dev[i].device}, "
            f"expected {device_objs[i]}"
        )
        assert target_scales_per_dev[i].device == device_objs[i], (
            f"replica target_scale landed on {target_scales_per_dev[i].device}, "
            f"expected {device_objs[i]}"
        )

        # Strong equality on values: anything else means the CPU-staged
        # transfer dropped data.
        assert torch.equal(
            target_means_per_dev[i].cpu(), target_mean.cpu()
        ), (
            f"target_mean replica on {device_objs[i]} (abs-sum {rm:.6g}) "
            f"does not equal master (abs-sum {mm:.6g}). This is the "
            f"GPU->GPU peer-copy silently-zeros bug returning — the "
            f"replica must come from .cpu().to(d), not .to(d)."
        )
        assert torch.equal(
            target_scales_per_dev[i].cpu(), target_scale.cpu()
        ), (
            f"target_scale replica on {device_objs[i]} (abs-sum {rs:.6g}) "
            f"does not equal master (abs-sum {ms:.6g}). This is the "
            f"GPU->GPU peer-copy silently-zeros bug returning — the "
            f"replica must come from .cpu().to(d), not .to(d)."
        )

        # Match the production tolerance form, in case the dtype/precision
        # ever loosens to float32.
        assert abs(rm - mm) < max(1e-3, 1e-5 * abs(mm)), (
            f"target_mean drifted on {device_objs[i]}: {rm:.6g} vs {mm:.6g}"
        )
        assert abs(rs - ms) < max(1e-3, 1e-5 * abs(ms)), (
            f"target_scale drifted on {device_objs[i]}: {rs:.6g} vs {ms:.6g}"
        )


def test_direct_to_other_cuda_would_be_caught_by_drift_check(tmp_path):
    """Confirm the inline drift check fires if someone reverts the fix.

    Simulates the broken pattern (``target_mean.to(d)`` without ``.cpu()``)
    and asserts that *if* this box exhibits the zero-fill bug, the drift
    check at acquire.py:~2023 would catch it. On a host where direct peer
    copy works, this test is a no-op.

    Runs in a **fresh subprocess** because once any CPU-staged copy has
    happened in a process (as ``test_direct_peer_copy_zeros_bug`` does),
    PyTorch lazily initialises peer access and subsequent direct copies
    succeed — masking the bug. A clean process is the only way to
    re-exercise the suspect path inside the same pytest run.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import torch
        if torch.cuda.device_count() < 2:
            print("SKIP")
            sys.exit(0)
        target_mean = torch.tensor([3.7e8], dtype=torch.float64, device="cuda:0")
        mm = float(target_mean.detach().abs().sum().cpu())
        broken = target_mean.to("cuda:1")
        torch.cuda.synchronize("cuda:1")
        rm = float(broken.detach().abs().sum().cpu())
        drift_tol = max(1e-3, 1e-5 * abs(mm))
        drift_caught = not (abs(rm - mm) < drift_tol)
        print(f"mm={mm}")
        print(f"rm={rm}")
        print(f"drift_tol={drift_tol}")
        print(f"drift_caught={drift_caught}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed: rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    if result.stdout.strip() == "SKIP":
        pytest.skip("subprocess saw <2 CUDA devices")

    out = dict(
        line.split("=", 1) for line in result.stdout.strip().splitlines() if "=" in line
    )
    mm = float(out["mm"])
    rm = float(out["rm"])
    drift_caught = out["drift_caught"] == "True"

    if rm == pytest.approx(mm, rel=1e-12):
        # Box is fine; direct copy works. The acquire.py workaround is
        # defensive only on this host.
        pytest.skip(
            "Direct cuda:0 -> cuda:1 copy works on a fresh process here — "
            "drift check has nothing to catch. The .cpu() workaround in "
            "acquire.py is defensive, not load-bearing on this box."
        )

    # Box is broken in a fresh process. The drift check MUST fire.
    assert drift_caught, (
        f"Direct cuda:0 -> cuda:1 copy returned {rm:.6g} (master {mm:.6g}) "
        f"but the production drift-check tolerance did not catch it. "
        f"Tighten the check in acquire.py:~2023."
    )
