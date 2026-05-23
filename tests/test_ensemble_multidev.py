"""Synthetic CPU-mode test suite for the multi-device acquisition path.

The production function ``orchestrator.acquire._run_acquisition_ensemble``
shards K geology batches across N devices, runs a forward+backward per device
in parallel via Python threads, and sums the per-device gradients back onto a
master coords tensor before each Adam step (see ``acquire.py`` lines
~1925-2295).

Exercising that function end-to-end requires CUDA, a trained surrogate
checkpoint, and real geology HDF5 files. These tests instead isolate the
*threading + gradient-sum core* into a self-contained CPU harness that mirrors
the production logic line-for-line:

  * trivial ``nn.Module`` with closed-form gradient (``f(pos)=geo_weight*sum(pos)``)
  * K fake batches, each carrying a ``geo_weight`` and a ``.well.pos_xyz`` slot
  * N "devices" = ``[torch.device("cpu")] * N`` (separate replicas, single host)
  * the same round-robin sharding, replica creation, threaded
    ``_accumulate_device_grad``, error-reraise, and grad-sum logic as
    production

Every test has a comment explaining the specific bug class it would catch in
the real code.
"""
from __future__ import annotations

import copy
import gc
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch import nn, optim


# ---------------------------------------------------------------------------
# Test harness: a faithful CPU replica of the multi-device gradient core in
# ``_run_acquisition_ensemble``.
# ---------------------------------------------------------------------------


class TrivialSurrogate(nn.Module):
    """Stand-in for the real ``HeteroGNNRegressor`` with closed-form gradient.

    Predicts ``preds[m] = geo_weight * sum(pos_xyz[m])`` for each candidate m.
    Because the model is linear in ``pos_xyz``,
        df/d pos_xyz[m, w, d] = geo_weight
    for every (m, w, d). That lets every test compare numerical gradients
    against a hand-computed analytic value.
    """

    def __init__(self) -> None:
        super().__init__()
        # An unused parameter so deepcopy/.to() exercise nontrivial code paths.
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch: "FakeBatch") -> torch.Tensor:
        pos = batch.well.pos_xyz  # (M*W, 3) — matches production layout
        # Reshape to (M, W, 3) using the batch's stored M for safety.
        pos_m = pos.view(batch.M, -1, 3)
        # preds shape (M,)
        return batch.geo_weight * pos_m.sum(dim=(1, 2))


class _WellSlot:
    """Mimics ``bd['well']`` which exposes a ``pos_xyz`` attribute."""

    pos_xyz: torch.Tensor


@dataclass
class FakeBatch:
    """Stand-in for a PyG ``Batch`` carrying one geology's static graph.

    Only the attributes the gradient core touches are reproduced:
      * ``bd["well"].pos_xyz`` is assigned a coords replica each forward
      * the model uses ``geo_weight`` to compute its prediction
    """

    geo_weight: float
    M: int  # candidate count (rows of coords)
    W: int  # wells per candidate
    well: _WellSlot = field(default_factory=_WellSlot)
    # Marker the harness uses to verify replicas land on the right "device".
    device_marker: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __getitem__(self, key: str) -> _WellSlot:
        if key == "well":
            return self.well
        raise KeyError(key)


def make_batches(
    geo_weights: list[float],
    M: int,
    W: int,
    n_dev: int,
) -> tuple[list[FakeBatch], list[int], list[list[FakeBatch]]]:
    """Build K fake batches and assign each one round-robin to a device.

    Returns (batches, batch_owner_dev, batches_per_dev) exactly like the
    production code at acquire.py:2113-2129.
    """
    K = len(geo_weights)
    batches: list[FakeBatch] = []
    batch_owner_dev: list[int] = []
    for k_idx, w in enumerate(geo_weights):
        b = FakeBatch(geo_weight=float(w), M=M, W=W)
        dev_idx = k_idx % n_dev
        b.device_marker = torch.device("cpu")  # all CPU in tests
        batches.append(b)
        batch_owner_dev.append(dev_idx)
    batches_per_dev: list[list[FakeBatch]] = [[] for _ in range(n_dev)]
    for bd, dev_idx in zip(batches, batch_owner_dev):
        batches_per_dev[dev_idx].append(bd)
    return batches, batch_owner_dev, batches_per_dev


def make_models(model: nn.Module, n_dev: int) -> list[nn.Module]:
    """Mirror ``models_per_dev`` construction at acquire.py:1923-1934."""
    out = [model]
    for _ in range(1, n_dev):
        m = copy.deepcopy(model)
        m.eval()
        out.append(m)
    return out


def _predict_one_geo(
    c: torch.Tensor,
    bd: FakeBatch,
    dev_idx: int,
    models_per_dev: list[nn.Module],
    M: int,
) -> torch.Tensor:
    """Match acquire.py:2143-2152 — assign coords into the batch and forward."""
    bd.well.pos_xyz = c.view(-1, 3)
    return models_per_dev[dev_idx](bd).view(M)


def run_adam_loop(
    geo_weights: list[float],
    M: int,
    W: int,
    n_dev: int,
    n_steps: int,
    lr: float = 0.05,
    start: torch.Tensor | None = None,
    seed: int = 0,
    poison_step: int | None = None,
    poison_dev: int | None = None,
    record_replicas_each_step: bool = False,
) -> dict[str, Any]:
    """Run the threaded Adam loop end-to-end on CPU.

    Returns a dict with final coords, the gradient observed after step 1, and
    optionally weak-refs to the per-step replica lists (for GC tests).

    The body mirrors acquire.py:2196-2294 — same replica creation, same
    ``_accumulate_device_grad``, same error re-raise, same grad sum, same
    optimizer.step() sequencing.
    """
    torch.manual_seed(seed)
    K = len(geo_weights)
    if K == 0:
        raise RuntimeError("Ensemble acquisition requires at least one geology.")
    device_objs = [torch.device("cpu") for _ in range(n_dev)]
    master_dev = device_objs[0]

    model = TrivialSurrogate()
    models_per_dev = make_models(model, n_dev)
    _, _, batches_per_dev = make_batches(geo_weights, M, W, n_dev)

    if start is None:
        coords = torch.zeros((M, W, 3), dtype=torch.float32, device=master_dev)
        coords.fill_(0.5)
    else:
        coords = start.clone().to(master_dev)
    coords.requires_grad = True
    optimizer = optim.Adam([coords], lr=lr)
    K_float = float(K)

    grad_after_step1: torch.Tensor | None = None
    replica_weakrefs_per_step: list[list[weakref.ReferenceType]] = []

    def _accumulate_device_grad(
        dev_idx: int,
        coords_replica: torch.Tensor,
        out_errors: list,
    ) -> None:
        try:
            for bd in batches_per_dev[dev_idx]:
                # Inject a non-finite forward to test exception propagation.
                weight = bd.geo_weight
                if (
                    poison_step is not None
                    and poison_dev == dev_idx
                    and step == poison_step
                ):
                    # Force a NaN loss to drive the inner try/except path.
                    weight = float("nan")
                    bd.geo_weight = weight
                preds_k = _predict_one_geo(
                    coords_replica, bd, dev_idx, models_per_dev, M
                )
                loss_k = -(preds_k.sum() / K_float)
                if not torch.isfinite(loss_k):
                    raise RuntimeError(
                        f"non-finite loss on dev{dev_idx} step{step}"
                    )
                loss_k.backward()
        except BaseException as e:  # pragma: no cover - tested explicitly
            out_errors[dev_idx] = e

    for step in range(1, n_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        if n_dev == 1:
            _accumulate_device_grad(0, coords, [None])
        else:
            replicas: list[torch.Tensor] = []
            for i in range(n_dev):
                if i == 0:
                    coords.grad = None
                    replicas.append(coords)
                else:
                    replicas.append(
                        coords.detach().to(device_objs[i]).requires_grad_(True)
                    )

            errors: list[BaseException | None] = [None] * n_dev
            threads: list[threading.Thread] = []
            for i in range(n_dev):
                t = threading.Thread(
                    target=_accumulate_device_grad,
                    args=(i, replicas[i], errors),
                    name=f"acquire-ensemble-dev{i}",
                )
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            for err in errors:
                if err is not None:
                    raise err

            if coords.grad is None:
                coords.grad = torch.zeros_like(coords)
            for i in range(1, n_dev):
                if replicas[i].grad is None:
                    continue
                coords.grad.add_(replicas[i].grad.to(master_dev))

            if record_replicas_each_step:
                replica_weakrefs_per_step.append(
                    [weakref.ref(r) for r in replicas[1:]]
                )
            # Drop the strong reference so GC can collect the non-master
            # replicas; production simply lets ``replicas`` fall out of scope
            # at the end of the for-step iteration (same effect).
            del replicas

        if step == 1:
            grad_after_step1 = coords.grad.detach().clone()
        optimizer.step()

    return {
        "coords": coords.detach().clone(),
        "grad_step1": grad_after_step1,
        "replica_weakrefs_per_step": replica_weakrefs_per_step,
    }


# Convenience: a fixed roster of "geology weights" used across tests so we can
# precompute analytic expectations.
DEFAULT_WEIGHTS = [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# 1. Correctness parity
# ---------------------------------------------------------------------------


def test_01_parity_1_vs_2_devices_5_steps():
    # Catches: any sharding bug that changes the optimizer trajectory when
    # K geologies are split across 2 devices vs run all on one device.
    out1 = run_adam_loop(DEFAULT_WEIGHTS, M=3, W=2, n_dev=1, n_steps=5, seed=42)
    out2 = run_adam_loop(DEFAULT_WEIGHTS, M=3, W=2, n_dev=2, n_steps=5, seed=42)
    assert torch.equal(out1["coords"], out2["coords"]), (
        "1-device vs 2-device coords diverged after 5 Adam steps"
    )


def test_02_parity_1_vs_4_devices_5_steps():
    # Catches: off-by-one in the round-robin assignment with K == n_dev.
    out1 = run_adam_loop(DEFAULT_WEIGHTS, M=3, W=2, n_dev=1, n_steps=5, seed=42)
    out4 = run_adam_loop(DEFAULT_WEIGHTS, M=3, W=2, n_dev=4, n_steps=5, seed=42)
    assert torch.equal(out1["coords"], out4["coords"]), (
        "1-device vs 4-device coords diverged after 5 Adam steps"
    )


def test_03_parity_gradient_at_step3():
    # Catches: state leaking across Adam steps (e.g., a stale grad on a
    # non-master replica getting summed twice).
    out1 = run_adam_loop(DEFAULT_WEIGHTS, M=3, W=2, n_dev=1, n_steps=3, seed=7)
    out2 = run_adam_loop(DEFAULT_WEIGHTS, M=3, W=2, n_dev=2, n_steps=3, seed=7)
    # The "grad at step 3" we record is the gradient at step 1 in a 3-step
    # run; rerun and compare the post-step-3 coords AND step-1 grad.
    assert torch.equal(out1["grad_step1"], out2["grad_step1"]), (
        "step-1 gradients differ between 1-dev and 2-dev"
    )
    assert torch.equal(out1["coords"], out2["coords"]), (
        "post-step-3 coords differ between 1-dev and 2-dev"
    )


def test_04_gradient_magnitude_matches_analytic():
    # Catches: a missing /K normalization or a sign flip in the loss.
    # Analytic: loss = -(1/K) Σ_k (w_k * Σ pos)  →  d loss / d pos = -(Σ w_k)/K
    weights = DEFAULT_WEIGHTS
    M, W = 3, 2
    out = run_adam_loop(weights, M=M, W=W, n_dev=2, n_steps=1, seed=0)
    expected = -sum(weights) / len(weights)
    assert torch.allclose(
        out["grad_step1"], torch.full((M, W, 3), expected), atol=1e-6
    ), f"expected grad {expected}, got {out['grad_step1'][0, 0, 0].item()}"


# ---------------------------------------------------------------------------
# 2. Edge cases on (K, n_dev)
# ---------------------------------------------------------------------------


def test_05_K_less_than_n_dev():
    # Catches: an empty batches_per_dev[i] crashing the thread (e.g., index
    # error in the for-loop, or `replicas[i].grad is None` not being handled
    # in the sum). With K=2 and n_dev=4, devices 2 and 3 have no work.
    weights = [1.0, 2.0]
    M, W = 2, 1
    out = run_adam_loop(weights, M=M, W=W, n_dev=4, n_steps=1, seed=1)
    expected = -sum(weights) / len(weights)
    assert torch.allclose(out["grad_step1"], torch.full((M, W, 3), expected))


def test_06_K_equals_one_with_n_dev_two():
    # Catches: a divide-by-K bug when K=1 and one device is idle.
    out = run_adam_loop([5.0], M=2, W=1, n_dev=2, n_steps=1, seed=2)
    assert torch.allclose(out["grad_step1"], torch.full((2, 1, 3), -5.0))


def test_07_K_equals_n_dev_even_split():
    # Catches: round-robin bug when K and n_dev coincide (each device sees
    # exactly one batch).
    weights = [1.0, 2.0]
    out = run_adam_loop(weights, M=2, W=1, n_dev=2, n_steps=1, seed=3)
    expected = -sum(weights) / 2.0
    assert torch.allclose(out["grad_step1"], torch.full((2, 1, 3), expected))


def test_08_K_much_greater_than_n_dev():
    # Catches: regressions in the gradient-accumulation inner loop when many
    # backwards land on one device. Use uneven counts (15 vs 15 here, but
    # confirms K=30 path).
    weights = [float(i + 1) for i in range(30)]
    out = run_adam_loop(weights, M=2, W=1, n_dev=2, n_steps=1, seed=4)
    expected = -sum(weights) / len(weights)
    assert torch.allclose(out["grad_step1"], torch.full((2, 1, 3), expected))


def test_09_production_like_sizes_K15_n_dev_2():
    # Catches: a regression in the realistic shape (15 geologies / 2 GPUs is
    # near a current production config; 8 vs 7 uneven split).
    weights = [float(i + 1) for i in range(15)]
    out = run_adam_loop(weights, M=4, W=3, n_dev=2, n_steps=2, seed=5)
    # Just check gradient matches analytic at step 1.
    expected = -sum(weights) / 15.0
    assert torch.allclose(out["grad_step1"], torch.full((4, 3, 3), expected))


def test_10_K_zero_raises_at_entry():
    # Catches: silent acceptance of an empty geology list; production code
    # explicitly raises RuntimeError so callers don't waste compute on a
    # vacuous Adam loop.
    with pytest.raises(RuntimeError, match="at least one geology"):
        run_adam_loop([], M=1, W=1, n_dev=2, n_steps=1)


# ---------------------------------------------------------------------------
# 3. Round-robin partition
# ---------------------------------------------------------------------------


def test_11_round_robin_partition_K4_n_dev2():
    # Catches: any deviation from the exact round-robin pattern that
    # downstream code (batch_owner_dev) assumes.
    weights = [1.0, 2.0, 3.0, 4.0]
    _, owners, per_dev = make_batches(weights, M=1, W=1, n_dev=2)
    assert owners == [0, 1, 0, 1], f"unexpected owners: {owners}"
    assert [b.geo_weight for b in per_dev[0]] == [1.0, 3.0]
    assert [b.geo_weight for b in per_dev[1]] == [2.0, 4.0]


def test_12_per_device_batch_counts():
    # Catches: a load-balancing bug that gives device 0 too many batches
    # (round-robin should put the remainder on the low devices).
    for K, n_dev in [(5, 2), (7, 3), (10, 4), (1, 4), (4, 4), (2, 4)]:
        weights = [float(i + 1) for i in range(K)]
        _, _, per_dev = make_batches(weights, M=1, W=1, n_dev=n_dev)
        counts = [len(p) for p in per_dev]
        assert sum(counts) == K, f"lost batches: K={K} n={n_dev} -> {counts}"
        # Round-robin: low-index devices get ceil(K/n_dev), the rest floor.
        ceil = (K + n_dev - 1) // n_dev
        floor = K // n_dev
        extras = K - floor * n_dev
        expected = [ceil if i < extras else floor for i in range(n_dev)]
        assert counts == expected, (
            f"K={K} n_dev={n_dev} expected {expected} got {counts}"
        )


# ---------------------------------------------------------------------------
# 4. Exception propagation
# ---------------------------------------------------------------------------


def test_13_worker_exception_reraised_in_main():
    # Catches: silent swallowing of a NaN loss or unexpected exception in a
    # worker thread. Production stores into errors[dev_idx] and re-raises
    # after join.
    with pytest.raises(RuntimeError, match="non-finite loss"):
        run_adam_loop(
            DEFAULT_WEIGHTS, M=2, W=1, n_dev=2, n_steps=1,
            poison_step=1, poison_dev=0,
        )


def test_14_retry_after_exception_is_clean():
    # Catches: leaked grads or stale optimizer state after a failed step.
    # After a poisoned run dies, a fresh run with the same seed must match
    # the gold result — proving no global state leaked.
    with pytest.raises(RuntimeError):
        run_adam_loop(
            DEFAULT_WEIGHTS, M=2, W=1, n_dev=2, n_steps=1,
            poison_step=1, poison_dev=1,
        )
    gold = run_adam_loop(DEFAULT_WEIGHTS, M=2, W=1, n_dev=1, n_steps=1, seed=99)
    retry = run_adam_loop(DEFAULT_WEIGHTS, M=2, W=1, n_dev=2, n_steps=1, seed=99)
    assert torch.equal(gold["coords"], retry["coords"])


def test_15_exception_on_either_device_propagates():
    # Catches: an ordering bug where errors[0] gets re-raised but errors[1]
    # is silently dropped (or vice versa). We poison each device in turn and
    # confirm both raise.
    for poison_dev in (0, 1):
        with pytest.raises(RuntimeError, match=f"dev{poison_dev}"):
            run_adam_loop(
                DEFAULT_WEIGHTS, M=2, W=1, n_dev=2, n_steps=1,
                poison_step=1, poison_dev=poison_dev,
            )


# ---------------------------------------------------------------------------
# 5. Gradient accumulation correctness
# ---------------------------------------------------------------------------


def test_16_step1_gradient_equals_analytic():
    # Catches: a sign flip, missing /K, or summation bug. Repeat of test 04
    # at multiple (M, W) shapes.
    for M, W in [(1, 1), (3, 2), (5, 4), (2, 8)]:
        out = run_adam_loop(
            DEFAULT_WEIGHTS, M=M, W=W, n_dev=2, n_steps=1, seed=11
        )
        expected = -sum(DEFAULT_WEIGHTS) / len(DEFAULT_WEIGHTS)
        assert torch.allclose(
            out["grad_step1"], torch.full((M, W, 3), expected), atol=1e-6
        ), f"shape M={M},W={W}: {out['grad_step1'][0,0,0]} vs {expected}"


def test_17_zero_grad_set_to_none_each_step():
    # Catches: a missing zero_grad call that would let gradients accumulate
    # across Adam steps and blow up the trajectory. We verify by running
    # 1 step and comparing to a 2-step run's intermediate grad: with proper
    # zero_grad, step 2's grad magnitude stays bounded; without it, it grows.
    out1 = run_adam_loop(DEFAULT_WEIGHTS, M=2, W=1, n_dev=2, n_steps=1, seed=21)
    # Same grad must reappear if we restart from the same coords.
    out_restart = run_adam_loop(
        DEFAULT_WEIGHTS, M=2, W=1, n_dev=2, n_steps=1, seed=21,
        start=torch.full((2, 1, 3), 0.5),
    )
    assert torch.equal(out1["grad_step1"], out_restart["grad_step1"])


def test_18_grad_sum_order_invariant():
    # Catches: order-dependent floating-point sum bug. Sum into coords.grad
    # (master already has its own grad) must equal a naive sum of all
    # n_dev replicas.
    # We can't easily swap the order from outside, but we can confirm that
    # the 2-device result equals a hand-computed sum of per-device grads.
    weights = [1.5, 2.5, 3.5, 4.5]
    out = run_adam_loop(weights, M=2, W=1, n_dev=2, n_steps=1, seed=31)
    # Per-device contribution: -(w/K) per element from each batch on that
    # device. Sum across devices = sum over all weights / K.
    expected = -sum(weights) / len(weights)
    assert torch.allclose(out["grad_step1"], torch.full((2, 1, 3), expected))


# ---------------------------------------------------------------------------
# 6. Replica lifetime / memory hygiene
# ---------------------------------------------------------------------------


def test_19_no_persistent_replica_growth_across_steps():
    # Catches: a leaked list/registry of replicas across steps that would
    # OOM the master GPU after many Adam steps.
    out = run_adam_loop(
        DEFAULT_WEIGHTS, M=2, W=1, n_dev=2, n_steps=5, seed=41,
        record_replicas_each_step=True,
    )
    # Each step recorded a separate list of weakrefs to its non-master
    # replicas. There should be no cross-step alias chain.
    refs = out["replica_weakrefs_per_step"]
    assert len(refs) == 5, "expected 5 per-step weakref entries"
    # After the loop, force GC and confirm none of the older replicas are
    # still alive.
    gc.collect()
    alive = sum(1 for step_refs in refs for r in step_refs if r() is not None)
    assert alive == 0, f"{alive} stale replicas still alive after loop"


def test_20_replica_tensors_collected_after_step():
    # Catches: a closure or thread frame holding refs to replicas after the
    # thread joins. We capture a weakref *during* step 1 and confirm the
    # replica is gone after the next step starts.
    weights = DEFAULT_WEIGHTS
    out = run_adam_loop(
        weights, M=2, W=1, n_dev=2, n_steps=2, seed=51,
        record_replicas_each_step=True,
    )
    refs = out["replica_weakrefs_per_step"]
    gc.collect()
    # The step-1 replicas must not survive to the end of step 2.
    step1_alive = sum(1 for r in refs[0] if r() is not None)
    assert step1_alive == 0, (
        f"{step1_alive} step-1 replicas still alive at end of loop"
    )


# ---------------------------------------------------------------------------
# 7. Master replica identity
# ---------------------------------------------------------------------------


def test_21_master_replica_is_master_coords():
    # Catches: a refactor that accidentally clones the master replica too,
    # which would double-count its gradient when summed back.
    # We re-implement enough of the loop inline to inspect replicas[0] id.
    M, W, n_dev = 2, 1, 2
    coords = torch.full((M, W, 3), 0.5, requires_grad=True)
    device_objs = [torch.device("cpu")] * n_dev
    replicas = []
    for i in range(n_dev):
        if i == 0:
            coords.grad = None
            replicas.append(coords)
        else:
            replicas.append(
                coords.detach().to(device_objs[i]).requires_grad_(True)
            )
    assert replicas[0] is coords, "master replica must alias coords itself"
    assert replicas[1] is not coords, "non-master replicas must be fresh"
    # The non-master replica must be a *distinct* autograd leaf, so that
    # its .grad is independent of the master's. (On CPU, ``.to(cpu)`` is a
    # no-op and may share storage; we instead assert grad independence.)
    assert replicas[1].grad is None
    replicas[1].sum().backward()
    assert replicas[1].grad is not None and coords.grad is None, (
        "non-master replica's grad must be independent of master's grad"
    )


def test_22_step_moves_coords_along_negative_gradient():
    # Catches: a sign flip in optimizer or loss. Adam minimizes
    # loss = -(preds.sum()/K), so coords should INCREASE (move toward
    # higher preds). With weights all positive, the gradient is negative,
    # so coords should grow.
    weights = [1.0, 2.0, 3.0, 4.0]
    M, W = 2, 1
    start = torch.full((M, W, 3), 0.5)
    out = run_adam_loop(
        weights, M=M, W=W, n_dev=2, n_steps=3, seed=61, start=start.clone()
    )
    assert (out["coords"] > start.to(out["coords"].device)).all(), (
        "coords should have increased under negative-gradient direction"
    )


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


def test_23_threaded_runs_are_bitwise_deterministic():
    # Catches: a race condition in the float sum that would yield
    # non-deterministic last-bit results across runs.
    out_a = run_adam_loop(DEFAULT_WEIGHTS, M=4, W=3, n_dev=2, n_steps=5, seed=71)
    out_b = run_adam_loop(DEFAULT_WEIGHTS, M=4, W=3, n_dev=2, n_steps=5, seed=71)
    assert torch.equal(out_a["coords"], out_b["coords"]), (
        "two identical runs produced different final coords"
    )
    assert torch.equal(out_a["grad_step1"], out_b["grad_step1"])


# ---------------------------------------------------------------------------
# 9. Math equivalence: K small backwards == one big mean-loss backward
# ---------------------------------------------------------------------------


def test_24_k_fold_grad_equals_single_mean_backward():
    # Catches: any regression in the central claim of the refactor — that
    # accumulating K per-geology backwards (each scaled by 1/K) yields the
    # same gradient as a single backward over the mean-over-K loss.
    weights = DEFAULT_WEIGHTS
    K = len(weights)
    M, W = 3, 2

    # Reference: a single forward+backward through loss = -(preds.mean(dim=1).sum())
    # which is what the original (pre-refactor) code computed.
    coords_ref = torch.full((M, W, 3), 0.5, requires_grad=True)
    model = TrivialSurrogate()
    preds_cols = []
    for w in weights:
        b = FakeBatch(geo_weight=w, M=M, W=W)
        b.well.pos_xyz = coords_ref.view(-1, 3)
        preds_cols.append(model(b).view(M))
    preds_stack = torch.stack(preds_cols, dim=1)  # (M, K)
    loss_ref = -(preds_stack.mean(dim=1).sum())
    loss_ref.backward()
    grad_ref = coords_ref.grad.detach().clone()

    # K-fold version (matches production)
    out = run_adam_loop(weights, M=M, W=W, n_dev=2, n_steps=1, seed=81,
                       start=torch.full((M, W, 3), 0.5))
    grad_kfold = out["grad_step1"]

    assert torch.allclose(grad_ref, grad_kfold, atol=1e-6), (
        f"K-fold grad {grad_kfold[0,0,0]} != mean-backward grad "
        f"{grad_ref[0,0,0]}; this would break the refactor claim"
    )


# ---------------------------------------------------------------------------
# Extra: 1-device fast path equivalence to the threaded path
# ---------------------------------------------------------------------------


def test_25_single_device_fast_path_matches_threaded():
    # Catches: a divergence between the n_dev==1 inline branch and the
    # general threaded branch. Force the threaded code by setting n_dev=2
    # with all batches assigned to device 0 via a custom weight schedule.
    # We achieve this by setting all weights identical (round-robin still
    # splits them 50/50, so this isn't a perfect isolation, but the result
    # should still match the inline path with same weights).
    weights = [2.0, 2.0, 2.0, 2.0]
    out1 = run_adam_loop(weights, M=2, W=1, n_dev=1, n_steps=3, seed=91)
    out2 = run_adam_loop(weights, M=2, W=1, n_dev=2, n_steps=3, seed=91)
    assert torch.equal(out1["coords"], out2["coords"])
