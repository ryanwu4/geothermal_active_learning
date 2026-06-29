"""Panel-gated INTERSECT sampling — decide, per geology, whether to run IX this iteration.

The AL loop's dominant cost is running every candidate well-config in all K geologies to
get a ground-truth EMV (mean revenue over the geology ensemble). Offline analysis showed the
surrogate's per-geology error is heteroscedastic: a small hard cluster stays miscalibrated
while most geologies fall to ~1-3% MAPE within a couple iterations. This module implements a
**causal MAPE-cutoff gate** that exploits that structure:

  1. Each iteration, run a config in IX only for geologies whose LAST-MEASURED per-geology
     MAPE exceeds ``mape_cutoff_pct`` (the "panel"). Geologies at/below the cutoff are
     surrogate-filled (no IX task emitted). "Last-measured" is read from prior iterations'
     ``state.history`` — never the current iteration — so the gate is causal: you cannot
     know a geology's MAPE this iteration until after its IX has run.
  2. Iteration 0 (and any iteration with no MAPE history) runs the full ensemble. The cutoff
     self-shrinks the panel as the surrogate calibrates.
  3. Rank configs by a control-variate estimate ``EMV_hat`` = mean over K of
     ``[real if panel-geo else surrogate prediction]``, take the top-M, and complete those M
     configs in the remaining ("fill") geologies. The incumbent updates only from these
     fully-completed configs, so it stays ground-truth (the strict-EMV rule in
     :mod:`orchestrator.emv` throws out any config with < K real geologies).

The gate is a **revenue**-MAPE / revenue-EMV gate even on NPV runs (NPV drives acquisition
only) — this matches the validated offline replay
(``analysis/ix_savings_panel/make_panel_heatmap.py``) and keeps the remote ingest cube-free.

This module is pure and I/O-light (only :func:`build_completion_manifest` touches the
filesystem, and only to write); it mirrors the dataclass+pure-function style of
:mod:`orchestrator.emv` and is unit/golden-replay tested in ``tests/test_gating.py``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _is_finite(v: Any) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


@dataclass
class GeologyGatingConfig:
    """Parsed from a top-level ``"geology_gating"`` block in the run config.

    Defaults reproduce the validated offline replay (causal, tau=5%, M=3) and leave the
    feature OFF, so an absent/disabled block is a guaranteed no-op.
    """

    enabled: bool = False
    mape_cutoff_pct: float = 5.0     # tau; a geology is queried while its MAPE exceeds this
    top_m: int = 3                   # configs completed to the full ensemble each iteration
    warmup_iters: int = 1            # leading iterations forced to the full ensemble
    audit_configs: int = 0           # RESERVED for a drift guard (random filled-geo completions);
    audit_seed: int = 0              #   NOT YET IMPLEMENTED — must be 0 (validate() enforces).
    # Per-geology metric to gate on. "mape" = raw |pred-real|/|real|, matching the validated
    # replay (analysis/ix_savings_panel/make_panel_heatmap.py) and more conservative on the
    # hard low-revenue geologies (geo-8-like) than the floored variant, which shrinks their
    # metric and would drop them from the panel sooner — exactly where the surrogate is worst.
    mape_key: str = "mape"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "GeologyGatingConfig":
        block: Mapping[str, Any] = {}
        if cfg:
            block = cfg.get("geology_gating") or {}
        known = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in dict(block).items() if k in known}
        out = cls(**kwargs)
        out.validate()
        return out

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.top_m < 1:
            raise ValueError("geology_gating.top_m must be >= 1 when enabled")
        if self.mape_cutoff_pct < 0:
            raise ValueError("geology_gating.mape_cutoff_pct must be >= 0")
        if self.warmup_iters < 0:
            raise ValueError("geology_gating.warmup_iters must be >= 0")
        if self.audit_configs != 0:
            # The drift guard is designed but not implemented; refuse rather than silently
            # ignore it so a run can't believe it has drift protection that doesn't run.
            raise NotImplementedError(
                "geology_gating.audit_configs is reserved but not yet implemented; set it to 0. "
                "Filled geologies are currently re-measured only via the top-M completions."
            )


@dataclass
class PanelDecision:
    """The geologies to run in IX this iteration, plus the signal that drove the choice."""

    panel: set[int]                          # geology_index values to RUN in IX
    last_mape_by_geo: dict[int, float | None]  # causal signal used (fraction, not pct)
    reason: str                              # "disabled" | "warmup" | "cold_start" | "gated"
    is_full: bool                            # panel == full ensemble (gate is a no-op this iter)


def _last_measured_mape(
    per_geology_history: Sequence[Mapping[str, Mapping[str, Any]] | None],
    all_geology_indices: Sequence[int],
    mape_key: str,
) -> dict[int, float | None]:
    """Most-recent finite per-geology MAPE for each geology, walking history newest->oldest.

    ``per_geology_history`` is ``[state.history[i].per_geology, ...]`` oldest->newest; each
    entry is keyed by ``str(geology_index)``. Returns the value of ``mape_key`` (falling back
    to ``"mape"``) from the most recent iteration in which that geology was actually measured,
    or ``None`` if it has never been measured. Values are fractions (e.g. 0.04 == 4%).
    """
    last: dict[int, float | None] = {int(g): None for g in all_geology_indices}
    for rec in reversed(list(per_geology_history)):
        if not rec:
            continue
        for g in last:
            if last[g] is not None:
                continue
            entry = rec.get(str(g))
            if not entry:
                continue
            v = entry.get(mape_key)
            if v is None:
                v = entry.get("mape")  # robust to records that pre-date mape_floored
            if _is_finite(v):
                last[g] = float(v)
        if all(v is not None for v in last.values()):
            break
    return last


def compute_panel(
    *,
    per_geology_history: Sequence[Mapping[str, Mapping[str, Any]] | None],
    all_geology_indices: Sequence[int],
    iteration: int,
    gcfg: GeologyGatingConfig,
) -> PanelDecision:
    """Decide which geologies to run in IX this iteration from prior MAPE (causal).

    Keys on the actual ``geology_index`` values (which may be non-contiguous), never
    ``range(K)``. A geology with no recent measurement is conservatively kept in the panel.
    An empty panel (full convergence) is valid: ranking falls back to the surrogate and the
    top-M are still completed in the full ensemble.
    """
    all_idx = [int(g) for g in all_geology_indices]
    full = set(all_idx)

    if not gcfg.enabled:
        return PanelDecision(panel=set(full), last_mape_by_geo={g: None for g in all_idx},
                             reason="disabled", is_full=True)

    last = _last_measured_mape(per_geology_history, all_idx, gcfg.mape_key)

    if iteration < gcfg.warmup_iters or all(v is None for v in last.values()):
        reason = "warmup" if iteration < gcfg.warmup_iters else "cold_start"
        return PanelDecision(panel=set(full), last_mape_by_geo=last, reason=reason, is_full=True)

    # `last` is a fraction; cutoff is a percentage. Compare in fraction space.
    tau_frac = gcfg.mape_cutoff_pct / 100.0
    panel = {g for g in all_idx if last[g] is None or last[g] > tau_frac}
    return PanelDecision(panel=panel, last_mape_by_geo=last, reason="gated",
                         is_full=(panel == full))


def compute_emv_hat(
    *,
    panel: set[int],
    all_geology_indices: Sequence[int],
    real_by_config: Mapping[str, Mapping[int, float]],
    pred_by_config: Mapping[str, Mapping[int, float]],
) -> dict[str, float]:
    """Control-variate EMV estimate per config: ``(sum_panel real + sum_fill pred) / K``.

    For each geology use the real IX value if the geology is in the panel and the value is
    finite, otherwise the surrogate prediction. Denominator is always ``K`` =
    ``len(all_geology_indices)``. ``panel == set()`` reduces to ``mean(predictions)``. A config
    is skipped (no entry returned) if any geology lacks both a usable real and a finite pred.

    This is a ranking signal only — it is NOT the incumbent. The incumbent comes from
    :mod:`orchestrator.emv`'s strict rule over fully-completed configs.
    """
    all_idx = [int(g) for g in all_geology_indices]
    k = len(all_idx)
    if k == 0:
        return {}
    out: dict[str, float] = {}
    for sid, preds in pred_by_config.items():
        reals = real_by_config.get(sid, {})
        acc = 0.0
        ok = True
        for g in all_idx:
            rv = reals.get(g)
            if g in panel and _is_finite(rv):
                acc += float(rv)
                continue
            pv = preds.get(g)
            if not _is_finite(pv):
                ok = False
                break
            acc += float(pv)
        if ok:
            out[sid] = acc / k
    return out


def rank_top_m(emv_hat: Mapping[str, float], m: int) -> list[str]:
    """Snapshot ids of the ``m`` highest ``EMV_hat`` configs (ties broken by id for determinism)."""
    if m <= 0:
        return []
    ordered = sorted(emv_hat.items(), key=lambda kv: (-kv[1], kv[0]))
    return [sid for sid, _ in ordered[:m]]


def slice_manifest(
    manifest: Mapping[str, Any] | str | Path,
    *,
    geos: Sequence[int],
    sids: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    gate_phase: str | None = None,
) -> dict[str, Any]:
    """Restrict a full ensemble manifest to a geology subset (and optionally a snapshot subset).

    The driver emits one FULL manifest (all K geologies per snapshot) and slices it for each
    phase: phase-1 keeps every snapshot but only the panel geologies
    (``geos=panel, sids=None``); phase-2 keeps the top-M snapshots and only the fill geologies
    (``geos=fill, sids=top_m``). Slicing a full manifest (rather than filtering at emit time)
    means the fill-geology entries — with their resolved geology files/metadata — are always
    available for the completion phase. The staged ``.jl`` well-config files are shared across
    geologies, so no re-acquisition is needed. Drops snapshots left with no entries. Writes
    JSON to ``out_path`` if given; stamps ``gate_phase`` when provided.
    """
    if isinstance(manifest, (str, Path)):
        with open(manifest, "r") as f:
            src: Mapping[str, Any] = json.load(f)
    else:
        src = manifest

    geo_set = {int(g) for g in geos}
    keep_sids = {str(s) for s in sids} if sids is not None else None
    snaps_out: list[dict[str, Any]] = []
    for snap in src.get("snapshots", []):
        if keep_sids is not None and str(snap.get("snapshot_id")) not in keep_sids:
            continue
        entries = [
            e for e in snap.get("well_config_paths_by_geology", [])
            if int(e.get("geology_index")) in geo_set
        ]
        if not entries:
            continue
        new_snap = dict(snap)
        new_snap["well_config_paths_by_geology"] = entries
        snaps_out.append(new_snap)

    out = dict(src)
    out["snapshots"] = snaps_out
    out["snapshot_count"] = len(snaps_out)
    if gate_phase is not None:
        out["gate_phase"] = gate_phase

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out
