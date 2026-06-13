"""Stopping criteria for the AL loop."""
from __future__ import annotations

from dataclasses import dataclass

from .state import RunState


@dataclass
class StoppingConfig:
    max_iterations: int
    plateau_window: int = 5
    plateau_threshold_relative: float = 0.005
    target_mape: float | None = None
    # Dedicated window for the target-MAPE check. Decoupled from `plateau_window`
    # so revenue-plateau and MAPE-target stops can use different sustain lengths.
    target_mape_window: int = 3
    # Bail out if IX completion rate has been zero for this many consecutive
    # iterations. Protects against silently burning compute when Intersect
    # systematically fails (bad submission, missing files, etc.).
    consecutive_zero_completion_limit: int = 2


@dataclass
class StopDecision:
    should_stop: bool
    reason: str | None = None


def _revenue_plateau(state: RunState, window: int, threshold_rel: float) -> bool:
    """True if best_real_revenue improvement over the last ``window`` iterations
    is below ``threshold_rel`` relative to the current best.

    We compare the most recent best to the best of all earlier iterations. If
    the recent runs regressed (improvement is negative), we treat that as zero
    improvement rather than a "negative" plateau — because best_real_revenue
    is recorded as best-so-far it shouldn't actually decrease, but if it does
    via a state edit we don't want a spurious stop.
    """
    revs = [r.best_real_revenue for r in state.history if r.best_real_revenue is not None]
    if len(revs) < window + 1:
        return False
    best_now = max(revs)  # robust to non-monotone recordings
    best_prior = max(revs[:-window])
    if best_now is None or best_prior is None:
        return False
    # Sign-safe relative improvement: normalize by the larger magnitude of the two
    # bests rather than abs(best_now). This keeps the ratio well-behaved even for a
    # sign-unbounded objective (e.g. NPV crossing zero), where dividing by abs(best_now)
    # alone would explode near break-even or invert when best_now straddles 0.
    denom = max(abs(best_now), abs(best_prior))
    if denom == 0:
        return False
    improvement = max(0.0, (best_now - best_prior) / denom)
    return improvement < threshold_rel


def _mape_plateau(state: RunState, window: int, target: float) -> bool:
    """True if the last ``window`` iterations all have floored MAPE <= target.

    Uses ``batch_mape_floored`` (denominator floored at 10% of cohort-median
    real revenue) so that small-revenue cohorts like geo 8 don't keep the
    raw MAPE artificially above the target indefinitely. Falls back to
    ``batch_mape`` for older state records that pre-date the floored metric.
    """
    mapes: list[float] = []
    for r in state.history:
        val = getattr(r, "batch_mape_floored", None)
        if val is None:
            val = r.batch_mape
        if val is not None:
            mapes.append(val)
    if len(mapes) < window:
        return False
    return all(m <= target for m in mapes[-window:])


def _consecutive_zero_completions(state: RunState, limit: int) -> int:
    """Count trailing iterations with zero completed IX runs.

    Returns the run-length of the consecutive 0-completion suffix, capped at
    `limit + 1` so the caller can compare cheaply.
    """
    if limit <= 0:
        return 0
    count = 0
    for rec in reversed(state.history):
        if rec.completed == 0 and rec.submitted > 0:
            count += 1
            if count > limit:
                return count
        else:
            break
    return count


def evaluate_stopping(state: RunState, cfg: StoppingConfig, objective: str = "revenue") -> StopDecision:
    # `state.iteration` is 0-indexed and is the *just-completed* iteration when
    # this check runs (in phase_ingest, before the increment that schedules the
    # next iter). `max_iterations` is the total count the user asked for, so
    # we stop once we've completed that many — meaning iteration index
    # `max_iterations - 1`.
    if state.iteration + 1 >= cfg.max_iterations:
        return StopDecision(
            should_stop=True,
            reason=f"max_iterations ({cfg.max_iterations}) reached "
                   f"(just completed iter {state.iteration})",
        )
    zero_streak = _consecutive_zero_completions(state, cfg.consecutive_zero_completion_limit)
    if zero_streak >= cfg.consecutive_zero_completion_limit:
        return StopDecision(
            should_stop=True,
            reason=(
                f"IX completion rate has been zero for {zero_streak} consecutive "
                f"iterations (limit {cfg.consecutive_zero_completion_limit}); "
                "investigate before continuing"
            ),
        )
    # The revenue plateau keys on best_real_revenue, which the remote ingest computes from IX.
    # In npv mode the optimized objective is NPV (computed only locally on the GPU host, not on
    # the Sherlock ingest where this check runs), so a revenue plateau is the WRONG stop signal —
    # NPV can still be improving while revenue plateaus (cheaper/shorter wells) or vice versa.
    # Skip it in npv mode and rely on max_iterations / the MAPE-target stop.
    if objective != "npv" and _revenue_plateau(state, cfg.plateau_window, cfg.plateau_threshold_relative):
        return StopDecision(
            should_stop=True,
            reason=f"revenue plateau over last {cfg.plateau_window} iters "
                   f"(< {cfg.plateau_threshold_relative} relative improvement)",
        )
    if cfg.target_mape is not None and _mape_plateau(state, cfg.target_mape_window, cfg.target_mape):
        return StopDecision(
            should_stop=True,
            reason=f"MAPE below target ({cfg.target_mape}) for {cfg.target_mape_window} iters",
        )
    return StopDecision(should_stop=False, reason=None)
