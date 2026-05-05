"""Thin wandb wrapper that resumes the same run across SLURM jobs.

Each AL run has one wandb run id, persisted in :class:`RunState`. Phase scripts
call ``init_run`` to get a wandb handle that resumes the existing run. The
helper degrades gracefully when wandb is unavailable or when the user disables
it via ``WANDB_MODE=disabled``.
"""
from __future__ import annotations

from typing import Any

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional dep at runtime
    wandb = None


class WandbHandle:
    def __init__(self, run: Any) -> None:
        self._run = run

    @property
    def is_active(self) -> bool:
        return self._run is not None

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.is_active:
            return
        try:
            if step is not None:
                self._run.log(payload, step=step)
            else:
                self._run.log(payload)
        except Exception:
            # Don't let wandb failures kill the AL loop.
            pass

    def finish(self) -> None:
        if not self.is_active:
            return
        try:
            self._run.finish()
        except Exception:
            pass


def init_run(
    *,
    run_id: str,
    project: str,
    entity: str | None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
    resume: str = "allow",
) -> WandbHandle:
    """Initialize or resume a wandb run with the given id.

    ``run_id`` here is the wandb run id stored in :class:`RunState` — *not* the
    AL run_id. They're separate so wandb can shuffle internally.
    """
    if wandb is None:
        return WandbHandle(None)
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            resume=resume,
            config=config or {},
            tags=tags,
            name=name,
        )
        return WandbHandle(run)
    except Exception:
        return WandbHandle(None)
