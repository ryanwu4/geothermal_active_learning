"""Centralized path resolution for an AL run.

A run lives under ``<scratch_root>/<run_id>/`` and contains everything except
the original surrogate/simulator repos and the bootstrap data they hold.
Iteration-scoped artifacts are placed under ``iter_NN/`` subdirectories so
re-running an iteration cannot stomp on a prior one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_root: Path

    @property
    def state_file(self) -> Path:
        return self.run_root / "state.json"

    @property
    def done_marker(self) -> Path:
        return self.run_root / "done.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_root / "logs"

    @property
    def raw_ix_archive(self) -> Path:
        """Symlinks to all raw IX outputs ever produced by this run."""
        return self.run_root / "raw_ix_archive"

    @property
    def compiled_h5_dir(self) -> Path:
        return self.run_root / "compiled"

    @property
    def models_dir(self) -> Path:
        return self.run_root / "models"

    @property
    def manifests_dir(self) -> Path:
        return self.run_root / "manifests"

    @property
    def sbatch_dir(self) -> Path:
        return self.run_root / "sbatch_rendered"

    def iter_dir(self, iteration: int) -> Path:
        return self.run_root / f"iter_{iteration:04d}"

    def iter_acquire_dir(self, iteration: int) -> Path:
        return self.iter_dir(iteration) / "acquire"

    def iter_ix_stage_dir(self, iteration: int) -> Path:
        return self.iter_dir(iteration) / "ix_stage"

    def iter_ix_output_dir(self, iteration: int) -> Path:
        return self.iter_dir(iteration) / "ix_output"

    def iter_preprocessed_h5(self, iteration: int) -> Path:
        return self.iter_dir(iteration) / "preprocessed.h5"

    def iter_compiled_h5(self, iteration: int) -> Path:
        return self.compiled_h5_dir / f"compiled_iter_{iteration:04d}.h5"

    def iter_model_dir(self, iteration: int) -> Path:
        return self.models_dir / f"iter_{iteration:04d}"

    def iter_manifest(self, iteration: int) -> Path:
        return self.manifests_dir / f"manifest_iter_{iteration:04d}.json"

    def ensure_dirs(self) -> None:
        for d in (
            self.run_root,
            self.logs_dir,
            self.raw_ix_archive,
            self.compiled_h5_dir,
            self.models_dir,
            self.manifests_dir,
            self.sbatch_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def ensure_iter_dirs(self, iteration: int) -> None:
        for d in (
            self.iter_dir(iteration),
            self.iter_acquire_dir(iteration),
            self.iter_ix_stage_dir(iteration),
            self.iter_ix_output_dir(iteration),
            self.iter_model_dir(iteration),
        ):
            d.mkdir(parents=True, exist_ok=True)


def resolve_run_paths(scratch_root: Path | str, run_id: str) -> RunPaths:
    return RunPaths(run_root=Path(scratch_root) / run_id)
