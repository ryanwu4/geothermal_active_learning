"""Wrapper around ``Geothermal_Graph_Surrogate/train.py`` for AL iterations.

Decides warm-start vs. from-scratch based on iteration index and the
``from_scratch_every_k`` policy. Returns the new checkpoint and scaler paths.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class TrainError(RuntimeError):
    pass


@dataclass
class TrainResult:
    checkpoint_path: Path
    scaler_path: Path
    log_dir: Path
    metrics_json: Path | None
    final_val_loss: float | None
    was_from_scratch: bool


def _resolve_run_outputs(output_root: Path, run_id: int) -> tuple[Path, Path, Path]:
    log_dir = output_root / "geothermal_hetero_gnn" / f"run_{run_id:02d}"
    ckpt_dir = log_dir / "checkpoints"
    plots_dir = log_dir / "plots"
    return log_dir, ckpt_dir, plots_dir


def _find_best_checkpoint(ckpt_dir: Path) -> Path:
    best = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not best:
        raise TrainError(f"No best-*.ckpt found in {ckpt_dir}")
    # Lightning saves only top-k=1, so there should be exactly one. If multiple
    # exist (re-runs), pick the one with lowest val_loss in filename.
    def _loss(p: Path) -> float:
        # filename pattern: best-EEE-LLL.LLLL.ckpt  → loss is the last token.
        try:
            return float(p.stem.split("-")[-1])
        except ValueError:
            return float("inf")
    return min(best, key=_loss)


def should_train_from_scratch(iteration: int, from_scratch_every_k: int) -> bool:
    """Iteration 0 trains from-scratch (bootstrap retrain) and every K-th iter
    after that resets to avoid drift accumulation.
    """
    if from_scratch_every_k <= 0:
        return False
    return iteration % from_scratch_every_k == 0


def run_train(
    *,
    surrogate_repo: Path,
    h5_path: Path,
    output_root: Path,
    target: str,
    seed: int = 42,
    run_id: int = 0,
    cache_to_gpu: bool = True,
    gpu: str = "0",
    edge_encoder: str = "cnn",
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    max_epochs: int = 180,
    early_stop_patience: int = 60,
    warm_start_checkpoint: Path | None = None,
    warm_start_scaler: Path | None = None,
    max_epochs_finetune: int | None = None,
    extra_args: list[str] | None = None,
    log_path: Path | None = None,
) -> TrainResult:
    """Invoke ``train.py`` as a subprocess and parse out the resulting checkpoint."""
    train_py = Path(surrogate_repo) / "train.py"
    if not train_py.exists():
        raise TrainError(f"Surrogate train script not found: {train_py}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(train_py),
        "--h5-path", str(h5_path),
        "--target", target,
        "--seed", str(seed),
        "--split-seed", str(seed),
        "--run-id", str(run_id),
        "--output-root", str(output_root),
        "--gpu", gpu,
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--max-epochs", str(max_epochs),
        "--early-stop-patience", str(early_stop_patience),
        "--edge-encoder", edge_encoder,
    ]
    if cache_to_gpu:
        cmd.append("--cache-to-gpu")
    # NOTE: train.py defaults to --stratified-split, so we don't pass it here.
    # The geology index is derived automatically from case_id patterns + the
    # filenum/scenario CSV. Pass --no-stratified-split via extra_args to revert.

    is_warm = warm_start_checkpoint is not None
    if is_warm:
        cmd.extend(["--checkpoint-path", str(warm_start_checkpoint)])
        if warm_start_scaler is not None:
            cmd.extend(["--scaler-path", str(warm_start_scaler)])
        if max_epochs_finetune is not None:
            cmd.extend(["--max-epochs-finetune", str(max_epochs_finetune)])

    if extra_args:
        cmd.extend(extra_args)

    log_handle = open(log_path, "w") if log_path else None
    try:
        proc = subprocess.run(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT if log_handle else None,
            check=False,
            cwd=str(surrogate_repo),
        )
    finally:
        if log_handle:
            log_handle.close()

    if proc.returncode != 0:
        raise TrainError(
            f"train.py exited with code {proc.returncode}. See {log_path} for details."
        )

    log_dir, ckpt_dir, plots_dir = _resolve_run_outputs(output_root, run_id)
    best_ckpt = _find_best_checkpoint(ckpt_dir)
    scaler_path = best_ckpt.parent / "scaler.pkl"
    if not scaler_path.exists():
        raise TrainError(f"Expected scaler at {scaler_path} after training")

    metrics_json = plots_dir / "metrics_summary.json"
    final_val_loss: float | None = None
    if metrics_json.exists():
        try:
            import json
            with open(metrics_json, "r") as f:
                payload = json.load(f)
            val = payload.get("splits", {}).get("val", {}).get("metrics", {})
            final_val_loss = val.get("rmse") or val.get("mae")
        except Exception:
            final_val_loss = None

    return TrainResult(
        checkpoint_path=best_ckpt,
        scaler_path=scaler_path,
        log_dir=log_dir,
        metrics_json=metrics_json if metrics_json.exists() else None,
        final_val_loss=final_val_loss,
        was_from_scratch=not is_warm,
    )
