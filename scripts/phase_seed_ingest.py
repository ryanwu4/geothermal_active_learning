#!/usr/bin/env python3
"""Seed-phase compile: preprocess the LHS-seed IX outputs into a standalone
compiled training H5 + a FRESH normalization config.

Counterpart to ``phase_ingest.py`` for the clean-start seed pre-step. There is
no surrogate at seed time, so this:
  * skips ALL MAPE/calibration bookkeeping, and
  * does NOT merge into a prior compiled archive (the seed IS the starting set).

Pipeline:
1. Symlink this seed batch's IX outputs into a scoped scratch dir
   (reuses ``_stage_iteration_outputs`` from ``phase_ingest.py``).
2. Run ``preprocess_h5.py`` over those outputs ONCE. Because the target
   ``--norm-config`` does not yet exist, preprocess computes fresh
   normalization stats over the seed data, saves them, and then builds the
   compiled H5 in the same pass (Geothermal_Graph_Surrogate/preprocess_h5.py:271-298).
   It also writes ``case_geology_map.json`` adjacent to the H5.

Outputs land at exactly the paths the clean-start AL config references via
``paths.bootstrap_compiled_h5`` and ``paths.norm_config_path``, so the AL driver
picks them up with no manual edits.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402

from orchestrator.ingest import _run_preprocess_h5  # type: ignore  # noqa: E402

# Reuse the IX-output scoping helper from phase_ingest so we don't duplicate the
# array_tasks.json → output_file_name symlink logic.
from phase_ingest import _stage_iteration_outputs  # type: ignore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Remote seed dir (holds scoped IX outputs + logs).")
    parser.add_argument("--array-tasks-json", type=Path, required=True)
    parser.add_argument("--ix-output-dir", type=Path, required=True,
                        help="Shared IX output dir (FILEPATHS.h5s_dir_out).")
    parser.add_argument("--surrogate-repo", type=Path, required=True)
    parser.add_argument("--seed-compiled-h5", type=Path, required=True,
                        help="Output compiled H5 (== AL config paths.bootstrap_compiled_h5).")
    parser.add_argument("--seed-norm-config", type=Path, required=True,
                        help="Output fresh norm config (== AL config paths.norm_config_path).")
    parser.add_argument("--economics-config", type=Path, default=None,
                        help="Defaults to <surrogate-repo>/configs/economics.json.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    surrogate_repo = args.surrogate_repo.resolve()
    economics_config = (
        args.economics_config.resolve() if args.economics_config
        else surrogate_repo / "configs" / "economics.json"
    )
    if not economics_config.exists():
        print(f"ERROR: economics config not found: {economics_config}", file=sys.stderr)
        return 1

    seed_compiled_h5 = args.seed_compiled_h5.resolve()
    seed_norm_config = args.seed_norm_config.resolve()
    seed_compiled_h5.parent.mkdir(parents=True, exist_ok=True)
    seed_norm_config.parent.mkdir(parents=True, exist_ok=True)

    # 1. Scope just this seed batch's IX outputs.
    scoped_dir = _stage_iteration_outputs(
        ix_output_root=args.ix_output_dir,
        array_tasks_json=args.array_tasks_json,
        iter_scratch_dir=args.run_root / "seed_ix_output",
    )
    # Fail fast if NO IX outputs landed (e.g. every seed IX run failed). preprocess
    # would otherwise exit 0 on empty input without producing anything; bail here
    # with a clear message and leave any prior good seed artifacts untouched.
    n_inputs = len(list(scoped_dir.glob("*.h5")))
    if n_inputs == 0:
        print("ERROR: no IX outputs found for the seed batch — every seed IX run "
              "may have failed. Existing seed artifacts left untouched.", file=sys.stderr)
        return 1
    print(f"[seed-ingest] scoped {n_inputs} IX outputs")

    # 2. Build into a private staging dir, then atomically promote on success.
    #    This both (a) forces a FRESH norm config — the staging norm path never
    #    pre-exists, so preprocess recomputes rather than loads it — and (b)
    #    guarantees a failed/empty rebuild never destroys a previously-good seed.
    build_dir = seed_compiled_h5.parent / ".seed_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    staged_compiled = build_dir / "seed_compiled.h5"
    staged_norm = build_dir / "seed_norm_config.json"

    log_path = args.run_root / "logs" / "seed_ingest.preprocess.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[seed-ingest] preprocess_h5 → {staged_compiled} (fresh norm)")
    _run_preprocess_h5(
        surrogate_repo=surrogate_repo,
        input_dir=scoped_dir,
        output_h5=staged_compiled,
        norm_config=staged_norm,
        economics_config=economics_config,
        workers=int(args.workers),
        log_path=log_path,
    )

    if not staged_compiled.exists() or not staged_norm.exists():
        print(f"ERROR: preprocess did not produce {staged_compiled} / {staged_norm}; "
              f"see {log_path}. Existing seed artifacts left untouched.", file=sys.stderr)
        return 1
    with h5py.File(staged_compiled, "r") as f:
        n_cases = len(list(f.keys()))
    if n_cases == 0:
        print("ERROR: seed compiled H5 has 0 cases — all IX runs may have failed. "
              "Existing seed artifacts left untouched.", file=sys.stderr)
        return 1

    # Promote staged artifacts to the configured paths (same filesystem → atomic
    # os.replace). case_geology_map.json sits next to the compiled H5.
    staged_map = build_dir / "case_geology_map.json"
    final_map = seed_compiled_h5.parent / "case_geology_map.json"
    os.replace(staged_compiled, seed_compiled_h5)
    os.replace(staged_norm, seed_norm_config)
    if staged_map.exists():
        os.replace(staged_map, final_map)
    elif final_map.exists():
        # No fresh map produced; drop any stale one so train.py resolves geology
        # at runtime rather than trusting an outdated map.
        final_map.unlink()
    shutil.rmtree(build_dir, ignore_errors=True)

    elapsed_min = (time.time() - started) / 60.0
    print(
        f"[seed-ingest] done: {n_cases} cases → {seed_compiled_h5}; "
        f"norm={seed_norm_config}; case_map={'present' if final_map.exists() else 'MISSING'} "
        f"({elapsed_min:.2f} min)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
