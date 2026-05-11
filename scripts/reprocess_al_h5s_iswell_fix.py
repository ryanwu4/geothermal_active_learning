"""One-off: rewrite Input/IsWell + Input/InjRate in each AL per-scenario H5
that was corrupted by the cli_post_format_scenario.jl place_default_wells
fallback (the "IsWell bug", 2026-05-08 to 2026-05-11).

The simulator outputs themselves (revenues, well_tp_profiles, WEPT timeseries,
PermX/Y/Z, Porosity, etc.) are correct in each per-scenario H5 — only the
input-side `Input/IsWell` and `Input/InjRate` grids were stamped with a
constant default-template. This script rewrites those two datasets from the
staged .jl spec that the simulator actually consumed during pre-sim.

After patching, the script optionally re-runs `preprocess_h5.py` over the
patched directory and merges the result with the bootstrap compiled H5 to
produce a clean `current_compiled.h5`.

This script is intentionally isolated and not wired into the orchestrator.
Run once, then delete or archive.

Convention (verified against the bug-default empirical mapping):
- The staged .jl records each well as `(I_jl, J_jl, K_jl, "TYPE", rate)`
  with Julia 1-based indices, written by Geothermal_Graph_Surrogate's
  `to_julia_wells_text` per `(j_idx=round(x)+1, i_idx=round(y)+1,
  k_idx=round(z)+1)`.
- Python's view of `Input/IsWell` has shape (NZ, A, B) where the bug-default
  analysis proved Python axis-1 ↔ Julia j and Python axis-2 ↔ Julia i. So
  a .jl entry `(I, J, K, ...)` marks `is_well[0:K, J-1, I-1] = 1`.
- The `Input/IsActive` mask is respected: cells flagged inactive get -999
  in both arrays (mirrors what `data_collect_initialize_reservoir` does on
  the Julia side).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


# Bumped if the patching logic ever changes in a way that should force a
# re-patch of previously-patched files. The script writes this as an HDF5
# root-group attr (`reprocess_iswell_fix_version`) once a patch completes.
PATCH_VERSION = 1
PATCH_VERSION_ATTR = "reprocess_iswell_fix_version"
PATCH_JL_ATTR = "reprocess_iswell_fix_jl_basename"
TMP_SUFFIX = ".patching.tmp"


# --------------------------------------------------------------------------- #
# .jl parsing
# --------------------------------------------------------------------------- #

_WELL_LINE_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\"(INJECTOR|PRODUCER)\"\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)


@dataclass(frozen=True)
class WellSpec:
    i_jl: int   # Julia 1-based i (axis-2 in Python is i-1)
    j_jl: int   # Julia 1-based j (axis-1 in Python is j-1)
    k_jl: int   # Julia 1-based deepest perforation (Python perforations 0..K-1)
    is_injector: bool
    rate: float


def parse_staged_jl(path: Path) -> list[WellSpec]:
    """Parse a .jl staged by acquire.to_julia_wells_text. Returns wells in file order."""
    text = path.read_text()
    # Scope to the wells = [ ... ] block to avoid matching tuples in comments.
    m = re.search(r"wells\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No `wells = [...]` block in {path}")
    block = m.group(1)
    wells: list[WellSpec] = []
    for line in block.splitlines():
        match = _WELL_LINE_RE.search(line)
        if not match:
            continue
        i_jl, j_jl, k_jl, well_type, rate_str = match.groups()
        wells.append(WellSpec(
            i_jl=int(i_jl),
            j_jl=int(j_jl),
            k_jl=int(k_jl),
            is_injector=(well_type == "INJECTOR"),
            rate=float(rate_str),
        ))
    if not wells:
        raise ValueError(f"Empty wells list in {path}")
    return wells


# --------------------------------------------------------------------------- #
# Task-id ↔ staged-jl mapping
# --------------------------------------------------------------------------- #

def build_output_to_staged_jl_map(stage_roots: Iterable[Path]) -> dict[str, str]:
    """Walk array_tasks.json files under each stage_root and return a dict
    mapping output_file_name → staged_jl_path. This is the canonical source of
    truth — it's exactly what the worker consumed when running IX.
    """
    mapping: dict[str, str] = {}
    for root in stage_roots:
        for tj in sorted(Path(root).rglob("array_tasks.json")):
            try:
                with open(tj) as f:
                    payload = json.load(f)
            except Exception as e:
                print(f"[reprocess] WARN: failed to parse {tj}: {e}", file=sys.stderr)
                continue
            for task in payload.get("tasks", []):
                out_name = task.get("output_file_name")
                jl_path = task.get("staged_jl_path") or task.get("staged_jl_relpath")
                if out_name and jl_path:
                    if not Path(jl_path).is_absolute():
                        jl_path = str(Path(tj).parent / jl_path)
                    mapping[out_name] = jl_path
    return mapping


# --------------------------------------------------------------------------- #
# H5 patching
# --------------------------------------------------------------------------- #

def rebuild_iswell_injrate(
    is_active: np.ndarray,  # shape (NZ, A, B), 0/1 mask (or -999 etc.)
    wells: list[WellSpec],
    default_float: float = -999.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct fresh IsWell + InjRate arrays from a .jl wells spec.

    The (i_jl, j_jl) → Python-(axis-1, axis-2) mapping is the swap proven by
    the bug-default analysis: Python_x = j_jl - 1 (axis 1), Python_y = i_jl - 1
    (axis 2). For each well, mark every k from 0..k_jl-1 inclusive.
    """
    nz, dim_a, dim_b = is_active.shape
    is_well = np.zeros((nz, dim_a, dim_b), dtype=np.int64)
    inj_rate = np.zeros((nz, dim_a, dim_b), dtype=np.float32)

    inactive_mask = is_active != 1  # True where cell is inactive

    for w in wells:
        ax1 = w.j_jl - 1
        ax2 = w.i_jl - 1
        if not (0 <= ax1 < dim_a and 0 <= ax2 < dim_b):
            raise ValueError(
                f"Well (i_jl={w.i_jl}, j_jl={w.j_jl}) out of grid bounds (A={dim_a}, B={dim_b})"
            )
        kf = w.k_jl
        if kf < 1 or kf > nz:
            raise ValueError(
                f"Well perforation depth k_jl={kf} outside [1, {nz}] for grid"
            )
        # Mirror data_collect_initialize_reservoir's inactive-cell handling.
        column_inactive = inactive_mask[:kf, ax1, ax2]
        is_well[:kf, ax1, ax2] = np.where(column_inactive, -999, 1)
        sign = 1.0 if w.is_injector else -1.0
        rate_value = sign * abs(w.rate)
        inj_rate[:kf, ax1, ax2] = np.where(column_inactive, default_float, rate_value)
    return is_well, inj_rate


def _already_patched(h5_path: Path) -> bool:
    """Return True iff the file at h5_path carries the idempotency marker for
    the current PATCH_VERSION. Tolerates a missing file (returns False).
    """
    if not h5_path.exists():
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            v = f.attrs.get(PATCH_VERSION_ATTR)
            return v is not None and int(v) >= PATCH_VERSION
    except (OSError, KeyError):
        # Truncated/corrupt file from an interrupted run — treat as not patched.
        return False


def patch_one_h5(
    src_h5: Path,
    jl_path: Path,
    target_h5: Path,
    *,
    default_float: float = -999.0,
    force: bool = False,
) -> dict:
    """Patch Input/IsWell + Input/InjRate using a write-temp-then-rename pattern.

    If `target_h5 == src_h5` we patch in place; otherwise we write a fresh copy
    at `target_h5`. In both cases the actual mutation happens on a sibling
    `target_h5.patching.tmp` file, and we `os.replace` it onto `target_h5` only
    after every write has completed and the H5 file handle has been closed.
    POSIX rename is atomic, so the operation is all-or-nothing from the
    perspective of any subsequent run.

    Idempotency: a completed patch leaves an HDF5 root attr
    `reprocess_iswell_fix_version=PATCH_VERSION` on `target_h5`. A re-run with
    `force=False` (default) sees this and short-circuits with status="skipped_already_patched".
    """
    if not force and _already_patched(target_h5):
        return {"status": "skipped_already_patched", "h5": str(target_h5), "jl": str(jl_path), "n_wells": 0, "unchanged": False}

    wells = parse_staged_jl(jl_path)
    tmp_h5 = target_h5.with_suffix(target_h5.suffix + TMP_SUFFIX)

    # Clean up any orphan tmp from a previous interrupted run before we start.
    if tmp_h5.exists():
        tmp_h5.unlink()

    # Stage a fresh copy of the source bytes into tmp. We always copy (rather
    # than open src in r+) so that a failure mid-patch leaves the original
    # untouched, and so the in-place and copy-dir paths share the same code.
    shutil.copy2(src_h5, tmp_h5)

    try:
        with h5py.File(tmp_h5, "r+") as f:
            is_active = f["Input/IsActive"][:]
            old_iswell = f["Input/IsWell"][:]
            if is_active.shape != old_iswell.shape:
                raise ValueError(
                    f"Shape mismatch in {src_h5}: IsActive {is_active.shape} vs IsWell {old_iswell.shape}"
                )
            new_iswell, new_injrate = rebuild_iswell_injrate(is_active, wells, default_float)
            unchanged = bool(np.array_equal(new_iswell, old_iswell))

            del f["Input/IsWell"]
            del f["Input/InjRate"]
            f.create_dataset("Input/IsWell", data=new_iswell, dtype=np.int64, compression="gzip", compression_opts=4)
            f.create_dataset("Input/InjRate", data=new_injrate, dtype=np.float32, compression="gzip", compression_opts=4)

            f.attrs[PATCH_VERSION_ATTR] = PATCH_VERSION
            f.attrs[PATCH_JL_ATTR] = jl_path.name
    except BaseException:
        # On any failure (including KeyboardInterrupt), drop the tmp so we
        # don't leave a half-written file that the next run would mistake for
        # a stale-but-existent target.
        try:
            tmp_h5.unlink()
        except FileNotFoundError:
            pass
        raise

    # Atomic swap. Anything before this and target_h5 is untouched; anything
    # after and target_h5 has the complete patched file.
    os.replace(str(tmp_h5), str(target_h5))

    return {
        "status": "patched",
        "h5": str(target_h5),
        "jl": str(jl_path),
        "n_wells": len(wells),
        "unchanged": unchanged,
    }


def cleanup_orphan_tmps(*dirs: Path) -> int:
    """Remove any *.h5.patching.tmp files left over from a previous run. These
    are always safe to delete: their existence means a prior process didn't
    reach the os.replace step, so the corresponding target either was never
    written or already held the pre-tmp content.
    """
    n = 0
    for d in dirs:
        if d is None or not d.exists():
            continue
        for p in d.glob(f"*.h5{TMP_SUFFIX}"):
            try:
                p.unlink()
                n += 1
            except FileNotFoundError:
                pass
    return n


# --------------------------------------------------------------------------- #
# Re-preprocess + merge
# --------------------------------------------------------------------------- #

def run_preprocess(
    surrogate_repo: Path,
    input_dir: Path,
    output_h5: Path,
    norm_config: Path,
    economics_config: Path | None,
    workers: int = 4,
) -> None:
    cmd = [
        sys.executable,
        str(surrogate_repo / "preprocess_h5.py"),
        "--input-dir", str(input_dir),
        "--output-h5", str(output_h5),
        "--norm-config", str(norm_config),
        "--workers", str(workers),
    ]
    if economics_config is not None:
        cmd.extend(["--economics-config", str(economics_config)])
    print(f"[reprocess] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(surrogate_repo), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"preprocess_h5.py failed with code {proc.returncode}")


def merge_compiled(prior: Path, delta: Path, out: Path) -> None:
    """Mirror orchestrator.ingest._merge_compiled_h5s: copy prior groups, then
    overlay delta groups (delta wins on collision).
    """
    with h5py.File(prior, "r") as p, h5py.File(delta, "r") as d, h5py.File(out, "w") as o:
        for k, v in p.attrs.items():
            o.attrs[k] = v
        delta_keys = set(d.keys())
        for key in p.keys():
            if key in delta_keys:
                continue
            p.copy(key, o)
        for key in d.keys():
            d.copy(key, o)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-h5-dir", type=Path, required=True,
                    help="Directory containing the per-scenario IX-output H5s to patch.")
    ap.add_argument("--stage-root", type=Path, action="append", required=True,
                    help="Stage root containing iter_*/array_tasks.json files. May be passed multiple times.")
    ap.add_argument("--patched-output-dir", type=Path, default=None,
                    help="If set, write patched H5s here instead of in-place. (Symlinks the originals first; "
                         "the patched copies live in this dir.) Default: in-place.")
    ap.add_argument("--rebuild-compiled", action="store_true",
                    help="After patching, run preprocess_h5.py over the patched dir and merge with --bootstrap-compiled-h5.")
    ap.add_argument("--surrogate-repo", type=Path, default=None,
                    help="Path to Geothermal_Graph_Surrogate (needed only with --rebuild-compiled).")
    ap.add_argument("--bootstrap-compiled-h5", type=Path, default=None,
                    help="Bootstrap compiled H5 to merge AL delta into.")
    ap.add_argument("--norm-config", type=Path, default=None,
                    help="Norm config used by preprocess_h5.py.")
    ap.add_argument("--economics-config", type=Path, default=None,
                    help="Optional economics config for preprocess_h5.py.")
    ap.add_argument("--output-compiled-h5", type=Path, default=None,
                    help="Where to write the final rebuilt compiled H5.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be patched without modifying files.")
    ap.add_argument("--force", action="store_true",
                    help="Re-patch files that already carry the idempotency marker. Default is to skip them.")
    args = ap.parse_args()

    # Sweep any *.patching.tmp leftover from a prior interrupted run.
    n_cleaned = cleanup_orphan_tmps(args.raw_h5_dir, args.patched_output_dir)
    if n_cleaned:
        print(f"[reprocess] cleaned {n_cleaned} orphan .patching.tmp file(s) from a prior interrupted run")

    out_to_jl = build_output_to_staged_jl_map(args.stage_root)
    if not out_to_jl:
        print(f"[reprocess] FATAL: no array_tasks.json entries found under {args.stage_root}", file=sys.stderr)
        return 2
    print(f"[reprocess] loaded {len(out_to_jl)} output_file_name → staged_jl_path mappings")

    # Drive the loop from the mapping, not the directory. The raw-h5-dir may
    # be a shared bucket (e.g. /scratch/users/rwu4/intersect_data/h5s_out)
    # containing H5s from many runs; we should only touch files this run's
    # stage roots actually claim.
    if args.patched_output_dir:
        args.patched_output_dir.mkdir(parents=True, exist_ok=True)

    n_patched = 0
    n_skipped_already = 0
    n_missing_in_h5_dir = 0
    n_missing_jl_on_disk = 0
    n_unchanged = 0
    n_errors = 0
    missing_h5_names: list[str] = []
    for out_name in sorted(out_to_jl):
        jl_path = Path(out_to_jl[out_name])
        src_h5 = args.raw_h5_dir / out_name
        if not src_h5.exists():
            n_missing_in_h5_dir += 1
            if len(missing_h5_names) < 20:
                missing_h5_names.append(out_name)
            continue
        if not jl_path.exists():
            print(f"[reprocess] WARN staged .jl missing for {out_name}: {jl_path}", file=sys.stderr)
            n_missing_jl_on_disk += 1
            continue
        target = args.patched_output_dir / out_name if args.patched_output_dir else src_h5
        if args.dry_run:
            label = "RE-PATCH (--force)" if (args.force and _already_patched(target)) else (
                "SKIP already-patched" if _already_patched(target) else "PATCH"
            )
            print(f"[reprocess] DRY {label} {target}  ← {jl_path}")
            continue
        try:
            res = patch_one_h5(src_h5, jl_path, target, force=args.force)
        except Exception as e:
            print(f"[reprocess] ERROR patching {target}: {e}", file=sys.stderr)
            n_errors += 1
            continue
        if res["status"] == "skipped_already_patched":
            n_skipped_already += 1
        else:
            n_patched += 1
            if res["unchanged"]:
                n_unchanged += 1
        done_so_far = n_patched + n_skipped_already
        if done_so_far and done_so_far % 200 == 0:
            print(f"[reprocess] progress: patched={n_patched} skipped_already={n_skipped_already} "
                  f"unchanged={n_unchanged} missing_h5={n_missing_in_h5_dir} missing_jl={n_missing_jl_on_disk} errors={n_errors}")
    print(
        f"[reprocess] done patching: expected={len(out_to_jl)} patched={n_patched} "
        f"skipped_already_patched={n_skipped_already} "
        f"unchanged={n_unchanged} missing_h5={n_missing_in_h5_dir} "
        f"missing_jl={n_missing_jl_on_disk} errors={n_errors}"
    )
    if n_missing_in_h5_dir:
        print(f"[reprocess] first missing h5s: {missing_h5_names[:10]}")

    if args.dry_run:
        return 0

    if args.rebuild_compiled:
        if not (args.surrogate_repo and args.bootstrap_compiled_h5 and args.norm_config and args.output_compiled_h5):
            print("[reprocess] FATAL: --rebuild-compiled requires --surrogate-repo, --bootstrap-compiled-h5, --norm-config, --output-compiled-h5", file=sys.stderr)
            return 2
        if not args.patched_output_dir:
            print(
                "[reprocess] FATAL: --rebuild-compiled requires --patched-output-dir so preprocess only sees this run's AL outputs. "
                "Running preprocess over the shared raw-h5-dir would compile every unrelated H5 too.",
                file=sys.stderr,
            )
            return 2
        patched_dir = args.patched_output_dir
        delta_h5 = args.output_compiled_h5.with_suffix(".al_delta.h5")
        print(f"[reprocess] running preprocess_h5.py over patched dir → {delta_h5}")
        run_preprocess(
            surrogate_repo=args.surrogate_repo,
            input_dir=patched_dir,
            output_h5=delta_h5,
            norm_config=args.norm_config,
            economics_config=args.economics_config,
            workers=args.workers,
        )
        print(f"[reprocess] merging {args.bootstrap_compiled_h5} + {delta_h5} → {args.output_compiled_h5}")
        merge_compiled(args.bootstrap_compiled_h5, delta_h5, args.output_compiled_h5)
        print(f"[reprocess] wrote {args.output_compiled_h5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
