"""Stress-test the six ensemble-mode plots in scripts/plot_convergence.py.

Builds tiny synthetic (history, rows) payloads and exercises degenerate cases
without modifying source. Reports crashes and visual issues to stdout.

Run with:
    conda activate geothermal-pomdp
    python tests/test_ensemble_plots.py
"""
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path("/home/rwu4/omv_geothermal/geothermal_active_learning")
SCRIPT = REPO_ROOT / "scripts" / "plot_convergence.py"

sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location("plot_convergence", SCRIPT)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

# --- the six plots under test ---
PLOTS = [
    ("plot_emv_distribution",                     "emv_distribution.png",                 "history+rows"),
    ("plot_best_emv_so_far",                      "best_emv_so_far.png",                  "history"),
    ("plot_exploit_seed_vs_terminal_predicted",   "exploit_seed_vs_terminal_predicted.png", "rows"),
    ("plot_exploit_seed_vs_terminal_real",        "exploit_seed_vs_terminal_real.png",    "rows"),
    ("plot_exploit_emv_progression",              "exploit_emv_progression.png",          "history"),
    ("plot_exploit_per_geology_progression",      "exploit_per_geology_progression.png",  "history"),
]


def call_plot(name: str, history: list[dict], rows: list[dict], out_dir: Path):
    """Dispatch to plot fn with the right signature; return (ok, err, stderr_text)."""
    fn = getattr(pc, name)
    buf_err = io.StringIO()
    buf_out = io.StringIO()
    try:
        with redirect_stderr(buf_err), redirect_stdout(buf_out):
            for entry in PLOTS:
                if entry[0] == name:
                    sig = entry[2]
                    break
            if sig == "history+rows":
                fn(history, rows, out_dir)
            elif sig == "history":
                fn(history, out_dir)
            elif sig == "rows":
                fn(rows, out_dir)
        return True, None, buf_err.getvalue()
    except Exception:
        return False, traceback.format_exc(), buf_err.getvalue()


def run_all(case_name: str, history: list[dict], rows: list[dict], expect_placeholder: dict | None = None):
    """Run all six plots; print outcomes; return dict of results."""
    tmp = Path(tempfile.mkdtemp(prefix=f"ensplots_{case_name}_"))
    print(f"\n=== CASE {case_name} -> {tmp}")
    results = {}
    for fn_name, fname, _sig in PLOTS:
        ok, err, stderr_text = call_plot(fn_name, history, rows, tmp)
        png = tmp / fname
        size = png.stat().st_size if png.exists() else 0
        results[fn_name] = {
            "ok": ok,
            "size": size,
            "err": err,
            "stderr": stderr_text,
        }
        status = "PASS" if (ok and size > 1024) else ("WARN" if ok else "FAIL")
        sz_kb = size / 1024
        print(f"  [{status}] {fn_name:48s} {fname:48s} size={sz_kb:7.2f} KB")
        if err:
            for line in err.strip().split("\n")[-3:]:
                print(f"      ! {line}")
        if stderr_text.strip():
            for line in stderr_text.strip().split("\n")[:3]:
                print(f"      stderr: {line}")
    return results, tmp


# ----------------------------------------------------------------------
# Synthetic payload builders
# ----------------------------------------------------------------------

def _exploit_row(it, sid, geo, real, seed_pred=100.0, term_pred=120.0,
                 seed_real=None, seed_src=None):
    return {
        "iteration": it,
        "snapshot_id": sid,
        "geology_index": geo,
        "kind": "exploit",
        "real_revenue": real,
        "predicted_revenue": term_pred,
        "seed_predicted_emv": seed_pred,
        "terminal_predicted_emv": term_pred,
        "seed_real_emv": seed_real,
        "seed_source_snapshot_id": seed_src,
    }


def _frontier_row(it, sid, geo, real, pred=110.0):
    return {
        "iteration": it,
        "snapshot_id": sid,
        "geology_index": geo,
        "kind": "frontier",
        "real_revenue": real,
        "predicted_revenue": pred,
    }


def case_A_no_ensemble():
    """Every record's best_emv_so_far is None."""
    history = [
        {"iteration": 0, "best_emv_so_far": None, "per_geology": {}},
        {"iteration": 1, "best_emv_so_far": None, "per_geology": {}},
    ]
    rows = []  # no rows at all
    return history, rows


def case_B_single_iter():
    history = [{
        "iteration": 0,
        "best_emv_so_far": 150.0,
        "best_emv_in_batch": 150.0,
        "exploit_best_emv": 140.0,
        "exploit_best_per_geology": {"0": 145.0, "1": 135.0},
        "per_geology": {
            "0": {"max_real_revenue": 145.0},
            "1": {"max_real_revenue": 135.0},
        },
    }]
    # One exploit snapshot across 2 geos; one frontier snapshot across 2 geos.
    rows = [
        _exploit_row(0, "exp_s0", 0, 145.0, seed_real=130.0, seed_src="prev"),
        _exploit_row(0, "exp_s0", 1, 135.0, seed_real=130.0, seed_src="prev"),
        _frontier_row(0, "fr_s0", 0, 150.0),
        _frontier_row(0, "fr_s0", 1, 140.0),
    ]
    return history, rows


def case_C_all_cold_start():
    """All exploit seeds have seed_source_snapshot_id=None (and seed_real_emv=None)."""
    history = [{
        "iteration": 0,
        "best_emv_so_far": 100.0,
        "exploit_best_emv": 100.0,
        "exploit_best_per_geology": {"0": 100.0},
        "per_geology": {"0": {"max_real_revenue": 100.0}},
    }]
    rows = [
        _exploit_row(0, "exp_a", 0, 100.0, seed_real=None, seed_src=None),
        _exploit_row(0, "exp_a", 1, 90.0,  seed_real=None, seed_src=None),
        _exploit_row(0, "exp_b", 0, 105.0, seed_real=None, seed_src=None),
        _exploit_row(0, "exp_b", 1, 95.0,  seed_real=None, seed_src=None),
    ]
    return history, rows


def case_D_iter_zero_frontier():
    """iter=0, kind=frontier -> position = -0.30 -> int(pos*100) = -30."""
    history = [{
        "iteration": 0,
        "best_emv_so_far": 110.0,
        "per_geology": {"0": {"max_real_revenue": 110.0}},
    }]
    rows = [
        _frontier_row(0, "fr_s0", 0, 110.0),
        _frontier_row(0, "fr_s0", 1, 108.0),
    ]
    return history, rows


def case_E_exploit_only():
    history = [{
        "iteration": 0,
        "best_emv_so_far": 120.0,
        "exploit_best_emv": 120.0,
        "exploit_best_per_geology": {"0": 120.0},
        "per_geology": {"0": {"max_real_revenue": 120.0}},
    }]
    rows = []
    for s in range(5):  # multiple snapshots so EMV groups appear
        rows.append(_exploit_row(0, f"exp_{s}", 0, 100.0 + s, seed_real=90.0 + s, seed_src=f"src_{s}"))
        rows.append(_exploit_row(0, f"exp_{s}", 1,  95.0 + s, seed_real=90.0 + s, seed_src=f"src_{s}"))
    return history, rows


def case_F_frontier_only():
    history = [{
        "iteration": 0,
        "best_emv_so_far": 130.0,
        "per_geology": {"0": {"max_real_revenue": 130.0}},
    }]
    rows = []
    for s in range(5):
        rows.append(_frontier_row(0, f"fr_{s}", 0, 120.0 + s))
        rows.append(_frontier_row(0, f"fr_{s}", 1, 115.0 + s))
    return history, rows


def case_G_mismatched_geos():
    history = [
        {
            "iteration": 0,
            "best_emv_so_far": 100.0,
            "exploit_best_emv": 100.0,
            "exploit_best_per_geology": {"0": 100.0, "1": 95.0, "2": 90.0},
            "per_geology": {
                "0": {"max_real_revenue": 100.0},
                "1": {"max_real_revenue": 95.0},
                "2": {"max_real_revenue": 90.0},
            },
        },
        {
            "iteration": 1,
            "best_emv_so_far": 110.0,
            "exploit_best_emv": 108.0,
            "exploit_best_per_geology": {"0": 108.0, "3": 100.0, "4": 99.0},
            "per_geology": {
                "0": {"max_real_revenue": 105.0},
                "3": {"max_real_revenue": 100.0},
                "4": {"max_real_revenue": 99.0},
            },
        },
    ]
    rows = [
        _exploit_row(0, "x0", 0, 100.0, seed_src="s"),
        _exploit_row(0, "x0", 1, 95.0,  seed_src="s"),
        _exploit_row(0, "x0", 2, 90.0,  seed_src="s"),
        _exploit_row(1, "x1", 0, 108.0, seed_real=100.0, seed_src="s0"),
        _exploit_row(1, "x1", 3, 100.0, seed_real=100.0, seed_src="s0"),
        _exploit_row(1, "x1", 4,  99.0, seed_real=100.0, seed_src="s0"),
    ]
    return history, rows


def case_H_nan_revenue():
    history = [{
        "iteration": 0,
        "best_emv_so_far": 100.0,
        "exploit_best_emv": 100.0,
        "exploit_best_per_geology": {"0": 100.0},
        "per_geology": {"0": {"max_real_revenue": 100.0}},
    }]
    rows = [
        _exploit_row(0, "exp_a", 0, float("nan"), seed_real=90.0, seed_src="s"),
        _exploit_row(0, "exp_a", 1, 100.0,        seed_real=90.0, seed_src="s"),
        _exploit_row(0, "exp_b", 0, 105.0,        seed_real=95.0, seed_src="s"),
        _exploit_row(0, "exp_b", 1, float("inf"), seed_real=95.0, seed_src="s"),
        _frontier_row(0, "fr_a", 0, float("nan")),
        _frontier_row(0, "fr_a", 1, 110.0),
    ]
    return history, rows


def case_I_tiny_cohort():
    """One exploit + one frontier per iteration -> violins fall back to scatter."""
    history = []
    rows = []
    for it in range(3):
        history.append({
            "iteration": it,
            "best_emv_so_far": 100.0 + it,
            "exploit_best_emv": 100.0 + it,
            "exploit_best_per_geology": {"0": 100.0 + it},
            "per_geology": {"0": {"max_real_revenue": 100.0 + it}},
        })
        rows.append(_exploit_row(it, f"exp_{it}", 0, 100.0 + it, seed_real=95.0 + it, seed_src="prev"))
        rows.append(_exploit_row(it, f"exp_{it}", 1,  95.0 + it, seed_real=95.0 + it, seed_src="prev"))
        rows.append(_frontier_row(it, f"fr_{it}", 0, 110.0 + it))
        rows.append(_frontier_row(it, f"fr_{it}", 1, 105.0 + it))
    return history, rows


def case_J_single_geo():
    """K=1: each snapshot has just one row -> any_multi check can't trigger."""
    history = [{
        "iteration": 0,
        "best_emv_so_far": 100.0,
        "exploit_best_emv": 100.0,
        "exploit_best_per_geology": {"0": 100.0},
        "per_geology": {"0": {"max_real_revenue": 100.0}},
    }]
    rows = []
    for s in range(4):
        rows.append(_exploit_row(0, f"exp_{s}", 0, 100.0 + s, seed_src="x"))
        rows.append(_frontier_row(0, f"fr_{s}", 0, 110.0 + s))
    return history, rows


def case_K_real_workspace():
    """Load real local_workspace_5_18 (per-geology run) via the script's helpers."""
    run_root = Path("/home/rwu4/omv_geothermal/geothermal_active_learning/local_workspace_5_18")
    if not run_root.exists():
        print(f"  SKIP: {run_root} does not exist")
        return None, None
    state = pc._load_state(run_root)
    history = state.get("history", [])
    rows = pc._all_candidate_rows(run_root, history)
    print(f"  loaded {len(history)} history records, {len(rows)} candidate rows")
    return history, rows


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

CASES = [
    ("A_no_ensemble",         case_A_no_ensemble),
    ("B_single_iter",         case_B_single_iter),
    ("C_all_cold_start",      case_C_all_cold_start),
    ("D_iter_zero_frontier",  case_D_iter_zero_frontier),
    ("E_exploit_only",        case_E_exploit_only),
    ("F_frontier_only",       case_F_frontier_only),
    ("G_mismatched_geos",     case_G_mismatched_geos),
    ("H_nan_revenue",         case_H_nan_revenue),
    ("I_tiny_cohort",         case_I_tiny_cohort),
    ("J_single_geo",          case_J_single_geo),
    ("K_real_workspace",      case_K_real_workspace),
]


def main():
    pc._set_style()
    summary = {}
    for case_name, builder in CASES:
        try:
            payload = builder()
        except Exception:
            print(f"\n=== CASE {case_name}: builder FAILED")
            traceback.print_exc()
            summary[case_name] = None
            continue
        if payload == (None, None):
            summary[case_name] = "skip"
            continue
        history, rows = payload
        results, _ = run_all(case_name, history, rows)
        summary[case_name] = results

    # ------------ summary table ------------
    print("\n" + "=" * 110)
    print(f"{'CASE':28s}  " + "  ".join(f"{n[5:30]:>26s}" for n, _, _ in PLOTS))
    print("-" * 110)
    for case_name, _ in CASES:
        res = summary.get(case_name)
        if res is None:
            print(f"{case_name:28s}  BUILDER-FAILED")
            continue
        if res == "skip":
            print(f"{case_name:28s}  SKIP")
            continue
        cells = []
        for fn_name, _, _ in PLOTS:
            r = res[fn_name]
            if not r["ok"]:
                cells.append("CRASH")
            elif r["size"] <= 1024:
                cells.append(f"SMALL:{r['size']}")
            else:
                cells.append(f"ok:{r['size']//1024}k")
        print(f"{case_name:28s}  " + "  ".join(f"{c:>26s}" for c in cells))
    print("=" * 110)

    # Crash summary.
    any_crash = False
    for case_name, res in summary.items():
        if res in (None, "skip"):
            continue
        for fn_name, r in res.items():
            if not r["ok"]:
                any_crash = True
                print(f"\nCRASH in {case_name}::{fn_name}:")
                print(r["err"])
    if not any_crash:
        print("\nNo crashes observed across all cases.")


if __name__ == "__main__":
    main()
