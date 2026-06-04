"""Contract tests for geothermal.data.resolve_geology_indices.

Locks the standard: an AL case's geology is resolved from its ``<scenario>``
(geology_config_id) token via the geologies config — correct for EVERY emit
path. In particular, ensemble / cma_surrogate / baseline emits use a small
``run_id=m`` shared across all geologies of a candidate, so resolving geology by
decoding ``run_id // 10000`` would collapse them all to geo 0. This test would
fail under that (now-fixed) bug.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LOCAL_SURROGATE_REPO = Path("/home/rwu4/omv_geothermal/Geothermal_Graph_Surrogate")


def _resolver_and_config():
    if not LOCAL_SURROGATE_REPO.exists():
        pytest.skip(f"surrogate repo missing: {LOCAL_SURROGATE_REPO}")
    if str(LOCAL_SURROGATE_REPO) not in sys.path:
        sys.path.insert(0, str(LOCAL_SURROGATE_REPO))
    data = pytest.importorskip("geothermal.data")
    _, cfg_p = data._search_geology_metadata_files()
    if cfg_p is None:
        pytest.skip("geologies config not discoverable by the resolver")
    geos = json.loads(Path(cfg_p).read_text())["geologies"]
    return data.resolve_geology_indices, geos


def _al_case_id(scenario: int, run_id: int, *, al_iter: int = 3, tail_iter: int = 45) -> str:
    """An AL IX-output stem in the canonical form the Julia stager emits:
    v2.5_{output_prefix}_{scenario}_run{run_id:04d}_iter{tail_iter:04d}.
    output_prefix carries its own _iter token (al_<id>_iter<NN>)."""
    return f"v2.5_al_demo_iter{al_iter:04d}_{scenario}_run{run_id:04d}_iter{tail_iter:04d}"


def test_per_geology_and_ensemble_resolve_by_scenario(tmp_path):
    resolve, geos = _resolver_and_config()
    sample = geos[:4]
    assert any(int(e["geology_index"]) != 0 for e in sample), \
        "need a non-zero geology_index in the sample for a meaningful regression lock"

    case_ids: list[str] = []
    expected: list[int] = []
    for e in sample:
        scen, gidx = int(e["scenario"]), int(e["geology_index"])
        # Per-geology emit: run_id encodes geology (geology_index*10000 + m).
        case_ids.append(_al_case_id(scen, gidx * 10000 + 5))
        expected.append(gidx)
        # Ensemble / cma_surrogate emit: run_id = m (small), shared across all
        # geologies of a candidate — geology lives ONLY in the scenario token.
        case_ids.append(_al_case_id(scen, 5))
        expected.append(gidx)

    # h5_path is only opened for the fingerprint fallback; every case resolves
    # via AL_RE + scenario here, so a non-existent path is never touched.
    out = resolve(case_ids, tmp_path / "unused.h5")
    assert out is not None, "resolver returned None — geologies config not found?"
    assert out.tolist() == expected


def test_ensemble_run_id_does_not_collapse_to_geo_zero(tmp_path):
    """The explicit bug lock: an ensemble case (small run_id) for a non-zero
    geology must resolve to that geology, not geo 0."""
    resolve, geos = _resolver_and_config()
    nonzero = next((e for e in geos if int(e["geology_index"]) != 0), None)
    if nonzero is None:
        pytest.skip("no non-zero geology_index in config")
    scen, gidx = int(nonzero["scenario"]), int(nonzero["geology_index"])
    # run_id=3 would decode to geo 0 under the old run//10000 rule.
    out = resolve([_al_case_id(scen, 3)], tmp_path / "unused.h5")
    assert out is not None and int(out[0]) == gidx
