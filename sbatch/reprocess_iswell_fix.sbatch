#!/usr/bin/bash
#
# One-off SLURM batch: patch IsWell/InjRate in every AL output H5 from the
# corrupted run, rebuild current_compiled.h5, and run a sanity check on the
# result. Submit with:
#     sbatch reprocess_iswell_fix.sbatch
#
# Edit the variables in the "CONFIG" block below before submitting.
# This script is idempotent — re-running after success short-circuits per file
# via the reprocess_iswell_fix_version HDF5 attr, so it's safe to re-submit
# after a partial or interrupted run.
#
#SBATCH --job-name=reprocess_iswell_fix
#SBATCH --output=/scratch/users/rwu4/al_runs/_logs/reprocess_iswell_fix.%j.out
#SBATCH --error=/scratch/users/rwu4/al_runs/_logs/reprocess_iswell_fix.%j.err
#SBATCH --mem=32GB
#SBATCH -c 8
#SBATCH -t 6:00:00

set -euo pipefail

# ───────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before submitting.
# ───────────────────────────────────────────────────────────────────────────
RUN_ID="al_20260508_204044_e7c7fa"
RAW_H5_DIR="/scratch/users/rwu4/intersect_data/h5s_out"
STAGE_ROOT="/scratch/users/rwu4/al_runs/${RUN_ID}"
PATCHED_OUTPUT_DIR="/scratch/users/rwu4/al_runs/${RUN_ID}/h5_outputs_patched"
BOOTSTRAP_COMPILED_H5="/scratch/users/rwu4/al_runs/compiled_CNN_patched.h5"
NORM_CONFIG="/scratch/users/rwu4/al_runs/norm_config.json"
OUTPUT_COMPILED_H5="/scratch/users/rwu4/al_runs/${RUN_ID}/current_compiled.fixed.h5"
SURROGATE_REPO="/home/users/rwu4/Geothermal_Graph_Surrogate"
AL_REPO="/home/users/rwu4/geothermal_active_learning"
# Optional: economics config for preprocess_h5.py. Leave blank to use the
# defaults baked into the bootstrap H5 attrs.
ECONOMICS_CONFIG=""
# Workers for preprocess_h5.py (independent of the patcher, which is
# single-threaded for crash-safety).
PREPROCESS_WORKERS=8
# ───────────────────────────────────────────────────────────────────────────

mkdir -p "$(dirname "${SLURM_JOB_NAME:-}".out 2>/dev/null || true)/../_logs" || true

# Match the conda/venv setup used by ingest.sbatch.template.
ml python/3.12.1 uv hdf5/1.14.4 openblas/0.3.20
export UV_PYTHON=$(which python3)
source ~/geothermal-pomdp/bin/activate

echo "=== reprocess_iswell_fix start ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-LOCAL} run_id=${RUN_ID}"
echo "RAW_H5_DIR=${RAW_H5_DIR}"
echo "STAGE_ROOT=${STAGE_ROOT}"
echo "PATCHED_OUTPUT_DIR=${PATCHED_OUTPUT_DIR}"
echo "OUTPUT_COMPILED_H5=${OUTPUT_COMPILED_H5}"
echo

# Pre-flight: validate inputs exist before doing real work.
for p in "${RAW_H5_DIR}" "${STAGE_ROOT}" "${BOOTSTRAP_COMPILED_H5}" "${NORM_CONFIG}" "${SURROGATE_REPO}" "${AL_REPO}"; do
    if [[ ! -e "$p" ]]; then
        echo "FATAL: missing input path: $p" >&2
        exit 2
    fi
done

# Stage 1: dry-run so the log captures the mapping count and any missing-h5
# warnings before we touch anything. Fast (no writes).
echo "--- stage 1: dry-run ---"
python "${AL_REPO}/scripts/reprocess_al_h5s_iswell_fix.py" \
    --raw-h5-dir "${RAW_H5_DIR}" \
    --stage-root "${STAGE_ROOT}" \
    --patched-output-dir "${PATCHED_OUTPUT_DIR}" \
    --dry-run

# Stage 2: patch + rebuild compiled H5.
echo "--- stage 2: patch + rebuild ---"
extra_args=()
if [[ -n "${ECONOMICS_CONFIG}" ]]; then
    extra_args+=(--economics-config "${ECONOMICS_CONFIG}")
fi
python "${AL_REPO}/scripts/reprocess_al_h5s_iswell_fix.py" \
    --raw-h5-dir "${RAW_H5_DIR}" \
    --stage-root "${STAGE_ROOT}" \
    --patched-output-dir "${PATCHED_OUTPUT_DIR}" \
    --rebuild-compiled \
    --surrogate-repo "${SURROGATE_REPO}" \
    --bootstrap-compiled-h5 "${BOOTSTRAP_COMPILED_H5}" \
    --norm-config "${NORM_CONFIG}" \
    --output-compiled-h5 "${OUTPUT_COMPILED_H5}" \
    --workers "${PREPROCESS_WORKERS}" \
    ${extra_args[@]+"${extra_args[@]}"}

# Stage 3: sanity-check the rebuilt compiled H5. We expect roughly one
# distinct well-xy template per AL row (every snapshot has a unique geometry).
# If the result is still 1, the patch didn't take and the job should fail loud.
echo "--- stage 3: sanity check ---"
python - <<PY
import sys, h5py
from collections import Counter

path = "${OUTPUT_COMPILED_H5}"
with h5py.File(path, "r") as f:
    al = [k for k in f if k.startswith("v2.5_al_")]
    if not al:
        print(f"FATAL: no v2.5_al_* rows in {path}", file=sys.stderr)
        sys.exit(3)
    patterns = Counter(
        tuple(sorted([(int(r['x']), int(r['y'])) for r in f[k]['wells'][:]]))
        for k in al
    )
print(f"AL rows: {len(al)}  distinct xy templates: {len(patterns)}")
ratio = len(patterns) / len(al)
if ratio < 0.5:
    print(
        f"FATAL: distinct/total = {ratio:.3f}, suspiciously low (expected ~1.0). "
        "The patch did not take — investigate before retraining.",
        file=sys.stderr,
    )
    sys.exit(4)
print("sanity check PASSED")
PY

echo
echo "=== reprocess_iswell_fix done — output: ${OUTPUT_COMPILED_H5} ==="
