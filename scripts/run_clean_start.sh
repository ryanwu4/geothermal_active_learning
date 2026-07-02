#!/usr/bin/env bash
# run_clean_start.sh — clean-start AL, end to end: (1) LHS seed initialization, then
# (2) the AL run, in sequence, from ONE config.
#
# The two steps chain via the config: run_seed_lhs.py writes
#   paths.bootstrap_compiled_h5 + paths.norm_config_path
# and BLOCKS until the remote seed IX + ingest COMPLETES (returns non-zero otherwise);
# run_al_local.py then consumes exactly those paths. So running them back-to-back is
# correct — the AL step never starts before the seed compiled H5 exists.
#
# Usage:
#   bash scripts/run_clean_start.sh [CONFIG]
#   SEED_ID=clean RUN_ID=myrun bash scripts/run_clean_start.sh configs/al_cma_surrogate_clean.json
#   SKIP_SEED=1 bash scripts/run_clean_start.sh        # AL only, reuse an existing seed
#   DRY_RUN=1  bash scripts/run_clean_start.sh         # print the commands, run nothing
set -euo pipefail

REPO=/home/rwu4/omv_geothermal/geothermal_active_learning
CONFIG="${1:-configs/al_cma_surrogate_clean.json}"
SEED_ID="${SEED_ID:-}"; RUN_ID="${RUN_ID:-}"
SKIP_SEED="${SKIP_SEED:-0}"; DRY_RUN="${DRY_RUN:-0}"

cd "$REPO"
[[ -f "$CONFIG" ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
for s in scripts/run_seed_lhs.py scripts/run_al_local.py; do
  [[ -f "$s" ]] || { echo "ERROR: missing $s" >&2; exit 1; }
done

run() { echo "+ $*"; [[ "$DRY_RUN" == "1" ]] || "$@"; }

if [[ "$DRY_RUN" != "1" ]]; then
  source ~/.bashrc 2>/dev/null || true
  conda activate geothermal-pomdp 2>/dev/null || true
fi

echo "=== clean-start: config=$CONFIG  seed_id=${SEED_ID:-<from config>}  run_id=${RUN_ID:-<auto>} ==="

seed_args=(--config "$CONFIG"); [[ -n "$SEED_ID" ]] && seed_args+=(--seed-id "$SEED_ID")
al_args=(--config "$CONFIG");   [[ -n "$RUN_ID" ]]  && al_args+=(--run-id "$RUN_ID")

if [[ "$SKIP_SEED" == "1" ]]; then
  echo "[1/2] SKIP_SEED=1 — reusing existing bootstrap seed"
else
  echo "[1/2] LHS seed initialization (blocks until remote seed ingest COMPLETES)"
  run python scripts/run_seed_lhs.py "${seed_args[@]}"
fi

echo "[2/2] AL run"
run python scripts/run_al_local.py "${al_args[@]}"
echo "=== clean-start complete ==="
