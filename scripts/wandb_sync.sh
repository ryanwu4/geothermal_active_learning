#!/usr/bin/env bash
# Sync an AL run's locally-buffered wandb runs to wandb.ai.
#
# Usage:
#   scripts/wandb_sync.sh /scratch/users/rwu4/al_runs/<run_id>
#
# This is needed because train_acquire.sbatch and ingest.sbatch run with
# WANDB_MODE=offline (Sherlock's GPU partition typically has no outbound
# internet). Each phase appends to the same offline run via the wandb_run_id
# stored in state.json. After the AL chain finishes, run this from a host with
# internet access (e.g. a Sherlock login node, or your local machine after
# scp'ing the wandb dir).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <al_run_root>" >&2
    exit 1
fi

RUN_ROOT="$1"
WANDB_DIR="$RUN_ROOT/wandb"

if [[ ! -d "$WANDB_DIR" ]]; then
    echo "ERROR: no wandb dir at $WANDB_DIR" >&2
    exit 1
fi

# wandb sync targets `offline-run-*` directories.
runs=( "$WANDB_DIR"/offline-run-* )
if [[ ! -e "${runs[0]}" ]]; then
    echo "ERROR: no offline-run-* dirs under $WANDB_DIR" >&2
    exit 1
fi

echo "Syncing ${#runs[@]} offline run(s) under $WANDB_DIR"
for r in "${runs[@]}"; do
    echo "=== $r ==="
    wandb sync "$r"
done

echo "Done. Visit your wandb dashboard to see the synced run(s)."
