# Geothermal Active Learning Orchestrator

Closes the loop between the GNN surrogate ([Geothermal_Graph_Surrogate](../Geothermal_Graph_Surrogate)) and the
INTERSECT reservoir simulator ([GeologicalSimulationWrapper.jl](../GeologicalSimulationWrapper.jl)) on the
Sherlock HPC cluster. Each iteration:

1. **Train / finetune** the GNN on all data accumulated so far.
2. **Acquire** new well-placement candidates by multi-start gradient descent on
   the surrogate. K_safe steps yield "frontier" candidates (top surrogate revenue
   in a trust region); a small subset is run on past the surrogate-vs-Intersect peak
   to K_adv to produce "adversarial" probes for free.
3. **Submit** candidates to INTERSECT via `cli_surrogate_array_prepare.jl`.
4. **Ingest** the simulation results, append them to the training set, and chain
   the next iteration.

The orchestrator is implemented as a chain of dependent SLURM jobs — there is no
long-running daemon. Each phase requests its own resources (GPU for training and
acquisition, CPU for ingest) and the 3-day SLURM cap is irrelevant.

## Behaviour notes (May 2026)

- **Stopping criteria**: in addition to `max_iterations` and the revenue-plateau
  check, an MAPE-target stop (`target_mape` / `target_mape_window`) fires when
  the *floored* MAPE has been below `target_mape` for `target_mape_window`
  consecutive iterations. The floor is `max(|real|, 0.1 × cohort_median_real)`
  so geo-8-style small-denominator candidates don't keep MAPE artificially high.
  See `configs/al_hybrid.json` for defaults.
- **Frontier slot allocation**: each iteration distributes frontier selection
  slots *equally* across geologies (was proportional to candidate count, which
  silently penalised geologies whose Adam runs went non-finite).
- **Stratified train/val/test split** is enabled by default during the retrain
  step (`--stratified-split` is appended to the surrogate-side `train.py`
  invocation by `orchestrator/retrain.py`). Train.py derives the per-case
  geology index automatically from the case_id pattern (AL cases via
  `run_num // 10000`) and from the `filenum_to_scenario_mapping.csv` +
  `geologies_full*.json` files for bootstrap cases — no auxiliary JSON map
  needs to be maintained.
- **Worker death-on-parent**: multi-GPU acquisition workers register
  `PR_SET_PDEATHSIG` so they receive `SIGTERM` automatically when the
  orchestrator process exits (Ctrl-C, SLURM timeout, OOM kill). They also
  install their own SIGTERM/SIGINT handlers for graceful shutdown. The parent
  installs a top-level handler that calls `terminate()`/`kill()` on every
  worker before raising `KeyboardInterrupt`. Combined, killing the orchestrator
  cleanly releases the GPUs — no orphan acquisition processes.
- **Surrogate hparam round-trip**: `orchestrator/acquire.py` reads
  `node_encoder` and `enrich_global_attr` from the loaded checkpoint's hparams
  (falling back to the legacy `profile` / `False` for older checkpoints) so the
  graphs built for acquisition always match the scaler the model was trained
  with. No more `"X has N features, but StandardScaler is expecting M"` errors
  when the surrogate side flips defaults.

## Layout

```
orchestrator/   Library code (state, acquisition, staging, ingest, retrain, logging)
sbatch/         SLURM job templates rendered by orchestrator/slurm.py
scripts/        Entry points and phase bodies invoked by sbatch
configs/        Default pipeline + geology selection config
tests/          Pytest suites for state and selection
```

## Usage

```bash
# Bootstrap a new AL run from existing checkpoint + compiled H5
python scripts/start_al_run.py --config configs/al_default.json

# Resume from the last successfully-completed phase
python scripts/resume_al_run.py /scratch/users/rwu4/al_runs/<run_id>

# Inspect state at any time
cat /scratch/users/rwu4/al_runs/<run_id>/state.json | jq .
```

See `configs/al_default.json` for all configurable parameters.

## Hybrid mode (local GPU, remote IX + ingest)

Waiting for a Sherlock GPU node dominates iteration latency in test runs. The
hybrid mode in `configs/al_hybrid.json` runs **train + acquire + select on the
local GPU host (bend)** while keeping IX simulations and ingest on Sherlock.

### One-time setup

Add a `Host sherlock` entry to `~/.ssh/config` so that *any* `ssh sherlock`,
`scp … sherlock:…`, or `rsync … sherlock:…` invocation transparently
multiplexes onto a single authenticated master socket:

```
Host sherlock
    HostName login.sherlock.stanford.edu
    User rwu4
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 24h
```

Then in a dedicated tmux pane on bend, kick the master once and authenticate
(Duo). With `ControlPersist 24h` the socket stays alive for 24h after you
exit the shell:

```bash
ssh sherlock          # accept Duo, run anything, exit
ssh -O check sherlock # → "Master running"
```

Edit `configs/al_hybrid.json` so the `compute.*` paths match the bend host
(`local_workspace`, `local_surrogate_repo`, `local_geologies_config`). The
local geologies config must reference local copies of the v2.5 H5 files. The
`ssh_control_path` in the config is only used for a friendly pre-flight check
that the socket exists — it should match the path your `~/.ssh/config`
`ControlPath` resolves to (e.g. `~/.ssh/cm-rwu4@login.sherlock.stanford.edu:22`).

### Run

```bash
# Fresh run (bootstraps a remote run dir on Sherlock first):
python scripts/run_al_local.py --config configs/al_hybrid.json

# Resume an existing run:
python scripts/run_al_local.py --config configs/al_hybrid.json --run-id <id>
```

The driver runs as a long-lived process — keep it in a tmux pane. Each
iteration it pulls only what it needs (compiled H5 for training, ~11–15 GB)
and overwrites a single local copy at `local_workspace/current_compiled.h5`.
Checkpoints live locally only.

### When ssh fails

If the ControlMaster socket dies (Duo expires, etc.) the driver writes
`local_workspace/NEEDS_AUTH.md` with the exact command to re-establish the
socket and the resume command, then exits cleanly with code 2. State on
Sherlock is untouched; restart the driver with `--run-id <id>`.
