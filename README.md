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
