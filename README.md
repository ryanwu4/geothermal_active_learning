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
