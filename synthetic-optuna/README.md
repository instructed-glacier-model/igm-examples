# Synthetic glacier — Optuna sweeper example

This example demonstrates the **IGM Optuna Hydra sweeper** (`igm_optuna`) on a synthetic glacier. It runs a multi-objective optimization with 3 controls and 2 cost functions selected from a score dictionary.

## Files overview

```
synthetic-optuna/
├── experiment/
│   └── params.yaml              # IGM experiment config (synthetic glacier)
├── optuna_params.yaml            # Optuna optimization config
├── user/
│   ├── code/
│   │   ├── inputs/
│   │   │   └── synthetic.py     # Synthetic bedrock generator
│   │   └── processes/
│   │       └── eval_objective.py # Computes state.score as a dict
│   └── conf/
│       ├── inputs/
│       │   └── synthetic.yaml
│       └── processes/
│           └── eval_objective.yaml  # Default target values
└── README.md
```

## How `state.score` works

The user module `eval_objective.py` computes **all available metrics** as a dictionary:

```python
state.score = {
    "cost_volume":    abs(volume - target_volume),
    "cost_speed":     abs(max_speed - target_max_speed),
    "cost_thickness": abs(max_thk - target_max_thk),
}
```

The `optuna_params.yaml` then **selects** which scores to use as objectives:

```yaml
objectives:
  - name: cost_volume      # picks this key from state.score
    direction: minimize
  - name: cost_speed       # picks this key from state.score
    direction: minimize
```

This separation means you can add new metrics to `eval_objective.py` and switch which ones you optimize without changing any code — just edit the yaml.

## Optuna config structure

Everything lives in `optuna_params.yaml` (loaded by default):

```yaml
# --- Which scores to optimize (selected from state.score dict) ---
objectives:
  - name: cost_volume
    direction: minimize
  - name: cost_speed
    direction: minimize

# --- Fixed overrides (target values, passed to every trial) ---
overrides:
  processes.eval_objective.target_volume: 18.0
  processes.eval_objective.target_max_speed: 200.0

# --- Control parameters (tuned by Optuna) ---
parameters:
  - name: processes.iceflow.physics.init_slidingco
    type: float
    low: 0.01
    high: 1.0
    log: true

  - name: processes.iceflow.physics.init_arrhenius
    type: float
    low: 30.0
    high: 150.0

  - name: processes.smb_simple.array.1.3   # ELA
    type: float
    low: 2500.0
    high: 3200.0

# --- Sampler, trials, parallelism ---
n_trials: 20
n_jobs: 1
sampler:
  method: TPESampler
```

The `overrides` section passes fixed Hydra overrides to every trial (target values, etc.), so both controls and targets are in the same file.

## Running the optimization

From inside this directory:

```bash
cd synthetic-optuna

# Uses optuna_params.yaml by default
igm_run -m +experiment=params hydra/sweeper=igm_optuna

# Or specify a custom config
igm_run -m +experiment=params hydra/sweeper=igm_optuna \
  hydra.sweeper.optuna_config=my_custom.yaml
```

The `-m` flag enables Hydra **multirun** mode (required for sweepers).

## Parallel trials

Each trial runs as a separate subprocess with its own TensorFlow session. By default trials run sequentially (`n_jobs: 1`). To run trials in parallel, increase `n_jobs`:

```yaml
n_jobs: 4   # launch 4 trials simultaneously
```

### GPU considerations

By default, TensorFlow grabs all available GPU memory, so multiple trials on the same GPU will crash. Two options:

**Option 1 — Allow GPU memory growth** (multiple trials share one GPU):

```yaml
n_jobs: 4
gpu_allow_growth: true    # sets TF_FORCE_GPU_ALLOW_GROWTH=true per trial
```

Each trial allocates only the GPU memory it actually needs. This works well when each trial's model fits comfortably in a fraction of the GPU.

**Option 2 — Distribute across GPUs** (one trial per GPU):

```yaml
n_jobs: 4
gpu_ids: [0, 1, 2, 3]    # trial i uses gpu_ids[i % len(gpu_ids)]
```

Trials are assigned to GPUs in round-robin order. On a 4-GPU node with `n_jobs: 4`, each trial gets its own GPU.

**Combining both** (multiple trials per GPU, across GPUs):

```yaml
n_jobs: 8
gpu_ids: [0, 1]           # 4 trials per GPU
gpu_allow_growth: true
```

### CPU-only runs

If running without a GPU, parallel trials work out of the box:

```yaml
n_jobs: 4    # runs 4 CPU trials in parallel
```

## Output

After the run, you will find:
- `multirun/` directory with one subfolder per trial
- `optimization_results.csv` with named objective columns (`cost_volume`, `cost_speed`) and all trial parameters
- Console summary of Pareto-optimal trials

## Using persistent storage / dashboard

The example configs already include `storage` and `study_name`. To visualize results:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db
```
