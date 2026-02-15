# IGM Optuna Sweeper — Full Reference

IGM includes a built-in Hydra sweeper plugin (`igm_optuna`) that uses [Optuna](https://optuna.org/) for parameter optimization. It supports single- and multi-objective optimization, parallel trials, GPU distribution, and persistent storage with dashboard visualization.

This replaces the upstream `hydra-optuna-sweeper` (which pins `optuna<3.0`) with a lightweight alternative that uses Optuna directly with no version ceiling.

---

## 1. Overview

The optimization workflow has two parts:

1. **User process** (`eval_objective.py`): computes `state.score` as a dictionary of all available metrics at the end of a simulation.
2. **Optuna config** (`optuna_params.yaml`): selects which scores to optimize, defines control parameters, target values, sampler, and parallelism.

Each trial runs as a **separate subprocess** with its own TensorFlow session, ensuring clean graph state between trials.

---

## 2. Defining scores in user code

Create a user process module (e.g. `user/code/processes/eval_objective.py`) that computes all available metrics in `finalize()`:

```python
import tensorflow as tf

def initialize(cfg, state):
    pass

def update(cfg, state):
    pass

def finalize(cfg, state):
    ecfg = cfg.processes.eval_objective
    dx = state.dx

    volume_km3 = float(tf.reduce_sum(state.thk) * dx * dx / 1.0e9)
    max_thk = float(tf.reduce_max(state.thk))
    velsurf_mag = tf.norm(
        tf.stack([state.uvelsurf, state.vvelsurf], axis=-1), axis=-1
    )
    max_speed = float(tf.reduce_max(velsurf_mag))

    state.score = {
        "cost_volume": abs(volume_km3 - ecfg.target_volume),
        "cost_speed": abs(max_speed - ecfg.target_max_speed),
        "cost_thickness": abs(max_thk - ecfg.target_max_thk),
    }
```

The corresponding config (`user/conf/processes/eval_objective.yaml`) provides default target values:

```yaml
target_volume: 18.0      # km^3
target_max_speed: 200.0  # m/y
target_max_thk: 300.0    # m
```

The `state.score` dictionary can contain as many metrics as desired. The `optuna_params.yaml` file then selects which ones to actually optimize.

---

## 3. The `optuna_params.yaml` config file

All optimization settings live in a single YAML file (default name: `optuna_params.yaml`). Below is a complete reference with all available options.

### 3.1 Objectives

Select which scores from `state.score` to optimize:

```yaml
objectives:
  - name: cost_volume      # key in state.score dict
    direction: minimize
  - name: cost_speed       # key in state.score dict
    direction: minimize
```

- Each entry must have `name` (matching a key in `state.score`) and `direction` (`minimize` or `maximize`).
- Use a single entry for single-objective optimization, multiple entries for multi-objective.
- You can change which scores are active without modifying any Python code.

### 3.2 Control parameters

Parameters tuned by Optuna. Each maps to a Hydra config path:

```yaml
parameters:
  - name: processes.iceflow.physics.init_slidingco
    type: float
    low: 0.01
    high: 1.0
    log: true       # sample in log-space (good for parameters spanning orders of magnitude)

  - name: processes.iceflow.physics.init_arrhenius
    type: float
    low: 30.0
    high: 150.0

  - name: processes.smb_simple.array.1.3    # ELA (row 1, column 3 of the SMB array)
    type: float
    low: 2500.0
    high: 3200.0
    log: false
```

Supported types:

| Type | Required fields | Optional |
|------|----------------|----------|
| `float` | `low`, `high` | `log` (default: `false`) |
| `int` | `low`, `high` | `log` (default: `false`) |
| `categorical` | `choices` (list) | — |

The `name` field is the full Hydra override path. Optuna suggests a value within the specified range, and it is passed to `igm_run` as `name=value`.

### 3.3 Fixed overrides (target values)

Values passed to every trial as Hydra overrides (not tuned by Optuna):

```yaml
overrides:
  processes.eval_objective.target_volume: 18.0
  processes.eval_objective.target_max_speed: 200.0
```

This keeps both controls and targets in the same file for clarity.

### 3.4 Trials and parallelism

```yaml
n_trials: 200       # total number of trials to run
n_jobs: 1           # number of parallel trials (default: 1 = sequential)
```

When `n_jobs > 1`, trials are launched as parallel subprocesses. See Section 5 for GPU considerations.

### 3.5 Sampler

```yaml
sampler:
  method: TPESampler       # Optuna sampler class name
  population_size: 10      # extra kwargs passed to the sampler constructor
```

Available samplers (all from `optuna.samplers`):

| Method | Multi-obj? | Best for |
|--------|-----------|----------|
| `TPESampler` | Yes (MOTPE) | General purpose, fast convergence, good default |
| `NSGAIISampler` | Yes | Multi-objective, diverse Pareto front (2-3 obj) |
| `NSGAIIISampler` | Yes | Multi-objective, best for 3+ objectives (Optuna >= 3.2) |
| `CmaEsSampler` | No | Continuous parameters, covariance-based (single-obj only) |
| `GPSampler` | No | Gaussian Process surrogate, small budget (single-obj only) |
| `RandomSampler` | Yes | Baseline / exploration |
| `QMCSampler` | Yes | Quasi-Monte Carlo, uniform space coverage |

Any keyword argument accepted by the sampler constructor can be added under `sampler:` (e.g. `population_size`, `seed`, `n_startup_trials`).

### 3.6 Pruner (optional)

```yaml
pruner:
  method: MedianPruner
  n_startup_trials: 5
  n_warmup_steps: 10
```

Any Optuna pruner from `optuna.pruners` can be used. Omit this section if pruning is not needed.

### 3.7 Storage and study name

```yaml
storage: sqlite:///optuna.db
study_name: my_experiment
```

- `storage`: Optuna storage URL. Use `sqlite:///optuna.db` for a local SQLite file. Omit for in-memory storage.
- `study_name`: name of the study. When `storage` is set, the study is created or resumed if it already exists (`load_if_exists=True`).

### 3.8 Trial timeout (optional)

```yaml
trial_timeout: 3600    # seconds, kill trial if it exceeds this
```

---

## 4. Running the optimization

```bash
# Uses optuna_params.yaml by default
igm_run -m +experiment=params hydra/sweeper=igm_optuna

# Specify a custom config file
igm_run -m +experiment=params hydra/sweeper=igm_optuna \
  hydra.sweeper.optuna_config=my_other_config.yaml
```

The `-m` flag enables Hydra **multirun** mode (required for sweepers).

---

## 5. GPU distribution

By default, TensorFlow grabs all available GPU memory. When running multiple trials in parallel, this causes crashes. Three strategies:

### Option A — Share one GPU with memory growth

```yaml
n_jobs: 4
gpu_allow_growth: true
```

Sets `TF_FORCE_GPU_ALLOW_GROWTH=true` per trial. Each trial allocates only the memory it needs.

### Option B — One trial per GPU (round-robin)

```yaml
n_jobs: 4
gpu_ids: [0, 1, 2, 3]
```

Trial *i* uses `gpu_ids[i % len(gpu_ids)]` via `CUDA_VISIBLE_DEVICES`.

### Option C — Multiple trials per GPU, across GPUs

```yaml
n_jobs: 8
gpu_ids: [0, 1]
gpu_allow_growth: true
```

8 trials distributed round-robin across 2 GPUs, 4 trials per GPU, with memory growth enabled.

### CPU-only

No special configuration needed:

```yaml
n_jobs: 4
```

---

## 6. Cluster deployment

For multi-node clusters, each node can run its own `igm_run -m` command pointing to the **same SQLite database** (via shared filesystem) and **same `study_name`**. Optuna's `load_if_exists=True` ensures all nodes contribute to the same study.

Example launch script (`run.sh`):

```bash
#!/bin/bash
GPU_ID=$1
N_TRIALS=$2
igm_run -m +experiment=params hydra/sweeper=igm_optuna \
  hydra.sweeper.optuna_config=optuna_params.yaml
```

Example cluster dispatch (SSH + tmux):

```bash
#!/bin/bash
NB=50
for GPU in 0 1 2 3; do
  for NODE in node01 node02 node03; do
    SESSION="${NODE}_GPU${GPU}"
    ssh $NODE "tmux new-session -d -s $SESSION 'cd /path/to/experiment && ./run.sh $GPU $NB'"
    sleep 5
  done
done
```

All trials land in the same `optuna.db` and can be monitored together.

---

## 7. Output

After a run completes:

- **`multirun/`** — one subfolder per trial containing the full IGM output and `_igm_score.json`
- **`optimization_results.csv`** — all trials with named objective columns and parameter values
- **Console summary** — Pareto-optimal trials (multi-objective) or best trial (single-objective)

---

## 8. Dashboard

To visualize optimization progress interactively:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db
```

The dashboard shows:
- Objective values per trial
- Best parameters
- Parameter importance
- Parallel coordinate plots
- Pareto front (multi-objective)

Alternatively, upload `optuna.db` to the [Optuna Dashboard web app](https://optuna.github.io/optuna-dashboard/).

---

## 9. Complete `optuna_params.yaml` template

```yaml
# --- Objectives (selected from state.score dict) ---
objectives:
  - name: cost_volume
    direction: minimize
  - name: cost_speed
    direction: minimize

# --- Trials and parallelism ---
n_trials: 200
n_jobs: 2
gpu_allow_growth: true
# gpu_ids: [0, 1, 2, 3]
# trial_timeout: 3600

# --- Persistent storage ---
storage: sqlite:///optuna.db
study_name: my_experiment

# --- Sampler ---
sampler:
#  method: TPESampler          # Bayesian, fast convergence, good default (1+ obj)
  method: NSGAIISampler        # Genetic, good Pareto diversity (2-3 obj)
#  method: NSGAIIISampler      # Genetic, best for 3+ objectives
#  method: CmaEsSampler        # Covariance-based (1 obj only)
#  method: GPSampler           # Gaussian Process surrogate (1 obj only)
  population_size: 10          # for NSGA-II/III only

# --- Pruner (optional) ---
# pruner:
#   method: MedianPruner
#   n_startup_trials: 5

# --- Fixed overrides (target values, applied to every trial) ---
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

  - name: processes.smb_simple.array.1.3    # ELA
    type: float
    low: 2500.0
    high: 3200.0
```
