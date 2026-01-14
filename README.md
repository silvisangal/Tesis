# Thesis Repo — RL-guided Linear Cryptanalysis for SPECK32/64

This repository contains the code and experiments for my thesis on **linear cryptanalysis of reduced-round SPECK32/64** using **Reinforcement Learning (Q-learning)** and classical baselines inspired by **Matsui-style search**.

The workflow is:
1. Build a **combinatorial Linear Approximation Table (cLAT)** for SPECK32/64.
2. Optionally **prune** the cLAT (e.g., remove entries below a correlation-weight threshold).
3. Run experiments comparing **RL agents** (different reward/penalty schemes) against **paper/Matsui baselines**, under different mask sets and experimental conditions.
4. Store outputs (paths/trails, logs, CSV summaries) in a structured way for analysis and reproducibility.

---

## Repository structure

At a high level, the repo is organized as:

- **`experiments/`**: all experiment runs and outputs.
- **`helpers/`**: scripts used to build the cLAT and generate random mask pairs (utilities used across experiments).
- **`Matsui trails/`**: precomputed `.csv` trails computed with methods from the reference paper(s).

A simplified tree:

.
├── experiments/
│ ├── Entorno jerarquico con diferentes distribuciones/
│ ├── Hull experiments/
│ ├── Mask variations/
│ ├── Matsui/
│ ├── Paper comparisons/
│ └── preliminares/
├── helpers/
│ ├── clat_speck32.py
│ └── generar_mascaras_col.py
└── Matsui trails/
└── *.csv


### Experiment families

The following experiment folders share the same internal layout:

- `Hull experiments/`
- `Mask variations/`
- `Paper comparisons/`

Each one contains:
- **`unpruned/`**: uses the full cLAT
- **`pruned/`**: uses a pruned cLAT (filtered by a minimum correlation weight)

Inside each, results are split by agent/strategy:

<experiment_family>/
├── unpruned/
│ ├── greedy/
│ ├── death penalty/
│ └── decrease/
└── pruned/
├── greedy/
├── death penalty/
└── decrease/


**Interpretation of subfolders**
- **`greedy/`**: greedy policy / greedy rollouts (baseline RL-style behavior).
- **`death penalty/`**: includes strong negative reward/termination penalties to discourage invalid/low-value transitions.
- **`decrease/`**: reward shaping where rewards decrease over time/steps or by a cost schedule (trade-off quality vs. speed/stability).

---

## Matsui trails

**`Matsui trails/`** contains CSV files with trails computed using the **paper-based methods** (Matsui-style / baseline trail generation). These are used for comparison and/or as inputs for key-recovery or ranking experiments.

---

## Helpers

**`helpers/`** contains utility scripts used to:
- build the **cLAT** for SPECK32/64,
- inspect/check intersections,
- **prune** the cLAT by correlation-weight threshold,
- generate **random mask pairs** to create experimental mask sets.

---

## Commands

> Run these from the repository root.  
> If your scripts are inside `helpers/`, call them as `python helpers/<script>.py`.

### 1) Build cLAT

```bash
python helpers/clat_speck32.py --build --out "<name>.pkl.gz"

Creates a compressed pickle with the cLAT.

2) Check intersections / query entries
python helpers/clat_speck32.py --check "<name>.pkl.gz" \
  --w16-for-u16-v16 0x0004 0x0006 \
  --max-bound 3 \
  --limit <limit>


--w16-for-u16-v16 <u16> <v16>: query using specific 16-bit masks (hex).

--max-bound: maximum correlation-weight bound to consider.

--limit: caps the amount of output (useful for large tables).

# 3) Build cLAT
python helpers/clat_speck32.py --build --out "<name>.pkl.gz"


Output is the complete cLAT file used in the unpruned/ experiment folders.

# 4) Prune cLAT
python helpers/clat_speck32.py --check "<name>.pkl.gz" \
  --prune --min-cw 1 \
  --out "clat_speck_32_cw>=1.pkl.gz"


--min-cw: minimum correlation weight to keep.

Output is a pruned cLAT file used in the pruned/ experiment folders.

# 5) Generate random mask pairs
python helpers/generar_mascaras_col.py --x mask1 --y mask2 \
  -n <num_pares_mascaras> \
  -o <nombre_archivo>.txt


Generates a text file containing random mask pairs for experiments (e.g., Mask variations/).

Reproducibility notes

- To reproduce results, keep consistent:
- the same cLAT file(s),

- the same mask-pair files,

- fixed random seeds (where applicable),

- consistent naming for output CSV/log files inside each run folder.

## Quick start (suggested)

- Baseline trails from the paper: **`Matsui trails/`** (`*.csv`)
- cLAT generation pipeline: **`helpers/clat_speck32.py`**
- RL comparisons across pruning and reward schemes:
  - **`Hull experiments/`**
  - **`Mask variations/`**
  - **`Paper comparisons/`**
