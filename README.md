# FRODO-V2X: Federated / Flexible Resource Allocation & Privacy Protection in V2X Offloading

Official repository for **FRODO**, a framework combining deep reinforcement learning (Rainbow-DQN) for flexible V2X task offloading and resource allocation with finite-field secret sharing privacy protection schemes (**GMDS**, **SPC**, and **FRODO**).

---

## 📌 Repository Overview

This repository contains the complete codebase, benchmark suite, simulation data, and pre-trained model checkpoints:

```text
FRODO-V2X/
├── src/                                         # Core Python Source Code
│   ├── main_rainbow_repaired.py                 # Simulation environment & Rainbow-DQN Agent
│   ├── baseline_agents.py                       # D3QN and Greedy Heuristic Agents
│   └── FRODO_Privacy_Tests_Aligned.py           # Finite-field GMDS/SPC/FRODO privacy test suite
│
├── experiments/                                 # Reproduction & Benchmark Suite
│   ├── run_rainbow_repaired_frozen_robustness_table_v1.py   # Frozen policy robustness evaluations
│   ├── run_cross_method_paired_task_size_sweep_rainbow_repaired.py # Task-size scaling benchmarks
│   ├── test_scarcity_sweep.py                   # Spectrum bandwidth scarcity sweep
│   ├── run_effective_sidelink_capacity_stress.py# Sidelink pool capacity stress test
│   ├── generate_persistent_uv_highway_test.py   # Persistent vehicle trajectory generator
│   └── analyze_persistent_uv_highway_test.py    # Persistent vehicle trajectory analyzer
│
├── notebooks/                                   # Analysis & Publication Figure Generators
│   ├── Ablation.ipynb                           # Ablation study notebook
│   ├── FRODO_Privacy_Tests.ipynb                # Privacy verification notebook
│   └── VPS_IEEE_Minimal_Figures_Updated_Style.ipynb # IEEE publication figures notebook
│
├── data/                                        # Datasets & Simulation Parameters
│   ├── lust_fcd_sample.csv                      # LuST SUMO mobility trace sample
│   ├── controlled_uv_trajectory_manifest.json   # Controlled vehicle trajectory schedule
│   └── tables/                                  # Parameter & benchmark CSVs
│
└── checkpoints/                                 # Pre-trained Model Checkpoints
    ├── rainbow_model_seed_0.pt
    ├── rainbow_model_seed_1.pt
    ├── rainbow_model_seed_2.pt
    └── rainbow_model_seed_3.pt
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/your-username/FRODO-V2X.git
cd FRODO-V2X
pip install -r requirements.txt
```

### 2. Privacy Test Suite

To run the finite-field privacy tests (Temporal Linkage MI & Same-Product Distinguishability Attack):

```bash
python src/FRODO_Privacy_Tests_Aligned.py
```

Outputs and high-resolution figures (`same_product_attack.png`, `temporal_linkage_mi.png`) will be saved to `privacy_test_outputs_aligned/`.

### 3. Training & Environment Simulation

To run the main Rainbow-DQN simulation and training pipeline:

```bash
python src/main_rainbow_repaired.py
```

### 4. Robustness & Benchmark Sweeps

To run the frozen-policy robustness benchmark across environmental parameter variations:

```bash
python experiments/run_rainbow_repaired_frozen_robustness_table_v1.py
```

---

## 🔒 Privacy Protection Schemes Summary

- **GMDS (Global Masking Secret Sharing)**: Uses global fixed nullspace masks across servers. Vulnerable under temporal differencing and same-product attacks.
- **SPC (Server-Specific Fixed Masking)**: Uses static server-specific nullspace masks. Prevents cross-server leakage but vulnerable under temporal differencing.
- **FRODO (Fresh Random Online Differential Offloading)**: Samples fresh nullspace masks per step/operand, achieving complete temporal linkage defense and chance-level distinguishability under active attacks.

---

## 📜 License & Citation

Distributed under the MIT License.
