# Phase-Aware & Hierarchical DQN Cache Replacement for ChampSim
**Course:** SEM 7 - Reinforcement Learning  
**Platform:** ChampSim Simulator (LLC Replacement Policy Evaluation)

---

## 📌 Project Overview

This project implements and evaluates **Reinforcement Learning (RL) and Deep Q-Network (DQN)** techniques for Last-Level Cache (LLC) block replacement, outperforming standard heuristic baselines like LRU by learning from **Belady's Optimal (MIN)** lookahead policy under time-varying phase behaviors:

1. **Phase-Aware DQN**: Integrates an online KMeans phase classifier ($K=4$) with normalized recency and access type features (28-dimensional state space).
2. **Two-Level Hierarchical DQN (HDQN)**: Combines a High-Level Meta Agent (workload phase encoder) with a Low-Level Action Agent (per-way eviction ranker).
3. **Hardware & C++ Integration**: Deploys lightweight feedforward inference within ChampSim C++ replacement policies (`phase_aware_rl` and `hierarchical_rl`).
4. **Interactive Dashboard & Visualization**: Complete web-based simulator and publication-quality review figures.

---

## 📁 Repository Structure

```
ChampSim/
├── PROJECT_README.md                             # This guide & file index
├── champsim_config.json                          # ChampSim JSON configuration (configured for phase_aware_rl)
├── Makefile, config.sh                           # Build configuration
│
├── 📄 Academic Research Paper & Report (report/)
│   ├── main.tex                                  # IEEE journal/conference format LaTeX source
│   ├── main.pdf                                  # Compiled 8-page publication-quality PDF
│   └── figures/                                  # 10 High-resolution figures and UI screenshots
│
├── 🧠 Trained Models & Weights
│   ├── kmeans_phase_model.json                  # K=4 KMeans centroids & scaler parameters
│   ├── phase_aware_dqn_weights.json              # 28-dim Phase-Aware DQN weights (loaded by C++ policy)
│   ├── hierarchical_dqn_weights.json             # Two-level Hierarchical DQN weights (loaded by C++ policy)
│   └── dqn_weights.json                          # Baseline DQN weights
│
├── 🐍 Core Python Pipeline
│   ├── prepare_hierarchical_dataset_and_cluster.py # Data prep, True Belady labeler, & KMeans phase clustering
│   ├── train_discrete_phase_dqn.py               # Phase-Aware DQN training & scrambled-phase ablation
│   ├── train_hierarchical_dqn.py                 # Two-Level HDQN training (Meta + Action Agent)
│   ├── train_dqn.py                              # Baseline DQN offline training
│   ├── export_phase_aware_weights.py             # Exports trained PyTorch model to C++ JSON format
│   ├── convert_weights_to_js.py                  # Exports model weights to interactive_simulator/weights.js
│   └── generate_review_figures.py                # Generates all 9 presentation figures
│
├── ⚙️ C++ Replacement Policies (ChampSim modules)
│   ├── replacement/phase_aware_rl/               # Online Phase-Aware RL policy (loads JSON weights)
│   ├── replacement/hierarchical_rl/              # Hierarchical RL policy (loads JSON weights)
│   ├── replacement/champsim_phase_logger/        # Instrumentation logger for phase features
│   ├── replacement/hierarchical_rl_logger/       # Instrumentation logger for hierarchical features
│   └── replacement/rl_logger/                    # Base RL instrumentation logger
│
├── 📊 Datasets (datasets/)
│   ├── champsim_phase_log_gcc.csv                # GCC trace phase features log
│   ├── champsim_phase_log_mcf.csv                # MCF trace phase features log
│   ├── combined_phase_belady_dataset.csv         # Combined labeled dataset with True Belady & phase clusters
│   ├── hierarchical_rl_log_belady.csv            # Labeled dataset for hierarchical DQN
│   └── champsim_rl_log_belady.csv                # Labeled dataset for baseline DQN
│
├── 📈 Review Figures (review_figures/)
│   ├── 00_summary_dashboard.png                 # Executive summary dashboard
│   ├── 01_system_architecture.png                # System block diagram
│   ├── 02_training_curves.png                    # Loss and accuracy curves
│   ├── 03_phase_clustering.png                   # 2D PCA phase cluster visualization
│   ├── 04_phase_ablation.png                     # Real vs Scrambled phase ablation
│   ├── 05_accuracy_comparison.png                # LRU vs DQN vs Phase-Aware vs HDQN
│   ├── 06_eviction_heatmap.png                   # Per-way eviction distribution
│   ├── 07_miss_rate_timeline.png                 # Rolling miss rate timeline with phase overlays
│   └── 08_nn_architecture.png                    # Neural network architecture diagram
│
├── 🌐 Interactive Web Simulator (interactive_simulator/)
│   ├── index.html                                # Web dashboard UI
│   ├── app.js                                    # Simulator logic & real-time cache animator
│   ├── styles.css                                # Modern dark-mode UI styling
│   └── weights.js                                # Auto-generated model weights for browser inference
│
└── 📦 Traces (traces/)
    ├── 400.perlbench-50B.champsimtrace.xz        # SPEC CPU perlbench trace
    ├── 403.gcc-16B.champsimtrace.xz              # SPEC CPU gcc trace
    └── 429.mcf-22B.champsimtrace.xz              # SPEC CPU mcf trace
```

---

## 🚀 Execution Workflow

### Step 1: Preprocess Data & Fit Phase Clusters
Reads raw phase logs from `datasets/`, computes True Belady MIN optimal victims, fits the $K=4$ KMeans phase clustering model, and saves the labeled dataset:
```powershell
python prepare_hierarchical_dataset_and_cluster.py
```

### Step 2: Train Phase-Aware DQN
Trains the 28-dimensional DQN policy using an 80/20 chronological time-series split, evaluating against a scrambled phase control baseline:
```powershell
python train_discrete_phase_dqn.py
```

### Step 3: Export Weights for C++ Simulation and Web UI
Exports network weights to `phase_aware_dqn_weights.json` and converts them for browser execution:
```powershell
python export_phase_aware_weights.py
python convert_weights_to_js.py
```

### Step 4: Train Hierarchical DQN (Optional Comparison)
Trains the Two-Level HDQN and exports `hierarchical_dqn_weights.json`:
```powershell
python train_hierarchical_dqn.py
```

### Step 5: Generate Review Figures
Generates all 9 high-resolution figures into `review_figures/`:
```powershell
python generate_review_figures.py
```

### Step 6: Run in ChampSim Simulator
Build ChampSim with the `phase_aware_rl` replacement policy and run against benchmark traces:
```bash
./config.sh champsim_config.json
make
bin/champsim --warmup_instructions 1000000 --simulation_instructions 10000000 traces/429.mcf-22B.champsimtrace.xz
```

### Step 7: Launch Interactive Simulator
Open `interactive_simulator/index.html` in any web browser to run interactive trace simulations, view real-time cache set heatmaps, and test replacement decisions step-by-step.
