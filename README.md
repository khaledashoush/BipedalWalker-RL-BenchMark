# 🤖 BipedalWalker Deep RL Benchmarking Project

> **Comprehensive benchmarking of SAC, PPO, TD3, and DDPG on continuous locomotion control**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.2.1-green)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏆 Key Results

| Environment | 🥇 Best | 🥈 2nd | 🥉 3rd | 4th |
|-------------|---------|--------|--------|-----|
| **Normal** | **TD3 (296.80)** | SAC (290.35) | PPO (289.34) | DDPG (226.92) |
| **Hardcore** | **SAC (205.73)** | TD3 (176.68) | PPO (-86.43) | — |

**Best Overall Performance:** SAC with GPU + SDE (297.05 on single track)

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Overview](#-project-overview)
- [Environment Details](#-environment-details)
- [File Structure](#-file-structure)
- [Training Pipeline](#-training-pipeline)
- [Results Summary](#-results-summary)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/bipedalwalker-rl.git
cd bipedalwalker-rl

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train baseline models (all 4 algorithms)
python train.py

# 5. Generate comparison visualization
python compare.py
```

---

## 📖 Project Overview

This project benchmarks **4 state-of-the-art Deep Reinforcement Learning algorithms** on the `BipedalWalker-v3` and `BipedalWalkerHardcore-v3` environments from [OpenAI Gymnasium](https://gymnasium.farama.org/).

### Algorithms Benchmarked

| # | Algorithm | Type | Policy | Key Feature |
|---|-----------|------|--------|-------------|
| 1 | **SAC** | Off-policy | Stochastic | Maximum Entropy Framework |
| 2 | **PPO** | On-policy | Stochastic | Clipped Surrogate Objective |
| 3 | **TD3** | Off-policy | Deterministic | Twin Delayed DDPG |
| 4 | **DDPG** | Off-policy | Deterministic | Baseline Actor-Critic |

### Environments

| # | Environment | Difficulty | Max Steps | Features |
|---|-------------|------------|-----------|----------|
| 1 | `BipedalWalker-v3` | Normal | 1,600 | Flat terrain, slight variations |
| 2 | `BipedalWalkerHardcore-v3` | Hardcore | 2,000 | Stairs, holes, obstacles, pits |

### State Space (24 dimensions)

| # | Component | Dimensions | Description |
|---|-----------|------------|-------------|
| 1 | Body State | 4 | Hull angle, angular velocity, horizontal & vertical velocity |
| 2 | Right Leg | 5 | Hip/knee angles, angular velocities, ground contact flag |
| 3 | Left Leg | 5 | Hip/knee angles, angular velocities, ground contact flag |
| 4 | LIDAR | 10 | Terrain distance measurements (160° forward arc) |

### Action Space (4 continuous dimensions)

| # | Action | Range | Description |
|---|--------|-------|-------------|
| 1 | a₁ | [-1, 1] | Right hip torque |
| 2 | a₂ | [-1, 1] | Right knee torque |
| 3 | a₃ | [-1, 1] | Left hip torque |
| 4 | a₄ | [-1, 1] | Left knee torque |

### Reward Function

```
R(t) = v_forward - c_control - c_fall
```

| Component | Value | Description |
|-----------|-------|-------------|
| v_forward | Variable | Positive reward for forward velocity |
| c_control | -Σ(action²) | Penalty for energy consumption |
| c_fall | -100 | Terminal penalty for falling |
| Completion Bonus | +300 | For completing full episode |

---

## 📁 File Structure

```
bipedalwalker-rl/
│
├── 📄 train.py                 # Baseline training (all 4 algorithms, 1M steps)
├── 📄 retrain_ddpg.py          # DDPG v3 improved (OU noise, larger network, 2M steps)
├── 📄 retain_ddpg.py           # DDPG v2 intermediate (larger network, 1M steps)
├── 📄 train_hardcore.py        # Transfer to Hardcore environment (5M steps)
├── 📄 train_hardcore_v2.py     # Refine Hardcore training (5M steps)
├── 📄 train_best.py            # Best training with auto-stop (1M + 10M steps)
├── 📄 train_curriculum.py      # Curriculum learning (500→1000→2000 steps)
├── 📄 train_gpu_sde.py         # GPU + State-Dependent Exploration (up to 15M steps)
├── 📄 train_SAC_only.py        # Intensive SAC training (10M steps)
│
├── 📄 compare.py               # Full 7-panel comparison visualization
├── 📄 watch.py                 # Live evaluation + video recording
├── 📄 record_compare.py        # Comparative video recording (Normal vs Hardcore)
├── 📄 record_seeds.py          # Fixed seed track evaluation (seeds: 42, 123, 456, 789)
├── 📄 record_hardcore.py       # Hardcore v2 model evaluation
├── 📄 record_curriculum.py     # Curriculum model evaluation
│
├── 📄 test.py                  # Random action environment test
├── 📄 test_best.py             # Quick evaluation of best SAC model
│
├── 📄 README.md                # This file
├── 📄 requirements.txt         # Python dependencies
├── 📄 .gitignore               # Git exclusions
└── 📄 LICENSE                  # MIT License
```

---

## 🔄 Training Pipeline

```
Phase 0: Baseline Training
    └── train.py (1M steps each algorithm)
        ├── models/SAC_final
        ├── models/PPO_final
        ├── models/TD3_final
        └── models/DDPG_final
            │
            ▼
Phase 1: DDPG Improvement
    ├── retain_ddpg.py (v2: [400,300] network, 1M steps)
    └── retrain_ddpg.py (v3: OU noise, 2M steps)
        └── models/DDPG_v3_final
            │
            ▼
Phase 2: Hardcore Transfer (v1)
    └── train_hardcore.py (5M steps)
        └── models/hardcore/*_v2_final
            │
            ▼
Phase 3: Hardcore Refinement (v2)
    └── train_hardcore_v2.py (5M steps)
        └── models/hardcore_v2/*_final
            │
            ▼
Phase 4: Best Training + Curriculum
    ├── train_best.py (Normal: 1M, Hardcore: 10M)
    └── train_curriculum.py (500→1000→2000 steps)
        └── models/curriculum/phase3_best
            │
            ▼
Phase 5: GPU + SDE Advanced
    └── train_gpu_sde.py (up to 15M steps, CUDA)
        └── models/gpu_sde/*_final
            │
            ▼
Phase 6: SAC Intensive
    └── train_SAC_only.py (10M steps)
        └── models/SAC_best/best/best_model
```

---

## 📊 Results Summary

### Normal Environment Results (1M Timesteps)

| Rank | Algorithm | Mean Reward | Std Dev | Status |
|------|-----------|-------------|---------|--------|
| 🥇 1 | **TD3** | **296.80** | 1.48 | ✅ Solved |
| 🥈 2 | **SAC** | **290.35** | 2.21 | ✅ Solved |
| 🥉 3 | **PPO** | **289.34** | 1.98 | ✅ Solved |
| 4 | DDPG v3 | 226.92 | 117.08 | ⚠️ Unstable |
| — | DDPG v1 | 151.6 | 166.8 | ❌ Poor |

### Hardcore Environment Results (After Full Training)

| Rank | Algorithm | Mean Reward | Std Dev | Training |
|------|-----------|-------------|---------|----------|
| 🥇 1 | **SAC** | **205.73** | 124.03 | GPU + SDE |
| 🥈 2 | **TD3** | **176.68** | 117.08 | GPU + SDE |
| 🥉 3 | **PPO** | **-86.43** | 25.88 | GPU + SDE |
| — | SAC (Best Track) | **297.05** | — | Single episode |

### SAC Training Progression

| Phase | Training | Mean Reward | Improvement | Cumulative Steps |
|-------|----------|-------------|-------------|------------------|
| 0 | Zero-shot | -77.0 | — | — |
| 1 | Hardcore v1 (5M) | 49.5 | +126.5 | 5M |
| 2 | Hardcore v2 (5M) | 146.7 | +97.2 | 10M |
| 3 | Curriculum (10M) | 187.1 | +40.4 | 20M |
| 4 | GPU + SDE (15M) | **297.1** | +110.0 | 35M |

### DDPG Improvement Trajectory

| Version | Changes | Mean Reward | Improvement |
|---------|---------|-------------|-------------|
| v1 | Original [256,256], Gaussian noise | 151.6 | Baseline |
| v2 | [400,300] network | 198.9 | +31% |
| v3 | + OU Noise + 2M steps | 226.9 | +49% |
| Best | Single episode peak | 305.98 | — |

### Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **TD3 best on Normal** | 296.80 mean, lowest std (1.48) |
| 2 | **SAC best on Hardcore** | 205.73 mean, best generalization |
| 3 | **PPO fast but limited** | Best early convergence, fails on Hardcore |
| 4 | **DDPG fundamentally unstable** | High variance even after tuning |
| 5 | **Transfer learning crucial** | +126.5 improvement from Normal→Hardcore |
| 6 | **Curriculum helps** | +40.4 from progressive difficulty |
| 7 | **GPU+SDE optimal** | 297.1 peak performance |
| 8 | **Fixed seeds enable fair comparison** | Eliminates terrain variability |

---

## 🔧 Prerequisites

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y swig cmake build-essential python3-dev ffmpeg
```

### macOS

```bash
brew install swig cmake ffmpeg
```

### Windows

1. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install [SWIG](http://www.swig.org/download.html)
3. Install [FFmpeg](https://ffmpeg.org/download.html) and add to PATH

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/bipedalwalker-rl.git
cd bipedalwalker-rl

# Create virtual environment (recommended)
python -m venv venv

# Activate environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

### Training

```bash
# Baseline training (all 4 algorithms, 1M steps each)
python train.py

# Improved DDPG (v3 with OU noise, 2M steps)
python retrain_ddpg.py

# Transfer to Hardcore environment
python train_hardcore.py

# Continue Hardcore refinement
python train_hardcore_v2.py

# Best training with automatic stopping
python train_best.py

# Curriculum learning (gradual difficulty)
python train_curriculum.py

# GPU + SDE advanced training
python train_gpu_sde.py

# Intensive SAC training
python train_SAC_only.py
```

### Evaluation

```bash
# Full comparison visualization (7-panel figure)
python compare.py

# Evaluate and watch best agent live
python watch.py

# Record comparison videos
python record_compare.py

# Fixed seed evaluation (fair comparison)
python record_seeds.py

# Evaluate Hardcore v2 models
python record_hardcore.py

# Evaluate curriculum model
python record_curriculum.py

# Quick test of best model
python test_best.py

# Random action test (environment check)
python test.py
```

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser at http://localhost:6006
```

---

## 🐛 Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `No module named 'gymnasium'` | Not installed | `pip install gymnasium[box2d]` |
| `Box2D not found` | Missing SWIG | Install SWIG (see Prerequisites) |
| `error: Microsoft Visual C++ 14.0 is required` | Missing compiler | Install Visual C++ Build Tools |
| `CUDA out of memory` | GPU memory full | Reduce `batch_size` or use `device="cpu"` |
| `Model file not found` | Training not done | Run `train.py` first |
| `Video recording fails` | Missing FFmpeg | `sudo apt-get install ffmpeg` |
| `Permission denied` | File locked | Close TensorBoard or other processes |
| `Segmentation fault` | Box2D conflict | Reinstall: `pip install --force-reinstall gymnasium[box2d]` |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate docstrings.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OpenAI Gymnasium](https://gymnasium.farama.org/) for the BipedalWalker environment
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithm implementations
- [PyTorch](https://pytorch.org/) for the deep learning backend
- [Farama Foundation](https://farama.org/) for maintaining Gymnasium

### References

| # | Paper | Authors | Year |
|---|-------|---------|------|
| 1 | Continuous Control with Deep Reinforcement Learning | Lillicrap et al. | 2016 (DDPG) |
| 2 | Addressing Function Approximation Error in Actor-Critic Methods | Fujimoto et al. | 2018 (TD3) |
| 3 | Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL | Haarnoja et al. | 2018 (SAC) |
| 4 | Proximal Policy Optimization Algorithms | Schulman et al. | 2017 (PPO) |
| 5 | Stable-Baselines3: Reliable RL Implementations | Raffin et al. | 2021 |

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 17 |
| Training Scripts | 9 |
| Evaluation Scripts | 6 |
| Test Scripts | 2 |
| Algorithms Benchmarked | 4 |
| Environment Variants | 2 |
| Total Training Steps | ~71M (all algorithms combined) |
| Evaluation Episodes | 50+ per model |
| Fixed Seed Tracks | 4 |
| DDPG Variants Tested | 3 |
| Training Phases | 7 |

---

<div align="center">

**🎯 Project Goal:** Identify optimal RL algorithms for continuous locomotion control

**🏆 Winner:** SAC for Hardcore | TD3 for Normal

**📊 Total Experiments:** 50+ training runs | 200+ evaluation episodes

</div>
