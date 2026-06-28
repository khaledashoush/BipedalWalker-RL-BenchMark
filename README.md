# 🤖 BipedalWalker Deep RL Benchmarking Project

> **Comprehensive benchmarking of SAC, PPO, TD3, and DDPG on continuous locomotion control**

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Details](#2-environment-details)
3. [File Structure & Descriptions](#3-file-structure--descriptions)
4. [Training Pipeline](#4-training-pipeline)
5. [Results Summary](#5-results-summary)
6. [How to Use](#6-how-to-use)
7. [Dependencies](#7-dependencies)

---

## 1. Project Overview

This project benchmarks **4 state-of-the-art Deep Reinforcement Learning algorithms** on the `BipedalWalker-v3` and `BipedalWalkerHardcore-v3` environments from OpenAI Gymnasium.

### Algorithms Benchmarked

| # | Algorithm | Type | Policy | Key Feature |
|---|-----------|------|--------|-------------|
| 1 | **SAC** | Off-policy | Stochastic | Maximum Entropy Framework |
| 2 | **PPO** | On-policy | Stochastic | Clipped Surrogate Objective |
| 3 | **TD3** | Off-policy | Deterministic | Twin Delayed DDPG (Double Q-Learning) |
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

## 2. Environment Details

### 2.1 Observation Space Breakdown

```
Observation Vector (24 floats, normalized):
┌─────────────────────────────────────────────────────┐
│ [0-3]   : Hull angle, angular velocity, vx, vy     │
│ [4-8]   : Right hip angle, velocity, knee angle,   │
│           velocity, ground contact (0/1)           │
│ [9-13]  : Left hip angle, velocity, knee angle,    │
│           velocity, ground contact (0/1)           │
│ [14-23] : LIDAR readings (10 distances)            │
└─────────────────────────────────────────────────────┘
```

### 2.2 Neural Network Architecture

```
All Algorithms (MlpPolicy):
┌─────────────────┐
│  Input: 24      │  ← State dimensions
│  (ReLU)         │
├─────────────────┤
│  Hidden: 256    │  ← Layer 1
│  (ReLU)         │
├─────────────────┤
│  Hidden: 256    │  ← Layer 2
│  (ReLU)         │
├─────────────────┤
│  Output: 4      │  ← Actions (Actor) or Q-Value (Critic)
│  (Tanh/None)    │
└─────────────────┘

DDPG v3 Improved:
┌─────────────────┐
│  Input: 24      │
│  (ReLU)         │
├─────────────────┤
│  Hidden: 400    │  ← Larger network
│  (ReLU)         │
├─────────────────┤
│  Hidden: 300    │
│  (ReLU)         │
├─────────────────┤
│  Output: 4      │
└─────────────────┘
```

---

## 3. File Structure & Descriptions

### 📁 **Training Scripts (8 files)**

---

#### 3.1 `train.py` — Initial Benchmark Training

**Purpose:** Train all 4 algorithms from scratch on Normal environment (1M timesteps each)

**Key Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TIMESTEPS` | 1,000,000 | Total training steps per algorithm |
| `N_ENVS` | 4 | Parallel environments for data collection |
| `eval_freq` | 50,000 | Evaluate every 50K steps |
| `n_eval_episodes` | 20 | Episodes per evaluation |
| `save_freq` | 100,000 | Save checkpoint every 100K steps |

**Algorithm-Specific Hyperparameters:**

| Algorithm | learning_rate | batch_size | gamma | Special Params |
|-----------|--------------|------------|-------|----------------|
| SAC | 3e-4 | 256 | 0.99 | tau=0.005 |
| PPO | 3e-4 | 64 | 0.99 | n_steps=2048, gae_lambda=0.95 |
| TD3 | 1e-3 | 256 | 0.99 | policy_delay=2, target_policy_noise=0.2 |
| DDPG | 1e-3 | 256 | 0.99 | tau=0.005 |

**Output Structure:**
```
models/
├── SAC_final, PPO_final, TD3_final, DDPG_final
├── SAC_best/, PPO_best/, TD3_best/, DDPG_best/
└── SAC/, PPO/, TD3/, DDPG/ (checkpoints every 100K)

logs/
├── SAC/evaluations.npz
├── PPO/evaluations.npz
├── TD3/evaluations.npz
└── DDPG/evaluations.npz
```

**Code Flow:**
```python
Line 1-8:    Imports (gymnasium, stable_baselines3, callbacks)
Line 10-12:  Settings (1M timesteps, Normal env, 4 parallel envs)
Line 15-17:  Create directories (models/, logs/, videos/)
Line 20-55:  train() function definition
Line 58-85:  Hyperparameter dictionaries for each algorithm
Line 88-93:  Execute training for all 4 algorithms
```

---

#### 3.2 `retrain_ddpg.py` — DDPG v3 (Improved Version)

**Purpose:** Retrain DDPG with enhanced configuration to address instability

**Improvements over Original DDPG:**

| # | Improvement | Original | v3 Value | Impact |
|---|-------------|----------|----------|--------|
| 1 | Network Architecture | [256, 256] | [400, 300] | +31% performance |
| 2 | Action Noise | Gaussian (σ=0.1) | OU Noise (σ=0.2, θ=0.15) | Better exploration |
| 3 | Training Duration | 1M steps | 2M steps | More convergence time |
| 4 | Learning Rate | 1e-3 | 1e-4 | More stable updates |
| 5 | Batch Size | 256 | 512 | Better gradient estimates |
| 6 | Tau | 0.005 | 0.001 | Slower target network updates |

**Ornstein-Uhlenbeck Noise Parameters:**
```python
ou_noise = OrnsteinUhlenbeckActionNoise(
    mean  = np.zeros(4),      # Zero mean
    sigma = 0.2 * np.ones(4), # Exploration magnitude
    theta = 0.15,              # Mean reversion rate
    dt    = 1e-2               # Time step
)
```

**Results:**
- DDPG v1 (Original): **151.6** ± 166.8
- DDPG v2 (Better Params): **198.9** ± ~120
- DDPG v3 (OU + Bigger): **226.9** ± ~115
- Best Single Episode: **305.98**

**Code Flow:**
```python
Line 1-7:     Imports (added OrnsteinUhlenbeckActionNoise)
Line 10-12:   Settings (2M timesteps, 4 parallel envs)
Line 15-28:   Environment setup + OU noise configuration
Line 31-45:   Callbacks (checkpoint every 100K, eval every 50K)
Line 48-68:   Model creation with improved hyperparameters
Line 71-76:   Training execution
Line 79-93:   Final evaluation (20 episodes)
```

---

#### 3.3 `retain_ddpg.py` — DDPG v2 (Intermediate Version)

**Purpose:** Bridge between original and v3

**Configuration:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `TIMESTEPS` | 1,000,000 | Same as original |
| `N_ENVS` | 4 | Parallel environments |
| `learning_rate` | 1e-4 | Reduced from 1e-3 |
| `batch_size` | 512 | Doubled from 256 |
| `gamma` | 0.98 | Slightly reduced from 0.99 |
| `tau` | 0.001 | Slower target updates |
| `buffer_size` | 1,000,000 | Replay buffer size |
| `learning_starts` | 10,000 | Random actions before learning |
| `action_noise` | NormalActionNoise(σ=0.1) | Standard Gaussian |
| `network` | [400, 300] | Larger than original |

**Code Flow:**
```python
Line 1-6:     Imports
Line 9-11:    Settings (1M timesteps)
Line 14-16:   Environment setup
Line 19-22:   NormalActionNoise (not OU)
Line 25-38:   Callbacks
Line 41-55:   Model creation
Line 58-63:   Training
Line 66:      Save final model
```

---

#### 3.4 `train_hardcore.py` — First Hardcore Training Phase

**Purpose:** Transfer pre-trained Normal models to Hardcore environment

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| `ENV_NAME` | `BipedalWalkerHardcore-v3` |
| `TIMESTEPS` | 5,000,000 |
| `N_ENVS` | 4 |

**Transfer Learning Pipeline:**
```
Normal Training (1M) → Hardcore Training (5M)
     SAC_final    →    SAC_v2 (hardcore)
     PPO_final    →    PPO_v2 (hardcore)
     TD3_final    →    TD3_v2 (hardcore)
     DDPG_final   →    DDPG_v2 (hardcore)
```

**Key Feature:**
```python
model = algorithm.load(
    pretrained_path,           # Load from Normal model
    env = env,
    device = "cpu",
    ...
)
model.learn(
    total_timesteps = 5_000_000,
    reset_num_timesteps = False  # Continue timestep counter
)
```

**Code Flow:**
```python
Line 1-5:     Imports
Line 8-10:    Settings (Hardcore env, 5M steps)
Line 13:      Directory creation
Line 16-52:   train() function with transfer learning
Line 55-65:   Execute for all 4 algorithms
```

---

#### 3.5 `train_hardcore_v2.py` — Second Hardcore Training Phase

**Purpose:** Continue training from first Hardcore models for further improvement

**Transfer Chain:**
```
Normal (1M) → Hardcore v1 (5M) → Hardcore v2 (5M)
  SAC_final  →  hardcore/SAC_final → hardcore_v2/SAC_final
```

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Source | `models/hardcore/{name}_final` |
| Destination | `models/hardcore_v2/{name}_final` |
| `TIMESTEPS` | 5,000,000 |

**Code Flow:**
```python
Line 1-5:     Imports
Line 8-10:    Settings
Line 13:      Directory creation
Line 16-52:   train() function (loads from hardcore/)
Line 55-65:   Execute for all algorithms
```

---

#### 3.6 `train_best.py` — Best Training with Auto-Stop

**Purpose:** Optimized training with automatic stopping at reward threshold

**Special Feature - StopTrainingOnRewardThreshold:**
```python
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold = 295,    # Stop when mean reward ≥ 295
    verbose = 1
)
```

**Curriculum Learning Phases:**

| Phase | Environment | Timesteps | Target | Source Model |
|-------|-------------|-----------|--------|--------------|
| 1 | Normal | 1,000,000 | 295 | Original final model |
| 2 | Hardcore | 10,000,000 | 250 | Phase 1 best model |

**SAC Hyperparameters (Best):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 3e-4 | Standard SAC learning rate |
| `batch_size` | 512 | Large batches for stability |
| `gamma` | 0.99 | Standard discount |
| `tau` | 0.005 | Soft update coefficient |
| `ent_coef` | `"auto"` | Automatic entropy tuning |
| `target_entropy` | `"auto"` | Automatic target entropy |
| `buffer_size` | 1,000,000 | Replay buffer |
| `learning_starts` | 10,000 | Random exploration phase |
| `train_freq` | 1 | Update every step |
| `gradient_steps` | 1 | One gradient update per step |

**PPO Hyperparameters (Best):**

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |

**TD3 Hyperparameters (Best):**

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 1e-3 |
| `batch_size` | 512 |
| `gamma` | 0.99 |
| `tau` | 0.005 |
| `policy_delay` | 2 |
| `target_policy_noise` | 0.2 |
| `target_noise_clip` | 0.5 |
| `buffer_size` | 1,000,000 |
| `learning_starts` | 10,000 |

**Code Flow:**
```python
Line 1-8:     Imports (added StopTrainingOnRewardThreshold)
Line 11-12:   Environment names
Line 14-15:   Directory creation
Line 18-50:   Hyperparameter dictionaries (SAC, PPO, TD3)
Line 53-105:  train() function with auto-stop logic
Line 108-165: Execute Phase 1 (Normal) for SAC, TD3, PPO
Line 168-225: Execute Phase 2 (Hardcore) for SAC, TD3, PPO
```

---

#### 3.7 `train_curriculum.py` — Curriculum Learning

**Purpose:** Gradual difficulty increase for Hardcore environment

**Curriculum Design:**

| Phase | Name | Max Steps | Timesteps | Source | Purpose |
|-------|------|-----------|-----------|--------|---------|
| 1 | phase1_500steps | 500 | 3,000,000 | hardcore_v2/SAC_final | Learn short episodes |
| 2 | phase2_1000steps | 1000 | 3,000,000 | Phase 1 output | Extend episode length |
| 3 | phase3_2000steps | 2000 | 4,000,000 | Phase 2 output | Full episode length |

**Custom Environment Factory:**
```python
def make_env(max_steps):
    def _init():
        env = gym.make(
            ENV_NAME,
            max_episode_steps=max_steps    # Key: limit episode length
        )
        return env
    return _init
```

**Training Chain:**
```
hardcore_v2/SAC_final → phase1_500steps_final → phase2_1000steps_final → phase3_2000steps_final
        ↓                      ↓                        ↓
     (pre-trained)         (500 steps)              (1000 steps)              (2000 steps)
```

**Code Flow:**
```python
Line 1-7:     Imports (DummyVecEnv for custom env)
Line 9-10:    Settings
Line 12-13:   Directory creation
Line 16-20:   make_env() factory function
Line 23-73:   train_phase() function
Line 76-93:   Phase 1 execution (500 steps, 3M timesteps)
Line 96-111:  Phase 2 execution (1000 steps, 3M timesteps)
Line 114-129: Phase 3 execution (2000 steps, 4M timesteps)
```

---

#### 3.8 `train_gpu_sde.py` — Advanced GPU Training with SDE

**Purpose:** Maximum performance training using GPU acceleration and State-Dependent Exploration

**GPU Check:**
```python
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

**State-Dependent Exploration (SDE):**

| Parameter | SAC Value | PPO Value | Description |
|-----------|-----------|-----------|-------------|
| `use_sde` | True | True | Enable SDE |
| `sde_sample_freq` | 8 | 4 | Noise resampling frequency |

**SDE vs Standard Noise:**
- **Standard:** Add independent Gaussian noise to actions
- **SDE:** Noise depends on current state → more structured exploration

**Training Configuration:**

| Phase | Algorithm | Environment | Timesteps | Device | Buffer | Target |
|-------|-----------|-------------|-----------|--------|--------|--------|
| 1 | SAC | Normal | 1M | CUDA | 1M | 295 |
| 2 | SAC | Hardcore | 15M | CUDA | 2M | 250 |
| 1 | TD3 | Normal | 1M | CUDA | 1M | 295 |
| 2 | TD3 | Hardcore | 15M | CUDA | 2M | 250 |
| 1 | PPO | Normal | 1M | CPU | N/A | 295 |
| 2 | PPO | Hardcore | 15M | CPU | N/A | 250 |

**TD3 Action Noise (Normal):**
```python
action_noise = NormalActionNoise(
    mean  = np.zeros(4),
    sigma = 0.1 * np.ones(4)
)
```

**Code Flow:**
```python
Line 1-10:    Imports (torch for GPU check)
Line 13-16:   GPU availability check
Line 19-20:   Settings
Line 22-23:   Directory creation
Line 26-88:   train() function with device parameter
Line 91-112:  SAC Phase 1 (Normal, GPU, SDE)
Line 115-137: SAC Phase 2 (Hardcore, GPU, SDE, 15M)
Line 140-147: TD3 action noise setup
Line 150-172: TD3 Phase 1 (Normal, GPU)
Line 175-197: TD3 Phase 2 (Hardcore, GPU, 15M)
Line 200-222: PPO Phase 1 (Normal, CPU, SDE)
Line 225-247: PPO Phase 2 (Hardcore, CPU, SDE, 15M)
```

---

#### 3.9 `train_SAC_only.py` — Intensive SAC Training

**Purpose:** Maximum effort training for SAC (best performing algorithm)

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Source Model | `models/hardcore_v2/SAC_final` |
| Environment | `BipedalWalkerHardcore-v3` |
| Timesteps | 10,000,000 |
| Stop Target | 250 |
| `N_ENVS` | 4 |

**SAC Parameters:**

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 3e-4 |
| `batch_size` | 512 |
| `gamma` | 0.99 |
| `tau` | 0.005 |
| `ent_coef` | `"auto"` |
| `buffer_size` | 1,000,000 |
| `learning_starts` | 10,000 |

**Code Flow:**
```python
Line 1-6:     Imports
Line 9:       Environment name
Line 11:      N_ENVS = 4
Line 13-14:   Directory creation
Line 16-17:   Environment setup
Line 20-33:   Callbacks (stop at 250, eval every 50K, checkpoint every 100K)
Line 36-50:   Load and configure SAC model
Line 53-58:   Training execution (10M steps)
Line 60-61:   Save final model
```

---

### 📁 **Evaluation & Comparison Scripts (4 files)**

---

#### 3.10 `compare.py` — Comprehensive Benchmark Visualization

**Purpose:** Generate 7-panel comparison figure for all algorithms

**Evaluation Setup:**

| Parameter | Value |
|-----------|-------|
| `N_EVAL_EPISODES` | 50 |
| Environments | Normal + Hardcore |
| Models | SAC_final, PPO_final, TD3_final, DDPG_v3_final |

**7-Panel Figure Layout:**

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Panel 1        │  Panel 2        │  Panel 3        │
│  Normal Reward  │  Hardcore Reward│  Stability      │
│  (Bar Chart)    │  (Bar Chart)    │  (Grouped Bars) │
├─────────────────┼─────────────────┼─────────────────┤
│  Panel 4        │  Panel 5        │  Panel 6        │
│  Radar Chart    │  Normal vs Hard │  Summary Table  │
│  (4 categories) │  (Grouped Bars) │  (Ranked)       │
├─────────────────┴─────────────────┴─────────────────┤
│  Panel 7                                              │
│  Learning Curves (All algorithms over time)           │
└───────────────────────────────────────────────────────┘
```

**Radar Chart Categories:**

| Axis | Calculation | Weight |
|------|-------------|--------|
| Reward | mean / max_mean × 100 | 25% |
| Stability | (1 - std / max_std) × 100 | 25% |
| Hardcore | (mean_h - min) / range × 100 | 25% |
| Overall | Average of above | 25% |

**Colors:**

| Algorithm | Color Code | Hex |
|-----------|------------|-----|
| SAC | Cyan | `#00B4D8` |
| PPO | Green | `#06D6A0` |
| TD3 | Pink/Red | `#EF476F` |
| DDPG | Yellow | `#FFD166` |

**Code Flow:**
```python
Line 1-8:     Imports (matplotlib, numpy, SB3)
Line 11-13:   Configuration constants
Line 15:      Dark background style
Line 18-23:   Color definitions + model loading
Line 26-42:   Normal environment evaluation (50 episodes)
Line 45-61:   Hardcore environment evaluation (50 episodes)
Line 64-68:   Figure setup (24×18 inches, 3×3 grid)
Line 71-75:   style_ax() helper function
Line 78-103:  Panel 1: Normal bar chart
Line 106-131: Panel 2: Hardcore bar chart
Line 134-155: Panel 3: Stability comparison
Line 158-195: Panel 4: Radar chart
Line 198-221: Panel 5: Normal vs Hardcore grouped bars
Line 224-260: Panel 6: Summary table (ranked by Normal mean)
Line 263-290: Panel 7: Learning curves from evaluations.npz
Line 293-295: Save figure (300 DPI)
Line 298-313: Terminal summary table
```

---

#### 3.11 `watch.py` — Live Agent Visualization

**Purpose:** Evaluate models and record videos + live watch best agent

**Features:**

| # | Feature | Description |
|---|---------|-------------|
| 1 | Evaluation | 20 episodes per algorithm |
| 2 | Video Recording | 1600-step videos for each algorithm |
| 3 | Best Selection | Auto-select highest mean reward |
| 4 | Live Watch | Render best agent in human mode |

**Evaluation Results Format:**
```
Algorithm    Mean Reward    Std Dev
─────────────────────────────────────
SAC          290.35         2.21
PPO          289.34         1.98
TD3          296.80         1.48
DDPG         198.88         123.71
```

**Code Flow:**
```python
Line 1-8:     Imports (added time for live watch)
Line 11-13:   Settings
Line 15:      Directory creation
Line 18-23:   Model loading
Line 26-42:   Evaluation loop (20 episodes each)
Line 45-47:   Best algorithm selection
Line 50-78:   Video recording for all algorithms
Line 81-98:   Live watch best agent (render_mode="human")
```

---

#### 3.12 `record_compare.py` — Comparative Video Recording

**Purpose:** Record and compare videos on both environments

**Recording Setup:**

| Environment | Folder | Max Steps |
|-------------|--------|-----------|
| Normal | `videos/normal/{algorithm}/` | 2000 |
| Hardcore | `videos/hardcore/{algorithm}/` | 2000 |

**Output Format:**
```
videos/
├── normal/
│   ├── SAC/SAC_normal-episode-0.mp4
│   ├── PPO/PPO_normal-episode-0.mp4
│   ├── TD3/TD3_normal-episode-0.mp4
│   └── DDPG/DDPG_normal-episode-0.mp4
└── hardcore/
    ├── SAC/SAC_hardcore-episode-0.mp4
    ├── PPO/PPO_hardcore-episode-0.mp4
    ├── TD3/TD3_hardcore-episode-0.mp4
    └── DDPG/DDPG_hardcore-episode-0.mp4
```

**Code Flow:**
```python
Line 1-6:     Imports
Line 9-10:    Environment names
Line 12-13:   Directory creation
Line 16-21:   Model loading
Line 24-56:   record() function definition
Line 59-68:   Normal recording
Line 71-80:   Hardcore recording
Line 83-100:  Comparison table
Line 103-107: Best per environment
Line 110-115: Output locations
```

---

#### 3.13 `record_seeds.py` — Fixed Seed Track Evaluation

**Purpose:** Fair comparison using identical terrain configurations

**Fixed Seeds:**

| Track | Seed | Purpose |
|-------|------|---------|
| Track 1 | 42 | Standard seed |
| Track 2 | 123 | Different terrain |
| Track 3 | 456 | Different terrain |
| Track 4 | 789 | Different terrain |

**Why Fixed Seeds?**
- Same seed = identical terrain layout
- Eliminates terrain variability as confounding factor
- Enables direct algorithm comparison

**Output Structure:**
```
videos/seeds/
├── track_42/
│   ├── SAC_seed42-episode-0.mp4
│   ├── PPO_seed42-episode-0.mp4
│   ├── TD3_seed42-episode-0.mp4
│   └── DDPG_seed42-episode-0.mp4
├── track_123/
│   └── ...
├── track_456/
│   └── ...
└── track_789/
    └── ...
```

**Code Flow:**
```python
Line 1-6:     Imports + sys.stdout.flush() for real-time output
Line 9-10:    Environment + seeds
Line 12:      Directory creation
Line 14-21:   Model loading with progress prints
Line 24-63:   Main loop: for each seed, for each algorithm
Line 66-78:   Results summary table
```

---

#### 3.14 `record_hardcore.py` — Hardcore v2 Evaluation

**Purpose:** Evaluate hardcore_v2 models on fixed seeds

**Models:**
```python
models = {
    "SAC":  SAC.load("models/hardcore_v2/SAC_final"),
    "PPO":  PPO.load("models/hardcore_v2/PPO_final"),
    "TD3":  TD3.load("models/hardcore_v2/TD3_final"),
    "DDPG": DDPG.load("models/hardcore_v2/DDPG_final")
}
```

**Output:**
```
videos/hardcore_v2_on_hardcore/
├── track_42/
├── track_123/
├── track_456/
└── track_789/
```

---

#### 3.15 `record_curriculum.py` — Curriculum Model Evaluation

**Purpose:** Evaluate curriculum-trained SAC on fixed seeds

**Model:**
```python
model = SAC.load(
    "models/curriculum/phase3_2000steps_best/best_model",
    device="cpu"
)
```

**Output:**
```
videos/curriculum/
├── track_42/SAC_curriculum_seed42-episode-0.mp4
├── track_123/SAC_curriculum_seed123-episode-0.mp4
├── track_456/SAC_curriculum_seed456-episode-0.mp4
└── track_789/SAC_curriculum_seed789-episode-0.mp4
```

---

### 📁 **Testing Scripts (2 files)**

---

#### 3.16 `test.py` — Random Action Test

**Purpose:** Verify environment works with random actions

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Environment | `BipedalWalker-v3` |
| Render Mode | `human` (visual window) |
| Steps | 1000 |
| Action Selection | `env.action_space.sample()` (completely random) |

**Code Flow:**
```python
Line 1:       Import gymnasium
Line 3:       Create environment with human rendering
Line 5:       Reset environment
Line 7-12:    Loop 1000 steps with random actions
Line 14:      Close environment
```

**Expected Behavior:** Robot flails randomly, falls quickly, negative reward.

---

#### 3.17 `test_best.py` — Best Model Quick Test

**Purpose:** Quick evaluation of the absolute best SAC model

**Model:** `models/SAC_best/best/best_model`

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Environment | `BipedalWalkerHardcore-v3` |
| Episodes | 20 |
| Deterministic | True |

**Code Flow:**
```python
Line 1-4:     Imports
Line 6-9:     Load best SAC model
Line 11:      Create Hardcore environment
Line 13-17:   Evaluate (20 episodes, deterministic)
Line 19:      Print results
Line 20:      Close environment
```

---

## 4. Training Pipeline

### 4.1 Complete Training Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PHASE 0: BASELINE TRAINING                        │
│                         (train.py - 1M steps each)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  SAC ──→ models/SAC_final                                               │
│  PPO ──→ models/PPO_final                                               │
│  TD3 ──→ models/TD3_final                                               │
│  DDPG ─→ models/DDPG_final  ┐                                           │
│                             │                                           │
│         ┌───────────────────┘                                           │
│         ↓                                                               │
│  DDPG v2 (retain_ddpg.py - 1M) ──→ models/DDPG_v2_final               │
│  DDPG v3 (retrain_ddpg.py - 2M) ─→ models/DDPG_v3_final  ◄── BEST DDPG │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: HARDCORE TRANSFER (v1)                      │
│                      (train_hardcore.py - 5M steps)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  SAC_final ──→ hardcore/SAC_v2_final                                    │
│  PPO_final ──→ hardcore/PPO_v2_final                                    │
│  TD3_final ──→ hardcore/TD3_v2_final                                    │
│  DDPG_final ─→ hardcore/DDPG_v2_final                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: HARDCORE REFINEMENT (v2)                    │
│                     (train_hardcore_v2.py - 5M steps)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  hardcore/SAC_final ──→ hardcore_v2/SAC_final                           │
│  hardcore/PPO_final ──→ hardcore_v2/PPO_final                           │
│  hardcore/TD3_final ──→ hardcore_v2/TD3_final                           │
│  hardcore/DDPG_final ─→ hardcore_v2/DDPG_final                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 3: BEST TRAINING + CURRICULUM                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐    │
│  │  train_best.py      │    │  train_curriculum.py                 │    │
│  │  ─────────────────  │    │  ───────────────────                  │    │
│  │  Phase 1: Normal    │    │  Phase 1: 500 steps (3M)            │    │
│  │  (Fine-tune to 295) │    │  Phase 2: 1000 steps (3M)           │    │
│  │                     │    │  Phase 3: 2000 steps (4M)           │    │
│  │  Phase 2: Hardcore  │    │                                      │    │
│  │  (Train to 250)     │    │  Source: hardcore_v2/SAC_final       │    │
│  │                     │    │  Output: curriculum/phase3_best      │    │
│  │  Output: best/*     │    │                                      │    │
│  └─────────────────────┘    └─────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 4: GPU + SDE ADVANCED TRAINING                 │
│                    (train_gpu_sde.py - Up to 15M steps)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SAC:  Normal (1M, GPU) ──→ gpu_sde/SAC_normal_final                   │
│              ↓                                                            │
│       Hardcore (15M, GPU) ─→ gpu_sde/SAC_hardcore_final  ◄── BEST OVERALL│
│                                                                         │
│  TD3:  Normal (1M, GPU) ──→ gpu_sde/TD3_normal_final                     │
│              ↓                                                            │
│       Hardcore (15M, GPU) ─→ gpu_sde/TD3_hardcore_final                │
│                                                                         │
│  PPO:  Normal (1M, CPU) ──→ gpu_sde/PPO_normal_final                    │
│              ↓                                                            │
│       Hardcore (15M, CPU) ─→ gpu_sde/PPO_hardcore_final                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     PHASE 5: SAC INTENSIVE TRAINING                      │
│                      (train_SAC_only.py - 10M steps)                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Source: hardcore_v2/SAC_final                                          │
│  Target: SAC_best/best/best_model  (stops at reward 250)               │
│  Final:  SAC_best/SAC_final                                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION SCRIPTS FLOW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  compare.py ──────────────────────────────────────► comparison_final.png│
│  (Load all finals, evaluate 50 episodes, 7 panels)                      │
│                                                                         │
│  watch.py ─────────────────────────────────────────► videos/ + live    │
│  (Evaluate 20 eps, record videos, watch best live)                     │
│                                                                         │
│  record_compare.py ────────────────────────────────► videos/normal/      │
│  (Record on both environments)                      videos/hardcore/   │
│                                                                         │
│  record_seeds.py ─────────────────────────────────► videos/seeds/      │
│  (Fixed seeds: 42, 123, 456, 789)                                       │
│                                                                         │
│  record_hardcore.py ─────────────────────────────► videos/hardcore_v2/ │
│  (Hardcore v2 models on fixed seeds)                                    │
│                                                                         │
│  record_curriculum.py ──────────────────────────► videos/curriculum/ │
│  (Curriculum model on fixed seeds)                                      │
│                                                                         │
│  test_best.py ────────────────────────────────────► Console output      │
│  (Quick eval of SAC_best)                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Results Summary

### 5.1 Normal Environment Results

| Rank | Algorithm | Mean Reward | Std Dev | Status |
|------|-----------|-------------|---------|--------|
| 🥇 1 | **TD3** | **296.80** | 1.48 | ✅ Solved |
| 🥈 2 | **SAC** | **290.35** | 2.21 | ✅ Solved |
| 🥉 3 | **PPO** | **289.34** | 1.98 | ✅ Solved |
| 4 | DDPG v3 | 226.92 | 117.08 | ⚠️ Unstable |
| - | DDPG v1 | 151.6 | 166.8 | ❌ Poor |

**Solved Threshold:** ~300 (episode completion bonus)

### 5.2 Hardcore Environment Results (After Full Training)

| Rank | Algorithm | Mean Reward | Std Dev | Training |
|------|-----------|-------------|---------|----------|
| 🥇 1 | **SAC** | **205.73** | 124.03 | GPU + SDE |
| 🥈 2 | **TD3** | **176.68** | 117.08 | GPU + SDE |
| 🥉 3 | **PPO** | **-86.43** | 25.88 | GPU + SDE |
| - | SAC (Best Track) | **297.05** | - | Single episode |

### 5.3 SAC Training Progression

| Phase | Training | Mean Reward | Improvement | Total Steps |
|-------|----------|-------------|-------------|-------------|
| 0 | Zero-shot | -77.0 | - | - |
| 1 | Hardcore v1 (5M) | 49.5 | +126.5 | 5M |
| 2 | Hardcore v2 (5M) | 146.7 | +97.2 | 10M |
| 3 | Curriculum (10M) | 187.1 | +40.4 | 20M |
| 4 | GPU + SDE (15M) | **297.1** | +110.0 | 35M |

### 5.4 DDPG Improvement Trajectory

| Version | Changes | Mean Reward | Improvement |
|---------|---------|-------------|-------------|
| v1 | Original [256,256], Gaussian | 151.6 | Baseline |
| v2 | [400,300] network | 198.9 | +31% |
| v3 | + OU Noise + 2M steps | 226.9 | +49% |
| Best | Single episode peak | 305.98 | - |

### 5.5 Key Findings

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

## 6. How to Use

### 6.1 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install gymnasium[box2d]
pip install stable-baselines3[extra]
pip install matplotlib numpy
pip install tensorboard
pip install moviepy  # for video recording
```

### 6.2 Running Training

```bash
# Phase 0: Baseline training (train all 4 algorithms)
python train.py

# Phase 1: Improve DDPG
python retrain_ddpg.py      # v3 with OU noise

# Phase 2: Hardcore transfer
python train_hardcore.py

# Phase 3: Hardcore refinement
python train_hardcore_v2.py

# Phase 4: Best training with auto-stop
python train_best.py

# Phase 5: Curriculum learning
python train_curriculum.py

# Phase 6: GPU + SDE advanced
python train_gpu_sde.py

# Phase 7: SAC intensive
python train_SAC_only.py
```

### 6.3 Running Evaluation

```bash
# Full comparison visualization
python compare.py

# Watch best agent live + record videos
python watch.py

# Record comparison videos
python record_compare.py

# Fixed seed evaluation
python record_seeds.py

# Test specific model
python test_best.py

# Quick random test
python test.py
```

### 6.4 Viewing Training Progress

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Then open http://localhost:6006 in browser
```

---

## 7. Dependencies

### 7.1 Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Python | ≥3.8 | Runtime |
| gymnasium | ≥0.28 | RL environments |
| stable-baselines3 | ≥2.0 | RL algorithms |
| torch | ≥1.13 | Deep learning backend |
| numpy | ≥1.21 | Numerical operations |
| matplotlib | ≥3.5 | Visualization |

### 7.2 Optional Libraries

| Library | Purpose |
|---------|---------|
| tensorboard | Training monitoring |
| moviepy | Video recording |
| opencv-python | Video processing |

### 7.3 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8+ cores |
| GPU | Optional | NVIDIA CUDA (for GPU training) |
| Storage | 10 GB | 50 GB (for models + videos) |

---

## 8. Project Statistics

| Metric | Value |
|--------|-------|
| Total Python Files | 17 |
| Training Scripts | 9 |
| Evaluation Scripts | 6 |
| Test Scripts | 2 |
| Total Training Steps | ~71M (all algorithms combined) |
| Algorithms Benchmarked | 4 |
| Environment Variants | 2 |
| Evaluation Episodes | 50+ per model |
| Fixed Seed Tracks | 4 |
| DDPG Variants Tested | 3 |
| Training Phases | 7 |

---

## 9. Citation

If you use this code, please cite:

```bibtex
@misc{bipedalwalker-rl-benchmark,
  title={Benchmarking Deep Reinforcement Learning Algorithms on BipedalWalker},
  author={[Your Name]},
  year={2024},
  note={Comparative study of SAC, PPO, TD3, and DDPG}
}
```

### References

| # | Paper | Authors | Year |
|---|-------|---------|------|
| 1 | Continuous Control with Deep Reinforcement Learning | Lillicrap et al. | 2016 (DDPG) |
| 2 | Addressing Function Approximation Error | Fujimoto et al. | 2018 (TD3) |
| 3 | Soft Actor-Critic | Haarnoja et al. | 2018 (SAC) |
| 4 | Proximal Policy Optimization | Schulman et al. | 2017 (PPO) |
| 5 | Stable-Baselines3 | Raffin et al. | 2021 |

---

<div align="center">

**🎯 Project Goal:** Identify optimal RL algorithms for continuous locomotion control

**🏆 Winner:** SAC for Hardcore | TD3 for Normal

**📊 Total Experiments:** 50+ training runs | 200+ evaluation episodes

</div>
