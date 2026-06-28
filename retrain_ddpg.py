import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os


# Settings
ENV_NAME  = "BipedalWalker-v3"
TIMESTEPS = 2_000_000 
N_ENVS    = 4

os.makedirs("models",  exist_ok=True)
os.makedirs("logs",    exist_ok=True)

# Environments
env      = make_vec_env(ENV_NAME, n_envs=N_ENVS)
eval_env = make_vec_env(ENV_NAME, n_envs=1)

n_actions = env.action_space.shape[-1]

ou_noise = OrnsteinUhlenbeckActionNoise(
    mean   = np.zeros(n_actions),
    sigma  = 0.2 * np.ones(n_actions),
    theta  = 0.15,
    dt     = 1e-2
)

normal_noise = NormalActionNoise(
    mean  = np.zeros(n_actions),
    sigma = 0.1 * np.ones(n_actions)
)

# Callbacks
checkpoint = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = "models/DDPG_v3/",
    name_prefix = "DDPG_v3"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = "models/DDPG_v3_best/",
    log_path             = "logs/DDPG_v3/",
    eval_freq            = 50_000,
    n_eval_episodes      = 20,
    verbose              = 1
)


# DDPG v3
model = DDPG(
    "MlpPolicy",
    env,

    # Learning
    learning_rate   = 0.0001,    
    batch_size      = 512,         
    gamma           = 0.99,      
    tau             = 0.001,     

    #Memory
    buffer_size     = 1_000_000, 
    learning_starts = 10_000,    

    # Network 
    policy_kwargs   = dict(
        net_arch = [400, 300]      # network 
    ),

    # Noise
    action_noise    = ou_noise,    # OrnsteinUhlenbeck

    verbose         = 1,
    device          = "cpu",
    tensorboard_log = "logs/DDPG_v3/"
)

# Training
print("\n" + "="*40)
print("Retraining DDPG v3...")
print("All improvements applied!")
print("="*40 + "\n")

model.learn(
    total_timesteps = TIMESTEPS,
    callback        = [checkpoint, eval_callback]
)
# Save
model.save("models/DDPG_v3_final")
print("\nDDPG v3 Done! Model Saved!")

env.close()
eval_env.close()

# Quick Evaluation
print("\nEvaluating DDPG v3...")
from stable_baselines3.common.evaluation import evaluate_policy

model    = DDPG.load("models/DDPG_v3_final", device="cpu")
eval_env = Monitor(gym.make(ENV_NAME))

mean, std = evaluate_policy(
    model, eval_env,
    n_eval_episodes = 20,
    deterministic   = True
)

print(f"\nDDPG v3 Results:")
print(f"Mean Reward: {mean:.2f}")
print(f"Std:         {std:.2f}")

eval_env.close()