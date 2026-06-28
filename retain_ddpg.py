import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import os

# Settings
ENV_NAME  = "BipedalWalker-v3"
TIMESTEPS = 1_000_000
N_ENVS    = 4

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

# Environments
env      = make_vec_env(ENV_NAME, n_envs=N_ENVS)
eval_env = make_vec_env(ENV_NAME, n_envs=1)

n_actions   = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean  = np.zeros(n_actions),
    sigma = 0.1 * np.ones(n_actions)
)

# Callbacks
checkpoint = CheckpointCallback(
    save_freq  = 100_000,
    save_path  = "models/DDPG_v2/",
    name_prefix= "DDPG_v2"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = "models/DDPG_v2_best/",
    log_path             = "logs/DDPG_v2/",
    eval_freq            = 50_000,
    n_eval_episodes      = 20,
    verbose              = 1
)

# DDPG Model 
model = DDPG(
    "MlpPolicy",
    env,
    learning_rate    = 0.0001,    
    batch_size       = 512,       
    gamma            = 0.98,      
    tau              = 0.001,     
    buffer_size      = 1_000_000, 
    learning_starts  = 10_000,    
    action_noise     = action_noise, 
    verbose          = 1,
    device           = "cpu",
    tensorboard_log  = "logs/DDPG_v2/"
)
# Training
print("\n" + "="*40)
print("Retraining DDPG v2...")
print("="*40 + "\n")

model.learn(
    total_timesteps = TIMESTEPS,
    callback        = [checkpoint, eval_callback]
)

# Save
model.save("models/DDPG_v2_final")
print("\nDDPG v2 Done! Model Saved!")

env.close()
eval_env.close()