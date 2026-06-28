import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold
)
import os

ENV_HARDCORE = "BipedalWalkerHardcore-v3"
N_ENVS       = 4

os.makedirs("models/SAC_best", exist_ok=True)
os.makedirs("logs/SAC_best",   exist_ok=True)

env      = make_vec_env(ENV_HARDCORE, n_envs=N_ENVS)
eval_env = make_vec_env(ENV_HARDCORE, n_envs=1)

# Stop when reward reaches 250
stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold = 250,
    verbose          = 1
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best = stop_callback,
    best_model_save_path = "models/SAC_best/best/",
    log_path             = "logs/SAC_best/",
    eval_freq            = 50_000,
    n_eval_episodes      = 20,
    verbose              = 1
)

checkpoint = CheckpointCallback(
    save_freq   = 100_000,
    save_path   = "models/SAC_best/checkpoints/",
    name_prefix = "SAC_best"
)

# Load best SAC
print("Loading SAC from hardcore_v2...", flush=True)
model = SAC.load(
    "models/hardcore_v2/SAC_final",
    env     = env,
    device  = "cpu",
    verbose = 1,
    learning_rate   = 3e-4,
    batch_size      = 512,
    gamma           = 0.99,
    tau             = 0.005,
    ent_coef        = "auto",
    buffer_size     = 1_000_000,
    learning_starts = 10_000,
    tensorboard_log = "logs/SAC_best/"
)
print("Model loaded!", flush=True)

print("Training SAC on Hardcore...", flush=True)
model.learn(
    total_timesteps     = 10_000_000,
    callback            = [checkpoint, eval_callback],
    reset_num_timesteps = False
)

model.save("models/SAC_best/SAC_final")
print("Done!", flush=True)

env.close()
eval_env.close()