import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold
)
from stable_baselines3.common.noise import NormalActionNoise
import torch
import numpy as np
import os

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}",
      flush=True)
print(f"GPU Name: {torch.cuda.get_device_name(0)}",
      flush=True)

ENV_NORMAL   = "BipedalWalker-v3"
ENV_HARDCORE = "BipedalWalkerHardcore-v3"
N_ENVS       = 4

os.makedirs("models/gpu_sde", exist_ok=True)
os.makedirs("logs/gpu_sde",   exist_ok=True)

# Train Function
def train(algorithm, name, env_name,
          model_params, timesteps,
          device, stop_reward=None,
          pretrained_path=None):

    print(f"\n{'='*50}", flush=True)
    print(f"Training {name}...", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Environment: {env_name}", flush=True)
    print(f"Timesteps: {timesteps:,}", flush=True)
    print(f"{'='*50}\n", flush=True)

    env      = make_vec_env(env_name, n_envs=N_ENVS)
    eval_env = make_vec_env(env_name, n_envs=1)

    callbacks = []

    callbacks.append(CheckpointCallback(
        save_freq   = 100_000,
        save_path   = f"models/gpu_sde/{name}/",
        name_prefix = f"{name}"
    ))

    if stop_reward:
        stop_cb = StopTrainingOnRewardThreshold(
            reward_threshold = stop_reward,
            verbose          = 1
        )
        eval_cb = EvalCallback(
            eval_env,
            callback_on_new_best = stop_cb,
            best_model_save_path =
                f"models/gpu_sde/{name}_best/",
            log_path        = f"logs/gpu_sde/{name}/",
            eval_freq       = 25_000,
            n_eval_episodes = 30,
            verbose         = 1
        )
    else:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path =
                f"models/gpu_sde/{name}_best/",
            log_path        = f"logs/gpu_sde/{name}/",
            eval_freq       = 25_000,
            n_eval_episodes = 30,
            verbose         = 1
        )
    callbacks.append(eval_cb)

    if pretrained_path:
        print(f"Loading from {pretrained_path}...",
              flush=True)
        model = algorithm.load(
            pretrained_path,
            env    = env,
            device = device,
            verbose = 1,
            tensorboard_log = f"logs/gpu_sde/{name}/",
            **model_params
        )
        print("Pretrained model loaded!", flush=True)
    else:
        print("Training from scratch...", flush=True)
        model = algorithm(
            "MlpPolicy",
            env,
            verbose = 1,
            device  = device,
            tensorboard_log = f"logs/gpu_sde/{name}/",
            **model_params
        )

    model.learn(
        total_timesteps     = timesteps,
        callback            = callbacks,
        reset_num_timesteps = True
    )

    save_path = f"models/gpu_sde/{name}_final"
    model.save(save_path)
    print(f"\n{name} Done!", flush=True)

    env.close()
    eval_env.close()
    return save_path

# Phase 1: Train SAC on Normal
sac_normal_path = train(
    SAC, "SAC_normal",
    env_name  = ENV_NORMAL,
    model_params = {
        "learning_rate"   : 3e-4,
        "batch_size"      : 256,
        "gamma"           : 0.99,
        "tau"             : 0.005,
        "ent_coef"        : "auto",
        "target_entropy"  : "auto",
        "buffer_size"     : 1_000_000,
        "learning_starts" : 10_000,
        "use_sde"         : True,      
        "sde_sample_freq" : 8,
    },
    timesteps   = 1_000_000,
    device      = "cuda",
    stop_reward = 295
)

# Phase 2: Train SAC on Hardcore
sac_hardcore_path = train(
    SAC, "SAC_hardcore",
    env_name  = ENV_HARDCORE,
    model_params = {
        "learning_rate"   : 3e-4,
        "batch_size"      : 512,
        "gamma"           : 0.99,
        "tau"             : 0.005,
        "ent_coef"        : "auto",
        "target_entropy"  : "auto",
        "buffer_size"     : 2_000_000,
        "learning_starts" : 10_000,
        "use_sde"         : True,      
        "sde_sample_freq" : 8,
    },
    timesteps      = 15_000_000,
    device         = "cuda",
    stop_reward    = 250,
    pretrained_path= sac_normal_path
)

# Phase 1: Train TD3 on Normal
n_actions = gym.make(ENV_NORMAL)\
    .action_space.shape[-1]

action_noise = NormalActionNoise(
    mean  = np.zeros(n_actions),
    sigma = 0.1 * np.ones(n_actions)
)

td3_normal_path = train(
    TD3, "TD3_normal",
    env_name  = ENV_NORMAL,
    model_params = {
        "learning_rate"       : 1e-4,
        "batch_size"          : 256,
        "gamma"               : 0.99,
        "tau"                 : 0.005,
        "policy_delay"        : 2,
        "target_policy_noise" : 0.2,
        "target_noise_clip"   : 0.5,
        "buffer_size"         : 1_000_000,
        "learning_starts"     : 10_000,
        "action_noise"        : action_noise,
    },
    timesteps   = 1_000_000,
    device      = "cuda",
    stop_reward = 295
)

# Phase 2: Train TD3 on Hardcore
td3_hardcore_path = train(
    TD3, "TD3_hardcore",
    env_name  = ENV_HARDCORE,
    model_params = {
        "learning_rate"       : 1e-4,
        "batch_size"          : 512,
        "gamma"               : 0.99,
        "tau"                 : 0.005,
        "policy_delay"        : 2,
        "target_policy_noise" : 0.2,
        "target_noise_clip"   : 0.5,
        "buffer_size"         : 2_000_000,
        "learning_starts"     : 10_000,
    },
    timesteps      = 15_000_000,
    device         = "cuda",
    stop_reward    = 250,
    pretrained_path= td3_normal_path
)

# PPO on CPU
ppo_normal_path = train(
    PPO, "PPO_normal",
    env_name  = ENV_NORMAL,
    model_params = {
        "learning_rate" : 3e-4,
        "n_steps"       : 2048,
        "batch_size"    : 64,
        "n_epochs"      : 10,
        "gamma"         : 0.99,
        "gae_lambda"    : 0.95,
        "clip_range"    : 0.2,
        "ent_coef"      : 0.01,
        "use_sde"       : True,    
        "sde_sample_freq": 4,
    },
    timesteps   = 1_000_000,
    device      = "cpu",
    stop_reward = 295
)

ppo_hardcore_path = train(
    PPO, "PPO_hardcore",
    env_name  = ENV_HARDCORE,
    model_params = {
        "learning_rate" : 3e-4,
        "n_steps"       : 2048,
        "batch_size"    : 64,
        "n_epochs"      : 10,
        "gamma"         : 0.99,
        "gae_lambda"    : 0.95,
        "clip_range"    : 0.2,
        "ent_coef"      : 0.01,
        "use_sde"       : True,    
        "sde_sample_freq": 4,
    },
    timesteps      = 15_000_000,
    device         = "cpu",
    stop_reward    = 250,
    pretrained_path= ppo_normal_path
)

print("\n" + "="*50, flush=True)
print("All SDE Training Done!", flush=True)
print("="*50, flush=True)