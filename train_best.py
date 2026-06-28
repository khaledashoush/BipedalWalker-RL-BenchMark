import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold
)
import os

# Settings
ENV_NORMAL   = "BipedalWalker-v3"
ENV_HARDCORE = "BipedalWalkerHardcore-v3"
N_ENVS       = 4

os.makedirs("models/best", exist_ok=True)
os.makedirs("logs/best",   exist_ok=True)

# Best Hyperparameters
SAC_PARAMS = {
    "learning_rate"  : 3e-4,
    "batch_size"     : 512,
    "gamma"          : 0.99,
    "tau"            : 0.005,
    "ent_coef"       : "auto",
    "target_entropy" : "auto",
    "buffer_size"    : 1_000_000,
    "learning_starts": 10_000,
    "train_freq"     : 1,
    "gradient_steps" : 1
}

PPO_PARAMS = {
    "learning_rate" : 3e-4,
    "n_steps"       : 2048,
    "batch_size"    : 64,
    "n_epochs"      : 10,
    "gamma"         : 0.99,
    "gae_lambda"    : 0.95,
    "clip_range"    : 0.2,
    "ent_coef"      : 0.01
}

TD3_PARAMS = {
    "learning_rate"       : 1e-3,
    "batch_size"          : 512,
    "gamma"               : 0.99,
    "tau"                 : 0.005,
    "policy_delay"        : 2,
    "target_policy_noise" : 0.2,
    "target_noise_clip"   : 0.5,
    "buffer_size"         : 1_000_000,
    "learning_starts"     : 10_000
}
# Train Function
def train(algorithm, name, params,
          pretrained_path, env_name,
          timesteps, stop_reward=None):

    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print(f"Environment: {env_name}")
    print(f"Timesteps: {timesteps:,}")
    print(f"{'='*50}\n")

    env      = make_vec_env(env_name, n_envs=N_ENVS)
    eval_env = make_vec_env(env_name, n_envs=1)

    callbacks = []

    # Checkpoint
    callbacks.append(CheckpointCallback(
        save_freq   = 100_000,
        save_path   = f"models/best/{name}/",
        name_prefix = f"{name}_best"
    ))

    # Stop on reward threshold
    if stop_reward:
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold = stop_reward,
            verbose          = 1
        )
        eval_cb = EvalCallback(
            eval_env,
            callback_on_new_best  = stop_callback,
            best_model_save_path  =
                f"models/best/{name}_best/",
            log_path    = f"logs/best/{name}/",
            eval_freq   = 50_000,
            n_eval_episodes = 20,
            verbose     = 1
        )
    else:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path =
                f"models/best/{name}_best/",
            log_path    = f"logs/best/{name}/",
            eval_freq   = 50_000,
            n_eval_episodes = 20,
            verbose     = 1
        )

    callbacks.append(eval_cb)

    # Load pretrained
    print(f"Loading from {pretrained_path}...")
    model = algorithm.load(
        pretrained_path,
        env     = env,
        device  = "cpu",
        verbose = 1,
        tensorboard_log = f"logs/best/{name}/",
        **params
    )

    model.learn(
        total_timesteps     = timesteps,
        callback            = callbacks,
        reset_num_timesteps = False
    )

    model.save(f"models/best/{name}_final")
    print(f"\n{name} Done!")

    env.close()
    eval_env.close()

    return model

# Curriculum Learning
print("\n" + "="*50)
print("Phase 1: Fine-tune SAC on Normal")
print("="*50)
train(
    SAC, "SAC",
    SAC_PARAMS,
    pretrained_path = "models/SAC_final",
    env_name        = ENV_NORMAL,
    timesteps       = 1_000_000,
    stop_reward     = 295  
)

print("\n" + "="*50)
print("Phase 2: SAC on Hardcore")
print("="*50)
train(
    SAC, "SAC_hardcore",
    SAC_PARAMS,
    pretrained_path = "models/best/SAC_final",
    env_name        = ENV_HARDCORE,
    timesteps       = 10_000_000,
    stop_reward     = 250  
)

#TD3
print("\n" + "="*50)
print("Phase 1: Fine-tune TD3 on Normal")
print("="*50)
train(
    TD3, "TD3",
    TD3_PARAMS,
    pretrained_path = "models/TD3_final",
    env_name        = ENV_NORMAL,
    timesteps       = 1_000_000,
    stop_reward     = 295
)

print("\n" + "="*50)
print("Phase 2: TD3 on Hardcore")
print("="*50)
train(
    TD3, "TD3_hardcore",
    TD3_PARAMS,
    pretrained_path = "models/best/TD3_final",
    env_name        = ENV_HARDCORE,
    timesteps       = 10_000_000,
    stop_reward     = 250
)

# PPO
print("\n" + "="*50)
print("Phase 1: Fine-tune PPO on Normal")
print("="*50)
train(
    PPO, "PPO",
    PPO_PARAMS,
    pretrained_path = "models/PPO_final",
    env_name        = ENV_NORMAL,
    timesteps       = 1_000_000,
    stop_reward     = 295
)

print("\n" + "="*50)
print("Phase 2: PPO on Hardcore")
print("="*50)
train(
    PPO, "PPO_hardcore",
    PPO_PARAMS,
    pretrained_path = "models/best/PPO_final",
    env_name        = ENV_HARDCORE,
    timesteps       = 10_000_000,
    stop_reward     = 250
)

print("\n" + "="*50)
print("All Training Done!")
print("="*50)