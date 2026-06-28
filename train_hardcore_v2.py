import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback
)
import os

ENV_NAME  = "BipedalWalkerHardcore-v3"
TIMESTEPS = 5_000_000
N_ENVS    = 4

os.makedirs("models/hardcore_v2", exist_ok=True)
os.makedirs("logs/hardcore_v2",   exist_ok=True)

def train(algorithm, name, pretrained_path):
    print(f"\n{'='*40}")
    print(f"Training {name} Hardcore v2...")
    print(f"{'='*40}\n")

    env      = make_vec_env(ENV_NAME, n_envs=N_ENVS)
    eval_env = make_vec_env(ENV_NAME, n_envs=1)

    checkpoint = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = f"models/hardcore_v2/{name}/",
        name_prefix = f"{name}_v2"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path =
            f"models/hardcore_v2/{name}_best/",
        log_path    = f"logs/hardcore_v2/{name}/",
        eval_freq   = 50_000,
        n_eval_episodes = 20,
        verbose     = 1
    )

    print(f"Loading {name} from hardcore training...")
    model = algorithm.load(
        pretrained_path,
        env     = env,
        device  = "cpu",
        verbose = 1,
        tensorboard_log =
            f"logs/hardcore_v2/{name}/"
    )
    print("Model loaded!")

    model.learn(
        total_timesteps     = TIMESTEPS,
        callback            = [checkpoint,
                               eval_callback],
        reset_num_timesteps = False
    )

    model.save(f"models/hardcore_v2/{name}_final")
    print(f"\n{name} v2 Done!")

    env.close()
    eval_env.close()

# Train from Hardcore models
train(SAC,  "SAC",
      "models/hardcore/SAC_final")

train(PPO,  "PPO",
      "models/hardcore/PPO_final")

train(TD3,  "TD3",
      "models/hardcore/TD3_final")

train(DDPG, "DDPG",
      "models/hardcore/DDPG_final")

print("\n" + "="*40)
print("All Hardcore v2 Training Done!")
print("="*40)