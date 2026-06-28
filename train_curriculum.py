import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback
)
import os

ENV_NAME = "BipedalWalkerHardcore-v3"
N_ENVS   = 4

os.makedirs("models/curriculum", exist_ok=True)
os.makedirs("logs/curriculum",   exist_ok=True)

# Custom Environment
def make_env(max_steps):
    def _init():
        env = gym.make(
            ENV_NAME,
            max_episode_steps=max_steps
        )
        return env
    return _init

def train_phase(name, pretrained_path,
                max_steps, timesteps):
    print(f"\n{'='*50}", flush=True)
    print(f"Phase: {name}", flush=True)
    print(f"Max Steps: {max_steps}", flush=True)
    print(f"Timesteps: {timesteps:,}", flush=True)
    print(f"{'='*50}\n", flush=True)

    from stable_baselines3.common.vec_env import (
        DummyVecEnv
    )

    env = DummyVecEnv([
        make_env(max_steps)
        for _ in range(N_ENVS)
    ])

    eval_env = DummyVecEnv([
        make_env(max_steps)
    ])

    checkpoint = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = f"models/curriculum/{name}/",
        name_prefix = f"{name}"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path =
            f"models/curriculum/{name}_best/",
        log_path    = f"logs/curriculum/{name}/",
        eval_freq   = 50_000,
        n_eval_episodes = 20,
        verbose     = 1
    )

    print(f"Loading from {pretrained_path}...",
          flush=True)
    model = SAC.load(
        pretrained_path,
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
        tensorboard_log =
            f"logs/curriculum/{name}/"
    )
    print("Model loaded!", flush=True)

    model.learn(
        total_timesteps     = timesteps,
        callback            = [checkpoint,
                               eval_callback],
        reset_num_timesteps = False
    )

    save_path = f"models/curriculum/{name}_final"
    model.save(save_path)
    print(f"\n{name} Done!", flush=True)
    print(f"Saved to {save_path}", flush=True)

    env.close()
    eval_env.close()

    return save_path

# Curriculum Training
phase1_path = train_phase(
    name           = "phase1_500steps",
    pretrained_path= "models/hardcore_v2/SAC_final",
    max_steps      = 500,
    timesteps      = 3_000_000
)

# Phase 2: (1000 steps)
phase2_path = train_phase(
    name           = "phase2_1000steps",
    pretrained_path= phase1_path,
    max_steps      = 1000,
    timesteps      = 3_000_000
)

# Phase 3: (2000 steps)
phase3_path = train_phase(
    name           = "phase3_2000steps",
    pretrained_path= phase2_path,
    max_steps      = 2000,
    timesteps      = 4_000_000
)

print("\n" + "="*50, flush=True)
print("Curriculum Training Done!", flush=True)
print("="*50, flush=True)