import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

# ==============================
# Settings
# ==============================
TIMESTEPS = 1_000_000
ENV_NAME = "BipedalWalker-v3"
N_ENVS = 4

# ==============================
# Create Folders
# ==============================
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# ==============================
# Train Function
# ==============================
def train(algorithm, name, hyperparams={}):
    print(f"\n{'='*40}")
    print(f"Training {name}...")
    print(f"{'='*40}\n")

    env = make_vec_env(ENV_NAME, n_envs=N_ENVS)
    eval_env = make_vec_env(ENV_NAME, n_envs=1)

    checkpoint = CheckpointCallback(
        save_freq=100_000,
        save_path=f"models/{name}/",
        name_prefix=name
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{name}_best/",
        log_path=f"logs/{name}/",
        eval_freq=50_000,
        n_eval_episodes=20,  # ✅ زودنا لـ 20
        verbose=1
    )

    model = algorithm(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",  # ✅ ضفنا device
        tensorboard_log=f"logs/{name}/",
        **hyperparams  # ✅ ضفنا hyperparams
    )

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[checkpoint, eval_callback]
    )

    model.save(f"models/{name}_final")
    print(f"\n{name} Done! Model Saved!")

    env.close()
    eval_env.close()

    return model

# ==============================
# Hyperparameters المحسّنة
# ==============================

# ✅ SAC hyperparams
sac_params = {
    "learning_rate": 0.0003,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
}

# ✅ PPO hyperparams
ppo_params = {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}

# ✅ TD3 hyperparams المحسّنة
td3_params = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
}

# ✅ DDPG hyperparams المحسّنة
ddpg_params = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
}

# ==============================
# Train All
# ==============================
sac_model  = train(SAC,  "SAC",  sac_params)
ppo_model  = train(PPO,  "PPO",  ppo_params)
td3_model  = train(TD3,  "TD3",  td3_params)
ddpg_model = train(DDPG, "DDPG", ddpg_params)

print("\n" + "="*40)
print("All Training Done!")
print("="*40)