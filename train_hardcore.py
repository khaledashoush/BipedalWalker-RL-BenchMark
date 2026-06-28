import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

ENV_NAME  = "BipedalWalkerHardcore-v3"
TIMESTEPS = 5_000_000
N_ENVS    = 4

os.makedirs("models/hardcore", exist_ok=True)
os.makedirs("logs/hardcore",   exist_ok=True)

def train(algorithm, name, pretrained_path=None):
    print(f"\n{'='*40}")
    print(f"Training {name} on Hardcore...")
    print(f"{'='*40}\n")

    env      = make_vec_env(ENV_NAME, n_envs=N_ENVS)
    eval_env = make_vec_env(ENV_NAME, n_envs=1)

    checkpoint = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = f"models/hardcore/{name}/",
        name_prefix = f"{name}_hardcore"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = f"models/hardcore/{name}_best/",
        log_path             = f"logs/hardcore/{name}/",
        eval_freq            = 50_000,
        n_eval_episodes      = 20,
        verbose              = 1
    )

    if pretrained_path:
        print(f"Loading pretrained {name}...")
        model = algorithm.load(
            pretrained_path,
            env     = env,
            device  = "cpu",
            verbose = 1,
            tensorboard_log = f"logs/hardcore/{name}/"
        )
        print("Pretrained model loaded!")
    else:
        model = algorithm(
            "MlpPolicy",
            env,
            device  = "cpu",
            verbose = 1,
            tensorboard_log = f"logs/hardcore/{name}/"
        )

    model.learn(
        total_timesteps  = TIMESTEPS,
        callback         = [checkpoint, eval_callback],
        reset_num_timesteps = False
    )

    model.save(f"models/hardcore/{name}_final")
    print(f"\n{name} Hardcore Done!")

    env.close()
    eval_env.close()


# Train All on Hardcore
train(SAC,  "SAC_v2",
      pretrained_path="models/hardcore/SAC_final")

train(PPO,  "PPO_v2",
      pretrained_path="models/hardcore/PPO_final")

train(TD3,  "TD3_v2",
      pretrained_path="models/hardcore/TD3_final")

train(DDPG, "DDPG_v2",
      pretrained_path="models/hardcore/DDPG_final")

print("\n" + "="*40)
print("All Hardcore Training Done!")
print("="*40)