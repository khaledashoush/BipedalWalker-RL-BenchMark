import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import numpy as np
import os

# Settings
ENV_NAME  = "BipedalWalker-v3"
ENV_HARD  = "BipedalWalkerHardcore-v3"

os.makedirs("videos/normal",   exist_ok=True)
os.makedirs("videos/hardcore", exist_ok=True)

# Load Models
models = {
    "SAC":  SAC.load("models/SAC_final",      device="cpu"),
    "PPO":  PPO.load("models/PPO_final",      device="cpu"),
    "TD3":  TD3.load("models/TD3_final",      device="cpu"),
    "DDPG": DDPG.load("models/DDPG_v3_final", device="cpu")
}

# Record Function
def record(model, name, env_name, folder):
    print(f"Recording {name} on {folder}...")
    try:
        env = RecordVideo(
            gym.make(env_name, render_mode="rgb_array"),
            video_folder = f"videos/{folder}/{name}",
            name_prefix  = f"{name}_{folder}"
        )
        obs, info    = env.reset()
        total_reward = 0
        steps        = 0

        for _ in range(2000):
            action, _ = model.predict(
                obs, deterministic=True
            )
            obs, reward, done, truncated, info = \
                env.step(action)
            total_reward += reward
            steps        += 1

            if done or truncated:
                break

        env.close()
        print(f"  {name}: Reward={total_reward:.2f}"
              f" Steps={steps}")
        return total_reward, steps

    except Exception as e:
        print(f"  Error: {e}")
        return 0, 0

# Record Normal
print("\n" + "="*50)
print("Recording Normal Environment...")
print("="*50)

normal_results = {}
for name, model in models.items():
    reward, steps = record(
        model, name, ENV_NAME, "normal"
    )
    normal_results[name] = {
        "reward": reward,
        "steps":  steps
    }

# Record Hardcore
print("\n" + "="*50)
print("Recording Hardcore Environment...")
print("="*50)

hard_results = {}
for name, model in models.items():
    reward, steps = record(
        model, name, ENV_HARD, "hardcore"
    )
    hard_results[name] = {
        "reward": reward,
        "steps":  steps
    }

# Compare Results
print("\n" + "="*60)
print(f"{'Algorithm':<10} {'Normal':>10} "
      f"{'Steps':>8} {'Hardcore':>10} {'Steps':>8}")
print("="*60)

for name in models.keys():
    n_r = normal_results[name]["reward"]
    n_s = normal_results[name]["steps"]
    h_r = hard_results[name]["reward"]
    h_s = hard_results[name]["steps"]
    print(f"{name:<10} {n_r:>10.2f} "
          f"{n_s:>8} {h_r:>10.2f} {h_s:>8}")

print("="*60)

# Best in Each Environment
best_normal   = max(normal_results,
                    key=lambda x:
                    normal_results[x]["reward"])
best_hardcore = max(hard_results,
                    key=lambda x:
                    hard_results[x]["reward"])

print(f"\nBest Normal:   {best_normal} = "
      f"{normal_results[best_normal]['reward']:.2f}")
print(f"Best Hardcore: {best_hardcore} = "
      f"{hard_results[best_hardcore]['reward']:.2f}")

# Videos Location
print("\n" + "="*50)
print("Videos saved in:")
print("  Normal:   videos/normal/")
print("  Hardcore: videos/hardcore/")
print("="*50)