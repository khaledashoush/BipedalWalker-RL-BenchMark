import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo
import time
import os

# Settings
ENV_NAME = "BipedalWalker-v3"
N_EVAL_EPISODES = 20

os.makedirs("videos", exist_ok=True)

# Load Models
models = {
    "SAC":  SAC.load("models/SAC_final",   device="cpu"),
    "PPO":  PPO.load("models/PPO_final",   device="cpu"),
    "TD3":  TD3.load("models/TD3_final",   device="cpu"),
    "DDPG": DDPG.load("models/DDPG_v3_final", device="cpu")
}
# Evaluate All
print("\n" + "="*40)
print("Evaluation Results:")
print("="*40)

results = {}
for name, model in models.items():
    env = Monitor(gym.make(ENV_NAME))
    mean_reward, std_reward = evaluate_policy(
        model, env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True
    )
    results[name] = {
        "mean": mean_reward,
        "std":  std_reward
    }
    print(f"{name}: {mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()

# Best Model
best = max(results, key=lambda x: results[x]["mean"])
print(f"\nBest Algorithm: {best}")
print(f"Best Reward: {results[best]['mean']:.2f}")

# Record Video
try:
    print("\nRecording Videos...")
    for name, model in models.items():
        print(f"Recording {name}...")
        env = RecordVideo(
            gym.make(ENV_NAME, render_mode="rgb_array"),
            video_folder=f"videos/{name}",
            name_prefix=name
        )
        obs, info = env.reset()
        total_reward = 0

        for _ in range(1600):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                print(f"{name} Total Reward: {total_reward:.2f}")
                break

        env.close()
    print("Videos Saved in videos/ folder!")

except Exception as e:
    print(f"Video Error: {e}")
    print("Skipping video recording...")

# Watch Best Model Live
print(f"\nWatching {best} Live...")
model = models[best]
env = Monitor(gym.make(ENV_NAME, render_mode="human"))

obs, info = env.reset()
total_reward = 0

for _ in range(3000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.01)

    if done or truncated:
        print(f"Total Reward: {total_reward:.2f}")
        total_reward = 0
        obs, info = env.reset()

env.close()