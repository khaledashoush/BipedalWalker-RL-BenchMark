import gymnasium as gym
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo
import os

ENV_NAME = "BipedalWalkerHardcore-v3"
SEEDS    = [42, 123, 456, 789]

os.makedirs("videos/curriculum", exist_ok=True)

model = SAC.load(
    "models/curriculum/phase3_2000steps_best/best_model",
    device="cpu"
)
print("Model loaded!", flush=True)

results = {}
for seed in SEEDS:
    folder = f"track_{seed}"
    os.makedirs(f"videos/curriculum/{folder}",
                exist_ok=True)

    print(f"\nTrack {SEEDS.index(seed)+1}"
          f" - Seed {seed}", flush=True)

    env = RecordVideo(
        gym.make(ENV_NAME, render_mode="rgb_array"),
        video_folder = f"videos/curriculum/{folder}",
        name_prefix  = f"SAC_curriculum_seed{seed}"
    )

    obs, info    = env.reset(seed=seed)
    total_reward = 0
    steps        = 0

    for _ in range(2000):
        action, _ = model.predict(
            obs, deterministic=True
        )
        obs, reward, done, truncated, \
            info = env.step(action)
        total_reward += reward
        steps        += 1
        if done or truncated:
            break

    env.close()
    print(f"  Reward={total_reward:.2f}"
          f" Steps={steps}", flush=True)

    results[seed] = {
        "reward": total_reward,
        "steps":  steps
    }

print("\n" + "="*50, flush=True)
print("Curriculum SAC Results:", flush=True)
print("="*50, flush=True)
for seed in SEEDS:
    track = f"Track {SEEDS.index(seed)+1}"
    r     = results[seed]["reward"]
    s     = results[seed]["steps"]
    print(f"{track}: {r:.2f} Steps={s}",
          flush=True)
print("="*50, flush=True)