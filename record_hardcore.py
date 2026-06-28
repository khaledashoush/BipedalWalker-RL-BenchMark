import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3, DDPG
from gymnasium.wrappers import RecordVideo
import os

ENV_NAME = "BipedalWalkerHardcore-v3"
SEEDS    = [42, 123, 456, 789]

os.makedirs("videos/hardcore_v2_on_hardcore",
            exist_ok=True)

print("Loading hardcore_v2 models...", flush=True)
models = {
    "SAC":  SAC.load(
        "models/hardcore_v2/SAC_final",
        device="cpu"),
    "PPO":  PPO.load(
        "models/hardcore_v2/PPO_final",
        device="cpu"),
    "TD3":  TD3.load(
        "models/hardcore_v2/TD3_final",
        device="cpu"),
    "DDPG": DDPG.load(
        "models/hardcore_v2/DDPG_final",
        device="cpu")
}
print("Models loaded!", flush=True)

results = {}

for seed in SEEDS:
    folder = f"track_{seed}"
    os.makedirs(
        f"videos/hardcore_v2_on_hardcore/{folder}",
        exist_ok=True
    )

    print(f"\n{'='*50}", flush=True)
    print(f"Track {SEEDS.index(seed)+1}"
          f" - Seed {seed}", flush=True)
    print(f"{'='*50}", flush=True)

    results[seed] = {}
    for name, model in models.items():
        print(f"  Recording {name}...", flush=True)

        env = RecordVideo(
            gym.make(ENV_NAME,
                     render_mode="rgb_array"),
            video_folder =
                f"videos/hardcore_v2_on_hardcore/{folder}",
            name_prefix  = f"{name}_seed{seed}"
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
        print(f"  {name}: {total_reward:.2f}"
              f" Steps={steps}", flush=True)

        results[seed][name] = {
            "reward": total_reward,
            "steps":  steps
        }

print("\n" + "="*70, flush=True)
print("Hardcore v2 Models on Hardcore Environment",
      flush=True)
print(f"{'Track':<10} {'SAC':>10}"
      f" {'PPO':>10} {'TD3':>10}"
      f" {'DDPG':>10}", flush=True)
print("="*70, flush=True)

for seed in SEEDS:
    track = f"Track {SEEDS.index(seed)+1}"
    sac   = results[seed]["SAC"]["reward"]
    ppo   = results[seed]["PPO"]["reward"]
    td3   = results[seed]["TD3"]["reward"]
    ddpg  = results[seed]["DDPG"]["reward"]
    print(f"{track:<10} {sac:>10.2f}"
          f" {ppo:>10.2f} {td3:>10.2f}"
          f" {ddpg:>10.2f}", flush=True)

print("="*70, flush=True)
print("Done!", flush=True)