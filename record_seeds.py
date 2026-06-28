# -*- coding: utf-8 -*-
import sys
import os

print("START", flush=True)
sys.stdout.flush()

try:
    import gymnasium as gym
    print("gym ok", flush=True)
    
    from stable_baselines3 import SAC, PPO, TD3, DDPG
    print("sb3 ok", flush=True)
    
    from gymnasium.wrappers import RecordVideo
    print("RecordVideo ok", flush=True)

    ENV_NAME = "BipedalWalkerHardcore-v3"
    SEEDS    = [42, 123, 456, 789]

    os.makedirs("videos/seeds", exist_ok=True)
    print("folders ok", flush=True)

    models = {
        "SAC":  SAC.load("models/SAC_final",
                         device="cpu"),
        "PPO":  PPO.load("models/PPO_final",
                         device="cpu"),
        "TD3":  TD3.load("models/TD3_final",
                         device="cpu"),
        "DDPG": DDPG.load("models/DDPG_v3_final",
                          device="cpu")
    }
    print("models ok", flush=True)

    results = {}

    for seed in SEEDS:
        folder = f"track_{seed}"
        os.makedirs(f"videos/seeds/{folder}",
                    exist_ok=True)

        print(f"\nTrack {SEEDS.index(seed)+1}"
              f" - Seed {seed}", flush=True)

        results[seed] = {}
        for name, model in models.items():
            print(f"  Recording {name}...",
                  flush=True)

            env = RecordVideo(
                gym.make(ENV_NAME,
                         render_mode="rgb_array"),
                video_folder=f"videos/seeds/{folder}",
                name_prefix=f"{name}_seed{seed}"
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
    print("DONE!", flush=True)

except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()