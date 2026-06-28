#test_best.py
from stable_baselines3 import SAC
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

model = SAC.load(
    "models/SAC_best/best/best_model",
    device="cpu"
)

env = Monitor(gym.make("BipedalWalkerHardcore-v3"))

mean, std = evaluate_policy(
    model, env,
    n_eval_episodes = 20,
    deterministic   = True
)

print(f"Best Model: {mean:.2f} +/- {std:.2f}")
env.close()