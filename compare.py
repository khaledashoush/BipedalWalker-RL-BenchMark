import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from stable_baselines3 import SAC, PPO, TD3, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import os

# Config
ENV_NAME        = "BipedalWalker-v3"
ENV_HARD        = "BipedalWalkerHardcore-v3"
N_EVAL_EPISODES = 50

plt.style.use('dark_background')

COLORS = {
    "SAC":  "#00B4D8",
    "PPO":  "#06D6A0",
    "TD3":  "#EF476F",
    "DDPG": "#FFD166"
}
# Load Models
models = {
    "SAC":  SAC.load("models/SAC_final",       device="cpu"),
    "PPO":  PPO.load("models/PPO_final",       device="cpu"),
    "TD3":  TD3.load("models/TD3_final",       device="cpu"),
    "DDPG": DDPG.load("models/DDPG_v3_final", device="cpu")
}

# Evaluate Normal
print("\n" + "="*50)
print("Evaluating on BipedalWalker-v3 (Normal)")
print("="*50)

names  = list(models.keys())
means  = []
stds   = []
colors = list(COLORS.values())

for name, model in models.items():
    env = Monitor(gym.make(ENV_NAME))
    mean, std = evaluate_policy(
        model, env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True
    )
    means.append(mean)
    stds.append(std)
    env.close()
    print(f"{name}: {mean:.2f} +/- {std:.2f}")

# Evaluate Hardcore
print("\n" + "="*50)
print("Evaluating on BipedalWalkerHardcore-v3")
print("="*50)

means_hard = []
stds_hard  = []

for name, model in models.items():
    env = Monitor(gym.make(ENV_HARD))
    mean, std = evaluate_policy(
        model, env,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True
    )
    means_hard.append(mean)
    stds_hard.append(std)
    env.close()
    print(f"{name}: {mean:.2f} +/- {std:.2f}")

# Figure Setup
fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('#0D1117')

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.6,
    wspace=0.4
)

fig.suptitle(
    "BipedalWalker - RL Algorithm Benchmarking\n"
    "Normal vs Hardcore Environment",
    fontsize=16,
    fontweight='bold',
    color='white'
)

# Helper: style axes
# ─────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor('#161B22')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    for spine in ax.spines.values():
        spine.set_color('#30363D')

# ─────────────────────────────────────────
# Plot 1 - Normal Environment
# ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)

bars = ax1.bar(
    names, means,
    yerr=stds,
    capsize=5,
    color=colors,
    alpha=0.85,
    width=0.6,
    error_kw=dict(ecolor='white', linewidth=2)
)
ax1.set_title(
    "Normal Environment\nMean Reward (Higher = Better)",
    fontsize=12, fontweight='bold', color='white', pad=10
)
ax1.set_xlabel("Algorithm", color='white')
ax1.set_ylabel("Mean Reward", color='white')
ax1.axhline(y=0, color='white', linestyle='--', alpha=0.3)

for bar, mean in zip(bars, means):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(stds) * 0.05,
        f'{mean:.1f}',
        ha='center', va='bottom',
        fontweight='bold', color='white', fontsize=10
    )

# ─────────────────────────────────────────
# Plot 2 - Hardcore Environment
# ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)

bars = ax2.bar(
    names, means_hard,
    yerr=stds_hard,
    capsize=5,
    color=colors,
    alpha=0.85,
    width=0.6,
    error_kw=dict(ecolor='white', linewidth=2)
)
ax2.set_title(
    "Hardcore Environment\nMean Reward (Higher = Better)",
    fontsize=12, fontweight='bold', color='white', pad=10
)
ax2.set_xlabel("Algorithm", color='white')
ax2.set_ylabel("Mean Reward", color='white')
ax2.axhline(y=0, color='white', linestyle='--', alpha=0.3)

for bar, mean in zip(bars, means_hard):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 3,
        f'{mean:.1f}',
        ha='center', va='top',
        fontweight='bold', color='white', fontsize=10
    )

# ─────────────────────────────────────────
# Plot 3 - Stability Comparison
# ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3)

x     = np.arange(len(names))
width = 0.35

ax3.bar(x - width/2, stds,      width,
        label='Normal',   color=colors, alpha=0.85)
ax3.bar(x + width/2, stds_hard, width,
        label='Hardcore', color=colors, alpha=0.45, hatch='//')

ax3.set_title(
    "Stability Comparison\nStd Dev (Lower = Better)",
    fontsize=12, fontweight='bold', color='white', pad=10
)
ax3.set_xlabel("Algorithm", color='white')
ax3.set_ylabel("Std Deviation", color='white')
ax3.set_xticks(x)
ax3.set_xticklabels(names)
ax3.legend(facecolor='#161B22', labelcolor='white')

# ─────────────────────────────────────────
# Plot 4 - Radar Chart
# ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0], polar=True)
ax4.set_facecolor('#161B22')

categories = ['Reward', 'Stability', 'Hardcore', 'Overall']
N          = len(categories)

max_mean      = max(means)
max_std       = max(stds)
min_mean_hard = min(means_hard)
max_mean_hard = max(means_hard)
range_hard    = max_mean_hard - min_mean_hard if max_mean_hard != min_mean_hard else 1

scores = []
for mean, std, mean_h in zip(means, stds, means_hard):
    r = mean / max_mean * 100
    s = (1 - std / max_std) * 100
    h = (mean_h - min_mean_hard) / range_hard * 100
    o = (r + s + h) / 3
    scores.append([r, s, h, o])

angles  = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, (name, score) in enumerate(zip(names, scores)):
    values = score + score[:1]
    ax4.plot(angles, values, 'o-',
             linewidth=2, label=name, color=colors[i])
    ax4.fill(angles, values, alpha=0.1, color=colors[i])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories, color='white')
ax4.set_title(
    "Overall Comparison",
    fontsize=12, fontweight='bold', color='white', pad=20
)
ax4.tick_params(colors='white')
ax4.grid(color='white', alpha=0.2)
ax4.legend(
    loc='upper right',
    bbox_to_anchor=(1.4, 1.2),
    facecolor='#161B22',
    labelcolor='white'
)

# ─────────────────────────────────────────
# Plot 5 - Normal vs Hardcore
# ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5)

x     = np.arange(len(names))
width = 0.35

ax5.bar(x - width/2, means,      width,
        label='Normal',   color=colors, alpha=0.85)
ax5.bar(x + width/2, means_hard, width,
        label='Hardcore', color=colors, alpha=0.45, hatch='//')

ax5.set_title(
    "Normal vs Hardcore\nMean Reward",
    fontsize=12, fontweight='bold', color='white', pad=10
)
ax5.set_xlabel("Algorithm", color='white')
ax5.set_ylabel("Mean Reward", color='white')
ax5.set_xticks(x)
ax5.set_xticklabels(names)
ax5.axhline(y=0, color='white', linestyle='--', alpha=0.3)
ax5.legend(facecolor='#161B22', labelcolor='white')

# ─────────────────────────────────────────
# Plot 6 - Summary Table
# ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#161B22')
ax6.axis('off')

ranked = sorted(
    zip(names, means, stds, means_hard, stds_hard),
    key=lambda x: x[1],
    reverse=True
)

medals_list = ['1st', '2nd', '3rd', '4th']
table_data  = []

for i, (name, mean, std, mean_h, std_h) in enumerate(ranked):
    table_data.append([
        f"{medals_list[i]} {name}",
        f"{mean:.1f}",
        f"{std:.1f}",
        f"{mean_h:.1f}",
        f"{std_h:.1f}"
    ])

table = ax6.table(
    cellText  = table_data,
    colLabels = ['Algorithm', 'Normal\nMean',
                 'Normal\nStd', 'Hard\nMean', 'Hard\nStd'],
    cellLoc   = 'center',
    loc       = 'center',
    bbox      = [0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(10)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor('#161B22')
    cell.set_text_props(color='white')
    cell.set_edgecolor('#30363D')
    if row == 0:
        cell.set_facecolor('#21262D')
        cell.set_text_props(color='white', fontweight='bold')

ax6.set_title(
    "Summary Table",
    fontsize=12, fontweight='bold', color='white', pad=10
)

# Plot 7 - Learning Curves
ax7 = fig.add_subplot(gs[2, :])
style_ax(ax7)

log_paths = {
    "SAC":  "logs/SAC/evaluations.npz",
    "PPO":  "logs/PPO/evaluations.npz",
    "TD3":  "logs/TD3/evaluations.npz",
    "DDPG": "logs/DDPG_v3/evaluations.npz"
}

for i, (name, log_path) in enumerate(log_paths.items()):
    if os.path.exists(log_path):
        data      = np.load(log_path)
        timesteps = data['timesteps']
        results_  = data['results']
        mean_r    = results_.mean(axis=1)
        std_r     = results_.std(axis=1)
        ax7.plot(timesteps, mean_r,
                 label=name, color=colors[i], linewidth=2.5)
        ax7.fill_between(
            timesteps,
            mean_r - std_r,
            mean_r + std_r,
            alpha=0.15, color=colors[i]
        )
    else:
        print(f"Warning: log not found → {log_path}")

ax7.set_title(
    "Learning Curves - All Algorithms",
    fontsize=14, fontweight='bold', color='white', pad=10
)
ax7.set_xlabel("Timesteps", color='white')
ax7.set_ylabel("Mean Reward", color='white')
ax7.axhline(y=0, color='white', linestyle='--', alpha=0.3)
ax7.legend(facecolor='#161B22', labelcolor='white', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.92])   
plt.savefig("comparison_final.png", dpi=300,
            facecolor='#0D1117')         


plt.show()
print("\nPlot saved as comparison_final.png")

# Terminal Summary
print("\n" + "="*70)
print(f"{'Rank':<6} {'Algorithm':<10} "
      f"{'Normal Mean':>12} {'Normal Std':>12} "
      f"{'Hard Mean':>12} {'Hard Std':>12}")
print("="*70)

medals = ['1st', '2nd', '3rd', '4th']
for i, (name, mean, std, mean_h, std_h) in enumerate(ranked):
    print(f"{medals[i]:<6} {name:<10} "
          f"{mean:>12.2f} {std:>12.2f} "
          f"{mean_h:>12.2f} {std_h:>12.2f}")

print("="*70)