"""Summarize and plot the three-seed persistent-UV frozen FRODO test."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "controlled_uv_experiment_v2_curriculum" / "persistent_uv_highway_v5"
RESULTS = BASE / "frozen_frodo_results"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)
PHASES = [
    ("Training-calibrated", 0, 13, "slategray"),
    ("Rush hour", 14, 33, "crimson"),
    ("Held-out motorway", 34, 49, "darkviolet"),
]


frames = []
episode_curves = []
for seed in (1, 2, 3):
    path = RESULTS / f"persistent_uv_training_rush_heldout_motorway_seed_{seed}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload)
    frame["seed"] = seed
    frame["selected_servers"] = np.minimum(frame["k_reduced"], frame["available_workers"])
    frame["server_utilization"] = np.where(
        frame["available_workers"] > 0,
        frame["selected_servers"] / frame["available_workers"],
        np.nan,
    )
    frames.append(frame)
    episode_curves.append(frame.groupby("episode")["reward"].mean().reindex(range(50)))

data = pd.concat(frames, ignore_index=True)
curves = np.vstack([curve.to_numpy(float) for curve in episode_curves])
mean_curve = np.nanmean(curves, axis=0)
ci_curve = 4.3026527299 * np.nanstd(curves, axis=0, ddof=1) / np.sqrt(3)
x = np.arange(1, 51)

fig, ax = plt.subplots(figsize=(20, 18))
for label, start, end, color in PHASES:
    ax.axvspan(start + 1, end + 1, color=color, alpha=.07, label=label)
    if start:
        ax.axvline(start + 1, color=color, linestyle="--", linewidth=5)
ax.plot(x, mean_curve, color="darkslategray", linewidth=6, marker="*", markersize=30,
        label="FRODO mean reward")
ax.fill_between(x, mean_curve - ci_curve, mean_curve + ci_curve,
                color="darkslategray", alpha=.2, label="95% CI across seeds")
ax.set_xlabel("Episodes", fontsize=70)
ax.set_ylabel("Mean per-step reward", fontsize=70)
ax.set_xticks([1, 10, 20, 30, 40, 50], ["1", "10", "20", "30", "40", "50"], fontsize=60)
ax.tick_params(axis="y", labelsize=60)
ax.grid(True, alpha=.4)
ax.legend(fontsize=38, loc="best", framealpha=0)
fig.tight_layout()
fig.savefig(PLOTS / "persistent_uv_frozen_frodo_reward.png", dpi=180)
plt.close(fig)

rows = []
for label, start, end, _ in PHASES:
    phase = data[data["episode"].between(start, end)]
    executed = phase[phase["total_latency_ms"] > 0]
    rows.append({
        "phase": label,
        "saved_decisions": int(len(phase)),
        "success_ratio": float(phase["success"].mean()),
        "conditional_success_ratio": float(executed["success"].mean()),
        "execution_availability_ratio": float(len(executed) / len(phase)),
        "mean_latency_ms": float(executed["total_latency_ms"].mean()),
        "mean_cost": float(executed.loc[executed["cost"] > 0, "cost"].mean()),
        "mean_available_servers": float(phase["available_workers"].mean()),
        "mean_selected_servers": float(phase["selected_servers"].mean()),
        "mean_server_utilization": float(phase["server_utilization"].mean()),
        "mean_relative_velocity_mps": float(phase["mean_relative_velocity_mps"].mean()),
        "mean_reward": float(phase["reward"].mean()),
    })
summary = pd.DataFrame(rows)
summary.to_csv(BASE / "persistent_uv_phase_metrics_three_seed.csv", index=False)
(BASE / "persistent_uv_phase_metrics_three_seed.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
metrics = [
    ("success_ratio", "Unconditional success ratio (%)", 100.0),
    ("mean_latency_ms", "Mean latency (ms)", 1.0),
    ("mean_cost", "Mean cost", 1.0),
    ("mean_server_utilization", "Mean server utilization", 100.0),
]
colors = [row[3] for row in PHASES]
for ax, (column, ylabel, multiplier) in zip(axes.flat, metrics):
    values = summary[column].to_numpy() * multiplier
    ax.bar(summary["phase"], values, color=colors, edgecolor="black", linewidth=2)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.tick_params(axis="both", labelsize=23)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=.4)
fig.tight_layout()
fig.savefig(PLOTS / "persistent_uv_phase_metrics.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(20, 18))
positions = np.arange(len(summary))
width = .24
for offset, column, label, color, hatch in (
    (-width, "success_ratio", "Unconditional success", "darkslategray", "//"),
    (0.0, "conditional_success_ratio", "Conditional success", "royalblue", "xx"),
    (width, "execution_availability_ratio", "Execution availability", "darkorange", ".."),
):
    ax.bar(positions + offset, summary[column] * 100, width=width,
           color=color, edgecolor="black", linewidth=3, hatch=hatch, label=label)
ax.set_xticks(positions, summary["phase"], fontsize=38)
ax.set_ylabel("Ratio (%)", fontsize=70)
ax.tick_params(axis="y", labelsize=60)
ax.set_ylim(95, 100.2)
ax.grid(axis="y", alpha=.4)
ax.legend(fontsize=26, loc="upper center", bbox_to_anchor=(.5, -.10),
          ncol=3, framealpha=0)
fig.tight_layout(rect=(0, .10, 1, 1))
fig.savefig(PLOTS / "persistent_uv_success_accounting.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(20, 18))
ax.bar(summary["phase"], summary["mean_relative_velocity_mps"],
       color=colors, edgecolor="black", linewidth=3)
ax.axhline(27.75, color="black", linestyle="--", linewidth=5)
ax.text(2.0, 27.45, "Original training/frozen reference: about 27.75 m/s",
        ha="right", va="top", fontsize=28, color="black")
ax.set_ylabel("Mean relative velocity (m/s)", fontsize=70)
ax.tick_params(axis="x", labelsize=38, rotation=12)
ax.tick_params(axis="y", labelsize=60)
ax.grid(axis="y", alpha=.4)
fig.tight_layout()
fig.savefig(PLOTS / "persistent_uv_mobility_calibration.png", dpi=180)
plt.close(fig)

print(summary.to_string(index=False))
print(f"Saved plots to {PLOTS}")
