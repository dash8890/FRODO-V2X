"""Run the seven frozen-policy tests used in the robustness table."""

import json
import random
from pathlib import Path

import numpy as np
import torch

import main_rainbow_repaired as frodo


ROOT = Path("controlled_uv_experiment_v2_curriculum")
OUTPUT = ROOT / "rainbow_repaired_frozen_robustness_table_v1"
SCHEDULE = ROOT / "schedules" / "evaluation_schedule_shared.json"
CHECKPOINTS = ROOT / "rainbow_repaired_v1" / "checkpoints"
SEEDS = (1, 2, 3)
EPISODES = 50
STEPS = 100
BASE_SEED = 4_912_192
CONDITIONS = (
    {"key": "training_distribution", "label": "Training dist.", "task_max": 120, "bandwidth": "uniform", "speed_scale": 1.0},
    {"key": "bandwidth_30mhz", "label": "$B_0=30$ MHz", "task_max": 120, "bandwidth": 30.0, "speed_scale": 1.0},
    {"key": "bandwidth_15mhz", "label": "$B_0=15$ MHz", "task_max": 120, "bandwidth": 15.0, "speed_scale": 1.0},
    {"key": "bandwidth_10mhz", "label": "$B_0=10$ MHz", "task_max": 120, "bandwidth": 10.0, "speed_scale": 1.0},
    {"key": "task_max_156", "label": r"$\omega_t^{\max}=156$", "task_max": 156, "bandwidth": "uniform", "speed_scale": 1.0},
    {"key": "task_max_192", "label": r"$\omega_t^{\max}=192$", "task_max": 192, "bandwidth": "uniform", "speed_scale": 1.0},
    {"key": "relative_speed_1p2", "label": r"$1.2\times$ rel. speed", "task_max": 120, "bandwidth": "uniform", "speed_scale": 1.2},
    {"key": "relative_speed_1p4", "label": r"$1.4\times$ rel. speed", "task_max": 120, "bandwidth": "uniform", "speed_scale": 1.4},
)


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class RobustnessEnvironment(frodo.LuSTMobileREPCEnvironment):
    def __init__(self):
        self.relative_velocity_scale = 1.0
        super().__init__(
            trace_file="lust_fcd_sample.csv", highway_only=True,
            controlled_uv=True, controlled_uv_schedule=str(SCHEDULE),
            controlled_uv_episodes=EPISODES, controlled_uv_steps=STEPS,
        )

    def set_bandwidth_mhz(self, bandwidth_mhz):
        self.current_bandwidth_mhz = float(bandwidth_mhz)
        self.total_bandwidth = float(bandwidth_mhz) * 1e6
        self.k_max = max(1, int(self.total_bandwidth / self.subchannel_bandwidth))
        self.noise_floor = -174 + 10 * np.log10(self.total_bandwidth)
        self.sbsps_pool = frodo.SBSPSResourcePool(k_max=self.k_max)

    def set_propagation_shift(self, delta_ex, sigma_sh):
        self.delta_ex = float(delta_ex)
        self.sigma_sh = float(sigma_sh)

    def update_coverage(self):
        super().update_coverage()
        if not self.active_workers or self.relative_velocity_scale == 1.0:
            return
        timestep = self.timesteps[self.current_step_idx]
        frame = self.trace_df[self.trace_df["timestep"] == timestep]
        source_id = self.controlled_uv_source_id if self.controlled_uv else self.client_id
        client_rows = frame[frame["vehicle_id"].astype(str) == str(source_id)]
        if client_rows.empty:
            return
        client = client_rows.iloc[0]
        angle = np.radians(client.get("angle", 0.0))
        client_velocity = np.asarray([
            client["speed"] * np.sin(angle), client["speed"] * np.cos(angle)], float)
        client_position = (client["x"], client["y"])
        for worker in self.current_frame_vehicles.values():
            worker_velocity = np.asarray([worker["vx"], worker["vy"]], float)
            relative_velocity = (worker_velocity - client_velocity) * self.relative_velocity_scale
            scaled_velocity = client_velocity + relative_velocity
            worker["vx"], worker["vy"] = map(float, scaled_velocity)
            worker["speed"] = float(np.linalg.norm(scaled_velocity))
            worker["relative_speed_mps"] = float(np.linalg.norm(relative_velocity))
            worker["time_to_exit"] = self._compute_2d_time_to_exit(
                (worker["x"], worker["y"]), client_position, scaled_velocity,
                client_velocity, self.coverage_radius)


def common_draws(seed):
    rng = np.random.default_rng(BASE_SEED + seed)
    return rng.random(EPISODES), rng.uniform(20.0, 40.0, EPISODES)


def evaluate(seed, condition):
    set_seed(BASE_SEED + seed)
    environment = RobustnessEnvironment()
    environment.relative_velocity_scale = condition["speed_scale"]
    environment.set_propagation_shift(condition.get("delta_ex", 0.0), condition.get("sigma_sh", 0.0))
    agent = frodo.load_rainbow_checkpoint(
        str(CHECKPOINTS / f"rainbow_repaired_seed_{seed}.pt"), evaluation=True)[0]
    task_quantiles, bandwidth_draws = common_draws(seed)
    maximum_p = condition["task_max"] // 12
    rows = []
    for episode in range(EPISODES):
        state = environment.reset()
        bandwidth = (float(bandwidth_draws[episode]) if condition["bandwidth"] == "uniform"
                     else float(condition["bandwidth"]))
        environment.set_bandwidth_mhz(bandwidth)
        p_value = min(maximum_p, int(task_quantiles[episode] * maximum_p) + 1)
        task_size = 12 * p_value
        environment.current_task = {
            "A_dim": (task_size, task_size), "B_dim": (task_size, task_size),
            "deadline": (0.5 + 1.5 * (p_value - 1) / 10) * 1000,
        }
        state = environment._get_state()
        for step in range(STEPS):
            available = frodo.get_available_workers(environment)
            if not available:
                rows.append({
                    "seed": seed, "episode": episode + 1, "step": step,
                    "task_size": task_size, "bandwidth_mhz": bandwidth,
                    "relative_speed_scale": condition["speed_scale"],
                    "success": 0, "reward": -1.0, "total_latency_ms": 0.0,
                    "cost": 0.0, "available_servers": 0, "selected_servers": 0,
                    "action_idx": -1,
                })
                break
            scenario_seed = BASE_SEED + seed * 100_000 + episode * 1_000 + step * 10
            set_seed(scenario_seed + 1)
            subchannels = max(1, environment.sbsps_pool.step(available, environment.current_slot))
            mask = frodo.feasible_action_mask(len(available), task_size, subchannels)
            action_idx = agent.act(state, action_mask=mask)
            l_value, m_value, n_value, epsilon = frodo.constrained_action_selection(
                action_idx, len(available), task_size, subchannels)
            selected_count = max(
                1, l_value*m_value*n_value + l_value - 1 - int(epsilon*l_value*m_value))
            selected = [item[0] for item in frodo.rank_workers(environment, available)[:selected_count]]
            set_seed(scenario_seed + 3)
            success, reward, done, latency, enc, up, down, comp, dec, cost = environment.step({
                "params": (l_value, m_value, n_value, epsilon), "workers": selected})
            rows.append({
                "seed": seed, "episode": episode + 1, "step": step,
                "task_size": task_size, "bandwidth_mhz": bandwidth,
                "relative_speed_scale": condition["speed_scale"],
                "success": int(success), "reward": float(reward),
                "total_latency_ms": float(latency), "cost": float(cost),
                "encoding_ms": float(enc), "uplink_ms": float(up),
                "downlink_ms": float(down), "compute_ms": float(comp),
                "decoding_ms": float(dec), "available_servers": len(available),
                "selected_servers": int(selected_count), "action_idx": int(action_idx),
            })
            state = environment._get_state()
            if done:
                break
    return rows


def statistics(rows):
    latency = np.asarray([row["total_latency_ms"] for row in rows if row["total_latency_ms"] > 0], float)
    cost = np.asarray([row["cost"] for row in rows if row["cost"] > 0], float)
    return {
        "decisions": len(rows),
        "success_percent": 100 * float(np.mean([row["success"] for row in rows])),
        "latency_mean_s": float(latency.mean() / 1000),
        "latency_p95_s": float(np.percentile(latency, 95) / 1000),
        "cost_mean": float(cost.mean()),
        "cost_p95": float(np.percentile(cost, 95)),
    }


def latex_table(aggregates):
    lines = [
        r"\begin{table}[!t]", r"\caption{Frozen-policy performance under different operating conditions.",
        r"The reported values are averages of the seed-wise statistics over three",
        r"independent seeds; cost is expressed in units of $10^{-15}$.}",
        r"\label{tab:frozen_robustness}", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"\multirow{2}{*}{\textbf{Configuration}} & \multirow{2}{*}{\textbf{Success (\%)}} &",
        r"\multicolumn{2}{c}{\textbf{Latency (s)}} & \multicolumn{2}{c}{\textbf{Cost ($10^{-15}$)}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r"& & \textbf{Mean} & \textbf{P95} & \textbf{Mean} & \textbf{P95} \\", r"\midrule",
    ]
    for index, condition in enumerate(CONDITIONS):
        values = aggregates[condition["key"]]
        lines.append(
            f'{condition["label"]} & ${values["success_percent"]:.2f}$ & '
            f'${values["latency_mean_s"]:.3f}$ & ${values["latency_p95_s"]:.3f}$ & '
            f'${values["cost_mean"]*1e15:.3f}$ & ${values["cost_p95"]*1e15:.3f}$ \\\\')
        if index in (0, 3, 5):
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_seed_stats = {}
    for condition in CONDITIONS:
        all_seed_stats[condition["key"]] = []
        for seed in SEEDS:
            rows = evaluate(seed, condition)
            path = OUTPUT / f'{condition["key"]}_seed_{seed}.json'
            path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            values = statistics(rows)
            all_seed_stats[condition["key"]].append(values)
            print(condition["key"], seed, values, flush=True)
    aggregates = {}
    for condition in CONDITIONS:
        key = condition["key"]
        aggregates[key] = {
            metric: float(np.mean([values[metric] for values in all_seed_stats[key]]))
            for metric in ("success_percent", "latency_mean_s", "latency_p95_s", "cost_mean", "cost_p95")
        }
    summary = {
        "method": "FRODO repaired frozen policy", "seeds": SEEDS,
        "episodes_per_seed_condition": EPISODES, "steps_per_episode": STEPS,
        "conditions": CONDITIONS, "seed_statistics": all_seed_stats,
        "averages_of_seedwise_statistics": aggregates,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "frozen_robustness_table.tex").write_text(latex_table(aggregates), encoding="utf-8")
    print(latex_table(aggregates), flush=True)


if __name__ == "__main__":
    run()
