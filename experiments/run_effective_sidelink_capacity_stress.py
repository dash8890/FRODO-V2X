"""Frozen FRODO/D3QN/heuristic stress test for reduced sidelink pool capacity."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch

import main
import main_rainbow_repaired as repaired
from baseline_agents import GreedyHeuristicAgent, load_d3qn_checkpoint


ROOT = Path("controlled_uv_experiment_v2_curriculum")
OUTPUT = ROOT / "effective_sidelink_capacity_stress_v1"
SCHEDULE = ROOT / "schedules" / "evaluation_schedule_shared.json"
METHODS = ("FRODO", "D3QN", "Heuristic")
SEEDS = (1, 2, 3)
ALPHAS = (1.0, 0.75, 0.5)
EPISODES = 50
STEPS = 100
BASE_SEED = 7_501_192
SUBCHANNEL_BANDWIDTH = 1.8e6


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def environment_for(method):
    module = repaired if method == "FRODO" else main
    environment = module.LuSTMobileREPCEnvironment(
        trace_file="lust_fcd_sample.csv", highway_only=True,
        controlled_uv=True, controlled_uv_schedule=str(SCHEDULE),
        controlled_uv_episodes=EPISODES, controlled_uv_steps=STEPS,
    )
    return environment, module


def configure_pool(environment, module, bandwidth_mhz, alpha):
    # Keep the nominal PHY bandwidth/k_max unchanged. Reduce only the resource
    # pool passed through SB-SPS so this is a capacity-abstraction stress test.
    environment.total_bandwidth = float(bandwidth_mhz) * 1e6
    nominal_capacity = max(1, int(environment.total_bandwidth / SUBCHANNEL_BANDWIDTH))
    effective_capacity = max(1, math.floor(alpha * nominal_capacity))
    environment.k_max = nominal_capacity
    environment.noise_floor = -174 + 10 * np.log10(environment.total_bandwidth)
    environment.sbsps_pool = module.SBSPSResourcePool(k_max=effective_capacity)
    return nominal_capacity, effective_capacity


def load_agent(method, seed, environment):
    if method == "FRODO":
        return repaired.load_rainbow_checkpoint(
            str(ROOT / "rainbow_repaired_v1" / "checkpoints" /
                f"rainbow_repaired_seed_{seed}.pt"), evaluation=True)[0]
    if method == "D3QN":
        return load_d3qn_checkpoint(
            str(ROOT / "baseline_experiments_v1" / "d3qn_corrected_v2" /
                "checkpoints" / f"d3qn_corrected_v2_seed_{seed}.pt"),
            evaluation=True)[0]
    return GreedyHeuristicAgent(environment)


def common_draws(seed):
    rng = np.random.default_rng(BASE_SEED + seed)
    task_p = rng.integers(1, 11, size=EPISODES)
    bandwidth = rng.uniform(20.0, 40.0, size=EPISODES)
    return task_p, bandwidth


def select_workers(method, module, environment, available, action_idx, count):
    if method == "FRODO":
        return [item[0] for item in repaired.rank_workers(environment, available)[:count]], 4
    strategy = (action_idx // 10) % 5
    if strategy == 4:
        selected = [item[0] for item in module.rank_workers(environment, available)[:count]]
    elif strategy == 0:
        selected = list(np.random.choice(available, size=min(count, len(available)), replace=False))
    elif strategy == 1:
        ranked = module.rank_workers(environment, available)
        selected = [item[0] for item in reversed(ranked[:count])]
    elif strategy == 2:
        selected = sorted(available, key=lambda wid: environment.current_frame_vehicles.get(wid, {}).get("time_to_exit", 999))[:count]
    else:
        selected = sorted(available, key=lambda wid: environment.current_frame_vehicles.get(wid, {}).get("distance", 0), reverse=True)[:count]
    return selected, strategy


def evaluate(method, seed, alpha):
    set_seed(BASE_SEED + seed)
    environment, module = environment_for(method)
    agent = load_agent(method, seed, environment)
    task_draws, bandwidth_draws = common_draws(seed)
    rows = []
    for episode in range(EPISODES):
        state = environment.reset()
        p_value = int(task_draws[episode])
        task_size = 12 * p_value
        bandwidth = float(bandwidth_draws[episode])
        nominal_capacity, effective_capacity = configure_pool(
            environment, module, bandwidth, alpha)
        environment.current_task = {
            "A_dim": (task_size, task_size), "B_dim": (task_size, task_size),
            "deadline": (0.5 + 1.5 * (p_value - 1) / 10) * 1000,
        }
        state = environment._get_state()
        for step in range(STEPS):
            available = module.get_available_workers(environment)
            if not available:
                rows.append({
                    "method": method, "seed": seed, "alpha": alpha,
                    "episode": episode + 1, "step": step, "task_size": task_size,
                    "bandwidth_mhz": bandwidth, "nominal_pool_capacity": nominal_capacity,
                    "effective_pool_capacity": effective_capacity,
                    "instantaneous_available_capacity": 0, "available_servers": 0,
                    "selected_servers": 0, "action_idx": -1, "worker_strategy": -1,
                    "success": 0, "reward": -1.0, "total_latency_ms": 0.0, "cost": 0.0,
                })
                break
            scenario_seed = BASE_SEED + seed * 100_000 + episode * 1_000 + step * 10
            set_seed(scenario_seed + 1)
            instantaneous_capacity = max(
                1, environment.sbsps_pool.step(available, environment.current_slot))
            mask = module.feasible_action_mask(
                len(available), task_size, instantaneous_capacity)
            action_idx = agent.act(state, action_mask=mask)
            l_value, m_value, n_value, epsilon = module.constrained_action_selection(
                action_idx, len(available), task_size, instantaneous_capacity)
            selected_count = max(
                1, l_value*m_value*n_value + l_value - 1 - int(epsilon*l_value*m_value))
            set_seed(scenario_seed + 2)
            selected, strategy = select_workers(
                method, module, environment, available, action_idx, selected_count)
            set_seed(scenario_seed + 3)
            success, reward, done, latency, enc, up, down, comp, dec, cost = environment.step({
                "params": (l_value, m_value, n_value, epsilon), "workers": selected})
            rows.append({
                "method": method, "seed": seed, "alpha": alpha,
                "episode": episode + 1, "step": step, "task_size": task_size,
                "bandwidth_mhz": bandwidth, "nominal_pool_capacity": nominal_capacity,
                "effective_pool_capacity": effective_capacity,
                "instantaneous_available_capacity": instantaneous_capacity,
                "available_servers": len(available), "selected_servers": selected_count,
                "action_idx": int(action_idx), "worker_strategy": int(strategy),
                "success": int(success), "reward": float(reward),
                "total_latency_ms": float(latency), "cost": float(cost),
                "encoding_ms": float(enc), "uplink_ms": float(up),
                "downlink_ms": float(down), "compute_ms": float(comp),
                "decoding_ms": float(dec),
            })
            state = environment._get_state()
            if done:
                break
    return rows


def summarize(rows):
    latency = np.asarray([row["total_latency_ms"] for row in rows if row["total_latency_ms"] > 0], float)
    cost = np.asarray([row["cost"] for row in rows if row["cost"] > 0], float)
    return {
        "decisions": len(rows),
        "success_percent": 100 * float(np.mean([row["success"] for row in rows])),
        "reward_mean": float(np.mean([row["reward"] for row in rows])),
        "latency_mean_s": float(latency.mean() / 1000),
        "latency_p95_s": float(np.percentile(latency, 95) / 1000),
        "cost_mean": float(cost.mean()), "cost_p95": float(np.percentile(cost, 95)),
        "selected_servers_mean": float(np.mean([row["selected_servers"] for row in rows])),
        "instantaneous_capacity_mean": float(np.mean([
            row["instantaneous_available_capacity"] for row in rows])),
    }


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    seed_statistics = {}
    for method in METHODS:
        for alpha in ALPHAS:
            key = f"{method}|{alpha}"
            seed_statistics[key] = []
            for seed in SEEDS:
                rows = evaluate(method, seed, alpha)
                output_path = OUTPUT / f"{method.lower()}_alpha_{str(alpha).replace('.', 'p')}_seed_{seed}.json"
                output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
                values = summarize(rows)
                seed_statistics[key].append(values)
                print(method, alpha, seed, values, flush=True)
    aggregate = {}
    metrics = (
        "success_percent", "reward_mean", "latency_mean_s", "latency_p95_s",
        "cost_mean", "cost_p95", "selected_servers_mean", "instantaneous_capacity_mean")
    for key, values in seed_statistics.items():
        aggregate[key] = {
            metric: float(np.mean([seed_value[metric] for seed_value in values]))
            for metric in metrics
        }
    summary = {
        "design": "Frozen-policy effective sidelink-capacity stress test",
        "methods": METHODS, "seeds": SEEDS, "alphas": ALPHAS,
        "episodes_per_method_seed_alpha": EPISODES, "steps_per_episode": STEPS,
        "capacity_definition": "max(1, floor(alpha * floor(B0 / 1.8 MHz)))",
        "seed_statistics": seed_statistics, "aggregate": aggregate,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    run()
