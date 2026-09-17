"""Fully paired task-size benchmark with repaired Rainbow checkpoints."""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import main
import main_rainbow_repaired as repaired
from baseline_agents import GreedyHeuristicAgent, load_d3qn_checkpoint


EXPERIMENT = Path("controlled_uv_experiment_v2_curriculum")
SCHEDULE = EXPERIMENT / "schedules" / "evaluation_schedule_shared.json"
OUTPUT = EXPERIMENT / "paired_cross_method_task_size_sweep_rainbow_repaired_v1"
TASK_SIZES = tuple(range(60, 205, 12))
SEEDS = (1, 2, 3)
EPISODES = 50
BASE_SEED = 917_300
METHODS = ("FRODO-Repaired", "D3QN", "Heuristic")


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def new_environment(method):
    environment_type = repaired.LuSTMobileREPCEnvironment if method == "FRODO-Repaired" else main.LuSTMobileREPCEnvironment
    return environment_type(
        trace_file="lust_fcd_sample.csv", highway_only=True,
        controlled_uv=True, controlled_uv_schedule=str(SCHEDULE),
        controlled_uv_episodes=EPISODES, controlled_uv_steps=100,
    )


def load_agent(method, seed, environment):
    if method == "FRODO-Repaired":
        return repaired.load_rainbow_checkpoint(
            str(EXPERIMENT / "rainbow_repaired_v1" / "checkpoints" /
                f"rainbow_repaired_seed_{seed}.pt"), evaluation=True)[0]
    if method == "D3QN":
        return load_d3qn_checkpoint(
            str(EXPERIMENT / "baseline_experiments_v1" / "d3qn_corrected_v2" /
                "checkpoints" / f"d3qn_corrected_v2_seed_{seed}.pt"),
            evaluation=True)[0]
    return GreedyHeuristicAgent(environment)


def select_workers(environment, available, action_idx, k_reduced, method):
    if method == "FRODO-Repaired":
        return [row[0] for row in repaired.rank_workers(environment, available)[:k_reduced]], 4
    rank_function = main.rank_workers
    strategy = (action_idx // 10) % 5
    if strategy == 4:
        selected = [row[0] for row in rank_function(environment, available)[:k_reduced]]
    elif strategy == 0:
        selected = list(np.random.choice(available, size=min(k_reduced, len(available)), replace=False))
    elif strategy == 1:
        ranked = rank_function(environment, available)
        selected = [row[0] for row in reversed(ranked[:k_reduced])]
    elif strategy == 2:
        selected = sorted(available, key=lambda wid: environment.current_frame_vehicles.get(wid, {}).get("time_to_exit", 999))[:k_reduced]
    else:
        selected = sorted(available, key=lambda wid: environment.current_frame_vehicles.get(wid, {}).get("distance", 0), reverse=True)[:k_reduced]
    return selected, strategy


def evaluate(method, seed, task_size):
    set_seed(BASE_SEED + seed)
    environment = new_environment(method)
    agent = load_agent(method, seed, environment)
    rows = []
    p_value = task_size // 12
    for episode in range(EPISODES):
        state = environment.reset()
        environment.current_task = {
            "A_dim": (task_size, task_size), "B_dim": (task_size, task_size),
            "deadline": (0.5 + 1.5 * (p_value - 1) / 10) * 1000,
        }
        state = environment._get_state()
        available = repaired.get_available_workers(environment)
        if not available:
            raise RuntimeError(f"No worker: {method=} {seed=} {task_size=} {episode=}")
        scenario_seed = BASE_SEED + seed * 10_000 + episode * 10
        set_seed(scenario_seed + 1)
        subchannels = max(1, environment.sbsps_pool.step(available, environment.current_slot))
        mask = repaired.feasible_action_mask(len(available), task_size, subchannels)
        action_idx = agent.act(state, action_mask=mask)
        l_value, m_value, n_value, epsilon = repaired.constrained_action_selection(
            action_idx, len(available), task_size, subchannels)
        k_reduced = max(1, l_value*m_value*n_value + l_value - 1 - int(epsilon*l_value*m_value))
        set_seed(scenario_seed + 2)
        selected, strategy = select_workers(environment, available, action_idx, k_reduced, method)
        set_seed(scenario_seed + 3)
        success, reward, _, latency, enc, up, down, comp, dec, cost = environment.step({
            "params": (l_value, m_value, n_value, epsilon), "workers": selected})
        rows.append({
            "method": method, "seed": seed, "episode": episode, "task_size": task_size,
            "source_vehicle": environment.controlled_uv_source_id,
            "start_time": environment.current_controlled_uv_segment.get("start_timestep"),
            "available_workers": len(available), "available_subchannels": subchannels,
            "action_idx": int(action_idx), "worker_strategy": int(strategy),
            "k_reduced": int(k_reduced), "success": int(success), "reward": float(reward),
            "total_latency_ms": float(latency), "encoding_ms": float(enc),
            "uplink_ms": float(up), "downlink_ms": float(down),
            "compute_ms": float(comp), "decoding_ms": float(dec), "cost": float(cost),
        })
    return rows


def summarize(rows):
    latency = np.asarray([row["total_latency_ms"] for row in rows], float)
    cost = np.asarray([row["cost"] for row in rows], float)
    return {
        "count": len(rows), "success_ratio": float(np.mean([row["success"] for row in rows])),
        "latency_mean_ms": float(latency.mean()), "latency_median_ms": float(np.median(latency)),
        "latency_p95_ms": float(np.percentile(latency, 95)), "latency_p99_ms": float(np.percentile(latency, 99)),
        "cost_mean": float(cost.mean()), "cost_median": float(np.median(cost)),
        "cost_p95": float(np.percentile(cost, 95)), "cost_p99": float(np.percentile(cost, 99)),
        "selected_servers_mean": float(np.mean([row["k_reduced"] for row in rows])),
        "available_servers_mean": float(np.mean([row["available_workers"] for row in rows])),
        "server_utilization_mean": float(np.mean([row["k_reduced"] / row["available_workers"] for row in rows])),
        "actions": Counter(row["action_idx"] for row in rows).most_common(),
    }


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for method in METHODS:
        for seed in SEEDS:
            method_path = OUTPUT / f"{method.lower()}_seed_{seed}.json"
            if method_path.exists():
                saved_rows = json.loads(method_path.read_text(encoding="utf-8"))
                all_rows.extend(saved_rows)
                print(f"Preserving and reusing {method_path}", flush=True)
                continue
            rows = []
            for size in TASK_SIZES:
                batch = evaluate(method, seed, size)
                rows.extend(batch)
                print(method, seed, size, summarize(batch), flush=True)
            method_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            all_rows.extend(rows)
    summary = {
        "design": "Fully paired one-decision counterfactual sweep with repaired Rainbow",
        "methods": METHODS, "seeds": SEEDS, "task_sizes": TASK_SIZES,
        "decisions_per_method_seed_size": EPISODES, "by_method_size": {},
    }
    for method in METHODS:
        for size in TASK_SIZES:
            rows = [row for row in all_rows if row["method"] == method and row["task_size"] == size]
            summary["by_method_size"][f"{method}|{size}"] = summarize(rows)
    (OUTPUT / "paired_cross_method_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
