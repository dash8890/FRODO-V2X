import json
import numpy as np
import torch
import main_rainbow_repaired as frodo
import run_rainbow_repaired_frozen_robustness_table_v1 as base

TEST_CONDITIONS = [
    {"key": "bandwidth_15mhz", "label": r"$B_0=15$ MHz", "task_max": 120, "bandwidth": 15.0, "speed_scale": 1.0, "pool_scale": 1.0},
    {"key": "bandwidth_10mhz", "label": r"$B_0=10$ MHz", "task_max": 120, "bandwidth": 10.0, "speed_scale": 1.0, "pool_scale": 1.0},
    {"key": "contention_stress_50", "label": r"Contention stress ($50\%$ pool)", "task_max": 120, "bandwidth": "uniform", "speed_scale": 1.0, "pool_scale": 0.5},
]

class ScarcityEnvironment(base.RobustnessEnvironment):
    def __init__(self, pool_scale=1.0):
        super().__init__()
        self.pool_scale = float(pool_scale)

    def set_bandwidth_mhz(self, bandwidth_mhz):
        super().set_bandwidth_mhz(bandwidth_mhz)
        if self.pool_scale < 1.0:
            # Scale down available subchannels by pool_scale factor
            self.k_max = max(1, int(self.k_max * self.pool_scale))
            self.sbsps_pool = frodo.SBSPSResourcePool(k_max=self.k_max)

def evaluate_scarcity(seed, condition):
    base.set_seed(base.BASE_SEED + seed)
    env = ScarcityEnvironment(pool_scale=condition.get("pool_scale", 1.0))
    env.relative_velocity_scale = condition["speed_scale"]
    agent = frodo.load_rainbow_checkpoint(
        str(base.CHECKPOINTS / f"rainbow_repaired_seed_{seed}.pt"), evaluation=True)[0]
    task_quantiles, bandwidth_draws = base.common_draws(seed)
    maximum_p = condition["task_max"] // 12
    rows = []
    for episode in range(base.EPISODES):
        state = env.reset()
        bandwidth = (float(bandwidth_draws[episode]) if condition["bandwidth"] == "uniform"
                     else float(condition["bandwidth"]))
        env.set_bandwidth_mhz(bandwidth)
        p_value = min(maximum_p, int(task_quantiles[episode] * maximum_p) + 1)
        task_size = 12 * p_value
        env.current_task = {
            "A_dim": (task_size, task_size), "B_dim": (task_size, task_size),
            "deadline": (0.5 + 1.5 * (p_value - 1) / 10) * 1000,
        }
        state = env._get_state()
        for step in range(base.STEPS):
            available = frodo.get_available_workers(env)
            if not available:
                break
            scenario_seed = base.BASE_SEED + seed * 100_000 + episode * 1_000 + step * 10
            base.set_seed(scenario_seed + 1)
            subchannels = max(1, env.sbsps_pool.step(available, env.current_slot))
            mask = frodo.feasible_action_mask(len(available), task_size, subchannels)
            action_idx = agent.act(state, action_mask=mask)
            l_val, m_val, n_val, eps = frodo.constrained_action_selection(
                action_idx, len(available), task_size, subchannels)
            psi_t = max(1, l_val * m_val * n_val + l_val - 1 - int(eps * l_val * m_val))
            selected = [item[0] for item in frodo.rank_workers(env, available)[:psi_t]]
            base.set_seed(scenario_seed + 3)
            success, reward, done, latency, enc, up, down, comp, dec, cost = env.step({
                "params": (l_val, m_val, n_val, eps), "workers": selected})
            rows.append({
                "success": int(success), "latency": float(latency), "cost": float(cost),
                "psi_t": int(psi_t), "subchannels": int(subchannels),
                "l": int(l_val), "m": int(m_val), "n": int(n_val)
            })
            state = env._get_state()
            if done:
                break
    return rows

if __name__ == "__main__":
    for cond in TEST_CONDITIONS:
        for seed in (1, 2, 3):
            rows = evaluate_scarcity(seed, cond)
            succ = np.mean([r["success"] for r in rows]) * 100
            lat_mean = np.mean([r["latency"] for r in rows if r["latency"] > 0]) / 1000.0
            lat_p95 = np.percentile([r["latency"] for r in rows if r["latency"] > 0], 95) / 1000.0
            cost_mean = np.mean([r["cost"] for r in rows if r["cost"] > 0]) * 1e15
            psi_mean = np.mean([r["psi_t"] for r in rows])
            sub_mean = np.mean([r["subchannels"] for r in rows])
            print(f"{cond['key']} Seed {seed}: Succ={succ:.2f}%, LatMean={lat_mean:.3f}s, LatP95={lat_p95:.3f}s, CostMean={cost_mean:.3f}, PsiMean={psi_mean:.2f}, SubchannelsMean={sub_mean:.2f}", flush=True)
