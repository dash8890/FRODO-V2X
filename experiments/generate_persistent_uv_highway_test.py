"""Generate a continuous persistent-UV highway trace and a 50-episode schedule.

The trace uses no post-hoc relative-velocity scaling.  One logical UV follows a
continuous 5,000-step trajectory.  Surrounding traffic changes from a
training-calibrated baseline to rush hour and then to a held-out motorway
configuration, while 100-step episode windows remain unchanged.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "controlled_uv_experiment_v2_curriculum" / "persistent_uv_highway_v5"
TRACE = OUTPUT / "persistent_uv_highway_trace.csv"
SCHEDULE = OUTPUT / "persistent_uv_highway_schedule.json"
STEPS = 5_000
DT = 1.0
PHASE_STARTS = (0, 1_400, 3_400)
RNG = np.random.default_rng(880031)


def smooth_target(step, baseline, rush, heldout, transition=150):
    if step < PHASE_STARTS[1]:
        return baseline
    if step < PHASE_STARTS[1] + transition:
        weight = (step - PHASE_STARTS[1]) / transition
        return baseline + weight * (rush - baseline)
    if step < PHASE_STARTS[2]:
        return rush
    if step < PHASE_STARTS[2] + transition:
        weight = (step - PHASE_STARTS[2]) / transition
        return rush + weight * (heldout - rush)
    return heldout


def phase_name(step):
    if step < PHASE_STARTS[1]:
        return "training_calibrated"
    if step < PHASE_STARTS[2]:
        return "rush_hour"
    return "heldout_motorway"


def generate():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    maximum_servers = 60
    relative_positions = RNG.uniform(-2_000, 2_000, maximum_servers)
    speed_offsets = RNG.normal(0.0, 3.9, maximum_servers)
    speeds = np.clip(29.3 + speed_offsets, 15.0, 40.0)
    lanes = RNG.integers(0, 3, maximum_servers)
    # The original training trace mixes both motorway carriageways.  Roughly
    # 45% opposing traffic reproduces its 27--28 m/s mean relative-speed level
    # without modifying any velocity after trace generation.
    directions = np.ones(maximum_servers)
    directions[RNG.choice(np.arange(36), size=15, replace=False)] = -1.0
    directions[RNG.choice(np.arange(36, maximum_servers), size=9, replace=False)] = -1.0
    uv_x = 0.0
    uv_speed = 29.3
    rows = []

    for step in range(STEPS + 1):
        phase = phase_name(step)
        uv_target = smooth_target(step, 29.3, 19.0, 25.0)
        speed_mean = smooth_target(step, 29.3, 18.0, 26.0)
        speed_sd = smooth_target(step, 3.9, 2.8, 5.5)
        active_target = int(round(smooth_target(step, 36, 58, 44)))
        lane_count = 3 if step < PHASE_STARTS[2] else 4
        edge = "motorway_training" if phase == "training_calibrated" else (
            "motorway_rush" if phase == "rush_hour" else "motorway_heldout_4lane"
        )

        uv_speed += 0.035 * (uv_target - uv_speed) + RNG.normal(0.0, 0.04)
        uv_speed = float(np.clip(uv_speed, 10.0, 38.0))
        if step:
            uv_x += uv_speed * DT

        rows.append({
            "timestep": step, "vehicle_id": "persistent_uv", "x": uv_x,
            "y": 3.7, "speed": uv_speed, "angle": 90.0, "edge": edge,
            "traffic_regime": phase,
        })

        desired = speed_mean + speed_offsets * (speed_sd / 3.9)
        speeds += 0.06 * (desired - speeds) + RNG.normal(0.0, 0.10, maximum_servers)
        speeds = np.clip(speeds, 7.0, 42.0)
        relative_positions += (directions * speeds - uv_speed) * DT
        wrapped_low = relative_positions < -2_000
        wrapped_high = relative_positions > 2_000
        relative_positions[wrapped_low] += 4_000
        relative_positions[wrapped_high] -= 4_000

        # The active population changes gradually with the traffic regime.  A
        # fixed ranking prevents stochastic on/off flicker at every timestep.
        active_ids = range(active_target)
        for index in active_ids:
            if index >= 36 and step < PHASE_STARTS[1]:
                continue
            if lane_count == 4 and index % 11 == 0:
                lanes[index] = 3
            rows.append({
                "timestep": step,
                "vehicle_id": f"server_{index:03d}",
                "x": uv_x + relative_positions[index],
                "y": float(lanes[index] * 3.7),
                "speed": float(speeds[index]),
                "angle": 90.0 if directions[index] > 0 else 270.0,
                "edge": edge,
                "traffic_regime": phase,
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(TRACE, index=False)

    def balanced_tasks(count, extra_values):
        values = list(range(12, 121, 12)) * (count // 10) + list(extra_values)
        RNG.shuffle(values)
        assert len(values) == count and np.isclose(np.mean(values), 66.0)
        return values

    phase_tasks = (
        balanced_tasks(14, [12, 48, 84, 120])
        + balanced_tasks(20, [])
        + balanced_tasks(16, [12, 36, 60, 72, 96, 120])
    )
    schedule_rows = []
    for episode in range(50):
        start = episode * 100
        schedule_rows.append({
            "episode": episode + 1,
            "segment_id": f"persistent_uv__step_{start:04d}",
            "source_vehicle": "persistent_uv",
            "source_vehicle_id": "persistent_uv",
            "start_time": float(start),
            "start_timestep": float(start),
            "start_step_idx": start,
            "segment_length": 100,
            "transitions": 100,
            "B0": 20.0,
            "B0_mhz": 20.0,
            "task_class": "balanced_full_training_range",
            "task_size": int(phase_tasks[episode]),
            "seed": None,
            "bank_split": "evaluation",
            "traffic_regime": phase_name(start),
            "persistent_uv": True,
        })
    digest = hashlib.sha256(TRACE.read_bytes()).hexdigest()
    payload = {
        "format_version": 1,
        "experiment": "persistent_uv_highway_v5",
        "trace_file": TRACE.name,
        "trace_sha256": digest,
        "logical_uv_id": "persistent_uv",
        "continuous_mobility": True,
        "transitions_per_episode": 100,
        "phase_boundaries": {
            "training_calibrated": [1, 14],
            "rush_hour": [15, 34],
            "heldout_motorway": [35, 50],
        },
        "velocity_scaling": False,
        "task_schedule": {
            "range": [12, 120],
            "increment": 12,
            "phase_mean": 66.0,
            "shared_across_checkpoint_seeds": True,
        },
        "episodes": schedule_rows,
    }
    SCHEDULE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {TRACE} ({len(frame):,} rows)")
    print(f"Saved {SCHEDULE}")


if __name__ == "__main__":
    generate()
