"""Comparable D3QN and deterministic heuristic baselines for FRODO-SUMO."""

import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import main


Transition = namedtuple("Transition", "state action reward next_state done")


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Identity(), nn.Linear(512, 256), nn.ReLU(),
            nn.Identity(), nn.Linear(256, 128), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_size)
        )

    def forward(self, state):
        feature = self.feature(state)
        value = self.value(feature)
        advantage = self.advantage(feature)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class D3QNAgent:
    def __init__(self, state_size, action_size, device=None, evaluation=False,
                 update_frequency=1, epsilon_decay_frames=15_000):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net = DuelingQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = deque(maxlen=100_000)
        self.batch_size = 32
        self.gamma = 0.99
        self.update_frequency = int(update_frequency)
        self.epsilon_decay_frames = int(epsilon_decay_frames)
        self.target_update_frequency = 1000
        self.frame = 1
        self.loss_history = []
        self.evaluation = bool(evaluation)
        if self.evaluation:
            self.policy_net.eval()
            self.target_net.eval()

    def epsilon(self):
        if self.evaluation:
            return 0.0
        fraction = min(1.0, self.frame / float(self.epsilon_decay_frames))
        return 1.0 + fraction * (0.05 - 1.0)

    def act(self, state, action_mask=None):
        mask = np.ones(self.action_size, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
        feasible = np.flatnonzero(mask)
        if feasible.size == 0:
            raise ValueError("Action mask contains no feasible actions")
        if not self.evaluation and random.random() < self.epsilon():
            return int(np.random.choice(feasible))
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_tensor.unsqueeze(0), float("-inf"))
            return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def replay(self):
        self.frame += 1
        if len(self.memory) < self.batch_size or self.frame % self.update_frequency:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.as_tensor(
            np.asarray([row.state for row in batch]),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.as_tensor(
            [row.action for row in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            [row.reward for row in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.asarray([row.next_state for row in batch]),
            dtype=torch.float32, device=self.device,
        )
        dones = torch.as_tensor(
            [row.done for row in batch], dtype=torch.float32, device=self.device
        )
        current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: policy selects; target evaluates.
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * next_values
        loss = F.smooth_l1_loss(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        if self.frame % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        loss_value = float(loss.detach().cpu())
        self.loss_history.append({"frame": int(self.frame), "loss": loss_value})
        return loss_value


def save_d3qn_checkpoint(agent, filename, metadata=None):
    torch.save({
        "algorithm": "D3QN",
        "policy_net_state_dict": agent.policy_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "training_frame": agent.frame,
        "update_frequency": agent.update_frequency,
        "epsilon_decay_frames": agent.epsilon_decay_frames,
        "loss_history": agent.loss_history,
        "metadata": metadata or {},
    }, filename)


def load_d3qn_checkpoint(filename, device=None, evaluation=True):
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(filename, map_location=selected_device, weights_only=False)
    agent = D3QNAgent(
        payload["state_size"], payload["action_size"],
        device=selected_device, evaluation=evaluation,
        update_frequency=payload.get("update_frequency", 4),
        epsilon_decay_frames=payload.get("epsilon_decay_frames", 30_000),
    )
    agent.policy_net.load_state_dict(payload["policy_net_state_dict"])
    agent.target_net.load_state_dict(payload["target_net_state_dict"])
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    agent.frame = payload.get("training_frame", 1)
    agent.loss_history = payload.get("loss_history", [])
    return agent, payload.get("metadata", {})


class GreedyHeuristicAgent:
    """Deterministic, model-based one-step controller with no learned parameters."""

    def __init__(self, environment):
        self.environment = environment
        self.frame = 1

    def act(self, _state, action_mask=None):
        mask = np.ones(100, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
        # Strategy 4 uses the environment's composite worker ranking. Restricting
        # to it avoids random worker selection and makes the baseline reproducible.
        candidates = [index for index in np.flatnonzero(mask) if (index // 10) % 5 == 4]
        if not candidates:
            candidates = list(np.flatnonzero(mask))
        workers = main.get_available_workers(self.environment)
        ranked_ids = [row[0] for row in main.rank_workers(self.environment, workers)]
        estimates = [
            (self._estimated_utility(index, workers, ranked_ids), index)
            for index in candidates
        ]
        return int(max(estimates)[1])

    def _estimated_utility(self, action_idx, workers, ranked_ids):
        env = self.environment
        size = int(env.current_task["A_dim"][0])
        available_subchannels = max(1, env.k_max)
        l_value, m_value, n_value, epsilon = main.constrained_action_selection(
            action_idx, len(workers), size, available_subchannels
        )
        k_value = max(
            1, l_value * m_value * n_value + l_value - 1
            - int(epsilon * l_value * m_value),
        )
        selected_ids = ranked_ids[:k_value]
        selected = [env.current_frame_vehicles[row] for row in selected_ids]
        if len(selected) < k_value:
            return -float("inf")
        distances = [row["distance"] for row in selected]
        tx_power_per_subchannel = env.tx_power - 10 * np.log10(env.k_max)
        sinr_values = []
        for index, distance in enumerate(distances):
            f_ghz = 5.9
            d_bp = 4 * 1.5 * 1.5 * f_ghz / 0.3
            distance = max(distance, 1.0)
            if distance <= d_bp:
                path_loss = 28.0 + 22 * np.log10(distance) + 20 * np.log10(f_ghz)
            else:
                path_loss = 28.0 + 20 * np.log10(distance) + 20 * np.log10(f_ghz) - 9 * np.log10(d_bp ** 2)
            signal = 10 ** ((tx_power_per_subchannel - path_loss) / 10)
            interference = 0.0
            for other_index, other_distance in enumerate(distances):
                if other_index == index:
                    continue
                other_distance = max(other_distance, 1.0)
                if other_distance <= d_bp:
                    other_loss = 28.0 + 22 * np.log10(other_distance) + 20 * np.log10(f_ghz)
                else:
                    other_loss = 28.0 + 20 * np.log10(other_distance) + 20 * np.log10(f_ghz) - 9 * np.log10(d_bp ** 2)
                interference += 10 ** ((tx_power_per_subchannel - other_loss) / 10)
            sinr_values.append(signal / (interference + 10 ** (env.noise_floor / 10)))
        spectral_efficiency = max(1e-9, np.log2(1 + np.mean(sinr_values)))
        mean_distance = float(np.mean(distances))
        distance_efficiency = 1.0 if mean_distance <= 50 else (0.5 if mean_distance <= 150 else 0.2)
        rate = 1.8e6 * spectral_efficiency * distance_efficiency
        elements = size * size
        encoding = elements * 4 / 1e9 * 1000 * 1.5
        uplink = elements / (l_value * m_value) * 32 / rate * 1000 * 1.2
        operations = (size / l_value) * (size / m_value) * (size / n_value) * k_value
        mean_compute = np.mean([row["compute"] for row in selected])
        compute = operations / (mean_compute * 0.3) * 1000
        result_elements = (size / m_value) * (size / n_value)
        downlink = result_elements * 32 / rate * 1000 * 1.2
        decoding = result_elements * 0.67 * k_value * math.log2(k_value) ** 2 / 1e9 * 1000 * 1.5
        latency = encoding + uplink + compute + downlink + decoding
        cost = sum(operations / row["compute"] * row["cost"] for row in selected)
        deadline = env.current_task["deadline"]
        if latency > deadline:
            return -0.5 - min(1.5, (latency - deadline) / deadline)
        time_saved = max(0.0, min(1.0, (deadline - latency) / deadline))
        spectral_bonus = min(1.0, spectral_efficiency / 6.0)
        cost_efficiency = max(0.0, 1.0 - cost * 5e4)
        return 0.5 * time_saved + 0.3 * spectral_bonus + 0.2 * cost_efficiency
