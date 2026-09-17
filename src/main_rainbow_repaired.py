import os
import json
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import pandas as pd
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from torch.distributions import Categorical
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

# Global Environment Configuration
NUM_EPSILON = 1


# ==========================================
# 0. 3GPP Mode 2 Sensing-Based SPS Resource Pool
# ==========================================
class SBSPSResourcePool:
    """
    3GPP TS 38.214 / TS 38.321 Mode 2 Sensing-Based Semi-Persistent Scheduling (SB-SPS) Resource Pool.
    Maintains energy sensing history across T0=20 slots, reservation counters, and calculates collision-free subchannels.
    """
    def __init__(self, k_max=11, sensing_window=20, rsrp_threshold_dbm=-95.0):
        self.k_max = k_max
        self.sensing_window = sensing_window
        self.rsrp_threshold_dbm = rsrp_threshold_dbm
        
        # Energy sensing memory grid: shape (sensing_window, k_max)
        self.sensing_history = deque(maxlen=sensing_window)
        for _ in range(sensing_window):
            self.sensing_history.append(np.full(k_max, -110.0))  # Noise floor baseline in dBm
            
        # Per-vehicle reservation states: {v_id: {'subchannel': int, 'counter': int}}
        self.vehicle_reservations = {}
        
    def step(self, active_vehicle_ids, current_slot=0):
        """
        Advances the SB-SPS state machine by one slot and returns K_t_av (number of available collision-free subchannels).
        """
        active_set = set(active_vehicle_ids)
        
        # Purge vehicles no longer active
        for v_id in list(self.vehicle_reservations.keys()):
            if v_id not in active_set:
                del self.vehicle_reservations[v_id]
                
        # Calculate subchannel sensing energy (average RSRP over sensing window T0)
        sensing_matrix = np.array(self.sensing_history)  # (T0, k_max)
        avg_sensed_energy = np.mean(sensing_matrix, axis=0)  # (k_max,)
        
        # Subchannel usage tracking for current slot
        subchannel_allocations = {k: [] for k in range(self.k_max)}
        
        for v_id in active_vehicle_ids:
            if v_id not in self.vehicle_reservations:
                # Initial allocation via sensing
                candidate_subchannels = [k for k in range(self.k_max) if avg_sensed_energy[k] <= self.rsrp_threshold_dbm]
                if not candidate_subchannels:
                    candidate_subchannels = list(range(self.k_max))
                # Select candidate with minimum sensed interference
                chosen_k = min(candidate_subchannels, key=lambda k: avg_sensed_energy[k] + np.random.uniform(-1, 1))
                res_counter = np.random.randint(5, 15)
                self.vehicle_reservations[v_id] = {'subchannel': chosen_k, 'counter': res_counter}
            else:
                res = self.vehicle_reservations[v_id]
                res['counter'] -= 1
                if res['counter'] <= 0:
                    # Reselection triggered
                    candidate_subchannels = [k for k in range(self.k_max) if avg_sensed_energy[k] <= self.rsrp_threshold_dbm]
                    if not candidate_subchannels:
                        candidate_subchannels = list(range(self.k_max))
                    chosen_k = min(candidate_subchannels, key=lambda k: avg_sensed_energy[k] + np.random.uniform(-1, 1))
                    res['subchannel'] = chosen_k
                    res['counter'] = np.random.randint(5, 15)
                    
            subchannel_allocations[self.vehicle_reservations[v_id]['subchannel']].append(v_id)
            
        # Update current slot energy sensing record
        slot_energy = np.full(self.k_max, -110.0)
        for k in range(self.k_max):
            num_users = len(subchannel_allocations[k])
            if num_users == 1:
                slot_energy[k] = -90.0  # Signal RSRP
            elif num_users > 1:
                slot_energy[k] = -90.0 + 10 * np.log10(num_users)  # Collision energy accumulation
                
        self.sensing_history.append(slot_energy)
        
        # Calculate collision-free available subchannels (K_t_av)
        collision_free_count = 0
        for k in range(self.k_max):
            if len(subchannel_allocations[k]) <= 1 and avg_sensed_energy[k] <= self.rsrp_threshold_dbm + 10.0:
                collision_free_count += 1
                
        # Ensure at least min subchannels
        k_av_t = max(1, collision_free_count)
        return k_av_t


# ==========================================
# 1. Original 1D Kinematic Environment
# ==========================================
class MobileREPCEnvironment:
    def __init__(self, num_workers=20, highway_length=100000, arrival_rate=0.2, min_workers_in_coverage=10):
        self.num_workers = num_workers
        self.highway_length = highway_length
        self.arrival_rate = arrival_rate
        self.min_workers_in_coverage = min_workers_in_coverage
        self.coverage_radius = 300  # 300m coverage radius
        self.task_size_range = list(range(1, 11))  # Default to full range
        
        # Sidelink Radio Resource Configuration
        self.total_bandwidth = 20e6  # 20 MHz
        self.subchannel_bandwidth = 1.8e6  # 1.8 MHz
        self.k_max = int(self.total_bandwidth / self.subchannel_bandwidth)  # K_max = 11
        
        # SB-SPS Mode 2 Pool
        self.sbsps_pool = SBSPSResourcePool(k_max=self.k_max)
        self.current_slot = 0
        
        # Speed parameters (truncated Gaussian)
        self.mean = 60  # km/h
        self.sd = 5    # km/h
        self.lower = self.mean - 3*self.sd  # 30 km/h
        self.upper = self.mean + 3*self.sd  # 90 km/h
        
        # Vehicle tracking
        self.all_vehicles = []
        self.active_workers = set()
        self.vehicle_id_counter = 0
        
        # Initialize client
        self.client = self.generate_new_vehicle(client=True)
        
        # Generate initial workers
        client_pos = self.client['position']
        for _ in range(num_workers):
            worker = self.generate_new_vehicle()
            worker['position'] = client_pos + np.random.uniform(-self.coverage_radius, self.coverage_radius)
            self.all_vehicles.append(worker)
        
        self.tx_power = 23
        self.noise_floor = -174 + 10*np.log10(20e6)
        self.update_coverage()
        if len(self.active_workers) < self.min_workers_in_coverage:
            workers_needed = self.min_workers_in_coverage - len(self.active_workers)
            for _ in range(workers_needed):
                new_worker = self.generate_new_vehicle()
                new_worker['position'] = self.client['position'] + np.random.uniform(-self.coverage_radius, self.coverage_radius)
                self.all_vehicles.append(new_worker)
            self.update_coverage()

    def set_parameters(self, min_workers_in_coverage=None, arrival_rate=None):
        """Update environment parameters"""
        if min_workers_in_coverage is not None:
            self.min_workers_in_coverage = min_workers_in_coverage
        if arrival_rate is not None:
            self.arrival_rate = arrival_rate

    def generate_new_vehicle(self, client=False):
        """Generate a new vehicle with random properties"""
        speed = stats.truncnorm.rvs(
            (self.lower - self.mean) / self.sd,
            (self.upper - self.mean) / self.sd,
            loc=self.mean,
            scale=self.sd
        ) / 3.6  # Convert to m/s
        
        vehicle = {
            'id': self.vehicle_id_counter,
            'position': 0,  # Start at beginning of highway
            'speed': speed,
            'compute': np.random.uniform(1e10, 5e10),
            'cost': np.random.uniform(1e-12, 1e-9),
            'transmission_delay': np.random.uniform(1e-6, 1e-3),
            'type': 'client' if client else 'worker'
        }
        self.vehicle_id_counter += 1
        return vehicle

    def update_vehicle_positions(self, time_step=1):
        """Update positions of all vehicles and manage arrivals/departures"""
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            current_workers_in_coverage = len(self.active_workers)
            if current_workers_in_coverage < self.min_workers_in_coverage:
                self.arrival_rate = min(
                    self.arrival_rate * (1 + (self.min_workers_in_coverage - current_workers_in_coverage)),
                    self.arrival_rate * 5
                )
            else:
                self.arrival_rate = self.arrival_rate
            
            num_new_vehicles = np.random.poisson(self.arrival_rate)
            for _ in range(num_new_vehicles):
                self.all_vehicles.append(self.generate_new_vehicle())
            
            for vehicle in self.all_vehicles[:]:
                vehicle['position'] += vehicle['speed'] * time_step
                if vehicle['position'] > self.highway_length:
                    self.all_vehicles.remove(vehicle)
                    if vehicle['id'] in self.active_workers:
                        self.active_workers.remove(vehicle['id'])
            
            self.client['position'] += self.client['speed'] * time_step
            if self.client['position'] > self.highway_length:
                self.client['position'] = 0
            
            self.update_coverage()
            if len(self.active_workers) >= self.min_workers_in_coverage:
                break
                
            if len(self.active_workers) < self.min_workers_in_coverage:
                workers_needed = self.min_workers_in_coverage - len(self.active_workers)
                for _ in range(workers_needed):
                    new_worker = self.generate_new_vehicle()
                    new_worker['position'] = self.client['position'] + np.random.uniform(-self.coverage_radius, self.coverage_radius)
                    self.all_vehicles.append(new_worker)
                self.update_coverage()
            
            iteration += 1

    def update_coverage(self):
        """Update which workers are in client's coverage"""
        client_pos = self.client['position']
        new_active_workers = set()
        
        for vehicle in self.all_vehicles:
            if vehicle['type'] == 'worker':
                distance = abs(vehicle['position'] - client_pos)
                if distance <= self.coverage_radius:
                    new_active_workers.add(vehicle['id'])
        
        self.active_workers = new_active_workers

    def reset(self):
        P = np.random.choice(self.task_size_range)
        size = 12 * P
        
        self.current_task = {
            'A_dim': (size, size),
            'B_dim': (size, size),
            'deadline': (0.5 + 1.5*(P-1)/10) * 1000
        }
        
        self.client = self.generate_new_vehicle(client=True)
        self.all_vehicles = []
        self.active_workers = set()
        
        # Reset SB-SPS pool and slot index
        self.current_slot = 0
        self.sbsps_pool = SBSPSResourcePool(k_max=self.k_max)
        
        client_pos = self.client['position']
        for _ in range(20):
            worker = self.generate_new_vehicle()
            worker['position'] = client_pos + np.random.uniform(-self.coverage_radius, self.coverage_radius)
            self.all_vehicles.append(worker)
        
        self.update_coverage()
        if len(self.active_workers) < self.min_workers_in_coverage:
            workers_needed = self.min_workers_in_coverage - len(self.active_workers)
            for _ in range(workers_needed):
                new_worker = self.generate_new_vehicle()
                new_worker['position'] = self.client['position'] + np.random.uniform(-self.coverage_radius, self.coverage_radius)
                self.all_vehicles.append(new_worker)
            self.update_coverage()
        
        return self._get_state()

    def _get_state(self):
        state = [
            self.current_task['A_dim'][0]/240,
            self.current_task['deadline']/1850,
            self.min_workers_in_coverage/100,
            self.arrival_rate/2.0
        ]
        
        workers_in_coverage = [v for v in self.all_vehicles if v['id'] in self.active_workers]
        client_pos = self.client['position']
        workers_in_coverage.sort(key=lambda w: abs(w['position'] - client_pos))
        
        for w in workers_in_coverage:
            distance = abs(w['position'] - client_pos)
            rel_speed = (w['speed'] - self.client['speed']) + 1e-5
            time_to_exit = (self.coverage_radius - distance) / rel_speed
            
            state.extend([
                distance/self.coverage_radius,
                rel_speed/33.33,
                min(time_to_exit/10.0, 1.0),
                w['compute']/5e10,
                w['cost']/1e-9
            ])
        
        max_workers = max(100, self.min_workers_in_coverage * 2)
        padding_size = (max_workers - len(workers_in_coverage)) * 5
        state.extend([0] * padding_size)
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        l, m, n, epsilon = action['params']
        selected_worker_ids = action['workers']
        size = self.current_task['A_dim'][0]
        
        if not (size % l == 0 and size % m == 0 and size % n == 0):
            valid_divisors = [i for i in range(1, min(size + 1, 6)) if size % i == 0]
            if not valid_divisors:
                valid_divisors = [1]
            
            target_k = l*m*n + l - 1
            best_l = min(valid_divisors, key=lambda x: abs(x - l))
            best_m = min(valid_divisors, key=lambda x: abs(x - m))
            best_n = min(valid_divisors, key=lambda x: abs(x - n))
            
            current_k = best_l * best_m * best_n + best_l - 1
            if current_k > target_k:
                for divisor in sorted(valid_divisors, reverse=True):
                    if size % divisor == 0 and divisor * best_m * best_n + divisor - 1 <= target_k:
                        best_l = divisor
                        break
            l, m, n = best_l, best_m, best_n
        
        self.update_vehicle_positions()
        self.update_coverage()
        
        # Advance simulation and compute available collision-free subchannels via SB-SPS
        active_sv_ids = list(self.active_workers)
        k_av_t = self.sbsps_pool.step(active_sv_ids, self.current_slot)
        self.current_slot += 1
        
        available_subchannels = max(1, k_av_t)
        tx_power_per_subchannel = self.tx_power - 10 * np.log10(self.k_max)
        
        valid_workers = []
        for worker_id in selected_worker_ids:
            worker = next((w for w in self.all_vehicles if w['id'] == worker_id), None)
            if worker and worker['id'] in self.active_workers:
                valid_workers.append(worker)
        
        worker_scores = []
        for worker in valid_workers:
            client_pos = self.client['position']
            worker_pos = worker['position']
            distance = abs(worker_pos - client_pos)
            
            wavelength = 0.125
            free_space_loss = 20 * np.log10(4 * np.pi * max(distance, 1e-3) / wavelength)
            urban_loss = 10 * np.log10(max(distance, 1e-3)/10)
            total_path_loss = free_space_loss + urban_loss
            
            received_power = tx_power_per_subchannel - total_path_loss
            noise_power = 10 ** (self.noise_floor / 10)
            sinr = 10 ** (received_power / 10) / noise_power
            spectral_efficiency = np.log2(1 + sinr)
            
            rel_speed = (worker['speed'] - self.client['speed']) + 1e-5
            time_to_exit = (self.coverage_radius - distance) / rel_speed
            distance_factor = 1 - (distance / self.coverage_radius)
            
            F = (0.4 * (worker['compute'] / 5e10) + 
                 0.3 * min(time_to_exit / 10.0, 1.0) + 
                 0.2 * distance_factor + 
                 0.1 * min(spectral_efficiency / 6.0, 1.0))
            worker_scores.append((worker, F))
        
        sorted_workers = sorted(worker_scores, key=lambda x: x[1], reverse=True)
        k_reduced = max(1, l*m*n + l - 1 - int(epsilon*l*m))
        if k_reduced > available_subchannels:
            k_reduced = available_subchannels
        
        final_workers = [w[0] for w in sorted_workers[:k_reduced]]
        if len(final_workers) < k_reduced:
            return (0, -1.0, True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        distances = [abs(w['position'] - self.client['position']) for w in final_workers]
        total_elements = size * size
        encoding_time_ms = (total_elements * 4 / 1e9) * 1000 * 1.5
        
        sinr_values = []
        for i, distance in enumerate(distances):
            f_ghz = 5.9
            h_bs, h_ms = 1.5, 1.5
            d_bp = 4 * h_bs * h_ms * f_ghz / 0.3
            dist_val = max(distance, 1.0)
            
            if dist_val <= d_bp:
                pl_los = 28.0 + 22 * np.log10(dist_val) + 20 * np.log10(f_ghz)
            else:
                pl_los = 28.0 + 20 * np.log10(dist_val) + 20 * np.log10(f_ghz) - 9 * np.log10((d_bp**2 + (h_bs - h_ms)**2))
            
            shadowing = np.random.normal(0, 3)
            excess_shadowing = np.random.normal(0, getattr(self, 'sigma_sh', 0.0)) if getattr(self, 'sigma_sh', 0.0) > 0 else 0.0
            total_path_loss = pl_los + shadowing + getattr(self, 'delta_ex', 0.0) + excess_shadowing
            received_power = tx_power_per_subchannel - total_path_loss
            
            interference = 0
            for j, other_distance in enumerate(distances):
                if i != j:
                    o_dist = max(other_distance, 1.0)
                    if o_dist <= d_bp:
                        int_pl = 28.0 + 22 * np.log10(o_dist) + 20 * np.log10(f_ghz)
                    else:
                        int_pl = 28.0 + 20 * np.log10(o_dist) + 20 * np.log10(f_ghz) - 9 * np.log10((d_bp**2 + (h_bs - h_ms)**2))
                    int_shadow = np.random.normal(0, 3)
                    int_excess_shadow = np.random.normal(0, getattr(self, 'sigma_sh', 0.0)) if getattr(self, 'sigma_sh', 0.0) > 0 else 0.0
                    interference += 10 ** ((tx_power_per_subchannel - (int_pl + int_shadow + getattr(self, 'delta_ex', 0.0) + int_excess_shadow)) / 10)
            
            noise_power = 10 ** (self.noise_floor / 10)
            sinr = 10 ** (received_power / 10) / (interference + noise_power)
            sinr_values.append(sinr)
        
        avg_sinr = np.mean(sinr_values)
        spectral_efficiency = np.log2(1 + avg_sinr)
        avg_distance = np.mean(distances) if distances else 150
        distance_efficiency = 1.0 if avg_distance <= 50 else (0.5 if avg_distance <= 150 else 0.2)
        
        subchannel_bandwidth = 1.8e6
        worker_data_rate = subchannel_bandwidth * spectral_efficiency * distance_efficiency
        
        bits_per_worker = (total_elements / (l * m)) * 32
        uplink_time_ms = (bits_per_worker / worker_data_rate) * 1000 * 1.2
        
        ops_per_worker = (size/l) * (size/m) * (size/n)
        total_ops = ops_per_worker * k_reduced
        avg_compute_power = np.mean([w['compute'] for w in final_workers])
        compute_time_ms = (total_ops / (avg_compute_power * 0.3)) * 1000
        
        result_elements = (size/m) * (size/n)
        bits_per_result = result_elements * 32
        downlink_time_ms = (bits_per_result / worker_data_rate) * 1000 * 1.2
        
        logged = np.log2(k_reduced)
        C_1 = 0.67
        Decode_complexity = k_reduced * logged**2
        decoding_time_ms = (result_elements * C_1 * Decode_complexity / 1e9) * 1000 * 1.5
        
        total_latency_ms = encoding_time_ms + uplink_time_ms + compute_time_ms + downlink_time_ms + decoding_time_ms
        computation_cost = sum((total_ops)/w['compute'] * w['cost'] for w in final_workers)
        
        deadline = self.current_task['deadline']
        success = 1 if total_latency_ms <= deadline else 0
        
        if success:
            time_saved_ratio = max(0.0, min(1.0, (deadline - total_latency_ms) / deadline))
            spectral_bonus = min(1.0, spectral_efficiency / 6.0)
            cost_efficiency = max(0.0, 1.0 - computation_cost * 5e4)
            reward = 0.5 * time_saved_ratio + 0.3 * spectral_bonus + 0.2 * cost_efficiency
        else:
            overrun_ratio = (total_latency_ms - deadline) / deadline
            reward = -0.5 - min(1.5, overrun_ratio)
        
        return (success, reward, False, total_latency_ms, encoding_time_ms, uplink_time_ms, downlink_time_ms, compute_time_ms, decoding_time_ms, computation_cost)

    def set_task_size_range(self, task_sizes):
        self.task_size_range = task_sizes


# ==========================================
# 2. LuST Trace Helpers & Utilities
# ==========================================
def generate_synthetic_lust_trace(filename="lust_fcd_sample.csv", num_timesteps=1500, num_vehicles=120, highway_only=True):
    """
    Generates a synthetic 2D LuST FCD mobility trace CSV focused strictly on Highway & Motorway corridors.
    Ensures high vehicle density (10-20 vehicles within coverage radius).
    """
    records = []
    np.random.seed(None)  # Dynamic unseeded generation for independent runs
    
    vehicles = {}
    for v_id in range(num_vehicles):
        if highway_only:
            speed = np.random.uniform(22.2, 36.1)  # 80 km/h to 130 km/h
            angle = float(np.random.choice([0.0, 90.0, 180.0, 270.0]))
            x = np.random.uniform(2000, 4000)
            y = np.random.uniform(2000, 4000)
        else:
            speed = np.random.uniform(8.3, 25.0)
            angle = np.random.uniform(0, 360)
            x = np.random.uniform(1000, 5000)
            y = np.random.uniform(1000, 5000)

        vehicles[f"veh_{v_id}"] = {
            'x': x,
            'y': y,
            'speed': speed,
            'angle': angle,
            'edge': f"motorway_A{np.random.choice([1, 3, 6])}"
        }
        
    for t in range(num_timesteps):
        for v_id, v in vehicles.items():
            rad = np.radians(v['angle'])
            vx = v['speed'] * np.sin(rad)
            vy = v['speed'] * np.cos(rad)
            
            v['x'] += vx * 1.0
            v['y'] += vy * 1.0
            
            # Wrap around highway corridor bounding box to maintain uniform high density
            if v['x'] > 4500: v['x'] = 1500
            elif v['x'] < 1500: v['x'] = 4500
            if v['y'] > 4500: v['y'] = 1500
            elif v['y'] < 1500: v['y'] = 4500
            
            if highway_only:
                v['angle'] += np.random.uniform(-0.5, 0.5)
                v['speed'] = float(np.clip(v['speed'] + np.random.uniform(-0.2, 0.2), 22.2, 36.1))
            else:
                v['angle'] += np.random.uniform(-5, 5)
                v['speed'] = float(np.clip(v['speed'] + np.random.uniform(-0.5, 0.5), 5.0, 30.0))
            
            records.append({
                'timestep': t,
                'vehicle_id': v_id,
                'x': v['x'],
                'y': v['y'],
                'speed': v['speed'],
                'angle': v['angle'],
                'edge': v['edge']
            })
            
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Generated LuST Highway 2D FCD trace: {filename} with {len(df)} records across {num_timesteps} timesteps.")
    return filename


def parse_fcd_xml(xml_file="lust_trace.xml", output_file="lust_fcd_sample.csv", highway_only=True):
    """
    Parses a SUMO FCD XML output file into a CSV or Parquet tabular trace, filtering for Highway/Motorway edges.
    Usage:
        sumo -c scenario/dua.actuated.sumocfg --fcd-output lust_trace.xml --fcd-output.attributes id,x,y,speed,angle,edge
        parse_fcd_xml('lust_trace.xml', 'lust_fcd_sample.csv', highway_only=True)
    """
    print(f"Parsing SUMO FCD XML file: {xml_file} (Highway Only: {highway_only})...")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    records = []
    for timestep_node in root.findall('timestep'):
        time = float(timestep_node.attrib['time'])
        for vehicle_node in timestep_node.findall('vehicle'):
            v_id = vehicle_node.attrib['id']
            x = float(vehicle_node.attrib['x'])
            y = float(vehicle_node.attrib['y'])
            speed = float(vehicle_node.attrib.get('speed', 0.0))
            angle = float(vehicle_node.attrib.get('angle', 0.0))
            edge = vehicle_node.attrib.get('edge', '')
            
            is_highway = (speed >= 19.4) or any(k in edge.lower() for k in ['motorway', 'primary', 'trunk', 'a1', 'a3', 'a6', 'expressway', 'autobahn'])
            if highway_only and not is_highway:
                continue
                
            records.append({
                'timestep': time,
                'vehicle_id': v_id,
                'x': x,
                'y': y,
                'speed': speed,
                'angle': angle,
                'edge': edge
            })
            
    df = pd.DataFrame(records)
    if output_file.endswith('.parquet'):
        df.to_parquet(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)
    print(f"Successfully exported {len(df)} Highway FCD entries to {output_file}.")
    return output_file


# ==========================================
# 3. LuST 2D Spatial Trace-Driven Environment
# ==========================================
class LuSTMobileREPCEnvironment:
    def __init__(self, trace_file="lust_fcd_sample.csv", coverage_radius=300,
                 min_workers_in_coverage=10, highway_only=True,
                 controlled_uv=True,
                 controlled_uv_manifest="controlled_uv_trajectory_manifest.json",
                 controlled_uv_episodes=350, controlled_uv_steps=100,
                 controlled_uv_min_servers=2, controlled_uv_schedule=None):
        self.trace_file = trace_file
        self.coverage_radius = coverage_radius
        self.min_workers_in_coverage = min_workers_in_coverage
        self.task_size_range = list(range(1, 11))
        self.highway_only = highway_only
        self.controlled_uv = bool(controlled_uv)
        self.controlled_uv_manifest_path = controlled_uv_manifest
        self.controlled_uv_episodes = int(controlled_uv_episodes)
        self.controlled_uv_steps = int(controlled_uv_steps)
        self.controlled_uv_min_servers = int(controlled_uv_min_servers)
        self.controlled_uv_schedule_path = controlled_uv_schedule
        
        # Sidelink Radio Resource Configuration
        self.total_bandwidth = 20e6  # 20 MHz
        self.subchannel_bandwidth = 1.8e6  # 1.8 MHz
        self.k_max = int(self.total_bandwidth / self.subchannel_bandwidth)  # K_max = 11
        
        # SB-SPS Mode 2 Pool
        self.sbsps_pool = SBSPSResourcePool(k_max=self.k_max)
        self.current_slot = 0
        
        self.tx_power = 23  # dBm
        self.noise_floor = -174 + 10 * np.log10(20e6)
        
        # Propagation shift parameters (for propagation-stress testing)
        self.delta_ex = 0.0
        self.sigma_sh = 0.0
        
        if not os.path.exists(trace_file):
            print(f"Trace file '{trace_file}' not found. Auto-generating synthetic LuST trace dataset...")
            generate_synthetic_lust_trace(trace_file, highway_only=self.highway_only)
            
        if trace_file.endswith('.parquet'):
            self.trace_df = pd.read_parquet(trace_file)
        else:
            self.trace_df = pd.read_csv(trace_file)
            
        if self.highway_only:
            if 'edge' in self.trace_df.columns:
                highway_mask = (self.trace_df['speed'] >= 19.4) | (self.trace_df['edge'].astype(str).str.contains('motorway|trunk|primary|A1|A3|A6|expressway', case=False, na=False))
                filtered_df = self.trace_df[highway_mask]
                if not filtered_df.empty:
                    self.trace_df = filtered_df
            else:
                filtered_df = self.trace_df[self.trace_df['speed'] >= 19.4]
                if not filtered_df.empty:
                    self.trace_df = filtered_df
            print(f"LuST Environment set to HIGHWAY & MOTORWAY ONLY ({len(self.trace_df)} records loaded).")
            
        self.timesteps = sorted(self.trace_df['timestep'].unique())
        self.current_step_idx = 0
        
        self.vehicle_registry = {}
        self.client_id = None
        self.active_workers = set()
        self.current_frame_vehicles = {}
        self.controlled_uv_source_id = None
        self.current_controlled_uv_segment = None
        self.controlled_uv_end_idx = None
        self.controlled_uv_episode_cursor = 0
        self.controlled_uv_segments = []
        if self.controlled_uv:
            if self.controlled_uv_schedule_path:
                self.controlled_uv_segments = self._load_controlled_uv_schedule(
                    self.controlled_uv_schedule_path
                )
            else:
                self.controlled_uv_segments = self._load_or_create_controlled_uv_manifest()

    def _load_controlled_uv_schedule(self, schedule_path):
        with open(schedule_path, 'r', encoding='utf-8') as stream:
            payload = json.load(stream)
        rows = payload.get('episodes', payload.get('segments', []))
        normalized = []
        for row in rows:
            item = dict(row)
            item['source_vehicle_id'] = str(
                item.get('source_vehicle_id', item.get('source_vehicle'))
            )
            item['start_timestep'] = float(
                item.get('start_timestep', item.get('start_time'))
            )
            normalized.append(item)
        valid = self._validate_controlled_uv_segments(normalized)
        if len(valid) != len(normalized) or not valid:
            raise RuntimeError(
                f"Controlled-UV schedule {schedule_path!r} contains invalid trace segments"
            )
        return valid

    def _eligible_controlled_uv_segments(self):
        """Find trace-faithful segments containing 100 transitions (101 frames)."""
        timestep_to_index = {value: index for index, value in enumerate(self.timesteps)}
        required_frames = self.controlled_uv_steps + 1
        coverage_counts = {}
        for timestep, frame in self.trace_df.groupby('timestep'):
            positions = frame[['x', 'y']].to_numpy(dtype=np.float64)
            squared_distances = np.sum(
                (positions[:, None, :] - positions[None, :, :]) ** 2, axis=2
            )
            counts = np.sum(squared_distances <= self.coverage_radius ** 2, axis=1) - 1
            time_index = timestep_to_index[timestep]
            for vehicle_id, count in zip(frame['vehicle_id'].astype(str), counts):
                coverage_counts.setdefault(
                    vehicle_id, np.full(len(self.timesteps), -1, dtype=np.int16)
                )[time_index] = int(count)
        segments = []
        for vehicle_id, rows in self.trace_df.groupby('vehicle_id'):
            vehicle_id = str(vehicle_id)
            indices = sorted({timestep_to_index[value] for value in rows['timestep'] if value in timestep_to_index})
            index_set = set(indices)
            for start_idx in indices[::10]:
                end_idx = start_idx + required_frames - 1
                if end_idx >= len(self.timesteps):
                    continue
                window_counts = coverage_counts[vehicle_id][start_idx:end_idx + 1]
                if (all(index in index_set for index in range(start_idx, end_idx + 1))
                        and np.min(window_counts) >= self.controlled_uv_min_servers):
                    segments.append({
                        'source_vehicle_id': vehicle_id,
                        'start_step_idx': int(start_idx),
                        'start_timestep': float(self.timesteps[start_idx]),
                        'end_timestep': float(self.timesteps[end_idx]),
                        'transitions': self.controlled_uv_steps,
                        'minimum_candidate_servers': int(np.min(window_counts)),
                        'mean_candidate_servers': float(np.mean(window_counts)),
                    })
        return segments

    def _validate_controlled_uv_segments(self, segments):
        timestep_to_index = {value: index for index, value in enumerate(self.timesteps)}
        valid = []
        for segment in segments:
            start_timestep = segment['start_timestep']
            if start_timestep not in timestep_to_index:
                continue
            start_idx = timestep_to_index[start_timestep]
            end_idx = start_idx + self.controlled_uv_steps
            if end_idx >= len(self.timesteps):
                continue
            vehicle_id = str(segment['source_vehicle_id'])
            rows = self.trace_df[self.trace_df['vehicle_id'].astype(str) == vehicle_id]
            available = set(rows['timestep'])
            if all(self.timesteps[index] in available for index in range(start_idx, end_idx + 1)):
                item = dict(segment)
                item['start_step_idx'] = int(start_idx)
                item['end_timestep'] = float(self.timesteps[end_idx])
                item['transitions'] = self.controlled_uv_steps
                valid.append(item)
        return valid

    def _load_or_create_controlled_uv_manifest(self):
        manifest_path = os.path.abspath(self.controlled_uv_manifest_path)
        segments = []
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as stream:
                payload = json.load(stream)
            if (payload.get('format_version') == 2
                    and payload.get('trace_file') == os.path.basename(self.trace_file)
                    and payload.get('minimum_candidate_servers') == self.controlled_uv_min_servers):
                segments = self._validate_controlled_uv_segments(payload.get('segments', []))
        if len(segments) < self.controlled_uv_episodes:
            candidates = self._eligible_controlled_uv_segments()
            if not candidates:
                raise RuntimeError('No continuous controlled-UV trajectory segments are available')
            rng = np.random.default_rng(246810)
            order = rng.permutation(len(candidates))
            segments = [candidates[index] for index in order[:min(len(order), self.controlled_uv_episodes)]]
            while len(segments) < self.controlled_uv_episodes:
                segments.extend(segments[:self.controlled_uv_episodes - len(segments)])
            payload = {
                'format_version': 2,
                'trace_file': os.path.basename(self.trace_file),
                'logical_uv_id': 'controlled_uv',
                'episodes': self.controlled_uv_episodes,
                'transitions_per_episode': self.controlled_uv_steps,
                'selection_seed': 246810,
                'minimum_candidate_servers': self.controlled_uv_min_servers,
                'segments': segments,
            }
            with open(manifest_path, 'w', encoding='utf-8') as stream:
                json.dump(payload, stream, indent=2)
        return segments[:self.controlled_uv_episodes]

    def rewind_controlled_uv(self):
        self.controlled_uv_episode_cursor = 0
        
    def _register_vehicle(self, v_id):
        if v_id not in self.vehicle_registry:
            self.vehicle_registry[v_id] = {
                'compute': np.random.uniform(1e10, 5e10),
                'cost': np.random.uniform(1e-12, 1e-9)
            }

    def set_bandwidth_mhz(self, bandwidth_mhz):
        self.current_bandwidth_mhz = float(bandwidth_mhz)
        self.total_bandwidth = self.current_bandwidth_mhz * 1e6
        self.k_max = max(1, int(self.total_bandwidth / self.subchannel_bandwidth))
        self.noise_floor = -174 + 10 * np.log10(self.total_bandwidth)
        self.sbsps_pool = SBSPSResourcePool(k_max=self.k_max)

    def reset(self):
        if self.controlled_uv:
            segment = self.controlled_uv_segments[
                self.controlled_uv_episode_cursor % len(self.controlled_uv_segments)
            ]
            self.current_controlled_uv_segment = dict(segment)
            self.controlled_uv_episode_cursor += 1
            self.current_step_idx = int(segment['start_step_idx'])
            self.controlled_uv_end_idx = self.current_step_idx + self.controlled_uv_steps
            self.controlled_uv_source_id = str(segment['source_vehicle_id'])
        else:
            max_start = max(1, len(self.timesteps) - 200)
            self.current_step_idx = np.random.randint(0, max_start)
            self.controlled_uv_end_idx = None
            self.controlled_uv_source_id = None
            self.current_controlled_uv_segment = None
        t = self.timesteps[self.current_step_idx]
        
        # Reset SB-SPS pool and slot index
        self.current_slot = 0
        self.sbsps_pool = SBSPSResourcePool(k_max=self.k_max)
        
        frame = self.trace_df[self.trace_df['timestep'] == t]
        if frame.empty:
            self.current_step_idx = 0
            t = self.timesteps[0]
            frame = self.trace_df[self.trace_df['timestep'] == t]
            
        self.client_id = 'controlled_uv' if self.controlled_uv else np.random.choice(frame['vehicle_id'].values)
        self._register_vehicle(self.client_id)
        
        P = np.random.choice(self.task_size_range)
        size = 12 * P
        self.current_task = {
            'A_dim': (size, size),
            'B_dim': (size, size),
            'deadline': (0.5 + 1.5 * (P - 1) / 10) * 1000
        }
        if self.current_controlled_uv_segment:
            scheduled_bandwidth = self.current_controlled_uv_segment.get(
                'B0_mhz', self.current_controlled_uv_segment.get('B0')
            )
            if scheduled_bandwidth is not None:
                self.set_bandwidth_mhz(float(scheduled_bandwidth))
            scheduled_size = self.current_controlled_uv_segment.get('task_size')
            if scheduled_size is not None:
                scheduled_size = int(scheduled_size)
                self.current_task = {
                    'A_dim': (scheduled_size, scheduled_size),
                    'B_dim': (scheduled_size, scheduled_size),
                    'deadline': (0.5 + 1.5 * (scheduled_size / 12 - 1) / 10) * 1000,
                }
        
        self.update_coverage()
        return self._get_state()

    def _compute_2d_time_to_exit(self, p_w, p_c, v_w, v_c, R):
        r = np.array([p_w[0] - p_c[0], p_w[1] - p_c[1]])
        v_rel = np.array([v_w[0] - v_c[0], v_w[1] - v_c[1]])
        
        v_rel_sq = np.dot(v_rel, v_rel)
        r_dist = np.linalg.norm(r)
        
        if v_rel_sq < 1e-6:
            return 999.0
            
        r_dot_v = np.dot(r, v_rel)
        discriminant = r_dot_v**2 - v_rel_sq * (r_dist**2 - R**2)
        
        if discriminant >= 0:
            t_exit = (-r_dot_v + np.sqrt(discriminant)) / v_rel_sq
            if t_exit > 0:
                return float(t_exit)
                
        rel_speed_scalar = np.linalg.norm(v_rel) + 1e-5
        return float((R - r_dist) / rel_speed_scalar)

    def update_coverage(self):
        t = self.timesteps[self.current_step_idx]
        frame = self.trace_df[self.trace_df['timestep'] == t]
        
        source_id = self.controlled_uv_source_id if self.controlled_uv else self.client_id
        client_row = frame[frame['vehicle_id'].astype(str) == str(source_id)]
        if client_row.empty:
            if self.controlled_uv:
                raise RuntimeError(
                    f"Controlled UV source {source_id!r} is missing at timestep {t}"
                )
            self.client_id = np.random.choice(frame['vehicle_id'].values)
            client_row = frame[frame['vehicle_id'] == self.client_id]
            
        c_x, c_y = client_row['x'].values[0], client_row['y'].values[0]
        c_speed = client_row['speed'].values[0]
        c_angle = client_row['angle'].values[0] if 'angle' in client_row.columns else 0.0
        c_rad = np.radians(c_angle)
        c_vx, c_vy = c_speed * np.sin(c_rad), c_speed * np.cos(c_rad)
        
        self.active_workers = set()
        self.current_frame_vehicles = {}
        
        for _, row in frame.iterrows():
            v_id = row['vehicle_id']
            if str(v_id) in {str(self.client_id), str(source_id)}:
                continue
            self._register_vehicle(v_id)
            
            w_x, w_y = row['x'], row['y']
            dist = np.sqrt((w_x - c_x)**2 + (w_y - c_y)**2)
            
            if dist <= self.coverage_radius:
                w_speed = row['speed']
                w_angle = row['angle'] if 'angle' in row else 0.0
                w_rad = np.radians(w_angle)
                w_vx, w_vy = w_speed * np.sin(w_rad), w_speed * np.cos(w_rad)
                
                time_to_exit = self._compute_2d_time_to_exit(
                    (w_x, w_y), (c_x, c_y), (w_vx, w_vy), (c_vx, c_vy), self.coverage_radius
                )
                
                self.active_workers.add(v_id)
                self.current_frame_vehicles[v_id] = {
                    'x': w_x,
                    'y': w_y,
                    'speed': w_speed,
                    'vx': w_vx,
                    'vy': w_vy,
                    'distance': dist,
                    'time_to_exit': time_to_exit,
                    'compute': self.vehicle_registry[v_id]['compute'],
                    'cost': self.vehicle_registry[v_id]['cost']
                }

    def _get_state(self):
        state = [
            self.current_task['A_dim'][0] / 240.0,
            self.current_task['deadline'] / 1850.0,
            len(self.active_workers) / 100.0,
            1.0
        ]
        
        sorted_workers = sorted(self.current_frame_vehicles.items(), key=lambda item: item[1]['distance'])
        
        for v_id, w in sorted_workers:
            rel_speed = w['speed'] + 1e-5
            time_to_exit = w['time_to_exit']
            state.extend([
                w['distance'] / self.coverage_radius,
                rel_speed / 33.33,
                min(max(time_to_exit, 0.0) / 10.0, 1.0),
                w['compute'] / 5e10,
                w['cost'] / 1e-9
            ])
            
        max_workers = 100
        padding_size = (max_workers - len(sorted_workers)) * 5
        state.extend([0.0] * padding_size)
        return np.array(state, dtype=np.float32)

    def step(self, action):
        l, m, n, epsilon = action['params']
        selected_worker_ids = action['workers']
        size = self.current_task['A_dim'][0]
        
        if not (size % l == 0 and size % m == 0 and size % n == 0):
            valid_divisors = [i for i in range(1, min(size + 1, 6)) if size % i == 0]
            if not valid_divisors:
                valid_divisors = [1]
            
            target_k = l*m*n + l - 1
            best_l = min(valid_divisors, key=lambda x: abs(x - l))
            best_m = min(valid_divisors, key=lambda x: abs(x - m))
            best_n = min(valid_divisors, key=lambda x: abs(x - n))
            
            current_k = best_l * best_m * best_n + best_l - 1
            if current_k > target_k:
                for divisor in sorted(valid_divisors, reverse=True):
                    if size % divisor == 0 and divisor * best_m * best_n + divisor - 1 <= target_k:
                        best_l = divisor
                        break
            l, m, n = best_l, best_m, best_n

        self.current_step_idx = min(self.current_step_idx + 1, len(self.timesteps) - 1)
        self.update_coverage()

        # Advance SB-SPS state machine and obtain empirical collision-free subchannels (K_t_av)
        active_sv_ids = list(self.active_workers)
        k_av_t = self.sbsps_pool.step(active_sv_ids, self.current_slot)
        self.current_slot += 1
        
        available_subchannels = max(1, k_av_t)
        tx_power_per_subchannel = self.tx_power - 10 * np.log10(self.k_max)

        valid_workers = []
        for v_id in selected_worker_ids:
            if v_id in self.active_workers and v_id in self.current_frame_vehicles:
                valid_workers.append((v_id, self.current_frame_vehicles[v_id]))

        worker_scores = []
        for v_id, worker in valid_workers:
            distance = worker['distance']
            wavelength = 0.125
            free_space_loss = 20 * np.log10(4 * np.pi * max(distance, 1e-3) / wavelength)
            urban_loss = 10 * np.log10(max(distance, 1e-3)/10)
            total_path_loss = free_space_loss + urban_loss
            
            received_power = tx_power_per_subchannel - total_path_loss
            noise_power = 10 ** (self.noise_floor / 10)
            sinr = 10 ** (received_power / 10) / noise_power
            spectral_efficiency = np.log2(1 + sinr)
            
            time_to_exit = worker['time_to_exit']
            distance_factor = 1 - (distance / self.coverage_radius)
            
            F = (0.4 * (worker['compute'] / 5e10) + 
                 0.3 * min(max(time_to_exit, 0) / 10.0, 1.0) + 
                 0.2 * distance_factor + 
                 0.1 * min(spectral_efficiency / 6.0, 1.0))
            worker_scores.append((v_id, worker, F))

        sorted_workers = sorted(worker_scores, key=lambda x: x[2], reverse=True)
        k_reduced = max(1, l*m*n + l - 1 - int(epsilon*l*m))
        if k_reduced > available_subchannels:
            k_reduced = available_subchannels
            
        final_workers = sorted_workers[:k_reduced]
        if len(final_workers) < k_reduced:
            return (0, -1.0, True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        distances = [w[1]['distance'] for w in final_workers]
        
        total_elements = size * size
        encoding_time_ms = (total_elements * 4 / 1e9) * 1000 * 1.5

        sinr_values = []
        for i, distance in enumerate(distances):
            f_ghz = 5.9
            h_bs, h_ms = 1.5, 1.5
            d_bp = 4 * h_bs * h_ms * f_ghz / 0.3
            dist_val = max(distance, 1.0)
            
            if dist_val <= d_bp:
                pl_los = 28.0 + 22 * np.log10(dist_val) + 20 * np.log10(f_ghz)
            else:
                pl_los = 28.0 + 20 * np.log10(dist_val) + 20 * np.log10(f_ghz) - 9 * np.log10((d_bp**2 + (h_bs - h_ms)**2))
                
            shadowing = np.random.normal(0, 3)
            excess_shadowing = np.random.normal(0, getattr(self, 'sigma_sh', 0.0)) if getattr(self, 'sigma_sh', 0.0) > 0 else 0.0
            total_path_loss = pl_los + shadowing + getattr(self, 'delta_ex', 0.0) + excess_shadowing
            received_power = tx_power_per_subchannel - total_path_loss
            
            interference = 0
            for j, other_distance in enumerate(distances):
                if i != j:
                    o_dist = max(other_distance, 1.0)
                    if o_dist <= d_bp:
                        int_pl = 28.0 + 22 * np.log10(o_dist) + 20 * np.log10(f_ghz)
                    else:
                        int_pl = 28.0 + 20 * np.log10(o_dist) + 20 * np.log10(f_ghz) - 9 * np.log10((d_bp**2 + (h_bs - h_ms)**2))
                    int_shadow = np.random.normal(0, 3)
                    int_excess_shadow = np.random.normal(0, getattr(self, 'sigma_sh', 0.0)) if getattr(self, 'sigma_sh', 0.0) > 0 else 0.0
                    interference += 10 ** ((tx_power_per_subchannel - (int_pl + int_shadow + getattr(self, 'delta_ex', 0.0) + int_excess_shadow)) / 10)
                    
            noise_power = 10 ** (self.noise_floor / 10)
            sinr = 10 ** (received_power / 10) / (interference + noise_power)
            sinr_values.append(sinr)

        avg_sinr = np.mean(sinr_values)
        spectral_efficiency = np.log2(1 + avg_sinr)
        
        avg_distance = np.mean(distances) if distances else 150
        distance_efficiency = 1.0 if avg_distance <= 50 else (0.5 if avg_distance <= 150 else 0.2)
        subchannel_bandwidth = 1.8e6
        worker_data_rate = subchannel_bandwidth * spectral_efficiency * distance_efficiency

        bits_per_worker = (total_elements / (l * m)) * 32
        uplink_time_ms = (bits_per_worker / worker_data_rate) * 1000 * 1.2

        ops_per_worker = (size/l) * (size/m) * (size/n)
        total_ops = ops_per_worker * k_reduced
        avg_compute_power = np.mean([w[1]['compute'] for w in final_workers])
        compute_time_ms = (total_ops / (avg_compute_power * 0.3)) * 1000

        result_elements = (size/m) * (size/n)
        bits_per_result = result_elements * 32
        downlink_time_ms = (bits_per_result / worker_data_rate) * 1000 * 1.2

        logged = np.log2(k_reduced)
        C_1 = 0.67
        Decode_complexity = k_reduced * logged**2
        decoding_time_ms = (result_elements * C_1 * Decode_complexity / 1e9) * 1000 * 1.5

        total_latency_ms = encoding_time_ms + uplink_time_ms + compute_time_ms + downlink_time_ms + decoding_time_ms
        computation_cost = sum((total_ops)/w[1]['compute'] * w[1]['cost'] for w in final_workers)

        deadline = self.current_task['deadline']
        success = 1 if total_latency_ms <= deadline else 0

        if success:
            time_saved_ratio = max(0.0, min(1.0, (deadline - total_latency_ms) / deadline))
            spectral_bonus = min(1.0, spectral_efficiency / 6.0)
            cost_efficiency = max(0.0, 1.0 - computation_cost * 5e4)
            reward = 0.5 * time_saved_ratio + 0.3 * spectral_bonus + 0.2 * cost_efficiency
        else:
            overrun_ratio = (total_latency_ms - deadline) / deadline
            reward = -0.5 - min(1.5, overrun_ratio)

        done = (self.current_step_idx >= len(self.timesteps) - 1)
        if self.controlled_uv_end_idx is not None:
            done = done or (self.current_step_idx >= self.controlled_uv_end_idx)
        return (success, reward, done, total_latency_ms, encoding_time_ms, uplink_time_ms, downlink_time_ms, compute_time_ms, decoding_time_ms, computation_cost)

    def set_task_size_range(self, task_sizes):
        self.task_size_range = task_sizes


# ==========================================
# 4. Rainbow DQN Components
# ==========================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.3):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    def __init__(self, input_size, action_size, num_atoms=121, v_min=-20, v_max=100):
        super(RainbowDQN, self).__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms)
        
        self.feature = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.advantage_hidden = NoisyLinear(128, 64)
        self.advantage = NoisyLinear(64, action_size * num_atoms)
        
        self.value_hidden = NoisyLinear(128, 64)
        self.value = NoisyLinear(64, num_atoms)
    
    def forward(self, x):
        batch_size = x.size(0)
        features = self.feature(x)
        
        advantage_hidden = F.relu(self.advantage_hidden(features))
        advantage = self.advantage(advantage_hidden).view(batch_size, self.action_size, self.num_atoms)
        
        value_hidden = F.relu(self.value_hidden(features))
        value = self.value(value_hidden).view(batch_size, 1, self.num_atoms)
        
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    def reset_noise(self):
        self.advantage_hidden.reset_noise()
        self.advantage.reset_noise()
        self.value_hidden.reset_noise()
        self.value.reset_noise()
    
    def act(self, state, action_mask=None):
        with torch.no_grad():
            device = next(self.parameters()).device
            state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_dist = self.forward(state)
            q_values = (q_dist * self.support.to(device)).sum(dim=2)
            if action_mask is not None:
                mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
                if mask.numel() != self.action_size:
                    raise ValueError("Action mask size does not match action space")
                if not mask.any():
                    raise ValueError("Action mask contains no feasible actions")
                q_values = q_values.masked_fill(~mask.unsqueeze(0), float('-inf'))
            return q_values.argmax(dim=1).item()


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
    
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.memory)]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta_by_frame(self.frame))
        weights /= weights.max()
        self.frame += 1
        
        batch = list(zip(*samples))
        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(np.array(batch[1]))
        rewards = torch.FloatTensor(np.array(batch[2]))
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(np.array(batch[4]))
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class RainbowAgent:
    def __init__(self, state_size, action_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        self.batch_size = 32
        self.gamma = 0.99  # Standard RL discount factor for long-horizon multi-step Q-learning
        self.tau = 0.005
        self.learning_rate = 0.0001
        self.update_frequency = 4
        self.target_update_frequency = 1000
        self.n_step = 3
        
        self.policy_net = RainbowDQN(state_size, action_size).to(device)
        self.target_net = RainbowDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = PrioritizedReplayBuffer(100000)
        self.frame = 1
    
    def act(self, state, action_mask=None):
        return self.policy_net.act(state, action_mask=action_mask)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        current_q_dist = self.policy_net(states)
        current_q_dist = current_q_dist[range(self.batch_size), actions]
        
        with torch.no_grad():
            next_q_dist = self.target_net(next_states)
            support = self.policy_net.support.to(self.device)
            next_actions = (next_q_dist * support).sum(dim=2).max(1)[1]
            next_q_dist = next_q_dist[range(self.batch_size), next_actions]
            target_q_dist = self._categorical_projection(next_q_dist, rewards, dones)
        
        loss = -(target_q_dist * current_q_dist.log()).sum(dim=1)
        weighted_loss = (loss * weights).mean()
        
        self.memory.update_priorities(indices, loss.detach().cpu().numpy() + 1e-6)
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        for param in self.policy_net.parameters():
            if torch.isnan(param.grad).any():
                print("NaN gradients detected!")
                param.grad = torch.nan_to_num(param.grad)
        
        self.optimizer.step()
        
        if self.frame % self.target_update_frequency == 0:
            self._update_target_network()
        
        self.frame += 1
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
    
    def _categorical_projection(self, next_q_dist, rewards, dones):
        batch_size = len(rewards)
        num_atoms = self.policy_net.num_atoms
        v_min = self.policy_net.v_min
        v_max = self.policy_net.v_max
        delta_z = (v_max - v_min) / (num_atoms - 1)
        support = self.policy_net.support.to(self.device)
        
        projected_dist = torch.zeros(batch_size, num_atoms, device=self.device)
        
        rewards_expanded = rewards.unsqueeze(1).expand(batch_size, num_atoms)
        dones_expanded = dones.unsqueeze(1).expand(batch_size, num_atoms)
        support_expanded = support.unsqueeze(0).expand(batch_size, num_atoms)
        
        tz = rewards_expanded + (1 - dones_expanded) * self.gamma * support_expanded
        tz = tz.clamp(v_min, v_max)
        
        b = (tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        l[(u > 0) & (l == u)] -= 1
        u[(l < (num_atoms - 1)) & (l == u)] += 1
        
        offset = (torch.arange(0, batch_size, device=self.device) * num_atoms).unsqueeze(1).expand(batch_size, num_atoms)
        
        l_idx = (l + offset).view(-1)
        u_idx = (u + offset).view(-1)
        
        l_val = (next_q_dist * (u.float() - b)).view(-1)
        u_val = (next_q_dist * (b - l.float())).view(-1)
        
        projected_dist_flat = projected_dist.view(-1)
        projected_dist_flat.index_add_(0, l_idx, l_val)
        projected_dist_flat.index_add_(0, u_idx, u_val)
        
        return projected_dist
    
    def _update_target_network(self):
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ==========================================
# 5. Worker Ranking & Training Infrastructure
# ==========================================
def rank_workers(env, available_workers):
    """
    Rank workers based on compute power, coverage duration, distance, and SINR.
    Supports both 1D MobileREPCEnvironment and 2D LuSTMobileREPCEnvironment.
    Returns list of (worker_id, rank) tuples sorted by rank (1 is best).
    """
    worker_scores = []
    is_lust = isinstance(env, LuSTMobileREPCEnvironment)
    
    for worker_id in available_workers:
        if is_lust:
            worker_info = env.current_frame_vehicles.get(worker_id)
            if not worker_info:
                continue
            distance = worker_info['distance']
            compute = worker_info['compute']
            speed = worker_info['speed']
            time_to_exit = worker_info['time_to_exit']
        else:
            worker = next((w for w in env.all_vehicles if w['id'] == worker_id), None)
            if not worker:
                continue
            client_pos = env.client['position']
            worker_pos = worker['position']
            distance = abs(worker_pos - client_pos)
            compute = worker['compute']
            speed = worker['speed']
            rel_speed = (speed - env.client['speed']) + 1e-5
            time_to_exit = (env.coverage_radius - distance) / rel_speed
            
        wavelength = 0.125
        free_space_loss = 20 * np.log10(4 * np.pi * max(distance, 1e-3) / wavelength)
        urban_loss = 10 * np.log10(max(distance, 1e-3)/10)
        total_path_loss = free_space_loss + urban_loss
        
        tx_power = env.tx_power - 10 * np.log10(11)
        received_power = tx_power - total_path_loss
        noise_power = 10 ** (env.noise_floor / 10)
        sinr = 10 ** (received_power / 10) / noise_power
        
        spectral_efficiency = np.log2(1 + sinr)
        distance_factor = 1 - (distance / env.coverage_radius)
        
        F = (0.4 * (compute / 5e10) + 
             0.3 * min(max(time_to_exit, 0) / 10.0, 1.0) + 
             0.2 * distance_factor + 
             0.1 * min(spectral_efficiency / 6.0, 1.0))
        
        worker_scores.append((worker_id, F))
    
    sorted_workers = sorted(worker_scores, key=lambda x: x[1], reverse=True)
    ranked_workers = [(w[0], i+1) for i, w in enumerate(sorted_workers)]
    return ranked_workers


def get_available_workers(env):
    """Get list of available worker IDs in client's coverage"""
    return list(env.active_workers)


def train_rainbow(env, agent, episodes=500, steps_per_episode=100, learn=True, verbose=True):
    metrics = {
        'episode': [],
        'reward': [],
        'success': [],
        'total_latency_ms': [],
        'encoding_delay_ms': [],
        'uplink_ms': [],
        'downlink_ms': [],
        'compute_ms': [],
        'decoding_ms': [],
        'cost': [],
        'k_reduced': [],
        'available_workers': [],
        'l': [],
        'm': [],
        'n': [],
        'epsilon_action': [],
        'worker_utilization': [],
        'avg_worker_distance': [],
        'avg_worker_speed': [],
        'task_size': [],
        'action_idx': [],
        'worker_strategy': [],
        'feasible_action_count': [],
        'infeasible_termination': [],
        'training_frame': []
    }
    
    for episode in range(episodes):
        # Fixed Task Difficulty Benchmark (P=5 -> 60x60 matrix task) for clean RL policy convergence
        env.set_task_size_range([5])

        state = env.reset()
        episode_reward = 0
        episode_success = 0
        episode_latency = 0
        episode_workers_used = set()
        episode_params = []
        
        for step in range(steps_per_episode):
            available_workers = get_available_workers(env)
            if not available_workers:
                for key in metrics:
                    if key == 'episode':
                        metrics[key].append(episode)
                    elif key == 'reward':
                        metrics[key].append(-1.0)
                    elif key == 'action_idx' or key == 'worker_strategy':
                        metrics[key].append(-1)
                    elif key == 'infeasible_termination':
                        metrics[key].append(1)
                    elif key == 'training_frame':
                        metrics[key].append(agent.frame)
                    else:
                        metrics[key].append(0)
                if learn:
                    agent.remember(state, 0, -1.0, state, True)
                    agent.replay()
                break

            # Obtain real-time empirical K_t_av from SB-SPS pool
            k_av_t = env.sbsps_pool.step(available_workers, env.current_slot)
            available_subchannels = max(1, k_av_t)

            action_mask = feasible_action_mask(
                len(available_workers),
                size=env.current_task['A_dim'][0],
                available_subchannels=available_subchannels,
            )
            feasible_count = int(action_mask.sum())
            if feasible_count == 0:
                raise RuntimeError("No feasible action despite available workers")
            action_idx = agent.act(state, action_mask=action_mask)
            l, m, n, epsilon = constrained_action_selection(
                action_idx, 
                len(available_workers), 
                size=env.current_task['A_dim'][0],
                available_subchannels=available_subchannels
            )
            
            metrics['l'].append(l)
            metrics['m'].append(m)
            metrics['n'].append(n)
            metrics['epsilon_action'].append(epsilon)
            k_reduced = max(1, l*m*n + l - 1 - int(epsilon*l*m))
            metrics['k_reduced'].append(k_reduced)
            metrics['available_workers'].append(len(available_workers))
            worker_strategy = (action_idx // 10) % 5
            metrics['action_idx'].append(action_idx)
            metrics['worker_strategy'].append(worker_strategy)
            metrics['feasible_action_count'].append(feasible_count)
            metrics['infeasible_termination'].append(0)
            
            if len(available_workers) < k_reduced:
                metrics['infeasible_termination'][-1] = 1
                for key in metrics:
                    if key not in [
                        'l', 'm', 'n', 'epsilon_action', 'k_reduced',
                        'available_workers', 'action_idx', 'worker_strategy',
                        'feasible_action_count', 'infeasible_termination'
                    ]:
                        if key == 'episode':
                            metrics[key].append(episode)
                        elif key == 'reward':
                            metrics[key].append(-1.0)
                        elif key == 'training_frame':
                            metrics[key].append(agent.frame)
                        else:
                            metrics[key].append(0)
                if learn:
                    agent.remember(state, action_idx, -1.0, state, True)
                    agent.replay()
                break
            
            # End-to-End Worker Selection Strategy in Rainbow DQN action space:
            # worker_strategy: 0=Random, 1=Worst Compute, 2=Shortest Dwell, 3=Furthest Distance, 4=Top Optimal Rank
            if worker_strategy == 4:
                ranked_workers = rank_workers(env, available_workers)
                selected_workers = [w[0] for w in ranked_workers[:k_reduced]]
            elif worker_strategy == 0:
                # Random selection
                selected_workers = list(np.random.choice(available_workers, size=min(k_reduced, len(available_workers)), replace=False))
            elif worker_strategy == 1:
                # Select worst compute / worst channels
                ranked_workers = rank_workers(env, available_workers)
                selected_workers = [w[0] for w in reversed(ranked_workers[:k_reduced])]
            elif worker_strategy == 2:
                # Select workers leaving coverage soonest
                if isinstance(env, LuSTMobileREPCEnvironment):
                    workers_by_exit = sorted(available_workers, key=lambda w_id: env.current_frame_vehicles.get(w_id, {}).get('time_to_exit', 999))
                else:
                    workers_by_exit = sorted(available_workers, key=lambda w_id: abs(next((w['position'] for w in env.all_vehicles if w['id'] == w_id), 0) - env.client['position']))
                selected_workers = workers_by_exit[:k_reduced]
            else:
                # Select furthest distance workers
                if isinstance(env, LuSTMobileREPCEnvironment):
                    workers_by_dist = sorted(available_workers, key=lambda w_id: env.current_frame_vehicles.get(w_id, {}).get('distance', 0), reverse=True)
                else:
                    workers_by_dist = sorted(available_workers, key=lambda w_id: abs(next((w['position'] for w in env.all_vehicles if w['id'] == w_id), 0) - env.client['position']), reverse=True)
                selected_workers = workers_by_dist[:k_reduced]
            
            utilization = len(selected_workers) / len(available_workers) if available_workers else 0
            metrics['worker_utilization'].append(utilization)
            episode_workers_used.update(selected_workers)
            
            action = {'params': (l, m, n, epsilon), 'workers': selected_workers}
            (success, reward, done, total_latency_ms,
             encoding_delay_ms, uplink_ms, downlink_ms,
             compute_ms, decoding_ms, cost) = env.step(action)
            
            next_state = env._get_state()
            if learn:
                agent.remember(state, action_idx, reward, next_state, done)
                agent.replay()
            metrics['training_frame'].append(agent.frame)
            
            metrics['episode'].append(episode)
            metrics['reward'].append(reward)
            metrics['success'].append(success)
            metrics['total_latency_ms'].append(total_latency_ms)
            metrics['encoding_delay_ms'].append(encoding_delay_ms)
            metrics['uplink_ms'].append(uplink_ms)
            metrics['downlink_ms'].append(downlink_ms)
            metrics['compute_ms'].append(compute_ms)
            metrics['decoding_ms'].append(decoding_ms)
            metrics['cost'].append(cost)
            metrics['task_size'].append(env.current_task['A_dim'][0])
            
            selected_distances = []
            selected_speeds = []
            for worker_id in selected_workers:
                if isinstance(env, LuSTMobileREPCEnvironment):
                    w_info = env.current_frame_vehicles.get(worker_id)
                    if w_info:
                        selected_distances.append(w_info['distance'])
                        selected_speeds.append(w_info['speed'])
                else:
                    worker = next((w for w in env.all_vehicles if w['id'] == worker_id), None)
                    if worker:
                        client_pos = env.client['position']
                        worker_pos = worker['position']
                        selected_distances.append(abs(worker_pos - client_pos))
                        selected_speeds.append(worker['speed'])
            
            metrics['avg_worker_distance'].append(np.mean(selected_distances) if selected_distances else 0)
            metrics['avg_worker_speed'].append(np.mean(selected_speeds) if selected_speeds else 0)
            
            episode_reward += reward
            episode_success += success
            episode_latency += total_latency_ms
            episode_params.append((l, m, n, epsilon))
            
            state = next_state
            
            if done:
                break
        
        if verbose and (episode + 1) % 10 == 0:
            start_idx = max(0, len(metrics['episode']) - 10*steps_per_episode)
            avg_reward = np.mean(metrics['reward'][start_idx:])
            avg_success = np.mean(metrics['success'][start_idx:])
            valid_latencies = [x for x in metrics['total_latency_ms'][start_idx:] if x > 0]
            avg_latency = np.mean(valid_latencies) if valid_latencies else 0
            avg_workers = np.mean(metrics['available_workers'][start_idx:])
            avg_k_reduced = np.mean(metrics['k_reduced'][start_idx:])
            avg_utilization = np.mean(metrics['worker_utilization'][start_idx:])
            print(f"Ep {episode}: Reward={avg_reward:.2f}, Success={avg_success:.2f}, "
                  f"Latency={avg_latency:.0f}ms, Workers={avg_workers:.1f}, "
                  f"k_reduced={avg_k_reduced:.1f}, Utilization={avg_utilization:.2f}")
    
    return metrics


def evaluate_rainbow(env, agent, episodes=20, steps_per_episode=100):
    """Run a greedy, frozen evaluation with dropout and NoisyNet noise disabled."""
    policy_was_training = agent.policy_net.training
    target_was_training = agent.target_net.training
    agent.policy_net.eval()
    agent.target_net.eval()
    try:
        return train_rainbow(
            env,
            agent,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            learn=False,
            verbose=False,
        )
    finally:
        agent.policy_net.train(policy_was_training)
        agent.target_net.train(target_was_training)


def plot_training_metrics(metrics, save_path):
    rewards = np.asarray(metrics['reward'], dtype=np.float64)
    reward_min = np.min(rewards)
    reward_max = np.max(rewards)
    if reward_max > reward_min:
        normalized_rewards = (rewards - reward_min) / (reward_max - reward_min)
    else:
        normalized_rewards = np.ones_like(rewards)

    df_metrics = pd.DataFrame({
        'episode': metrics['episode'],
        'reward': normalized_rewards,
        'success': metrics['success']
    })
    ep_summary = df_metrics.groupby('episode').mean().reset_index()
    
    # Compute episode-level rolling statistics over 50 episodes.
    window_size = 50
    ep_summary['reward_rolling'] = ep_summary['reward'].rolling(window=window_size, min_periods=1).mean()
    rolling_std = ep_summary['reward'].rolling(window=window_size, min_periods=2).std().fillna(0.0)
    rolling_count = ep_summary['reward'].rolling(window=window_size, min_periods=1).count()
    reward_ci = 1.96 * rolling_std / np.sqrt(rolling_count)
    reward_ci_lower = np.clip(ep_summary['reward_rolling'] - reward_ci, 0.0, 1.0)
    reward_ci_upper = np.clip(ep_summary['reward_rolling'] + reward_ci, 0.0, 1.0)
    ep_summary['success_rolling'] = ep_summary['success'].rolling(window=window_size, min_periods=1).mean()
    
    # Plot Reward with Rolling Average (x-axis: Episode 0 to 350)
    plt.figure(figsize=(10, 6))
    plt.plot(ep_summary['episode'], ep_summary['reward'], color='skyblue', alpha=0.35, label='Normalized Episodic Mean Reward')
    plt.fill_between(
        ep_summary['episode'], reward_ci_lower, reward_ci_upper,
        color='cornflowerblue', alpha=0.25, label='95% CI'
    )
    plt.plot(ep_summary['episode'], ep_summary['reward_rolling'], color='darkblue', linewidth=2.5, label=f'Rolling Avg (Window={window_size} episodes)')
    plt.title(f'Normalized Training Reward (Rolling Average, Window = {window_size} Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Normalized Reward')
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.savefig(save_path.replace('.png', '_reward.png'))
    plt.savefig(save_path.replace('.png', '_rolling_reward.png'))
    plt.close()
    
    # Plot Success Rate with Rolling Average
    plt.figure(figsize=(10, 6))
    plt.plot(ep_summary['episode'], ep_summary['success'], color='lightgreen', alpha=0.35, label='Raw Episodic Success Rate')
    plt.plot(ep_summary['episode'], ep_summary['success_rolling'], color='darkgreen', linewidth=2.5, label=f'Rolling Avg (Window={window_size} episodes)')
    plt.title(f'Training Success Rate (Rolling Average, Window = {window_size} Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.savefig(save_path.replace('.png', '_success.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['encoding_delay_ms'], label='Encoding')
    plt.plot(metrics['episode'], metrics['uplink_ms'], label='Uplink')
    plt.plot(metrics['episode'], metrics['downlink_ms'], label='Downlink')
    plt.plot(metrics['episode'], metrics['decoding_ms'], label='Decoding')
    plt.title('Communication Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.savefig(save_path.replace('.png', '_comm_latency.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['compute_ms'], label='Computation')
    plt.title('Processing Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency (ms)')
    plt.legend()
    plt.savefig(save_path.replace('.png', '_proc_latency.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['total_latency_ms'])
    plt.title('Total Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency (ms)')
    plt.savefig(save_path.replace('.png', '_total_latency.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode'], metrics['cost'])
    plt.title('Computation Cost')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.savefig(save_path.replace('.png', '_cost.png'))
    plt.close()


# ==========================================
# 6. PSDD Model & Action Selection
# ==========================================
# ==========================================
# 6. Deterministic PSDD Model & Action Space Table
# ==========================================
DETERMINISTIC_ACTION_TABLE = []
for l_val in [1, 2, 3, 4, 6]:
    for m_val in [1, 2, 3, 4, 6]:
        for n_val in [1, 2, 3, 4]:
            for eps_val in [0.0, 0.25, 0.5, 0.75]:
                DETERMINISTIC_ACTION_TABLE.append((l_val, m_val, n_val, eps_val))

DETERMINISTIC_ACTION_TABLE = DETERMINISTIC_ACTION_TABLE[:100]

def action_parameters(action_idx, size):
    """Map one network output to one stable, task-compatible action."""
    l_val, m_val, n_val, eps_val = DETERMINISTIC_ACTION_TABLE[action_idx]
    valid_divisors = [i for i in range(1, min(size + 1, 7)) if size % i == 0]
    if not valid_divisors:
        valid_divisors = [1]
    l_val = min(valid_divisors, key=lambda x: abs(x - l_val))
    m_val = min(valid_divisors, key=lambda x: abs(x - m_val))
    n_val = min(valid_divisors, key=lambda x: abs(x - n_val))
    return l_val, m_val, n_val, eps_val


def feasible_action_mask(available_workers_count, size, available_subchannels=11):
    max_k = max(0, min(available_workers_count, available_subchannels))
    mask = np.zeros(len(DETERMINISTIC_ACTION_TABLE), dtype=bool)
    for action_idx in range(len(DETERMINISTIC_ACTION_TABLE)):
        l_val, m_val, n_val, eps_val = action_parameters(action_idx, size)
        k = l_val * m_val * n_val + l_val - 1 - int(eps_val * l_val * m_val)
        mask[action_idx] = 1 <= k <= max_k
    return mask


def constrained_action_selection(action_idx, available_workers_count, size, available_subchannels=11):
    """Return the stable action represented by action_idx; never modulo-remap it."""
    max_k = max(1, min(available_workers_count, available_subchannels))
    selected_action = action_parameters(action_idx, size)
    l_val, m_val, n_val, eps_val = selected_action
    k = l_val * m_val * n_val + l_val - 1 - int(eps_val * l_val * m_val)
    if not 1 <= k <= max_k:
        raise ValueError(f"Action {action_idx} is infeasible for max_k={max_k}")
    return selected_action


import json

def save_rainbow_checkpoint(agent, filename, metadata=None):
    checkpoint = {
        "policy_net_state_dict": agent.policy_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "state_size": agent.state_size,
        "action_size": agent.action_size,
        "training_frame": agent.frame,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, filename)
    print(f"Successfully saved Rainbow checkpoint to '{filename}'.")


def load_rainbow_checkpoint(filename, device=None, evaluation=True):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    agent = RainbowAgent(
        checkpoint["state_size"], checkpoint["action_size"], device=device
    )
    agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.frame = checkpoint.get("training_frame", 1)
    if evaluation:
        agent.policy_net.eval()
        agent.target_net.eval()
    return agent, checkpoint.get("metadata", {})


def save_training_results(metrics, filename="training_results.json"):
    serializable_metrics = {}
    for key, value in metrics.items():
        serializable_metrics[key] = [float(x) if isinstance(x, (np.floating, np.integer, float, int)) else x for x in value]
    with open(filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"Successfully saved training metrics to '{filename}' for later plotting.")


# ==========================================
# 7. Main Execution Entry Point
# ==========================================
if __name__ == '__main__':
    use_lust_environment = True
    
    if use_lust_environment:
        print("Initializing LuST 2D Trace-Driven Environment (Highway Only + 3GPP SB-SPS Mode 2)...")
        env = LuSTMobileREPCEnvironment(trace_file="lust_fcd_sample.csv", highway_only=True)
    else:
        print("Initializing 1D MobileREPC Environment (3GPP SB-SPS Mode 2)...")
        env = MobileREPCEnvironment()

    state_dim = len(env.reset())
    env.rewind_controlled_uv()
    action_dim = 100
    agent = RainbowAgent(state_dim, action_dim)
    metrics = train_rainbow(env, agent, episodes=350, steps_per_episode=100)
    save_training_results(metrics, 'training_results.json')
    plot_training_metrics(metrics, 'training_metrics.png')
