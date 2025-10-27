"""
Metrics logging and analysis utilities.

UPDATED FOR MUTATION: Changed flow_mods_count → mutation_count
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class MetricsLogger:
    """Logs per-episode metrics to CSV file."""
    
    def __init__(self, csv_path: Path):
        """
        Args:
            csv_path: Path to output CSV file
        """
        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None
        self.headers_written = False
        
        # Open file
        self.csv_file = open(csv_path, 'w', newline='')
        
    def log_episode(self, metrics: Dict):
        """Log metrics for one episode."""
        if not self.headers_written:
            # Write header on first call
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=metrics.keys())
            self.csv_writer.writeheader()
            self.headers_written = True
        
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()  # Ensure data is written
    
    def close(self):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()


def compute_windowed_metrics(all_metrics: List[Dict], window_size: int) -> List[Dict]:
    """
    Compute windowed aggregate metrics.
    
    Args:
        all_metrics: List of per-episode metrics
        window_size: Size of sliding window
        
    Returns:
        List of windowed metric dicts
    """
    n_episodes = len(all_metrics)
    windowed = []
    
    for window_start in range(0, n_episodes, window_size):
        window_end = min(window_start + window_size, n_episodes)
        window_metrics = all_metrics[window_start:window_end]
        
        if not window_metrics:
            continue
        
        # Extract values
        tsh_values = [m['tsh'] for m in window_metrics]
        coverage_values = [m['coverage'] for m in window_metrics]
        entropy_values = [m['policy_entropy'] for m in window_metrics]
        
        # UPDATED: Use mutation_count instead of flow_mods_count
        mutation_values = [m.get('mutation_count', m.get('flow_mods_count', 0)) 
                          for m in window_metrics]
        
        qos_values = [m['qos_penalty_proxy_ms'] for m in window_metrics]
        
        windowed_dict = {
            'window_start': window_start,
            'window_end': window_end,
            'n_episodes': len(window_metrics),
            'tsh_mean': float(np.mean(tsh_values)),
            'tsh_median': float(np.median(tsh_values)),
            'tsh_std': float(np.std(tsh_values)),
            'tsh_auc': float(np.sum(tsh_values)),  # Area under curve for this window
            'coverage_mean': float(np.mean(coverage_values)),
            'coverage_final': float(coverage_values[-1]),
            'policy_entropy_mean': float(np.mean(entropy_values)),
            'mutation_mean': float(np.mean(mutation_values)),  # RENAMED
            'qos_penalty_mean': float(np.mean(qos_values))
        }
        
        windowed.append(windowed_dict)
    
    return windowed


def compute_adaptation_lag(
    all_metrics: List[Dict],
    switch_points: List[int],
    window_size: int,
    epsilon: float
) -> List[int]:
    """
    Compute adaptation lag after each attacker switch.
    
    Adaptation lag = number of episodes until defender's windowed TSH
    returns to within epsilon of pre-switch baseline.
    
    Args:
        all_metrics: List of per-episode metrics
        switch_points: List of episode numbers where attacker switched
        window_size: Window size for computing baseline
        epsilon: Threshold for considering TSH "recovered" (e.g., 0.10 for 10%)
        
    Returns:
        List of adaptation lags (one per switch)
    """
    adaptation_lags = []
    
    for switch_ep in switch_points:
        # Compute baseline: mean TSH in window before switch
        baseline_start = max(0, switch_ep - window_size)
        baseline_end = switch_ep
        
        if baseline_start >= baseline_end:
            continue  # Not enough history
        
        baseline_metrics = all_metrics[baseline_start:baseline_end]
        baseline_tsh = np.mean([m['tsh'] for m in baseline_metrics])
        
        # Find recovery point starting from switch_ep + 1
        recovery_ep = None
        
        for ep in range(switch_ep + 1, len(all_metrics)):
            # Compute windowed TSH using ONLY post-switch episodes
            window_start = max(switch_ep + 1, ep - window_size + 1)
            window_metrics = all_metrics[window_start:ep+1]
            
            if len(window_metrics) < window_size // 2:
                continue  # Need minimum window size
            
            windowed_tsh = np.mean([m['tsh'] for m in window_metrics])
            
            # Check if within epsilon of baseline
            if abs(windowed_tsh - baseline_tsh) <= epsilon * baseline_tsh:
                recovery_ep = ep
                break
        
        if recovery_ep is not None:
            lag = recovery_ep - switch_ep
            adaptation_lags.append(lag)
        else:
            # Never recovered within experiment duration
            # Use remaining episodes as upper bound
            lag = len(all_metrics) - switch_ep
            adaptation_lags.append(lag)
    
    return adaptation_lags


def compute_success_criteria(
    aggregated: Dict,
    baseline_aggregated: Optional[Dict],
    config: Dict
) -> Dict:
    """
    Evaluate success criteria as defined in experiment prompt.
    
    Args:
        aggregated: Aggregated results for this config
        baseline_aggregated: Aggregated results for baseline-static (if available)
        config: Experiment configuration
        
    Returns:
        Dict with success flags and explanations
    """
    success = {}
    
    # Criterion 1: TSH reduction
    if baseline_aggregated is not None:
        tsh_reduction_pct = (
            (baseline_aggregated['windowed_tsh_auc_median'] - 
             aggregated['windowed_tsh_auc_median']) / 
            baseline_aggregated['windowed_tsh_auc_median']
        ) * 100
        
        success['tsh_reduction_pct'] = float(tsh_reduction_pct)
        success['meets_tsh_threshold'] = tsh_reduction_pct >= 20.0
    else:
        success['tsh_reduction_pct'] = None
        success['meets_tsh_threshold'] = None
    
    # Criterion 2: QoS cost acceptable
    qos_threshold_ms = 100.0
    success['mean_qos_ms'] = aggregated['mean_qos_proxy_ms']
    success['meets_qos_threshold'] = aggregated['mean_qos_proxy_ms'] <= qos_threshold_ms
    
    # Criterion 3: Mutations not excessive
    # UPDATED: Use mutation_count instead of flow_mods
    if baseline_aggregated is not None:
        mutation_ratio = (
            aggregated.get('mean_mutation_count', 0) / 
            (baseline_aggregated.get('mean_mutation_count', 0) + 1e-10)
        )
        success['mutation_ratio'] = float(mutation_ratio)
        success['meets_mutation_threshold'] = mutation_ratio <= 2.0
    else:
        success['mutation_ratio'] = None
        success['meets_mutation_threshold'] = None
    
    # Criterion 4: Adaptation lag sufficient
    if aggregated.get('adaptation_lag_median') is not None:
        min_lag_threshold = 200  # episodes
        success['median_adaptation_lag'] = aggregated['adaptation_lag_median']
        success['meets_adaptation_lag_threshold'] = (
            aggregated['adaptation_lag_median'] >= min_lag_threshold
        )
    else:
        success['median_adaptation_lag'] = None
        success['meets_adaptation_lag_threshold'] = None
    
    # Overall success: all criteria met
    criteria_results = [
        success['meets_tsh_threshold'],
        success['meets_qos_threshold'],
        success.get('meets_mutation_threshold'),
        success['meets_adaptation_lag_threshold']
    ]
    
    # Filter out None values
    valid_criteria = [c for c in criteria_results if c is not None]
    success['overall_success'] = all(valid_criteria) if valid_criteria else None
    success['criteria_met'] = sum(valid_criteria) if valid_criteria else 0
    success['criteria_total'] = len(valid_criteria)
    
    return success


def load_csv_metrics(csv_path: Path) -> List[Dict]:
    """Load per-episode metrics from CSV file."""
    metrics = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['episode', 'tsh', 'hits_count', 'discovered_hosts_count',
                       'discovered_hosts_cumulative',
                       'mutation_count', 'mutation_flag',  # UPDATED: mutation instead of flow_mods
                       'probe_budget', 'partition_id', 'seed']:
                if key in row:
                    row[key] = int(row[key])
            
            for key in ['coverage', 'policy_entropy', 'qos_penalty_proxy_ms',
                       'tsh_per_100_probes', 'address_entropy',  # ADDED: address_entropy
                       'mutation_saturation_active_partition']:  # UPDATED: mutation_saturation
                if key in row:
                    row[key] = float(row[key])
            
            metrics.append(row)
    
    return metrics