#!/usr/bin/env python3
"""
ID-HAM Divide & Conquer + Timing Variations Experiment Runner
Runs experiments as specified in the experiment prompt document.
"""

import argparse
import json
import yaml
import numpy as np
import torch
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys

from src.attacker import DivideConquerAttacker
from src.defender import IDHAMDefender
from src.environment import NetworkEnvironment
from src.metrics import MetricsLogger, compute_windowed_metrics, compute_adaptation_lag
from src.utils import set_seed, create_directories


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path}")
    return config


def run_single_episode(
    env: NetworkEnvironment,
    attacker: DivideConquerAttacker,
    defender: IDHAMDefender,
    episode: int,
    probe_budget: int
) -> Dict:
    """
    Run a single episode and return metrics.
    
    Returns dict with keys: tsh, hits_count, discovered_hosts_count, coverage,
    policy_entropy, flow_mods_count, mask_change_flag, qos_penalty_proxy_ms
    """
    # Reset episode state
    env.reset()
    defender.reset_episode()
    
    # Get current attacker partition
    partition_id = attacker.get_current_partition(episode)
    target_hosts = attacker.get_partition_targets(partition_id)
    
    # Attacker performs probing
    probe_results = []
    hits_count = 0
    discovered_hosts = set()
    
    for probe_idx in range(probe_budget):
        # Attacker selects target from current partition
        target = attacker.select_target(target_hosts, probe_idx)
        
        # Defender applies masking/defense
        is_masked, qos_cost = defender.apply_defense(target, probe_idx)
        
        # Check if probe hits a real host
        is_hit = env.probe(target, is_masked)
        
        if is_hit:
            hits_count += 1
            discovered_hosts.add(target)
            probe_results.append({
                'probe_idx': probe_idx,
                'target': target,
                'is_hit': True,
                'is_masked': is_masked
            })
    
    # Calculate Times of Scanning Hits (TSH)
    tsh = len([p for p in probe_results if p['is_hit']])
    
    # Coverage
    coverage = len(discovered_hosts) / env.n_real_hosts
    
    # Update defender (learning step if training)
    # MOVED HERE - before getting metrics so flow_mods are captured
    defender.update(hits_count, discovered_hosts, episode)
    
    # Get defender metrics (AFTER update so flow_mods are set)
    policy_entropy = defender.get_policy_entropy()
    flow_mods_count = defender.get_flow_mods_count()
    mask_change_flag = defender.did_mask_change()
    qos_penalty_proxy_ms = defender.get_qos_penalty()
    
    metrics = {
        'tsh': tsh,
        'hits_count': hits_count,
        'discovered_hosts_count': len(discovered_hosts),
        'coverage': coverage,
        'policy_entropy': policy_entropy,
        'flow_mods_count': flow_mods_count,
        'mask_change_flag': int(mask_change_flag),
        'qos_penalty_proxy_ms': qos_penalty_proxy_ms,
        'partition_id': partition_id,
        'probe_budget': probe_budget
    }
    
    return metrics

def run_experiment(config: Dict, seed: int, output_dir: Path) -> Dict:
    """
    Run a complete experiment for a single seed.
    
    Returns summary statistics for this seed.
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    config_id = config['config_id']
    n_hosts = config['n_hosts']
    episodes_train = config['episodes']['train']
    eval_window = config['episodes']['eval_window']
    probe_budget = config['probe_budget']
    
    print(f"[Seed {seed}] Starting experiment: {config_id}")
    
    # Initialize components
    env = NetworkEnvironment(n_hosts=n_hosts, config=config, seed=seed)
    
    attacker = DivideConquerAttacker(
        n_hosts=n_hosts,
        partitioning_config=config['partitioning'],
        switch_config=config['attacker']['switch'],
        seed=seed
    )
    
    defender = IDHAMDefender(
        n_hosts=n_hosts,
        config=config,
        seed=seed
    )
    
    # Setup logging
    log_dir = output_dir / 'logs' / config_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = log_dir / f'seed_{seed}.csv'
    metrics_logger = MetricsLogger(csv_path)
    
    # Track switch points for adaptation lag calculation
    switch_points = []
    last_partition = None
    
    # Main training/evaluation loop
    all_metrics = []
    
    for episode in range(episodes_train):
        # Check for attacker switch
        current_partition = attacker.get_current_partition(episode)
        if last_partition is not None and current_partition != last_partition:
            switch_points.append(episode)
            print(f"[Seed {seed}] Episode {episode}: Attacker switched to partition {current_partition}")
        last_partition = current_partition
        
        # Run episode
        episode_metrics = run_single_episode(
            env, attacker, defender, episode, probe_budget
        )
        
        # Add metadata
        episode_metrics.update({
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            'config_id': config_id,
            'episode': episode,
            'attacker_mode': config['attacker']['strategy']
        })
        
        # Log to CSV
        metrics_logger.log_episode(episode_metrics)
        all_metrics.append(episode_metrics)
        
        # Periodic progress update
        if (episode + 1) % 500 == 0:
            recent_tsh = np.mean([m['tsh'] for m in all_metrics[-100:]])
            recent_coverage = np.mean([m['coverage'] for m in all_metrics[-100:]])
            print(f"[Seed {seed}] Episode {episode+1}/{episodes_train}: "
                  f"TSH={recent_tsh:.2f}, Coverage={recent_coverage:.3f}")
    
    metrics_logger.close()
    
    print(f"[Seed {seed}] Computing windowed metrics...")
    
    # Compute windowed metrics
    window_size = config['metrics']['window']
    windowed_metrics = compute_windowed_metrics(all_metrics, window_size)
    
    # Compute adaptation lag
    adaptation_epsilon = config['metrics']['adaptation_epsilon']
    adaptation_lags = compute_adaptation_lag(
        all_metrics, switch_points, window_size, adaptation_epsilon
    )
    
    # Compute summary statistics
    tsh_values = [m['tsh'] for m in all_metrics]
    coverage_values = [m['coverage'] for m in all_metrics]
    
    summary = {
        'seed': seed,
        'config_id': config_id,
        'n_episodes': len(all_metrics),
        'n_switches': len(switch_points),
        'switch_points': switch_points,
        'tsh_median': float(np.median(tsh_values)),
        'tsh_mean': float(np.mean(tsh_values)),
        'tsh_std': float(np.std(tsh_values)),
        'coverage_final': float(coverage_values[-1]),
        'coverage_mean': float(np.mean(coverage_values)),
        'windowed_tsh_auc': float(np.sum([w['tsh_mean'] for w in windowed_metrics])),
        'adaptation_lags': adaptation_lags,
        'median_adaptation_lag': float(np.median(adaptation_lags)) if adaptation_lags else None,
        'mean_qos_proxy_ms': float(np.mean([m['qos_penalty_proxy_ms'] for m in all_metrics])),
        'mean_flow_mods': float(np.mean([m['flow_mods_count'] for m in all_metrics]))
    }
    
    # Save windowed metrics
    windowed_path = log_dir / f'seed_{seed}_windowed.json'
    with open(windowed_path, 'w') as f:
        json.dump(windowed_metrics, f, indent=2)
    
    print(f"[Seed {seed}] Completed. TSH median: {summary['tsh_median']:.2f}, "
          f"Adaptation lag median: {summary['median_adaptation_lag']}")
    
    return summary


def aggregate_results(summaries: List[Dict], config: Dict, output_dir: Path):
    """Aggregate results across all seeds and save summary."""
    config_id = config['config_id']
    
    # Aggregate statistics
    tsh_medians = [s['tsh_median'] for s in summaries]
    windowed_tsh_aucs = [s['windowed_tsh_auc'] for s in summaries]
    adaptation_lags_all = []
    for s in summaries:
        if s['adaptation_lags']:
            adaptation_lags_all.extend(s['adaptation_lags'])
    
    qos_means = [s['mean_qos_proxy_ms'] for s in summaries]
    flow_mods_means = [s['mean_flow_mods'] for s in summaries]
    
    aggregated = {
        'config_id': config_id,
        'n_seeds': len(summaries),
        'tsh_median_across_seeds': float(np.median(tsh_medians)),
        'tsh_median_iqr': [float(np.percentile(tsh_medians, 25)), 
                           float(np.percentile(tsh_medians, 75))],
        'windowed_tsh_auc_median': float(np.median(windowed_tsh_aucs)),
        'windowed_tsh_auc_iqr': [float(np.percentile(windowed_tsh_aucs, 25)),
                                  float(np.percentile(windowed_tsh_aucs, 75))],
        'adaptation_lag_median': float(np.median(adaptation_lags_all)) if adaptation_lags_all else None,
        'adaptation_lag_90pct': float(np.percentile(adaptation_lags_all, 90)) if adaptation_lags_all else None,
        'adaptation_lag_iqr': [float(np.percentile(adaptation_lags_all, 25)),
                               float(np.percentile(adaptation_lags_all, 75))] if adaptation_lags_all else None,
        'mean_qos_proxy_ms': float(np.mean(qos_means)),
        'mean_flow_mods': float(np.mean(flow_mods_means)),
        'seed_summaries': summaries
    }
    
    # Save aggregated results
    results_dir = output_dir / 'results' / config_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = results_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS: {config_id}")
    print(f"{'='*60}")
    print(f"Windowed TSH AUC (median): {aggregated['windowed_tsh_auc_median']:.2f}")
    print(f"Adaptation Lag (median): {aggregated['adaptation_lag_median']}")
    print(f"Mean QoS Proxy (ms): {aggregated['mean_qos_proxy_ms']:.2f}")
    print(f"Mean Flow Mods: {aggregated['mean_flow_mods']:.2f}")
    print(f"{'='*60}\n")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Run ID-HAM divide-and-conquer + timing variations experiment'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config file (YAML or JSON)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for logs and results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Run single seed only (for debugging)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    create_directories(output_dir, config['config_id'])
    
    # Determine seeds to run
    if args.seed is not None:
        seeds = [args.seed]
        print(f"Running single seed: {args.seed}")
    else:
        seeds = list(range(config['seeds']))
        print(f"Running {len(seeds)} seeds")
    
    # Run experiments
    summaries = []
    for seed in seeds:
        try:
            summary = run_experiment(config, seed, output_dir)
            summaries.append(summary)
        except Exception as e:
            print(f"ERROR in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not summaries:
        print("ERROR: No successful runs!")
        sys.exit(1)
    
    # Aggregate results
    aggregate_results(summaries, config, output_dir)
    
    print(f"\nExperiment complete! Results saved to {output_dir}")
    print(f"Config: {config['config_id']}")
    print(f"Successful seeds: {len(summaries)}/{len(seeds)}")


if __name__ == '__main__':
    main()