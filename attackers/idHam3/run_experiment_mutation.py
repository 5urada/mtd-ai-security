#!/usr/bin/env python3
"""
ID-HAM Experiment Runner with TRUE ADDRESS MUTATION

This replaces the masking approach with actual address mutation.
Compatible with all new defender types: idham_mutation, frvm, rhm, static_mutation
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

# Import mutation-based components
from src.attacker import DivideConquerAttacker

# Import MUTATION defenders
try:
    from src.defender_mutation import (
        IDHAMMutationDefender,
        FRVMDefender,
        RHMDefender,
        StaticMutationDefender
    )
except ImportError:
    print("ERROR: Could not import mutation defenders.")
    print("Make sure defender_mutation.py is in src/ directory")
    sys.exit(1)

# Import mutation-aware environment
try:
    from src.environment_mutation import NetworkEnvironment
except ImportError:
    print("ERROR: Could not import mutation environment.")
    print("Make sure environment_mutation.py is in src/ directory")
    sys.exit(1)

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


def run_single_episode_mutation(
    env: NetworkEnvironment,
    attacker: DivideConquerAttacker,
    defender,  # Mutation defender (any type)
    episode: int,
    probe_budget: int,
    discovered_hosts_global: set,  # Cumulative PHYSICAL hosts discovered
    dwell_time_active: int = 0,
    last_partition_id: Optional[int] = None
) -> Dict:
    """
    Run a single episode with TRUE ADDRESS MUTATION.
    
    Key differences from masking:
    1. Attacker probes VIRTUAL IPs
    2. Defender resolves virtual IP → physical host  
    3. Environment checks if PHYSICAL host is real
    4. Track mutations instead of masks
    
    Returns dict with metrics including mutation instrumentation
    """
    # Reset episode state
    env.reset()
    defender.reset_episode()
    
    # Get current attacker partition
    partition_id = attacker.get_current_partition(episode)
    target_hosts = attacker.get_partition_targets(partition_id)
    
    # Detect switch
    is_switch = (partition_id != last_partition_id) if last_partition_id is not None else False
    
    # Attacker performs probing
    hits_count = 0
    discovered_hosts_this_episode = set()  # PHYSICAL hosts discovered
    mutated_probe_count = 0
    
    for probe_idx in range(probe_budget):
        # 1. Attacker selects target from partition (treated as VIRTUAL IP)
        virtual_ip = attacker.select_target(target_hosts, probe_idx)
        
        # 2. Defender resolves VIRTUAL IP → PHYSICAL HOST
        #    THIS IS THE KEY MUTATION STEP
        physical_host, is_mutated, qos_cost = defender.resolve_probe(virtual_ip, probe_idx)
        
        # Track mutated probes
        if is_mutated:
            mutated_probe_count += 1
        
        # 3. Environment checks if PHYSICAL HOST is real
        #    Mutation provides some protection even if host is real
        is_hit = env.probe_with_mutation(virtual_ip, physical_host, is_mutated)
        
        if is_hit:
            hits_count += 1
            discovered_hosts_this_episode.add(physical_host)  # Track PHYSICAL host
            discovered_hosts_global.add(physical_host)
    
    # Calculate TSH
    tsh = hits_count
    tsh_per_100 = (tsh / probe_budget) * 100 if probe_budget > 0 else 0
    
    # Coverage is cumulative across all episodes (of PHYSICAL hosts)
    coverage = len(discovered_hosts_global) / env.n_real_hosts
    
    # Update defender (learning step if applicable)
    defender.update(hits_count, discovered_hosts_this_episode, episode)
    
    # Get defender metrics
    policy_entropy = defender.get_policy_entropy()
    mutation_count = defender.get_mutation_count()
    mutation_flag = defender.did_mutation_occur()
    qos_data_ms = defender.get_qos_data_plane_ms()
    qos_ctrl_ms = defender.get_qos_control_plane_ms()
    
    # Mutation saturation (fraction of partition that's mutated)
    mutation_saturation = defender.get_mutation_fraction(target_hosts)
    
    # Address entropy (Shannon entropy of address mapping)
    address_entropy = defender.get_address_entropy()
    
    metrics = {
        'tsh': tsh,
        'tsh_per_100_probes': tsh_per_100,
        'hits_count': hits_count,
        'discovered_hosts_count': len(discovered_hosts_this_episode),
        'discovered_hosts_cumulative': len(discovered_hosts_global),
        'coverage': coverage,
        'policy_entropy': policy_entropy,
        'mutation_count': mutation_count,  # RENAMED from flow_mods_count
        'mutation_flag': int(mutation_flag),  # RENAMED from mask_change_flag
        'qos_penalty_proxy_ms': qos_data_ms + qos_ctrl_ms,
        'qos_data_plane_ms': qos_data_ms,
        'qos_control_plane_ms': qos_ctrl_ms,
        'partition_id': partition_id,
        'probe_budget': probe_budget,
        # Mutation instrumentation
        'is_switch': int(is_switch),
        'dwell_time_active': dwell_time_active,
        'focus_partition_id': partition_id,
        'mutated_probe_count': mutated_probe_count,
        'mutation_saturation_active_partition': mutation_saturation,  # RENAMED
        'address_entropy': address_entropy,  # NEW metric
    }
    
    return metrics


def run_experiment(config: Dict, seed: int, output_dir: Path) -> Dict:
    """
    Run a complete experiment for a single seed with mutation.
    
    Returns summary statistics for this seed.
    """
    # Seed matching for strata
    seed_strategy = config.get('seed_strategy', 'independent')
    if seed_strategy == 'matched':
        matched_group_id = config.get('matched_seed_group_id', 'default')
        base_seed = hash(matched_group_id) & 0x7FFFFFFF
        actual_seed = base_seed + seed
        set_seed(actual_seed)
        print(f"[Seed {seed}] Using matched seed: {actual_seed} (group: {matched_group_id})")
    else:
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
    
    # Initialize MUTATION defender based on type
    defender_cfg = config.get('defender', {'type': 'idham_mutation'})
    defender_type = defender_cfg.get('type', 'idham_mutation')
    
    if defender_type == 'idham_mutation':
        defender = IDHAMMutationDefender(n_hosts=n_hosts, config=config, seed=seed)
    elif defender_type == 'frvm':
        defender = FRVMDefender(n_hosts=n_hosts, config=config, seed=seed)
    elif defender_type == 'rhm':
        defender = RHMDefender(n_hosts=n_hosts, config=config, seed=seed)
    elif defender_type == 'static_mutation':
        defender = StaticMutationDefender(n_hosts=n_hosts, config=config, seed=seed)
    else:
        raise ValueError(f"Unknown defender type: {defender_type}")
    
    print(f"[Seed {seed}] Using defender type: {defender_type}")
    
    # Setup logging
    log_dir = output_dir / 'logs' / config_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = log_dir / f'seed_{seed}.csv'
    metrics_logger = MetricsLogger(csv_path)
    
    # Track switch points
    switch_points = []
    last_partition = None
    last_switch_episode = 0
    
    # Initialize global cumulative discovered hosts set (PHYSICAL hosts)
    discovered_hosts_global = set()
    
    # Main training/evaluation loop
    all_metrics = []
    
    for episode in range(episodes_train):
        # Check for attacker switch
        current_partition = attacker.get_current_partition(episode)
        if last_partition is not None and current_partition != last_partition:
            switch_points.append(episode)
            last_switch_episode = episode
            print(f"[Seed {seed}] Episode {episode}: Attacker switched to partition {current_partition}")
        last_partition = current_partition
        
        # Calculate dwell time
        dwell_time = episode - last_switch_episode
        
        # Run episode with MUTATION
        episode_metrics = run_single_episode_mutation(
            env, attacker, defender, episode, probe_budget,
            discovered_hosts_global,
            dwell_time_active=dwell_time,
            last_partition_id=last_partition if episode > 0 else None
        )
        
        # Add metadata
        episode_metrics.update({
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            'config_id': config_id,
            'episode': episode,
            'attacker_mode': config['attacker']['strategy'],
            'defender_type': defender_type,  # Add defender type
        })
        
        # Log to CSV
        metrics_logger.log_episode(episode_metrics)
        all_metrics.append(episode_metrics)
        
        # Periodic progress update
        if (episode + 1) % 500 == 0:
            recent_tsh = np.mean([m['tsh'] for m in all_metrics[-100:]])
            recent_coverage = np.mean([m['coverage'] for m in all_metrics[-100:]])
            recent_mutations = np.mean([m['mutation_count'] for m in all_metrics[-100:]])
            print(f"[Seed {seed}] Episode {episode+1}/{episodes_train}: "
                  f"TSH={recent_tsh:.2f}, Coverage={recent_coverage:.3f}, "
                  f"Mutations={recent_mutations:.1f}")
    
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
    tsh_per_100_values = [m['tsh_per_100_probes'] for m in all_metrics]
    coverage_values = [m['coverage'] for m in all_metrics]
    dwell_times = [m['dwell_time_active'] for m in all_metrics]
    mutation_saturations = [m['mutation_saturation_active_partition'] for m in all_metrics]
    address_entropies = [m['address_entropy'] for m in all_metrics]
    mutation_counts = [m['mutation_count'] for m in all_metrics]
    
    summary = {
        'seed': seed,
        'config_id': config_id,
        'defender_type': defender_type,
        'n_episodes': len(all_metrics),
        'n_switches': len(switch_points),
        'switch_points': switch_points,
        'tsh_median': float(np.median(tsh_values)),
        'tsh_mean': float(np.mean(tsh_values)),
        'tsh_std': float(np.std(tsh_values)),
        'tsh_per_100_median': float(np.median(tsh_per_100_values)),
        'tsh_per_100_mean': float(np.mean(tsh_per_100_values)),
        'coverage_final': float(coverage_values[-1]),
        'coverage_mean': float(np.mean(coverage_values)),
        'windowed_tsh_auc': float(np.sum([w['tsh_mean'] for w in windowed_metrics])),
        'adaptation_lags': adaptation_lags,
        'median_adaptation_lag': float(np.median(adaptation_lags)) if adaptation_lags else None,
        'mean_qos_proxy_ms': float(np.mean([m['qos_penalty_proxy_ms'] for m in all_metrics])),
        'mean_mutation_count': float(np.mean(mutation_counts)),  # RENAMED
        # Mutation-specific statistics
        'mean_dwell_time': float(np.mean(dwell_times)),
        'mean_mutation_saturation': float(np.mean(mutation_saturations)),
        'mean_address_entropy': float(np.mean(address_entropies)),
        'total_mutation_events': int(np.sum(mutation_counts)),
    }
    
    # Save windowed metrics
    windowed_path = log_dir / f'seed_{seed}_windowed.json'
    with open(windowed_path, 'w') as f:
        json.dump(windowed_metrics, f, indent=2)
    
    print(f"[Seed {seed}] Completed. TSH median: {summary['tsh_median']:.2f}, "
          f"TSH/100 median: {summary['tsh_per_100_median']:.2f}, "
          f"Coverage final: {summary['coverage_final']:.3f}, "
          f"Mutations: {summary['total_mutation_events']}")
    
    return summary


def aggregate_results(summaries: List[Dict], config: Dict, output_dir: Path):
    """Aggregate results across all seeds and save summary."""
    config_id = config['config_id']
    
    # Aggregate statistics
    tsh_medians = [s['tsh_median'] for s in summaries]
    tsh_per_100_medians = [s['tsh_per_100_median'] for s in summaries]
    windowed_tsh_aucs = [s['windowed_tsh_auc'] for s in summaries]
    adaptation_lags_all = []
    for s in summaries:
        if s['adaptation_lags']:
            adaptation_lags_all.extend(s['adaptation_lags'])
    
    qos_means = [s['mean_qos_proxy_ms'] for s in summaries]
    mutation_counts = [s['mean_mutation_count'] for s in summaries]
    address_entropies = [s['mean_address_entropy'] for s in summaries]
    
    aggregated = {
        'config_id': config_id,
        'defender_type': config.get('defender', {}).get('type', 'unknown'),
        'n_seeds': len(summaries),
        'tsh_median_across_seeds': float(np.median(tsh_medians)),
        'tsh_median_iqr': [float(np.percentile(tsh_medians, 25)),
                           float(np.percentile(tsh_medians, 75))],
        'tsh_per_100_median_across_seeds': float(np.median(tsh_per_100_medians)),
        'tsh_per_100_iqr': [float(np.percentile(tsh_per_100_medians, 25)),
                            float(np.percentile(tsh_per_100_medians, 75))],
        'windowed_tsh_auc_median': float(np.median(windowed_tsh_aucs)),
        'windowed_tsh_auc_iqr': [float(np.percentile(windowed_tsh_aucs, 25)),
                                  float(np.percentile(windowed_tsh_aucs, 75))],
        'adaptation_lag_median': float(np.median(adaptation_lags_all)) if adaptation_lags_all else None,
        'adaptation_lag_90pct': float(np.percentile(adaptation_lags_all, 90)) if adaptation_lags_all else None,
        'adaptation_lag_iqr': [float(np.percentile(adaptation_lags_all, 25)),
                               float(np.percentile(adaptation_lags_all, 75))] if adaptation_lags_all else None,
        'mean_qos_proxy_ms': float(np.mean(qos_means)),
        'mean_mutation_count': float(np.mean(mutation_counts)),  # RENAMED
        'mean_address_entropy': float(np.mean(address_entropies)),  # NEW
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
    print(f"Defender Type: {aggregated['defender_type']}")
    print(f"Windowed TSH AUC (median): {aggregated['windowed_tsh_auc_median']:.2f}")
    print(f"TSH per 100 probes (median): {aggregated['tsh_per_100_median_across_seeds']:.2f}")
    print(f"Adaptation Lag (median): {aggregated['adaptation_lag_median']}")
    print(f"Mean QoS Proxy (ms): {aggregated['mean_qos_proxy_ms']:.2f}")
    print(f"Mean Mutation Count: {aggregated['mean_mutation_count']:.2f}")
    print(f"Mean Address Entropy: {aggregated['mean_address_entropy']:.2f}")
    print(f"{'='*60}\n")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Run ID-HAM experiment with TRUE ADDRESS MUTATION'
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
    
    # Validate defender type
    defender_type = config.get('defender', {}).get('type', 'unknown')
    valid_types = ['idham_mutation', 'frvm', 'rhm', 'static_mutation']
    if defender_type not in valid_types:
        print(f"WARNING: Unknown defender type: {defender_type}")
        print(f"Valid types: {valid_types}")
    
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
    print(f"Defender: {config.get('defender', {}).get('type', 'unknown')}")
    print(f"Successful seeds: {len(summaries)}/{len(seeds)}")


if __name__ == '__main__':
    main()