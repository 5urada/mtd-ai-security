#!/usr/bin/env python3
"""
Plotting and analysis utilities for experiment results.
Generates plots as specified in the experiment prompt.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import json
from pathlib import Path
from typing import List, Dict, Optional
import scipy.stats as stats

from src.metrics import load_csv_metrics


def plot_tsh_timeseries(
    config_id: str,
    output_dir: Path,
    seeds: List[int],
    switch_points: Optional[List[int]] = None,
    smoothing_window: int = 50
):
    """
    Plot TSH time series with per-seed thin lines and median-smoothed curve.
    
    Args:
        config_id: Configuration identifier
        output_dir: Base output directory
        seeds: List of seed numbers to plot
        switch_points: Episode numbers where attacker switched (for annotations)
        smoothing_window: Window size for median smoothing
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_tsh_series = []
    
    # Plot individual seeds
    for seed in seeds:
        csv_path = output_dir / 'logs' / config_id / f'seed_{seed}.csv'
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue
        
        metrics = load_csv_metrics(csv_path)
        episodes = [m['episode'] for m in metrics]
        tsh_values = [m['tsh'] for m in metrics]
        
        ax.plot(episodes, tsh_values, alpha=0.15, linewidth=0.5, color='blue')
        all_tsh_series.append(tsh_values)
    
    # Compute and plot median curve
    if all_tsh_series:
        tsh_array = np.array(all_tsh_series)
        median_tsh = np.median(tsh_array, axis=0)
        
        # Smooth median with moving average
        if len(median_tsh) >= smoothing_window:
            kernel = np.ones(smoothing_window) / smoothing_window
            smoothed_median = np.convolve(median_tsh, kernel, mode='same')
        else:
            smoothed_median = median_tsh
        
        episodes = range(len(median_tsh))
        ax.plot(episodes, smoothed_median, linewidth=2.5, color='darkblue',
                label='Median (smoothed)', zorder=10)
    
    # Annotate switch points
    if switch_points:
        ylim = ax.get_ylim()
        for switch_ep in switch_points:
            ax.axvline(x=switch_ep, color='red', linestyle='--', 
                      alpha=0.6, linewidth=1.5, label='Attacker switch' if switch_ep == switch_points[0] else '')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Times of Scanning Hits (TSH)', fontsize=12)
    ax.set_title(f'TSH Time Series: {config_id}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save
    fig_dir = output_dir / 'figs' / config_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'tsh_timeseries.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'tsh_timeseries.pdf', bbox_inches='tight')
    plt.close()


def plot_tsh_auc_comparison(
    config_id: str,
    baseline_config_id: str,
    output_dir: Path,
    seeds: List[int]
):
    """
    Bar chart comparing windowed TSH AUC between dynamic and baseline.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Load aggregated results
    results_path = output_dir / 'results' / config_id / 'summary.json'
    baseline_path = output_dir / 'results' / baseline_config_id / 'summary.json'
    
    with open(results_path) as f:
        results = json.load(f)
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    # Extract values
    dynamic_auc = results['windowed_tsh_auc_median']
    dynamic_iqr = results['windowed_tsh_auc_iqr']
    
    baseline_auc = baseline['windowed_tsh_auc_median']
    baseline_iqr = baseline['windowed_tsh_auc_iqr']
    
    # Compute error bars (half of IQR range)
    dynamic_err = (dynamic_iqr[1] - dynamic_iqr[0]) / 2
    baseline_err = (baseline_iqr[1] - baseline_iqr[0]) / 2
    
    # Bar chart
    x = np.arange(2)
    heights = [baseline_auc, dynamic_auc]
    errors = [baseline_err, dynamic_err]
    labels = ['Baseline\n(Static)', 'Dynamic\n(Timing Variations)']
    colors = ['lightblue', 'coral']
    
    bars = ax.bar(x, heights, yerr=errors, capsize=10, alpha=0.8,
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Windowed TSH AUC (Median)', fontsize=12)
    ax.set_title(f'TSH AUC Comparison\n{config_id}', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add percentage reduction text
    pct_reduction = ((baseline_auc - dynamic_auc) / baseline_auc) * 100
    ax.text(1, dynamic_auc + dynamic_err, f'{pct_reduction:.1f}% reduction',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')
    
    plt.tight_layout()
    
    fig_dir = output_dir / 'figs' / config_id
    plt.savefig(fig_dir / 'tsh_auc_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'tsh_auc_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_coverage_vs_budget(
    config_id: str,
    output_dir: Path,
    seeds: List[int],
    max_probes: int = 500
):
    """
    Plot coverage vs probe count.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For simplicity, plot final coverage per episode
    # (Full implementation would track within-episode coverage)
    
    for seed in seeds[:5]:  # Plot first 5 seeds to avoid clutter
        csv_path = output_dir / 'logs' / config_id / f'seed_{seed}.csv'
        if not csv_path.exists():
            continue
        
        metrics = load_csv_metrics(csv_path)
        episodes = [m['episode'] for m in metrics]
        coverage = [m['coverage'] for m in metrics]
        
        ax.plot(episodes, coverage, alpha=0.3, linewidth=1, label=f'Seed {seed}')
    
    # Plot median across all seeds
    all_coverage = []
    for seed in seeds:
        csv_path = output_dir / 'logs' / config_id / f'seed_{seed}.csv'
        if not csv_path.exists():
            continue
        metrics = load_csv_metrics(csv_path)
        coverage = [m['coverage'] for m in metrics]
        all_coverage.append(coverage)
    
    if all_coverage:
        coverage_array = np.array(all_coverage)
        median_coverage = np.median(coverage_array, axis=0)
        episodes = range(len(median_coverage))
        ax.plot(episodes, median_coverage, linewidth=3, color='darkgreen',
                label='Median', zorder=10)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Coverage (Discovered Hosts / Total Real Hosts)', fontsize=12)
    ax.set_title(f'Coverage vs Episode: {config_id}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_dir = output_dir / 'figs' / config_id
    plt.savefig(fig_dir / 'coverage_vs_budget.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'coverage_vs_budget.pdf', bbox_inches='tight')
    plt.close()


def plot_adaptation_lag_distribution(
    config_id: str,
    output_dir: Path,
    seeds: List[int]
):
    """
    Plot distribution of adaptation lags (CDF and violin plot).
    """
    # Load aggregated results
    results_path = output_dir / 'results' / config_id / 'summary.json'
    with open(results_path) as f:
        results = json.load(f)
    
    # Collect all adaptation lags
    all_lags = []
    for summary in results['seed_summaries']:
        if summary['adaptation_lags']:
            all_lags.extend(summary['adaptation_lags'])
    
    if not all_lags:
        print(f"No adaptation lags available for {config_id}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # CDF plot
    sorted_lags = np.sort(all_lags)
    cdf = np.arange(1, len(sorted_lags) + 1) / len(sorted_lags)
    
    ax1.plot(sorted_lags, cdf, linewidth=2, color='purple')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    ax1.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    
    median_lag = np.median(all_lags)
    p90_lag = np.percentile(all_lags, 90)
    
    ax1.set_xlabel('Adaptation Lag (episodes)', fontsize=12)
    ax1.set_ylabel('Cumulative Probability', fontsize=12)
    ax1.set_title(f'Adaptation Lag CDF\nMedian: {median_lag:.0f}, 90%: {p90_lag:.0f}',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    parts = ax2.violinplot([all_lags], positions=[0], widths=0.7,
                           showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    ax2.set_ylabel('Adaptation Lag (episodes)', fontsize=12)
    ax2.set_title('Adaptation Lag Distribution', fontsize=12, fontweight='bold')
    ax2.set_xticks([0])
    ax2.set_xticklabels([config_id], rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig_dir = output_dir / 'figs' / config_id
    plt.savefig(fig_dir / 'adaptation_lag_dist.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'adaptation_lag_dist.pdf', bbox_inches='tight')
    plt.close()


def plot_policy_entropy(
    config_id: str,
    output_dir: Path,
    seeds: List[int],
    switch_points: Optional[List[int]] = None
):
    """Plot policy entropy over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for seed in seeds[:5]:
        csv_path = output_dir / 'logs' / config_id / f'seed_{seed}.csv'
        if not csv_path.exists():
            continue
        
        metrics = load_csv_metrics(csv_path)
        episodes = [m['episode'] for m in metrics]
        entropy = [m['policy_entropy'] for m in metrics]
        
        ax.plot(episodes, entropy, alpha=0.3, linewidth=1)
    
    # Median
    all_entropy = []
    for seed in seeds:
        csv_path = output_dir / 'logs' / config_id / f'seed_{seed}.csv'
        if not csv_path.exists():
            continue
        metrics = load_csv_metrics(csv_path)
        entropy = [m['policy_entropy'] for m in metrics]
        all_entropy.append(entropy)
    
    if all_entropy:
        entropy_array = np.array(all_entropy)
        median_entropy = np.median(entropy_array, axis=0)
        episodes = range(len(median_entropy))
        ax.plot(episodes, median_entropy, linewidth=2.5, color='darkviolet',
                label='Median', zorder=10)
    
    # Annotate switches
    if switch_points:
        for switch_ep in switch_points:
            ax.axvline(x=switch_ep, color='red', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Policy Entropy', fontsize=12)
    ax.set_title(f'Policy Entropy Over Time: {config_id}', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_dir = output_dir / 'figs' / config_id
    plt.savefig(fig_dir / 'policy_entropy.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'policy_entropy.pdf', bbox_inches='tight')
    plt.close()


def plot_qos_summary(
    config_id: str,
    baseline_config_id: str,
    output_dir: Path
):
    """Plot QoS proxy and flow mods comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Load results
    with open(output_dir / 'results' / config_id / 'summary.json') as f:
        results = json.load(f)
    with open(output_dir / 'results' / baseline_config_id / 'summary.json') as f:
        baseline = json.load(f)
    
    # QoS proxy
    qos_values = [results['mean_qos_proxy_ms'], baseline['mean_qos_proxy_ms']]
    flow_values = [results['mean_flow_mods'], baseline['mean_flow_mods']]
    
    labels = ['Dynamic', 'Baseline']
    x = np.arange(len(labels))
    
    ax1.bar(x, qos_values, color=['coral', 'lightblue'], alpha=0.8, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Mean QoS Proxy (ms)', fontsize=12)
    ax1.set_title('QoS Cost', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    ax2.bar(x, flow_values, color=['coral', 'lightblue'], alpha=0.8, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Mean Flow Modifications', fontsize=12)
    ax2.set_title('Flow Mods', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig_dir = output_dir / 'figs' / config_id
    plt.savefig(fig_dir / 'qos_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'qos_summary.pdf', bbox_inches='tight')
    plt.close()


def generate_all_plots(config_id: str, baseline_config_id: str, 
                       output_dir: Path, n_seeds: int = 10):
    """Generate all plots for a configuration."""
    seeds = list(range(n_seeds))
    
    # Load switch points from first seed
    csv_path = output_dir / 'logs' / config_id / 'seed_0.csv'
    if csv_path.exists():
        # Extract switch points (simplified - look for partition changes)
        metrics = load_csv_metrics(csv_path)
        switch_points = []
        last_partition = None
        for m in metrics:
            if 'partition_id' in m:
                if last_partition is not None and m['partition_id'] != last_partition:
                    switch_points.append(m['episode'])
                last_partition = m['partition_id']
    else:
        switch_points = None
    
    print(f"Generating plots for {config_id}...")
    
    plot_tsh_timeseries(config_id, output_dir, seeds, switch_points)
    plot_tsh_auc_comparison(config_id, baseline_config_id, output_dir, seeds)
    plot_coverage_vs_budget(config_id, output_dir, seeds)
    plot_adaptation_lag_distribution(config_id, output_dir, seeds)
    plot_policy_entropy(config_id, output_dir, seeds, switch_points)
    plot_qos_summary(config_id, baseline_config_id, output_dir)
    
    print(f"Plots saved to {output_dir / 'figs' / config_id}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python src/plotting.py <config_id> <baseline_config_id> [output_dir]")
        sys.exit(1)
    
    config_id = sys.argv[1]
    baseline_config_id = sys.argv[2]
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('output')
    
    generate_all_plots(config_id, baseline_config_id, output_dir)