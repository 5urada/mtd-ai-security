#!/usr/bin/env python3
"""
Analyze partial experiment results for preliminary findings.
Works with incomplete data to show trends and project final outcomes.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all available summary.json files into a DataFrame."""
    results = []
    
    for summary_file in results_dir.glob('*/summary.json'):
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {summary_file}: {e}")
            continue
    
    if not results:
        print("ERROR: No results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    return df


def parse_config_id(config_id: str) -> Dict:
    """Extract parameters from config_id string."""
    params = {
        'config_id': config_id,
        'is_baseline': 'baseline' in config_id
    }
    
    # Extract partitions (p4, p8, p16, p32)
    if 'p4' in config_id:
        params['partitions'] = 4
    elif 'p8' in config_id:
        params['partitions'] = 8
    elif 'p16' in config_id:
        params['partitions'] = 16
    elif 'p32' in config_id:
        params['partitions'] = 32
    else:
        params['partitions'] = None
    
    # Extract switch mode
    if 'unif' in config_id:
        params['timing'] = 'uniform'
    elif 'expo' in config_id:
        params['timing'] = 'exponential'
    elif 'stoc' in config_id:
        params['timing'] = 'stochastic'
    else:
        params['timing'] = 'static'
    
    # Extract mean interval (int200, int500, int1000)
    if 'int200' in config_id:
        params['interval'] = 200
    elif 'int500' in config_id:
        params['interval'] = 500
    elif 'int1000' in config_id:
        params['interval'] = 1000
    else:
        params['interval'] = None
    
    # Extract jitter (j10, j20)
    if 'j10' in config_id:
        params['jitter'] = 10
    elif 'j20' in config_id:
        params['jitter'] = 20
    else:
        params['jitter'] = None
    
    # Extract switch type (peri, stoc)
    if 'peri' in config_id:
        params['switch_type'] = 'periodic'
    elif 'stoc' in config_id:
        params['switch_type'] = 'stochastic'
    else:
        params['switch_type'] = None
    
    return params


def generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics from results."""
    # Add parsed parameters
    parsed = df['config_id'].apply(parse_config_id)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df, parsed_df], axis=1)
    
    # Summary stats
    summary = df.groupby(['is_baseline']).agg({
        'tsh_median_across_seeds': ['mean', 'median', 'std'],
        'windowed_tsh_auc_median': ['mean', 'median', 'std'],
        'adaptation_lag_median': ['mean', 'median', 'std'],
        'mean_qos_proxy_ms': ['mean', 'median', 'std'],
        'mean_flow_mods': ['mean', 'median', 'std'],
        'config_id': 'count'
    }).round(2)
    
    return summary


def compare_baseline_vs_dynamic(df: pd.DataFrame):
    """Compare baseline vs dynamic attacker performance."""
    # Add parsed parameters
    parsed = df['config_id'].apply(parse_config_id)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df, parsed_df], axis=1)
    
    baseline = df[df['is_baseline']]
    dynamic = df[~df['is_baseline']]
    
    print("\n" + "="*70)
    print("BASELINE vs DYNAMIC ATTACKER COMPARISON")
    print("="*70)
    
    if len(baseline) == 0:
        print("   No baseline results yet!")
        baseline_tsh = None
    else:
        baseline_tsh = baseline['tsh_median_across_seeds'].median()
        print(f"\n  BASELINE (Static Attacker):")
        print(f"   Configs completed: {len(baseline)}")
        print(f"   Median TSH: {baseline_tsh:.2f}")
        print(f"   Median QoS Cost: {baseline['mean_qos_proxy_ms'].median():.2f} ms")
        print(f"   Median Flow Mods: {baseline['mean_flow_mods'].median():.3f}")
    
    if len(dynamic) == 0:
        print("\n   No dynamic attacker results yet!")
        return
    
    dynamic_tsh = dynamic['tsh_median_across_seeds'].median()
    print(f"\n  DYNAMIC ATTACKER:")
    print(f"   Configs completed: {len(dynamic)}")
    print(f"   Median TSH: {dynamic_tsh:.2f}")
    print(f"   Median Adaptation Lag: {dynamic['adaptation_lag_median'].median():.0f} episodes")
    print(f"   Median QoS Cost: {dynamic['mean_qos_proxy_ms'].median():.2f} ms")
    print(f"   Median Flow Mods: {dynamic['mean_flow_mods'].median():.3f}")
    
    if baseline_tsh:
        improvement = ((dynamic_tsh - baseline_tsh) / baseline_tsh) * 100
        print(f"\n  KEY FINDING:")
        print(f"   Dynamic attacker TSH is {abs(improvement):.1f}% {'HIGHER' if improvement > 0 else 'LOWER'} than baseline")
        print(f"     Dynamic attacker is {'MORE' if improvement > 0 else 'LESS'} effective at discovery!")


def analyze_by_parameters(df: pd.DataFrame):
    """Analyze results by different parameters."""
    # Add parsed parameters
    parsed = df['config_id'].apply(parse_config_id)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df, parsed_df], axis=1)
    
    # Filter to dynamic attackers only
    dynamic = df[~df['is_baseline']].copy()
    
    if len(dynamic) == 0:
        print("\n   No dynamic attacker results to analyze by parameters")
        return
    
    print("\n" + "="*70)
    print("PARAMETER ANALYSIS")
    print("="*70)
    
    # By partitions
    if dynamic['partitions'].notna().any():
        print("\n  By Number of Partitions:")
        partition_stats = dynamic.groupby('partitions')['tsh_median_across_seeds'].agg(['count', 'mean', 'median']).round(2)
        print(partition_stats.to_string())
    
    # By timing distribution
    if dynamic['timing'].notna().any():
        print("\n  By Switch Timing Distribution:")
        timing_stats = dynamic.groupby('timing')['tsh_median_across_seeds'].agg(['count', 'mean', 'median']).round(2)
        print(timing_stats.to_string())
    
    # By mean interval
    if dynamic['interval'].notna().any():
        print("\n  By Mean Switch Interval:")
        interval_stats = dynamic.groupby('interval')['tsh_median_across_seeds'].agg(['count', 'mean', 'median']).round(2)
        print(interval_stats.to_string())
    
    # By jitter
    if dynamic['jitter'].notna().any():
        print("\n  By Jitter Percentage:")
        jitter_stats = dynamic.groupby('jitter')['tsh_median_across_seeds'].agg(['count', 'mean', 'median']).round(2)
        print(jitter_stats.to_string())


def find_best_configs(df: pd.DataFrame, top_n: int = 10):
    """Find top performing configurations for the attacker."""
    # Add parsed parameters
    parsed = df['config_id'].apply(parse_config_id)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df, parsed_df], axis=1)
    
    # Filter to dynamic attackers only
    dynamic = df[~df['is_baseline']].copy()
    
    if len(dynamic) == 0:
        print("\n   No dynamic attacker results yet!")
        return
    
    print("\n" + "="*70)
    print(f"TOP {min(top_n, len(dynamic))} ATTACKER CONFIGURATIONS (Highest TSH)")
    print("="*70)
    
    top_configs = dynamic.nlargest(top_n, 'tsh_median_across_seeds')
    
    for i, row in enumerate(top_configs.itertuples(), 1):
        print(f"\n{i}. {row.config_id}")
        print(f"   TSH: {row.tsh_median_across_seeds:.2f}")
        print(f"   TSH AUC: {row.windowed_tsh_auc_median:.2f}")
        print(f"   Adaptation Lag: {row.adaptation_lag_median:.0f} episodes")
        print(f"   QoS Cost: {row.mean_qos_proxy_ms:.2f} ms")
        print(f"   Parameters: {row.partitions}p, {row.timing}, int={row.interval}, j={row.jitter}%")


def project_completion(results_dir: Path):
    """Project completion time and coverage."""
    completed = len(list(results_dir.glob('*/summary.json')))
    total = 433
    
    print("\n" + "="*70)
    print("EXPERIMENT PROGRESS")
    print("="*70)
    print(f"\n  Completed: {completed} / {total} ({completed/total*100:.1f}%)")
    print(f"  Remaining: {total - completed}")
    
    if completed > 10:
        # Estimate time per config
        # This is rough - you can improve with actual timing data
        avg_time_per_config = 10  # minutes (adjust based on your observation)
        remaining_time = (total - completed) * avg_time_per_config
        
        print(f"\n   Estimated time remaining: {remaining_time/60:.1f} hours")
        print(f"   (Assuming ~{avg_time_per_config} min per config)")


def export_summary_csv(df: pd.DataFrame, output_file: Path):
    """Export summary table to CSV."""
    # Add parsed parameters
    parsed = df['config_id'].apply(parse_config_id)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df, parsed_df], axis=1)
    
    # Select key columns
    summary_cols = [
        'config_id', 'is_baseline', 'partitions', 'timing', 'interval', 'jitter',
        'tsh_median_across_seeds', 'windowed_tsh_auc_median', 
        'adaptation_lag_median', 'mean_qos_proxy_ms', 'mean_flow_mods'
    ]
    
    export_df = df[summary_cols].sort_values('tsh_median_across_seeds', ascending=False)
    export_df.to_csv(output_file, index=False)
    print(f"\n  Exported summary to: {output_file}")


def main():
    print("\n" + "="*70)
    print("ID-HAM PARTIAL RESULTS ANALYSIS")
    print("="*70)
    
    # Load results
    results_dir = Path('output/results')
    if not results_dir.exists():
        print(f"\nERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"\n  Loading results from: {results_dir}")
    df = load_all_results(results_dir)
    
    if df.empty:
        print("\n  No results found! Run some experiments first.")
        sys.exit(1)
    
    print(f"  Loaded {len(df)} experiment results")
    
    # Project completion
    project_completion(results_dir)
    
    # Compare baseline vs dynamic
    compare_baseline_vs_dynamic(df)
    
    # Parameter analysis
    analyze_by_parameters(df)
    
    # Find best configs
    find_best_configs(df, top_n=10)
    
    # Export summary
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)
    export_summary_csv(df, output_dir / 'partial_results_summary.csv')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\n  TIP: Run this script periodically to monitor experiment progress")
    print("  For presentation: Use the TOP 10 configs and baseline comparison\n")


if __name__ == '__main__':
    main()