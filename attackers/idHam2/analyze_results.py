#!/usr/bin/env python3
"""
Statistical analysis and result aggregation across experiments.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import List, Dict
import sys


def mann_whitney_test(group1: List[float], group2: List[float]) -> Dict:
    """
    Perform Mann-Whitney U test between two groups.
    
    Returns dict with statistic, p-value, and Cliff's delta.
    """
    statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Compute Cliff's delta (effect size)
    n1, n2 = len(group1), len(group2)
    comparisons = np.array([[1 if g1 > g2 else (-1 if g1 < g2 else 0)
                            for g2 in group2] for g1 in group1])
    cliffs_delta = comparisons.sum() / (n1 * n2)
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'cliffs_delta': float(cliffs_delta),
        'significant': pvalue < 0.05
    }


def kruskal_wallis_test(groups: List[List[float]]) -> Dict:
    """
    Perform Kruskal-Wallis H-test across multiple groups.
    """
    statistic, pvalue = stats.kruskal(*groups)
    
    return {
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < 0.05
    }


def pairwise_mann_whitney_bonferroni(groups: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Perform pairwise Mann-Whitney tests with Bonferroni correction.
    """
    group_names = list(groups.keys())
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2
    alpha_corrected = 0.05 / n_comparisons
    
    results = []
    
    for i, name1 in enumerate(group_names):
        for j, name2 in enumerate(group_names[i+1:], start=i+1):
            test_result = mann_whitney_test(groups[name1], groups[name2])
            
            results.append({
                'group1': name1,
                'group2': name2,
                'pvalue': test_result['pvalue'],
                'cliffs_delta': test_result['cliffs_delta'],
                'significant_bonf': test_result['pvalue'] < alpha_corrected
            })
    
    return pd.DataFrame(results)


def analyze_partition_effect(output_dir: Path, baseline_config_id: str):
    """
    Analyze effect of number of partitions on TSH AUC.
    """
    print("\n" + "="*60)
    print("ANALYSIS: Effect of Number of Partitions")
    print("="*60)
    
    results_dir = output_dir / 'results'
    
    # Load baseline
    with open(results_dir / baseline_config_id / 'summary.json') as f:
        baseline = json.load(f)
    
    baseline_auc = baseline['windowed_tsh_auc_median']
    
    # Group configs by number of partitions
    partition_groups = {2: [], 4: [], 8: [], 16: []}
    
    for config_dir in results_dir.iterdir():
        if not config_dir.is_dir() or 'baseline' in config_dir.name:
            continue
        
        summary_path = config_dir / 'summary.json'
        if not summary_path.exists():
            continue
        
        with open(summary_path) as f:
            data = json.load(f)
        
        # Extract partition count from config_id
        config_id = data['config_id']
        for p in [2, 4, 8, 16]:
            if f'_p{p}_' in config_id:
                partition_groups[p].append(data['windowed_tsh_auc_median'])
                break
    
    # Statistical test
    groups_list = [partition_groups[p] for p in [2, 4, 8, 16] if partition_groups[p]]
    
    if len(groups_list) >= 2:
        kw_result = kruskal_wallis_test(groups_list)
        print(f"\nKruskal-Wallis test across partition counts:")
        print(f"  H-statistic: {kw_result['statistic']:.3f}")
        print(f"  p-value: {kw_result['pvalue']:.4f}")
        print(f"  Significant: {kw_result['significant']}")
    
    # Summary table
    print(f"\nPartition Count | N configs | Median TSH AUC | % Reduction vs Baseline")
    print("-" * 70)
    for p in [2, 4, 8, 16]:
        if partition_groups[p]:
            median_auc = np.median(partition_groups[p])
            reduction = ((baseline_auc - median_auc) / baseline_auc) * 100
            print(f"{p:15} | {len(partition_groups[p]):9} | {median_auc:14.2f} | {reduction:23.1f}%")


def analyze_timing_effect(output_dir: Path):
    """
    Analyze effect of timing jitter on adaptation lag.
    """
    print("\n" + "="*60)
    print("ANALYSIS: Effect of Timing Jitter on Adaptation Lag")
    print("="*60)
    
    results_dir = output_dir / 'results'
    
    # Group by jitter level
    jitter_groups = {'j0': [], 'j10': [], 'j30': []}
    
    for config_dir in results_dir.iterdir():
        if not config_dir.is_dir() or 'baseline' in config_dir.name:
            continue
        
        summary_path = config_dir / 'summary.json'
        if not summary_path.exists():
            continue
        
        with open(summary_path) as f:
            data = json.load(f)
        
        config_id = data['config_id']
        median_lag = data.get('adaptation_lag_median')
        
        if median_lag is not None:
            if '_j0_' in config_id:
                jitter_groups['j0'].append(median_lag)
            elif '_j10_' in config_id:
                jitter_groups['j10'].append(median_lag)
            elif '_j30_' in config_id:
                jitter_groups['j30'].append(median_lag)
    
    # Statistical test
    groups_dict = {k: v for k, v in jitter_groups.items() if v}
    
    if len(groups_dict) >= 2:
        pairwise_df = pairwise_mann_whitney_bonferroni(groups_dict)
        print("\nPairwise Mann-Whitney tests (Bonferroni corrected):")
        print(pairwise_df.to_string(index=False))
    
    # Summary
    print(f"\nJitter Level | N configs | Median Adaptation Lag (episodes)")
    print("-" * 60)
    for jitter_name in ['j0', 'j10', 'j30']:
        if jitter_groups[jitter_name]:
            median_lag = np.median(jitter_groups[jitter_name])
            jitter_pct = jitter_name[1:]
            print(f"{jitter_pct}%{' '*9} | {len(jitter_groups[jitter_name]):9} | {median_lag:30.1f}")


def generate_results_table(output_dir: Path, output_csv: Path):
    """
    Generate consolidated results table across all experiments.
    """
    results_dir = output_dir / 'results'
    
    table_data = []
    
    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        
        summary_path = config_dir / 'summary.json'
        if not summary_path.exists():
            continue
        
        with open(summary_path) as f:
            data = json.load(f)
        
        row = {
            'config_id': data['config_id'],
            'windowed_tsh_auc_median': data['windowed_tsh_auc_median'],
            'tsh_auc_iqr_lower': data['windowed_tsh_auc_iqr'][0],
            'tsh_auc_iqr_upper': data['windowed_tsh_auc_iqr'][1],
            'median_adaptation_lag': data.get('adaptation_lag_median'),
            'adaptation_lag_90pct': data.get('adaptation_lag_90pct'),
            'mean_qos_proxy_ms': data['mean_qos_proxy_ms'],
            'mean_flow_mods': data['mean_flow_mods']
        }
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    df.to_csv(output_csv, index=False)
    
    print(f"\nResults table saved to: {output_csv}")
    print(f"Total configurations: {len(df)}")
    
    return df


def main():
    if len(sys.argv) < 2:
        output_dir = Path('output')
    else:
        output_dir = Path(sys.argv[1])
    
    baseline_config_id = 'baseline_static_p4_b200'
    
    print("="*60)
    print("STATISTICAL ANALYSIS OF EXPERIMENT RESULTS")
    print("="*60)
    
    # Generate results table
    results_csv = output_dir / 'consolidated_results.csv'
    df = generate_results_table(output_dir, results_csv)
    
    # Analyze partition effect
    analyze_partition_effect(output_dir, baseline_config_id)
    
    # Analyze timing effect
    analyze_timing_effect(output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()