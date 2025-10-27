#!/usr/bin/env python3
"""
Equivalence testing (TOST) for NDV experiments.
Tests whether ID-HAM and non-learning MTD baselines are equivalent under NDV.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import re


def parse_config_id(config_id: str) -> Dict:
    """
    Parse NDV config ID to extract parameters.
    
    Format: {defender}-divconq_{p}p_{family}-{mu}mu_{jit}_{B}b
    Example: idham-divconq_2p_poisson-100mu_j0_200b
    """
    pattern = r'([a-z_]+)-divconq_(\d+)p_([a-z\d_]+)-(\d+)mu_j(\d+)_(\d+)b'
    match = re.match(pattern, config_id)
    
    if not match:
        return {'config_id': config_id, 'parse_error': True}
    
    defender, p, family, mu, jit, B = match.groups()
    
    return {
        'config_id': config_id,
        'defender': defender,
        'partitions': int(p),
        'family': family,
        'mean_interval': int(mu),
        'jitter_pct': int(jit) / 100.0,
        'probe_budget': int(B),
    }


def run_tost(x: np.ndarray, y: np.ndarray, delta_rel: float = 0.10, 
             alpha: float = 0.05) -> Dict:
    """
    Two One-Sided Tests (TOST) for equivalence.
    
    Tests if two groups are equivalent within ±delta_rel of the reference mean.
    
    Args:
        x: Reference group (e.g., ID-HAM)
        y: Comparison group (e.g., static mask)
        delta_rel: Relative equivalence margin (0.10 = ±10%)
        alpha: Significance level (0.05 = 95% CI)
    
    Returns:
        Dict with test results
    """
    n_x, n_y = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
    
    # Pooled standard error
    se = np.sqrt(std_x**2/n_x + std_y**2/n_y)
    
    # Absolute equivalence bounds based on reference mean
    delta_abs = delta_rel * mean_x
    lower_bound = mean_x - delta_abs
    upper_bound = mean_x + delta_abs
    
    # TOST: Test H0: |mean_y - mean_x| >= delta
    # Equivalent to two one-sided tests:
    #   1) H0: mean_y - mean_x <= -delta  vs  H1: mean_y - mean_x > -delta
    #   2) H0: mean_y - mean_x >= delta   vs  H1: mean_y - mean_x < delta
    
    diff = mean_y - mean_x
    
    # Test 1: Is y greater than lower bound?
    t1 = (diff - (-delta_abs)) / se
    p1 = stats.t.cdf(-t1, df=n_x + n_y - 2)  # One-sided
    
    # Test 2: Is y less than upper bound?
    t2 = (diff - delta_abs) / se  
    p2 = stats.t.cdf(t2, df=n_x + n_y - 2)  # One-sided
    
    # TOST p-value is max of the two
    p_tost = max(p1, p2)
    
    # Confidence interval (90% for alpha=0.05 TOST)
    t_crit = stats.t.ppf(1 - alpha, df=n_x + n_y - 2)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se
    
    # Equivalent if TOST rejects (p < alpha)
    equivalent = p_tost < alpha
    
    return {
        'mean_x': float(mean_x),
        'mean_y': float(mean_y),
        'diff': float(diff),
        'diff_rel_pct': float((diff / mean_x) * 100) if mean_x != 0 else 0,
        'delta_abs': float(delta_abs),
        'delta_rel_pct': float(delta_rel * 100),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': float(p_tost),
        'p1': float(p1),
        'p2': float(p2),
        't1': float(t1),
        't2': float(t2),
        'equivalent': bool(equivalent),
        'n_x': int(n_x),
        'n_y': int(n_y),
    }


def run_wilcoxon_tost(x: np.ndarray, y: np.ndarray, delta_rel: float = 0.10,
                      alpha: float = 0.05) -> Dict:
    """
    Non-parametric TOST using Wilcoxon rank-sum test.
    Fallback for non-normal distributions.
    """
    mean_x = np.median(x)
    mean_y = np.median(y)
    delta_abs = delta_rel * mean_x
    
    # Shift y by bounds and test
    # Test 1: Is y shifted by -delta greater than x?
    y_shifted_lower = y + delta_abs
    stat1, p1 = stats.mannwhitneyu(y_shifted_lower, x, alternative='greater')
    
    # Test 2: Is y shifted by +delta less than x?
    y_shifted_upper = y - delta_abs
    stat2, p2 = stats.mannwhitneyu(y_shifted_upper, x, alternative='less')
    
    p_wilcoxon_tost = max(p1, p2)
    equivalent = p_wilcoxon_tost < alpha
    
    return {
        'median_x': float(mean_x),
        'median_y': float(mean_y),
        'diff': float(mean_y - mean_x),
        'p_value': float(p_wilcoxon_tost),
        'p1': float(p1),
        'p2': float(p2),
        'equivalent': bool(equivalent),
    }


def load_results_for_stratum(results_dir: Path, p: int, B: int, mu: int, 
                             family: str) -> Dict[str, List[float]]:
    """
    Load results for a specific stratum (p, B, μ, family).
    
    Returns dict mapping defender type to list of per-seed metrics.
    """
    results = {'idham': [], 'static_mask': [], 'random_mtd': []}
    
    for defender_type in results.keys():
        # Build config ID pattern
        config_id_pattern = f"{defender_type}-divconq_{p}p_{family}-{mu}mu"
        
        # Find matching summary file
        for summary_file in results_dir.glob(f"*{config_id_pattern}*/summary.json"):
            with open(summary_file, 'r') as f:
                data = json.load(f)
                
                # Extract per-seed TSH/100 values
                for seed_summary in data.get('seed_summaries', []):
                    tsh_per_100 = seed_summary.get('tsh_per_100_median')
                    if tsh_per_100 is not None:
                        results[defender_type].append(tsh_per_100)
    
    return results


def analyze_stratum(results_dir: Path, p: int, B: int, mu: int, family: str,
                    delta_rel: float = 0.10, alpha: float = 0.05) -> Dict:
    """
    Analyze one stratum: run TOST tests comparing defenders.
    """
    # Load data
    data = load_results_for_stratum(results_dir, p, B, mu, family)
    
    if not data['idham']:
        return {
            'stratum': f"p{p}_b{B}_mu{mu}_{family}",
            'error': 'No ID-HAM data found'
        }
    
    idham_vals = np.array(data['idham'])
    
    # Test ID-HAM vs Static Mask
    tost_vs_static = None
    wilcoxon_vs_static = None
    if data['static_mask']:
        static_vals = np.array(data['static_mask'])
        tost_vs_static = run_tost(idham_vals, static_vals, delta_rel, alpha)
        wilcoxon_vs_static = run_wilcoxon_tost(idham_vals, static_vals, delta_rel, alpha)
    
    # Test ID-HAM vs Random MTD
    tost_vs_random = None
    wilcoxon_vs_random = None
    if data['random_mtd']:
        random_vals = np.array(data['random_mtd'])
        tost_vs_random = run_tost(idham_vals, random_vals, delta_rel, alpha)
        wilcoxon_vs_random = run_wilcoxon_tost(idham_vals, random_vals, delta_rel, alpha)
    
    return {
        'stratum': f"p{p}_b{B}_mu{mu}_{family}",
        'p': p,
        'B': B,
        'mu': mu,
        'family': family,
        'n_seeds': {
            'idham': len(data['idham']),
            'static_mask': len(data['static_mask']),
            'random_mtd': len(data['random_mtd']),
        },
        'idham_mean': float(np.mean(idham_vals)),
        'idham_median': float(np.median(idham_vals)),
        'static_mask_mean': float(np.mean(data['static_mask'])) if data['static_mask'] else None,
        'static_mask_median': float(np.median(data['static_mask'])) if data['static_mask'] else None,
        'random_mtd_mean': float(np.mean(data['random_mtd'])) if data['random_mtd'] else None,
        'random_mtd_median': float(np.median(data['random_mtd'])) if data['random_mtd'] else None,
        'tost_vs_static': tost_vs_static,
        'tost_vs_random': tost_vs_random,
        'wilcoxon_vs_static': wilcoxon_vs_static,
        'wilcoxon_vs_random': wilcoxon_vs_random,
    }


def format_results_table(results: List[Dict]) -> pd.DataFrame:
    """Format results as a table for easy interpretation."""
    rows = []
    
    for r in results:
        if 'error' in r:
            continue
        
        row = {
            'stratum': r['stratum'],
            'p': r['p'],
            'B': r['B'],
            'mu': r['mu'],
            'family': r['family'],
            'idham_mean': r['idham_mean'],
            'idham_median': r['idham_median'],
        }
        
        # Static mask comparison
        if r['tost_vs_static']:
            row['static_mean'] = r['static_mask_mean']
            row['static_diff_pct'] = r['tost_vs_static']['diff_rel_pct']
            row['static_equiv'] = r['tost_vs_static']['equivalent']
            row['static_p'] = r['tost_vs_static']['p_value']
        
        # Random MTD comparison
        if r['tost_vs_random']:
            row['random_mean'] = r['random_mtd_mean']
            row['random_diff_pct'] = r['tost_vs_random']['diff_rel_pct']
            row['random_equiv'] = r['tost_vs_random']['equivalent']
            row['random_p'] = r['tost_vs_random']['p_value']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze equivalence across NDV experiments')
    parser.add_argument('--root', type=str, default='output/results',
                       help='Root directory containing results')
    parser.add_argument('--delta', type=float, default=0.10,
                       help='Equivalence margin (relative, e.g., 0.10 = ±10%%)')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level')
    parser.add_argument('--output-dir', type=str, default='analysis/equiv',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing equivalence with δ = ±{args.delta*100}%")
    print(f"Results directory: {results_dir}")
    
    # Define strata to analyze
    # For minimal grid: p=2, B=200, mu=100
    strata = [
        (2, 200, 100, 'deterministic'),
        (2, 200, 100, 'uniform_j10'),
        (2, 200, 100, 'uniform_j30'),
        (2, 200, 100, 'poisson'),
    ]
    
    results = []
    for p, B, mu, family in strata:
        print(f"\nAnalyzing stratum: p={p}, B={B}, μ={mu}, family={family}")
        result = analyze_stratum(results_dir, p, B, mu, family, args.delta, args.alpha)
        results.append(result)
        
        # Print summary
        if 'error' not in result:
            print(f"  ID-HAM mean: {result['idham_mean']:.2f}")
            
            if result['tost_vs_static']:
                equiv_static = '✅ EQUIVALENT' if result['tost_vs_static']['equivalent'] else '❌ NOT EQUIV'
                print(f"  vs Static: {result['static_mask_mean']:.2f} "
                      f"({result['tost_vs_static']['diff_rel_pct']:+.1f}%) {equiv_static}")
            
            if result['tost_vs_random']:
                equiv_random = '✅ EQUIVALENT' if result['tost_vs_random']['equivalent'] else '❌ NOT EQUIV'
                print(f"  vs Random: {result['random_mtd_mean']:.2f} "
                      f"({result['tost_vs_random']['diff_rel_pct']:+.1f}%) {equiv_random}")
    
    # Save detailed results
    output_file = output_dir / 'detailed_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary table
    table = format_results_table(results)
    table_file = output_dir / 'summary_table.csv'
    table.to_csv(table_file, index=False)
    print(f"Summary table saved to: {table_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("EQUIVALENCE TEST SUMMARY")
    print("="*80)
    print(table.to_string(index=False))
    
    # Overall conclusion
    if len(table) > 0:
        static_equiv_rate = table['static_equiv'].mean() if 'static_equiv' in table.columns else 0
        random_equiv_rate = table['random_equiv'].mean() if 'random_equiv' in table.columns else 0
        
        print(f"\n{'='*80}")
        print(f"Overall Equivalence Rates (within ±{args.delta*100}%):")
        print(f"  ID-HAM vs Static Mask: {static_equiv_rate*100:.0f}% of strata")
        print(f"  ID-HAM vs Random MTD:  {random_equiv_rate*100:.0f}% of strata")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()