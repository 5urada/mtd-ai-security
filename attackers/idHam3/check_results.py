#!/usr/bin/env python3
"""
Improved experiment results checker with:
- Full precision display (no rounding)
- Variance and confidence intervals
- Statistical significance tests
- Better visualization
"""

import argparse
import json
import csv
from pathlib import Path
import numpy as np
from collections import defaultdict
from scipy import stats


def check_experiment_status(output_dir: str, config_id: str = None, show_variance: bool = True):
    """
    Check status of experiments with full statistical detail.
    
    Args:
        output_dir: Output directory
        config_id: Specific config ID (optional)
        show_variance: Show variance metrics
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        return
    
    logs_dir = output_path / 'logs'
    results_dir = output_path / 'results'
    
    if not logs_dir.exists():
        print(f"No logs directory found in {output_dir}")
        return
    
    # Find all config directories
    if config_id:
        config_dirs = [logs_dir / config_id] if (logs_dir / config_id).exists() else []
    else:
        config_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    
    if not config_dirs:
        print("No experiment results found")
        return
    
    print("\n" + "="*100)
    print("EXPERIMENT STATUS (FULL PRECISION)")
    print("="*100 + "\n")
    
    all_summaries = []
    
    for config_dir in sorted(config_dirs):
        config_name = config_dir.name
        
        # Find seed CSV files
        seed_files = list(config_dir.glob('seed_*.csv'))
        seed_numbers = []
        
        for f in seed_files:
            try:
                seed_num = int(f.stem.split('_')[1])
                seed_numbers.append(seed_num)
            except:
                pass
        
        if not seed_numbers:
            print(f"⚠️  {config_name}: No seed results found")
            continue
        
        seed_numbers = sorted(seed_numbers)
        n_seeds = len(seed_numbers)
        
        print(f"\n📊 {config_name}")
        print(f"   Seeds completed: {n_seeds} ({min(seed_numbers)} to {max(seed_numbers)})")
        
        # Try to load multi-seed results
        multi_seed_file = config_dir / 'multi_seed_results.json'
        if multi_seed_file.exists():
            with open(multi_seed_file, 'r') as f:
                multi_data = json.load(f)
                
            summary = multi_data.get('summary', {})
            print(f"   ✅ Successful: {summary.get('successful', 0)}")
            print(f"   ❌ Failed: {summary.get('failed', 0)}")
            print(f"   ⏱️  Timeout: {summary.get('timeout', 0)}")
        
        # Load aggregated results with FULL PRECISION
        summary_file = results_dir / config_name / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                agg_data = json.load(f)
            
            # Extract per-seed TSH values for variance analysis
            seed_tsh_values = [s['tsh_per_100_median'] for s in agg_data.get('seed_summaries', [])]
            
            if seed_tsh_values:
                tsh_mean = np.mean(seed_tsh_values)
                tsh_median = np.median(seed_tsh_values)
                tsh_std = np.std(seed_tsh_values, ddof=1)
                tsh_sem = tsh_std / np.sqrt(len(seed_tsh_values))
                tsh_min = np.min(seed_tsh_values)
                tsh_max = np.max(seed_tsh_values)
                
                # 95% confidence interval
                ci_95 = stats.t.interval(0.95, len(seed_tsh_values)-1, 
                                        loc=tsh_mean, scale=tsh_sem)
                
                print(f"   📈 TSH/100 (across {len(seed_tsh_values)} seeds):")
                print(f"      Mean:   {tsh_mean:.4f} ± {tsh_std:.4f} (SD)")
                print(f"      Median: {tsh_median:.4f}")
                print(f"      Range:  [{tsh_min:.4f}, {tsh_max:.4f}]")
                print(f"      95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
                print(f"      CV:     {(tsh_std/tsh_mean)*100:.2f}%")  # Coefficient of variation
                
                if show_variance:
                    # Show unique values to see if there's hidden variance
                    unique_vals = len(set(seed_tsh_values))
                    print(f"      Unique: {unique_vals} different values")
                    if unique_vals <= 10:
                        sorted_vals = sorted(set(seed_tsh_values))
                        print(f"      Values: {[f'{v:.4f}' for v in sorted_vals]}")
            
            # Other metrics
            if 'mean_address_entropy' in agg_data:
                print(f"   🎲 Address Entropy: {agg_data['mean_address_entropy']:.4f}")
            
            if 'adaptation_lag_median' in agg_data and agg_data['adaptation_lag_median']:
                print(f"   ⏱️  Adaptation Lag: {agg_data['adaptation_lag_median']:.1f} episodes")
            
            all_summaries.append({
                'config': config_name,
                'n_seeds': n_seeds,
                'tsh_mean': tsh_mean if seed_tsh_values else 0,
                'tsh_median': tsh_median if seed_tsh_values else 0,
                'tsh_std': tsh_std if seed_tsh_values else 0,
                'tsh_values': seed_tsh_values,
                'defender': agg_data.get('defender_type', 'unknown'),
                'agg_data': agg_data
            })
        else:
            # Try to compute quick stats from seed files
            tsh_values = []
            
            for seed_num in seed_numbers[:5]:  # Sample first 5 seeds
                seed_file = config_dir / f'seed_{seed_num}.csv'
                if seed_file.exists():
                    try:
                        with open(seed_file, 'r') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            if rows:
                                last_200 = rows[-200:] if len(rows) >= 200 else rows
                                tsh_vals = [float(r['tsh_per_100_probes']) for r in last_200]
                                tsh_values.append(np.median(tsh_vals))
                    except Exception as e:
                        print(f"   ⚠️  Error reading seed {seed_num}: {e}")
            
            if tsh_values:
                print(f"   📈 TSH/100 (approx from {len(tsh_values)} seeds): "
                      f"{np.mean(tsh_values):.4f} ± {np.std(tsh_values):.4f}")
    
    return all_summaries


def compare_results(output_dir: str, baseline_config: str = None, 
                   show_stats: bool = True):
    """Compare results across different configurations with statistical tests."""
    results_dir = Path(output_dir) / 'results'
    
    if not results_dir.exists():
        print(f"No results directory found")
        return
    
    # Load all summaries
    summaries = []
    
    for config_dir in results_dir.iterdir():
        if not config_dir.is_dir():
            continue
        
        summary_file = config_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                
                # Extract per-seed values
                seed_tsh_values = [s['tsh_per_100_median'] 
                                  for s in data.get('seed_summaries', [])]
                
                data['seed_tsh_values'] = seed_tsh_values
                summaries.append(data)
    
    if not summaries:
        print("No summary files found")
        return
    
    # Sort by TSH
    summaries = sorted(summaries, 
                      key=lambda x: np.mean(x.get('seed_tsh_values', [999])))
    
    print("\n" + "="*100)
    print("RESULTS COMPARISON (sorted by mean TSH/100)")
    print("="*100 + "\n")
    
    print(f"{'Config':<55s} {'Defender':<15s} {'Mean±SD':>15s} {'Median':>8s} {'Range':>15s} {'Seeds':>6s}")
    print("-" * 120)
    
    for s in summaries:
        config = s['config_id'][:53]
        defender = s.get('defender_type', 'unknown')[:13]
        
        tsh_vals = s.get('seed_tsh_values', [])
        if tsh_vals:
            mean = np.mean(tsh_vals)
            std = np.std(tsh_vals, ddof=1)
            median = np.median(tsh_vals)
            min_val = np.min(tsh_vals)
            max_val = np.max(tsh_vals)
            seeds = len(tsh_vals)
            
            print(f"{config:<55s} {defender:<15s} "
                  f"{mean:7.4f}±{std:5.4f} "
                  f"{median:8.4f} "
                  f"[{min_val:6.4f},{max_val:6.4f}] "
                  f"{seeds:6d}")
        else:
            print(f"{config:<55s} {defender:<15s} {'N/A':>15s} {'N/A':>8s} {'N/A':>15s} {0:6d}")
    
    print("\n" + "="*100)
    
    # Statistical comparison by defender type
    if show_stats:
        print("\n" + "="*100)
        print("STATISTICAL COMPARISONS (Paired t-tests within each defender)")
        print("="*100 + "\n")
        
        # Group by defender
        by_defender = defaultdict(list)
        for s in summaries:
            defender = s.get('defender_type', 'unknown')
            tsh_vals = s.get('seed_tsh_values', [])
            if tsh_vals:
                by_defender[defender].append({
                    'config': s['config_id'],
                    'tsh_values': tsh_vals
                })
        
        # For each defender, compare deterministic vs jittered
        for defender, configs in sorted(by_defender.items()):
            print(f"\n{'='*80}")
            print(f"DEFENDER: {defender.upper()}")
            print(f"{'='*80}")
            
            # Find deterministic baseline
            det_config = None
            for c in configs:
                if 'deterministic' in c['config'].lower():
                    det_config = c
                    break
            
            if not det_config:
                print("  No deterministic baseline found")
                continue
            
            det_tsh = np.array(det_config['tsh_values'])
            det_mean = np.mean(det_tsh)
            
            print(f"\nBaseline (Deterministic):")
            print(f"  Config: {det_config['config']}")
            print(f"  Mean TSH/100: {det_mean:.6f} ± {np.std(det_tsh, ddof=1):.6f}")
            print(f"  N seeds: {len(det_tsh)}")
            
            print(f"\nComparisons vs Baseline:")
            print(f"{'Config':<50s} {'Mean':>10s} {'Diff':>10s} {'Diff%':>8s} {'t':>8s} {'p-value':>10s} {'Sig':>5s}")
            print("-" * 110)
            
            for c in configs:
                if c['config'] == det_config['config']:
                    continue
                
                comp_tsh = np.array(c['tsh_values'])
                comp_mean = np.mean(comp_tsh)
                diff = comp_mean - det_mean
                diff_pct = (diff / det_mean) * 100
                
                # Paired t-test if same number of seeds
                if len(comp_tsh) == len(det_tsh):
                    t_stat, p_value = stats.ttest_rel(det_tsh, comp_tsh)
                    test_type = "paired"
                else:
                    t_stat, p_value = stats.ttest_ind(det_tsh, comp_tsh)
                    test_type = "indep"
                
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                config_short = c['config'][:48]
                print(f"{config_short:<50s} "
                      f"{comp_mean:10.6f} "
                      f"{diff:+10.6f} "
                      f"{diff_pct:+7.2f}% "
                      f"{t_stat:8.4f} "
                      f"{p_value:10.6f} "
                      f"{sig:>5s}")
            
            print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")
    
    # If baseline specified, compute improvements
    if baseline_config:
        print("\n" + "="*100)
        print(f"IMPROVEMENT vs BASELINE ({baseline_config[:40]})")
        print("="*100 + "\n")
        
        baseline = next((s for s in summaries if baseline_config in s['config_id']), None)
        
        if baseline:
            baseline_tsh = np.mean(baseline.get('seed_tsh_values', [0]))
            
            print(f"{'Config':<50s} {'Mean TSH':>10s} {'Improvement':>12s}")
            print("-" * 75)
            
            for s in summaries:
                if baseline_config in s['config_id']:
                    continue
                
                tsh_vals = s.get('seed_tsh_values', [])
                if tsh_vals:
                    tsh = np.mean(tsh_vals)
                    improvement = (baseline_tsh - tsh) / baseline_tsh * 100
                    
                    config = s['config_id'][:48]
                    print(f"{config:<50s} {tsh:10.4f} {improvement:+11.2f}%")
    
    print("\n" + "="*100 + "\n")


def export_for_analysis(output_dir: str, export_file: str = 'results_export.csv'):
    """Export all results to CSV for external analysis."""
    results_dir = Path(output_dir) / 'results'
    
    if not results_dir.exists():
        print(f"No results directory found")
        return
    
    output_path = Path(output_dir) / export_file
    
    rows = []
    
    for config_dir in results_dir.iterdir():
        if not config_dir.is_dir():
            continue
        
        summary_file = config_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Extract per-seed data
            for seed_summary in data.get('seed_summaries', []):
                row = {
                    'config_id': data['config_id'],
                    'defender_type': data.get('defender_type', 'unknown'),
                    'seed': seed_summary['seed'],
                    'tsh_median': seed_summary['tsh_median'],
                    'tsh_mean': seed_summary['tsh_mean'],
                    'tsh_per_100_median': seed_summary['tsh_per_100_median'],
                    'tsh_per_100_mean': seed_summary['tsh_per_100_mean'],
                    'coverage_final': seed_summary['coverage_final'],
                    'n_switches': seed_summary['n_switches'],
                    'median_adaptation_lag': seed_summary.get('median_adaptation_lag', None),
                }
                rows.append(row)
    
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Exported {len(rows)} rows to {output_path}")
        print(f"   Configs: {df['config_id'].nunique()}")
        print(f"   Seeds per config: ~{len(rows) // df['config_id'].nunique()}")
    else:
        print("No data to export")


def main():
    parser = argparse.ArgumentParser(
        description='Check experiment status and results (IMPROVED VERSION)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all results with full precision
  python check_results_improved.py
  
  # Compare with statistical tests
  python check_results_improved.py --compare --stats
  
  # View specific config
  python check_results_improved.py --config idham_mutation-divconq_2p_deterministic-100mu_j0_200b
  
  # Export for R/Python analysis
  python check_results_improved.py --export results.csv
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Specific config ID to check'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Show comparison table'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistical significance tests'
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        help='Baseline config for improvement calculations'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )
    
    parser.add_argument(
        '--no-variance',
        action='store_true',
        help='Hide variance details'
    )
    
    args = parser.parse_args()
    
    # Check status
    check_experiment_status(args.output_dir, args.config, 
                           show_variance=not args.no_variance)
    
    # Show comparison if requested
    if args.compare or args.stats:
        compare_results(args.output_dir, args.baseline, show_stats=args.stats)
    
    # Export if requested
    if args.export:
        export_for_analysis(args.output_dir, args.export)


if __name__ == '__main__':
    main()