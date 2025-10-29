#!/usr/bin/env python3
"""
Check experiment results and progress.

Quickly view status of multi-seed experiments.
"""

import argparse
import json
import csv
from pathlib import Path
import numpy as np
from collections import defaultdict


def check_experiment_status(output_dir: str, config_id: str = None):
    """
    Check status of experiments.
    
    Args:
        output_dir: Output directory
        config_id: Specific config ID (optional)
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
    
    print("\n" + "="*80)
    print("EXPERIMENT STATUS")
    print("="*80 + "\n")
    
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
        
        # Load aggregated results
        summary_file = results_dir / config_name / 'summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                agg_data = json.load(f)
            
            tsh_median = agg_data.get('tsh_per_100_median_across_seeds', 0)
            tsh_iqr = agg_data.get('tsh_per_100_iqr', [0, 0])
            
            print(f"   📈 TSH/100 (median): {tsh_median:.2f} (IQR: {tsh_iqr[0]:.2f}-{tsh_iqr[1]:.2f})")
            
            if 'mean_address_entropy' in agg_data:
                print(f"   🎲 Address Entropy: {agg_data['mean_address_entropy']:.2f}")
            
            if 'adaptation_lag_median' in agg_data and agg_data['adaptation_lag_median']:
                print(f"   ⏱️  Adaptation Lag: {agg_data['adaptation_lag_median']:.0f} episodes")
            
            all_summaries.append({
                'config': config_name,
                'n_seeds': n_seeds,
                'tsh_median': tsh_median,
                'defender': agg_data.get('defender_type', 'unknown')
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
                print(f"   📈 TSH/100 (approx from {len(tsh_values)} seeds): {np.median(tsh_values):.2f}")
    
    # Summary table
    if all_summaries:
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80 + "\n")
        
        # Group by defender
        by_defender = defaultdict(list)
        for s in all_summaries:
            by_defender[s['defender']].append(s)
        
        for defender, configs in sorted(by_defender.items()):
            print(f"\n{defender.upper()}:")
            for c in sorted(configs, key=lambda x: x['config']):
                print(f"  {c['config'][:60]:60s}  TSH={c['tsh_median']:5.2f}  seeds={c['n_seeds']:2d}")
    
    # Check for batch summary
    batch_summary = output_path / 'batch_experiment_summary.json'
    if batch_summary.exists():
        print("\n" + "="*80)
        print("BATCH EXPERIMENT SUMMARY")
        print("="*80 + "\n")
        
        with open(batch_summary, 'r') as f:
            batch_data = json.load(f)
        
        print(f"Total configs: {batch_data.get('configs_total', 0)}")
        print(f"✅ Successful: {batch_data.get('configs_successful', 0)}")
        print(f"❌ Failed: {batch_data.get('configs_failed', 0)}")
        print(f"Duration: {batch_data.get('duration_seconds', 0)/3600:.2f} hours")
    
    print("\n" + "="*80 + "\n")


def compare_results(output_dir: str, baseline_config: str = None):
    """Compare results across different configurations."""
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
                summaries.append(data)
    
    if not summaries:
        print("No summary files found")
        return
    
    # Sort by TSH
    summaries = sorted(summaries, key=lambda x: x.get('tsh_per_100_median_across_seeds', 999))
    
    print("\n" + "="*80)
    print("RESULTS COMPARISON (sorted by TSH/100)")
    print("="*80 + "\n")
    
    print(f"{'Config':<50s} {'Defender':<15s} {'TSH/100':>8s} {'Seeds':>6s} {'Entropy':>8s}")
    print("-" * 95)
    
    for s in summaries:
        config = s['config_id'][:48]
        defender = s.get('defender_type', 'unknown')[:13]
        tsh = s.get('tsh_per_100_median_across_seeds', 0)
        seeds = s.get('n_seeds', 0)
        entropy = s.get('mean_address_entropy', 0)
        
        print(f"{config:<50s} {defender:<15s} {tsh:8.2f} {seeds:6d} {entropy:8.2f}")
    
    print("\n" + "="*80)
    
    # If baseline specified, compute improvements
    if baseline_config:
        baseline = next((s for s in summaries if baseline_config in s['config_id']), None)
        
        if baseline:
            baseline_tsh = baseline['tsh_per_100_median_across_seeds']
            
            print(f"\nIMPROVEMENT vs BASELINE ({baseline_config[:40]}):")
            print("-" * 80)
            
            for s in summaries:
                if baseline_config in s['config_id']:
                    continue
                
                tsh = s['tsh_per_100_median_across_seeds']
                improvement = (baseline_tsh - tsh) / baseline_tsh * 100
                
                config = s['config_id'][:45]
                print(f"{config:<47s} {improvement:+6.1f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Check experiment status and results')
    
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
        '--baseline',
        type=str,
        help='Baseline config for improvement calculations'
    )
    
    args = parser.parse_args()
    
    # Check status
    check_experiment_status(args.output_dir, args.config)
    
    # Show comparison if requested
    if args.compare:
        compare_results(args.output_dir, args.baseline)


if __name__ == '__main__':
    main()