#!/usr/bin/env python3
"""
Batch experiment runner for multiple configurations and seeds.

Runs the full experimental grid across all attacker strategies and defender types.
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys


def find_config_files(config_dir: str, pattern: str = "*.yaml") -> list:
    """Find all config files in directory."""
    config_path = Path(config_dir)
    
    if not config_path.exists():
        print(f"ERROR: Config directory not found: {config_dir}")
        return []
    
    configs = list(config_path.glob(pattern))
    configs.extend(config_path.glob("**/" + pattern))
    
    return sorted(set(configs))


def run_batch_experiments(config_files: list, seed_start: int, seed_end: int,
                          output_dir: str, timeout: int = 3600,
                          dry_run: bool = False):
    """
    Run experiments for multiple configs.
    
    Args:
        config_files: List of config file paths
        seed_start: Starting seed
        seed_end: Ending seed (exclusive)
        output_dir: Output directory
        timeout: Timeout per seed
        dry_run: If True, just print what would run
    """
    n_seeds = seed_end - seed_start
    total_experiments = len(config_files) * n_seeds
    
    print("\n" + "="*80)
    print("BATCH EXPERIMENT RUNNER")
    print("="*80)
    print(f"Configurations: {len(config_files)}")
    print(f"Seeds per config: {n_seeds} (seeds {seed_start}-{seed_end-1})")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    print(f"Timeout per seed: {timeout}s ({timeout/60:.1f} min)")
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No experiments will actually run")
    
    print("="*80 + "\n")
    
    # Show configs
    print("Configurations to run:")
    for i, config in enumerate(config_files, 1):
        print(f"  {i}. {config.name}")
    
    if dry_run:
        print("\n" + "="*80)
        print("DRY RUN COMPLETE")
        print("="*80)
        print("\nTo run for real, remove the --dry-run flag")
        return
    
    # Confirm
    print("\n" + "="*80)
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Track results
    all_results = []
    start_time = datetime.now()
    
    # Run each config
    for i, config_file in enumerate(config_files, 1):
        print("\n" + "="*80)
        print(f"CONFIG {i}/{len(config_files)}: {config_file.name}")
        print("="*80 + "\n")
        
        # Run multi-seed experiment for this config
        # Use the SAME Python executable that's running this script
        python_exe = sys.executable
        
        cmd = [
            python_exe, 'run_multi_seed_experiments.py',
            '--config', str(config_file),
            '--seed-range', str(seed_start), str(seed_end),
            '--output-dir', output_dir,
            '--timeout', str(timeout)
        ]
        
        print(f"Using Python: {python_exe}")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=False)
            
            status = 'success' if result.returncode == 0 else 'failed'
            all_results.append({
                'config': str(config_file),
                'status': status,
                'returncode': result.returncode
            })
            
        except Exception as e:
            print(f"ERROR running config {config_file}: {e}")
            all_results.append({
                'config': str(config_file),
                'status': 'error',
                'error': str(e)
            })
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("BATCH EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration/3600:.2f} hours")
    
    successes = [r for r in all_results if r['status'] == 'success']
    failures = [r for r in all_results if r['status'] != 'success']
    
    print(f"\n✅ Successful configs: {len(successes)}/{len(config_files)}")
    print(f"❌ Failed configs: {len(failures)}/{len(config_files)}")
    
    if failures:
        print("\nFailed configurations:")
        for r in failures:
            print(f"  - {Path(r['config']).name}")
    
    # Save summary
    summary_file = Path(output_dir) / 'batch_experiment_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'configs_total': len(config_files),
        'configs_successful': len(successes),
        'configs_failed': len(failures),
        'seeds_per_config': n_seeds,
        'total_experiments': total_experiments,
        'results': all_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Summary saved to: {summary_file}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run batch experiments across multiple configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configs in a directory with seeds 0-19
  python run_batch_experiments.py --config-dir configs/ndv_mutation/p2_b200_mu100 --seed-range 0 20
  
  # Dry run to see what would execute
  python run_batch_experiments.py --config-dir configs/ndv_mutation/p2_b200_mu100 --seed-range 0 5 --dry-run
  
  # Run only ID-HAM configs
  python run_batch_experiments.py --config-dir configs/ndv_mutation/p2_b200_mu100 --pattern "idham_mutation*.yaml" --seed-range 0 20
        """
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        required=True,
        help='Directory containing config files'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.yaml',
        help='Filename pattern for configs (default: *.yaml)'
    )
    
    parser.add_argument(
        '--seed-range',
        type=int,
        nargs=2,
        required=True,
        metavar=('START', 'END'),
        help='Range of seeds [start, end) (e.g., 0 20 for seeds 0-19)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout per seed in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would run without actually running'
    )
    
    args = parser.parse_args()
    
    # Find config files
    config_files = find_config_files(args.config_dir, args.pattern)
    
    if not config_files:
        print(f"ERROR: No config files found in {args.config_dir} matching {args.pattern}")
        sys.exit(1)
    
    # Run batch
    run_batch_experiments(
        config_files=config_files,
        seed_start=args.seed_range[0],
        seed_end=args.seed_range[1],
        output_dir=args.output_dir,
        timeout=args.timeout,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()