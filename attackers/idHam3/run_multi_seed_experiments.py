#!/usr/bin/env python3
"""
Multi-seed experiment runner for ID-HAM neural network defender.

This script:
1. Runs experiments across multiple seeds
2. Provides progress tracking
3. Handles errors gracefully
4. Generates summary statistics
"""

import argparse
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
import sys
import time


def load_config(config_path: str):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def run_single_seed(config_path: str, seed: int, output_dir: str, 
                    timeout: int = 3600) -> dict:
    """
    Run experiment for a single seed.
    
    Returns:
        dict with status, runtime, and error info
    """
    start_time = time.time()
    
    # Use the SAME Python executable that's running this script
    # This ensures we use the activated virtual environment
    python_exe = sys.executable
    
    cmd = [
        python_exe, 'run_experiment_mutation.py',
        '--config', config_path,
        '--seed', str(seed),
        '--output-dir', output_dir
    ]
    
    print(f"\n{'='*60}")
    print(f"Starting seed {seed}")
    print(f"Using Python: {python_exe}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Seed {seed} completed successfully in {runtime:.1f}s")
            return {
                'seed': seed,
                'status': 'success',
                'runtime': runtime,
                'stdout': result.stdout[-500:],  # Last 500 chars
                'stderr': result.stderr[-500:] if result.stderr else ''
            }
        else:
            print(f"❌ Seed {seed} failed with return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            return {
                'seed': seed,
                'status': 'failed',
                'runtime': runtime,
                'returncode': result.returncode,
                'stdout': result.stdout[-500:],
                'stderr': result.stderr[-500:]
            }
            
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print(f"⏱️  Seed {seed} timed out after {timeout}s")
        return {
            'seed': seed,
            'status': 'timeout',
            'runtime': runtime,
            'timeout': timeout
        }
        
    except Exception as e:
        runtime = time.time() - start_time
        print(f"💥 Seed {seed} crashed with exception: {e}")
        return {
            'seed': seed,
            'status': 'error',
            'runtime': runtime,
            'error': str(e)
        }


def run_multi_seed_experiment(config_path: str, seeds: list, output_dir: str,
                               parallel: bool = False, max_parallel: int = 4,
                               timeout: int = 3600):
    """
    Run experiments across multiple seeds.
    
    Args:
        config_path: Path to config YAML
        seeds: List of seed values to run
        output_dir: Output directory
        parallel: Whether to run in parallel (not implemented yet)
        max_parallel: Max parallel jobs
        timeout: Timeout per seed (seconds)
    """
    config = load_config(config_path)
    config_id = config['config_id']
    
    print("\n" + "="*80)
    print(f"MULTI-SEED EXPERIMENT: {config_id}")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(seeds)}")
    print(f"Defender type: {config.get('defender', {}).get('type', 'unknown')}")
    print(f"Output directory: {output_dir}")
    print(f"Timeout per seed: {timeout}s ({timeout/60:.1f} min)")
    print("="*80 + "\n")
    
    # Track results
    results = []
    start_time = time.time()
    
    # Run seeds sequentially (parallel coming later)
    for i, seed in enumerate(seeds, 1):
        print(f"\n>>> Running seed {seed} ({i}/{len(seeds)}) <<<")
        
        result = run_single_seed(config_path, seed, output_dir, timeout)
        results.append(result)
        
        # Progress update
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (len(seeds) - i)
        
        print(f"\n📊 Progress: {i}/{len(seeds)} ({i/len(seeds)*100:.1f}%)")
        print(f"⏱️  Elapsed: {elapsed/60:.1f} min | "
              f"Avg: {avg_time/60:.1f} min/seed | "
              f"Est. remaining: {remaining/60:.1f} min")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    successes = [r for r in results if r['status'] == 'success']
    failures = [r for r in results if r['status'] == 'failed']
    timeouts = [r for r in results if r['status'] == 'timeout']
    errors = [r for r in results if r['status'] == 'error']
    
    print(f"✅ Successful: {len(successes)}/{len(seeds)}")
    print(f"❌ Failed: {len(failures)}/{len(seeds)}")
    print(f"⏱️  Timeout: {len(timeouts)}/{len(seeds)}")
    print(f"💥 Errors: {len(errors)}/{len(seeds)}")
    print(f"\n⏱️  Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    
    if successes:
        avg_runtime = sum(r['runtime'] for r in successes) / len(successes)
        print(f"📈 Average runtime: {avg_runtime/60:.1f} minutes")
    
    # Save detailed results
    output_path = Path(output_dir)
    results_file = output_path / 'logs' / config_id / 'multi_seed_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'config_id': config_id,
        'config_path': str(config_path),
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'summary': {
            'total': len(seeds),
            'successful': len(successes),
            'failed': len(failures),
            'timeout': len(timeouts),
            'errors': len(errors)
        },
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {results_file}")
    
    # Print failures if any
    if failures or timeouts or errors:
        print("\n" + "="*80)
        print("FAILED SEEDS DETAILS")
        print("="*80)
        
        for r in failures + timeouts + errors:
            print(f"\nSeed {r['seed']} ({r['status']}):")
            if 'stderr' in r and r['stderr']:
                print(f"  Error: {r['stderr'][:200]}")
            if 'error' in r:
                print(f"  Exception: {r['error']}")
    
    print("\n" + "="*80 + "\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run ID-HAM experiments across multiple seeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run seeds 0-4 for a config
  python run_multi_seed_experiments.py --config configs/ndv_mutation/p2_b200_mu100/idham_mutation-divconq_2p_deterministic-100mu_j0_200b.yaml --seeds 0 1 2 3 4
  
  # Run seeds 0-19 (20 total)
  python run_multi_seed_experiments.py --config my_config.yaml --seed-range 0 20
  
  # Run with custom timeout (2 hours per seed)
  python run_multi_seed_experiments.py --config my_config.yaml --seed-range 0 10 --timeout 7200
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config YAML file'
    )
    
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        help='List of seeds to run (e.g., 0 1 2 3 4)'
    )
    
    parser.add_argument(
        '--seed-range',
        type=int,
        nargs=2,
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
        '--parallel',
        action='store_true',
        help='Run seeds in parallel (NOT YET IMPLEMENTED)'
    )
    
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=4,
        help='Maximum parallel jobs (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Determine seeds to run
    if args.seeds:
        seeds = args.seeds
    elif args.seed_range:
        seeds = list(range(args.seed_range[0], args.seed_range[1]))
    else:
        print("ERROR: Must specify either --seeds or --seed-range")
        parser.print_help()
        sys.exit(1)
    
    # Validate config exists
    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run experiments
    summary = run_multi_seed_experiment(
        config_path=args.config,
        seeds=seeds,
        output_dir=args.output_dir,
        parallel=args.parallel,
        max_parallel=args.max_parallel,
        timeout=args.timeout
    )
    
    # Exit code based on success rate
    success_rate = summary['summary']['successful'] / summary['summary']['total']
    if success_rate >= 0.95:
        sys.exit(0)  # All good
    elif success_rate >= 0.80:
        sys.exit(1)  # Some failures
    else:
        sys.exit(2)  # Many failures


if __name__ == '__main__':
    main()