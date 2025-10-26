import subprocess
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_config(config_path):
    """Run a single config using the current Python interpreter."""
    # Use sys.executable to ensure same Python as parent process
    cmd = [sys.executable, 'run_experiment.py', '--config', config_path]
    print(f"Starting: {config_path}")
    result = subprocess.run(cmd, capture_output=False)
    print(f"Completed: {config_path}")
    return config_path, result.returncode

def main():
    # Get all configs
    config_dir = Path('configs/ndv/p2_b200_mu100')
    configs = list(config_dir.glob('*.yaml'))
    
    print(f"Found {len(configs)} configs")
    print(f"Using Python: {sys.executable}")  # Show which Python
    print("Running with 8 parallel workers...")
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_config, str(cfg)): cfg for cfg in configs}
        
        completed = 0
        for future in as_completed(futures):
            config_path, returncode = future.result()
            completed += 1
            print(f"\n[{completed}/{len(configs)}] Finished: {config_path}")
            if returncode != 0:
                print(f"  WARNING: Exit code {returncode}")
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*60)
    print(f"\nRun analysis:")
    print("python analyze_equivalence.py --root output/results --delta 0.10")

if __name__ == '__main__':
    main()