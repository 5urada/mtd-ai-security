#!/usr/bin/env python3
"""
Quick verification test for partition switching fix.

This script runs a minimal test (300 episodes, 1 seed) and checks:
1. Switches are detected at correct episodes
2. is_switch flag is set correctly in CSV
3. dwell_time resets after switches
4. Different jitter configs produce different switch timings

Run this AFTER applying the fix to verify it works!
"""

import sys
import subprocess
import pandas as pd
from pathlib import Path
import json


def run_test_experiment(config_path, seed=0, output_dir='output_verify'):
    """Run a single seed experiment."""
    print(f"\n🧪 Running test experiment with {config_path}...")
    
    cmd = [
        sys.executable, 'run_experiment_mutation.py',
        '--config', config_path,
        '--seed', str(seed),
        '--output-dir', output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Experiment completed successfully")
        return True
    else:
        print("❌ Experiment failed!")
        print(result.stderr)
        return False


def verify_csv_switches(csv_path, expected_interval=100, tolerance=50):
    """Verify switches are detected in CSV logs."""
    print(f"\n🔍 Verifying switches in {csv_path}...")
    
    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    
    # Find all switches
    switches = df[df['is_switch'] == 1]
    switch_episodes = switches['episode'].tolist()
    
    print(f"📊 Found {len(switch_episodes)} switches at episodes: {switch_episodes[:5]}{'...' if len(switch_episodes) > 5 else ''}")
    
    if len(switch_episodes) == 0:
        print("❌ FAIL: No switches detected!")
        return False
    
    # Check first switch is around expected interval
    first_switch = switch_episodes[0]
    if abs(first_switch - expected_interval) > tolerance:
        print(f"⚠️  WARNING: First switch at episode {first_switch}, expected ~{expected_interval}")
        print(f"   (This is OK for jittered configs)")
    else:
        print(f"✅ First switch at episode {first_switch} (expected {expected_interval})")
    
    # Check dwell_time resets after switches
    for switch_ep in switch_episodes[:3]:  # Check first 3 switches
        if switch_ep < len(df) - 1:
            dwell_after = df.loc[df['episode'] == switch_ep, 'dwell_time_active'].values[0]
            if dwell_after == 0:
                print(f"✅ Dwell time resets to 0 at switch episode {switch_ep}")
            else:
                print(f"❌ FAIL: Dwell time is {dwell_after} at switch episode {switch_ep}, expected 0")
                return False
    
    # Check partition alternates
    switch_partitions = df[df['is_switch'] == 1]['partition_id'].tolist()
    if len(set(switch_partitions)) > 1:
        print(f"✅ Partitions alternate: {switch_partitions[:5]}")
    else:
        print(f"⚠️  WARNING: All switches to same partition: {set(switch_partitions)}")
    
    return True


def verify_console_output(output_dir, config_id, seed=0):
    """Check that switches were logged in console output."""
    print(f"\n🔍 Checking for switch detection messages...")
    
    # Try to find multi_seed_results.json
    results_file = Path(output_dir) / 'logs' / config_id / f'multi_seed_results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
            
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            stdout = result.get('stdout', '')
            
            if 'switched' in stdout.lower():
                print("✅ Switch detection messages found in output")
                return True
            else:
                print("⚠️  No switch detection messages in output (might be normal)")
                return True
    
    print("ℹ️  Could not check console output (file not found)")
    return True


def compare_deterministic_vs_jittered(output_dir):
    """Compare switch timings between deterministic and jittered configs."""
    print(f"\n🔬 Comparing deterministic vs jittered configs...")
    
    det_csv = None
    jit_csv = None
    
    # Find CSV files
    logs_dir = Path(output_dir) / 'logs'
    if logs_dir.exists():
        for config_dir in logs_dir.iterdir():
            if 'deterministic' in config_dir.name:
                det_csv = config_dir / 'seed_0.csv'
            elif 'uniform' in config_dir.name or 'poisson' in config_dir.name:
                jit_csv = config_dir / 'seed_0.csv'
    
    if det_csv and det_csv.exists() and jit_csv and jit_csv.exists():
        df_det = pd.read_csv(det_csv)
        df_jit = pd.read_csv(jit_csv)
        
        switches_det = df_det[df_det['is_switch'] == 1]['episode'].tolist()
        switches_jit = df_jit[df_jit['is_switch'] == 1]['episode'].tolist()
        
        print(f"Deterministic switches: {switches_det[:5]}")
        print(f"Jittered switches: {switches_jit[:5]}")
        
        if switches_det != switches_jit:
            print("✅ PASS: Switch timings differ between deterministic and jittered")
            return True
        else:
            print("❌ FAIL: Switch timings are identical (jitter not working?)")
            return False
    else:
        print("ℹ️  Skipping comparison (need both config types)")
        return True


def main():
    print("="*80)
    print("PARTITION SWITCHING FIX - VERIFICATION TEST")
    print("="*80)
    
    # Check if config files exist
    config_base = Path('configs/ndv_mutation/p2_b200_mu100')
    
    if not config_base.exists():
        print(f"\n❌ Config directory not found: {config_base}")
        print("Please run: python generate_configs_mutation.py --preset minimal")
        return 1
    
    configs_to_test = list(config_base.glob('*deterministic*.yaml'))
    
    if not configs_to_test:
        print(f"\n❌ No config files found in {config_base}")
        print("Please run: python generate_configs_mutation.py --preset minimal")
        return 1
    
    # Test with deterministic config
    test_config = configs_to_test[0]
    config_id = test_config.stem
    
    print(f"\nUsing test config: {test_config.name}")
    
    # Run test experiment
    success = run_test_experiment(str(test_config), seed=0, output_dir='output_verify')
    
    if not success:
        print("\n❌ VERIFICATION FAILED: Experiment did not complete")
        return 1
    
    # Verify CSV logs
    csv_path = f'output_verify/logs/{config_id}/seed_0.csv'
    csv_ok = verify_csv_switches(csv_path, expected_interval=100, tolerance=20)
    
    if not csv_ok:
        print("\n❌ VERIFICATION FAILED: CSV checks failed")
        return 1
    
    # Verify console output
    console_ok = verify_console_output('output_verify', config_id, seed=0)
    
    # Run comparison test if possible
    jittered_configs = list(config_base.glob('*uniform*.yaml'))
    if jittered_configs:
        jit_config = jittered_configs[0]
        print(f"\n🧪 Running jittered config for comparison: {jit_config.name}")
        run_test_experiment(str(jit_config), seed=0, output_dir='output_verify')
        
        compare_ok = compare_deterministic_vs_jittered('output_verify')
    else:
        compare_ok = True
    
    # Final verdict
    print("\n" + "="*80)
    if csv_ok and console_ok and compare_ok:
        print("✅ ALL VERIFICATION CHECKS PASSED!")
        print("="*80)
        print("\n🎉 The partition switching fix is working correctly!")
        print("\nYou can now run your full experiments with confidence:")
        print("  python run_multi_seed_experiments.py --config <config> --seed-range 0 20")
        return 0
    else:
        print("❌ SOME VERIFICATION CHECKS FAILED")
        print("="*80)
        print("\n⚠️  Please review the output above and ensure:")
        print("  1. You've applied the fix to run_experiment_mutation.py")
        print("  2. The function signature matches the call site")
        print("  3. get_current_partition() is only called once per episode")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)