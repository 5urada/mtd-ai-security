#!/usr/bin/env python3
"""
Quick test script to validate experiment framework.
Runs a minimal experiment to check all components work.
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.attacker import DivideConquerAttacker
from src.defender import IDHAMDefender
from src.environment import NetworkEnvironment
from src.utils import set_seed


def test_basic_components():
    """Test that all basic components can be instantiated."""
    print("Testing basic component initialization...")
    
    config = {
        'n_hosts': 100,
        'partitioning': {'mode': 'uniform', 'partitions': 4},
        'attacker': {
            'strategy': 'divide_and_conquer',
            'switch': {
                'mode': 'periodic',
                'mean_interval': 500,
                'jitter_pct': 0.10
            }
        }
    }
    
    set_seed(0)
    
    # Test environment
    env = NetworkEnvironment(n_hosts=100, config=config, seed=0)
    print(f"✓ Environment created: {env.n_real_hosts} real hosts")
    
    # Test attacker
    attacker = DivideConquerAttacker(
        n_hosts=100,
        partitioning_config=config['partitioning'],
        switch_config=config['attacker']['switch'],
        seed=0
    )
    print(f"✓ Attacker created: {attacker.n_partitions} partitions")
    print(f"  Partition sizes: {attacker.get_partition_sizes()}")
    
    # Test defender
    defender = IDHAMDefender(n_hosts=100, config=config, seed=0)
    print(f"✓ Defender created: {len(defender.current_masked)} initially masked")
    
    # Test episode simulation
    print("\nTesting episode simulation...")
    for episode in range(5):
        env.reset()
        defender.reset_episode()
        
        partition_id = attacker.get_current_partition(episode)
        targets = attacker.get_partition_targets(partition_id)
        
        hits = 0
        for probe_idx in range(10):
            target = attacker.select_target(targets, probe_idx)
            is_masked, qos_cost = defender.apply_defense(target, probe_idx)
            is_hit = env.probe(target, is_masked)
            if is_hit:
                hits += 1
        
        defender.update(hits, set(), episode)
        print(f"  Episode {episode}: partition={partition_id}, hits={hits}")
    
    print("\n✓ All basic components working!")
    return True


def test_config_generation():
    """Test config file generation."""
    print("\nTesting config generation...")
    
    from generate_configs import generate_baseline_config, generate_experiment_configs
    
    baseline = generate_baseline_config()
    print(f"✓ Baseline config: {baseline['config_id']}")
    
    exp_configs = generate_experiment_configs()
    print(f"✓ Generated {len(exp_configs)} experimental configs")
    
    # Check first config is valid
    first_config = exp_configs[0]
    print(f"  Sample config: {first_config['config_id']}")
    
    return True


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics computation...")
    
    from src.metrics import compute_windowed_metrics, compute_adaptation_lag
    
    # Create dummy metrics
    dummy_metrics = []
    for ep in range(100):
        dummy_metrics.append({
            'episode': ep,
            'tsh': 10 + np.random.randint(-3, 3),
            'coverage': 0.5 + ep * 0.001,
            'policy_entropy': 2.5,
            'flow_mods_count': 5,
            'qos_penalty_proxy_ms': 10.0
        })
    
    # Test windowed metrics
    windowed = compute_windowed_metrics(dummy_metrics, window_size=20)
    print(f"✓ Windowed metrics computed: {len(windowed)} windows")
    
    # Test adaptation lag
    switch_points = [30, 60]
    lags = compute_adaptation_lag(dummy_metrics, switch_points, window_size=10, epsilon=0.1)
    print(f"✓ Adaptation lags computed: {lags}")
    
    return True


def run_mini_experiment():
    """Run a very small experiment end-to-end."""
    print("\n" + "="*60)
    print("Running mini experiment (100 episodes, 1 seed)...")
    print("="*60)
    
    # Create mini config
    mini_config = {
        'config_id': 'test_mini',
        'n_hosts': 50,
        'partitioning': {'mode': 'uniform', 'partitions': 2},
        'attacker': {
            'strategy': 'divide_and_conquer',
            'switch': {
                'mode': 'periodic',
                'mean_interval': 25,
                'jitter_pct': 0.0
            }
        },
        'probe_budget': 50,
        'episodes': {'train': 100, 'eval_window': 20},
        'seeds': 1,
        'metrics': {'window': 20, 'adaptation_epsilon': 0.10},
        'logging': {'per_episode_csv': True, 'aggregate_json': True}
    }
    
    # Save config
    config_path = Path('test_mini_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(mini_config, f)
    
    print(f"Config saved to: {config_path}")
    
    # Run experiment
    import subprocess
    result = subprocess.run([
        sys.executable, 'run_experiment.py',
        '--config', str(config_path),
        '--output-dir', 'test_output',
        '--seed', '0'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Mini experiment completed successfully!")
        print("\nOutput files created:")
        test_output = Path('test_output')
        for path in sorted(test_output.rglob('*')):
            if path.is_file():
                print(f"  {path}")
        return True
    else:
        print("✗ Mini experiment failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False


if __name__ == '__main__':
    import numpy as np
    
    print("="*60)
    print("ID-HAM EXPERIMENT FRAMEWORK TEST SUITE")
    print("="*60)
    
    tests = [
        ("Component Initialization", test_basic_components),
        ("Config Generation", test_config_generation),
        ("Metrics Computation", test_metrics),
        ("End-to-End Mini Experiment", run_mini_experiment)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        try:
            if test_func():
                passed += 1
                print(f"✓ PASSED: {test_name}")
            else:
                failed += 1
                print(f"✗ FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"✗ FAILED: {test_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Framework is ready to use.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        sys.exit(1)