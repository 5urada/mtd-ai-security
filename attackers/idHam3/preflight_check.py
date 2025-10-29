#!/usr/bin/env python3
"""
Pre-flight check before running experiments.

Verifies:
1. All required files exist
2. Configs are valid
3. Python dependencies are installed
4. Neural networks can be initialized
"""

import sys
import os
from pathlib import Path
import importlib.util


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(path).exists():
        print(f"  ✅ {description}")
        return True
    else:
        print(f"  ❌ {description} NOT FOUND: {path}")
        return False


def check_directory_exists(path: str, description: str) -> bool:
    """Check if a directory exists."""
    if Path(path).exists() and Path(path).is_dir():
        print(f"  ✅ {description}")
        return True
    else:
        print(f"  ❌ {description} NOT FOUND: {path}")
        return False


def check_python_module(module_name: str) -> bool:
    """Check if a Python module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"  ✅ {module_name}")
            return True
        else:
            print(f"  ❌ {module_name} NOT INSTALLED")
            return False
    except Exception as e:
        print(f"  ❌ {module_name} ERROR: {e}")
        return False


def check_config_files(config_dir: str) -> tuple:
    """Check for config files."""
    if not Path(config_dir).exists():
        return [], False
    
    yaml_files = list(Path(config_dir).glob("*.yaml"))
    yaml_files.extend(Path(config_dir).glob("**/*.yaml"))
    
    return yaml_files, len(yaml_files) > 0


def main():
    print("\n" + "="*80)
    print("PRE-FLIGHT CHECK FOR ID-HAM EXPERIMENTS")
    print("="*80 + "\n")
    
    all_checks_passed = True
    
    # Check 1: Required Python files
    print("1️⃣  Checking required Python files...")
    
    required_files = [
        ('run_experiment_mutation.py', 'Main experiment runner'),
        ('run_multi_seed_experiments.py', 'Multi-seed runner'),
        ('run_batch_experiments.py', 'Batch runner'),
        ('check_results.py', 'Results checker'),
        ('src/defender_mutation.py', 'Defender implementation'),
        ('src/attacker.py', 'Attacker implementation'),
        ('src/environment_mutation.py', 'Environment simulation'),
        ('src/metrics.py', 'Metrics logger'),
        ('src/utils.py', 'Utilities'),
    ]
    
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    print()
    
    # Check 2: Python dependencies
    print("2️⃣  Checking Python dependencies...")
    
    required_modules = [
        'numpy',
        'torch',
        'yaml',
        'csv',
        'json',
        'pathlib',
    ]
    
    for module in required_modules:
        if not check_python_module(module):
            all_checks_passed = False
    
    # Special check for PyTorch with version
    try:
        import torch
        print(f"     PyTorch version: {torch.__version__}")
        print(f"     CUDA available: {torch.cuda.is_available()}")
    except:
        pass
    
    print()
    
    # Check 3: Config files
    print("3️⃣  Checking configuration files...")
    
    config_dirs = [
        'configs/ndv_mutation/p2_b200_mu100',
        'configs/ndv_mutation',
        'configs'
    ]
    
    configs_found = False
    for config_dir in config_dirs:
        yaml_files, found = check_config_files(config_dir)
        if found:
            print(f"  ✅ Found {len(yaml_files)} config files in {config_dir}")
            configs_found = True
            
            # Show first few
            for i, f in enumerate(yaml_files[:3]):
                print(f"     - {f.name}")
            if len(yaml_files) > 3:
                print(f"     ... and {len(yaml_files) - 3} more")
            break
    
    if not configs_found:
        print(f"  ⚠️  No config files found. You may need to generate them:")
        print(f"     python generate_configs_mutation.py --preset minimal")
        all_checks_passed = False
    
    print()
    
    # Check 4: Output directory
    print("4️⃣  Checking output directory...")
    
    output_dir = Path('output')
    if not output_dir.exists():
        print(f"  ℹ️  Output directory will be created automatically")
    else:
        print(f"  ✅ Output directory exists")
        
        # Check if there are existing results
        logs = list((output_dir / 'logs').glob('*')) if (output_dir / 'logs').exists() else []
        if logs:
            print(f"     Found {len(logs)} existing experiment logs")
    
    print()
    
    # Check 5: Test neural network initialization
    print("5️⃣  Testing neural network initialization...")
    
    try:
        sys.path.insert(0, 'src')
        from defender_mutation import IDHAMNeuralDefender, PolicyNetwork, ValueNetwork
        
        # Try to create networks
        test_state_dim = 301
        test_n_actions = 3
        
        actor = PolicyNetwork(test_state_dim, test_n_actions)
        critic = ValueNetwork(test_state_dim)
        
        print(f"  ✅ Actor network initialized ({sum(p.numel() for p in actor.parameters())} params)")
        print(f"  ✅ Critic network initialized ({sum(p.numel() for p in critic.parameters())} params)")
        
        # Try creating a defender
        test_config = {
            'defender': {
                'type': 'idham_mutation',
                'mutation_capacity': 30,
                'mutation_interval': 50
            }
        }
        
        defender = IDHAMNeuralDefender(
            n_hosts=100,
            config=test_config,
            feasible_actions=[{i: [i] for i in range(100)}],
            seed=42
        )
        
        print(f"  ✅ Neural defender initialized successfully")
        print(f"     - State dimension: {defender.state_dim}")
        print(f"     - Feasible actions: {defender.n_feasible_actions}")
        
    except Exception as e:
        print(f"  ❌ Neural network initialization failed: {e}")
        all_checks_passed = False
    
    print()
    
    # Check 6: Test imports
    print("6️⃣  Testing module imports...")
    
    try:
        from src.attacker import DivideConquerAttacker
        from src.environment_mutation import NetworkEnvironment
        from src.metrics import MetricsLogger
        from src.utils import set_seed
        
        print(f"  ✅ All modules import successfully")
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        all_checks_passed = False
    
    print()
    
    # Final summary
    print("="*80)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - READY TO RUN EXPERIMENTS!")
        print("="*80)
        print("\n🚀 Next steps:")
        print("\n1. Start with a test run (5 seeds):")
        print("   python run_multi_seed_experiments.py \\")
        print("     --config configs/ndv_mutation/p2_b200_mu100/idham_mutation-divconq_2p_deterministic-100mu_j0_200b.yaml \\")
        print("     --seed-range 0 5\n")
        print("2. Once verified, run full experiments (20 seeds):")
        print("   python run_batch_experiments.py \\")
        print("     --config-dir configs/ndv_mutation/p2_b200_mu100 \\")
        print("     --pattern 'idham_mutation*.yaml' \\")
        print("     --seed-range 0 20\n")
        print("3. Monitor progress:")
        print("   python check_results.py\n")
        
        return 0
    else:
        print("❌ SOME CHECKS FAILED - PLEASE FIX ISSUES BEFORE RUNNING")
        print("="*80)
        print("\n🔧 Common fixes:")
        print("\n1. Install missing Python packages:")
        print("   pip install numpy torch pyyaml --break-system-packages\n")
        print("2. Make sure you're in the correct directory:")
        print("   cd ~/Documents/CSCE491/mtd-ai-security/attackers/idHam3\n")
        print("3. Generate config files if missing:")
        print("   python generate_configs_mutation.py --preset minimal\n")
        
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)