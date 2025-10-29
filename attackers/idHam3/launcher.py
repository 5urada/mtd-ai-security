#!/usr/bin/env python3
"""
Simple launcher for ID-HAM experiments.

Provides an interactive menu for running experiments.
"""

import sys
import subprocess
from pathlib import Path


def print_banner():
    print("\n" + "="*80)
    print(" ID-HAM NEURAL NETWORK EXPERIMENT LAUNCHER")
    print("="*80 + "\n")


def run_command(cmd: list, description: str):
    """Run a command with nice output."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}\n")
    print("-"*80)
    
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    print_banner()
    
    print("What would you like to do?\n")
    print("1. 🔍 Pre-flight check (verify setup)")
    print("2. 🧪 Quick test (5 seeds, ~30 min)")
    print("3. 📊 Full ID-HAM experiments (20 seeds, ~4-6 hours)")
    print("4. 🏁 Complete comparison (all defenders, ~16-24 hours)")
    print("5. 📈 Check results")
    print("6. 🔬 Test neural networks")
    print("7. ⚙️  Generate configs")
    print("8. 🆘 Help / Documentation")
    print("9. ❌ Exit")
    
    choice = input("\nEnter choice (1-9): ").strip()
    
    print("\n" + "="*80 + "\n")
    
    if choice == '1':
        # Pre-flight check
        cmd = [sys.executable, 'preflight_check.py']
        run_command(cmd, "Running pre-flight check")
        
    elif choice == '2':
        # Quick test
        config_dir = 'configs/ndv_mutation/p2_b200_mu100'
        
        # Find first config
        configs = list(Path(config_dir).glob('idham_mutation*.yaml'))
        
        if not configs:
            print(f"❌ No configs found in {config_dir}")
            print(f"\nRun option 7 to generate configs first")
            return
        
        config = configs[0]
        
        print(f"Using config: {config.name}\n")
        cmd = [
            sys.executable, 'run_multi_seed_experiments.py',
            '--config', str(config),
            '--seed-range', '0', '5',
            '--output-dir', 'output'
        ]
        
        run_command(cmd, "Running quick test (5 seeds)")
        
        # Offer to check results
        print("\n" + "="*80)
        if input("\nView results? [y/N]: ").lower() == 'y':
            subprocess.run([sys.executable, 'check_results.py'])
        
    elif choice == '3':
        # Full ID-HAM experiments
        config_dir = 'configs/ndv_mutation/p2_b200_mu100'
        
        if not Path(config_dir).exists():
            print(f"❌ Config directory not found: {config_dir}")
            print(f"\nRun option 7 to generate configs first")
            return
        
        # Confirm
        print("This will run 3 configs × 20 seeds = 60 experiments")
        print("Estimated time: 4-6 hours\n")
        
        if input("Continue? [y/N]: ").lower() != 'y':
            print("Cancelled")
            return
        
        cmd = [
            sys.executable, 'run_batch_experiments.py',
            '--config-dir', config_dir,
            '--pattern', 'idham_mutation*.yaml',
            '--seed-range', '0', '20',
            '--output-dir', 'output'
        ]
        
        success = run_command(cmd, "Running full ID-HAM experiments")
        
        if success:
            print("\n" + "="*80)
            if input("\nView results? [y/N]: ").lower() == 'y':
                subprocess.run([sys.executable, 'check_results.py', '--compare'])
        
    elif choice == '4':
        # Complete comparison
        config_dir = 'configs/ndv_mutation/p2_b200_mu100'
        
        if not Path(config_dir).exists():
            print(f"❌ Config directory not found: {config_dir}")
            return
        
        # Count configs
        n_configs = len(list(Path(config_dir).glob('*.yaml')))
        
        print(f"This will run {n_configs} configs × 20 seeds = {n_configs * 20} experiments")
        print("Estimated time: 16-24 hours\n")
        
        if input("Continue? [y/N]: ").lower() != 'y':
            print("Cancelled")
            return
        
        cmd = [
            sys.executable, 'run_batch_experiments.py',
            '--config-dir', config_dir,
            '--seed-range', '0', '20',
            '--output-dir', 'output',
            '--timeout', '7200'
        ]
        
        success = run_command(cmd, "Running complete comparison")
        
        if success:
            print("\n" + "="*80)
            if input("\nView results? [y/N]: ").lower() == 'y':
                subprocess.run([sys.executable, 'check_results.py', '--compare'])
        
    elif choice == '5':
        # Check results
        print("Results options:\n")
        print("1. View all results")
        print("2. View comparison table")
        print("3. View specific config")
        
        sub_choice = input("\nEnter choice (1-3): ").strip()
        
        if sub_choice == '1':
            subprocess.run([sys.executable, 'check_results.py'])
        elif sub_choice == '2':
            subprocess.run([sys.executable, 'check_results.py', '--compare'])
        elif sub_choice == '3':
            config_id = input("Enter config ID: ").strip()
            subprocess.run([sys.executable, 'check_results.py', '--config', config_id])
        
    elif choice == '6':
        # Test neural networks
        cmd = [sys.executable, 'test_neural_defender.py']
        run_command(cmd, "Testing neural networks")
        
    elif choice == '7':
        # Generate configs
        print("Config generation options:\n")
        print("1. Minimal grid (p=2, B=200, μ=100)")
        print("2. Full grid (multiple strata)")
        
        sub_choice = input("\nEnter choice (1-2): ").strip()
        
        if sub_choice == '1':
            cmd = [
                sys.executable, 'generate_configs_mutation.py',
                '--preset', 'minimal',
                '--output-dir', 'configs/ndv_mutation/p2_b200_mu100'
            ]
            run_command(cmd, "Generating minimal config grid")
        elif sub_choice == '2':
            cmd = [
                sys.executable, 'generate_configs_mutation.py',
                '--preset', 'full',
                '--output-dir', 'configs/ndv_mutation/full_grid'
            ]
            run_command(cmd, "Generating full config grid")
        
    elif choice == '8':
        # Help
        print("📚 Documentation Files:\n")
        print("1. README_MULTI_SEED.md - Main README")
        print("2. EXPERIMENT_GUIDE.md - Detailed experiment guide")
        print("3. CONTINUE_CHAT_PROMPT.md - Project context")
        print("4. NEURAL_IMPLEMENTATION_GUIDE.md - Technical details")
        
        print("\n💡 Quick Commands:\n")
        print("Pre-flight check:")
        print("  python preflight_check.py\n")
        print("Quick test (5 seeds):")
        print("  python run_multi_seed_experiments.py --config <config.yaml> --seed-range 0 5\n")
        print("Full experiments (20 seeds):")
        print("  python run_batch_experiments.py --config-dir <dir> --seed-range 0 20\n")
        print("Check results:")
        print("  python check_results.py\n")
        
        input("\nPress Enter to continue...")
        
    elif choice == '9':
        print("Goodbye! 👋")
        return
    
    else:
        print("Invalid choice")
        return
    
    print("\n" + "="*80)
    
    # Ask if user wants to continue
    if input("\nReturn to menu? [y/N]: ").lower() == 'y':
        main()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)