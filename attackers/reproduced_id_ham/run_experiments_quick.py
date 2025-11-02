"""
Quick Test Script for ID-HAM
Runs reduced experiments to verify setup (5-10 minutes)
"""

import numpy as np
from run_experiments import ExperimentRunner

def quick_test():
    """Run quick test with reduced parameters"""
    
    print("\n" + "="*70)
    print("ID-HAM QUICK TEST")
    print("This will take approximately 5-10 minutes")
    print("="*70)
    
    runner = ExperimentRunner(results_dir="results_quick_test")
    
    # Run with reduced parameters
    print("\nRunning quick test with:")
    print("  - 30 hosts, 50 blocks, 5 switches")
    print("  - 500 epochs (instead of 3000)")
    print("  - 5 steps per epoch (instead of 10)")
    print("  - Testing only 2 scanning strategies")
    
    results = runner.run_defense_performance_experiment(
        num_hosts=30,
        num_blocks=50,
        num_switches=5,
        num_epochs=500,  # Reduced from 3000
        steps_per_epoch=10  # Match paper (was 5)
    )
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETED!")
    print("\nCheck results in 'results_quick_test/' directory:")
    print("  - defense_performance_30h_50b.png")
    print("  - defense_comparison_bars_30h_50b.png")
    print("  - defense_performance_30h_50b.json")
    print("\nIf plots show ID-HAM learning (TSH decreasing), the setup works!")
    print("="*70)
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 70)
    for strategy, data in results.items():
        print(f"\n{strategy.upper()} SCANNING:")
        for method, tsh in data.items():
            initial_avg = np.mean(tsh[:50])
            final_avg = np.mean(tsh[-50:])
            improvement = ((initial_avg - final_avg) / initial_avg) * 100
            print(f"  {method:10s}: Initial={initial_avg:.2f}, Final={final_avg:.2f}, "
                  f"Improvement={improvement:.1f}%")

if __name__ == '__main__':
    quick_test()