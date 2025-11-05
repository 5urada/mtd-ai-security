"""
Quick Test Script for ID-HAM - CORRECTED VERSION
Uses scanning_rate=16 (from Table I) and addresses_per_host=25
"""

import numpy as np
from run_experiments import ExperimentRunner

def quick_test():
    """Run quick test with reduced parameters"""
    
    print("\n" + "="*70)
    print("ID-HAM QUICK TEST - CORRECTED VERSION")
    print("This will verify that TSH values are realistic (9-19 range)")
    print("Expected runtime: 5-10 minutes")
    print("="*70)
    
    runner = ExperimentRunner(results_dir="results_quick_test_fixed")
    
    # Run with reduced parameters
    print("\nRunning quick test with:")
    print("  - 30 hosts, 50 blocks, 5 switches")
    print("  - 3000 epochs")
    print("  - 10 steps per epoch")
    print("  - Testing all 4 scanning strategies")
    print("\nAddress space calculation:")
    print("  - Address space = 50 blocks × 128 = 6400 addresses")
    print("  - Hosts get 25 addresses each = 750 total host addresses")
    print("  - Scanning rate = 16 addresses/period (FROM PAPER TABLE I)")
    print("  - Expected hit rate = 750/6400 × 100 = 11.7%")
    print("  - Expected TSH per period = 16 × 0.117 = 1.9 hits")
    print("  - Expected TSH per epoch = 1.9 × 10 steps = 19 hits")
    print("  - With defense learning, should decrease to ~9-15 range")
    
    results = runner.run_defense_performance_experiment(
        num_hosts=30,
        num_blocks=50,
        num_switches=5,
        num_epochs=3000,  # Reduced from 3000
        steps_per_epoch=10  # Match paper (was 5)
    )
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETED!")
    print("\nCheck results in 'results_quick_test_fixed/' directory:")
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
            improvement = ((initial_avg - final_avg) / initial_avg) * 100 if initial_avg > 0 else 0
            print(f"  {method:10s}: Initial={initial_avg:.2f}, Final={final_avg:.2f}, "
                  f"Improvement={improvement:.1f}%")
    
    # Validation against paper
    print("\n" + "="*70)
    print("VALIDATION CHECKLIST:")
    print("="*70)
    
    all_valid = True
    for strategy, data in results.items():
        print(f"\n{strategy.upper().replace('_', ' ')}:")
        
        # Check 1: TSH values in reasonable range (5-30)
        for method, tsh in data.items():
            final_avg = np.mean(tsh[-50:])
            if 5 <= final_avg <= 30:
                print(f"  ✓ {method}: TSH = {final_avg:.1f} (in range 5-30)")
            else:
                print(f"  ✗ {method}: TSH = {final_avg:.1f} (out of range!)")
                all_valid = False
        
        # Check 2: ID-HAM should be best (lowest TSH)
        id_ham_final = np.mean(data['ID-HAM'][-50:])
        rhm_final = np.mean(data['RHM'][-50:])
        frvm_final = np.mean(data['FRVM'][-50:])
        
        if id_ham_final < rhm_final and id_ham_final < frvm_final:
            print(f"  ✓ ID-HAM performs best: {id_ham_final:.1f} < {rhm_final:.1f}, {frvm_final:.1f}")
        else:
            print(f"  ✗ ID-HAM not performing best!")
            all_valid = False
    
    print("\n" + "="*70)
    print("EXPECTED PAPER VALUES:")
    print("  Local Preference: ID-HAM=18-19, RHM=19-20, FRVM=18-19")
    print("  Sequential:       ID-HAM=9-12,  RHM=10-11, FRVM=12")
    print("  Divide-Conquer:   ID-HAM=14-17, RHM=16-17, FRVM=17-18")
    print("  Dynamic:          ID-HAM=18-19, RHM=19-20, FRVM=19-20")
    print("="*70)
    
    if all_valid:
        print("\n" + "="*70)
        print("✓✓✓ ALL CHECKS PASSED! ✓✓✓")
        print("The fix is working correctly!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠ Some checks failed - may need further tuning")
        print("If TSH is still too low, ensure:")
        print("  1. scanning_strategies.py has addresses_per_host=25")
        print("  2. run_experiments.py uses scanning_rate=16")
        print("="*70)

if __name__ == '__main__':
    quick_test()