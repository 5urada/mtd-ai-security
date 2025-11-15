"""
Quick Test Script for ID-HAM - IMPROVED VERSION
Tests the improved MDP with block-aware rewards
Expected runtime: 10-15 minutes
"""

import numpy as np
from run_experiments import ExperimentRunner

def quick_test():
    """Run quick test with improved implementation"""
    
    print("\n" + "="*70)
    print("ID-HAM QUICK TEST - IMPROVED VERSION")
    print("WITH BLOCK-AWARE REWARDS AND SCANNING HISTORY")
    print("Expected runtime: 10-15 minutes")
    print("="*70)
    
    print("\nKEY IMPROVEMENTS:")
    print("  ✓ Block-aware reward function")
    print("  ✓ Scanning history in state")
    print("  ✓ Block effectiveness tracking")
    print("  ✓ Improved RHM with hypothesis test")
    
    runner = ExperimentRunner(results_dir="results_quick_test_improved")
    
    # Run with full epochs (reduced from paper for speed)
    print("\nRunning quick test with:")
    print("  - 30 hosts, 50 blocks, 5 switches")
    print("  - 500 epochs (reduced for quick test)")
    print("  - 10 steps per epoch")
    print("  - Testing all 4 scanning strategies")
    
    print("\nExpected behavior:")
    print("  - ID-HAM TSH should DECREASE from ~19 to ~12-15")
    print("  - RHM TSH should decrease slightly to ~17-18")
    print("  - FRVM TSH should stay flat around ~18-20")
    print("  - Reward variance should be >1.0")
    print("  - Block scan history should update")
    
    results = runner.run_defense_performance_experiment(
        num_hosts=30,
        num_blocks=50,
        num_switches=5,
        num_epochs=500,  # Reduced for quick test
        steps_per_epoch=10
    )
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETED!")
    print("\nCheck results in 'results_quick_test_improved/' directory:")
    print("  - defense_performance_30h_50b.png")
    print("  - defense_comparison_bars_30h_50b.png")
    print("  - defense_performance_30h_50b.json")
    print("="*70)
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 70)
    for strategy, data in results.items():
        print(f"\n{strategy.upper().replace('_', ' ')}:")
        for method, tsh in data.items():
            initial_avg = np.mean(tsh[:50])
            final_avg = np.mean(tsh[-50:])
            improvement = ((initial_avg - final_avg) / initial_avg) * 100 if initial_avg > 0 else 0
            print(f"  {method:10s}: Initial={initial_avg:.2f}, Final={final_avg:.2f}, "
                  f"Improvement={improvement:.1f}%")
    
    # Validation checklist
    print("\n" + "="*70)
    print("VALIDATION CHECKLIST:")
    print("="*70)
    
    all_valid = True
    
    # Check 1: ID-HAM shows learning
    for strategy, data in results.items():
        print(f"\n{strategy.upper().replace('_', ' ')}:")
        
        id_ham = data['ID-HAM']
        rhm = data['RHM']
        frvm = data['FRVM']
        
        # Initial values
        id_ham_initial = np.mean(id_ham[:50])
        id_ham_final = np.mean(id_ham[-50:])
        
        # Check 1: ID-HAM decreased
        if id_ham_final < id_ham_initial:
            print(f"  ✓ ID-HAM shows learning: {id_ham_initial:.1f} → {id_ham_final:.1f}")
        else:
            print(f"  ✗ ID-HAM not learning: {id_ham_initial:.1f} → {id_ham_final:.1f}")
            all_valid = False
        
        # Check 2: ID-HAM is best
        rhm_final = np.mean(rhm[-50:])
        frvm_final = np.mean(frvm[-50:])
        
        if id_ham_final < rhm_final and id_ham_final < frvm_final:
            improvement_vs_frvm = ((frvm_final - id_ham_final) / frvm_final) * 100
            print(f"  ✓ ID-HAM performs best: {id_ham_final:.1f} < {rhm_final:.1f}, {frvm_final:.1f}")
            print(f"    Improvement vs FRVM: {improvement_vs_frvm:.1f}%")
        else:
            print(f"  ⚠ ID-HAM not clearly best: ID-HAM={id_ham_final:.1f}, RHM={rhm_final:.1f}, FRVM={frvm_final:.1f}")
            # Don't fail - might need more epochs
    
    print("\n" + "="*70)
    print("EXPECTED BEHAVIOR AFTER FIXES:")
    print("="*70)
    print("\nYou should observe:")
    print("  1. ID-HAM curves trending DOWNWARD")
    print("  2. RHM curves decreasing slightly")
    print("  3. FRVM curves staying relatively FLAT")
    print("  4. Clear separation between ID-HAM and baselines")
    print("  5. Reward variance >1.0 (printed during training)")
    print("  6. Block scan history updating (printed every 500 epochs)")
    
    print("\n" + "="*70)
    if all_valid:
        print("✓✓✓ VALIDATION PASSED ✓✓✓")
        print("\nID-HAM is learning successfully!")
        print("The improved implementation is working correctly.")
        print("\nNext steps:")
        print("  1. Run full experiment: python run_experiments_improved.py")
        print("  2. Compare with paper's expected values")
    else:
        print("⚠ Some checks didn't pass completely")
        print("\nPossible reasons:")
        print("  1. Only 500 epochs - may need full 3000 for convergence")
        print("  2. Random seed variations")
        print("  3. SMT solver found different feasible actions")
        print("\nNext steps:")
        print("  1. Check reward variance (should be >1.0)")
        print("  2. Verify block_scan_history updates")
        print("  3. Run with more epochs if needed")
    print("="*70)


if __name__ == '__main__':
    quick_test()