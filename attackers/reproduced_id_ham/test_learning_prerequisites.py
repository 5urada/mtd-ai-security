"""
Test Script: Verify that Reward Signal Differentiates Actions
This MUST pass before training can work!
"""

import numpy as np
import sys

# Add path if needed
sys.path.append('/mnt/user-data/uploads/attackers/reproduced_id_ham')

def test_reward_differentiation():
    """
    CRITICAL TEST: Do different actions get different rewards?
    
    Current Bug: All actions with same # hits get same reward
    Expected: Actions avoiding hot blocks should get better rewards
    """
    
    print("=" * 70)
    print("REWARD SIGNAL TEST")
    print("=" * 70)
    
    try:
        from mdp_model_improved import ImprovedHAM_MDP
        print("✓ Using ImprovedHAM_MDP (with block-aware rewards)")
        mdp_class = ImprovedHAM_MDP
    except ImportError:
        print("⚠ ImprovedHAM_MDP not found, using original HAM_MDP")
        from mdp_model import HAM_MDP as mdp_class
    
    # Create small MDP for testing
    mdp = mdp_class(num_hosts=10, num_blocks=20)
    mdp.reset()
    
    # Simulate adversary scanning blocks [0, 1, 2] heavily
    if hasattr(mdp, 'block_scan_history'):
        mdp.block_scan_history[0:3] = 10.0  # Hot blocks!
        mdp.block_scan_history[10:13] = 0.1  # Cool blocks
        print("\n✓ Adversary scan pattern set:")
        print(f"  Hot blocks [0,1,2]: {mdp.block_scan_history[0:3]}")
        print(f"  Cool blocks [10,11,12]: {mdp.block_scan_history[10:13]}")
    else:
        print("\n✗ MDP doesn't track block scan history - using original model")
        print("  This is the ROOT CAUSE of learning failure!")
    
    # Create two actions with SAME number of hits
    action_hot = np.zeros((10, 20))
    action_cool = np.zeros((10, 20))
    
    # Action 1: Assigns HOT blocks [0,1,2] to hosts
    for i in range(10):
        action_hot[i, 0:3] = 1
    
    # Action 2: Assigns COOL blocks [10,11,12] to hosts
    for i in range(10):
        action_cool[i, 10:13] = 1
    
    print("\nAction 1: Assigns hot blocks [0,1,2]")
    print("Action 2: Assigns cool blocks [10,11,12]")
    
    # Simulate SAME scan results (5 hits) for both actions
    scan_results = {0: 3, 1: 2}  # 5 total hits
    scanned_addresses = {5: 1, 10: 1, 15: 1}  # Addresses that were scanned
    
    print(f"\nScan results: {sum(scan_results.values())} total hits (same for both actions)")
    
    # Calculate rewards
    if hasattr(mdp, 'calculate_block_aware_reward'):
        reward_hot = mdp.calculate_block_aware_reward(action_hot, scan_results)
        reward_cool = mdp.calculate_block_aware_reward(action_cool, scan_results)
    else:
        # Original reward (broken)
        mdp.current_allocation = action_hot
        reward_hot = mdp.calculate_reward(scan_results)
        
        mdp.current_allocation = action_cool
        reward_cool = mdp.calculate_reward(scan_results)
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Reward for HOT blocks action:  {reward_hot:8.3f}")
    print(f"Reward for COOL blocks action: {reward_cool:8.3f}")
    print(f"Difference:                    {abs(reward_cool - reward_hot):8.3f}")
    print("=" * 70)
    
    # Verify reward signal
    print("\nANALYSIS:")
    
    if abs(reward_hot - reward_cool) < 0.1:
        print("✗ REWARD SIGNAL IS BROKEN!")
        print("  Both actions get nearly identical rewards despite different block choices.")
        print("  Agent CANNOT learn which blocks to avoid.")
        print("\n  ROOT CAUSE: Reward only considers total hits, not which blocks were assigned.")
        print("\n  FIX REQUIRED:")
        print("    1. Track which blocks are being scanned (block_scan_history)")
        print("    2. Penalize actions that use frequently-scanned blocks")
        print("    3. Reward actions that avoid hot blocks")
        return False
    else:
        print("✓ REWARD SIGNAL IS WORKING!")
        print(f"  Actions get different rewards (difference = {abs(reward_cool - reward_hot):.3f})")
        
        if reward_cool > reward_hot:
            print("  ✓ Cool blocks correctly get BETTER reward")
        else:
            print("  ⚠ Unexpected: Hot blocks got better reward")
        
        print("\n  Agent CAN learn to prefer actions that avoid scanned blocks.")
        return True


def test_state_representation():
    """
    Test if state contains scanning history
    """
    print("\n\n" + "=" * 70)
    print("STATE REPRESENTATION TEST")
    print("=" * 70)
    
    try:
        from mdp_model_improved import ImprovedHAM_MDP
        mdp = ImprovedHAM_MDP(num_hosts=10, num_blocks=20)
    except ImportError:
        from mdp_model import HAM_MDP
        mdp = HAM_MDP(num_hosts=10, num_blocks=20)
    
    state = mdp.reset()
    
    print(f"\nState dimension: {len(state)}")
    print(f"Expected minimum: {mdp.num_hosts} (host types only)")
    print(f"Expected improved: {mdp.num_hosts + mdp.num_blocks} (host types + block history)")
    
    if len(state) <= mdp.num_hosts:
        print("\n✗ STATE IS TOO SIMPLE!")
        print("  State only contains host types (moving vs static).")
        print("  Agent CANNOT see scanning patterns.")
        print("\n  FIX REQUIRED:")
        print("    Include block_scan_history in state")
        print("    State should be: [host_types, block_history]")
        return False
    else:
        print("\n✓ STATE CONTAINS ADDITIONAL INFORMATION")
        print(f"  State has {len(state) - mdp.num_hosts} extra features")
        print("  This allows agent to learn from scanning patterns")
        return True


def test_action_space():
    """
    Test if action space is reasonable
    """
    print("\n\n" + "=" * 70)
    print("ACTION SPACE TEST")
    print("=" * 70)
    
    from smt_constraints import generate_feasible_actions
    
    print("\nGenerating feasible actions (this may take ~30s)...")
    
    try:
        actions = generate_feasible_actions(
            num_hosts=10,
            num_blocks=20,
            num_switches=3,
            max_solutions=50
        )
        
        print(f"\n✓ Generated {len(actions)} feasible actions")
        print(f"  Action shape: {actions[0].shape}")
        print(f"  Avg blocks per host: {np.mean([a.sum() / 10 for a in actions]):.1f}")
        
        # Check diversity
        action_hashes = [hash(a.tobytes()) for a in actions]
        unique = len(set(action_hashes))
        print(f"  Unique actions: {unique}/{len(actions)}")
        
        if unique == len(actions):
            print("\n✓ ACTION SPACE IS DIVERSE")
            return True
        else:
            print("\n⚠ WARNING: Some duplicate actions")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR generating actions: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("ID-HAM DIAGNOSTIC TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Reward Signal
    results['reward'] = test_reward_differentiation()
    
    # Test 2: State Representation  
    results['state'] = test_state_representation()
    
    # Test 3: Action Space
    results['actions'] = test_action_space()
    
    # Summary
    print("\n\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.upper():15s}: {status}")
    
    all_pass = all(results.values())
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nYour implementation should be able to learn!")
        print("Run: python run_experiments_quick.py")
    else:
        print("✗✗✗ CRITICAL ISSUES FOUND ✗✗✗")
        print("\nYour implementation CANNOT learn until these are fixed.")
        print("\nNext steps:")
        print("1. Read DEBUGGING_GUIDE.md")
        print("2. Implement mdp_model_improved.py")
        print("3. Re-run this test script")
        print("4. Only proceed to training when all tests pass")
    print("=" * 70)


if __name__ == '__main__':
    main()
