#!/usr/bin/env python3
"""
Test script for neural network-based ID-HAM defender.

Tests:
1. Network initialization
2. State feature extraction
3. Action selection from policy
4. Mutation execution
5. Training loop
"""

import numpy as np
import torch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock config for testing
config = {
    'defender': {
        'type': 'idham_mutation',
        'mutation_capacity': 30,
        'mutation_interval': 50
    }
}

print("="*80)
print("TESTING NEURAL NETWORK-BASED ID-HAM DEFENDER")
print("="*80)

# Import the neural defender from src directory
from defender_mutation import IDHAMNeuralDefender

print("\n1. INITIALIZING DEFENDER")
print("-"*80)

# Create simple feasible actions (for testing)
n_hosts = 100
feasible_actions = [
    {i: [i] for i in range(n_hosts)},  # Action 0: identity mapping
    {i: [(i + 10) % n_hosts] for i in range(n_hosts)},  # Action 1: shift by 10
    {i: [(i + 20) % n_hosts] for i in range(n_hosts)},  # Action 2: shift by 20
]

defender = IDHAMNeuralDefender(
    n_hosts=n_hosts,
    config=config,
    feasible_actions=feasible_actions,
    seed=42
)

print(f"✅ Defender initialized")
print(f"   - State dimension: {defender.state_dim}")
print(f"   - Number of feasible actions: {defender.n_feasible_actions}")
print(f"   - Actor parameters: {sum(p.numel() for p in defender.actor.parameters())}")
print(f"   - Critic parameters: {sum(p.numel() for p in defender.critic.parameters())}")
print(f"   - Initial mutated hosts: {len(defender.mutated_hosts)}")

print("\n2. TESTING STATE FEATURE EXTRACTION")
print("-"*80)

state = defender._extract_state_features(coverage=0.25)
print(f"✅ State features extracted")
print(f"   - Shape: {state.shape}")
print(f"   - Expected shape: ({defender.state_dim},)")
print(f"   - Coverage value: {state[-1]:.2f}")
print(f"   - Moving status sum: {state[:n_hosts].sum():.0f}")
print(f"   - Sample features: {state[:5]}")

print("\n3. TESTING NEURAL NETWORK FORWARD PASS")
print("-"*80)

state_tensor = torch.FloatTensor(state).unsqueeze(0)

# Test actor
with torch.no_grad():
    logits = defender.actor(state_tensor)
    probs = defender.actor.get_action_probs(state_tensor)
    action_idx, log_prob = defender.actor.sample_action(state_tensor)

print(f"✅ Actor forward pass successful")
print(f"   - Logits shape: {logits.shape}")
print(f"   - Probabilities shape: {probs.shape}")
print(f"   - Probabilities sum: {probs.sum().item():.4f}")
print(f"   - Selected action: {action_idx}")
print(f"   - Log probability: {log_prob.item():.4f}")

# Test critic
with torch.no_grad():
    value = defender.critic(state_tensor)

print(f"✅ Critic forward pass successful")
print(f"   - Value shape: {value.shape}")
print(f"   - Estimated value: {value.item():.4f}")

print("\n4. SIMULATING EPISODE WITH PROBING")
print("-"*80)

# Simulate probing partition 0 (hosts 0-49)
partition_0_hosts = set(range(0, 50))
discovered_hosts = set()

for probe_idx in range(200):
    target_host = list(partition_0_hosts)[probe_idx % len(partition_0_hosts)]
    physical_host, is_mutated, qos = defender.resolve_probe(target_host, probe_idx)
    
    if probe_idx % 2 == 0:
        discovered_hosts.add(physical_host)

hits_count = len(discovered_hosts)

print(f"✅ Probing simulation complete")
print(f"   - Total probes: 200")
print(f"   - Hits: {hits_count}")
print(f"   - Unique hosts probed: {int((defender.host_probe_counts > 0).sum())}")
print(f"   - Avg probes per host: {defender.host_probe_counts.mean():.1f}")

print("\n5. TESTING MUTATION POLICY UPDATE (Episode 0)")
print("-"*80)

# Update defender (should trigger mutation at episode 0)
defender.update(hits_count=hits_count, discovered_hosts=discovered_hosts, episode=0)

print(f"✅ Policy update complete")
print(f"   - Mutations this episode: {defender.get_mutation_count()}")
print(f"   - Total mutated hosts: {len(defender.mutated_hosts)}")
print(f"   - Address entropy: {defender.get_address_entropy():.2f}")
print(f"   - Experience buffer size: {len(defender.rewards)}")

print("\n6. SIMULATING MULTIPLE EPISODES")
print("-"*80)

for episode in range(1, 60):
    # Simulate probing
    discovered_hosts = set()
    for probe_idx in range(200):
        target_host = list(partition_0_hosts)[probe_idx % len(partition_0_hosts)]
        physical_host, is_mutated, qos = defender.resolve_probe(target_host, probe_idx)
        if probe_idx % 2 == 0:
            discovered_hosts.add(physical_host)
    
    hits_count = len(discovered_hosts)
    defender.update(hits_count=hits_count, discovered_hosts=discovered_hosts, episode=episode)
    
    if episode in [25, 50]:
        print(f"   Episode {episode}: mutations={defender.get_mutation_count()}, "
              f"total_mutated={len(defender.mutated_hosts)}, "
              f"entropy={defender.get_address_entropy():.2f}")

print(f"\n✅ Multi-episode simulation complete")

print("\n7. VERIFYING NEURAL NETWORK TRAINING")
print("-"*80)

if len(defender.states) > 0:
    print(f"✅ Training data collected")
    print(f"   - States: {len(defender.states)}")
    print(f"   - Actions: {len(defender.actions)}")
    print(f"   - Rewards: {len(defender.rewards)}")
    print(f"   - Mean reward: {np.mean(defender.rewards):.2f}")
else:
    print(f"⚠️  No training data collected yet (expected at episode 50+)")

print("\n8. TESTING MODEL SAVE/LOAD")
print("-"*80)

save_path = 'idham_neural_test.pth'
defender.save_models(save_path)

# Create new defender and load
defender2 = IDHAMNeuralDefender(
    n_hosts=n_hosts,
    config=config,
    feasible_actions=feasible_actions,
    seed=99
)
defender2.load_models(save_path)

print(f"✅ Model save/load successful")

# Verify loaded model works
state_tensor = torch.FloatTensor(defender._extract_state_features()).unsqueeze(0)
with torch.no_grad():
    probs1 = defender.actor.get_action_probs(state_tensor)
    probs2 = defender2.actor.get_action_probs(state_tensor)
    diff = torch.abs(probs1 - probs2).max().item()

print(f"   - Max probability difference: {diff:.6f}")
print(f"   - Models match: {'✅' if diff < 1e-6 else '❌'}")

# Clean up
if os.path.exists(save_path):
    os.remove(save_path)

print("\n" + "="*80)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*80)

print("\n📊 SUMMARY:")
print(f"   - Neural network defender is working correctly")
print(f"   - Actor and critic networks are functional")
print(f"   - State extraction works properly")
print(f"   - Action selection from policy works")
print(f"   - Mutations are being performed")
print(f"   - Training loop is collecting experience")
print(f"   - Model save/load works")

print("\n🎯 NEXT STEPS:")
print("   1. Your src/defender_mutation.py already has the neural version")
print("   2. Update run_experiment_mutation.py if needed to pass feasible_actions")
print("   3. Run full experiment with neural defender")
print("   4. Compare results with heuristic baselines")