import torch
import numpy as np
from config import Config
from policy import ActorCritic
from env import IDHAMEnv, AttackerType
from constraints import generate_feasible_assignments

# Load checkpoint
checkpoint = torch.load('runs/local_pref/ckpt_latest.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']

# Initialize policy
policy = ActorCritic(config.N + config.N * config.B + config.N, config.N, config.B)
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval()

# Create environment
env = IDHAMEnv(config, AttackerType.LOCAL_PREFERENCE)
obs = env.reset()

# Decode state
S = obs[:config.N].astype(np.int32)
A_prev = np.argmax(obs[config.N:config.N + config.N * config.B].reshape(config.N, config.B), axis=1)

print("="*70)
print("MANUAL VALIDATION: Testing 5 different host flips")
print("="*70)

for test_host in [0, 5, 10, 15, 20]:
    print(f"\n--- Testing flip of host {test_host} (S[{test_host}] = {S[test_host]}) ---")
    
    # Original state
    base_seed = abs(hash(tuple(A_prev.tolist()))) % (2**31)
    feasible_orig = generate_feasible_assignments(config, A_prev, S, config.K, 
                                                   use_ortools=False, seed=base_seed)
    
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        logits_orig, _ = policy(obs_tensor, feasible_orig)
        probs_orig = torch.softmax(logits_orig, dim=-1).cpu().numpy()[0]
        action_orig = np.argmax(probs_orig)
    
    print(f"Original - Top action: {action_orig}, Prob: {probs_orig[action_orig]:.4f}")
    print(f"Original - Top 3 probs: {sorted(probs_orig, reverse=True)[:3]}")
    
    # Flipped state
    S_flipped = S.copy()
    S_flipped[test_host] = 1 - S_flipped[test_host]
    
    feasible_flip = generate_feasible_assignments(config, A_prev, S_flipped, config.K,
                                                   use_ortools=False, seed=base_seed)
    
    # Reconstruct observation
    A_onehot = np.zeros((config.N, config.B), dtype=np.float32)
    A_onehot[np.arange(config.N), A_prev] = 1.0
    Z_norm = obs[-config.N:]
    obs_flip = np.concatenate([S_flipped.astype(np.float32), A_onehot.flatten(), Z_norm])
    
    with torch.no_grad():
        obs_flip_tensor = torch.FloatTensor(obs_flip).unsqueeze(0)
        logits_flip, _ = policy(obs_flip_tensor, feasible_flip)
        probs_flip = torch.softmax(logits_flip, dim=-1).cpu().numpy()[0]
        action_flip = np.argmax(probs_flip)
    
    print(f"Flipped - Top action: {action_flip}, Prob: {probs_flip[action_flip]:.4f}")
    print(f"Flipped - Top 3 probs: {sorted(probs_flip, reverse=True)[:3]}")
    
    # Compare
    orig_set = {tuple(a.tolist()) for a in feasible_orig}
    flip_set = {tuple(a.tolist()) for a in feasible_flip}
    common = orig_set & flip_set
    
    print(f"Common assignments: {len(common)}/{config.K}")
    
    if action_orig < len(feasible_orig) and action_flip < len(feasible_flip):
        same_assignment = np.array_equal(feasible_orig[action_orig], feasible_flip[action_flip])
        print(f"Same chosen assignment: {same_assignment}")
        
        if not same_assignment:
            print(f"  FLIP DETECTED!")
            print(f"  Original assignment: {feasible_orig[action_orig]}")
            print(f"  Flipped assignment:  {feasible_flip[action_flip]}")