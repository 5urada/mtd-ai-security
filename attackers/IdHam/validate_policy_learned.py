import torch
import numpy as np
from config import Config
from policy import ActorCritic
from env import IDHAMEnv, AttackerType
from constraints import generate_feasible_assignments

checkpoint = torch.load('runs/local_pref/ckpt_latest.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']

policy = ActorCritic(config.N + config.N * config.B + config.N, config.N, config.B)
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval()

env = IDHAMEnv(config, AttackerType.LOCAL_PREFERENCE)

print("Checking if policy learned (not uniform random)...")
print("="*70)

entropy_values = []

for _ in range(20):
    obs = env.reset()
    S = obs[:config.N].astype(np.int32)
    A_prev = np.argmax(obs[config.N:config.N + config.N * config.B].reshape(config.N, config.B), axis=1)
    base_seed = abs(hash(tuple(A_prev.tolist()))) % (2**31)
    
    feasible = generate_feasible_assignments(config, A_prev, S, config.K,
                                             use_ortools=False, seed=base_seed)
    
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        logits, _ = policy(obs_tensor, feasible)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Calculate entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    entropy_values.append(entropy)
    
    print(f"State {_+1}: Entropy={entropy:.4f}, Max prob={probs.max():.4f}, "
          f"Top action prob={probs[np.argmax(probs)]:.4f}")

uniform_entropy = -np.log(1.0 / config.K)  # log(16) = 2.77

print(f"\n{'='*70}")
print(f"Average entropy: {np.mean(entropy_values):.4f}")
print(f"Uniform random entropy: {uniform_entropy:.4f}")
print(f"Policy is {'LEARNED' if np.mean(entropy_values) < uniform_entropy * 0.8 else 'NOT LEARNED'}")
print(f"{'='*70}")