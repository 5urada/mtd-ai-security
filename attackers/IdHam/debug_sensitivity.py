import torch
from config import Config
from policy import ActorCritic
from env import IDHAMEnv, AttackerType
from constraints import generate_feasible_assignments
import numpy as np

# Load checkpoint
checkpoint = torch.load('runs/exp3/ckpt_latest.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']

# Initialize policy with correct architecture
policy = ActorCritic(config.N + config.N * config.B + config.N, config.N, config.B)
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval()

print("✓ Policy loaded successfully")

# Test with one state
env = IDHAMEnv(config, AttackerType.STATIC)
obs = env.reset()

S = obs[:config.N].astype(np.int32)
A_prev = np.argmax(obs[config.N:config.N + config.N * config.B].reshape(config.N, config.B), axis=1)

# Generate candidates
feasible = generate_feasible_assignments(config, A_prev, S, config.K, use_ortools=False, seed=42)
print(f"✓ Generated {len(feasible)} feasible candidates")

# Get policy predictions
with torch.no_grad():
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    logits, value = policy(obs_tensor, feasible)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

print(f"✓ Policy forward pass successful")
print(f"Probabilities: {probs}")
print(f"Min: {probs.min():.4f}, Max: {probs.max():.4f}, Std: {probs.std():.4f}")

if probs.std() > 0.01:
    print("✓ Policy is producing varied probabilities (learning worked!)")
else:
    print("✗ Policy still producing uniform probabilities (learning failed)")