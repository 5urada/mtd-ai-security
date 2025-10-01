import torch
import numpy as np
import json
import os
from pathlib import Path
from env import AttackerType

def evaluate_sensitivity(
    ckpt_path: str,
    num_samples: int = 200,
    attacker_type: AttackerType = AttackerType.STATIC,
    output_path: str = "reports/sensitivity.json"
):
    """Evaluate flip-sensitivity of trained policy."""
    from env import IDHAMEnv
    from policy import ActorCritic
    from metrics import compute_flip_sensitivity
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize policy
    env = IDHAMEnv(config, attacker_type)
    policy = ActorCritic(env.observation_dim, config.N, config.B).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    print(f"Loaded policy from {ckpt_path}")
    print(f"Collecting {num_samples} state samples...")
    
    # Collect state samples
    states = []
    obs = env.reset()
    for _ in range(num_samples * 2):  # Oversample
        states.append(obs.copy())
        
        # Generate feasible actions and take random step
        S = obs[:config.N].astype(np.int32)
        A_prev_flat = obs[config.N:config.N + config.N * config.B]
        A_prev = np.argmax(A_prev_flat.reshape(config.N, config.B), axis=1)
        
        from constraints import generate_feasible_assignments
        feasible = generate_feasible_assignments(
            config, A_prev, S, config.K, use_ortools=False, seed=None
        )
        
        if len(feasible) > 0:
            action = np.random.randint(0, len(feasible))
            obs, _, done, _ = env.step(action, feasible)
            if done:
                obs = env.reset()
        else:
            obs = env.reset()
    
    # Sample uniformly
    states = states[:num_samples]
    
    print("Computing flip-sensitivity metrics...")
    results = compute_flip_sensitivity(policy, states, config, device)
    
    # Save report
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSensitivity Report:")
    print(f"  Overall Flip Rate: {results['flip_rate_overall']:.3f}")
    print(f"  Mean KL Divergence: {results['kl_mean']:.3f}")
    print(f"  Mean Total Variation: {results['tv_mean']:.3f}")
    print(f"  Mean Assignment Hamming: {results['assignment_hamming_mean']:.3f}")
    print(f"\nFull report saved to: {output_path}")
