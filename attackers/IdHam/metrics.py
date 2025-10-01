import numpy as np
import torch
from typing import List, Dict, Tuple
from scipy.special import kl_div
from config import Config
from policy import ActorCritic

def compute_flip_sensitivity(
    policy: "ActorCritic",
    states: List[np.ndarray],
    config: Config,
    device: torch.device
) -> Dict:
    """
    Compute flip-sensitivity metrics for sampled states.
    
    For each state S and each host j:
    - Flip bit j to create S^(j)
    - Compare policy distributions and chosen actions
    """
    from constraints import generate_feasible_assignments
    
    policy.eval()
    results = {
        "flip_rate_overall": 0.0,
        "kl_mean": 0.0,
        "tv_mean": 0.0,
        "assignment_hamming_mean": 0.0,
        "by_host": []
    }
    
    total_flips = 0
    total_kl = 0.0
    total_tv = 0.0
    total_hamming = 0.0
    
    host_flip_counts = [0] * config.N
    host_total_counts = [0] * config.N
    
    for state_vec in states:
        # Decode state
        S = state_vec[:config.N].astype(np.int32)
        A_prev_flat = state_vec[config.N:config.N + config.N * config.B]
        A_prev = np.argmax(A_prev_flat.reshape(config.N, config.B), axis=1)
        
        # Generate feasible actions for original state
        feasible_orig = generate_feasible_assignments(
            config, A_prev, S, config.K, use_ortools=False, seed=config.seed
        )
        
        if len(feasible_orig) == 0:
            continue
        
        # Get policy distribution for original state
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            logits_orig, _ = policy(obs_tensor, len(feasible_orig))
            probs_orig = torch.softmax(logits_orig, dim=-1).cpu().numpy()[0]
            action_orig = np.argmax(probs_orig)
        
        # Test each host flip
        for j in range(config.N):
            S_flipped = S.copy()
            S_flipped[j] = 1 - S_flipped[j]
            
            # Generate feasible actions for flipped state
            feasible_flipped = generate_feasible_assignments(
                config, A_prev, S_flipped, config.K, use_ortools=False, seed=config.seed
            )
            
            if len(feasible_flipped) == 0:
                continue
            
            # Reconstruct observation for flipped state
            A_onehot = np.zeros((config.N, config.B), dtype=np.float32)
            A_onehot[np.arange(config.N), A_prev] = 1.0
            Z_norm = state_vec[-config.N:]
            obs_flipped = np.concatenate([
                S_flipped.astype(np.float32),
                A_onehot.flatten(),
                Z_norm
            ])
            
            # Get policy distribution for flipped state
            with torch.no_grad():
                obs_flipped_tensor = torch.FloatTensor(obs_flipped).unsqueeze(0).to(device)
                logits_flipped, _ = policy(obs_flipped_tensor, len(feasible_flipped))
                probs_flipped = torch.softmax(logits_flipped, dim=-1).cpu().numpy()[0]
                action_flipped = np.argmax(probs_flipped)
            
            # Top-1 flip rate
            if action_orig < len(feasible_orig) and action_flipped < len(feasible_flipped):
                if not np.array_equal(feasible_orig[action_orig], feasible_flipped[action_flipped]):
                    host_flip_counts[j] += 1
                    total_flips += 1
                
                # Assignment Hamming distance
                hamming_dist = np.mean(feasible_orig[action_orig] != feasible_flipped[action_flipped])
                total_hamming += hamming_dist
            
            # KL divergence (handle different dimensions)
            min_len = min(len(probs_orig), len(probs_flipped))
            if min_len > 0:
                p1 = probs_orig[:min_len] + 1e-10
                p2 = probs_flipped[:min_len] + 1e-10
                p1 = p1 / p1.sum()
                p2 = p2 / p2.sum()
                kl = np.sum(p1 * np.log(p1 / p2))
                total_kl += kl
            
            # Total Variation
            if min_len > 0:
                tv = 0.5 * np.sum(np.abs(probs_orig[:min_len] - probs_flipped[:min_len]))
                total_tv += tv
            
            host_total_counts[j] += 1
    
    # Aggregate metrics
    total_comparisons = sum(host_total_counts)
    if total_comparisons > 0:
        results["flip_rate_overall"] = total_flips / total_comparisons
        results["kl_mean"] = total_kl / total_comparisons
        results["tv_mean"] = total_tv / total_comparisons
        results["assignment_hamming_mean"] = total_hamming / total_comparisons
    
    # Per-host breakdown
    for j in range(config.N):
        if host_total_counts[j] > 0:
            results["by_host"].append({
                "host": j,
                "flip_rate": host_flip_counts[j] / host_total_counts[j]
            })
    
    policy.train()
    return results