import numpy as np
import torch
from typing import List, Dict, Tuple
from scipy.special import kl_div
from config import Config
from policy import ActorCritic

def compute_flip_sensitivity(
    policy: ActorCritic,
    states: List[np.ndarray],
    config: Config,
    device: torch.device
) -> Dict:
    """
    Compute flip-sensitivity metrics for sampled states.
    
    For each state S and each host j:
    - Flip bit j to create S^(j)
    - Compare policy distributions and chosen actions
    - Separate metrics for same-support vs different-support cases
    """
    from constraints import generate_feasible_assignments
    
    policy.eval()
    results = {
        "flip_rate_overall": 0.0,
        "flip_rate_same_support": 0.0,
        "flip_rate_diff_support": 0.0,
        "support_change_rate": 0.0,
        "kl_mean": 0.0,
        "tv_mean": 0.0,
        "assignment_hamming_mean": 0.0,
        "by_host": []
    }
    
    # Counters for decomposed metrics
    total_same_support = 0
    total_diff_support = 0
    flip_same_support = 0
    flip_diff_support = 0
    
    total_kl = 0.0
    total_tv = 0.0
    total_hamming = 0.0
    total_comparisons = 0
    
    host_flip_counts = [0] * config.N
    host_total_counts = [0] * config.N
    host_same_support_counts = [0] * config.N
    host_diff_support_counts = [0] * config.N
    
    for state_vec in states:
        # Decode state
        S = state_vec[:config.N].astype(np.int32)
        A_prev_flat = state_vec[config.N:config.N + config.N * config.B]
        A_prev = np.argmax(A_prev_flat.reshape(config.N, config.B), axis=1)
        
        # Generate feasible actions for original state
        # Use base seed from A_prev only - S affects constraints but not RNG path
        base_seed = abs(hash(tuple(A_prev.tolist()))) % (2**31)
        feasible_orig = generate_feasible_assignments(
            config, A_prev, S, config.K, use_ortools=False, seed=base_seed
        )
        
        if len(feasible_orig) == 0:
            continue
        
        # Get policy distribution for original state
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            logits_orig, _ = policy(obs_tensor, feasible_orig)
            probs_orig = torch.softmax(logits_orig, dim=-1).cpu().numpy()[0]
            action_orig_idx = np.argmax(probs_orig)
        
        # Test each host flip
        for j in range(config.N):
            S_flipped = S.copy()
            S_flipped[j] = 1 - S_flipped[j]
            
            # Generate feasible actions for flipped state
            # Use SAME base seed - let constraints naturally determine differences
            feasible_flipped = generate_feasible_assignments(
                config, A_prev, S_flipped, config.K, use_ortools=False, seed=base_seed
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
                logits_flipped, _ = policy(obs_flipped_tensor, feasible_flipped)
                probs_flipped = torch.softmax(logits_flipped, dim=-1).cpu().numpy()[0]
                action_flipped_idx = np.argmax(probs_flipped)
            
            # Hash candidates to find common support
            orig_hashes = {tuple(a.tolist()): i for i, a in enumerate(feasible_orig)}
            flip_hashes = {tuple(a.tolist()): i for i, a in enumerate(feasible_flipped)}
            common_assignments = set(orig_hashes.keys()) & set(flip_hashes.keys())
            
            host_total_counts[j] += 1
            total_comparisons += 1
            
            if len(common_assignments) > 0:
                # Case 1: Same support - measure pure policy sensitivity
                total_same_support += 1
                host_same_support_counts[j] += 1
                
                # Map common assignments to their indices
                common_orig_indices = [orig_hashes[c] for c in common_assignments]
                common_flip_indices = [flip_hashes[c] for c in common_assignments]
                
                # Get policy preferences over common assignments
                probs_orig_common = probs_orig[common_orig_indices]
                probs_flipped_common = probs_flipped[common_flip_indices]
                
                # Renormalize
                probs_orig_common = probs_orig_common / probs_orig_common.sum()
                probs_flipped_common = probs_flipped_common / probs_flipped_common.sum()
                
                # Find top actions in common set
                best_orig_common_idx = np.argmax(probs_orig_common)
                best_flip_common_idx = np.argmax(probs_flipped_common)
                
                # Get actual assignments
                best_orig_assignment = feasible_orig[common_orig_indices[best_orig_common_idx]]
                best_flip_assignment = feasible_flipped[common_flip_indices[best_flip_common_idx]]
                
                # Check if policy changed its mind on common support
                if not np.array_equal(best_orig_assignment, best_flip_assignment):
                    flip_same_support += 1
                    host_flip_counts[j] += 1
                
                # Compute KL divergence on common support
                kl = np.sum(probs_orig_common * np.log((probs_orig_common + 1e-10) / (probs_flipped_common + 1e-10)))
                total_kl += kl
                
                # Compute Total Variation on common support
                tv = 0.5 * np.sum(np.abs(probs_orig_common - probs_flipped_common))
                total_tv += tv
                
                # Assignment Hamming distance
                hamming_dist = np.mean(best_orig_assignment != best_flip_assignment)
                total_hamming += hamming_dist
                
            else:
                # Case 2: Different support - constraint-driven change
                total_diff_support += 1
                host_diff_support_counts[j] += 1
                
                # Compare top actions from different supports
                action_orig_assignment = feasible_orig[action_orig_idx]
                action_flip_assignment = feasible_flipped[action_flipped_idx]
                
                if not np.array_equal(action_orig_assignment, action_flip_assignment):
                    flip_diff_support += 1
                    host_flip_counts[j] += 1
                
                # KL/TV not meaningful when supports differ, but compute Hamming
                hamming_dist = np.mean(action_orig_assignment != action_flip_assignment)
                total_hamming += hamming_dist
    
    # Aggregate overall metrics
    if total_comparisons > 0:
        results["flip_rate_overall"] = float(
            (flip_same_support + flip_diff_support) / total_comparisons
        )
        results["assignment_hamming_mean"] = float(total_hamming / total_comparisons)
    
    if total_same_support > 0:
        results["flip_rate_same_support"] = float(flip_same_support / total_same_support)
        results["kl_mean"] = float(total_kl / total_same_support)
        results["tv_mean"] = float(total_tv / total_same_support)
    
    if total_diff_support > 0:
        results["flip_rate_diff_support"] = float(flip_diff_support / total_diff_support)
    
    if (total_same_support + total_diff_support) > 0:
        results["support_change_rate"] = float(
            total_diff_support / (total_same_support + total_diff_support)
        )
    
    # Per-host breakdown
    for j in range(config.N):
        if host_total_counts[j] > 0:
            results["by_host"].append({
                "host": int(j),
                "flip_rate": float(host_flip_counts[j] / host_total_counts[j]),
                "same_support_count": int(host_same_support_counts[j]),
                "diff_support_count": int(host_diff_support_counts[j])
            })
    
    policy.train()
    return results