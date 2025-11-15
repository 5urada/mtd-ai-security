"""
Improved MDP Model with Block-Aware Rewards
CRITICAL FIX: Rewards should differentiate which blocks are being scanned
"""

import numpy as np
from typing import List, Tuple, Dict
import random

class ImprovedHAM_MDP:
    """
    Enhanced MDP Model that tracks block-level scanning history
    """
    
    def __init__(self, 
                 num_hosts: int,
                 num_blocks: int,
                 block_size: int = 128,
                 alpha: float = 1.0,
                 reward_constant: float = 10.0,
                 gamma: float = 0.99):
        """Initialize improved MDP"""
        self.num_hosts = num_hosts
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.alpha = alpha
        self.reward_constant = reward_constant
        self.gamma = gamma
        
        # Network state: 1 = moving host, 0 = static host
        self.state = np.zeros(num_hosts, dtype=int)
        
        # NEW: Track which blocks are being scanned (heat map)
        self.block_scan_history = np.zeros(num_blocks, dtype=float)
        self.block_scan_decay = 0.95  # Decay factor for history
        
        # NEW: Track effectiveness of block assignments
        self.block_effectiveness = np.ones(num_blocks, dtype=float)
        
        # Current action (address block allocation)
        self.current_allocation = None
        
        # State transition probabilities
        self.P_s = 0.3
        self.P_m = 0.1
        
    def get_state(self) -> np.ndarray:
        """
        Get enhanced state representation
        Includes: host types + block scan history
        """
        # Normalize scan history to [0, 1]
        max_scans = np.max(self.block_scan_history) if np.max(self.block_scan_history) > 0 else 1
        normalized_history = self.block_scan_history / max_scans
        
        # Concatenate: [host_states, block_history]
        enhanced_state = np.concatenate([
            self.state.astype(float),
            normalized_history
        ])
        
        return enhanced_state
    
    def get_state_dim(self) -> int:
        """Enhanced state dimension: hosts + blocks"""
        return self.num_hosts + self.num_blocks
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.state = np.random.binomial(1, 0.5, size=self.num_hosts)
        self.block_scan_history = np.zeros(self.num_blocks, dtype=float)
        self.block_effectiveness = np.ones(self.num_blocks, dtype=float)
        return self.get_state()
    
    def step(self, 
             action: np.ndarray, 
             scan_results: Dict[int, int],
             scanned_addresses: Dict[int, int]) -> Tuple[np.ndarray, float, bool]:
        """
        Execute step with enhanced reward calculation
        
        Args:
            action: Address block allocation
            scan_results: Dict mapping host_id to hits
            scanned_addresses: Dict mapping address to host_id (NEW)
        """
        self.current_allocation = action
        
        # Update block scan history based on which addresses were scanned
        self._update_block_history(scanned_addresses)
        
        # Calculate block-aware reward
        reward = self.calculate_block_aware_reward(action, scan_results)
        
        # Update block effectiveness
        self._update_block_effectiveness(action, scan_results)
        
        # State transitions
        next_state = self.transition_state(scan_results)
        
        done = False
        return next_state, reward, done
    
    def _update_block_history(self, scanned_addresses: Dict[int, int]):
        """
        Update which blocks are being targeted by adversary
        
        Args:
            scanned_addresses: Dict of address -> count
        """
        # Decay old history
        self.block_scan_history *= self.block_scan_decay
        
        # Add new scans
        for address, count in scanned_addresses.items():
            block_id = address // self.block_size
            if 0 <= block_id < self.num_blocks:
                self.block_scan_history[block_id] += count
    
    def _update_block_effectiveness(self, action: np.ndarray, scan_results: Dict[int, int]):
        """
        Track which block assignments led to hits
        Blocks that got scanned are less effective
        """
        for host_id, hits in scan_results.items():
            if hits > 0:
                # Which blocks does this host use?
                assigned_blocks = np.where(action[host_id] == 1)[0]
                
                # Penalize these blocks (they got scanned)
                for block_id in assigned_blocks:
                    self.block_effectiveness[block_id] *= 0.95  # Decay effectiveness
        
        # Restore effectiveness slowly for unused blocks
        self.block_effectiveness = np.clip(
            self.block_effectiveness * 1.01,
            0.1, 1.0
        )
    
    def calculate_block_aware_reward(self, 
                                    action: np.ndarray, 
                                    scan_results: Dict[int, int]) -> float:
        """
        CRITICAL FIX: Reward that considers which blocks were assigned
        
        Rewards actions that:
        1. Avoid heavily scanned blocks
        2. Use blocks with high historical effectiveness
        3. Minimize total hits
        """
        total_hits = sum(scan_results.values())
        
        # Component 1: Penalty for hits (original)
        hit_penalty = -self.alpha * total_hits
        
        # Component 2: Reward for avoiding hot blocks
        hot_block_bonus = 0.0
        for host_id in range(self.num_hosts):
            assigned_blocks = np.where(action[host_id] == 1)[0]
            if len(assigned_blocks) > 0:
                # Average scan heat of assigned blocks
                avg_heat = np.mean(self.block_scan_history[assigned_blocks])
                # Lower heat = better (inverse reward)
                max_heat = np.max(self.block_scan_history) if np.max(self.block_scan_history) > 0 else 1
                hot_block_bonus += (1.0 - avg_heat / max_heat) * 0.5
        
        # Component 3: Reward for using effective blocks
        effectiveness_bonus = 0.0
        for host_id in range(self.num_hosts):
            assigned_blocks = np.where(action[host_id] == 1)[0]
            if len(assigned_blocks) > 0:
                avg_effectiveness = np.mean(self.block_effectiveness[assigned_blocks])
                effectiveness_bonus += avg_effectiveness * 0.5
        
        # Combined reward
        if total_hits > 0:
            reward = hit_penalty + hot_block_bonus + effectiveness_bonus
        else:
            # Big bonus for completely avoiding scans
            reward = self.reward_constant + hot_block_bonus + effectiveness_bonus
        
        return reward
    
    def transition_state(self, scan_results: Dict[int, int]) -> np.ndarray:
        """State transitions (unchanged)"""
        next_state = self.state.copy()
        
        for host_id in range(self.num_hosts):
            if host_id in scan_results and scan_results[host_id] > 0:
                if self.state[host_id] == 0:  # Static host scanned
                    if random.random() < self.P_s:
                        next_state[host_id] = 1
            else:
                if self.state[host_id] == 1:  # Moving host not scanned
                    if random.random() < self.P_m:
                        next_state[host_id] = 0
        
        self.state = next_state
        return self.get_state()
    
    def get_moving_hosts(self) -> List[int]:
        """Get list of moving host indices"""
        return [i for i in range(self.num_hosts) if self.state[i] == 1]
    
    def get_static_hosts(self) -> List[int]:
        """Get list of static host indices"""
        return [i for i in range(self.num_hosts) if self.state[i] == 0]


if __name__ == '__main__':
    print("Testing Improved HAM MDP")
    print("=" * 50)
    
    mdp = ImprovedHAM_MDP(num_hosts=30, num_blocks=50)
    
    state = mdp.reset()
    print(f"Initial state dimension: {len(state)}")
    print(f"  - Host states: {state[:30]}")
    print(f"  - Block history: {state[30:]}")
    
    # Simulate some scans
    action = np.zeros((30, 50))
    for i in range(30):
        blocks = np.random.choice(50, 3, replace=False)
        action[i, blocks] = 1
    
    scan_results = {5: 2, 10: 1}
    scanned_addresses = {100: 1, 150: 1, 200: 1}  # Addresses that were scanned
    
    next_state, reward, done = mdp.step(action, scan_results, scanned_addresses)
    print(f"\nAfter step:")
    print(f"Reward: {reward:.2f}")
    print(f"Block scan history (first 10): {mdp.block_scan_history[:10]}")
