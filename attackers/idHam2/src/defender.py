"""
Defender (ID-HAM) implementation with adaptive masking.
Simplified model that learns to mask addresses based on attacker behavior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Set, Tuple, Dict


class IDHAMDefender:
    """
    Simplified ID-HAM defender that learns adaptive address masking.
    Uses a policy network to decide which addresses to mask.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        """
        Args:
            n_hosts: Number of hosts in network
            config: Full experiment config
            seed: Random seed
        """
        self.n_hosts = n_hosts
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Hyperparameters
        self.mask_capacity = int(n_hosts * 0.3)  # Can mask up to 30% of addresses
        self.learning_rate = 0.001
        self.exploration_rate = 0.1
        self.qos_cost_per_mask = 0.5  # ms penalty per masked address
        
        # State tracking
        self.current_masked = set()
        self.previous_masked = set()
        self.hit_history = []
        self.episode_flow_mods = 0
        self.episode_qos_penalty = 0.0
        
        # Simple policy: track hit frequencies and mask high-hit addresses
        self.address_hit_counts = np.zeros(n_hosts)
        self.address_probe_counts = np.zeros(n_hosts)
        
        # For more sophisticated version: neural policy
        self.use_neural_policy = False  # Set to True for deep RL
        if self.use_neural_policy:
            self.policy_net = self._build_policy_network()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize with random masking instead of starting empty
        initial_mask_count = int(self.mask_capacity * 0.5)  # Start at 50% capacity
        initial_masked_indices = self.rng.choice(n_hosts, initial_mask_count, replace=False)
        self.current_masked = set(initial_masked_indices)
        self.previous_masked = self.current_masked.copy()
        
    def _build_policy_network(self) -> nn.Module:
        """Build simple neural network for masking policy."""
        return nn.Sequential(
            nn.Linear(self.n_hosts, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_hosts),
            nn.Sigmoid()  # Output: probability of masking each address
        )
    
    def reset_episode(self):
        """Reset per-episode state."""
        self.previous_masked = self.current_masked.copy()
        self.episode_flow_mods = 0
        self.episode_qos_penalty = 0.0
    
    def apply_defense(self, target: int, probe_idx: int) -> Tuple[bool, float]:
        """
        Apply defense for a probe to target address.
        
        Returns:
            is_masked: Whether this address is currently masked
            qos_cost: QoS penalty incurred (ms)
        """
        is_masked = target in self.current_masked
        qos_cost = self.qos_cost_per_mask if is_masked else 0.0
        self.episode_qos_penalty += qos_cost
        
        # Track probe
        self.address_probe_counts[target] += 1
        
        return is_masked, qos_cost
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """
        Update defender policy based on episode results.
        
        Args:
            hits_count: Number of successful probes this episode
            discovered_hosts: Set of hosts discovered this episode
            episode: Current episode number
        """
        # Update hit statistics
        for host in discovered_hosts:
            self.address_hit_counts[host] += 1
        
        # Update masking policy (every N episodes to avoid thrashing)
        # Skip episode 0 to keep initial random masking
        if episode > 0 and episode % 50 == 0:  # Changed from: if episode % 50 == 0:
            self._update_masking_policy()
    
    def _update_masking_policy(self):
        """Update which addresses to mask based on learned statistics."""
        # Simple heuristic policy: mask addresses with highest hit rates
        # Hit rate = hits / probes (with Laplace smoothing)
        hit_rates = (self.address_hit_counts + 1) / (self.address_probe_counts + 2)
        
        # Only consider addresses that have been probed at least once
        probed_mask = self.address_probe_counts > 0
        n_probed = probed_mask.sum()
        
        # If we don't have enough data yet, keep current masking
        # Require at least 10 probed addresses before updating
        if n_probed < 10:  # Changed from: if n_probed < self.mask_capacity:
            return  # Not enough data, keep current masks
        
        # Set hit rate to -1 for unprobed addresses so they won't be selected
        hit_rates_filtered = hit_rates.copy()
        hit_rates_filtered[~probed_mask] = -1
        
        # Add exploration noise
        if self.rng.random() < self.exploration_rate:
            noise = self.rng.normal(0, 0.1, size=self.n_hosts)
            noise[~probed_mask] = 0  # Don't add noise to unprobed addresses
            hit_rates_filtered = hit_rates_filtered + noise
        
        # Select top-k addresses to mask (from probed addresses)
        # Mask up to capacity, but only from addresses we've seen
        k = min(self.mask_capacity, n_probed)
        top_k_indices = np.argsort(hit_rates_filtered)[-k:]
        
        new_masked = set(top_k_indices)
        
        # Count flow modifications (addresses whose mask state changed)
        changed_addresses = (new_masked - self.current_masked) | (self.current_masked - new_masked)
        self.episode_flow_mods = len(changed_addresses)
        
        self.current_masked = new_masked
    
    def get_policy_entropy(self) -> float:
        """
        Calculate policy entropy as a measure of exploration vs exploitation.
        Higher entropy = more exploration.
        """
        # For heuristic policy: entropy based on hit rate distribution
        hit_rates = (self.address_hit_counts + 1) / (self.address_probe_counts + 2)
        hit_rates = hit_rates / (hit_rates.sum() + 1e-10)  # Normalize
        
        entropy = -np.sum(hit_rates * np.log(hit_rates + 1e-10))
        return float(entropy)
    
    def get_flow_mods_count(self) -> int:
        """Return number of flow table modifications this episode."""
        return self.episode_flow_mods
    
    def did_mask_change(self) -> bool:
        """Return whether masking configuration changed this episode."""
        return len(self.current_masked - self.previous_masked) > 0 or \
               len(self.previous_masked - self.current_masked) > 0
    
    def get_qos_penalty(self) -> float:
        """Return accumulated QoS penalty this episode (in ms)."""
        return self.episode_qos_penalty
    
    def get_current_masked_set(self) -> Set[int]:
        """Return current set of masked addresses."""
        return self.current_masked.copy()


class StaticDefender(IDHAMDefender):
    """Static baseline defender that doesn't adapt."""
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        super().__init__(n_hosts, config, seed)
        # Note: Parent __init__ now initializes with random masking
        # So StaticDefender will start with 15 random masked addresses
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """Static defender never updates its policy."""
        pass
    
    def _update_masking_policy(self):
        """Override to prevent updates."""
        pass