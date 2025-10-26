"""
Baseline defender implementations for comparison with ID-HAM.
Provides non-learning MTD alternatives to test NDV effectiveness.
"""

import numpy as np
from typing import Set, Tuple, Dict


class StaticMaskDefender:
    """
    Fixed mask set for entire run (no learning).
    
    This defender establishes a mask at initialization and never changes it,
    serving as a baseline to test whether ID-HAM's learning provides benefit.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        """
        Args:
            n_hosts: Number of hosts in network
            config: Full experiment config including defender.mask_capacity
            seed: Random seed
        """
        self.n_hosts = n_hosts
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Extract defender config
        defender_cfg = config.get('defender', {})
        self.mask_capacity = defender_cfg.get('mask_capacity', int(n_hosts * 0.3))
        initial_strategy = defender_cfg.get('initial_mask_strategy', 'random')
        
        # QoS costs (same as ID-HAM for fair comparison)
        self.qos_cost_per_mask_data = 0.5  # ms per masked probe
        
        # State tracking
        self.episode_qos_data_ms = 0.0
        self.episode_qos_ctrl_ms = 0.0  # Always 0 for static
        
        # Initialize static mask
        self.current_masked = self._initialize_mask(initial_strategy)
        
    def _initialize_mask(self, strategy: str) -> Set[int]:
        """Initialize mask based on strategy."""
        if strategy == 'random':
            # Random selection of addresses to mask
            masked_indices = self.rng.choice(
                self.n_hosts, 
                size=min(self.mask_capacity, self.n_hosts),
                replace=False
            )
            return set(masked_indices)
        
        elif strategy == 'top-k-seeded':
            # Deterministic selection based on seed
            # Use hash of addresses to create consistent "importance" ranking
            importance = [(i, hash((self.rng.get_state()[1][0], i))) for i in range(self.n_hosts)]
            importance.sort(key=lambda x: x[1], reverse=True)
            return set([i for i, _ in importance[:self.mask_capacity]])
        
        else:
            raise ValueError(f"Unknown initial_mask_strategy: {strategy}")
    
    def reset_episode(self):
        """Reset per-episode state."""
        self.episode_qos_data_ms = 0.0
        self.episode_qos_ctrl_ms = 0.0
    
    def apply_defense(self, target: int, probe_idx: int) -> Tuple[bool, float]:
        """
        Apply defense for a probe to target address.
        
        Returns:
            is_masked: Whether this address is currently masked
            qos_cost: QoS penalty incurred (ms)
        """
        is_masked = target in self.current_masked
        qos_cost = self.qos_cost_per_mask_data if is_masked else 0.0
        self.episode_qos_data_ms += qos_cost
        return is_masked, qos_cost
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """
        Update defender policy (no-op for static defender).
        
        Args:
            hits_count: Number of successful probes this episode
            discovered_hosts: Set of hosts discovered this episode
            episode: Current episode number
        """
        # Static defender never updates
        pass
    
    def get_flow_mods_count(self) -> int:
        """Return number of flow table modifications this episode."""
        return 0  # Static never modifies
    
    def did_mask_change(self) -> bool:
        """Return whether masking configuration changed this episode."""
        return False  # Static never changes
    
    def get_qos_penalty(self) -> float:
        """Return accumulated QoS penalty this episode (in ms)."""
        return self.episode_qos_data_ms + self.episode_qos_ctrl_ms
    
    def get_qos_data_plane_ms(self) -> float:
        """Return data-plane QoS cost (masked probes) for this episode."""
        return self.episode_qos_data_ms
    
    def get_qos_control_plane_ms(self) -> float:
        """Return control-plane QoS cost (flow modifications) for this episode."""
        return self.episode_qos_ctrl_ms  # Always 0
    
    def get_current_masked_set(self) -> Set[int]:
        """Return current set of masked addresses."""
        return self.current_masked.copy()
    
    def get_mask_saturation(self, current_partition_hosts: Set[int]) -> float:
        """
        Calculate what fraction of current partition is masked.
        
        Args:
            current_partition_hosts: Set of hosts in current active partition
            
        Returns:
            Fraction of partition that is masked (0.0 to 1.0)
        """
        if len(current_partition_hosts) == 0:
            return 0.0
        
        masked_in_partition = len(self.current_masked & current_partition_hosts)
        return masked_in_partition / len(current_partition_hosts)
    
    def get_policy_entropy(self) -> float:
        """Return policy entropy (0 for static)."""
        return 0.0


class RandomizedMTDDefender:
    """
    Non-learning randomized MTD: re-masks by simple policy on a fixed cadence.
    
    This defender periodically randomizes its mask set without learning from
    attacker behavior, serving as a "moving target" baseline.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        """
        Args:
            n_hosts: Number of hosts in network
            config: Full experiment config including defender settings
            seed: Random seed
        """
        self.n_hosts = n_hosts
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Extract defender config
        defender_cfg = config.get('defender', {})
        self.mask_capacity = defender_cfg.get('mask_capacity', int(n_hosts * 0.3))
        self.remask_interval = defender_cfg.get('remask_interval', 50)
        self.remask_mode = defender_cfg.get('remask_mode', 'uniform')
        initial_strategy = defender_cfg.get('initial_mask_strategy', 'random')
        
        # QoS costs
        self.qos_cost_per_mask_data = 0.5  # ms per masked probe
        self.qos_cost_per_flow_mod_ctrl = 2.0  # ms per flow mod
        
        # State tracking
        self.current_masked = set()
        self.previous_masked = set()
        self.episode_qos_data_ms = 0.0
        self.episode_qos_ctrl_ms = 0.0
        self.episode_flow_mods = 0
        
        # For reservoir sampling mode
        self.seen_addresses = set()
        
        # Initialize first mask
        self.current_masked = self._initialize_mask(initial_strategy)
        self.previous_masked = self.current_masked.copy()
    
    def _initialize_mask(self, strategy: str) -> Set[int]:
        """Initialize mask based on strategy."""
        if strategy == 'random':
            masked_indices = self.rng.choice(
                self.n_hosts,
                size=min(self.mask_capacity, self.n_hosts),
                replace=False
            )
            return set(masked_indices)
        elif strategy == 'top-k-seeded':
            importance = [(i, hash((self.rng.get_state()[1][0], i))) for i in range(self.n_hosts)]
            importance.sort(key=lambda x: x[1], reverse=True)
            return set([i for i, _ in importance[:self.mask_capacity]])
        else:
            raise ValueError(f"Unknown initial_mask_strategy: {strategy}")
    
    def reset_episode(self):
        """Reset per-episode state."""
        self.previous_masked = self.current_masked.copy()
        self.episode_qos_data_ms = 0.0
        self.episode_qos_ctrl_ms = 0.0
        self.episode_flow_mods = 0
    
    def apply_defense(self, target: int, probe_idx: int) -> Tuple[bool, float]:
        """
        Apply defense for a probe to target address.
        
        Returns:
            is_masked: Whether this address is currently masked
            qos_cost: QoS penalty incurred (ms)
        """
        is_masked = target in self.current_masked
        qos_cost = self.qos_cost_per_mask_data if is_masked else 0.0
        self.episode_qos_data_ms += qos_cost
        
        # Track seen addresses for reservoir mode
        if self.remask_mode == 'reservoir':
            self.seen_addresses.add(target)
        
        return is_masked, qos_cost
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """
        Update defender policy (remask on cadence).
        
        Args:
            hits_count: Number of successful probes this episode
            discovered_hosts: Set of hosts discovered this episode
            episode: Current episode number
        """
        # Check if it's time to remask
        if episode > 0 and episode % self.remask_interval == 0:
            self._remask()
    
    def _remask(self):
        """Perform remasking based on configured mode."""
        if self.remask_mode == 'uniform':
            # Uniform random remasking across all addresses
            new_masked_indices = self.rng.choice(
                self.n_hosts,
                size=min(self.mask_capacity, self.n_hosts),
                replace=False
            )
            new_masked = set(new_masked_indices)
        
        elif self.remask_mode == 'reservoir':
            # Reservoir sampling: prefer addresses we've seen
            if len(self.seen_addresses) >= self.mask_capacity:
                # Sample from seen addresses
                seen_list = list(self.seen_addresses)
                new_masked_indices = self.rng.choice(
                    seen_list,
                    size=min(self.mask_capacity, len(seen_list)),
                    replace=False
                )
                new_masked = set(new_masked_indices)
            else:
                # Not enough seen addresses, use uniform
                new_masked_indices = self.rng.choice(
                    self.n_hosts,
                    size=min(self.mask_capacity, self.n_hosts),
                    replace=False
                )
                new_masked = set(new_masked_indices)
        
        else:
            raise ValueError(f"Unknown remask_mode: {self.remask_mode}")
        
        # Calculate flow modifications
        changed_addresses = (new_masked - self.current_masked) | (self.current_masked - new_masked)
        self.episode_flow_mods = len(changed_addresses)
        
        # Charge control-plane cost
        self.episode_qos_ctrl_ms = self.episode_flow_mods * self.qos_cost_per_flow_mod_ctrl
        
        # Update mask
        self.current_masked = new_masked
    
    def get_flow_mods_count(self) -> int:
        """Return number of flow table modifications this episode."""
        return self.episode_flow_mods
    
    def did_mask_change(self) -> bool:
        """Return whether masking configuration changed this episode."""
        return len(self.current_masked - self.previous_masked) > 0 or \
               len(self.previous_masked - self.current_masked) > 0
    
    def get_qos_penalty(self) -> float:
        """Return accumulated QoS penalty this episode (in ms)."""
        return self.episode_qos_data_ms + self.episode_qos_ctrl_ms
    
    def get_qos_data_plane_ms(self) -> float:
        """Return data-plane QoS cost (masked probes) for this episode."""
        return self.episode_qos_data_ms
    
    def get_qos_control_plane_ms(self) -> float:
        """Return control-plane QoS cost (flow modifications) for this episode."""
        return self.episode_qos_ctrl_ms
    
    def get_current_masked_set(self) -> Set[int]:
        """Return current set of masked addresses."""
        return self.current_masked.copy()
    
    def get_mask_saturation(self, current_partition_hosts: Set[int]) -> float:
        """
        Calculate what fraction of current partition is masked.
        
        Args:
            current_partition_hosts: Set of hosts in current active partition
            
        Returns:
            Fraction of partition that is masked (0.0 to 1.0)
        """
        if len(current_partition_hosts) == 0:
            return 0.0
        
        masked_in_partition = len(self.current_masked & current_partition_hosts)
        return masked_in_partition / len(current_partition_hosts)
    
    def get_policy_entropy(self) -> float:
        """Return policy entropy (constant for random MTD)."""
        # Entropy of uniform distribution over n_hosts choose mask_capacity
        # Simplified: return log of capacity
        return float(np.log(self.mask_capacity + 1))