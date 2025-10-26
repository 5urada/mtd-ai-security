"""
Attacker strategies for ID-HAM experiment.
Implements divide-and-conquer partitioning with timing variations.

FIX APPLIED:
2. Probe selection without replacement within each episode
"""

import numpy as np
from typing import List, Dict, Set, Tuple


class DivideConquerAttacker:
    """
    Divide-and-conquer attacker that partitions the address space
    and rotates between partitions with configurable timing.
    """
    
    def __init__(self, n_hosts: int, partitioning_config: Dict, 
                 switch_config: Dict, seed: int = 0):
        """
        Args:
            n_hosts: Total number of hosts in address space
            partitioning_config: Dict with 'mode' and 'partitions'
            switch_config: Dict with 'mode', 'mean_interval', 'jitter_pct'
            seed: Random seed
        """
        self.n_hosts = n_hosts
        self.n_partitions = partitioning_config['partitions']
        self.partition_mode = partitioning_config['mode']
        
        self.switch_mode = switch_config['mode']
        self.mean_interval = switch_config['mean_interval']
        self.jitter_pct = switch_config['jitter_pct']
        
        self.rng = np.random.RandomState(seed)
        
        # Create partitions
        self.partitions = self._create_partitions()
        
        # Initialize partition schedule
        self.current_partition_idx = 0
        self.next_switch_episode = self._schedule_next_switch(0)
        self.switch_history = []
        
        # FIX 2: Add state for sampling without replacement
        self.current_shuffle = None
        self.shuffle_position = 0
        self.last_partition_idx = None
        
    def _create_partitions(self) -> List[Set[int]]:
        """Create address space partitions based on mode."""
        all_hosts = np.arange(self.n_hosts)
        partitions = []
        
        if self.partition_mode == 'uniform':
            # Equal-sized partitions
            partition_size = self.n_hosts // self.n_partitions
            for i in range(self.n_partitions):
                start_idx = i * partition_size
                end_idx = start_idx + partition_size if i < self.n_partitions - 1 else self.n_hosts
                partitions.append(set(all_hosts[start_idx:end_idx]))
                
        elif self.partition_mode == 'skewed':
            # One large partition (50%) + rest equally split
            large_size = self.n_hosts // 2
            small_size = (self.n_hosts - large_size) // (self.n_partitions - 1)
            
            # Large partition
            partitions.append(set(all_hosts[:large_size]))
            
            # Small partitions
            remaining_start = large_size
            for i in range(self.n_partitions - 1):
                start_idx = remaining_start + i * small_size
                end_idx = start_idx + small_size if i < self.n_partitions - 2 else self.n_hosts
                partitions.append(set(all_hosts[start_idx:end_idx]))
        else:
            raise ValueError(f"Unknown partition mode: {self.partition_mode}")
        
        return partitions
    
    def _schedule_next_switch(self, current_episode: int) -> int:
        """Schedule the next partition switch based on timing mode."""
        if self.switch_mode == 'periodic':
            # Periodic with jitter
            base_interval = self.mean_interval
            if self.jitter_pct > 0:
                jitter_range = base_interval * self.jitter_pct
                jitter = self.rng.uniform(-jitter_range, jitter_range)
                interval = int(base_interval + jitter)
            else:
                interval = base_interval
            return current_episode + interval
            
        elif self.switch_mode == 'poisson':
            # Poisson-distributed intervals (exponential inter-arrival)
            lambd = 1.0 / self.mean_interval
            interval = int(self.rng.exponential(1.0 / lambd))
            interval = max(interval, 10)  # Minimum 10 episodes between switches
            return current_episode + interval
        else:
            raise ValueError(f"Unknown switch mode: {self.switch_mode}")
    
    def get_current_partition(self, episode: int) -> int:
        """
        Get current partition index for this episode.
        Updates partition if switch is due.
        """
        # Check if we should switch
        if episode >= self.next_switch_episode:
            # Switch to next partition
            self.current_partition_idx = (self.current_partition_idx + 1) % self.n_partitions
            self.switch_history.append({
                'episode': episode,
                'partition': self.current_partition_idx
            })
            # Schedule next switch
            self.next_switch_episode = self._schedule_next_switch(episode)
            
            # FIX 2: Reset shuffle when partition changes
            self.current_shuffle = None
            self.shuffle_position = 0
        
        # FIX 2: Track partition changes
        self.last_partition_idx = self.current_partition_idx
        
        return self.current_partition_idx
    
    def get_partition_targets(self, partition_idx: int) -> Set[int]:
        """Get set of host addresses in the specified partition."""
        return self.partitions[partition_idx]
    
    def select_target(self, target_hosts: Set[int], probe_idx: int) -> int:
        """
        FIX 2: Select a target host from the current partition WITHOUT REPLACEMENT.
        
        Within each episode, we shuffle the partition once and cycle through it.
        This prevents duplicate probes to the same target within an episode.
        
        Args:
            target_hosts: Set of hosts in current partition
            probe_idx: Index of current probe (resets each episode at 0)
            
        Returns:
            Selected target host address
        """
        # FIX 2: Initialize shuffle at start of episode (probe_idx == 0)
        # or when partition changes
        if probe_idx == 0 or self.current_shuffle is None:
            targets_list = list(target_hosts)
            self.rng.shuffle(targets_list)
            self.current_shuffle = targets_list
            self.shuffle_position = 0
        
        # FIX 2: Select target using modulo indexing (cycle through shuffle)
        target = self.current_shuffle[self.shuffle_position % len(self.current_shuffle)]
        self.shuffle_position += 1
        
        return target
    
    def get_partition_sizes(self) -> List[int]:
        """Return sizes of all partitions."""
        return [len(p) for p in self.partitions]
    
    def get_switch_history(self) -> List[Dict]:
        """Return history of partition switches."""
        return self.switch_history