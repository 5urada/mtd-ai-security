"""
Network environment simulation for ID-HAM experiment.
"""

import numpy as np
from typing import Dict


class NetworkEnvironment:
    """
    Simulates a network with real and fake hosts.
    Handles probe responses with masking effects.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        """
        Args:
            n_hosts: Total number of addresses in network
            config: Experiment configuration
            seed: Random seed
        """
        self.n_hosts = n_hosts
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Determine real vs fake hosts
        # For simplicity: 60% of addresses are real hosts
        self.real_host_ratio = 0.6
        n_real = int(n_hosts * self.real_host_ratio)
        
        # Randomly assign which addresses are real hosts
        all_addresses = np.arange(n_hosts)
        self.rng.shuffle(all_addresses)
        self.real_hosts = set(all_addresses[:n_real])
        self.fake_hosts = set(all_addresses[n_real:])
        
        self.n_real_hosts = len(self.real_hosts)
        self.n_fake_hosts = len(self.fake_hosts)
        
        # Masking reduces probability of response
        self.mask_effectiveness = 0.7  # 70% chance masked probe gets no response
        
    def reset(self):
        """Reset environment state for new episode."""
        # For static topology, nothing to reset
        pass
    
    def probe(self, target: int, is_masked: bool) -> bool:
        """
        Simulate a probe to target address.
        
        Args:
            target: Target address (0 to n_hosts-1)
            is_masked: Whether defender has masked this address
            
        Returns:
            is_hit: True if probe receives a response (discovered a real host)
        """
        # Check if target is a real host
        if target not in self.real_hosts:
            # Fake host - no response
            return False
        
        # Real host
        if is_masked:
            # Masking reduces probability of response
            if self.rng.random() < self.mask_effectiveness:
                return False  # Masked successfully
            else:
                return True  # Probe got through despite masking
        else:
            # Not masked - always responds
            return True
    
    def is_real_host(self, address: int) -> bool:
        """Check if address is a real host."""
        return address in self.real_hosts
    
    def get_topology_info(self) -> Dict:
        """Return topology information."""
        return {
            'n_hosts': self.n_hosts,
            'n_real_hosts': self.n_real_hosts,
            'n_fake_hosts': self.n_fake_hosts,
            'real_host_ratio': self.real_host_ratio
        }