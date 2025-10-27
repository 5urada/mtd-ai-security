"""
Network environment simulation with TRUE ADDRESS MUTATION support.

Key changes:
- probe() now takes virtual_ip and uses defender to resolve to physical_host
- Supports both mutation and legacy masking for backward compatibility
"""

import numpy as np
from typing import Dict, Tuple, Optional


class NetworkEnvironment:
    """
    Simulates a network with real and fake hosts.
    Handles probe responses with ADDRESS MUTATION (not masking).
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
        
        # Randomly assign which PHYSICAL addresses are real hosts
        all_addresses = np.arange(n_hosts)
        self.rng.shuffle(all_addresses)
        self.real_hosts = set(all_addresses[:n_real])
        self.fake_hosts = set(all_addresses[n_real:])
        
        self.n_real_hosts = len(self.real_hosts)
        self.n_fake_hosts = len(self.fake_hosts)
        
        # Mutation effectiveness (small chance mutated probe still gets through)
        self.mutation_evasion_rate = 0.1  # 10% chance probe penetrates mutation
        
    def reset(self):
        """Reset environment state for new episode."""
        pass
    
    def probe_with_mutation(self, virtual_ip: int, physical_host: int, 
                           is_mutated: bool) -> bool:
        """
        Simulate a probe with address mutation.
        
        Args:
            virtual_ip: Virtual IP address being probed (what attacker sees)
            physical_host: Physical host ID (resolved by defender)
            is_mutated: Whether this address is currently mutated
            
        Returns:
            is_hit: True if probe receives a response (discovered a real host)
        """
        # Check if physical host is real
        if physical_host not in self.real_hosts:
            return False  # Fake host - no response
        
        # Real host
        if is_mutated:
            # Mutation provides some protection
            if self.rng.random() < self.mutation_evasion_rate:
                return True  # Probe got through despite mutation
            else:
                return False  # Mutation successfully confused attacker
        else:
            # Not mutated - always responds
            return True
    
    def probe(self, target: int, is_masked: bool) -> bool:
        """
        Legacy probe interface for backward compatibility with masking.
        
        Args:
            target: Target address (treated as physical host in legacy mode)
            is_masked: Whether address is masked (legacy semantics)
            
        Returns:
            is_hit: True if probe receives a response
        """
        # In legacy mode, target is physical host
        return self.probe_with_mutation(
            virtual_ip=target,
            physical_host=target,
            is_mutated=is_masked
        )
    
    def is_real_host(self, physical_host: int) -> bool:
        """Check if PHYSICAL host address is real."""
        return physical_host in self.real_hosts
    
    def get_topology_info(self) -> Dict:
        """Return topology information."""
        return {
            'n_hosts': self.n_hosts,
            'n_real_hosts': self.n_real_hosts,
            'n_fake_hosts': self.n_fake_hosts,
            'real_host_ratio': self.real_host_ratio
        }