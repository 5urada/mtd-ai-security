"""
Markov Decision Process Model for Host Address Mutation
Based on Section III-C of the paper
"""

import numpy as np
from typing import List, Tuple, Dict
import random

class HAM_MDP:
    """
    MDP Model for Host Address Mutation
    
    State: Network state represented as types of hosts (moving vs static)
    Action: IP address block assignments
    Reward: Based on scanning hits
    """
    
    def __init__(self, 
                 num_hosts: int,
                 num_blocks: int,
                 block_size: int = 128,
                 alpha: float = 1.0,
                 reward_constant: float = 10.0,
                 gamma: float = 0.99):
        """
        Initialize MDP
        
        Args:
            num_hosts: Number of hosts in network
            num_blocks: Number of IP address blocks
            block_size: Size of each IP address block (Z in paper)
            alpha: Coefficient for negative reward
            reward_constant: Positive constant C for avoiding scans
            gamma: Discount factor
        """
        self.num_hosts = num_hosts
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.alpha = alpha
        self.reward_constant = reward_constant
        self.gamma = gamma
        
        # Network state: 1 = moving host, 0 = static host
        self.state = np.zeros(num_hosts, dtype=int)
        
        # Scanning hits per host
        self.scan_hits = np.zeros(num_hosts, dtype=int)
        
        # State transition probabilities
        self.P_s = 0.3  # Probability static -> moving (after being scanned)
        self.P_m = 0.1  # Probability moving -> static
        
        # Current action (address block allocation)
        self.current_allocation = None
        
    def get_state(self) -> np.ndarray:
        """Get current network state"""
        return self.state.copy()
    
    def get_state_dim(self) -> int:
        """Get state dimension"""
        return self.num_hosts
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Initialize with some hosts as moving (e.g., 50%)
        self.state = np.random.binomial(1, 0.5, size=self.num_hosts)
        self.scan_hits = np.zeros(self.num_hosts, dtype=int)
        return self.get_state()
    
    def step(self, action: np.ndarray, scan_results: Dict[int, int]) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step in the MDP
        
        Args:
            action: Address block allocation (num_hosts x num_blocks binary matrix)
            scan_results: Dictionary mapping host_id to number of successful scans
        
        Returns:
            next_state: Next network state
            reward: Reward for this step
            done: Whether episode is complete
        """
        self.current_allocation = action
        
        # Update scan hits
        for host_id, hits in scan_results.items():
            self.scan_hits[host_id] = hits
        
        # Calculate reward based on Equation (2)
        reward = self.calculate_reward(scan_results)
        
        # State transitions
        next_state = self.transition_state(scan_results)
        
        # Episode never really "done" in continuous defense
        done = False
        
        return next_state, reward, done
    
    def calculate_reward(self, scan_results: Dict[int, int]) -> float:
        """
        Calculate reward based on Equation (2) in paper
        
        Rt = -α * Σ(Θi) if hosts are scanned successfully
        Rt = C if hosts avoid scanning
        """
        total_hits = sum(scan_results.values())
        
        if total_hits > 0:
            # Negative reward proportional to scan hits
            reward = -self.alpha * total_hits
        else:
            # Positive reward for avoiding scans
            reward = self.reward_constant
        
        return reward
    
    def transition_state(self, scan_results: Dict[int, int]) -> np.ndarray:
        """
        Transition network state based on scan results
        
        - Static host scanned -> may become moving with probability P_s
        - Moving host -> may become static with probability P_m
        """
        next_state = self.state.copy()
        
        for host_id in range(self.num_hosts):
            if host_id in scan_results and scan_results[host_id] > 0:
                # Host was scanned
                if self.state[host_id] == 0:  # Static host
                    # May transition to moving
                    if random.random() < self.P_s:
                        next_state[host_id] = 1
            else:
                # Host was not scanned
                if self.state[host_id] == 1:  # Moving host
                    # May transition to static
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
    
    def validate_action(self, action: np.ndarray) -> bool:
        """
        Validate that action satisfies basic constraints
        
        Args:
            action: Address block allocation matrix (num_hosts x num_blocks)
        
        Returns:
            True if action is valid
        """
        if action.shape != (self.num_hosts, self.num_blocks):
            return False
        
        # Each block should be assigned to at most one host
        block_assignments = action.sum(axis=0)
        if np.any(block_assignments > 1):
            return False
        
        # Each moving host should have at least one block
        moving_hosts = self.get_moving_hosts()
        for host_id in moving_hosts:
            if action[host_id].sum() == 0:
                return False
        
        return True


class AddressSpace:
    """Manage IP address space for mutation"""
    
    def __init__(self, subnet: str = "10.0.0.0/16", block_size: int = 128):
        """
        Initialize address space
        
        Args:
            subnet: Network subnet (CIDR notation)
            block_size: Size of each address block
        """
        self.subnet = subnet
        self.block_size = block_size
        
        # Parse subnet
        network, prefix = subnet.split('/')
        self.prefix_len = int(prefix)
        
        # Calculate total addresses
        self.total_addresses = 2 ** (32 - self.prefix_len)
        self.num_blocks = self.total_addresses // block_size
        
        # Track used addresses
        self.used_addresses = set()
        
    def get_num_blocks(self) -> int:
        """Get number of available blocks"""
        return self.num_blocks
    
    def allocate_blocks(self, num_blocks: int) -> List[Tuple[int, int]]:
        """
        Allocate IP address blocks
        
        Returns:
            List of (start_addr, end_addr) tuples
        """
        blocks = []
        for i in range(num_blocks):
            start = i * self.block_size
            end = start + self.block_size - 1
            blocks.append((start, end))
        return blocks


if __name__ == '__main__':
    # Test MDP model
    print("Testing HAM MDP Model")
    print("=" * 50)
    
    # Create MDP
    mdp = HAM_MDP(num_hosts=30, num_blocks=50)
    
    # Reset to initial state
    state = mdp.reset()
    print(f"Initial state: {state}")
    print(f"Moving hosts: {len(mdp.get_moving_hosts())}")
    print(f"Static hosts: {len(mdp.get_static_hosts())}")
    
    # Simulate a step
    action = np.zeros((30, 50))
    # Assign blocks to moving hosts
    for i, host_id in enumerate(mdp.get_moving_hosts()[:10]):
        action[host_id, i] = 1
    
    # Simulate some scans
    scan_results = {5: 2, 10: 1, 15: 3}  # host_id: num_hits
    
    next_state, reward, done = mdp.step(action, scan_results)
    print(f"\nAfter step:")
    print(f"Reward: {reward}")
    print(f"Next state: {next_state}")
    print(f"Moving hosts: {len(mdp.get_moving_hosts())}")
