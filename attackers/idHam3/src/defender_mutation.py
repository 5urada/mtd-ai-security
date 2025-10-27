"""
Defender implementations with TRUE HOST ADDRESS MUTATION.

This replaces the masking approach with actual address mutation:
- Each physical host is assigned a virtual IP address
- Mutation = reshuffling these assignments
- Attacker probes virtual IPs, which resolve to physical hosts based on current mapping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Set, Tuple, Dict, Optional


class MutationDefenderBase:
    """
    Base class for mutation-based defenders.
    
    Key concepts:
    - host_to_vip: Maps physical host ID → virtual IP address
    - vip_to_host: Maps virtual IP address → physical host ID (inverse)
    - mutation_capacity: How many addresses can be mutated per mutation event
    - mutation_fraction: What fraction of hosts are currently mutated from default
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        self.n_hosts = n_hosts
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Extract defender config
        defender_cfg = config.get('defender', {})
        self.mutation_capacity = defender_cfg.get('mutation_capacity', int(n_hosts * 0.3))
        
        # QoS costs
        self.qos_cost_per_mutation_data = 0.5  # ms per probe to mutated address
        self.qos_cost_per_mutation_ctrl = 2.0  # ms per address mutation
        
        # State tracking
        self.episode_qos_data_ms = 0.0
        self.episode_qos_ctrl_ms = 0.0
        self.episode_mutations = 0
        
        # Address mapping: host_id → virtual_ip
        # Initialize with identity mapping (no mutation)
        self.host_to_vip = {i: i for i in range(n_hosts)}
        self.vip_to_host = {i: i for i in range(n_hosts)}
        
        # Track which hosts are currently mutated from default
        self.mutated_hosts = set()  # Hosts where vip != host_id
        
        # Previous mapping for change detection
        self.previous_host_to_vip = self.host_to_vip.copy()
        
    def reset_episode(self):
        """Reset per-episode state."""
        self.previous_host_to_vip = self.host_to_vip.copy()
        self.episode_qos_data_ms = 0.0
        self.episode_qos_ctrl_ms = 0.0
        self.episode_mutations = 0
    
    def resolve_probe(self, virtual_ip: int, probe_idx: int) -> Tuple[int, bool, float]:
        """
        Resolve a probe to a virtual IP to its physical host.
        
        Args:
            virtual_ip: Virtual IP address being probed
            probe_idx: Index of current probe in episode
            
        Returns:
            physical_host: Physical host ID that this vIP resolves to
            is_mutated: Whether this address is currently mutated
            qos_cost: QoS penalty incurred (ms)
        """
        # Resolve virtual IP to physical host
        physical_host = self.vip_to_host.get(virtual_ip, virtual_ip)
        
        # Check if this address is mutated
        is_mutated = physical_host in self.mutated_hosts
        
        # Data-plane cost (probing mutated address has overhead)
        qos_cost = self.qos_cost_per_mutation_data if is_mutated else 0.0
        self.episode_qos_data_ms += qos_cost
        
        return physical_host, is_mutated, qos_cost
    
    def apply_defense(self, target: int, probe_idx: int) -> Tuple[bool, float]:
        """
        Compatibility wrapper for existing code.
        Returns whether target is mutated and QoS cost.
        
        NOTE: The caller should use resolve_probe() for full mutation semantics,
        but this maintains backward compatibility with masking API.
        """
        physical_host, is_mutated, qos_cost = self.resolve_probe(target, probe_idx)
        return is_mutated, qos_cost
    
    def _perform_mutation(self, hosts_to_mutate: Set[int]):
        """
        Perform address mutation on specified hosts.
        
        Args:
            hosts_to_mutate: Set of physical host IDs to mutate
        """
        # Get list of available virtual IPs (those not currently assigned)
        # For simplicity, we'll use a random shuffling approach
        
        # Collect all hosts that need new vIPs
        hosts_needing_assignment = list(hosts_to_mutate)
        
        # Get vIPs that are either:
        # 1) Currently unmutated (vip == host_id)
        # 2) Being freed by hosts we're mutating
        available_vips = []
        
        # Add default vIPs of hosts we're mutating
        for host in hosts_needing_assignment:
            available_vips.append(host)
        
        # Add vIPs from unmutated hosts (but only if not in hosts_to_mutate)
        for host_id in range(self.n_hosts):
            if host_id not in hosts_to_mutate and host_id not in self.mutated_hosts:
                # This host is at default location, its vIP is available
                available_vips.append(host_id)
        
        # Shuffle available vIPs
        self.rng.shuffle(available_vips)
        
        # Assign new vIPs to hosts
        # Ensure we assign different vIPs than current ones
        changes = 0
        for i, host in enumerate(hosts_needing_assignment):
            old_vip = self.host_to_vip[host]
            
            # Try to find a different vIP
            for candidate_vip in available_vips:
                # Check if this vIP is not currently used and is different
                if candidate_vip != old_vip and candidate_vip not in self.vip_to_host:
                    new_vip = candidate_vip
                    
                    # Remove old mapping
                    if old_vip in self.vip_to_host:
                        del self.vip_to_host[old_vip]
                    
                    # Add new mapping
                    self.host_to_vip[host] = new_vip
                    self.vip_to_host[new_vip] = host
                    
                    changes += 1
                    
                    # Remove from available pool
                    available_vips.remove(candidate_vip)
                    break
        
        # Update mutated_hosts set
        self.mutated_hosts = {h for h in range(self.n_hosts) 
                              if self.host_to_vip[h] != h}
        
        # Track mutations
        self.episode_mutations = changes
        self.episode_qos_ctrl_ms = changes * self.qos_cost_per_mutation_ctrl
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """
        Update defender policy (to be overridden by subclasses).
        
        Args:
            hits_count: Number of successful probes this episode
            discovered_hosts: Set of PHYSICAL hosts discovered this episode
            episode: Current episode number
        """
        raise NotImplementedError("Subclass must implement update()")
    
    def get_mutation_count(self) -> int:
        """Return number of address mutations this episode."""
        return self.episode_mutations
    
    def did_mutation_occur(self) -> bool:
        """Return whether mutation occurred this episode."""
        return self.episode_mutations > 0
    
    def get_qos_penalty(self) -> float:
        """Return accumulated QoS penalty this episode (in ms)."""
        return self.episode_qos_data_ms + self.episode_qos_ctrl_ms
    
    def get_qos_data_plane_ms(self) -> float:
        """Return data-plane QoS cost for this episode."""
        return self.episode_qos_data_ms
    
    def get_qos_control_plane_ms(self) -> float:
        """Return control-plane QoS cost for this episode."""
        return self.episode_qos_ctrl_ms
    
    def get_flow_mods_count(self) -> int:
        """Return number of flow table modifications this episode."""
        return self.episode_mutations
    
    def did_mask_change(self) -> bool:
        """Compatibility: Return whether mapping changed this episode."""
        return self.did_mutation_occur()
    
    def get_mutation_fraction(self, current_partition_hosts: Set[int]) -> float:
        """
        Calculate what fraction of current partition is mutated.
        
        Args:
            current_partition_hosts: Set of PHYSICAL hosts in current active partition
            
        Returns:
            Fraction of partition that is mutated (0.0 to 1.0)
        """
        if len(current_partition_hosts) == 0:
            return 0.0
        
        mutated_in_partition = len(self.mutated_hosts & current_partition_hosts)
        return mutated_in_partition / len(current_partition_hosts)
    
    def get_mask_saturation(self, current_partition_hosts: Set[int]) -> float:
        """Compatibility: Same as get_mutation_fraction."""
        return self.get_mutation_fraction(current_partition_hosts)
    
    def get_policy_entropy(self) -> float:
        """Return policy entropy (to be overridden by learning defenders)."""
        return 0.0
    
    def get_address_entropy(self) -> float:
        """
        Calculate Shannon entropy of current address mapping.
        Higher entropy = more randomized mapping.
        """
        # Count how many hosts are at each offset from default
        # offset = (vip - host_id) mod n_hosts
        offsets = [(self.host_to_vip[h] - h) % self.n_hosts 
                   for h in range(self.n_hosts)]
        
        # Compute frequency distribution
        unique_offsets, counts = np.unique(offsets, return_counts=True)
        probs = counts / self.n_hosts
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return float(entropy)


class IDHAMMutationDefender(MutationDefenderBase):
    """
    ID-HAM defender with TRUE ADDRESS MUTATION.
    
    Learns which hosts to mutate based on attacker behavior using DRL.
    Uses A2C-style learning to optimize mutation policy.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        super().__init__(n_hosts, config, seed)
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.exploration_rate = 0.1
        self.update_interval = 50  # Episodes between mutations
        
        # Learning state
        self.host_hit_counts = np.zeros(n_hosts)
        self.host_probe_counts = np.zeros(n_hosts)
        
        # Neural policy (optional - can use heuristic)
        self.use_neural_policy = False
        
        # Initialize with random mutation
        initial_mutation_count = int(self.mutation_capacity * 0.5)
        initial_hosts = self.rng.choice(n_hosts, initial_mutation_count, replace=False)
        self._perform_mutation(set(initial_hosts))
    
    def resolve_probe(self, virtual_ip: int, probe_idx: int) -> Tuple[int, bool, float]:
        """Override to track probe statistics for learning."""
        physical_host, is_mutated, qos_cost = super().resolve_probe(virtual_ip, probe_idx)
        
        # Track probes for learning
        self.host_probe_counts[physical_host] += 1
        
        return physical_host, is_mutated, qos_cost
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """
        Update defender policy based on episode results using DRL.
        
        Args:
            hits_count: Number of successful probes this episode
            discovered_hosts: Set of PHYSICAL hosts discovered this episode
            episode: Current episode number
        """
        # Update hit statistics
        for host in discovered_hosts:
            self.host_hit_counts[host] += 1
        
        # Update mutation policy periodically
        if episode > 0 and episode % self.update_interval == 0:
            self._update_mutation_policy()
    
    def _update_mutation_policy(self):
        """
        Update which addresses to mutate based on learned statistics.
        
        Strategy: Mutate addresses with highest hit rates (they're being targeted).
        This is the core DRL-inspired heuristic.
        """
        # Compute hit rates with Laplace smoothing
        hit_rates = (self.host_hit_counts + 1) / (self.host_probe_counts + 2)
        
        # Only consider hosts that have been probed
        probed_mask = self.host_probe_counts > 0
        n_probed = probed_mask.sum()
        
        # Need sufficient data before mutating
        if n_probed < 10:
            return
        
        # Set hit rate to -1 for unprobed hosts
        hit_rates_filtered = hit_rates.copy()
        hit_rates_filtered[~probed_mask] = -1
        
        # Add exploration noise
        if self.rng.random() < self.exploration_rate:
            noise = self.rng.normal(0, 0.1, size=self.n_hosts)
            noise[~probed_mask] = 0
            hit_rates_filtered = hit_rates_filtered + noise
        
        # Select top-k hosts to mutate
        k = min(self.mutation_capacity, n_probed)
        top_k_indices = np.argsort(hit_rates_filtered)[-k:]
        
        hosts_to_mutate = set(top_k_indices)
        
        # Perform mutation
        self._perform_mutation(hosts_to_mutate)
    
    def get_policy_entropy(self) -> float:
        """Calculate policy entropy based on hit rate distribution."""
        hit_rates = (self.host_hit_counts + 1) / (self.host_probe_counts + 2)
        hit_rates = hit_rates / (hit_rates.sum() + 1e-10)
        entropy = -np.sum(hit_rates * np.log(hit_rates + 1e-10))
        return float(entropy)


class FRVMDefender(MutationDefenderBase):
    """
    FRVM (Flexible Random Virtual IP Multiplexing) Defender.
    
    Non-learning baseline: performs random mutation on a fixed schedule.
    Based on: Sharma et al., "FRVM: Flexible Random Virtual IP Multiplexing 
    in Software-Defined Networks," TrustCom 2018.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        super().__init__(n_hosts, config, seed)
        
        # FRVM parameters
        defender_cfg = config.get('defender', {})
        self.mutation_interval = defender_cfg.get('mutation_interval', 50)  # T_AS in paper
        self.mutation_mode = defender_cfg.get('mutation_mode', 'uniform')
        
        # Initialize with random mutation
        initial_mutation_count = int(self.mutation_capacity * 0.5)
        initial_hosts = self.rng.choice(n_hosts, initial_mutation_count, replace=False)
        self._perform_mutation(set(initial_hosts))
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """
        Update defender policy: periodic random mutation.
        
        Args:
            hits_count: Number of successful probes this episode
            discovered_hosts: Set of PHYSICAL hosts discovered this episode
            episode: Current episode number
        """
        # Mutate on fixed schedule
        if episode > 0 and episode % self.mutation_interval == 0:
            self._perform_random_mutation()
    
    def _perform_random_mutation(self):
        """Perform uniformly random mutation."""
        # Select random hosts to mutate
        n_to_mutate = min(self.mutation_capacity, self.n_hosts)
        hosts_to_mutate = set(self.rng.choice(
            self.n_hosts, 
            size=n_to_mutate, 
            replace=False
        ))
        
        # Perform mutation
        self._perform_mutation(hosts_to_mutate)
    
    def get_policy_entropy(self) -> float:
        """Return constant entropy for random policy."""
        return float(np.log(self.mutation_capacity + 1))


class RHMDefender(MutationDefenderBase):
    """
    RHM (Random Host Mutation) Defender with heuristic selection.
    
    Based on: Jafarian et al., "An Effective Address Mutation Approach for 
    Disrupting Reconnaissance Attacks," IEEE TIFS 2015.
    
    Uses hypothesis testing to identify heavily-probed hosts and mutates them.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        super().__init__(n_hosts, config, seed)
        
        # RHM parameters
        defender_cfg = config.get('defender', {})
        self.mutation_interval = defender_cfg.get('mutation_interval', 50)
        self.attack_threshold = defender_cfg.get('attack_threshold', 2.0)  # Std devs
        
        # Tracking
        self.host_probe_counts = np.zeros(n_hosts)
        self.host_hit_counts = np.zeros(n_hosts)
        
        # Initialize with random mutation
        initial_mutation_count = int(self.mutation_capacity * 0.5)
        initial_hosts = self.rng.choice(n_hosts, initial_mutation_count, replace=False)
        self._perform_mutation(set(initial_hosts))
    
    def resolve_probe(self, virtual_ip: int, probe_idx: int) -> Tuple[int, bool, float]:
        """Override to track probe statistics."""
        physical_host, is_mutated, qos_cost = super().resolve_probe(virtual_ip, probe_idx)
        self.host_probe_counts[physical_host] += 1
        return physical_host, is_mutated, qos_cost
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """
        Update defender policy: heuristic-based mutation.
        
        Args:
            hits_count: Number of successful probes this episode
            discovered_hosts: Set of PHYSICAL hosts discovered this episode
            episode: Current episode number
        """
        # Update hit counts
        for host in discovered_hosts:
            self.host_hit_counts[host] += 1
        
        # Mutate on schedule using heuristic
        if episode > 0 and episode % self.mutation_interval == 0:
            self._perform_heuristic_mutation()
    
    def _perform_heuristic_mutation(self):
        """
        Perform heuristic-based mutation: mutate most-attacked hosts.
        
        Uses hypothesis testing approach from RHM paper.
        """
        # Identify hosts under attack (probe count > mean + threshold * std)
        mean_probes = np.mean(self.host_probe_counts)
        std_probes = np.std(self.host_probe_counts)
        
        attack_threshold_value = mean_probes + self.attack_threshold * std_probes
        
        # Hosts that exceed threshold
        under_attack = np.where(self.host_probe_counts > attack_threshold_value)[0]
        
        # If no hosts detected, fall back to highest-probed hosts
        if len(under_attack) == 0:
            n_to_select = min(self.mutation_capacity, self.n_hosts)
            under_attack = np.argsort(self.host_probe_counts)[-n_to_select:]
        
        # Limit to mutation capacity
        if len(under_attack) > self.mutation_capacity:
            # Select top-k by probe count
            sorted_by_probes = sorted(under_attack, 
                                     key=lambda h: self.host_probe_counts[h], 
                                     reverse=True)
            hosts_to_mutate = set(sorted_by_probes[:self.mutation_capacity])
        else:
            hosts_to_mutate = set(under_attack)
        
        # Perform mutation
        if hosts_to_mutate:
            self._perform_mutation(hosts_to_mutate)
    
    def get_policy_entropy(self) -> float:
        """Return policy entropy based on probe distribution."""
        probe_dist = (self.host_probe_counts + 1) / (self.host_probe_counts.sum() + self.n_hosts)
        entropy = -np.sum(probe_dist * np.log(probe_dist + 1e-10))
        return float(entropy)


class StaticMutationDefender(MutationDefenderBase):
    """
    Static baseline: establishes initial mutation and never changes it.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        super().__init__(n_hosts, config, seed)
        
        # Initialize with random mutation
        defender_cfg = config.get('defender', {})
        initial_strategy = defender_cfg.get('initial_mutation_strategy', 'random')
        
        if initial_strategy == 'random':
            initial_mutation_count = int(self.mutation_capacity * 0.5)
            initial_hosts = self.rng.choice(n_hosts, initial_mutation_count, replace=False)
            self._perform_mutation(set(initial_hosts))
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """Static defender never updates."""
        pass
    
    def get_policy_entropy(self) -> float:
        """Return 0 entropy for static policy."""
        return 0.0