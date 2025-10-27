"""
FIXED + NEURAL NETWORK: Defender implementations with TRUE HOST ADDRESS MUTATION + DRL

This implements the full ID-HAM approach from the paper:
- Advantage Actor-Critic (A2C) framework
- Neural network policy for selecting IP block assignments
- Learning from scanning behaviors

BUG FIXES INCLUDED:
- Fixed _perform_mutation() logic (shuffling instead of checking availability)
- Episode 0 mutations allowed
- Reduced probe threshold

NEW: Full DRL implementation with PyTorch networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Set, Tuple, Dict, Optional, List
from collections import deque


class PolicyNetwork(nn.Module):
    """
    Actor network for learning mutation policy.
    
    Input: State features (host status, hit rates, probe counts, etc.)
    Output: Probability distribution over feasible IP block assignments
    """
    
    def __init__(self, state_dim: int, n_feasible_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_feasible_actions = n_feasible_actions
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (outputs logits for each feasible action)
        self.policy_head = nn.Linear(hidden_dim, n_feasible_actions)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim] tensor of state features
        
        Returns:
            logits: [batch_size, n_feasible_actions] logits for action distribution
        """
        features = self.encoder(state)
        logits = self.policy_head(features)
        return logits
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over actions."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action_idx: Index of selected action
            log_prob: Log probability of selected action
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        return action_idx.item(), log_prob


class ValueNetwork(nn.Module):
    """
    Critic network for estimating state value.
    
    Input: State features
    Output: Estimated value V(s)
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim] tensor
        
        Returns:
            value: [batch_size, 1] estimated value
        """
        features = self.encoder(state)
        value = self.value_head(features)
        return value


class MutationDefenderBase:
    """
    Base class for mutation-based defenders.
    Includes fixed _perform_mutation() logic.
    """
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        self.n_hosts = n_hosts
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Extract defender config
        defender_cfg = config.get('defender', {})
        self.mutation_capacity = defender_cfg.get('mutation_capacity', int(n_hosts * 0.3))
        
        # QoS costs
        self.qos_cost_per_mutation_data = 0.5
        self.qos_cost_per_mutation_ctrl = 2.0
        
        # State tracking
        self.episode_qos_data_ms = 0.0
        self.episode_qos_ctrl_ms = 0.0
        self.episode_mutations = 0
        
        # Address mapping
        self.host_to_vip = {i: i for i in range(n_hosts)}
        self.vip_to_host = {i: i for i in range(n_hosts)}
        self.mutated_hosts = set()
        self.previous_host_to_vip = self.host_to_vip.copy()
        
    def reset_episode(self):
        """Reset per-episode state."""
        self.previous_host_to_vip = self.host_to_vip.copy()
        self.episode_qos_data_ms = 0.0
        self.episode_qos_ctrl_ms = 0.0
        self.episode_mutations = 0
    
    def resolve_probe(self, virtual_ip: int, probe_idx: int) -> Tuple[int, bool, float]:
        """Resolve virtual IP to physical host."""
        physical_host = self.vip_to_host.get(virtual_ip, virtual_ip)
        is_mutated = physical_host in self.mutated_hosts
        qos_cost = self.qos_cost_per_mutation_data if is_mutated else 0.0
        self.episode_qos_data_ms += qos_cost
        return physical_host, is_mutated, qos_cost
    
    def _perform_mutation(self, hosts_to_mutate: Set[int]):
        """
        FIXED: Perform address mutation by shuffling vIP assignments.
        """
        if not hosts_to_mutate:
            return
        
        hosts_list = list(hosts_to_mutate)
        n_to_mutate = len(hosts_list)
        
        if n_to_mutate < 2:
            host = hosts_list[0]
            old_vip = self.host_to_vip[host]
            candidates = [h for h in range(self.n_hosts) if h not in hosts_to_mutate]
            if candidates:
                swap_host = self.rng.choice(candidates)
                swap_vip = self.host_to_vip[swap_host]
                self.host_to_vip[host] = swap_vip
                self.host_to_vip[swap_host] = old_vip
                self.vip_to_host[old_vip] = swap_host
                self.vip_to_host[swap_vip] = host
                self.episode_mutations = 2
            return
        
        old_vips = [self.host_to_vip[h] for h in hosts_list]
        new_vips = old_vips.copy()
        
        # Shuffle until enough changes
        attempts = 0
        while attempts < 100:
            self.rng.shuffle(new_vips)
            changes = sum(1 for i in range(n_to_mutate) if old_vips[i] != new_vips[i])
            if changes >= max(2, int(n_to_mutate * 0.8)):
                break
            attempts += 1
        
        # Apply new assignments
        changes_made = 0
        for i, host in enumerate(hosts_list):
            old_vip = old_vips[i]
            new_vip = new_vips[i]
            
            if old_vip != new_vip:
                if old_vip in self.vip_to_host:
                    del self.vip_to_host[old_vip]
                self.host_to_vip[host] = new_vip
                self.vip_to_host[new_vip] = host
                changes_made += 1
        
        self.mutated_hosts = {h for h in range(self.n_hosts) 
                              if self.host_to_vip[h] != h}
        self.episode_mutations = changes_made
        self.episode_qos_ctrl_ms = changes_made * self.qos_cost_per_mutation_ctrl
    
    def get_mutation_count(self) -> int:
        return self.episode_mutations
    
    def did_mutation_occur(self) -> bool:
        return self.episode_mutations > 0
    
    def get_qos_data_plane_ms(self) -> float:
        return self.episode_qos_data_ms
    
    def get_qos_control_plane_ms(self) -> float:
        return self.episode_qos_ctrl_ms
    
    def get_mutation_fraction(self, current_partition_hosts: Set[int]) -> float:
        if len(current_partition_hosts) == 0:
            return 0.0
        mutated_in_partition = len(self.mutated_hosts & current_partition_hosts)
        return mutated_in_partition / len(current_partition_hosts)
    
    def get_policy_entropy(self) -> float:
        return 0.0
    
    def get_address_entropy(self) -> float:
        offsets = [(self.host_to_vip[h] - h) % self.n_hosts 
                   for h in range(self.n_hosts)]
        unique_offsets, counts = np.unique(offsets, return_counts=True)
        probs = counts / self.n_hosts
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return float(entropy)
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        raise NotImplementedError("Subclass must implement update()")


class IDHAMNeuralDefender(MutationDefenderBase):
    """
    ID-HAM defender with FULL NEURAL NETWORK implementation.
    
    Uses Advantage Actor-Critic (A2C) to learn mutation policy.
    Learns which IP address blocks to assign to hosts based on scanning behavior.
    """
    
    def __init__(self, n_hosts: int, config: Dict, 
                 feasible_actions: List[Dict[int, List[int]]] = None,
                 seed: int = 0):
        """
        Args:
            n_hosts: Number of hosts
            config: Configuration dict
            feasible_actions: List of feasible IP block assignments
                             Each action is dict: {host_id: [block_ids]}
            seed: Random seed
        """
        super().__init__(n_hosts, config, seed)
        
        # Hyperparameters
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001
        self.gamma = 0.99  # Discount factor
        self.beta = 0.01   # Entropy coefficient
        self.update_interval = 50  # Episodes between mutations
        
        # Learning state
        self.host_hit_counts = np.zeros(n_hosts)
        self.host_probe_counts = np.zeros(n_hosts)
        self.episode_count = 0
        
        # Feasible actions (from SMT solver)
        self.feasible_actions = feasible_actions if feasible_actions else [{}]
        self.n_feasible_actions = len(self.feasible_actions)
        
        # State dimension: host_status + hit_rates + probe_counts + coverage
        self.state_dim = n_hosts * 3 + 1  # [moving_status, hit_rates, probe_counts, coverage]
        
        # Initialize neural networks
        self.actor = PolicyNetwork(
            state_dim=self.state_dim,
            n_feasible_actions=self.n_feasible_actions,
            hidden_dim=128
        )
        
        self.critic = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dim=128
        )
        
        # Optimizers
        self.actor_optimizer = optim.RMSprop(
            self.actor.parameters(), 
            lr=self.learning_rate_actor
        )
        
        self.critic_optimizer = optim.RMSprop(
            self.critic.parameters(),
            lr=self.learning_rate_critic
        )
        
        # Experience buffer for current episode
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        
        # Initialize with random mutation
        print(f"[IDHAMNeural] Initializing with random mutation...")
        initial_mutation_count = int(self.mutation_capacity * 0.5)
        initial_hosts = self.rng.choice(n_hosts, initial_mutation_count, replace=False)
        self._perform_mutation(set(initial_hosts))
        print(f"[IDHAMNeural] Initial mutation: {len(self.mutated_hosts)} hosts mutated")
    
    def _extract_state_features(self, coverage: float = 0.0) -> np.ndarray:
        """
        Extract state features for neural network input.
        
        Features:
        - Host moving status (binary): n_hosts
        - Normalized hit rates: n_hosts
        - Normalized probe counts: n_hosts
        - Coverage: 1
        
        Returns:
            state: [state_dim] numpy array
        """
        # Moving host status (1 if host is mutated, 0 otherwise)
        moving_status = np.array([1.0 if i in self.mutated_hosts else 0.0 
                                  for i in range(self.n_hosts)])
        
        # Normalized hit rates (with Laplace smoothing)
        hit_rates = (self.host_hit_counts + 1) / (self.host_probe_counts + 2)
        hit_rates = hit_rates / (hit_rates.max() + 1e-10)
        
        # Normalized probe counts
        probe_counts_norm = self.host_probe_counts / (self.host_probe_counts.max() + 1e-10)
        
        # Combine features
        state = np.concatenate([
            moving_status,
            hit_rates,
            probe_counts_norm,
            [coverage]
        ])
        
        return state.astype(np.float32)
    
    def resolve_probe(self, virtual_ip: int, probe_idx: int) -> Tuple[int, bool, float]:
        """Override to track probe statistics for learning."""
        physical_host, is_mutated, qos_cost = super().resolve_probe(virtual_ip, probe_idx)
        self.host_probe_counts[physical_host] += 1
        return physical_host, is_mutated, qos_cost
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        """
        Update defender policy using DRL.
        
        Args:
            hits_count: Number of successful probes this episode
            discovered_hosts: Set of PHYSICAL hosts discovered
            episode: Current episode number
        """
        self.episode_count = episode
        
        # Update hit statistics
        for host in discovered_hosts:
            self.host_hit_counts[host] += 1
        
        # Calculate reward based on scanning hits
        # Reward function from paper (Equation 2)
        alpha = 1.0  # Coefficient
        C = 10.0     # Positive constant for avoiding scans
        
        if hits_count > 0:
            reward = -alpha * hits_count
        else:
            reward = C
        
        # Perform mutation at regular intervals
        if episode % self.update_interval == 0:
            print(f"[IDHAMNeural] Episode {episode}: Mutation check triggered")
            
            # Store reward ONLY when we also store state/action
            self.rewards.append(reward)
            
            self._update_mutation_policy_neural(episode)
            
            # Train networks if we have enough experience
            if len(self.rewards) >= 2:  # Need at least 2 for returns calculation
                self._train_networks()
                
                # Clear experience buffer after training
                self.states = []
                self.actions = []
                self.log_probs = []
                self.rewards = []
                self.values = []
    
    def _update_mutation_policy_neural(self, episode: int):
        """
        Update mutation policy using neural networks.
        
        This is the core DRL-based mutation selection.
        """
        # Check if we have enough data
        probed_mask = self.host_probe_counts > 0
        n_probed = int(probed_mask.sum())
        
        print(f"[IDHAMNeural]   Probed hosts: {n_probed}")
        
        if n_probed < 5:
            print(f"[IDHAMNeural]   Not enough probed hosts ({n_probed} < 5), using random")
            # Fall back to random selection
            n_to_mutate = min(self.mutation_capacity, self.n_hosts)
            hosts_to_mutate = set(self.rng.choice(
                self.n_hosts, 
                size=n_to_mutate, 
                replace=False
            ))
            self._perform_mutation(hosts_to_mutate)
            return
        
        # Extract current state
        coverage = len(discovered_hosts) / self.n_hosts if hasattr(self, 'discovered_hosts') else 0.0
        state = self._extract_state_features(coverage)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Select action using actor network
        with torch.no_grad():
            action_idx, log_prob = self.actor.sample_action(state_tensor)
            value = self.critic(state_tensor)
        
        # Store experience
        self.states.append(state)
        self.actions.append(action_idx)
        self.log_probs.append(log_prob)
        self.values.append(value.item())
        
        # Get the selected feasible action
        if action_idx < len(self.feasible_actions):
            selected_action = self.feasible_actions[action_idx]
            
            # Extract hosts to mutate from action
            # Action format: {host_id: [block_ids]}
            hosts_to_mutate = set(selected_action.keys()) if selected_action else set()
            
            # If no specific assignment or using simple mode, use top-k by hit rate
            if not hosts_to_mutate:
                hit_rates = (self.host_hit_counts + 1) / (self.host_probe_counts + 2)
                hit_rates[~probed_mask] = -1
                k = min(self.mutation_capacity, n_probed)
                top_k_indices = np.argsort(hit_rates)[-k:]
                hosts_to_mutate = set(top_k_indices)
        else:
            # Fallback
            hit_rates = (self.host_hit_counts + 1) / (self.host_probe_counts + 2)
            hit_rates[~probed_mask] = -1
            k = min(self.mutation_capacity, n_probed)
            top_k_indices = np.argsort(hit_rates)[-k:]
            hosts_to_mutate = set(top_k_indices)
        
        print(f"[IDHAMNeural]   Mutating {len(hosts_to_mutate)} hosts (action {action_idx})")
        
        # Perform mutation
        self._perform_mutation(hosts_to_mutate)
        
        print(f"[IDHAMNeural]   Mutation complete: {len(self.mutated_hosts)} total mutated hosts")
    
    def _train_networks(self):
        """
        Train actor and critic networks using collected experience.
        
        Implements the advantage actor-critic algorithm from the paper.
        """
        if len(self.rewards) < 2:
            return
        
        # Safety check: ensure all buffers have the same length
        buffer_lengths = {
            'states': len(self.states),
            'actions': len(self.actions),
            'log_probs': len(self.log_probs),
            'rewards': len(self.rewards),
            'values': len(self.values)
        }
        
        if len(set(buffer_lengths.values())) > 1:
            print(f"[IDHAMNeural] WARNING: Buffer length mismatch: {buffer_lengths}")
            print(f"[IDHAMNeural] Skipping training, clearing buffers")
            # Clear mismatched buffers
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.values = []
            return
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states))
        rewards_tensor = torch.FloatTensor(self.rewards)
        log_probs_tensor = torch.stack(self.log_probs)
        values_tensor = torch.FloatTensor(self.values)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        
        R = 0
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.gamma * R
            returns.insert(0, R)
        
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalize returns
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Calculate advantages: A(s,a) = R - V(s)
        advantages_tensor = returns_tensor - values_tensor
        
        # Update actor (policy network)
        # Loss: -A(s,a) * log π(a|s) - β * H(π)
        action_probs = self.actor.get_action_probs(states_tensor)
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()
        
        actor_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()
        actor_loss = actor_loss - self.beta * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update critic (value network)
        # Loss: (R - V(s))^2
        current_values = self.critic(states_tensor).squeeze()
        critic_loss = F.mse_loss(current_values, returns_tensor)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        print(f"[IDHAMNeural] Training: actor_loss={actor_loss.item():.4f}, "
              f"critic_loss={critic_loss.item():.4f}, entropy={entropy.item():.4f}")
    
    def get_policy_entropy(self) -> float:
        """Calculate policy entropy from actor network."""
        if len(self.states) == 0:
            return 0.0
        
        state = torch.FloatTensor(self.states[-1]).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor.get_action_probs(state)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        
        return float(entropy)
    
    def save_models(self, path: str):
        """Save actor and critic networks."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        print(f"[IDHAMNeural] Models saved to {path}")
    
    def load_models(self, path: str):
        """Load actor and critic networks."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"[IDHAMNeural] Models loaded from {path}")


# Keep the other defenders (FRVM, RHM, Static) with fixed mutation logic
class FRVMDefender(MutationDefenderBase):
    """FRVM defender - inherits fixed _perform_mutation"""
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        super().__init__(n_hosts, config, seed)
        defender_cfg = config.get('defender', {})
        self.mutation_interval = defender_cfg.get('mutation_interval', 50)
        initial_mutation_count = int(self.mutation_capacity * 0.5)
        initial_hosts = self.rng.choice(n_hosts, initial_mutation_count, replace=False)
        self._perform_mutation(set(initial_hosts))
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        if episode % self.mutation_interval == 0:
            n_to_mutate = min(self.mutation_capacity, self.n_hosts)
            hosts_to_mutate = set(self.rng.choice(self.n_hosts, size=n_to_mutate, replace=False))
            self._perform_mutation(hosts_to_mutate)
    
    def get_policy_entropy(self) -> float:
        return float(np.log(self.mutation_capacity + 1))


class RHMDefender(MutationDefenderBase):
    """RHM defender - inherits fixed _perform_mutation"""
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        super().__init__(n_hosts, config, seed)
        defender_cfg = config.get('defender', {})
        self.mutation_interval = defender_cfg.get('mutation_interval', 50)
        self.attack_threshold = defender_cfg.get('attack_threshold', 2.0)
        self.host_probe_counts = np.zeros(n_hosts)
        self.host_hit_counts = np.zeros(n_hosts)
        initial_mutation_count = int(self.mutation_capacity * 0.5)
        initial_hosts = self.rng.choice(n_hosts, initial_mutation_count, replace=False)
        self._perform_mutation(set(initial_hosts))
    
    def resolve_probe(self, virtual_ip: int, probe_idx: int) -> Tuple[int, bool, float]:
        physical_host, is_mutated, qos_cost = super().resolve_probe(virtual_ip, probe_idx)
        self.host_probe_counts[physical_host] += 1
        return physical_host, is_mutated, qos_cost
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        for host in discovered_hosts:
            self.host_hit_counts[host] += 1
        
        if episode % self.mutation_interval == 0:
            mean_probes = np.mean(self.host_probe_counts)
            std_probes = np.std(self.host_probe_counts)
            threshold = mean_probes + self.attack_threshold * std_probes
            under_attack = np.where(self.host_probe_counts > threshold)[0]
            
            if len(under_attack) == 0:
                n_to_select = min(self.mutation_capacity, self.n_hosts)
                under_attack = np.argsort(self.host_probe_counts)[-n_to_select:]
            
            if len(under_attack) > self.mutation_capacity:
                sorted_by_probes = sorted(under_attack, 
                                         key=lambda h: self.host_probe_counts[h], 
                                         reverse=True)
                hosts_to_mutate = set(sorted_by_probes[:self.mutation_capacity])
            else:
                hosts_to_mutate = set(under_attack)
            
            if hosts_to_mutate:
                self._perform_mutation(hosts_to_mutate)


class StaticMutationDefender(MutationDefenderBase):
    """Static defender - inherits fixed _perform_mutation"""
    
    def __init__(self, n_hosts: int, config: Dict, seed: int = 0):
        super().__init__(n_hosts, config, seed)
        defender_cfg = config.get('defender', {})
        initial_strategy = defender_cfg.get('initial_mutation_strategy', 'random')
        
        if initial_strategy == 'random':
            initial_mutation_count = int(self.mutation_capacity * 0.5)
            initial_hosts = self.rng.choice(n_hosts, initial_mutation_count, replace=False)
            self._perform_mutation(set(initial_hosts))
    
    def update(self, hits_count: int, discovered_hosts: Set[int], episode: int):
        pass
    
    def get_policy_entropy(self) -> float:
        return 0.0


# Backward compatibility: keep old name
IDHAMMutationDefender = IDHAMNeuralDefender