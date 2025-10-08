# ============================================================================
# FILE: env.py
# ============================================================================
import numpy as np
from typing import Tuple, List, Dict, Optional
from enum import Enum
from config import Config

class AttackerType(Enum):
    LOCAL_PREFERENCE = "local_preference"
    SEQUENTIAL = "sequential"
    DIVIDE_CONQUER = "divide_conquer"
    DYNAMIC = "dynamic"

class IDHAMEnv:
    """ID-HAM MDP environment simulator."""
    
    def __init__(self, config: Config, attacker_type: AttackerType = AttackerType.SEQUENTIAL):
        self.config = config
        self.attacker_type = attacker_type
        self.rng = np.random.RandomState(config.seed)
        
        # State variables
        self.S = None  # Binary vector: 1=moving, 0=static
        self.A_prev = None  # Previous assignment
        self.Z = None  # Recent scan hits per host
        self.step_count = 0
        self.episode_step = 0
        
        # Attacker state
        self.current_focus_block = 0
        self.sweep_direction = 1
        self.dynamic_step = 0
        self.attacker_infected = set()  # For divide-conquer
        self.previous_hit_locations = []  # For local preference
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.S = self.rng.binomial(1, 0.5, self.config.N)
        self.A_prev = self.rng.randint(0, self.config.B, self.config.N)
        self.Z = np.zeros(self.config.N, dtype=np.float32)
        self.step_count = 0
        self.episode_step = 0
        self.current_focus_block = 0
        self.sweep_direction = 1
        self.dynamic_step = 0
        self.attacker_infected = set()
        self.previous_hit_locations = []
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from state."""
        # One-hot encode previous assignment
        A_onehot = np.zeros((self.config.N, self.config.B), dtype=np.float32)
        A_onehot[np.arange(self.config.N), self.A_prev] = 1.0
        A_flat = A_onehot.flatten()
        
        # Normalize scan hits
        Z_norm = self.Z / (1.0 + self.Z.max()) if self.Z.max() > 0 else self.Z
        
        # Concatenate: [S, one-hot(A_prev), normalized_Z]
        obs = np.concatenate([
            self.S.astype(np.float32),
            A_flat,
            Z_norm
        ])
        return obs
    
    def step(self, action_idx: int, feasible_actions: List[np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment."""
        # Select action from feasible candidates
        if action_idx >= len(feasible_actions):
            action_idx = 0  # Fallback
        A_new = feasible_actions[action_idx]
        
        # Calculate churn
        churn = np.mean(A_new != self.A_prev)
        
        # Simulate attacker scanning
        scan_hits = self._simulate_scan(A_new)
        self.Z = scan_hits.astype(np.float32)
        
        # Calculate reward
        reward = -self.config.alpha * self.Z.sum() - self.config.lambda_ * churn
        
        # Update state
        self.A_prev = A_new.copy()
        self.episode_step += 1
        self.step_count += 1
        
        # Update attacker state for dynamic regime
        if self.attacker_type == AttackerType.DYNAMIC:
            self.dynamic_step += 1
            if self.dynamic_step >= self.config.dynamic_period:
                self.dynamic_step = 0
                self.current_focus_block = self.rng.randint(0, self.config.B)
                self.sweep_direction = self.rng.choice([-1, 1])
        
        done = self.episode_step >= self.config.T
        info = {"churn": churn, "scan_hits": self.Z.sum()}
        
        return self._get_observation(), reward, done, info
    
    def _simulate_scan(self, A: np.ndarray) -> np.ndarray:
        """Simulate attacker scanning behavior."""
        hits = np.zeros(self.config.N, dtype=np.int32)
        
        if self.attacker_type == AttackerType.LOCAL_PREFERENCE:
            # Scan uniformly, but prefer areas near previous successful hits
            for i in range(self.config.N):
                # Base probability
                base_p = self.config.p_moving if self.S[i] == 1 else self.config.p_static
                
                # Check if near any previous hit locations (within distance 2)
                boost = 0.0
                for prev_hit in self.previous_hit_locations[-10:]:  # Last 10 hits
                    distance = abs(i - prev_hit)
                    if distance <= 2:
                        boost += 0.15 * (1.0 - distance / 3.0)  # Closer = higher boost
                
                # Apply boost with cap
                final_p = min(0.95, base_p + boost)
                hit = self.rng.binomial(1, final_p)
                hits[i] = hit
                
                # Track successful hits
                if hit:
                    self.previous_hit_locations.append(i)
        
        elif self.attacker_type == AttackerType.SEQUENTIAL:
            # Focus on current block, sweep sequentially
            for i in range(self.config.N):
                if A[i] == self.current_focus_block:
                    p = self.config.p_focus
                else:
                    p = self.config.p_moving if self.S[i] == 1 else self.config.p_static
                hits[i] = self.rng.binomial(1, p)
            
            # Move to next block
            self.current_focus_block = (self.current_focus_block + 1) % self.config.B
        
        elif self.attacker_type == AttackerType.DIVIDE_CONQUER:
            # Multiple coordinated scanners with infection propagation
            n_scanners = 3
            
            # Initial scanners focus on different blocks
            for scanner_id in range(n_scanners):
                focus_block = (self.current_focus_block + scanner_id) % self.config.B
                
                for i in range(self.config.N):
                    if A[i] == focus_block:
                        p = self.config.p_focus
                        hit = self.rng.binomial(1, p)
                        
                        if hit:
                            hits[i] = 1
                            # Infected hosts can become new attackers with probability ρ
                            if self.rng.random() < 0.1:  # ρ = 0.1
                                self.attacker_infected.add(i)
            
            # Previously infected hosts also scan their neighbors
            for infected_host in list(self.attacker_infected):
                # Scan nearby hosts
                for i in range(max(0, infected_host - 2), min(self.config.N, infected_host + 3)):
                    if i != infected_host:
                        p = self.config.p_static * 0.7  # Moderate scan probability
                        hit = self.rng.binomial(1, p)
                        if hit:
                            hits[i] = max(hits[i], 1)  # Don't overwrite existing hits
            
            # Advance focus for next step
            self.current_focus_block = (self.current_focus_block + 1) % self.config.B
        
        elif self.attacker_type == AttackerType.DYNAMIC:
            # Mix of behaviors with periodic regime changes
            # Randomly alternate between local-preference and sequential patterns
            if self.dynamic_step < self.config.dynamic_period // 2:
                # First half: sequential-like behavior
                for i in range(self.config.N):
                    if A[i] == self.current_focus_block:
                        p = self.config.p_focus
                    else:
                        p = self.config.p_moving if self.S[i] == 1 else self.config.p_static
                    hits[i] = self.rng.binomial(1, p)
            else:
                # Second half: local-preference-like behavior
                for i in range(self.config.N):
                    base_p = self.config.p_moving if self.S[i] == 1 else self.config.p_static
                    
                    # Boost near previous hits
                    boost = 0.0
                    for prev_hit in self.previous_hit_locations[-5:]:
                        if abs(i - prev_hit) <= 2:
                            boost += 0.1
                    
                    final_p = min(0.95, base_p + boost)
                    hit = self.rng.binomial(1, final_p)
                    hits[i] = hit
                    
                    if hit:
                        self.previous_hit_locations.append(i)
        
        return hits
    
    @property
    def observation_dim(self) -> int:
        """Dimension of observation space."""
        return self.config.N + self.config.N * self.config.B + self.config.N