import numpy as np
from typing import Tuple, List, Dict, Optional
from enum import Enum
from config import Config

class AttackerType(Enum):
    STATIC = "static"
    SEQUENTIAL = "sequential"
    DYNAMIC = "dynamic"

class IDHAMEnv:
    """ID-HAM MDP environment simulator."""
    
    def __init__(self, config: Config, attacker_type: AttackerType = AttackerType.STATIC):
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
        
        if self.attacker_type == AttackerType.STATIC:
            # Each host scanned independently
            for i in range(self.config.N):
                p = self.config.p_moving if self.S[i] == 1 else self.config.p_static
                hits[i] = self.rng.binomial(1, p)
        
        elif self.attacker_type == AttackerType.SEQUENTIAL:
            # Focus on current block
            for i in range(self.config.N):
                if A[i] == self.current_focus_block:
                    p = self.config.p_focus
                else:
                    p = self.config.p_moving if self.S[i] == 1 else self.config.p_static
                hits[i] = self.rng.binomial(1, p)
            
            # Move to next block
            self.current_focus_block = (self.current_focus_block + 1) % self.config.B
        
        elif self.attacker_type == AttackerType.DYNAMIC:
            # Mix of behaviors with periodic changes
            for i in range(self.config.N):
                if A[i] == self.current_focus_block:
                    p = self.config.p_focus
                else:
                    p = self.config.p_moving if self.S[i] == 1 else self.config.p_static
                hits[i] = self.rng.binomial(1, p)
        
        return hits
    
    @property
    def observation_dim(self) -> int:
        """Dimension of observation space."""
        return self.config.N + self.config.N * self.config.B + self.config.N
