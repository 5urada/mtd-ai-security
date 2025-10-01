from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    """Configuration for ID-HAM simulator with sane defaults."""
    
    # Network topology
    N: int = 24  # Number of hosts
    B: int = 6   # Number of blocks
    block_size: int = 128  # Size of each IP address block
    
    # Constraint parameters
    mu_min: float = 0.10  # Minimum mutation rate
    mu_max: float = 0.40  # Maximum mutation rate
    delta: int = 2  # Adjacency window for block assignments
    forbidden_blocks: List[int] = field(default_factory=list)  # Forbidden block IDs
    
    # Capacity: uniform distribution
    @property
    def capacities(self) -> List[int]:
        import math
        base_cap = math.ceil(self.N / self.B) + 2
        return [base_cap] * self.B
    
    # Attacker parameters
    p_static: float = 0.25   # Scan hit prob for static hosts
    p_moving: float = 0.10   # Scan hit prob for moving hosts
    p_focus: float = 0.55    # Focused scan hit prob
    dynamic_period: int = 8  # Steps before attacker changes regime
    
    # Episode parameters
    T: int = 64  # Episode length
    K: int = 16  # Number of candidate assignments per step
    
    # Reward parameters
    alpha: float = 1.0  # Weight for scan hits
    lambda_: float = 0.1  # Weight for churn cost
    
    # Training parameters
    total_steps: int = 100_000
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    seed: int = 1337
    
    # Paths
    runs_dir: str = "runs/exp1"
    reports_dir: str = "reports"
