"""
Utility functions for experiment framework.
"""

import random
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_directories(output_dir: Path, config_id: str):
    """Create necessary output directories."""
    dirs = [
        output_dir / 'logs' / config_id,
        output_dir / 'results' / config_id,
        output_dir / 'figs' / config_id,
        output_dir / 'models' / config_id
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def format_config_id(config: dict) -> str:
    """Generate a readable config ID from experiment parameters."""
    parts = []
    
    # Partitioning
    p = config['partitioning']
    parts.append(f"p{p['partitions']}")
    parts.append(p['mode'][:4])  # uniform -> unif, skewed -> skew
    
    # Switch config
    s = config['attacker']['switch']
    parts.append(f"int{s['mean_interval']}")
    
    jitter_str = f"j{int(s['jitter_pct']*100)}"
    parts.append(jitter_str)
    
    parts.append(s['mode'][:4])  # periodic -> peri, poisson -> pois
    
    # Budget
    parts.append(f"b{config['probe_budget']}")
    
    return "_".join(parts)