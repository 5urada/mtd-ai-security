"""
Configuration Parameters for ID-HAM
All values from Table I and throughout the paper
"""

# Network Configuration (Section III-B, VII)
NETWORK_CONFIG = {
    'small': {
        'num_hosts': 30,
        'num_switches': 5,
        'num_blocks': 50,
        'description': 'Small network scenario (Figures 5-6)'
    },
    'large': {
        'num_hosts': 100,
        'num_switches': 30,
        'num_blocks': 150,
        'description': 'Large network scenario (Figures 7-8)'
    }
}

# Topology Parameters (Section VII, Table I)
TOPOLOGY_PARAMS = {
    'model': 'Waxman',
    'alpha': 0.2,  # Waxman parameter
    'beta': 0.15,  # Waxman parameter
    'bandwidth': '10 Mbps',  # Link bandwidth
}

# Address Space Parameters (Section III-B)
ADDRESS_SPACE = {
    'block_size': 128,  # Z in paper (size of IP address block)
    'subnet': '10.0.0.0/16',  # Example subnet
    'forbidden_blocks': 5,  # First 5 blocks reserved
}

# Mutation Parameters (Section III-B)
MUTATION_CONFIG = {
    'T_AS': 64,  # Address Shuffling interval (seconds)
    'T_RM_min': 10,  # Minimum Random mutation interval (seconds)
    'T_RM_max': 15,  # Maximum Random mutation interval (seconds)
    'sliding_window_N': 3,  # N previous periods for seamless mutation
}

# MDP Parameters (Section III-C)
MDP_PARAMS = {
    'gamma': 0.99,  # Discount factor (γ)
    'alpha': 1.0,  # Reward coefficient (α in Equation 2)
    'reward_constant': 10.0,  # Positive constant C in Equation 2
    'P_s': 0.3,  # Probability: static → moving (after scan)
    'P_m': 0.1,  # Probability: moving → static
}

# SMT Constraint Parameters (Section V)
SMT_PARAMS = {
    'omega': 0.01,  # Maximum repetition probability (ω in Eq. 6)
    'theta': 2,  # Minimum adjacent block pairs (Θ in Eq. 11)
    'max_solutions': 500,  # Number of feasible actions to generate
    'timeout': 300,  # Timeout in seconds
}

# Deep Reinforcement Learning Parameters (Section VI, Table I)
DRL_PARAMS = {
    'learning_rate_actor': 0.001,  # α_a
    'learning_rate_critic': 0.001,  # α_c
    'beta': 0.01,  # Entropy regularization coefficient (β)
    'gamma': 0.99,  # Discount factor (same as MDP)
    
    # Network Architecture
    'actor_hidden_layers': [256, 256],  # From Table I
    'critic_hidden_layers': [256, 256],  # From Table I
    'activation': 'ReLU',  # Activation function
    'optimizer': 'RMSProp',  # From paper
}

# Training Parameters (Section VII, Table I)
TRAINING_PARAMS = {
    'num_epochs': 3000,  # U in Algorithm 2 (full experiment)
    'steps_per_epoch': 10,  # T in Algorithm 2  
    'num_epochs_quick': 500,  # For quick testing
    'steps_per_epoch_quick': 5,  # For quick testing
}

# Scanning Parameters (Section III-A, Table I)
SCANNING_PARAMS = {
    'scanning_rate': 16,  # η hosts per ΔT (increased from 16 for realistic TSH)
    'address_space_multiplier': 20,  # Address space = num_hosts * 20 (more realistic)
    
    # Local Preference Scanning
    'local_preference': {
        'locality_prob': 0.7,  # Probability to scan near previous hits
        'locality_range': 256,  # Range around previous hits
    },
    
    # Sequential Scanning
    'sequential': {
        'reselect_period': 1000,  # Reselect start address every N scans
    },
    
    # Divide-Conquer Scanning
    'divide_conquer': {
        'num_initial_attackers': 3,  # Initial compromised hosts
        'infection_prob': 0.1,  # ρ: probability to infect scanned host
    },
    
    # Dynamic Scanning
    'dynamic': {
        'strategy_switch_period': 500,  # Switch strategy every N scans
    }
}

# Comparison Methods (Section VII-B)
BASELINE_METHODS = {
    'ID-HAM': {
        'description': 'Proposed method with DRL',
        'learns': True,
        'uses_hypothesis_test': False,
    },
    'RHM': {
        'description': 'Random Host Mutation with hypothesis test [16]',
        'learns': True,
        'uses_hypothesis_test': True,
    },
    'FRVM': {
        'description': 'Flexible Random Virtual IP Multiplexing [13]',
        'learns': False,
        'uses_hypothesis_test': False,
    }
}

# Expected Results (From paper Section VII-B)
EXPECTED_RESULTS = {
    'small_network': {
        'local_preference': {
            'ID-HAM': 18.6,
            'RHM': 19.7,
            'FRVM': 18.6,
        },
        'sequential': {
            'ID-HAM': 9.0,
            'RHM': 10.8,
            'FRVM': 12.0,
        },
        'divide_conquer': {
            'ID-HAM': 14.0,
            'RHM': 16.0,
            'FRVM': 17.5,
        },
        'dynamic': {
            'ID-HAM': 18.0,
            'RHM': 19.7,
            'FRVM': 19.3,
        }
    },
    'large_network': {
        'local_preference': {
            'ID-HAM': 21.5,
            'RHM': 23.0,
            'FRVM': 24.5,
        },
        'sequential': {
            'ID-HAM': 18.0,
            'RHM': 20.0,
            'FRVM': 21.5,
        },
        'divide_conquer': {
            'ID-HAM': 24.0,
            'RHM': 25.5,
            'FRVM': 27.5,
        },
        'dynamic': {
            'ID-HAM': 21.8,
            'RHM': 24.0,
            'FRVM': 25.0,
        }
    }
}

# SMT Performance (Section VII-A, Table II)
SMT_EXPECTED_RESULTS = {
    'solving_times': {
        # Format: (hosts, blocks): time_seconds
        (10, 30): 5,
        (15, 30): 15,
        (20, 30): 45,
        (25, 30): 95,
        (30, 30): 180,
    },
    'feasible_actions': {
        # Format: (hosts, blocks): approximate_count
        # Note: These are very large numbers, shown as exponents
        (10, 30): '1.1e31',
        (15, 30): '8.9e37',
        (20, 30): '7.5e48',
        (25, 30): '2.2e57',
    }
}

# Hardware Configuration (Section VII, prototype system)
PROTOTYPE_HARDWARE = {
    'sdn_controller': {
        'cpu': 'Intel Core i9-10920X @ 3.5GHz',
        'gpu': 'NVIDIA GeForce RTX 5000',
        'ram': '64GB',
    },
    'switches': {
        'model': 'H3C S5560X-30C-EI',
        'count': 4,
    },
    'hosts': {
        'model': 'Raspberry Pi',
        'count': 13,
    }
}

# Validation Tolerances
VALIDATION_TOLERANCES = {
    'tsh_absolute': 0.20,  # ±20% on absolute TSH values
    'tsh_improvement': 0.10,  # ±10% on improvement percentages
    'solving_time': 0.30,  # ±30% on SMT solving times
    'convergence_epoch': 500,  # ±500 epochs for convergence
}

# Reproducibility Seeds (not in paper, but useful)
RANDOM_SEEDS = {
    'numpy': 42,
    'tensorflow': 42,
    'python': 42,
}

# Output Configuration
OUTPUT_CONFIG = {
    'results_dir': 'results',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'plot_format': 'png',
    'plot_dpi': 300,
    'save_frequency': 100,  # Save every N epochs
}

# Plotting Configuration
PLOT_CONFIG = {
    'figure_size': (14, 10),
    'font_size': 11,
    'title_size': 12,
    'line_width': 2,
    'marker_size': 8,
    'grid_alpha': 0.3,
    'smoothing_window': 50,  # Moving average window
    'colors': {
        'ID-HAM': '#1f77b4',  # Blue
        'RHM': '#ff7f0e',     # Orange
        'FRVM': '#2ca02c',    # Green
    }
}


def get_config(config_name='small'):
    """
    Get configuration for specific experiment
    
    Args:
        config_name: 'small', 'large', or 'quick'
    
    Returns:
        Dictionary with all relevant parameters
    """
    if config_name == 'quick':
        config = NETWORK_CONFIG['small'].copy()
        config.update({
            'num_epochs': TRAINING_PARAMS['num_epochs_quick'],
            'steps_per_epoch': TRAINING_PARAMS['steps_per_epoch_quick'],
        })
    else:
        config = NETWORK_CONFIG[config_name].copy()
        config.update({
            'num_epochs': TRAINING_PARAMS['num_epochs'],
            'steps_per_epoch': TRAINING_PARAMS['steps_per_epoch'],
        })
    
    config.update({
        'topology': TOPOLOGY_PARAMS,
        'address_space': ADDRESS_SPACE,
        'mutation': MUTATION_CONFIG,
        'mdp': MDP_PARAMS,
        'smt': SMT_PARAMS,
        'drl': DRL_PARAMS,
        'scanning': SCANNING_PARAMS,
    })
    
    return config


def print_config(config_name='small'):
    """Print configuration in readable format"""
    config = get_config(config_name)
    
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name.upper()}")
    print(f"{'='*60}")
    
    for section, params in config.items():
        if isinstance(params, dict):
            print(f"\n{section.upper()}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {params}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Print all configurations
    print_config('small')
    print_config('large')
    print_config('quick')
    
    # Show expected results
    print("\nEXPECTED RESULTS (from paper):")
    print("="*60)
    for network, strategies in EXPECTED_RESULTS.items():
        print(f"\n{network.upper().replace('_', ' ')}:")
        for strategy, methods in strategies.items():
            print(f"  {strategy}:")
            for method, tsh in methods.items():
                print(f"    {method}: {tsh}")