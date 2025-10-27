#!/usr/bin/env python3
"""
Generate NDV experiment configurations with MUTATION defenders.

This replaces masking semantics with true address mutation.
"""

import yaml
from pathlib import Path
from itertools import product


def get_matched_seed_group_id(p, B, mu, family):
    """Generate consistent seed group ID for matched strata."""
    return f"p{p}_b{B}_mu{mu}_{family}"


def generate_ndv_mutation_configs(p=2, B=200, mu=100, seeds=20):
    """
    Generate NDV grid with MUTATION-based defenders.
    
    Defender types:
    - idham_mutation: Learning-based mutation (ID-HAM with true mutation)
    - frvm: Random mutation baseline (Sharma et al. 2018)
    - rhm: Heuristic mutation baseline (Jafarian et al. 2015)
    - static_mutation: Static baseline (no adaptation)
    
    Args:
        p: Number of partitions
        B: Probe budget
        mu: Mean switch interval
        seeds: Number of seeds per config
    """
    configs = []
    n_hosts = 100
    
    # NDV families to test
    ndv_families = [
        {'name': 'deterministic', 'mode': 'periodic', 'jitter_pct': 0.0},
        {'name': 'uniform_j10', 'mode': 'periodic', 'jitter_pct': 0.10},
        {'name': 'uniform_j30', 'mode': 'periodic', 'jitter_pct': 0.30},
        {'name': 'poisson', 'mode': 'poisson', 'jitter_pct': 0.0},
    ]
    
    # MUTATION Defender types
    defender_types = [
        {
            'type': 'idham_mutation',
            'mutation_capacity': int(n_hosts * 0.3),
            'mutation_interval': 50,  # Episodes between mutations
        },
        {
            'type': 'frvm',
            'mutation_capacity': int(n_hosts * 0.3),
            'mutation_interval': 50,  # T_AS period
            'mutation_mode': 'uniform',
        },
        {
            'type': 'rhm',
            'mutation_capacity': int(n_hosts * 0.3),
            'mutation_interval': 50,
            'attack_threshold': 2.0,  # Standard deviations for attack detection
        },
        {
            'type': 'static_mutation',
            'mutation_capacity': int(n_hosts * 0.3),
            'initial_mutation_strategy': 'random',
        },
    ]
    
    # Generate configs for each combination
    for family in ndv_families:
        seed_group_id = get_matched_seed_group_id(p, B, mu, family['name'])
        
        for defender_cfg in defender_types:
            # Build config ID
            defender_name = defender_cfg['type']
            jitter_str = f"j{int(family['jitter_pct']*100)}" if family['jitter_pct'] > 0 else "j0"
            config_id = f"{defender_name}-divconq_{p}p_{family['name']}-{mu}mu_{jitter_str}_{B}b"
            
            config = {
                'config_id': config_id,
                'n_hosts': n_hosts,
                'partitioning': {
                    'mode': 'uniform',
                    'partitions': p
                },
                'attacker': {
                    'strategy': 'divide_and_conquer',
                    'switch': {
                        'mode': family['mode'],
                        'mean_interval': mu,
                        'jitter_pct': family['jitter_pct']
                    }
                },
                'defender': defender_cfg,
                'probe_budget': B,
                'episodes': {
                    'train': 5000,
                    'eval_window': 200
                },
                'seeds': seeds,
                'seed_strategy': 'matched',
                'matched_seed_group_id': seed_group_id,
                'metrics': {
                    'window': 200,
                    'adaptation_epsilon': 0.10
                },
                'logging': {
                    'per_episode_csv': True,
                    'aggregate_json': True
                }
            }
            
            configs.append(config)
    
    return configs


def generate_ndv_full_grid(seeds=20):
    """
    Generate full NDV grid across multiple strata with MUTATION defenders.
    """
    configs = []
    n_hosts = 100
    
    # Parameter ranges
    partitions_list = [2, 4, 8]
    budgets = [100, 200, 500]
    intervals = [100, 500]
    
    # NDV families
    ndv_families = [
        {'name': 'deterministic', 'mode': 'periodic', 'jitter_pct': 0.0},
        {'name': 'uniform_j10', 'mode': 'periodic', 'jitter_pct': 0.10},
        {'name': 'uniform_j30', 'mode': 'periodic', 'jitter_pct': 0.30},
        {'name': 'poisson', 'mode': 'poisson', 'jitter_pct': 0.0},
    ]
    
    # MUTATION Defender types
    defender_types = [
        {
            'type': 'idham_mutation',
            'mutation_capacity': int(n_hosts * 0.3),
            'mutation_interval': 50,
        },
        {
            'type': 'frvm',
            'mutation_capacity': int(n_hosts * 0.3),
            'mutation_interval': 50,
            'mutation_mode': 'uniform',
        },
        {
            'type': 'rhm',
            'mutation_capacity': int(n_hosts * 0.3),
            'mutation_interval': 50,
            'attack_threshold': 2.0,
        },
        {
            'type': 'static_mutation',
            'mutation_capacity': int(n_hosts * 0.3),
            'initial_mutation_strategy': 'random',
        },
    ]
    
    for p, B, mu in product(partitions_list, budgets, intervals):
        for family in ndv_families:
            seed_group_id = get_matched_seed_group_id(p, B, mu, family['name'])
            
            for defender_cfg in defender_types:
                defender_name = defender_cfg['type']
                jitter_str = f"j{int(family['jitter_pct']*100)}" if family['jitter_pct'] > 0 else "j0"
                config_id = f"{defender_name}-divconq_{p}p_{family['name']}-{mu}mu_{jitter_str}_{B}b"
                
                config = {
                    'config_id': config_id,
                    'n_hosts': n_hosts,
                    'partitioning': {
                        'mode': 'uniform',
                        'partitions': p
                    },
                    'attacker': {
                        'strategy': 'divide_and_conquer',
                        'switch': {
                            'mode': family['mode'],
                            'mean_interval': mu,
                            'jitter_pct': family['jitter_pct']
                        }
                    },
                    'defender': defender_cfg,
                    'probe_budget': B,
                    'episodes': {
                        'train': 5000,
                        'eval_window': 200
                    },
                    'seeds': seeds,
                    'seed_strategy': 'matched',
                    'matched_seed_group_id': seed_group_id,
                    'metrics': {
                        'window': 200,
                        'adaptation_epsilon': 0.10
                    },
                    'logging': {
                        'per_episode_csv': True,
                        'aggregate_json': True
                    }
                }
                
                configs.append(config)
    
    return configs


def save_configs(configs: list, output_dir: Path):
    """Save all configs to YAML files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for config in configs:
        filename = f"{config['config_id']}.yaml"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated {len(configs)} configuration files in {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate NDV mutation experiment configurations')
    parser.add_argument('--preset', choices=['minimal', 'full'], default='minimal',
                       help='Configuration preset to generate')
    parser.add_argument('--p', type=int, default=2,
                       help='Number of partitions (for minimal preset)')
    parser.add_argument('--B', type=int, default=200,
                       help='Probe budget (for minimal preset)')
    parser.add_argument('--mu', type=int, default=100,
                       help='Mean switch interval (for minimal preset)')
    parser.add_argument('--seeds', type=int, default=20,
                       help='Number of seeds per configuration')
    parser.add_argument('--output-dir', type=str, default='configs/ndv_mutation',
                       help='Output directory for configs')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.preset == 'minimal':
        print(f"Generating minimal NDV mutation grid: p={args.p}, B={args.B}, μ={args.mu}")
        configs = generate_ndv_mutation_configs(
            p=args.p, B=args.B, mu=args.mu, seeds=args.seeds
        )
        subdir = output_dir / f"p{args.p}_b{args.B}_mu{args.mu}"
    else:
        print("Generating full NDV mutation grid")
        configs = generate_ndv_full_grid(seeds=args.seeds)
        subdir = output_dir / "full_grid"
    
    save_configs(configs, subdir)
    
    # Summary
    print(f"\nSummary:")
    print(f"  Total configs: {len(configs)}")
    print(f"  Seeds per config: {args.seeds}")
    print(f"  Total experiments: {len(configs) * args.seeds}")
    
    # Count by defender type
    by_defender = {}
    for cfg in configs:
        def_type = cfg['defender']['type']
        by_defender[def_type] = by_defender.get(def_type, 0) + 1
    
    print(f"\nBy defender type:")
    for def_type, count in sorted(by_defender.items()):
        print(f"  {def_type}: {count} configs")
    
    print(f"\nDefender types included:")
    print(f"  idham_mutation: Learning-based mutation (ID-HAM)")
    print(f"  frvm: Random mutation baseline (Sharma et al. 2018)")
    print(f"  rhm: Heuristic mutation baseline (Jafarian et al. 2015)")
    print(f"  static_mutation: Static baseline (no learning)")
    
    print(f"\nTo run experiments:")
    print(f"  python run_experiment_mutation.py --config {subdir}/<config>.yaml")
    print(f"  # Or parallel:")
    print(f"  ls {subdir}/*.yaml | parallel -j 8 python run_experiment_mutation.py --config {{}}")


if __name__ == '__main__':
    main()