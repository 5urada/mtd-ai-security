#!/usr/bin/env python3
"""
Generate all experiment configuration files for the full sweep.
Creates YAML files for all combinations of parameters.
"""

import yaml
from pathlib import Path
from itertools import product


def generate_baseline_config():
    """Generate baseline-static configuration."""
    config = {
        'config_id': 'baseline_static_p4_b200',
        'n_hosts': 100,
        'partitioning': {
            'mode': 'uniform',
            'partitions': 4
        },
        'attacker': {
            'strategy': 'divide_and_conquer',
            'switch': {
                'mode': 'periodic',
                'mean_interval': 100000,  # Very long interval = effectively static
                'jitter_pct': 0.0
            }
        },
        'probe_budget': 200,
        'episodes': {
            'train': 5000,
            'eval_window': 200
        },
        'seeds': 10,
        'metrics': {
            'window': 200,
            'adaptation_epsilon': 0.10
        },
        'logging': {
            'per_episode_csv': True,
            'aggregate_json': True
        }
    }
    return config


def generate_experiment_configs():
    """Generate all experimental configurations for the full sweep."""
    configs = []
    
    # Parameter ranges
    partitions_list = [2, 4, 8, 16]
    partition_modes = ['uniform', 'skewed']
    switch_intervals = [100, 500, 2000]
    jitter_pcts = [0.0, 0.10, 0.30]
    switch_modes = ['periodic', 'poisson']
    probe_budgets = [100, 200, 500]
    
    n_hosts = 100
    n_seeds = 10
    
    # Generate all combinations
    for p, mode, interval, jitter, switch_mode, budget in product(
        partitions_list, partition_modes, switch_intervals,
        jitter_pcts, switch_modes, probe_budgets
    ):
        # Generate config ID
        mode_abbr = 'unif' if mode == 'uniform' else 'skew'
        jitter_str = f"j{int(jitter*100)}"
        switch_abbr = 'peri' if switch_mode == 'periodic' else 'pois'
        
        config_id = f"divconq_p{p}_{mode_abbr}_int{interval}_{jitter_str}_{switch_abbr}_b{budget}"
        
        config = {
            'config_id': config_id,
            'n_hosts': n_hosts,
            'partitioning': {
                'mode': mode,
                'partitions': p
            },
            'attacker': {
                'strategy': 'divide_and_conquer',
                'switch': {
                    'mode': switch_mode,
                    'mean_interval': interval,
                    'jitter_pct': jitter
                }
            },
            'probe_budget': budget,
            'episodes': {
                'train': 5000,
                'eval_window': 200
            },
            'seeds': n_seeds,
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


def generate_batch_run_script(configs: list, output_dir: Path):
    """Generate a shell script to run all experiments."""
    script_lines = [
        "#!/bin/bash",
        "# Batch run script for all experiments",
        "# Generated automatically",
        "",
        "set -e",  # Exit on error
        "",
        "OUTPUT_DIR=output",
        "CONFIG_DIR=configs",
        "",
        "echo 'Starting batch experiment run...'",
        "echo 'Total configs: " + str(len(configs)) + "'",
        ""
    ]
    
    for i, config in enumerate(configs, 1):
        config_file = f"{config['config_id']}.yaml"
        script_lines.extend([
            f"echo '--- Running experiment {i}/{len(configs)}: {config['config_id']} ---'",
            f"python run_experiment.py --config $CONFIG_DIR/{config_file} --output-dir $OUTPUT_DIR",
            ""
        ])
    
    script_lines.append("echo 'All experiments complete!'")
    
    script_path = output_dir.parent / "run_all_experiments.sh"
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"Generated batch run script: {script_path}")


def main():
    configs_dir = Path('configs')
    
    # Generate baseline config
    baseline = generate_baseline_config()
    save_configs([baseline], configs_dir)
    
    # Generate full experiment sweep
    exp_configs = generate_experiment_configs()
    save_configs(exp_configs, configs_dir)
    
    # Generate batch run script
    all_configs = [baseline] + exp_configs
    generate_batch_run_script(all_configs, configs_dir)
    
    print(f"\nSummary:")
    print(f"  Baseline configs: 1")
    print(f"  Experiment configs: {len(exp_configs)}")
    print(f"  Total configs: {len(all_configs)}")
    print(f"\nParameter space:")
    print(f"  Partitions: [2, 4, 8, 16]")
    print(f"  Modes: [uniform, skewed]")
    print(f"  Switch intervals: [100, 500, 2000]")
    print(f"  Jitter: [0%, 10%, 30%]")
    print(f"  Switch modes: [periodic, poisson]")
    print(f"  Probe budgets: [100, 200, 500]")
    print(f"\nTo run all experiments: ./run_all_experiments.sh")
    print(f"To run single config: python run_experiment.py --config configs/<config_name>.yaml")


if __name__ == '__main__':
    main()