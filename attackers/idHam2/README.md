# ID-HAM Experiment Framework

Experimental framework for evaluating adaptive network defense against divide-and-conquer reconnaissance attackers.

## Quick Setup

### Prerequisites

- Python 3.8+
- pip

### Installation
```bash
# Clone repository
git clone <repository-url>
cd idham_experiment_framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

### 1. Generate Configurations
```bash
# Generate all 433 experiment configs
python generate_configs.py

# Verify
ls configs/*.yaml | wc -l  # Should show 433
```

### 2. Test Single Experiment
```bash
# Run baseline
python run_experiment.py --config configs/baseline_static_p4_b200.yaml --seed 0

# Run dynamic attacker
python run_experiment.py --config configs/divconq_p4_unif_int500_j10_peri_b200.yaml --seed 0
```

### 3. Run Full Sweep

**Local (Sequential - ~72 hours):**
```bash
nohup ./run_all_experiments.sh > experiment.log 2>&1 &
```

**Local (Parallel - ~18 hours):**
```bash
# Edit run_all_experiments.sh: set MAX_PARALLEL=4
nohup ./run_all_experiments.sh > experiment.log 2>&1 &
```

**HPC with SLURM (~2-4 hours):**
```bash
sbatch submit_job.sh
```

### 4. Monitor Progress
```bash
# Check completed configs
ls output/results/*/summary.json | wc -l

# Watch log
tail -f experiment.log
```

## HPC Setup (SLURM)

### Setup on HPC
```bash
# Transfer framework
scp -r idham_experiment_framework/ username@hpc.university.edu:~/

# On HPC
ssh username@hpc.university.edu
cd idham_experiment_framework

# Setup environment
module load python/3.9
python -m venv idham_env
source idham_env/bin/activate
pip install -r requirements.txt

# Test
python run_experiment.py --config configs/baseline_static_p4_b200.yaml --seed 0
```

### Submit Job Array

Create `submit_job.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=idham
#SBATCH --output=idham_%A_%a.out
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --array=0-432

module load python/3.9
source ~/idham_experiment_framework/idham_env/bin/activate

CONFIG_FILES=(configs/*.yaml)
CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

python run_experiment.py --config "$CONFIG"
```

Submit:
```bash
sbatch submit_job.sh
squeue -u $USER
```

## Project Structure
```
├── src/                    # Source code
│   ├── attacker.py        # Divide-and-conquer attacker
│   ├── defender.py        # ID-HAM defender
│   ├── environment.py     # Network simulation
│   ├── metrics.py         # Metrics computation
│   └── utils.py           # Utilities
├── configs/               # 433 YAML configs (generated)
├── run_experiment.py      # Main runner
├── generate_configs.py    # Config generator
├── analyze_results.py     # Results analysis
├── run_all_experiments.sh # Batch runner
└── requirements.txt       # Dependencies
```

## Output
```
output/
├── logs/
│   └── <config_id>/
│       └── seed_*.csv     # Per-episode metrics
└── results/
    └── <config_id>/
        └── summary.json   # Aggregated stats
```

## Troubleshooting

**Experiments not starting?**
```bash
ps aux | grep python
```

**Check progress:**
```bash
ls output/logs/*/seed_*.csv | wc -l
```

**Kill running experiment:**
```bash
kill $(cat experiment.pid)
```
