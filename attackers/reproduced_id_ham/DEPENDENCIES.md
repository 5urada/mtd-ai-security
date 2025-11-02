# ID-HAM Component Dependencies

## Module Dependency Graph

```
                         ┌─────────────────────┐
                         │   config.py         │
                         │  (Parameters from   │
                         │   Table I)          │
                         └──────────┬──────────┘
                                    │
                                    ↓
                    ┌───────────────────────────────┐
                    │                               │
         ┌──────────▼────────┐          ┌──────────▼────────┐
         │  mdp_model.py     │          │ smt_constraints.py│
         │  ----------------  │          │ ----------------- │
         │  • State Space    │          │  • Z3 Solver      │
         │  • Action Space   │          │  • Constraints    │
         │  • Rewards        │←────────→│  • Feasible       │
         │  • Transitions    │          │    Actions        │
         └──────────┬────────┘          └──────────┬────────┘
                    │                              │
                    │    ┌──────────────────┐      │
                    └────▶│ actor_critic.py │◀─────┘
                         │ --------------- │
                         │ • Actor Network │
                         │ • Critic Network│
                         │ • A2C Algorithm │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
         ┌──────────▼──────────┐    ┌──────────▼──────────┐
         │ scanning_strategies.py│    │ topology_generator.py│
         │ --------------------│    │ -------------------- │
         │ • Local Preference │    │ • Waxman Model       │
         │ • Sequential       │    │ • NetworkX           │
         │ • Divide-Conquer   │    │ • Mininet            │
         │ • Dynamic          │    │                      │
         └──────────┬──────────┘    └──────────┬──────────┘
                    │                           │
                    └───────────┬───────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  run_experiments.py    │
                    │  ---------------------  │
                    │  • Experiment Runner   │
                    │  • Evaluation Logic    │
                    │  • Results Generation  │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │      results/          │
                    │  • PNG plots           │
                    │  • JSON data           │
                    │  • Performance metrics │
                    └────────────────────────┘
```

## Data Flow

```
1. Configuration
   config.py
      ↓
   [Network parameters, DRL hyperparameters, Constraints]

2. Constraint Solving
   smt_constraints.py
      ↓
   [Feasible Actions: L << 2^(n×m)]

3. Environment Setup
   mdp_model.py + topology_generator.py
      ↓
   [MDP Environment with Network State]

4. Adversarial Scanning
   scanning_strategies.py
      ↓
   [Scan Results: host_id → num_hits]

5. Deep RL Training
   actor_critic.py
      ↓
   [Policy π(A|S), Value V(S)]

6. Evaluation Loop
   run_experiments.py
      ↓
   [TSH over Episodes for ID-HAM, RHM, FRVM]

7. Results
   results/
      ↓
   [Plots, JSON, Performance Metrics]
```

## Import Dependencies

### mdp_model.py
```python
import numpy as np
from typing import List, Tuple, Dict
import random
# No internal dependencies
```

### smt_constraints.py
```python
from z3 import *
import numpy as np
from typing import List, Tuple, Set
import time
# Uses: mdp_model (conceptually, for dimensions)
```

### actor_critic.py
```python
import tensorflow as tf
import numpy as np
from typing import List, Tuple
import random

from mdp_model import HAM_MDP  # ← Direct dependency
# Uses feasible actions from smt_constraints
```

### scanning_strategies.py
```python
import numpy as np
from typing import List, Dict, Set
import random
# No internal dependencies
```

### topology_generator.py
```python
import networkx as nx
import matplotlib.pyplot as plt
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
# No internal dependencies
```

### run_experiments.py
```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import json, os
from datetime import datetime

from mdp_model import HAM_MDP                    # ← Dependency 1
from smt_constraints import generate_feasible_actions  # ← Dependency 2
from actor_critic import IDHAMAgent              # ← Dependency 3
from scanning_strategies import (                # ← Dependency 4
    LocalPreferenceScanning,
    SequentialScanning,
    DivideConquerScanning,
    DynamicScanning,
    NetworkScanner
)
```

### evaluate_smt_performance.py
```python
import numpy as np
import matplotlib.pyplot as plt
import time

from smt_constraints import AddressBlockConstraints  # ← Dependency
```

## Execution Order

### Quick Test
```
1. run_experiments_quick.py
   ├─→ smt_constraints.py (generate actions)
   ├─→ mdp_model.py (create environment)
   ├─→ scanning_strategies.py (create scanners)
   ├─→ actor_critic.py (train ID-HAM)
   └─→ Generate plots and save results
```

### Full Experiments
```
1. run_experiments.py
   ├─→ Small Network (30h, 50b, 5sw)
   │   ├─→ Generate feasible actions (SMT)
   │   ├─→ Test 4 scanning strategies
   │   │   ├─→ Train ID-HAM
   │   │   ├─→ Evaluate RHM
   │   │   └─→ Evaluate FRVM
   │   └─→ Generate plots
   │
   └─→ Large Network (100h, 150b, 30sw)
       ├─→ Generate feasible actions (SMT)
       ├─→ Test 4 scanning strategies
       │   ├─→ Train ID-HAM
       │   ├─→ Evaluate RHM
       │   └─→ Evaluate FRVM
       └─→ Generate plots
```

### SMT Evaluation
```
1. evaluate_smt_performance.py
   ├─→ Test different network sizes
   │   ├─→ 10, 15, 20, 25, 30 hosts
   │   └─→ 30, 40, 50 blocks
   ├─→ Measure solving time
   └─→ Generate Figure 4
```

## Testing Individual Components

```bash
# Test MDP model
python mdp_model.py
# → Creates MDP, simulates steps, shows state transitions

# Test SMT constraints
python smt_constraints.py
# → Generates feasible actions for 30h, 50b network

# Test Actor-Critic
python actor_critic.py
# → Creates networks, trains for 20 epochs

# Test scanning strategies
python scanning_strategies.py
# → Tests all 4 strategies, shows hit patterns

# Test topology generator (requires sudo)
sudo python topology_generator.py
# → Creates Mininet network, shows topology
```

## Paper Section Mapping

```
Section III-A (Threat Model)
   └─→ scanning_strategies.py
       • LocalPreferenceScanning
       • SequentialScanning
       • DivideConquerScanning
       • DynamicScanning

Section III-B (Network Model)
   └─→ topology_generator.py
       • Waxman topology (α=0.2, β=0.15)
       • OF-switches and hosts
       • Address space management

Section III-C (MDP Model)
   └─→ mdp_model.py
       • State space: {S1, ..., SΓ}
       • Action space: {A1, ..., AL}
       • Reward function: Equation (2)
       • State transitions: Ps, Pm

Section V (Constraints)
   └─→ smt_constraints.py
       • Mutation rate: Equation (6)
       • Forbidden blocks: Equations (7-8)
       • Flow table size: Equations (9-11)
       • Z3 solver integration

Section VI (DRL Algorithm)
   └─→ actor_critic.py
       • Algorithm 2 implementation
       • Actor network: π(At|St; θa)
       • Critic network: V(St; θc)
       • Advantage function: Equation (16)
       • Loss functions: Equations (14, 18)

Section VII-A (SMT Performance)
   └─→ evaluate_smt_performance.py
       • Figure 4: Solving time
       • Table II: Feasible actions

Section VII-B (Defense Performance)
   └─→ run_experiments.py
       • Figures 5-8: TSH over episodes
       • ID-HAM vs RHM vs FRVM
       • All 4 scanning strategies
```

## Key Interfaces

### MDP → Actor-Critic
```python
# MDP provides:
state = mdp.get_state()  # Network state
next_state, reward, done = mdp.step(action, scan_results)

# Actor-Critic uses:
action_idx = agent.ac_network.select_action(state)
agent.ac_network.update(state, action_idx, reward, next_state)
```

### SMT → Actor-Critic
```python
# SMT provides:
feasible_actions = generate_feasible_actions(...)
# List of valid address block allocations

# Actor-Critic uses:
action = feasible_actions[action_idx]  # Select from feasible set
```

### Scanning → MDP
```python
# Scanner provides:
scan_results = network_scanner.perform_scan()
# Dict: {host_id: num_hits}

# MDP uses:
next_state, reward, done = mdp.step(action, scan_results)
# Updates state and calculates reward
```

## Critical Paths

### Training Path (Most Important)
```
config.py → smt_constraints.py → mdp_model.py → 
actor_critic.py → scanning_strategies.py → run_experiments.py
```

### Evaluation Path
```
config.py → evaluate_smt_performance.py
```

### Visualization Path
```
run_experiments.py → matplotlib → results/*.png
```

## Summary

- **Total Modules**: 7 core Python files
- **External Dependencies**: TensorFlow, Z3, NetworkX, NumPy, Matplotlib
- **Internal Dependencies**: Mostly independent, main coupling in run_experiments.py
- **Execution Time**: 
  - Quick test: 10-15 minutes
  - Full experiments: 6-10 hours
  - SMT evaluation: 30 minutes
