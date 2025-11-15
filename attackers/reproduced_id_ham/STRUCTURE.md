# ID-HAM Testbed Documentation for Scanning Strategy Development

## Context

I have a working implementation of the ID-HAM (Intelligence-Driven Host Address Mutation) defense system from the paper "How to Disturb Network Reconnaissance: A Moving Target Defense Approach Based on Deep Reinforcement Learning" (Zhang et al., IEEE TIFS 2023).

This testbed is designed to evaluate adversarial scanning strategies against a moving target defense (MTD) that periodically mutates IP addresses to disrupt network reconnaissance.

## Testbed Architecture

### Network Configuration

**Small Network (Default):**
- 30 hosts distributed across 5 OpenFlow switches
- 50 IP address blocks, each containing 128 addresses
- Total address space: 6,400 addresses
- Active host addresses: 30 hosts × 25 addresses each = 750 addresses
- Address space utilization: ~11.7%

**Large Network (Optional):**
- 100 hosts distributed across 30 OpenFlow switches  
- 150 IP address blocks
- Total address space: 19,200 addresses

### Defense Mechanism (ID-HAM)

**Address Mutation:**
- Hosts are classified as "moving" (mutate IPs) or "static" (fixed IPs)
- Mutation occurs at two levels:
  - **Address Block Shuffling (AS):** Every T_AS = 64 seconds, the defender reassigns which IP blocks are allocated to which hosts
  - **Random vIP Mutation (RM):** Every T_RM = 10-15 seconds, moving hosts select new virtual IP addresses from their assigned blocks

**Defense Learning:**
- Uses Deep Reinforcement Learning (Actor-Critic) to learn optimal block assignments
- Tracks which address blocks are being scanned (block_scan_history)
- Rewards actions that avoid heavily-scanned blocks
- Penalizes actions that use frequently-targeted blocks

**State Representation:**
```python
state = [
    host_types (30 values),           # 1=moving, 0=static
    block_scan_history (50 values)    # Scanning heat per block
]
```

### Scanning Environment

**Adversary Capabilities:**
- Scanning rate: η = 16 addresses per scan period
- No prior knowledge of which addresses are active
- Must probe the 6,400-address space to find 750 active addresses
- Can choose different scanning strategies

**Temporal Structure:**
- Basic time unit: ΔT (one scan period)
- Steps per epoch: 10 scan periods
- Episodes: Multiple epochs of scanning attempts

**Network Visibility:**
- Adversary sees: {address → host_id} mapping only for successfully scanned hosts
- Adversary doesn't know:
  - Which hosts are moving vs static
  - When mutations occur
  - Future address allocations
  - Total number of hosts

## Performance Metric: TSH (Times of Scanning Hits)

### Definition

**TSH = Average number of successful scans per step**

Where:
- A "scanning hit" occurs when the adversary scans an address that is currently active (mapped to a host)
- A "step" is one scan period (16 address probes)

### Calculation Method

```python
# Per step:
scan_results = scanner.scan(active_hosts)  # Returns {host_id: num_hits}
tsh_this_step = sum(scan_results.values())  # Total hits this step

# Per epoch (10 steps):
epoch_tsh = sum(tsh_per_step for all 10 steps)
avg_tsh_per_step = epoch_tsh / 10

# Reported value:
TSH = avg_tsh_per_step  # Average hits per step
```

### Example Calculation

```
Scan period 1: Probe 16 addresses → 2 hits (found 2 active hosts)
Scan period 2: Probe 16 addresses → 1 hit
Scan period 3: Probe 16 addresses → 3 hits
... (7 more periods)
Scan period 10: Probe 16 addresses → 1 hit

Epoch total: 17 hits across 10 steps
TSH = 17 / 10 = 1.7 hits per step

# Note: Paper reports "per epoch" values
Paper's TSH = 1.7 × 10 = 17 hits per epoch
```

### Interpretation

**Lower TSH = More effective defense**
- TSH = 1.0 per step → Defense is working well (only 10 hits per epoch)
- TSH = 2.0 per step → Defense is moderate (20 hits per epoch)
- TSH = 3.0 per step → Defense is weak (30 hits per epoch)

**Baseline expectations (without defense):**
- Random scanning: ~11.7% hit rate → 16 × 0.117 = 1.87 hits per step
- With ID-HAM learning: Should decrease to 0.9-1.7 hits per step

**Paper's reported values (per epoch, scaled ×10):**
- Sequential scanning: ID-HAM achieves 9.0 hits/epoch (0.9 per step)
- Divide-Conquer scanning: ID-HAM achieves 14.0 hits/epoch (1.4 per step)
- Local Preference: ID-HAM achieves 18.6 hits/epoch (1.86 per step)

## Existing Scanning Strategies (Implemented)

### 1. Local Preference Scanning

```python
class LocalPreferenceScanning:
    def scan(self, active_hosts):
        # With 70% probability: scan near previous successful hits
        # With 30% probability: scan randomly
        # Locality range: ±256 addresses around previous hits
```

**Characteristics:**
- Exploits spatial locality of hosts
- Remembers last 100 successful scans
- Effective when hosts cluster in address space

**My testbed results:**
- Initial TSH: 1.92 per step (19.2 per epoch)
- After 500 epochs: 2.28 per step (22.8 per epoch)
- Paper's expected: 1.86 per step (18.6 per epoch)

### 2. Sequential Scanning

```python
class SequentialScanning:
    def scan(self, active_hosts):
        # Scan addresses sequentially: addr, addr+1, addr+2, ...
        # Every 160 scans: randomly reselect starting address
```

**Characteristics:**
- Most predictable pattern
- Easiest for defense to learn
- Comprehensive coverage over time

**My testbed results:**
- Initial TSH: 1.96 per step (19.6 per epoch)
- After 500 epochs: 1.70 per step (17.0 per epoch)
- Paper's expected: 0.90 per step (9.0 per epoch)
- **Note:** Needs more epochs to reach paper's value

### 3. Divide-Conquer Scanning (Current Baseline)

```python
class DivideConquerScanning:
    def __init__(self, num_initial_attackers=3, infection_prob=0.1):
        # Start with 3 compromised hosts
        # Each scanned host has 10% chance of becoming compromised
        # Each compromised host runs sequential scanning
        
    def scan(self, active_hosts):
        # Distribute 16 total scans among all compromised hosts
        # If N compromised hosts: each gets 16/N scans per period
        # Successfully scanned hosts may join as new scanners
```

**Characteristics:**
- Epidemic/worm-like behavior
- Number of attackers grows over time
- Most challenging for defense (paper shows highest TSH)

**My testbed results:**
- Initial TSH: 1.49 per step (14.9 per epoch)
- After 500 epochs: 1.86 per step (18.6 per epoch)
- Paper's expected: 1.40 per step (14.0 per epoch)

**Key implementation details:**
```python
# Critical constraint: Total scanning rate stays constant at 16
if len(compromised_hosts) <= 16:
    scans_per_host = 16 // len(compromised_hosts)
else:
    # Only activate 16 hosts at once, each scans 1 address
    active_scanners = compromised_hosts[:16]
```

### 4. Dynamic Scanning

```python
class DynamicScanning:
    def scan(self, active_hosts):
        # Every 500 scans: randomly switch between:
        #   - Local Preference
        #   - Sequential  
        #   - Divide-Conquer
```

**Characteristics:**
- Unpredictable strategy switching
- Hardest to characterize
- Combines strengths of all strategies

**My testbed results:**
- Initial TSH: 1.92 per step (19.2 per epoch)
- After 500 epochs: 1.91 per step (19.1 per epoch)
- Paper's expected: 1.80 per step (18.0 per epoch)

## Testbed API for New Scanning Strategies

### Base Class Interface

```python
class ScanningStrategy:
    def __init__(self, address_space_size: int, scanning_rate: int):
        """
        Args:
            address_space_size: Total addresses (default: 6400)
            scanning_rate: Addresses to scan per period (default: 16)
        """
        self.address_space_size = address_space_size
        self.scanning_rate = scanning_rate
        self.scanned_addresses = set()  # All addresses ever scanned
        self.current_period_scans = []  # Addresses scanned this period
    
    def scan(self, active_hosts: Dict[int, int]) -> Dict[int, int]:
        """
        Perform one scan period (probe 'scanning_rate' addresses)
        
        Args:
            active_hosts: Dict mapping {address: host_id} for currently active hosts
        
        Returns:
            Dict mapping {host_id: num_successful_scans}
        """
        raise NotImplementedError
```

### Example: Implementing a New Strategy

```python
class MyNewScanning(ScanningStrategy):
    def __init__(self, address_space_size, scanning_rate):
        super().__init__(address_space_size, scanning_rate)
        # Your initialization here
        
    def scan(self, active_hosts):
        scan_results = {}
        self.current_period_scans = []
        
        for _ in range(self.scanning_rate):
            # Your logic to select target_addr
            target_addr = self.select_target()
            
            # Record the scan
            self.scanned_addresses.add(target_addr)
            self.current_period_scans.append(target_addr)
            
            # Check if hit
            if target_addr in active_hosts:
                host_id = active_hosts[target_addr]
                scan_results[host_id] = scan_results.get(host_id, 0) + 1
        
        return scan_results
    
    def select_target(self):
        # Your strategy logic here
        return target_address
```

### Integration into Testbed

```python
# In run_experiments_improved.py
from my_scanning_strategy import MyNewScanning

strategies = {
    'divide_conquer': DivideConquerScanning(address_space, scanning_rate),
    'my_strategy': MyNewScanning(address_space, scanning_rate),  # Add here
}

# Testbed will automatically:
# 1. Train ID-HAM against your strategy for 500-3000 epochs
# 2. Calculate TSH per step and per epoch
# 3. Generate learning curves
# 4. Compare with baselines (RHM, FRVM)
```

## Evaluation Framework

### Running a Test

```python
from run_experiments_improved import ExperimentRunner

runner = ExperimentRunner(results_dir="my_results")

results = runner.run_defense_performance_experiment(
    num_hosts=30,
    num_blocks=50,
    num_switches=5,
    num_epochs=500,      # Quick test: 500, Full: 3000
    steps_per_epoch=10
)

# Results structure:
# {
#   'my_strategy': {
#       'ID-HAM': [tsh_epoch_0, tsh_epoch_1, ..., tsh_epoch_499],
#       'RHM': [tsh_epoch_0, ...],
#       'FRVM': [tsh_epoch_0, ...]
#   }
# }
```

### Output Metrics

**Per-strategy results:**
- TSH over time (learning curve)
- Final average TSH (last 100 epochs)
- Improvement percentage vs initial
- Comparison: ID-HAM vs RHM vs FRVM

**Diagnostic outputs:**
- Reward variance (should be >1.0 for learning)
- Block scan history (shows which blocks are targeted)
- Actor/critic losses (convergence indicators)

### Success Criteria

**Your strategy is effective against ID-HAM if:**
1. Final TSH > Divide-Conquer's TSH (currently 1.86 per step)
2. TSH doesn't decrease significantly over epochs (defense can't learn)
3. ID-HAM performs worse than random (FRVM)

**Your strategy is ineffective if:**
1. Final TSH < Sequential's TSH (currently 1.70 per step)
2. TSH decreases >20% over epochs (defense learns pattern)
3. ID-HAM performs better than baselines

## Research Questions for New Strategies

### Goals

1. **Beat Divide-Conquer:** Can you achieve higher TSH than 1.86 per step?
2. **Resist Learning:** Can you prevent ID-HAM from improving over time?
3. **Realistic Attack:** Is your strategy plausible for real adversaries?

### Constraints

- **Total scanning budget:** 16 addresses per period (realistic rate)
- **No omniscience:** Can't see defender's state or future mutations
- **Ethical boundaries:** Strategy should be defensible as security research

### Suggested Directions

1. **Adaptive scanning:** Learn defender's patterns (like ID-HAM learns yours)
2. **Deception-aware:** Detect honeypots or decoy addresses
3. **Timing-based:** Exploit mutation periods (if you can detect them)
4. **Coordinated:** Multiple attackers with shared intelligence
5. **Hybrid:** Combine multiple strategies intelligently

## Getting Started Prompt Template

Use this template to ask for help developing a new strategy:

```
I have an ID-HAM testbed for evaluating scanning strategies against moving target defense.

TESTBED PARAMETERS:
- Address space: 6,400 addresses
- Active addresses: 750 (30 hosts × 25 addresses each)
- Scanning rate: 16 addresses per period
- Defense: DRL-based address mutation every 10-64 seconds

PERFORMANCE METRIC:
- TSH = Average scanning hits per step (16-probe period)
- Lower TSH = Better defense
- Current baseline (Divide-Conquer): 1.86 TSH per step

MY GOAL:
[Describe what you want: beat divide-conquer, resist learning, etc.]

STRATEGY IDEA:
[Describe your approach]

QUESTIONS:
1. How would I implement this in the ScanningStrategy interface?
2. What TSH would you expect this to achieve?
3. How can I make it resist ID-HAM's learning?
```

## Additional Resources

**Code files:**
- `scanning_strategies_improved.py` - Base classes and existing strategies
- `run_experiments_improved.py` - Evaluation framework
- `mdp_model_improved.py` - Defense implementation
- `test_learning_prerequisites.py` - Validation tests

**Current testbed status:**
- ✅ Reward differentiation working (variance >1.0)
- ✅ Block history tracking working
- ✅ Learning demonstrated (Sequential: 13% improvement)
- ⚠️ Needs 3000 epochs for full convergence (currently tested at 500)

**Known limitations:**
- Simplified network model (no packet loss, latency)
- Perfect intrusion detection (all scans detected immediately)
- No adversary countermeasures to detection
- Fixed mutation periods (not adaptive)