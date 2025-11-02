"""
Evaluate SMT Constraint Solving Performance
Reproduces Figure 4 from the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from smt_constraints import AddressBlockConstraints

def evaluate_smt_performance():
    """
    Evaluate SMT solving time for different network sizes
    Reproduces Figure 4: SMT solving time vs number of hosts
    """
    
    print("\n" + "="*70)
    print("SMT PERFORMANCE EVALUATION")
    print("Reproducing Figure 4 from the paper")
    print("="*70)
    
    # Test configurations
    host_counts = [10, 15, 20, 25, 30]
    block_counts = [30, 40, 50]
    max_solutions = 500
    
    results = {}
    
    for num_blocks in block_counts:
        print(f"\nTesting with {num_blocks} IP address blocks:")
        print("-" * 50)
        
        solving_times = []
        actual_hosts = []
        
        for num_hosts in host_counts:
            print(f"  {num_hosts} hosts, {num_blocks} blocks...", end=" ", flush=True)
            
            try:
                # Setup
                num_switches = max(5, num_hosts // 6)
                hosts_per_switch = {}
                hosts_per_sw = num_hosts // num_switches
                
                for sw in range(num_switches):
                    start_host = sw * hosts_per_sw
                    if sw == num_switches - 1:
                        end_host = num_hosts
                    else:
                        end_host = start_host + hosts_per_sw
                    hosts_per_switch[sw] = list(range(start_host, end_host))
                
                # Create constraint solver
                constraints = AddressBlockConstraints(
                    num_hosts=num_hosts,
                    num_blocks=num_blocks,
                    block_size=128,
                    num_switches=num_switches,
                    hosts_per_switch=hosts_per_switch
                )
                
                # Add constraints
                mutation_periods = [np.random.randint(10, 16) for _ in range(num_hosts)]
                T_AS = 64
                
                constraints.add_mutation_rate_constraint(mutation_periods, T_AS, omega=0.25)
                constraints.add_forbidden_block_constraint(set(range(5)))
                constraints.add_flow_table_size_constraint(theta=1)
                constraints.add_basic_constraints()
                
                # Measure solving time
                start_time = time.time()
                solutions = constraints.solve(max_solutions=max_solutions, timeout=120)
                elapsed_time = time.time() - start_time
                
                solving_times.append(elapsed_time)
                actual_hosts.append(num_hosts)
                
                print(f"✓ {elapsed_time:.2f}s ({len(solutions)} solutions)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                solving_times.append(None)
        
        results[num_blocks] = {
            'hosts': actual_hosts,
            'times': solving_times
        }
    
    # Plot results (reproduce Figure 4)
    print("\n" + "="*70)
    print("Generating plot...")
    
    plt.figure(figsize=(10, 6))
    
    markers = ['o-', 's-', '^-']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, num_blocks in enumerate(block_counts):
        data = results[num_blocks]
        hosts = data['hosts']
        times = [t for t in data['times'] if t is not None]
        
        if times:
            plt.plot(hosts[:len(times)], times, markers[i], 
                    label=f'M={num_blocks}', linewidth=2, markersize=8,
                    color=colors[i])
    
    plt.xlabel('Number of hosts', fontsize=12)
    plt.ylabel('SMT solving time (s)', fontsize=12)
    plt.title('SMT Solving Time vs Network Size\n(max_solutions=500)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = 'results/smt_solving_time.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved plot: {filename}")
    plt.close()
    
    # Print summary table (reproduce Table II concept)
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Hosts':<10} {'Blocks':<10} {'Time (s)':<15} {'Notes'}")
    print("-" * 70)
    
    for num_blocks in block_counts:
        data = results[num_blocks]
        for host, time_val in zip(data['hosts'], data['times']):
            if time_val is not None:
                # Calculate action space dimensions
                total_dims = host * num_blocks
                print(f"{host:<10} {num_blocks:<10} {time_val:<15.2f} "
                      f"{total_dims} dimensions")
    
    print("\n" + "="*70)
    print("COMPARISON WITH PAPER (Table II):")
    print("-" * 70)
    print("Paper values (from Table II):")
    print("  - L(m=30): 1.1×10³¹, 8.9×10³⁷, 7.5×10⁴⁸, 2.2×10⁵⁷")
    print("  - L(m=40): 3.1×10⁴⁰, 1.3×10⁴⁹, 3.5×10⁶⁴, 5.8×10⁷⁶")
    print("  - L(m=50): 3.7×10⁴⁹, 4.3×10⁶⁰, 1.2×10⁷⁹, 1.7×10⁹⁴")
    print("\nYour results should show:")
    print("  ✓ Solving time increases with number of hosts")
    print("  ✓ Solving time increases with number of blocks")
    print("  ✓ Feasible actions << 2^(n×m) total action space")
    print("="*70)


def count_feasible_actions():
    """
    Count feasible actions for different network sizes
    Reproduces data for Table II
    """
    
    print("\n" + "="*70)
    print("COUNTING FEASIBLE ACTIONS")
    print("Reproducing Table II data")
    print("="*70)
    
    configurations = [
        (10, 30), (15, 30), (20, 30), (25, 30),
        (10, 40), (15, 40), (20, 40), (25, 40),
        (10, 50), (15, 50), (20, 50), (25, 50),
    ]
    
    print(f"\n{'Hosts':<10} {'Blocks':<10} {'Total Space':<20} {'Feasible':<20} {'Ratio'}")
    print("-" * 80)
    
    for num_hosts, num_blocks in configurations:
        # Total action space: 2^(n×m)
        total_space_exp = num_hosts * num_blocks
        
        # Try to count feasible actions (this may take time)
        print(f"{num_hosts:<10} {num_blocks:<10} 2^{total_space_exp:<17} ", end="", flush=True)
        
        try:
            from smt_constraints import generate_feasible_actions
            
            # Count with short timeout
            solutions = generate_feasible_actions(
                num_hosts=num_hosts,
                num_blocks=num_blocks,
                num_switches=5,
                max_solutions=100,  # Limited for speed
                T_AS=64
            )
            
            feasible = len(solutions)
            ratio = f"~1/{2**(total_space_exp) / feasible:.2e}" if feasible > 0 else "N/A"
            
            print(f"{feasible:<20} {ratio}")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    # Evaluate SMT performance
    evaluate_smt_performance()
    
    # Optionally count feasible actions (can be slow)
    # count_feasible_actions()
