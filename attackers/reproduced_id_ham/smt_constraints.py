"""
SMT-based Constraint Solver for Address Block Allocation
Based on Section V of the paper
"""

from z3 import *
import numpy as np
from typing import List, Tuple, Set
import time

class AddressBlockConstraints:
    """
    Formalize address block allocation as constrained satisfaction problem
    Using Z3 SMT Solver
    """
    
    def __init__(self, 
                 num_hosts: int,
                 num_blocks: int,
                 block_size: int,
                 num_switches: int,
                 hosts_per_switch: dict):
        """
        Initialize constraint solver
        
        Args:
            num_hosts: Number of hosts (n in paper)
            num_blocks: Number of IP address blocks (m in paper)
            block_size: Size of each block (Z in paper)
            num_switches: Number of OF-switches (z in paper)
            hosts_per_switch: Dict mapping switch_id to list of host_ids
        """
        self.num_hosts = num_hosts
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_switches = num_switches
        self.hosts_per_switch = hosts_per_switch
        
        # Create Z3 solver
        self.solver = Solver()
        
        # Decision variables: b[i][k] = 1 if block k assigned to host i
        self.b = [[Bool(f'b_{i}_{k}') for k in range(num_blocks)] 
                  for i in range(num_hosts)]
        
    def add_mutation_rate_constraint(self, 
                                     mutation_periods: List[int],
                                     T_AS: int,
                                     omega: float = 0.01):
        """
        Add mutation rate constraint (Equation 6)
        
        Ensures average repetition probability is below threshold
        
        Args:
            mutation_periods: T_RM for each host (mutation intervals)
            T_AS: Address shuffling interval
            omega: Maximum repetition probability threshold
        """
        for i in range(self.num_hosts):
            # Calculate N_i: number of mutations in T_AS
            N_i = (T_AS + mutation_periods[i] - 1) // mutation_periods[i]
            
            # Minimum blocks needed: ⌈(N_i - 1) / (2 * Z * ω)⌉
            min_blocks = int(np.ceil((N_i - 1) / (2 * self.block_size * omega)))
            
            # Sum of assigned blocks >= min_blocks
            self.solver.add(
                Sum([If(self.b[i][k], 1, 0) for k in range(self.num_blocks)]) >= min_blocks
            )
    
    def add_forbidden_block_constraint(self, forbidden_blocks: Set[int]):
        """
        Add forbidden block constraint (Equations 7-8)
        
        Some blocks cannot be used due to operational constraints
        
        Args:
            forbidden_blocks: Set of forbidden block indices
        """
        # Each non-forbidden block assigned to exactly one host (Eq. 7)
        for k in range(self.num_blocks):
            if k not in forbidden_blocks:
                self.solver.add(
                    Sum([If(self.b[i][k], 1, 0) for i in range(self.num_hosts)]) == 1
                )
        
        # Forbidden blocks not assigned to any host (Eq. 8)
        for k in forbidden_blocks:
            for i in range(self.num_hosts):
                self.solver.add(Not(self.b[i][k]))
    
    def add_flow_table_size_constraint(self, theta: int = 2):
        """
        Add flow table size constraint (Equations 9-11)
        
        Encourage supernetting by assigning adjacent blocks to same switch
        
        Args:
            theta: Minimum number of adjacent block pairs per switch
        """
        # For each switch, count adjacent block pairs
        for switch_id in range(self.num_switches):
            if switch_id not in self.hosts_per_switch:
                continue
            
            hosts_in_switch = self.hosts_per_switch[switch_id]
            
            # W[k][i]: block k assigned to at least one host in switch i
            # Simplified: count adjacent blocks assigned to this switch
            adjacent_pairs = []
            
            for k1 in range(self.num_blocks - 1):
                k2 = k1 + 1  # Adjacent block
                
                # Check if both blocks assigned to same switch
                block1_in_switch = Or([self.b[h][k1] for h in hosts_in_switch])
                block2_in_switch = Or([self.b[h][k2] for h in hosts_in_switch])
                
                # Both blocks in this switch = adjacent pair
                adjacent_pairs.append(And(block1_in_switch, block2_in_switch))
            
            if adjacent_pairs:
                # Minimum theta adjacent pairs per switch
                self.solver.add(
                    Sum([If(pair, 1, 0) for pair in adjacent_pairs]) >= theta
                )
    
    def add_basic_constraints(self):
        """Add basic feasibility constraints"""
        # Each host gets at least one block
        for i in range(self.num_hosts):
            self.solver.add(
                Or([self.b[i][k] for k in range(self.num_blocks)])
            )
    
    def solve(self, max_solutions: int = 500, timeout: int = 300) -> List[np.ndarray]:
        """
        Solve constraint satisfaction problem
        
        Args:
            max_solutions: Maximum number of solutions to generate
            timeout: Timeout in seconds
        
        Returns:
            List of feasible allocation matrices
        """
        solutions = []
        self.solver.set("timeout", timeout * 1000)  # Z3 uses milliseconds
        
        start_time = time.time()
        
        print(f"Solving SMT constraints...")
        print(f"Variables: {self.num_hosts * self.num_blocks} ({self.num_hosts}x{self.num_blocks})")
        
        solution_count = 0
        while solution_count < max_solutions:
            if time.time() - start_time > timeout:
                print(f"Timeout reached after {solution_count} solutions")
                break
            
            result = self.solver.check()
            
            if result == sat:
                model = self.solver.model()
                
                # Extract allocation matrix
                allocation = np.zeros((self.num_hosts, self.num_blocks), dtype=int)
                for i in range(self.num_hosts):
                    for k in range(self.num_blocks):
                        if is_true(model[self.b[i][k]]):
                            allocation[i, k] = 1
                
                solutions.append(allocation)
                solution_count += 1
                
                if solution_count % 100 == 0:
                    print(f"Found {solution_count} solutions...")
                
                # Add constraint to find different solution
                blocking_clause = []
                for i in range(self.num_hosts):
                    for k in range(self.num_blocks):
                        if allocation[i, k] == 1:
                            blocking_clause.append(Not(self.b[i][k]))
                        else:
                            blocking_clause.append(self.b[i][k])
                
                self.solver.add(Or(blocking_clause))
            else:
                print(f"No more solutions found. Result: {result}")
                break
        
        elapsed = time.time() - start_time
        print(f"SMT solving completed in {elapsed:.2f}s")
        print(f"Total solutions found: {len(solutions)}")
        
        return solutions


def generate_feasible_actions(num_hosts: int,
                              num_blocks: int,
                              num_switches: int,
                              block_size: int = 128,
                              T_AS: int = 64,
                              max_solutions: int = 500) -> List[np.ndarray]:
    """
    Generate feasible address block allocations
    
    Args:
        num_hosts: Number of hosts
        num_blocks: Number of IP address blocks
        num_switches: Number of switches
        block_size: Size of each block
        T_AS: Address shuffling interval
        max_solutions: Maximum solutions to generate
    
    Returns:
        List of feasible allocation matrices
    """
    # Create host-to-switch mapping (distribute evenly)
    hosts_per_switch = {}
    hosts_per_sw = num_hosts // num_switches
    for sw in range(num_switches):
        start_host = sw * hosts_per_sw
        if sw == num_switches - 1:
            # Last switch gets remaining hosts
            end_host = num_hosts
        else:
            end_host = start_host + hosts_per_sw
        hosts_per_switch[sw] = list(range(start_host, end_host))
    
    # Create constraint solver
    constraints = AddressBlockConstraints(
        num_hosts=num_hosts,
        num_blocks=num_blocks,
        block_size=block_size,
        num_switches=num_switches,
        hosts_per_switch=hosts_per_switch
    )
    
    # Add constraints with relaxed parameters for better success rate
    # Mutation periods (example: 10-15 seconds)
    mutation_periods = [np.random.randint(10, 16) for _ in range(num_hosts)]
    constraints.add_mutation_rate_constraint(mutation_periods, T_AS, omega=0.25)  # Relaxed from 0.01
    
    # Forbidden blocks (example: reserve first 5 blocks)
    forbidden_blocks = set(range(5))
    constraints.add_forbidden_block_constraint(forbidden_blocks)
    
    # Flow table size constraint (relaxed)
    constraints.add_flow_table_size_constraint(theta=1)  # Relaxed from 2
    
    # Basic constraints
    constraints.add_basic_constraints()
    
    # Solve with shorter timeout
    solutions = constraints.solve(max_solutions=max_solutions, timeout=60)  # Reduced from 300
    
    return solutions


if __name__ == '__main__':
    print("Testing SMT Constraint Solver")
    print("=" * 50)
    
    # Small network scenario
    print("\nSmall network: 30 hosts, 50 blocks, 5 switches")
    solutions = generate_feasible_actions(
        num_hosts=30,
        num_blocks=50,
        num_switches=5,
        max_solutions=100
    )
    
    if solutions:
        print(f"\nSample allocation matrix:")
        print(solutions[0])
        print(f"\nBlocks per host: {solutions[0].sum(axis=1)}")