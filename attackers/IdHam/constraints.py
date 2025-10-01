import numpy as np
from typing import List, Optional
from config import Config

def generate_feasible_assignments(
    config: Config,
    A_prev: np.ndarray,
    S: np.ndarray,
    K: int,
    use_ortools: bool = True,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Generate K feasible block assignments satisfying all constraints.
    Falls back to greedy repair if OR-Tools unavailable or fails.
    """
    if use_ortools:
        try:
            return _generate_with_ortools(config, A_prev, S, K, seed)
        except Exception as e:
            print(f"OR-Tools failed: {e}. Falling back to greedy repair.")
    
    return _generate_with_greedy(config, A_prev, S, K, seed)

def _generate_with_ortools(
    config: Config,
    A_prev: np.ndarray,
    S: np.ndarray,
    K: int,
    seed: Optional[int]
) -> List[np.ndarray]:
    """Generate assignments using OR-Tools CP-SAT."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        raise ImportError("OR-Tools not available")
    
    N, B = config.N, config.B
    rng = np.random.RandomState(seed)
    
    # Compute mutation constraints
    mu_min_count = int(np.ceil(config.mu_min * N))
    mu_max_count = int(np.floor(config.mu_max * N))
    
    results = []
    attempts = 0
    max_attempts = K * 10
    
    while len(results) < K and attempts < max_attempts:
        attempts += 1
        
        model = cp_model.CpModel()
        
        # Variables: b[i][k] = 1 if host i assigned to block k
        b = {}
        for i in range(N):
            for k in range(B):
                b[i, k] = model.NewBoolVar(f'b_{i}_{k}')
        
        # Each host assigned to exactly one block
        for i in range(N):
            model.Add(sum(b[i, k] for k in range(B)) == 1)
        
        # Block capacity constraints
        for k in range(B):
            if k in config.forbidden_blocks:
                for i in range(N):
                    model.Add(b[i, k] == 0)
            else:
                model.Add(sum(b[i, k] for i in range(N)) <= config.capacities[k])
        
        # Mutation rate constraint
        changed = [model.NewBoolVar(f'changed_{i}') for i in range(N)]
        for i in range(N):
            model.Add(b[i, int(A_prev[i])] == 0).OnlyEnforceIf(changed[i])
            model.Add(b[i, int(A_prev[i])] == 1).OnlyEnforceIf(changed[i].Not())
        
        total_changed = sum(changed)
        model.Add(total_changed >= mu_min_count)
        model.Add(total_changed <= mu_max_count)
        
        # Adjacency constraint
        for i in range(N):
            for k in range(B):
                if abs(k - A_prev[i]) > config.delta:
                    model.Add(b[i, k] == 0)
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = rng.randint(0, 2**31)
        solver.parameters.max_time_in_seconds = 1.0
        
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            A_new = np.zeros(N, dtype=np.int32)
            for i in range(N):
                for k in range(B):
                    if solver.Value(b[i, k]) == 1:
                        A_new[i] = k
                        break
            
            # Verify uniqueness
            if not any(np.array_equal(A_new, existing) for existing in results):
                results.append(A_new)
    
    return results if results else _generate_with_greedy(config, A_prev, S, K, seed)

def _generate_with_greedy(
    config: Config,
    A_prev: np.ndarray,
    S: np.ndarray,
    K: int,
    seed: Optional[int]
) -> List[np.ndarray]:
    """Greedy repair heuristic for generating feasible assignments."""
    rng = np.random.RandomState(seed)
    N, B = config.N, config.B
    
    mu_min_count = int(np.ceil(config.mu_min * N))
    mu_max_count = int(np.floor(config.mu_max * N))
    
    results = []
    
    for _ in range(K * 3):  # Try multiple times
        A_new = A_prev.copy()
        
        # Determine how many to mutate
        n_mutate = rng.randint(mu_min_count, mu_max_count + 1)
        
        # Select hosts to mutate
        hosts_to_mutate = rng.choice(N, size=n_mutate, replace=False)
        
        # Assign new blocks with constraints
        for i in hosts_to_mutate:
            # Valid blocks: within delta, not forbidden, capacity available
            valid_blocks = []
            for k in range(B):
                if k == A_prev[i]:
                    continue
                if k in config.forbidden_blocks:
                    continue
                if abs(k - A_prev[i]) > config.delta:
                    continue
                if np.sum(A_new == k) >= config.capacities[k]:
                    continue
                valid_blocks.append(k)
            
            if valid_blocks:
                A_new[i] = rng.choice(valid_blocks)
        
        # Verify constraints and add if unique
        if _verify_constraints(config, A_prev, A_new):
            if not any(np.array_equal(A_new, existing) for existing in results):
                results.append(A_new)
        
        if len(results) >= K:
            break
    
    # Ensure at least one solution (keep A_prev if needed)
    if not results:
        results.append(A_prev.copy())
    
    return results

def _verify_constraints(config: Config, A_prev: np.ndarray, A_new: np.ndarray) -> bool:
    """Verify all constraints are satisfied."""
    N, B = config.N, config.B
    
    # Mutation rate
    n_changed = np.sum(A_new != A_prev)
    mu_min_count = int(np.ceil(config.mu_min * N))
    mu_max_count = int(np.floor(config.mu_max * N))
    if not (mu_min_count <= n_changed <= mu_max_count):
        return False
    
    # Capacity
    for k in range(B):
        if np.sum(A_new == k) > config.capacities[k]:
            return False
    
    # Forbidden blocks
    for k in config.forbidden_blocks:
        if np.any(A_new == k):
            return False
    
    # Adjacency
    for i in range(N):
        if abs(A_new[i] - A_prev[i]) > config.delta:
            return False
    
    return True
