import numpy as np
from config import Config
from constraints import generate_feasible_assignments

config = Config(seed=42)

# Create a test state
S = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 
              1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
A_prev = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
                   0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])

# Generate for original state using base seed from A_prev only
base_seed = abs(hash(tuple(A_prev.tolist()))) % (2**31)
feasible_orig = generate_feasible_assignments(config, A_prev, S, config.K, 
                                               use_ortools=False, seed=base_seed)

print(f"Original state (S[0]={S[0]}): {len(feasible_orig)} candidates")
if len(feasible_orig) > 0:
    print(f"First candidate: {feasible_orig[0]}")
    print(f"Second candidate: {feasible_orig[1] if len(feasible_orig) > 1 else 'N/A'}")

# Flip one bit and regenerate with SAME base seed
S_flipped = S.copy()
S_flipped[0] = 1 - S_flipped[0]

# Use SAME base seed - let constraints naturally determine differences
feasible_flip = generate_feasible_assignments(config, A_prev, S_flipped, config.K,
                                               use_ortools=False, seed=base_seed)

print(f"\nFlipped state (S[0]={S_flipped[0]}): {len(feasible_flip)} candidates")
if len(feasible_flip) > 0:
    print(f"First candidate: {feasible_flip[0]}")
    print(f"Second candidate: {feasible_flip[1] if len(feasible_flip) > 1 else 'N/A'}")

# Check overlap
orig_set = {tuple(a.tolist()) for a in feasible_orig}
flip_set = {tuple(a.tolist()) for a in feasible_flip}
common = orig_set & flip_set

print(f"\n{'='*60}")
print(f"Common candidates: {len(common)} out of {config.K}")
print(f"Overlap percentage: {100 * len(common) / config.K:.1f}%")
print(f"Support changed: {len(common) == 0}")
print(f"Support overlap rate: {len(common) / config.K:.2f}")
print(f"{'='*60}")

# Show which candidates are common
if len(common) > 0:
    print(f"\nCommon assignments (first 3):")
    for i, assignment in enumerate(list(common)[:3]):
        print(f"  {i+1}. {assignment}")
else:
    print("\nNo common assignments found!")

# Show unique to each
orig_only = orig_set - flip_set
flip_only = flip_set - orig_set
print(f"\nUnique to original: {len(orig_only)}")
print(f"Unique to flipped: {len(flip_only)}")