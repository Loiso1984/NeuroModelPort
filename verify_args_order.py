"""verify_args_order.py - Verify RHS args parameter order matches"""

import inspect
from core.rhs import rhs_multicompartment

# Get RHS function signature
sig = inspect.signature(rhs_multicompartment.__wrapped__)  # __wrapped__ for Numba functions
params = list(sig.parameters.keys())

print("RHS Function Parameters (in order):")
for i, param in enumerate(params, 1):
    print(f"  {i:2d}. {param}")

print(f"\nTotal parameters: {len(params)}")
