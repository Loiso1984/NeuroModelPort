"""check_param_count.py - Verify args count matches RHS parameters"""

from core.rhs import rhs_multicompartment
import inspect

# Try to get signature via source code inspection
import core.rhs as rhs_module

# Read the source to count parameters manually
with open(r'c:\NeuroModelPort\core\rhs.py', 'r') as f:
    content = f.read()

# Extract RHS function definition
import re
pattern = r'def rhs_multicompartment\((.*?)\):'
match = re.search(pattern, content, re.DOTALL)

if match:
    params_text = match.group(1)
    # Split by comma but be careful with newlines
    params_text = params_text.replace('\n', ' ').replace('    ', ' ')
    
    # Count actual parameters (lines with parameter names)
    lines = content.split('def rhs_multicompartment(\n')[1].split('\n):')[0].split('\n')
    params = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Extract parameter name
            param_name = line.split('=')[0].split(',')[0].strip()
            if param_name:
                params.append(param_name)
    
    print("RHS Parameters (in order):")
    for i, param in enumerate(params, 1):
        print(f"  {i:2d}. {param}")
    
    print(f"\nTotal RHS params: {len(params)} (excluding t, y, n_comp)")

# Now let's count args in solver.py
with open(r'c:\NeuroModelPort\core\solver.py', 'r') as f:
    solver_content = f.read()

# Find the args = ( ... ) tuple
pattern = r'args = \((.*)\)'
match = re.search(pattern, solver_content, re.DOTALL)

if match:
    args_text = match.group(1)
    # Count items
    items = [item.strip() for item in args_text.split(',') if item.strip()]
    
    print("\nArgs items (in order):")
    for i, item in enumerate(items, 1):
        print(f"  {i:2d}. {item}")
    
    print(f"\nTotal args items: {len(items)}")
