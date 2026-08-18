import os
import sys

# 1. Ensure src directory and __init__.py exist
os.makedirs("src", exist_ok=True)
with open("src/__init__.py", "w") as f:
    pass

# 2. Force current directory into Python path BEFORE imports
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

# 3. Create/overwrite src/quantum_sim.py
