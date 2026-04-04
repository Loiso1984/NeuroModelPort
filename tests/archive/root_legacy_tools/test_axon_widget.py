#!/usr/bin/env python3
"""Test axon_biophysics widget rendering."""
import sys
import os
# Disable GUI by setting Qt to offscreen platform
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from gui.axon_biophysics import AxonBiophysicsWidget
from PySide6.QtWidgets import QApplication

# Create QApplication (required for PySide6)
app = QApplication(sys.argv)

# Create a simple simulation result
cfg = FullModelConfig()
apply_preset(cfg, "Cerebellar Purkinje (De Schutter)")  # Multi-compartment by default

try:
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    print(f"✓ Simulation completed")
    print(f"  n_comp: {result.n_comp}")
    print(f"  currents keys: {list(result.currents.keys())}")
    print(f"  morph keys: {list(result.morph.keys())}")
    
    # Try to create and use the widget
    widget = AxonBiophysicsWidget()
    print(f"✓ Widget created")
    
    # Try to plot data
    widget.plot_axon_data(result, cfg)
    print(f"✓ Widget plotting successful")
    
except Exception as e:
    import traceback
    print(f"✗ ERROR: {e}")
    traceback.print_exc()
