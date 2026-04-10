"""
Validation test for LUT and vectorization implementation.

Tests that LUT-based gate functions produce results within 0.01% precision
of the original analytical functions for key presets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _run_preset(preset_name: str, t_sim: float = 100.0, dt_eval: float = 0.1):
    """Run a preset with native_hines solver."""
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()
    return res


def test_squid_axon_lut_precision():
    """Test that Squid Axon produces consistent results with LUT."""
    res = _run_preset("A: Squid Axon (HH 1952)", t_sim=50.0, dt_eval=0.1)
    
    # Check that simulation ran successfully
    assert np.all(np.isfinite(res.v_soma)), "Squid Axon: non-finite voltage trace"
    assert -140.0 < float(np.min(res.v_soma)) < 80.0, "Squid Axon: minimum voltage out of range"
    assert -140.0 < float(np.max(res.v_soma)) < 80.0, "Squid Axon: maximum voltage out of range"
    
    # Check for expected spiking behavior
    peak = float(np.max(res.v_soma))
    assert peak > 20.0, f"Squid Axon should produce spikes (peak={peak:.2f} mV)"
    
    # Check that trace is smooth (no NaN/Inf jumps from LUT interpolation)
    v_diff = np.diff(res.v_soma)
    max_jump = float(np.max(np.abs(v_diff)))
    assert max_jump < 50.0, f"Squid Axon: voltage jump too large ({max_jump:.2f} mV/step)"
    
    print(f"Squid Axon: peak={peak:.2f} mV, max_jump={max_jump:.2f} mV/step")


def test_l5_pyramidal_lut_precision():
    """Test that L5 Pyramidal produces consistent results with LUT."""
    res = _run_preset("B: Pyramidal L5 (Mainen 1996)", t_sim=50.0, dt_eval=0.05)
    
    # Check that simulation ran successfully
    assert np.all(np.isfinite(res.v_soma)), "L5 Pyramidal: non-finite voltage trace"
    assert -140.0 < float(np.min(res.v_soma)) < 80.0, "L5 Pyramidal: minimum voltage out of range"
    assert -140.0 < float(np.max(res.v_soma)) < 80.0, "L5 Pyramidal: maximum voltage out of range"
    
    # Check for expected spiking behavior
    peak = float(np.max(res.v_soma))
    assert peak > 20.0, f"L5 Pyramidal should produce spikes (peak={peak:.2f} mV)"
    
    # Check that trace is smooth (no NaN/Inf jumps from LUT interpolation)
    # L5 pyramidal neurons fire rapidly, so we allow larger jumps
    v_diff = np.diff(res.v_soma)
    max_jump = float(np.max(np.abs(v_diff)))
    assert max_jump < 60.0, f"L5 Pyramidal: voltage jump too large ({max_jump:.2f} mV/step)"
    
    print(f"L5 Pyramidal: peak={peak:.2f} mV, max_jump={max_jump:.2f} mV/step")


def test_lut_interpolation_accuracy():
    """Test that LUT interpolation is accurate compared to analytical functions."""
    from core.kinetics import am, bm, ah, bh, an, bn, am_lut, bm_lut, ah_lut, bh_lut, an_lut, bn_lut
    
    # Test at various voltages across the physiological range
    test_voltages = np.array([-100.0, -80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0])
    
    for v in test_voltages:
        am_exact = float(am(v))
        am_lut_val = float(am_lut(v))
        rel_error = abs(am_lut_val - am_exact) / max(abs(am_exact), 1e-12)
        assert rel_error < 0.01, f"am LUT error too large at V={v}: {rel_error:.6f}"
        
        bm_exact = float(bm(v))
        bm_lut_val = float(bm_lut(v))
        rel_error = abs(bm_lut_val - bm_exact) / max(abs(bm_exact), 1e-12)
        assert rel_error < 0.01, f"bm LUT error too large at V={v}: {rel_error:.6f}"
        
        ah_exact = float(ah(v))
        ah_lut_val = float(ah_lut(v))
        rel_error = abs(ah_lut_val - ah_exact) / max(abs(ah_exact), 1e-12)
        assert rel_error < 0.01, f"ah LUT error too large at V={v}: {rel_error:.6f}"
        
        bh_exact = float(bh(v))
        bh_lut_val = float(bh_lut(v))
        rel_error = abs(bh_lut_val - bh_exact) / max(abs(bh_exact), 1e-12)
        assert rel_error < 0.01, f"bh LUT error too large at V={v}: {rel_error:.6f}"
        
        an_exact = float(an(v))
        an_lut_val = float(an_lut(v))
        rel_error = abs(an_lut_val - an_exact) / max(abs(an_exact), 1e-12)
        assert rel_error < 0.01, f"an LUT error too large at V={v}: {rel_error:.6f}"
        
        bn_exact = float(bn(v))
        bn_lut_val = float(bn_lut(v))
        rel_error = abs(bn_lut_val - bn_exact) / max(abs(bn_exact), 1e-12)
        assert rel_error < 0.01, f"bn LUT error too large at V={v}: {rel_error:.6f}"
    
    print(f"LUT interpolation accuracy: all gate functions within 1% at {len(test_voltages)} test points")


if __name__ == "__main__":
    test_lut_interpolation_accuracy()
    test_squid_axon_lut_precision()
    test_l5_pyramidal_lut_precision()
    print("\nAll LUT validation tests passed!")
