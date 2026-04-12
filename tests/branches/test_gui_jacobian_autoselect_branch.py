"""
Branch checks for GUI Jacobian auto-selection on heavy presets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.config_manager import ConfigManager


def test_heavy_preset_keeps_preset_native_hines_choice():
    mgr = ConfigManager()
    mgr.load_preset("F: Multiple Sclerosis (Demyelination)")
    mgr.auto_select_jacobian_for_preset()
    assert mgr.config.stim.jacobian_mode == "native_hines"


def test_nonheavy_preset_keeps_preset_default_choice():
    mgr = ConfigManager()
    mgr.load_preset("A: Squid Giant Axon (HH 1952)")
    mgr.auto_select_jacobian_for_preset()
    assert mgr.config.stim.jacobian_mode == "native_hines"


def test_preset_load_resets_dual_stim_to_disabled():
    mgr = ConfigManager()
    from core.dual_stimulation import DualStimulationConfig

    mgr.config.dual_stimulation = DualStimulationConfig(enabled=True, secondary_Iext=42.0)
    assert mgr.config.dual_stimulation is not None and mgr.config.dual_stimulation.enabled

    mgr.load_preset("B: Pyramidal L5 (Mainen 1996)")
    assert mgr.config.dual_stimulation is None


def _run_as_script() -> int:
    tests = [
        test_heavy_preset_keeps_preset_native_hines_choice,
        test_nonheavy_preset_keeps_preset_default_choice,
        test_preset_load_resets_dual_stim_to_disabled,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"\nSummary: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
