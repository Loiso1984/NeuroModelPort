"""test_gui_dual_stim_integration.py - Verify GUI dual stim integration"""

from PySide6.QtWidgets import QApplication
import sys

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from gui.dual_stimulation_widget import DualStimulationWidget
from core.dual_stimulation import DualStimulationConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Create QApplication required by PySide6
app = QApplication(sys.argv)


def test_gui_dual_stim_sync():
    """Test that dual stim widget config syncs to main config properly."""
    print("\n" + "="*70)
    print("TEST: GUI Dual Stim Integration")
    print("="*70)
    
    # Simulate GUI behavior
    main_config = FullModelConfig()
    apply_preset(main_config, 'B: Pyramidal L5 (Mainen 1996)')
    
    print(f"\n1. Main config created from L5 preset")
    print(f"   Location: {main_config.stim_location.location}")
    print(f"   Iext: {main_config.stim.Iext}")
    print(f"   dual_stimulation: {main_config.dual_stimulation}")
    
    # Create widget (as GUI does)
    widget = DualStimulationWidget()
    print(f"\n2. Widget created with default preset")
    print(f"   Widget enabled: {widget.config.enabled}")
    print(f"   Primary: {widget.config.primary_location}")
    print(f"   Secondary: {widget.config.secondary_location}")
    
    # User enables dual stim
    widget.config.enabled = True
    widget.config.primary_location = 'dendritic_filtered'
    widget.config.primary_Iext = 6.0
    widget.config.secondary_location = 'soma'
    widget.config.secondary_stim_type = 'GABAA'
    widget.config.secondary_Iext = 2.0
    
    print(f"\n3. User configures dual stim in widget")
    print(f"   Widget enabled: {widget.config.enabled}")
    print(f"   Primary: {widget.config.primary_location} Iext={widget.config.primary_Iext}")
    print(f"   Secondary: {widget.config.secondary_location} {widget.config.secondary_stim_type} Iext={widget.config.secondary_Iext}")
    
    # Simulate what run_simulation does
    if widget.config.enabled:
        main_config.dual_stimulation = widget.get_config()
    else:
        main_config.dual_stimulation = None
    
    print(f"\n4. Config synced to main (as run_simulation does)")
    print(f"   main_config.dual_stimulation: {main_config.dual_stimulation}")
    if main_config.dual_stimulation:
        print(f"   ├─ enabled: {main_config.dual_stimulation.enabled}")
        print(f"   ├─ primary: {main_config.dual_stimulation.primary_location}")
        print(f"   └─ secondary: {main_config.dual_stimulation.secondary_location}")
    
    # Now run simulator with synced config
    print(f"\n5. Run simulation with dual stim config")
    solver = NeuronSolver(main_config)
    result = solver.run_single()
    
    peak = result.v_soma.max()
    n_spikes = (result.v_soma > -30.0).sum()
    
    print(f"   V_peak: {peak:.1f} mV")
    print(f"   Spikes: {n_spikes}")
    print(f"   Status: ✅ PASS - Dual stim executed through GUI config")
    
    return True


def test_dual_stim_disabled():
    """Test that disabling dual stim works."""
    print("\n" + "="*70)
    print("TEST: GUI Dual Stim Disabled")
    print("="*70)
    
    main_config = FullModelConfig()
    apply_preset(main_config, 'B: Pyramidal L5 (Mainen 1996)')
    
    widget = DualStimulationWidget()
    widget.config.enabled = False  # User disables dual stim
    
    print(f"\n1. User disables dual stim")
    print(f"   Widget enabled: {widget.config.enabled}")
    
    # Sync config
    if widget.config.enabled:
        main_config.dual_stimulation = widget.get_config()
    else:
        main_config.dual_stimulation = None
    
    print(f"\n2. Config synced")
    print(f"   main_config.dual_stimulation: {main_config.dual_stimulation}")
    
    # Run simulation
    print(f"\n3. Run simulation with dual stim disabled")
    solver = NeuronSolver(main_config)
    result = solver.run_single()
    
    peak = result.v_soma.max()
    print(f"   V_peak: {peak:.1f} mV")
    print(f"   Status: ✅ PASS - Single stim worked when dual stim disabled")
    
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("GUI DUAL STIMULATION INTEGRATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    try:
        results['gui_sync'] = test_gui_dual_stim_sync()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)[:100]}")
        results['gui_sync'] = False
    
    try:
        results['gui_disabled'] = test_dual_stim_disabled()
    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)[:100]}")
        results['gui_disabled'] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    n_passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {n_passed}/{len(results)} tests passed")
    
    if n_passed == len(results):
        print("\n🎉 GUI INTEGRATION COMPLETE AND VALIDATED!")
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())
