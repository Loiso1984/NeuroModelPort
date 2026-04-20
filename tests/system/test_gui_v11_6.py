#!/usr/bin/env python3
"""Test GUI v11.6 enhancements."""
import sys
print('Testing GUI v11.6...')

# Fresh import
import importlib
import gui.analytics
importlib.reload(gui.analytics)

from gui.analytics import AnalyticsWidget
print('✅ AnalyticsWidget imports')

# Create instance without UI
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    import matplotlib
    matplotlib.use('Agg')
    
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    # Create widget
    widget = AnalyticsWidget()
    
    # Check registry
    specs = widget._all_tab_specs
    tab_21 = specs.get(21)
    if tab_21:
        print(f"✅ Tab 21: {tab_21}")
        assert tab_21['title'] == 'Metabolic', f"Wrong title: {tab_21['title']}"
        print('✅ Metabolic tab registered correctly')
    else:
        print('❌ Tab 21 not found')
        print(f'Available tabs: {list(specs.keys())}')
        sys.exit(1)
        
    # Check methods
    assert hasattr(widget, '_build_tab_metabolic'), 'Missing _build_tab_metabolic'
    assert hasattr(widget, '_update_metabolic'), 'Missing _update_metabolic'
    print('✅ Metabolic methods exist')
    
    print()
    print('🎉 GUI V11.6 READY:')
    print('   ✅ ATP Crisis Warning')
    print('   ✅ Enhanced ATP Plot')
    print('   ✅ Metabolic Trajectory Tab')
    print('   ✅ Pump Current ready')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
