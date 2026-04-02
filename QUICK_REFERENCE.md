# Quick Reference Guide — Channel Currents Tab

## For Developers

### File Location
```
c:\NeuroModelPort\gui\analytics.py
```

### Key Lines
| Section | Line Range | Purpose |
|---------|-----------|---------|
| Tab Registration | 128-132 | Create figure, canvas, add to widget |
| Update Call | 208 | Trigger update from main pipeline |
| Method Implementation | 401-438 | Core plotting logic |

---

## To Modify the Tab

### Change Tab Label
**Location**: Line 131
```python
self.addTab(_tab_with_toolbar(cvs), "⚡ Currents")  # ← Change this string
```

### Change Colors
**Location**: Line 35-41 (CHAN_COLORS constant)
```python
CHAN_COLORS = {
    'Na':   '#DC3232',  # Change hex code here
    'K':    '#3264DC',
    # ...
}
```

### Change Figure Size
**Location**: Line 129
```python
self.fig_currents, cvs = _mpl_fig(3, 1)  # (height_multiplier, num_cols)
```

### Change Grid Appearance
**Location**: Line 431 (_configure_ax_interactive call)
```python
_configure_ax_interactive(ax, show_legend=True, grid_alpha=0.15)
#                                                 ↑ Change transparency
```

### Change Zero-Line Appearance
**Location**: Line 428
```python
ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.8)
#                  ↑        ↑            ↑              ↑
#               color    line type   transparency   thickness
```

---

## To Debug Issues

### Issue: Tab appears empty
**Check**:
1. Is `result.currents` populated? (solver.py lines 168-193)
2. Are all channels below threshold (1e-9 pA)? → Use smaller threshold
3. Run verification: `python verify_implementation.py`

**Fix**:
```python
# Line 407 - Lower amplitude threshold
currents = {name: curr for name, curr in result.currents.items() 
           if np.max(np.abs(curr)) > 1e-12}  # ← Change 1e-9 to smaller value
```

### Issue: Some channels not showing
**Check**:
1. Are channels in `result.currents`?
2. Is channel amplitude > 1e-9 pA?

**Debug**:
```python
# Add debug print to _update_currents
print("Available currents:", list(result.currents.keys()))
print("Active currents:", list(currents.keys()))
```

### Issue: Colors don't match other tabs
**Check**:
1. Using `CHAN_COLORS` constant? (Line 425)
2. Is constant defined at top of file? (Line 35-41)

**Fix**: Ensure color mapping is consistent:
```python
color = CHAN_COLORS.get(name, '#888888')  # Default gray if not found
```

---

## Testing Checklist

Before deploying changes:

```bash
# 1. Syntax check
python -m py_compile gui/analytics.py

# 2. Import test
python -c "from gui.analytics import AnalyticsWidget; print('✓')"

# 3. Verification
python verify_implementation.py

# 4. Visual test
# Run full GUI and check tab appears correctly
python main.py
```

---

## Common Customizations

### Add another subplot reference
```python
# After membrane potential plot (line 415), add:
ax_stim = self.fig_currents.add_subplot(n_rows + 1, 1, n_rows + 1)
ax_stim.plot(t, result.stim_trace, color='gray')
```

### Log current statistics
```python
# Inside the loop (line 423), add:
I_max = np.max(np.abs(curr))
I_mean = np.mean(np.abs(curr))
print(f"{name}: max={I_max:.2f}, mean={I_mean:.2f} pA")
```

### Filter specific channels (hide SK)
```python
# Line 406-407, modify filter:
currents = {name: curr for name, curr in result.currents.items() 
           if np.max(np.abs(curr)) > 1e-9 and name != 'SK'}
```

### Custom axis limits
```python
# After plot command (line 427), add:
if name == 'Na':
    ax.set_ylim(-500, 100)  # Custom limits for Na only
```

---

## Integration Points

### Where currents come from
**File**: `core/solver.py` lines 168-193  
**Function**: `_post_process_physics()`  
**Output**: `result.currents` dictionary

### Where tab is displayed
**File**: `gui/main_window.py` (or wherever AnalyticsWidget is used)  
**Method**: Creates AnalyticsWidget and calls `update_analytics()`

### What triggers update
**File**: `gui/main_window.py` (wherever simulation results are received)  
**Call**: `self.analytics_widget.update_analytics(result)`

---

## Performance Notes

### Optimization Tips
1. **Filter zeros early** (Line 406-407) - Avoids plotting empty traces
2. **Reuse colors** - Use CHAN_COLORS dict, don't compute colors per frame
3. **Single figure clear** - Line 402 clears entire figure at once
4. **Tight layout** - Line 403 minimizes drawing time

### Potential Bottlenecks
1. Large time arrays (>100k points) - May slow rendering
2. Many channels (>7) - Increases subplot count and drawing time
3. Frequent redraws - Avoid calling update_currents unnecessarily

---

## Related Code

**Similar methods to reference for patterns**:
- `_update_gates()` (Line 370-397) - Same layout pattern
- `_update_energy()` (Line 597-625) - Uses CHAN_COLORS
- `_update_balance()` (Line 551-571) - Dual subplot pattern

**Helper functions used**:
- `_configure_ax_interactive()` (Line 68-86) - Formatting utility
- `_mpl_fig()` (Line 49-54) - Figure creation utility
- `_tab_with_toolbar()` (Line 57-64) - Tab wrapping utility

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 10.1 | 2024 Q1 | Added Channel Currents tab (+45 lines) |
| 10.0 | 2023 Q4 | Previous version baseline |

---

## Support Resources

**Documentation Files**:
- `FINAL_IMPLEMENTATION_REPORT.md` - Complete implementation details
- `CURRENTS_TAB_SUMMARY.md` - Feature overview
- `ARCHITECTURE_DIAGRAM.md` - System architecture diagrams

**Related Code**:
- `core/solver.py` - Currents calculation
- `core/analysis.py` - Analysis utilities
- `gui/plots.py` - Other visualization patterns

---

## Quick Copy-Paste Examples

### To add a new checkbox for toggling current visibility
```python
# In _update_currents method, wrap plot in condition:
if self.show_Na_current:  # Assume checkbox exists
    ax.plot(t, curr, color=color, lw=2.5)
```

### To export currents to CSV
```python
# After plotting, add:
import pandas as pd
data = {'t': t}
data.update(currents)
df = pd.DataFrame(data)
df.to_csv('currents.csv', index=False)
```

### To add current annotations
```python
# After plot command, add:
peak_idx = np.argmax(np.abs(curr))
ax.annotate(f'Peak: {curr[peak_idx]:.0f} pA',
            xy=(t[peak_idx], curr[peak_idx]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round', fc='yellow', alpha=0.5))
```

---

**Last Updated**: 2024  
**Maintainer**: NeuroModel Analytics Team  
**Quick Reference v1.0**
