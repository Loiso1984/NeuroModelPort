"""
gui/config_manager.py — Configuration Management Service

Handles all operations related to the model state:
- Config object lifecycle
- Preset loading and application
- Save/Load JSON operations
- Dual stimulation synchronization
- Parameter hint text generation
"""

import copy
import numpy as np
from PySide6.QtCore import QObject, Signal
from typing import Optional

from core.models import FullModelConfig
from core.presets import apply_preset, apply_synaptic_stimulus
from core.unit_converter import density_to_absolute_current


class ConfigManager(QObject):
    """
    Manages FullModelConfig lifecycle and related operations.
    
    Separates configuration logic from MainWindow UI code.
    """
    
    # Signal emitted when config changes (preset loaded, file opened, etc.)
    config_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config: FullModelConfig = FullModelConfig()
        self._current_preset_name = ""
        self._dual_stim_widget = None  # Will be set by MainWindow
        self._form_stim = None  # Will be set by MainWindow
        self._form_stim_loc = None  # Will be set by MainWindow
        self._form_preset_modes = None  # Will be set by MainWindow
    
    def set_dual_stim_widget(self, widget):
        """Set reference to dual stimulation widget."""
        self._dual_stim_widget = widget
    
    def set_form_widgets(self, form_stim, form_stim_loc, form_preset_modes):
        """Set references to form widgets for sync operations."""
        self._form_stim = form_stim
        self._form_stim_loc = form_stim_loc
        self._form_preset_modes = form_preset_modes
    
    @property
    def current_preset_name(self) -> str:
        """Get the current preset name."""
        return self._current_preset_name
    
    def load_preset(self, name: str) -> bool:
        """
        Load a preset by name.
        
        Parameters
        ----------
        name : str
            Preset name to load
            
        Returns
        -------
        bool
            True if preset was loaded, False if invalid name
        """
        if "—" in name or "Select" in name:
            return False
        
        self._current_preset_name = name
        apply_preset(self.config, name)
        self.config_changed.emit()
        return True
    
    def save_config_as(self, file_path: str) -> bool:
        """
        Save current configuration to JSON file.
        
        Parameters
        ----------
        file_path : str
            Path to save the configuration
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            self.config.save_to_file(file_path)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def load_config_from(self, file_path: str) -> bool:
        """
        Load configuration from JSON file.
        
        Parameters
        ----------
        file_path : str
            Path to load the configuration from
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            new_cfg = FullModelConfig.load_from_file(file_path)
            
            # Helper to recursively update Pydantic models without replacing them
            def _deep_update(target_obj, source_obj):
                for field_name in target_obj.model_fields:
                    target_val = getattr(target_obj, field_name)
                    source_val = getattr(source_obj, field_name)
                    
                    # If it's a nested Pydantic model, recurse
                    if hasattr(target_val, 'model_fields') and hasattr(source_val, 'model_fields'):
                        _deep_update(target_val, source_val)
                    else:
                        # Otherwise, just set the value (preserves the parent object identity)
                        setattr(target_obj, field_name, source_val)
            
            _deep_update(self.config, new_cfg)
            
            # Handle dual stimulation tab if present
            if self.config.dual_stimulation is not None and self._dual_stim_widget:
                # Same recursive update for dual stim
                _deep_update(self._dual_stim_widget.config, self.config.dual_stimulation)
                self._dual_stim_widget.update_ui_from_config()
            
            self.config_changed.emit()
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def sync_dual_stim_into_config(self) -> bool:
        """
        Sync dual-stimulation GUI config into main model config.
        
        Returns
        -------
        bool
            True if dual stimulation is enabled
        """
        if self._dual_stim_widget is None:
            return False
        
        dual_enabled = bool(self._dual_stim_widget.config.enabled)
        if dual_enabled:
            self.config.dual_stimulation = self._dual_stim_widget.get_config()
        else:
            self.config.dual_stimulation = None
        return dual_enabled
    
    def get_hint_text(self) -> str:
        """
        Generate hint text for parameter priority display.
        
        Returns
        -------
        str
            Formatted hint text showing dual stim status, mode flags, etc.
        """
        dual_enabled = bool(
            self._dual_stim_widget is not None 
            and self._dual_stim_widget.config.enabled
        )
        p = (self._current_preset_name or "").lower()
        pm = self.config.preset_modes
        
        mode_note = "Mode flags: none for this preset."
        if "thalamic" in p:
            mode_note = f"Mode flags: K mode={pm.k_mode}."
        elif "alzheimer" in p:
            mode_note = f"Mode flags: N mode={pm.alzheimer_mode}."
        elif "hypoxia" in p:
            mode_note = f"Mode flags: O mode={pm.hypoxia_mode}."
        elif "multiple sclerosis" in p:
            mode_note = "Mode flags: F is single-stage (no progressive/terminal switch)."
        
        if dual_enabled:
            priority_note = (
                "Priority: Dual Stim is ON, so primary stimulation values from the Dual Stim tab "
                "override main Stimulation/Stimulus Location fields."
            )
        else:
            priority_note = (
                "Priority: Dual Stim is OFF, so main Stimulation/Stimulus Location fields are active."
            )
        
        stim_note = ""
        if self.config.stim.stim_type == "const" and (
            "interneuron" in p or "hippocampal ca1" in p or "purkinje" in p
        ):
            stim_note = (
                " Const here represents tonic drive proxy (current-clamp style), "
                "not a single synaptic event; switch stim_type to alpha/AMPA/NMDA for event-like input."
            )
        
        jac = str(getattr(self.config.stim, "jacobian_mode", "dense_fd"))
        is_multi = not bool(self.config.morphology.single_comp)
        heavy_family = any(k in p for k in ("thalamic", "alzheimer", "hypoxia", "multiple sclerosis"))
        if is_multi and heavy_family:
            jac_note = (
                f" Jacobian: {jac}. Recommended for heavy presets: sparse_fd/analytic_sparse "
                "(faster than dense_fd)."
            )
        else:
            jac_note = f" Jacobian: {jac}."
        
        i_abs_nA = self.compute_display_iext_absolute_nA()
        iext_note = f" | Iext(abs): {i_abs_nA:.3f} nA (read-only)"
        
        return f"{priority_note}  {mode_note}{stim_note}{jac_note}{iext_note}"
    
    def compute_display_iext_absolute_nA(self) -> float:
        """
        Compute display-only absolute current from the active source of truth.
        
        - Dual Stim ON  -> dual primary Iext is effective input.
        - Dual Stim OFF -> canonical config.stim.Iext is effective input.
        
        Returns
        -------
        float
            Absolute current in nanoamperes (nA)
        """
        d = float(self.config.morphology.d_soma)
        area = np.pi * d * d
        
        if self._dual_stim_widget is not None and bool(self._dual_stim_widget.config.enabled):
            i_density = float(self._dual_stim_widget.config.primary_Iext)
        else:
            i_density = float(self.config.stim.Iext)
        
        return float(density_to_absolute_current(i_density, area))
    
    def get_dual_priority_text(self) -> str:
        """
        Get text for dual priority label.
        
        Returns
        -------
        str
            Priority note text
        """
        dual_enabled = bool(
            self._dual_stim_widget is not None 
            and self._dual_stim_widget.config.enabled
        )
        
        if dual_enabled:
            return (
                "Priority: Dual Stim is ON, so primary stimulation values from the Dual Stim tab "
                "override main Stimulation/Stimulus Location fields."
            )
        else:
            return (
                "Priority: Dual Stim is OFF, so main Stimulation/Stimulus Location fields are active."
            )
    
    def auto_select_jacobian_for_preset(self):
        """
        Prefer sparse_fd for computationally heavy multi-compartment presets
        unless user explicitly selected another non-dense mode or native_hines.
        """
        p = (self._current_preset_name or "").lower()
        heavy_family = any(k in p for k in ("thalamic", "alzheimer", "hypoxia", "multiple sclerosis"))
        if not heavy_family or bool(self.config.morphology.single_comp):
            return
        current_mode = str(getattr(self.config.stim, "jacobian_mode", "dense_fd"))
        # Don't override user's Hines selection or other non-dense modes
        if current_mode == "dense_fd":
            self.config.stim.jacobian_mode = "sparse_fd"
    
    def active_mode_suffix(self) -> str:
        """
        Compact status suffix for active preset mode selector state.
        
        Returns
        -------
        str
            Mode suffix string
        """
        if not self._current_preset_name:
            return ""
        name = self._current_preset_name
        pm = self.config.preset_modes
        if "Thalamic" in name:
            return f" | K mode={pm.k_mode}"
        if "Alzheimer" in name:
            return f" | N mode={pm.alzheimer_mode}"
        if "Hypoxia" in name:
            return f" | O mode={pm.hypoxia_mode}"
        return ""
