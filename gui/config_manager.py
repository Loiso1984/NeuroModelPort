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
        Apply a configuration from a JSON file onto the existing manager state.
        
        Loads a FullModelConfig from the given file and copies its field values into the existing
        ConfigManager.config object in-place (preserving object identity). If the loaded config
        contains a dual_stimulation section and a dual-stimulation widget is set, the widget's
        config is updated in-place and its UI is refreshed. Emits the `config_changed` signal on success.
        
        Parameters
        ----------
        file_path : str
            Path to the JSON file containing the configuration to load.
        
        Returns
        -------
        bool
            `True` if the configuration was applied successfully, `False` if an error occurred.
        """
        try:
            new_cfg = FullModelConfig.load_from_file(file_path)
            
            # Helper to recursively update Pydantic models without replacing them
            def _deep_update(target_obj, source_obj):
                """
                Recursively copy values from `source_obj` into `target_obj`, updating fields in-place.
                
                Iterates over `target_obj.model_fields`; for each field, if both the target and source values are Pydantic models (have `model_fields`), recurse into them, otherwise assign the source value onto the target attribute. This preserves the identity of parent objects while updating their contained data.
                
                Parameters:
                    target_obj: A Pydantic model instance to be updated in-place.
                    source_obj: A Pydantic model instance supplying values to copy.
                """
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
        Synchronize the dual-stimulation widget's configuration into the active model configuration.
        
        @returns `true` if dual stimulation is enabled, `false` otherwise.
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
        Builds a multi-part hint string describing preset-specific mode flags, dual-stimulation priority, stimulus notes, Jacobian guidance, and the absolute external current for UI display.
        
        The returned text summarizes:
        - whether Dual Stim is enabled and which controls take priority,
        - preset-specific mode flags (K/N/O/F) when applicable,
        - a note about `const` stim_type for certain presets,
        - Jacobian mode with recommendations for multi-compartment heavy presets,
        - a read-only absolute Iext value in nA.
        
        Returns:
            str: Formatted hint text suitable for showing parameter priorities and related recommendations in the UI.
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
                "Priority: Dual Stim is ON, secondary stimulation is active in addition to "
                "main Stimulation/Stimulus Location fields."
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
        
        - Display uses dual primary Iext when Dual Stim is ON.
        - Display uses canonical config.stim.Iext when Dual Stim is OFF.
        
        Returns
        -------
        float
            Absolute current in nanoamperes (nA)
        """
        d = float(self.config.morphology.d_soma)
        area = np.pi * d * d
        
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
                "Priority: Dual Stim is ON, secondary stimulation is active in addition to "
                "main Stimulation/Stimulus Location fields."
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
