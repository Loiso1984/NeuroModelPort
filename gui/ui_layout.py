"""Layout presets and guards for the dock-based PySide UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutPreset:
    name: str
    width: int
    height: int
    sidebar_max_width: int
    show_secondary_actions: bool
    float_stimulation: bool


LAYOUT_PRESETS: dict[str, LayoutPreset] = {
    "Laptop": LayoutPreset(
        name="Laptop",
        width=1100,
        height=700,
        sidebar_max_width=350,
        show_secondary_actions=False,
        float_stimulation=False,
    ),
    "Desktop": LayoutPreset(
        name="Desktop",
        width=1400,
        height=900,
        sidebar_max_width=520,
        show_secondary_actions=True,
        float_stimulation=True,
    ),
    "Presentation": LayoutPreset(
        name="Presentation",
        width=1280,
        height=800,
        sidebar_max_width=360,
        show_secondary_actions=False,
        float_stimulation=False,
    ),
    "Debug": LayoutPreset(
        name="Debug",
        width=1500,
        height=950,
        sidebar_max_width=520,
        show_secondary_actions=True,
        float_stimulation=True,
    ),
}


def preset_for_width(width: int) -> LayoutPreset:
    if width < 1250:
        return LAYOUT_PRESETS["Laptop"]
    if width < 1350:
        return LAYOUT_PRESETS["Presentation"]
    return LAYOUT_PRESETS["Desktop"]
