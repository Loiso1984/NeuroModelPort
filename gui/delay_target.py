"""Shared delay-target resolution utilities for GUI widgets.

Centralizes morphology-index semantics used by oscilloscope and topology views.
"""

from __future__ import annotations


def junction_index(n_comp: int, n_ais: int, n_trunk: int) -> int:
    if n_comp <= 1:
        return 0
    if int(n_trunk) > 0:
        return 1 + int(n_ais) + int(n_trunk) - 1
    if int(n_ais) > 0:
        return int(n_ais)
    return 0


def resolve_delay_target(
    *,
    target_name: str,
    custom_index: int,
    n_comp: int,
    n_ais: int,
    n_trunk: int,
    terminal_idx: int | None = None,
) -> tuple[int | None, str, str]:
    """Resolve delay target to concrete compartment index.

    Returns (idx, display_label, semantic_key).
    """
    if n_comp <= 1:
        return None, "", "n/a"

    terminal = int(n_comp - 1 if terminal_idx is None else max(1, min(terminal_idx, n_comp - 1)))
    target = str(target_name or "Terminal")

    if target == "AIS":
        if int(n_ais) > 0 and n_comp > 1:
            return 1, "AIS", "ais"
        return None, "", "ais"

    if target == "Trunk Junction":
        j = junction_index(n_comp, n_ais, n_trunk)
        if 1 <= j < n_comp:
            return j, "junction", "junction"
        return None, "", "junction"

    if target == "Custom Compartment":
        idx = max(1, min(terminal, int(custom_index)))
        return idx, f"comp[{idx}]", "custom"

    return terminal, "terminal", "terminal"
