from __future__ import annotations

from typing import Dict


_ENCODINGS = ("cp1250", "cp1252", "latin1")
_INVERSE_CACHE: Dict[str, Dict[str, int]] = {}
_SUSPICIOUS_FRAGMENTS = (
    "â",
    "Ã",
    "Â",
    "Đ",
    "Ń",
    "đź",
    "\ufffd",
)
_DIRECT_REPLACEMENTS = {
    "\r\n": "\n",
    "â€¦": "...",
    "â€”": "—",
    "â€“": "–",
    "âˆ’": "-",
    "â†’": "->",
    "Â·": "·",
    "Âµ": "µ",
    "Â°": "°",
    "Â±": "±",
    "Â²": "²",
    "Â³": "³",
    "ÂΩ": "Ω",
    "Î©": "Ω",
    "Î»": "λ",
    "Ïƒ": "σ",
    "Ď„": "τ",
    "Ă—": "x",
    "âšˇ": "HINES",
    "â–¶": ">",
    "â—€": "<",
}


def _inverse_table(encoding: str) -> Dict[str, int]:
    cached = _INVERSE_CACHE.get(encoding)
    if cached is not None:
        return cached
    table: Dict[str, int] = {}
    for byte in range(256):
        char = bytes([byte]).decode(encoding, errors="ignore")
        if char and char not in table:
            table[char] = byte
    _INVERSE_CACHE[encoding] = table
    return table


def _recover_via_single_byte(text: str, encoding: str) -> str:
    inverse = _inverse_table(encoding)
    payload = bytearray()
    for char in text:
        if char in inverse:
            payload.append(inverse[char])
        elif ord(char) < 256:
            payload.append(ord(char))
        else:
            raise ValueError(f"Cannot map {char!r} via {encoding}")
    return payload.decode("utf-8")


def _score_text(text: str) -> int:
    score = 0
    for fragment in _SUSPICIOUS_FRAGMENTS:
        score += text.count(fragment) * 10
    for char in text:
        code = ord(char)
        if 0x80 <= code <= 0x9F:
            score += 20
    return score


def repair_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text

    best = text
    for _ in range(2):
        current = best
        for src, dst in _DIRECT_REPLACEMENTS.items():
            current = current.replace(src, dst)

        current_score = _score_text(current)
        best_candidate = current
        best_score = current_score

        for encoding in _ENCODINGS:
            try:
                candidate = _recover_via_single_byte(current, encoding)
            except Exception:
                continue
            for src, dst in _DIRECT_REPLACEMENTS.items():
                candidate = candidate.replace(src, dst)
            candidate_score = _score_text(candidate)
            if candidate_score < best_score:
                best_candidate = candidate
                best_score = candidate_score

        if best_candidate == best:
            break
        best = best_candidate

    return best


def repair_widget_tree(root) -> None:
    from PySide6.QtWidgets import (
        QAbstractButton,
        QComboBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QTabWidget,
        QTextEdit,
        QWidget,
    )

    pending = [root]
    while pending:
        widget = pending.pop()

        if hasattr(widget, "windowTitle") and hasattr(widget, "setWindowTitle"):
            widget.setWindowTitle(repair_text(widget.windowTitle()))
        if hasattr(widget, "toolTip") and hasattr(widget, "setToolTip"):
            widget.setToolTip(repair_text(widget.toolTip()))

        if isinstance(widget, QAbstractButton):
            widget.setText(repair_text(widget.text()))
        elif isinstance(widget, QLabel):
            widget.setText(repair_text(widget.text()))
        elif isinstance(widget, QGroupBox):
            widget.setTitle(repair_text(widget.title()))
        elif isinstance(widget, QComboBox):
            for idx in range(widget.count()):
                widget.setItemText(idx, repair_text(widget.itemText(idx)))
            widget.setToolTip(repair_text(widget.toolTip()))
        elif isinstance(widget, QLineEdit):
            widget.setPlaceholderText(repair_text(widget.placeholderText()))
        elif isinstance(widget, QTextEdit):
            widget.setPlaceholderText(repair_text(widget.placeholderText()))
        elif isinstance(widget, QTabWidget):
            for idx in range(widget.count()):
                widget.setTabText(idx, repair_text(widget.tabText(idx)))

        for child in widget.children():
            if isinstance(child, QWidget):
                pending.append(child)
