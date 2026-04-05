from __future__ import annotations

import re


def missing_dependency_name(exc: BaseException) -> str:
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        return str(exc.name)
    msg = str(exc)
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg)
    if match:
        return match.group(1)
    return msg or "<unknown>"


def dependency_diagnostic(tool_label: str, exc: BaseException) -> str:
    dep = missing_dependency_name(exc)
    return (
        f"[{tool_label}] dependency diagnostic: missing runtime dependency {dep!r}. "
        "Install requirements.txt and retry."
    )
