"""
Branch checks for hybrid neuron passport classification (rules + lightweight ML).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis import classify_neuron_hybrid, classify_neuron_ml, full_analysis
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def test_ml_classifier_matches_fs_like_signature():
    label, conf = classify_neuron_ml(
        fi_hz=190.0,
        fs_hz=170.0,
        ai=0.04,
        hw_ms=0.42,
        cv_isi=0.08,
    )
    assert label == "FS", f"expected FS prototype match, got {label}"
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5, f"expected confident FS estimate, got {conf}"


def test_hybrid_prefers_ml_for_intermediate_rule_when_confident():
    hybrid, source, conf = classify_neuron_hybrid(
        "Intermediate",
        fi_hz=30.0,
        fs_hz=12.0,
        ai=0.45,
        hw_ms=1.1,
        cv_isi=0.22,
    )
    assert source in {"ml_only", "rule+ml", "rule_priority"}
    assert 0.0 <= conf <= 1.0
    # For this RS-like point, hybrid should not stay in generic intermediate.
    assert "Intermediate" not in hybrid, f"expected a typed hybrid label, got {hybrid}"


def test_full_analysis_exposes_hybrid_classification_fields():
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.t_sim = 120.0
    cfg.stim.dt_eval = 0.2
    result = NeuronSolver(cfg).run_single()
    stats = full_analysis(result)
    for key in (
        "neuron_type_rule",
        "neuron_type_ml",
        "neuron_type_ml_confidence",
        "neuron_type_hybrid",
        "neuron_type_hybrid_source",
        "neuron_type_hybrid_confidence",
    ):
        assert key in stats, f"missing field {key} in full_analysis output"
    assert 0.0 <= float(stats["neuron_type_ml_confidence"]) <= 1.0
    assert 0.0 <= float(stats["neuron_type_hybrid_confidence"]) <= 1.0


def _run_as_script() -> int:
    tests = [
        test_ml_classifier_matches_fs_like_signature,
        test_hybrid_prefers_ml_for_intermediate_rule_when_confident,
        test_full_analysis_exposes_hybrid_classification_fields,
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

