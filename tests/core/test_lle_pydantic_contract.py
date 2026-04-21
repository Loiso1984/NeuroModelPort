from __future__ import annotations


def test_analysis_params_use_native_benettin_lle_fields_only():
    from core.models import AnalysisParams

    fields = AnalysisParams.model_fields
    for legacy in (
        "lyapunov_embedding_dim",
        "lyapunov_lag_steps",
        "lyapunov_min_separation_ms",
        "lyapunov_fit_start_ms",
        "lyapunov_fit_end_ms",
    ):
        assert legacy not in fields

    for expected in (
        "lle_delta",
        "lle_t_evolve_ms",
        "lle_subspace",
        "lle_custom_v",
        "lle_custom_gates",
        "lle_custom_ca",
        "lle_custom_atp",
    ):
        assert expected in fields
