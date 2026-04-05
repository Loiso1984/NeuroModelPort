from core.rhs_contract import (
    RHS_ARG_COUNT,
    RHS_ARG_INDEX,
    RHS_ARG_ORDER,
    pack_rhs_args,
    unpack_rhs_args,
    validate_rhs_args_values,
)


def test_rhs_arg_index_map_is_contiguous_and_stable_keys_present():
    assert len(RHS_ARG_ORDER) == len(RHS_ARG_INDEX)
    assert set(RHS_ARG_ORDER) == set(RHS_ARG_INDEX)
    assert RHS_ARG_INDEX["use_dfilter_primary"] < RHS_ARG_INDEX["use_dfilter_secondary"]
    assert RHS_ARG_ORDER[RHS_ARG_INDEX["l_indices"]] == "l_indices"


def test_pack_rhs_args_rejects_missing_or_extra_keys():
    baseline = {name: i for i, name in enumerate(RHS_ARG_ORDER)}
    packed = pack_rhs_args(baseline)
    assert packed[0] == baseline["n_comp"]
    assert packed[-1] == baseline["dfilter_tau_ms_2"]

    missing = dict(baseline)
    missing.pop("n_comp")
    try:
        pack_rhs_args(missing)
        assert False, "Expected KeyError for missing keys"
    except KeyError:
        pass

    extra = dict(baseline)
    extra["ghost"] = 123
    try:
        pack_rhs_args(extra)
        assert False, "Expected KeyError for extra keys"
    except KeyError:
        pass


def test_unpack_rhs_args_requires_exact_contract_length():
    baseline = {name: i for i, name in enumerate(RHS_ARG_ORDER)}
    packed = pack_rhs_args(baseline)
    unpacked = unpack_rhs_args(packed)
    assert len(packed) == RHS_ARG_COUNT
    assert unpacked["n_comp"] == baseline["n_comp"]
    assert unpacked["dfilter_tau_ms_2"] == baseline["dfilter_tau_ms_2"]

    try:
        unpack_rhs_args(packed[:-1])
        assert False, "Expected ValueError for short rhs args"
    except ValueError:
        pass


def test_validate_rhs_args_values_rejects_shape_mismatch():
    n = 3
    values = {name: 0 for name in RHS_ARG_ORDER}
    values["n_comp"] = n
    for key in (
        "gna_v", "gk_v", "gl_v", "gih_v", "gca_v", "ga_v", "gsk_v", "gtca_v", "gim_v", "gnap_v", "gnar_v",
        "cm_v",
        "phi_na", "phi_k", "phi_ih", "phi_ca", "phi_ia", "phi_tca", "phi_im", "phi_nap", "phi_nar",
        "b_ca",
    ):
        values[key] = [0.0] * n
    values["l_data"] = [1.0, 2.0]
    values["l_indices"] = [0, 1]
    values["l_indptr"] = [0, 1, 2, 2]
    values["event_times_arr"] = []
    values["n_events"] = 0
    values["stim_mode"] = 0
    values["stim_mode_2"] = 0
    values["stim_comp"] = 0
    values["stim_comp_2"] = 0
    values["dfilter_tau_ms"] = 0.0
    values["dfilter_tau_ms_2"] = 0.0

    validate_rhs_args_values(values)

    bad = dict(values)
    bad["phi_nar"] = [0.0] * (n - 1)
    try:
        validate_rhs_args_values(bad)
        assert False, "Expected ValueError for vector shape mismatch"
    except ValueError as exc:
        assert "phi_nar" in str(exc)

    bad2 = dict(values)
    bad2["l_indptr"] = [0, 1, 2]
    try:
        validate_rhs_args_values(bad2)
        assert False, "Expected ValueError for l_indptr mismatch"
    except ValueError as exc:
        assert "l_indptr" in str(exc)

    bad2b = dict(values)
    bad2b["l_indptr"] = [1, 1, 2, 2]
    try:
        validate_rhs_args_values(bad2b)
        assert False, "Expected ValueError for l_indptr head mismatch"
    except ValueError as exc:
        assert "first element" in str(exc)

    bad2c = dict(values)
    bad2c["l_indices"] = [0, n]
    try:
        validate_rhs_args_values(bad2c)
        assert False, "Expected ValueError for out-of-range l_indices entry"
    except ValueError as exc:
        assert "l_indices" in str(exc)

    bad3 = dict(values)
    bad3["n_events"] = 2
    bad3["event_times_arr"] = [1.0]
    try:
        validate_rhs_args_values(bad3)
        assert False, "Expected ValueError for n_events/event_times mismatch"
    except ValueError as exc:
        assert "n_events" in str(exc)

    bad4 = dict(values)
    bad4["stim_mode"] = 3
    try:
        validate_rhs_args_values(bad4)
        assert False, "Expected ValueError for invalid stim_mode"
    except ValueError as exc:
        assert "stim_mode" in str(exc)

    bad5 = dict(values)
    bad5["stim_comp_2"] = n
    try:
        validate_rhs_args_values(bad5)
        assert False, "Expected ValueError for invalid stim_comp_2"
    except ValueError as exc:
        assert "stim_comp_2" in str(exc)
