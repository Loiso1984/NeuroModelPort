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
    values["cm_v"] = [1.0] * n
    values["b_ca"] = [0.0] * n
    values["gbar_mat"] = [[0.0] * n for _ in range(11)]
    values["phi_mat"] = [[1.0] * n for _ in range(9)]
    values["l_data"] = [1.0, 2.0]
    values["l_indices"] = [0, 1]
    values["l_indptr"] = [0, 1, 2, 2]
    values["event_times_arr"] = []
    values["n_events"] = 0
    values["stim_mode"] = 0
    values["stim_mode_2"] = 0
    values["stim_comp"] = 0
    values["stim_comp_2"] = 0
    values["dual_stim_enabled"] = 0
    values["use_dfilter_primary"] = 0
    values["use_dfilter_secondary"] = 0
    values["t_kelvin"] = 310.0
    values["tau_ca"] = 200.0
    values["tau_sk"] = 5.0
    values["atau"] = 2.0
    values["atau_2"] = 2.0
    values["ca_ext"] = 2.0
    values["ca_rest"] = 5e-5
    values["mg_ext"] = 1.0
    values["dfilter_tau_ms"] = 0.0
    values["dfilter_tau_ms_2"] = 0.0

    validate_rhs_args_values(values)

    bad = dict(values)
    bad["phi_mat"] = [[0.0] * n for _ in range(9)]
    bad["phi_mat"][8] = [0.0] * (n - 1)
    try:
        validate_rhs_args_values(bad)
        assert False, "Expected ValueError for phi_mat row shape mismatch"
    except ValueError as exc:
        assert "phi_mat" in str(exc)

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

    bad6 = dict(values)
    bad6["dual_stim_enabled"] = 2
    try:
        validate_rhs_args_values(bad6)
        assert False, "Expected ValueError for invalid dual_stim_enabled"
    except ValueError as exc:
        assert "dual_stim_enabled" in str(exc)

    bad7 = dict(values)
    bad7["use_dfilter_primary"] = 3
    try:
        validate_rhs_args_values(bad7)
        assert False, "Expected ValueError for invalid use_dfilter_primary"
    except ValueError as exc:
        assert "use_dfilter_primary" in str(exc)

    bad8 = dict(values)
    bad8["use_dfilter_secondary"] = -1
    try:
        validate_rhs_args_values(bad8)
        assert False, "Expected ValueError for invalid use_dfilter_secondary"
    except ValueError as exc:
        assert "use_dfilter_secondary" in str(exc)

    bad9 = dict(values)
    bad9["gbar_mat"] = [[0.0] * n for _ in range(11)]
    bad9["gbar_mat"][0][0] = -1.0
    try:
        validate_rhs_args_values(bad9)
        assert False, "Expected ValueError for negative gbar_mat entry"
    except ValueError as exc:
        assert "gbar_mat" in str(exc)

    bad10 = dict(values)
    bad10["phi_mat"] = [[1.0] * n for _ in range(9)]
    bad10["phi_mat"][0][1] = 0.0
    try:
        validate_rhs_args_values(bad10)
        assert False, "Expected ValueError for non-positive phi_mat entry"
    except ValueError as exc:
        assert "phi_mat" in str(exc)

    bad11 = dict(values)
    bad11["cm_v"] = [1.0, 0.0, 1.0]
    try:
        validate_rhs_args_values(bad11)
        assert False, "Expected ValueError for non-positive cm_v entry"
    except ValueError as exc:
        assert "cm_v" in str(exc)

    bad12 = dict(values)
    bad12["b_ca"] = [0.0, -1.0, 0.0]
    try:
        validate_rhs_args_values(bad12)
        assert False, "Expected ValueError for negative b_ca entry"
    except ValueError as exc:
        assert "b_ca" in str(exc)

    bad13 = dict(values)
    bad13["n_events"] = 3
    bad13["event_times_arr"] = [2.0, 1.0, 3.0]
    try:
        validate_rhs_args_values(bad13)
        assert False, "Expected ValueError for non-monotonic event_times_arr"
    except ValueError as exc:
        assert "event_times_arr" in str(exc)

    bad14 = dict(values)
    bad14["n_events"] = 2
    bad14["event_times_arr"] = [1.0, float("inf")]
    try:
        validate_rhs_args_values(bad14)
        assert False, "Expected ValueError for non-finite event_times_arr"
    except ValueError as exc:
        assert "event_times_arr" in str(exc)

    bad15 = dict(values)
    bad15["tau_ca"] = 0.0
    try:
        validate_rhs_args_values(bad15)
        assert False, "Expected ValueError for non-positive tau_ca"
    except ValueError as exc:
        assert "tau_ca" in str(exc)

    bad16 = dict(values)
    bad16["ca_ext"] = -1.0
    try:
        validate_rhs_args_values(bad16)
        assert False, "Expected ValueError for negative ca_ext"
    except ValueError as exc:
        assert "ca_ext" in str(exc)

    bad17 = dict(values)
    bad17["stype"] = 3
    try:
        validate_rhs_args_values(bad17)
        assert False, "Expected ValueError for unsupported stype"
    except ValueError as exc:
        assert "stype" in str(exc)

    bad18 = dict(values)
    bad18["stype_2"] = 42
    try:
        validate_rhs_args_values(bad18)
        assert False, "Expected ValueError for unsupported stype_2"
    except ValueError as exc:
        assert "stype_2" in str(exc)

    bad19 = dict(values)
    bad19["iext"] = float("nan")
    try:
        validate_rhs_args_values(bad19)
        assert False, "Expected ValueError for non-finite iext"
    except ValueError as exc:
        assert "iext" in str(exc)

    bad20 = dict(values)
    bad20["td_2"] = -1.0
    try:
        validate_rhs_args_values(bad20)
        assert False, "Expected ValueError for negative td_2"
    except ValueError as exc:
        assert "td_2" in str(exc)
