from core.rhs_contract import RHS_ARG_INDEX, RHS_ARG_ORDER, pack_rhs_args


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
