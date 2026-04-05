from gui.delay_target import junction_index, resolve_delay_target


def test_junction_index_prefers_trunk_then_ais():
    assert junction_index(10, n_ais=3, n_trunk=4) == 7
    assert junction_index(10, n_ais=3, n_trunk=0) == 3
    assert junction_index(10, n_ais=0, n_trunk=0) == 0


def test_resolve_delay_target_tau_independent_indexing():
    idx, label, key = resolve_delay_target(
        target_name="Trunk Junction",
        custom_index=5,
        n_comp=12,
        n_ais=2,
        n_trunk=5,
        terminal_idx=11,
    )
    assert idx == 7
    assert label == "junction"
    assert key == "junction"


def test_resolve_delay_target_custom_and_terminal():
    idx, label, key = resolve_delay_target(
        target_name="Custom Compartment",
        custom_index=99,
        n_comp=8,
        n_ais=2,
        n_trunk=2,
        terminal_idx=7,
    )
    assert idx == 7
    assert label == "comp[7]"
    assert key == "custom"

    idx2, label2, key2 = resolve_delay_target(
        target_name="Terminal",
        custom_index=1,
        n_comp=8,
        n_ais=2,
        n_trunk=2,
        terminal_idx=7,
    )
    assert idx2 == 7
    assert label2 == "terminal"
    assert key2 == "terminal"
