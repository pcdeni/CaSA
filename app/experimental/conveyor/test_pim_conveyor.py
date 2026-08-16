#!/usr/bin/env python3
"""test_pim_conveyor.py -- DRY unit tests for the #67 prefetch scheduler.

Card-free. No /dev/xdma, no card binary, no server. Exercises pim_conveyor.py
against the two shapes the task pins:
  * BitNet-2B  (native, fits one DIMM -> conveyor DEGENERATE, schedule trivial)
  * Llama2-13B-q4  (mainstream CROSSOVER: 162 MB/layer vs 21.2 ms/layer)
and the KV-contention crossover (13B + 4k KV stream: infeasible on 1 storage
channel, feasible on 2).

Run:  python3 test_pim_conveyor.py         (prints PASS/FAIL summary, exit 0/1)
"""
import os, sys, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "python"))
from pim_grid_config import GridConfig, BankState, DimmRole            # noqa: E402
import pim_conveyor as pc                                             # noqa: E402


def _grid():
    return GridConfig.default_bitnet(os.environ.get("BN_DIR"),
                                     active_banks=range(16))


RESULTS = []
def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"[PASS] {name}")
    except Exception as e:                                            # noqa: BLE001
        RESULTS.append((name, False, str(e)))
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


# --------------------------------------------------------------------------
def t_bitnet_degenerate():
    m = pc.bitnet_2b()
    sch = pc.ConveyorScheduler(_grid())
    depth, _ = sch.required_depth(m, mover_channels=1)
    assert depth == 0, f"BitNet must be depth 0 (fits resident), got {depth}"
    plan = sch.plan_token(m, mover_channels=1)
    assert plan.degenerate, "BitNet plan must be degenerate (all resident)"
    assert plan.per_token_staged_bytes == 0, "degenerate plan stages 0 bytes"
    checks = pc.validate(plan, m)
    assert len(checks) >= 3, "degenerate plan must pass the structural checks"
    # every layer has an ACTIVE assignment and NO staging
    for st in plan.steps:
        assert st.active_banks, f"L{st.layer} has no active banks"
        assert not st.staging_banks, f"L{st.layer} stages in a degenerate plan"


def t_13b_conveyor_valid():
    m = pc.llama2_13b_q4()
    sch = pc.ConveyorScheduler(_grid())
    depth, notes = sch.required_depth(m, mover_channels=1)
    assert depth == 1, f"13B (no KV) should hide at depth 1, got {depth}"
    plan = sch.plan_token(m, mover_channels=1)
    assert not plan.degenerate
    # the whole model streams once per token in the working-set regime
    assert abs(plan.per_token_staged_bytes - m.weight_bytes) / m.weight_bytes < 0.05
    checks = pc.validate(plan, m)          # raises on any violation
    # the three named properties are present and passed
    joined = " ".join(checks)
    assert "no bank computes while staging-in" in joined
    assert "resident before its layer starts" in joined
    assert "read-ahead depth respected" in joined
    assert "HIDES" in joined


def t_13b_no_compute_while_staging():
    """Property (1): the ACTIVE group and the STAGING group are always disjoint."""
    m = pc.llama2_13b_q4()
    plan = pc.ConveyorScheduler(_grid()).plan_token(m, mover_channels=1)
    for st in plan.steps:
        assert not (set(st.active_banks) & set(st.staging_banks)), \
            f"L{st.layer}: overlap of ACTIVE and STAGING banks"


def t_13b_resident_before_layer():
    """Property (2): every non-layer-0 slice is staged during an EARLIER layer."""
    m = pc.llama2_13b_q4()
    plan = pc.ConveyorScheduler(_grid()).plan_token(m, mover_channels=1)
    staged = {}
    for st in plan.steps:
        for mv in st.moves:
            staged[(mv.slice.layer, mv.slice.name)] = mv.staged_during_layer
    for layer in m.layers:
        if layer.layer == 0:
            continue
        for sl in layer.slices:
            k = (sl.layer, sl.name)
            assert k in staged, f"{sl.name} L{sl.layer} never staged"
            assert staged[k] < sl.layer, \
                f"{sl.name} L{sl.layer} staged during L{staged[k]} (not before)"


def t_13b_readahead_and_no_2x():
    """Property (3): lead <= depth, lead >= 1, no SAME_DIMM_2X tier."""
    m = pc.llama2_13b_q4()
    plan = pc.ConveyorScheduler(_grid()).plan_token(m, mover_channels=1)
    for st in plan.steps:
        for mv in st.moves:
            lead = mv.slice.layer - mv.staged_during_layer
            assert 1 <= lead <= plan.depth, f"{mv.slice.name} lead {lead} vs depth {plan.depth}"
            assert mv.tier != pc.MoveTier.SAME_DIMM_2X, "2x tier must never appear"
            assert mv.tier == pc.MoveTier.CROSS_DIMM_1X, "conveyor tier is cross-DIMM 1x"


def t_13b_kv_crossover():
    """The ladder crossover: 13B + 4k KV stream is INFEASIBLE on 1 storage
    channel (bandwidth-bound) and FEASIBLE on 2."""
    m = pc.llama2_13b_q4_kv4k()
    sch = pc.ConveyorScheduler(_grid())
    # 1 channel -> validate must RAISE (a slice can't be resident in time)
    plan1 = sch.plan_token(m, mover_channels=1, depth=1)
    raised = False
    try:
        pc.validate(plan1, m, strict_timing=True)
    except pc.ScheduleError:
        raised = True
    assert raised, "13B+4kKV on 1 channel must be flagged INFEASIBLE"
    # 2 mover channels (17 GB/s) -> feasible
    plan2 = sch.plan_token(m, mover_channels=2, depth=1)
    checks = pc.validate(plan2, m, strict_timing=True)
    assert any("HIDES" in c for c in checks), "13B+4kKV on 2 channels must hide"


def t_ratios_match_ladder():
    """The stage/compute ratios must reproduce the bus_bound_ladder anchors:
    7B +37% margin (ratio ~0.73), 13B +11% (ratio ~0.90)."""
    sch = pc.ConveyorScheduler(_grid())
    bw = pc.MOVER_BW_BPS
    m7 = pc.llama2_7b_q4()
    r7 = max(m7.storage_demand_per_layer(i) / bw / l.compute_s
             for i, l in enumerate(m7.layers))
    assert 0.68 <= r7 <= 0.76, f"7B ratio {r7:.3f} not near ladder 0.73"
    m13 = pc.llama2_13b_q4()
    r13 = max(m13.storage_demand_per_layer(i) / bw / l.compute_s
              for i, l in enumerate(m13.layers))
    assert 0.85 <= r13 <= 0.92, f"13B ratio {r13:.3f} not near ladder 0.90"


def t_magic_config_wire_roundtrip():
    """The emitted CFG_SET_STATE payloads encode/decode through the #65 codec."""
    m = pc.llama2_13b_q4()
    plan = pc.ConveyorScheduler(_grid()).plan_token(m, mover_channels=1)
    # pick a step that emits transitions and round-trip through the grid codec
    step = next(s for s in plan.steps if s.set_state)
    for dimm, changes in step.magic_config.items():
        enc = GridConfig.encode_set_state(changes)
        assert enc[:4] == (0xB17EF00A).to_bytes(4, "little"), "MAGIC_CONFIG header"
        # opcode CFG_SET_STATE == 3
        assert enc[4:8] == (3).to_bytes(4, "little"), "CFG_SET_STATE opcode"
        assert len(enc) == 12 + 8 * len(changes), "wire size = 12 + 8/change"


def t_transitions_legal_65_rule():
    """No step sets a bank ACTIVE and STAGING at once; demotions are counted."""
    for factory, ch in [(pc.llama2_7b_q4, 1), (pc.llama2_13b_q4, 1)]:
        m = factory()
        plan = pc.ConveyorScheduler(_grid()).plan_token(m, mover_channels=ch)
        for st in plan.steps:
            act = {(d, b) for (d, b, s) in st.set_state if s == BankState.ACTIVE}
            stg = {(d, b) for (d, b, s) in st.set_state if s == BankState.STAGING}
            assert not (act & stg), f"{m.name} L{st.layer}: bank ACTIVE+STAGING in one step"


if __name__ == "__main__":
    print("#" * 74)
    print("  #67 conveyor scheduler -- DRY unit tests (card-free)")
    print("#" * 74)
    check("bitnet-2b degenerate (fits resident, depth 0)", t_bitnet_degenerate)
    check("13b-q4 conveyor schedule valid (3 named properties)", t_13b_conveyor_valid)
    check("13b-q4 (1) no compute while staging-in", t_13b_no_compute_while_staging)
    check("13b-q4 (2) every slice resident before its layer", t_13b_resident_before_layer)
    check("13b-q4 (3) read-ahead depth respected & no 2x tier", t_13b_readahead_and_no_2x)
    check("13b-q4+4kKV crossover: 1ch INFEASIBLE / 2ch feasible", t_13b_kv_crossover)
    check("stage/compute ratios match ladder (7B 0.73, 13B 0.90)", t_ratios_match_ladder)
    check("MAGIC_CONFIG CFG_SET_STATE wire round-trip (#65 codec)", t_magic_config_wire_roundtrip)
    check("all transitions legal under the #65 invalidation rule", t_transitions_legal_65_rule)
    print("#" * 74)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"  {npass}/{len(RESULTS)} PASS")
    print("#" * 74)
    sys.exit(0 if npass == len(RESULTS) else 1)
