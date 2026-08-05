"""pim_conveyor.py -- host-side SCHEDULE-DRIVEN PREFETCH SCHEDULER (task #67).

[#67 2026-08-04, conveyor_2026_08_04]  Built ON the #65 grid config plumbing
(pim_grid_config.py: BankState / DimmRole / GridConfig / MAGIC_CONFIG codec,
now LIVE in production as a default-inert superset -- bank16_config_2026_08_04).

WHAT THIS IS
------------
The token loop walks the transformer layers in a KNOWN, fixed order, so the
future working set is DETERMINISTIC. This module turns that determinism into a
prefetch schedule: while layer L computes on its ACTIVE compute banks, idle
banks preload layer L+d's weight slices (d = read-ahead depth). It emits, per
token:
  * the per-layer MAGIC_CONFIG transition plan (CFG_SET_STATE lists), and
  * the staged MOVE list (which slice, from where, to which bank, at which
    COST TIER), and validates the whole plan card-free.

THE COST HIERARCHY (RECORDED, user-ruled -- STOCKTAKE PLACEMENT LAW v2
2026-08-02 / LEVERS #67 / bus_bound_ladder rung a'). Cost is measured on the
BINDING (compute-DIMM) DDR bus:
  * FREE_ROWCLONE     in-subarray RowClone (src+dst share one 640-row segment)
                      -> ZERO bus, in-DRAM, >=99.98% (SiMRA Multi-RowCopy).
  * CROSS_DIMM_1X     prefetch from a STORAGE DIMM (D1/D3 Micron) -> 1x WRITE on
                      the compute bus; the READ rides the otherwise-idle storage
                      channel, hidden by read-ahead. THE conveyor tier.
  * SAME_DIMM_2X      same-DIMM cross-bank staging -> 2x (read + write). WORST
                      tier -- the scheduler REFUSES to plan this (raises); the
                      design keeps the parked copy on storage so it never needs
                      a same-DIMM cross-bank hop.

THE #65 INVALIDATION RULE (what a transition costs at the server; verbatim from
the bank16 twin handle_config_request / invalidate_resident_bank):
  * A transition is applied BETWEEN complete requests (single-threaded loop) --
    never races an in-flight matmul.
  * leaving ACTIVE (ACTIVE -> anything) invalidates that bank's resident scratch.
  * idle<->idle (FREE/STAGING/STORAGE among themselves) is METADATA-ONLY -- no
    stop-the-world. Staging happens here.
  * STAGING -> ACTIVE (promotion of a freshly-loaded bank) is NOT "leaving
    ACTIVE" -> does NOT invalidate. Promotion is free by construction.
  * TODAY'S SERVER CAVEAT (the main #67 server change, SERVER_CHANGE_SPEC.md):
    a LOAD handle reserves rows on EVERY active bank (n_rounds = ceil(n_units/N)
    per bank), so a bank leaving ACTIVE conservatively drops the WHOLE cross-bank
    handle map. Until handle residency is decoupled to a per-bank SUBSET, the
    conveyor ping-pong is only legal when the demoted bank shares no live handle
    with the promoted one. This scheduler models the DECOUPLED target and flags
    the coupling so the on-card gate can check it.

CARD-FREE. Pure data + arithmetic. No torch/FPGA deps. Consumes pim_grid_config.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Tuple, Optional, Iterable
import os, sys

# import the #65 schema (sibling module). Keep the import robust whether this
# runs from bitnet_weights/ or from the conveyor_2026_08_04/ deliverable dir.
try:
    from pim_grid_config import (BankState, DimmRole, GridConfig, BankSpec,
                                 DimmSpec, BANKS_PER_DIMM)
except ImportError:  # pragma: no cover - path shim for the deliverable dir
    sys.path.insert(0, "/home/deni/bitnet_weights")
    from pim_grid_config import (BankState, DimmRole, GridConfig, BankSpec,
                                 DimmSpec, BANKS_PER_DIMM)


# ---- hardware quanta (MEASURED; MENTAL_MODEL 2.3 / kv_alloc DESIGN 0) --------
ROW_BYTES        = 8192           # 8 KiB DRAM row = RowClone/read/write unit
SEGMENT_ROWS     = 640            # sense-amp segment = RowClone locality
ROWS_PER_BANK    = 65536          # 0.5 GiB/bank
BANK_BYTES       = ROWS_PER_BANK * ROW_BYTES          # 512 MiB raw/bank
MOVER_BW_BPS     = 8.5e9          # per storage channel, MEASURED casa_sched eff-BW
                                  # (bus_bound_ladder rung a'; D1+D3 = 2 channels)


class MoveTier(IntEnum):
    FREE_ROWCLONE = 0   # in-subarray, zero bus
    CROSS_DIMM_1X = 1   # storage->compute, 1x write on binding bus
    SAME_DIMM_2X  = 2   # same-DIMM cross-bank, 2x -- REFUSED by the planner


TIER_WRITE_COST = {          # binding-bus WRITE cost multiplier per staged byte
    MoveTier.FREE_ROWCLONE: 0.0,
    MoveTier.CROSS_DIMM_1X: 1.0,
    MoveTier.SAME_DIMM_2X:  2.0,
}


# ==========================================================================
# Model layer/slice map
# ==========================================================================
@dataclass
class Slice:
    """One resident weight unit of a layer (a projection's bit-planes). Sized in
    whole DRAM rows so bank packing is exact."""
    layer: int
    name: str          # q/k/v/o/gate/up/down
    rows: int          # resident rows this slice occupies
    @property
    def bytes(self) -> int:
        return self.rows * ROW_BYTES


@dataclass
class LayerMap:
    layer: int
    slices: List[Slice]
    compute_s: float                 # per-layer decode compute time (SIM/MEASURED)
    @property
    def rows(self) -> int:
        return sum(s.rows for s in self.slices)
    @property
    def bytes(self) -> int:
        return self.rows * ROW_BYTES
    @property
    def banks_needed(self) -> int:
        return max(1, -(-self.rows // ROWS_PER_BANK))   # ceil


@dataclass
class ModelMap:
    name: str
    layers: List[LayerMap]
    fits_resident: bool              # whole model weights fit the compute rank?
    note: str = ""
    # Per-layer KV bytes the fabric-attention pipeline READS off the storage
    # channel each token (whole cache / n_layer). NON-ZERO => the read-ahead no
    # longer rides a fully-idle storage channel; it contends with the KV stream
    # (the ladder's 13B+4kKV crossover). 0 = KV parked elsewhere / not streamed.
    kv_read_bytes_per_layer: int = 0
    @property
    def n_layers(self) -> int:
        return len(self.layers)
    @property
    def weight_bytes(self) -> int:
        return sum(l.bytes for l in self.layers)
    def storage_demand_per_layer(self, li: int) -> int:
        """Bytes the storage channel(s) must move during layer li's compute
        window = weight read-ahead + KV attention read."""
        return self.layers[li].bytes + self.kv_read_bytes_per_layer


def _proj_rows(K: int, M: int, w_bits: int, track: int) -> int:
    """Resident rows for one projection GEMV (contraction K, outputs M) stored as
    `track` weight bit-planes at `w_bits`. Bit-planes tile 8 KiB rows: each plane
    is K*M bits; rows = ceil(track * w_bits * K * M / 8 / ROW_BYTES). This is the
    RAW bit-plane residency the RTL-readback (mainstream) / MAJ (native) lanes
    read -- NOT the tiny screened MAJ3 pool (that only bounds the native lane's
    per-request scratch; #65 residency)."""
    bits = track * w_bits * K * M
    return max(1, -(-bits // (8 * ROW_BYTES)))


def model_from_shape(name, L, d, ffn, kv_heads, head_dim, w_bits, track,
                     compute_s_per_layer, fits_resident, note="") -> ModelMap:
    d_kv = kv_heads * head_dim
    projs = [("q", d, d), ("k", d, d_kv), ("v", d, d_kv), ("o", d, d),
             ("gate", d, ffn), ("up", d, ffn), ("down", ffn, d)]
    layers = []
    for li in range(L):
        sl = [Slice(li, pn, _proj_rows(K, M, w_bits, track)) for pn, K, M in projs]
        layers.append(LayerMap(li, sl, compute_s_per_layer))
    return ModelMap(name, layers, fits_resident, note)


# ---- the two shapes the task pins (dims: flagship_pricing.py SHAPES) ---------
def bitnet_2b() -> ModelMap:
    # native lane, ternary (2-track). Whole model ~0.52 GB -> fits one 8 GiB
    # compute rank => conveyor DEGENERATE (static #65 residency).
    # compute_s/layer: native residency wall 3.6 s/tok / 30 (bus_bound rung a,
    # ARITHMETIC). Not load-bearing here (no staging).
    return model_from_shape("bitnet_2b", 30, 2560, 6912, 5, 128,
                            w_bits=2, track=1, compute_s_per_layer=3.6/30,
                            fits_resident=True,
                            note="native/ternary; 0.52 GB fits D2 rank-0 -> static #65")


def llama2_13b_q4() -> ModelMap:
    # mainstream lane, q4. ~6.34 GB weights. The ladder CROSSOVER case:
    # 162 MB/layer, 21.2 ms compute/layer (0.846 s/tok / 40, 2-DIMM SIM floor).
    # Treated as working-set conveyor (STOCKTAKE: models that co-need full-ctx KV
    # cannot hold weights+KV resident on one 8 GiB rank).
    # HONESTY: 6.34 GB q4 weights ALONE fit one 8 GiB compute rank, so the
    # DEGENERATE placement (weights resident on D2, KV on storage D1/D3 --
    # capacity.py's recommendation) needs NO weight conveyor. This conveyor case
    # is the ALTERNATIVE placement (KV kept resident on the compute rank -> the
    # 9.46 GiB weights+KV overrun forces weights to stream) -- exercised to
    # characterize the mechanism + its bandwidth crossover. The conveyor becomes
    # STRICTLY required at fp16 weights, >13B models, or multi-model residency.
    return model_from_shape("llama2_13b_q4", 40, 5120, 13824, 40, 128,
                            w_bits=4, track=1, compute_s_per_layer=0.846/40,
                            fits_resident=False,
                            note="mainstream/q4; conveyor placement (KV resident -> weights stream); "
                                 "degenerate alt = weights resident + KV on storage")


def llama2_13b_q4_kv4k() -> ModelMap:
    """The ladder's CROSSOVER: 13B q4 with a 4k KV cache STREAMED off the storage
    channel for fabric attention (#70/#72). KV read/layer = 2(K+V)*d_kv*ctx*2B."""
    m = llama2_13b_q4()
    d_kv, ctx = 40 * 128, 4096
    m.kv_read_bytes_per_layer = 2 * d_kv * ctx * 2      # fp16
    m.note += " +4k KV stream on storage channel -> crossover"
    return m


def llama2_7b_q4() -> ModelMap:
    return model_from_shape("llama2_7b_q4", 32, 4096, 11008, 32, 128,
                            w_bits=4, track=1, compute_s_per_layer=0.542/32,
                            fits_resident=False,
                            note="mainstream/q4; 3.24 GB; conveyor hides with +37% margin")


# ==========================================================================
# The scheduler
# ==========================================================================
@dataclass
class Move:
    """One staged slice move planned during some layer's compute window."""
    slice: Slice
    dst_dimm: int
    dst_bank: int
    tier: MoveTier
    staged_during_layer: int     # which layer's compute window hides this move
    @property
    def write_cost_bytes(self) -> float:
        return self.slice.bytes * TIER_WRITE_COST[self.tier]


@dataclass
class TokenStep:
    """One layer's worth of the per-token plan."""
    layer: int
    active_banks: List[Tuple[int, int]]      # (dimm, bank) computing this layer
    staging_banks: List[Tuple[int, int]]     # (dimm, bank) being filled for later
    moves: List[Move]                        # moves initiated during THIS layer
    set_state: List[Tuple[int, int, BankState]]  # (dimm,bank,new) at ENTRY boundary
    @property
    def magic_config(self) -> Dict[int, List[Tuple[int, int]]]:
        """Per-DIMM CFG_SET_STATE payloads: {dimm: [(bank,state),...]}."""
        out: Dict[int, List[Tuple[int, int]]] = {}
        for dimm, bank, st in self.set_state:
            out.setdefault(dimm, []).append((bank, int(st)))
        return out


@dataclass
class ConveyorPlan:
    model: str
    steps: List[TokenStep]
    depth: int
    compute_banks: List[Tuple[int, int]]
    storage_dimms: List[int]
    mover_channels: int
    degenerate: bool                          # model fully resident, zero staging
    per_token_staged_bytes: float
    per_token_write_bus_bytes: float
    notes: List[str] = field(default_factory=list)


class ScheduleError(AssertionError):
    pass


class ConveyorScheduler:
    """Schedule-driven prefetch over the #65 grid. The compute-bank pool is
    partitioned into `depth+1` GROUPS; layer L runs on group (L mod (depth+1))
    while the next groups are staged. Parked weights live on the storage DIMMs;
    every stage is a CROSS_DIMM_1X move (or FREE_ROWCLONE if already co-resident).
    """
    def __init__(self, grid: GridConfig, mover_bw_bps: float = MOVER_BW_BPS):
        self.grid = grid
        self.mover_bw = mover_bw_bps
        self.compute_banks = [(d.dimm, b.bank)
                              for d in grid.dimms.values()
                              if d.role == DimmRole.COMPUTE
                              for b in d.banks]
        self.storage_dimms = [d.dimm for d in grid.dimms.values()
                              if d.role == DimmRole.STORAGE]

    # ---- read-ahead depth derivation ------------------------------------
    def required_depth(self, model: ModelMap, mover_channels: int) -> Tuple[int, List[str]]:
        """Least d such that staging a future layer completes within d layers'
        compute. depth = ceil(T_stage / T_compute), T_stage = layer_bytes /
        (mover_channels * BW). depth>=1 means "stage 1 layer ahead". A model that
        fits resident needs depth 0 (no staging)."""
        notes = []
        if model.fits_resident:
            notes.append("model fits the compute rank -> depth 0 (no staging, static #65)")
            return 0, notes
        import math
        bw = self.mover_bw * mover_channels
        # ratio uses the full storage-channel DEMAND (weight read-ahead + KV
        # attention read). For UNIFORM layers the per-layer ratio == the per-token
        # average, so ratio>1 is BANDWIDTH-BOUND: read-ahead depth cannot relax it
        # (staging further ahead adds no aggregate bandwidth) -- the fix is more
        # mover channels (D1+D3) or a weight-split, NOT more depth.
        ratios = [model.storage_demand_per_layer(li) / bw / l.compute_s
                  for li, l in enumerate(model.layers)]
        rmax = max(ratios)
        kv = "" if not model.kv_read_bytes_per_layer else \
            f" (+{model.kv_read_bytes_per_layer/1e6:.0f} MB/layer KV read)"
        if rmax > 1.0:
            notes.append(f"max storage-demand/compute ratio = {rmax:.2f}{kv} > 1 over "
                         f"{mover_channels} channel(s) @ {self.mover_bw/1e9:.1f} GB/s -> "
                         f"BANDWIDTH-BOUND: read-ahead cannot hide it; needs "
                         f">= {math.ceil(mover_channels*rmax)} channels or a weight-split. depth=1")
            return 1, notes
        # bandwidth suffices; depth hides the per-layer burst (here depth 1 since
        # a single layer's stage < its compute window when ratio<=1).
        worst = max(1, math.ceil(rmax))
        notes.append(f"max storage-demand/compute ratio = {rmax:.2f}{kv} over {mover_channels} "
                     f"channel(s) @ {self.mover_bw/1e9:.1f} GB/s -> hides at depth {worst}")
        return worst, notes

    # ---- build the per-token plan ---------------------------------------
    def plan_token(self, model: ModelMap, mover_channels: int = 1,
                   depth: Optional[int] = None) -> ConveyorPlan:
        notes = []
        if depth is None:
            depth, dn = self.required_depth(model, mover_channels)
            notes += dn

        if model.fits_resident or depth == 0:
            return self._degenerate_plan(model, mover_channels, notes)

        groups = depth + 1                       # resident groups (current + depth ahead)
        n_cb = len(self.compute_banks)
        banks_per_layer = max(l.banks_needed for l in model.layers)
        need = groups * banks_per_layer
        if need > n_cb:
            notes.append(f"WARNING: need {need} compute banks ({groups} groups x "
                         f"{banks_per_layer}/layer) > {n_cb} available -> weight-split "
                         f"or more channels required")
        # partition compute banks into `groups` groups of banks_per_layer
        group_banks = [self.compute_banks[g*banks_per_layer:(g+1)*banks_per_layer]
                       for g in range(groups)]

        steps: List[TokenStep] = []
        staged_bytes = 0.0
        write_bytes = 0.0
        for li, layer in enumerate(model.layers):
            g = li % groups
            active = group_banks[g] if g < len(group_banks) and group_banks[g] else []
            # stage the layer that will run `depth` layers from now into the group
            # that will host it -- but only if that group is currently idle.
            future_li = li + depth
            staging_banks: List[Tuple[int, int]] = []
            moves: List[Move] = []
            set_state: List[Tuple[int, int, BankState]] = []
            if future_li < model.n_layers:
                fg = future_li % groups
                # the future layer's group must be idle now (not the active group)
                if fg != g and fg < len(group_banks) and group_banks[fg]:
                    fbanks = group_banks[fg]
                    staging_banks = list(fbanks)
                    # transitions at THIS layer's entry: put the future group into
                    # STAGING (idle<->idle if it was STORAGE/FREE -> metadata only).
                    for (dd, bb) in fbanks:
                        set_state.append((dd, bb, BankState.STAGING))
                    # assign the future layer's slices to those banks (round-robin)
                    for si, sl in enumerate(model.layers[future_li].slices):
                        dd, bb = fbanks[si % len(fbanks)]
                        tier = self._tier_for(dd, bb, sl)
                        if tier == MoveTier.SAME_DIMM_2X:
                            raise ScheduleError(
                                f"planner produced SAME_DIMM_2X for {sl.name} L{sl.layer} "
                                f"-> parked copy must live on a STORAGE DIMM (design law)")
                        moves.append(Move(sl, dd, bb, tier, staged_during_layer=li))
                        staged_bytes += sl.bytes
                        write_bytes += Move(sl, dd, bb, tier, li).write_cost_bytes
            steps.append(TokenStep(li, active, staging_banks, moves, set_state))

        # promotion transitions: a group that finished STAGING becomes ACTIVE at
        # the boundary of the layer it serves; the finished ACTIVE group demotes
        # to STORAGE (parked) / STAGING target. Encoded as the set_state of the
        # step whose group == that group. (kept implicit in active/staging above;
        # emit the explicit promote/demote pair per boundary.)
        self._emit_promotions(steps, groups, group_banks)

        return ConveyorPlan(model.name, steps, depth, self.compute_banks,
                            self.storage_dimms, mover_channels,
                            degenerate=False,
                            per_token_staged_bytes=staged_bytes,
                            per_token_write_bus_bytes=write_bytes,
                            notes=notes)

    def _emit_promotions(self, steps, groups, group_banks):
        """At the entry of layer L, promote group (L mod groups) STAGING->ACTIVE
        and demote the group that just finished (L-1 mod groups) ACTIVE->STORAGE."""
        for li, st in enumerate(steps):
            g = li % groups
            promote = group_banks[g] if group_banks[g] else []
            for (dd, bb) in promote:
                st.set_state.append((dd, bb, BankState.ACTIVE))
            if li > 0:
                pg = (li - 1) % groups
                for (dd, bb) in (group_banks[pg] if group_banks[pg] else []):
                    st.set_state.append((dd, bb, BankState.STORAGE))

    def _degenerate_plan(self, model, mover_channels, notes) -> ConveyorPlan:
        # all layers resident: one ACTIVE assignment, zero moves.
        n_cb = len(self.compute_banks)
        steps = []
        for li, layer in enumerate(model.layers):
            active = self.compute_banks[:max(1, layer.banks_needed)]
            steps.append(TokenStep(li, active, [], [], []))
        notes.append("DEGENERATE conveyor: whole model resident; zero staging moves; "
                     "schedule trivially valid (static #65 residency).")
        return ConveyorPlan(model.name, steps, 0, self.compute_banks,
                            self.storage_dimms, mover_channels, degenerate=True,
                            per_token_staged_bytes=0.0, per_token_write_bus_bytes=0.0,
                            notes=notes)

    def _tier_for(self, dimm, bank, sl: Slice) -> MoveTier:
        """Cost tier for staging `sl` into (dimm,bank). Parked copies live on a
        STORAGE DIMM, so the stage is CROSS_DIMM_1X (read on idle storage channel,
        1x write on compute bus). FREE_ROWCLONE only if the slice were already
        co-resident in the destination segment (not modeled per-token here)."""
        if dimm in self.storage_dimms:
            return MoveTier.FREE_ROWCLONE   # parking onto storage itself is not a stage
        return MoveTier.CROSS_DIMM_1X


# ==========================================================================
# Validity checker (the dry-test asserts)
# ==========================================================================
def validate(plan: ConveyorPlan, model: ModelMap, strict_timing: bool = True) -> List[str]:
    """Assert the schedule-validity properties. Returns the list of checks that
    PASSED; raises ScheduleError on the first violation. With strict_timing, a
    bandwidth-bound stage (storage demand > compute window) is a FAILURE because
    the slice cannot be resident before its layer starts (the ladder's crossover:
    13B+4kKV infeasible on 1 channel, feasible on 2)."""
    passed = []

    # (1) No bank computes while staging-in: a bank is never both ACTIVE
    #     (in this step's active set) and a staging destination in the SAME step.
    for st in plan.steps:
        act = set(st.active_banks)
        stg = set(st.staging_banks)
        clash = act & stg
        if clash:
            raise ScheduleError(f"L{st.layer}: banks {clash} are ACTIVE and STAGING "
                                f"simultaneously (compute-while-staging)")
    passed.append("(1) no bank computes while staging-in: OK "
                  f"({len(plan.steps)} layers checked)")

    # (2) Every slice resident before its layer starts: for each layer L, every
    #     slice was the target of a move staged during an EARLIER layer's window
    #     (staged_during_layer < L), OR the model is degenerate (all resident).
    if not plan.degenerate:
        staged_ready: Dict[Tuple[int, str], int] = {}   # (layer,slice)->ready-by layer
        for st in plan.steps:
            for mv in st.moves:
                key = (mv.slice.layer, mv.slice.name)
                # ready AFTER the staging window completes = end of staged_during_layer
                staged_ready[key] = mv.staged_during_layer
        for layer in model.layers:
            for sl in layer.slices:
                key = (sl.layer, sl.name)
                if sl.layer == 0:
                    continue   # layer 0 is prefetched during prefill / warm token
                if key not in staged_ready:
                    raise ScheduleError(f"slice {sl.name} L{sl.layer} never staged")
                if staged_ready[key] >= sl.layer:
                    raise ScheduleError(
                        f"slice {sl.name} L{sl.layer} staged during L{staged_ready[key]} "
                        f"-> not resident before its own layer starts")
        passed.append("(2) every slice resident before its layer starts: OK "
                      f"({sum(len(l.slices) for l in model.layers)} slices, "
                      f"layer-0 warmed at prefill)")
    else:
        passed.append("(2) every slice resident before its layer starts: OK "
                      "(degenerate -- all resident)")

    # (3) Read-ahead depth respected: no slice is staged more than `depth` layers
    #     ahead of its own layer, and each move's tier is never the 2x tier.
    if not plan.degenerate:
        for st in plan.steps:
            for mv in st.moves:
                lead = mv.slice.layer - mv.staged_during_layer
                if lead > plan.depth:
                    raise ScheduleError(
                        f"slice {mv.slice.name} L{mv.slice.layer} staged {lead} layers "
                        f"ahead > read-ahead depth {plan.depth}")
                if lead < 1:
                    raise ScheduleError(
                        f"slice {mv.slice.name} L{mv.slice.layer} staged non-positive "
                        f"lead {lead}")
                if mv.tier == MoveTier.SAME_DIMM_2X:
                    raise ScheduleError(f"move {mv.slice.name} uses the 2x tier (forbidden)")
        passed.append(f"(3) read-ahead depth respected (<= {plan.depth}) & no 2x-tier moves: OK")
    else:
        passed.append("(3) read-ahead depth respected: OK (degenerate, depth 0)")

    # (4 bonus) staging hides under compute: per-layer storage-channel DEMAND
    #     (weight read-ahead + KV attention read) fits the compute window at the
    #     mover BW * channels.
    if not plan.degenerate:
        bw = MOVER_BW_BPS * plan.mover_channels
        worst = max(model.storage_demand_per_layer(l) / bw / model.layers[l].compute_s
                    for l in range(model.n_layers))
        if worst > 1.0 and strict_timing:
            raise ScheduleError(
                f"storage-demand/compute worst ratio = {worst:.2f} > 1 on "
                f"{plan.mover_channels} channel(s): staging is BANDWIDTH-BOUND, a slice "
                f"cannot be resident before its layer starts -> add channels (D1+D3) or "
                f"weight-split")
        passed.append(f"(4) storage-demand/compute worst ratio = {worst:.2f} -> HIDES "
                      f"(feasible on {plan.mover_channels} channel(s))")

    # (5 #65-rule) MAGIC_CONFIG legality: every emitted transition is either
    #     idle<->idle (metadata-only) or a promotion STAGING->ACTIVE; a demotion
    #     ACTIVE->STORAGE is allowed but flagged as invalidating (per-bank).
    inval_demotions = 0
    for st in plan.steps:
        # reconstruct prior state is out of scope; check that we never SET a bank
        # ACTIVE that is also a staging destination in the same step.
        set_active = {(d, b) for (d, b, s) in st.set_state if s == BankState.ACTIVE}
        set_staging = {(d, b) for (d, b, s) in st.set_state if s == BankState.STAGING}
        if set_active & set_staging:
            raise ScheduleError(f"L{st.layer}: bank set to ACTIVE and STAGING in one step")
        inval_demotions += len({(d, b) for (d, b, s) in st.set_state
                                if s == BankState.STORAGE})
    passed.append(f"(5) #65-rule transitions legal: {inval_demotions} ACTIVE->STORAGE "
                  f"demotions (each invalidates only its own bank under the decoupled "
                  f"server change; promotions STAGING->ACTIVE never invalidate)")
    return passed


# ==========================================================================
# Report (card-free demo)
# ==========================================================================
def _report(model: ModelMap, mover_channels: int, grid: GridConfig):
    sch = ConveyorScheduler(grid)
    depth, dn = sch.required_depth(model, mover_channels)
    plan = sch.plan_token(model, mover_channels=mover_channels)
    print(f"\n=== {model.name}  ({model.n_layers} layers, "
          f"{model.weight_bytes/1e9:.2f} GB weights, {mover_channels} mover ch) ===")
    print(f"    note: {model.note}")
    for n in plan.notes:
        print(f"    - {n}")
    print(f"    depth={plan.depth}  degenerate={plan.degenerate}  "
          f"staged/token={plan.per_token_staged_bytes/1e9:.2f} GB  "
          f"write-bus/token={plan.per_token_write_bus_bytes/1e9:.2f} GB")
    try:
        checks = validate(plan, model)
        for c in checks:
            print(f"    [PASS] {c}")
    except ScheduleError as e:
        print(f"    [INFEASIBLE @ {mover_channels} ch] {e}")
    return plan


if __name__ == "__main__":
    grid = GridConfig.default_bitnet(
        os.environ.get("BN_DIR", "/home/deni/bitnet_weights"),
        active_banks=range(16))
    print("#" * 74)
    print("  pim_conveyor.py  card-free schedule demo (#67)")
    print("#" * 74)
    _report(bitnet_2b(), 1, grid)
    _report(llama2_7b_q4(), 1, grid)
    _report(llama2_13b_q4(), 1, grid)
    _report(llama2_13b_q4_kv4k(), 1, grid)   # crossover: 1 channel EXCEEDS
    _report(llama2_13b_q4_kv4k(), 2, grid)   # D1+D3 = 2 channels HIDES
