"""pim_grid_config.py — host-side CONFIG SCHEMA for the 4-DIMM x 16-bank grid.

[#65 2026-08-04, bank16_config_2026_08_04]  DESIGN LAW (user, verbatim):
"NO HARD CODED VALUES, EVERYTHING CAN BE CONFIGURED BY THE HOST ON POWER UP,
AND CAN BE CHANGED LATER."  And (user, 2026-08-04): "YOU HAVE 4 DIMMS, EACH
WITH 16 BANKS. NOT ALL OF THE 64 ARE BEING USED AT THE SAME TIME, IF IT DOESN'T
FIT, MOVE THE DATA BEFORE IT IS NEEDED WHILE THE OTHER BANK/DIMM IS UTILIZED."

This module is the pure-data representation of that grid: which DIMMs exist,
their ROLE (compute / storage), and per-bank residency STATE + geometry. It is
the working-set / conveyor model's host-side home. It has NO torch / FPGA deps,
so it is testable card-free.

WHO DRIVES IT:
  * This pass (#65) implements the plumbing: the schema, its serialization to
    the server's MAGIC_CONFIG wire format, and the PimServer.reconfigure /
    set_bank_state transport (in pim_linear.py).
  * The PREFETCH SCHEDULER (task #67) is what actually decides transitions and
    moves data ("move it before it is needed while the other bank/DIMM is
    utilized"). It consumes this schema; it is NOT implemented here.

DIMM MODEL (physical fact, see memory dimm_population_2026_07):
  * DIMMs 0 & 2 = SK hynix, MAJ3-capable  -> role "compute".
  * DIMMs 1 & 3 = Micron, MAJ3-dead       -> role "storage" (valid RowClone
    store / staging area even though they cannot compute MAJ3).
One bitnet-proj-server process today drives ONE DIMM (bender == dimm). Multi-
DIMM orchestration (routing compute to 0/2, staging on 1/3) is the python
orchestrator's job across several PimServer processes — the server carries the
dimm/role as config so nothing is hardcoded, but cross-DIMM data movement is
#67.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Iterable, Optional
import struct


# ---- wire constants (MUST match test_bitnet_server_bank16.cpp) ----
MAGIC_CONFIG = 0xB17EF00A
CFG_QUERY, CFG_RECONFIG, CFG_SET_STATE = 1, 2, 3


class BankState(IntEnum):
    ACTIVE = 0    # in the compute working set; serves matmul now
    STAGING = 1   # being preloaded with the next stage while idle
    STORAGE = 2   # holds a parked slice; not in the active set
    FREE = 3      # empty; available for allocation


class DimmRole:
    COMPUTE = "compute"   # DIMMs 0, 2 (SK hynix, MAJ3-capable)
    STORAGE = "storage"   # DIMMs 1, 3 (Micron, MAJ3-dead but valid store)


# Banks per DIMM on this hardware (BCU1525 QUAD). Not a hardcode in the sense
# the design law forbids — it is a queryable hardware fact; overridable here.
BANKS_PER_DIMM = 16


@dataclass
class BankSpec:
    dimm: int
    bank: int
    state: BankState = BankState.ACTIVE
    win_start: int = 0     # 0/0 => inherit the server's global PIM_SUB_START/END
    win_end: int = 0

    def wire(self) -> bytes:
        # [i32 dimm][i32 bank][u32 state][u32 ws][u32 we]  (20 B) — RECONFIG input
        # order (pool_size is output-only, absent here; present in the response).
        return struct.pack('<iiIII', self.dimm, self.bank,
                           int(self.state), self.win_start, self.win_end)


@dataclass
class DimmSpec:
    dimm: int                         # == bender id
    role: str = DimmRole.COMPUTE
    calib_path: str = ""
    pool_pattern: str = ""            # PIM_POOL_LIST_FILE pattern with {bank}
    banks: List[BankSpec] = field(default_factory=list)

    def active_banks(self) -> List[int]:
        return [b.bank for b in self.banks if b.state == BankState.ACTIVE]

    def bank_arg(self) -> str:
        """The comma-separated bank string for this DIMM's server argv[3]
        (only banks that should exist at startup — ACTIVE + STAGING + STORAGE;
        FREE banks are simply omitted until allocated)."""
        return ",".join(str(b.bank) for b in self.banks
                        if b.state != BankState.FREE)


@dataclass
class GridConfig:
    dimms: Dict[int, DimmSpec] = field(default_factory=dict)

    # ---- construction ----
    @staticmethod
    def default_bitnet(bn_dir: str,
                       compute_dimms: Iterable[int] = (0, 2),
                       storage_dimms: Iterable[int] = (1, 3),
                       active_banks: Iterable[int] = (0, 1, 2, 3)) -> "GridConfig":
        """Seed the standard grid: compute DIMMs 0/2 (SK hynix), storage DIMMs
        1/3 (Micron). ACTIVE banks default to 0-3 (today's working set); the
        remaining banks are declared STORAGE so the conveyor (#67) can stage
        into them without a restart. Every value here is a SEED default — the
        host may override any of it."""
        g = GridConfig()
        for d in compute_dimms:
            calib = f"{bn_dir}/calib_dimm{d}.txt"
            pool = f"{bn_dir}/pool_layout_dimm{d}_cloneok_bank{{bank}}.txt"
            banks = []
            for b in range(BANKS_PER_DIMM):
                st = BankState.ACTIVE if b in set(active_banks) else BankState.STORAGE
                banks.append(BankSpec(dimm=d, bank=b, state=st))
            g.dimms[d] = DimmSpec(dimm=d, role=DimmRole.COMPUTE,
                                  calib_path=calib, pool_pattern=pool, banks=banks)
        for d in storage_dimms:
            calib = f"{bn_dir}/calib_dimm{d}.txt"
            pool = f"{bn_dir}/pool_layout_dimm{d}_cloneok_bank{{bank}}.txt"
            banks = [BankSpec(dimm=d, bank=b, state=BankState.STORAGE)
                     for b in range(BANKS_PER_DIMM)]
            g.dimms[d] = DimmSpec(dimm=d, role=DimmRole.STORAGE,
                                  calib_path=calib, pool_pattern=pool, banks=banks)
        return g

    # ---- serialization to the server's MAGIC_CONFIG wire format ----
    @staticmethod
    def encode_query() -> bytes:
        return struct.pack('<II', MAGIC_CONFIG, CFG_QUERY)

    @staticmethod
    def encode_reconfig(bank_specs: Iterable[BankSpec]) -> bytes:
        specs = list(bank_specs)
        out = struct.pack('<III', MAGIC_CONFIG, CFG_RECONFIG, len(specs))
        for s in specs:
            out += s.wire()
        return out

    @staticmethod
    def encode_set_state(changes: Iterable) -> bytes:
        """changes = iterable of (bank, state)."""
        chg = list(changes)
        out = struct.pack('<III', MAGIC_CONFIG, CFG_SET_STATE, len(chg))
        for bank, state in chg:
            out += struct.pack('<iI', bank, int(state))
        return out

    @staticmethod
    def decode_table(resp: bytes):
        """Decode a MAGIC_CONFIG response ([u32 status][u32 n] + n*24)."""
        status, n = struct.unpack('<II', resp[:8])
        rows = []
        for i in range(n):
            off = 8 + i * 24
            dimm, bank, state, pool, ws, we = struct.unpack(
                '<iiIIII', resp[off:off + 24])
            rows.append({'dimm': dimm, 'bank': bank,
                         'state': BankState(state if state <= 3 else 0),
                         'pool_size': pool, 'win_start': ws, 'win_end': we})
        return status, rows

    # ---- capacity helper ----
    def working_set_capacity(self, pool_rows_per_bank: int = 197,
                             rows_per_subhandle: int = 9) -> dict:
        """Rough residency capacity of the ACTIVE compute banks. Defaults from
        the dimm2 gate (197 screened pool rows/bank; a sub-handle reserves
        n_rounds rows, ~9 for a d_in=128 shape minus the V2 scratch tail)."""
        active = sum(len(d.active_banks()) for d in self.dimms.values()
                     if d.role == DimmRole.COMPUTE)
        rows = active * pool_rows_per_bank
        return {'active_compute_banks': active,
                'resident_rows': rows,
                'approx_subhandles': rows // max(1, rows_per_subhandle)}


if __name__ == "__main__":
    # Card-free smoke: build the default grid, encode/decode a round-trip.
    import os
    bn = os.environ.get("BN_DIR",
        "/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet")
    g = GridConfig.default_bitnet(bn, active_banks=range(16))
    print("DIMMs:", {d: s.role for d, s in g.dimms.items()})
    print("DIMM0 bank_arg:", g.dimms[0].bank_arg())
    cap = g.working_set_capacity()
    print("capacity (all 16 active on 0+2):", cap)
    # wire round-trip
    specs = g.dimms[2].banks
    enc = GridConfig.encode_reconfig(specs)
    print("encode_reconfig bytes:", len(enc), "=> 12 +", len(specs), "* 20 =",
          12 + len(specs) * 20)
    # fake a server response table for decode
    fake = struct.pack('<II', 0, 1) + struct.pack('<iiIIII', 2, 5, 2, 197, 45312, 45952)
    print("decode_table:", GridConfig.decode_table(fake))
