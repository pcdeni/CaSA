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

DIMM MODEL:
  A channel's ROLE is a deployment choice, not a property of the part.
  "compute" runs the charge-sharing gates and holds resident weights and
  constants; "storage" is addressable capacity only — RowClone and ordinary
  read/write. A MAJ3-capable module can be given either role; a module with no
  usable MAJ3 yield can only be given storage. Which module sits in which
  socket, and the role each is given, live in calibration/DIMM_POPULATION.conf
  and are resolved by python/dimm_population.py — never assumed here.
One bitnet-proj-server process drives ONE DIMM (bender == dimm). Multi-DIMM
orchestration (routing compute work to the compute channels, staging on the
storage ones) is the Python orchestrator's job across several PimServer
processes — the server carries the dimm/role as config so nothing is
hardcoded; cross-DIMM data movement belongs to the prefetch scheduler.
"""
from dataclasses import dataclass, field
import os
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
    COMPUTE = "compute"   # runs the charge-sharing gates; holds resident state
    STORAGE = "storage"   # RowClone store / staging area only


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
    def default_bitnet(bn_dir: str = None,
                       compute_dimms: Iterable[int] = None,
                       storage_dimms: Iterable[int] = None,
                       active_banks: Iterable[int] = (0, 1, 2, 3)) -> "GridConfig":
        """Seed the grid from the population file.

        Roles and per-channel fixtures both come from
        calibration/DIMM_POPULATION.conf via python/dimm_population.py, so a
        module swap is a one-file edit and no channel can inherit a fixture
        set measured on a part that is no longer installed. Pass
        compute_dimms / storage_dimms explicitly to override the file's roles,
        and bn_dir to override where the fixtures are looked up.

        ACTIVE banks default to 0-3; the rest are declared STORAGE so the
        prefetch scheduler can stage into them without a restart. Every value
        here is a SEED default — the host may override any of it, including
        putting a compute-grade channel into the STORAGE role."""
        import dimm_population

        if bn_dir is not None:
            os.environ["PIM_BN"] = bn_dir
        if compute_dimms is None or storage_dimms is None:
            roles = dimm_population.lane_roles().split(",")
            file_compute = tuple(i for i, r in enumerate(roles)
                                 if r.strip().lower().startswith("c"))
            file_storage = tuple(i for i, r in enumerate(roles)
                                 if not r.strip().lower().startswith("c"))
            if compute_dimms is None:
                compute_dimms = file_compute
            if storage_dimms is None:
                storage_dimms = file_storage

        def _fixtures(d):
            t = dimm_population.trio(d, check_exists=False)
            return t["calib"], t["pool"]

        g = GridConfig()
        for d in compute_dimms:
            calib, pool = _fixtures(d)
            banks = []
            for b in range(BANKS_PER_DIMM):
                st = BankState.ACTIVE if b in set(active_banks) else BankState.STORAGE
                banks.append(BankSpec(dimm=d, bank=b, state=st))
            g.dimms[d] = DimmSpec(dimm=d, role=DimmRole.COMPUTE,
                                  calib_path=calib, pool_pattern=pool, banks=banks)
        for d in storage_dimms:
            calib, pool = _fixtures(d)
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
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    g = GridConfig.default_bitnet(os.environ.get("BN_DIR"),
                                  active_banks=range(16))
    print("DIMMs:", {d: s.role for d, s in g.dimms.items()})
    print("DIMM0 bank_arg:", g.dimms[0].bank_arg())
    cap = g.working_set_capacity()
    print("capacity (all 16 banks active on every compute channel):", cap)
    # wire round-trip
    specs = g.dimms[2].banks
    enc = GridConfig.encode_reconfig(specs)
    print("encode_reconfig bytes:", len(enc), "=> 12 +", len(specs), "* 20 =",
          12 + len(specs) * 20)
    # fake a server response table for decode
    fake = struct.pack('<II', 0, 1) + struct.pack('<iiIIII', 2, 5, 2, 197, 45312, 45952)
    print("decode_table:", GridConfig.decode_table(fake))
