#!/usr/bin/env python3
"""#67 CONVEYOR on-card gate (DESIGN.md promote/demote choreography).

Two compute GROUPS on DIMM 2:  A = banks {0,1,2,3},  B = banks {4,5,6,7}.
Handles are LOADed pinned to a group's bank SUBSET (PIM_HANDLE_SUBSET=1 + the new
trailing subset field). Then, in the SAME server process, we ping-pong the groups
with CFG_SET_STATE and assert, after each move:

  (a) the demoted group's handle is INVALIDATED (server log: "left ACTIVE: dropped
      N resident handle(s)") -- surgical, NOT the old coarse handles.clear();
  (b) a handle resident ONLY on the SURVIVING group still serves BYTE-EXACT vs the
      CPU oracle with NO reload (same process);
  (c) a freshly PROMOTED+reloaded group serves byte-exact;
  (d) >= 3 ping-pong cycles, zero corruption.

Ends with a DESTRUCTIVE ENOENT probe: serve a demoted handle -> the server errors
("unknown handle") / tears the session (proves it is not stale-served).
"""
import os, sys, struct, time
import numpy as np
sys.path.insert(0, "/home/deni/Claude/numerics_gate")
sys.path.insert(0, "/home/deni/bitnet_weights")
import v2_oracle as O
import pim_linear

DIN    = int(os.environ.get("CG_DIN", "128"))
NREAL  = 512
BENDER = 2
BANK   = "0,1,2,3,4,5,6,7"
CALIB  = os.environ["CG_CALIB"]
SERVER = os.environ["PIM_SERVER_BIN"]
SRVLOG = "/tmp/pim_server_b2_0_1_2_3_4_5_6_7.log"
GROUP_A = [0, 1, 2, 3]
GROUP_B = [4, 5, 6, 7]
ST_ACTIVE  = pim_linear.PimServer.ST_ACTIVE
ST_STORAGE = pim_linear.PimServer.ST_STORAGE
NCH = DIN // 32

fails = 0
def ok(cond, msg):
    global fails
    print(("  PASS " if cond else "  FAIL ") + msg, flush=True)
    if not cond: fails += 1

H = {}   # handle_id -> problem dict
def make_handle(hid, seed):
    W, x = O.make_problem(DIN, NREAL, seed, single=False)
    pos, neg = O.masks_from_W(W)
    xbp = O.pack_xbp(x, NCH)
    yref = O._internal_consistency(pos, neg, xbp, O.BITPLANE_FACTORS, W, x, NREAL)
    H[hid] = dict(seed=seed, pos=pos, neg=neg, xbp=xbp, yref=yref)

def build_load_subset(hid, bank_ids):
    base = O.build_load_request(hid, H[hid]["pos"], H[hid]["neg"])
    sub = struct.pack("<I", len(bank_ids)) + b"".join(struct.pack("<i", b) for b in bank_ids)
    return base + sub

def load(server, hid, bank_ids):
    st = struct.unpack("<I", server.request(build_load_subset(hid, bank_ids),
                                            expect_resp_len=4))[0]
    ok(st == 0, f"LOAD handle {hid} pinned to banks {bank_ids} (ack={st})")

def serve_exact(server, hid, note):
    body = O.build_mm3d_request(hid, H[hid]["xbp"], O.BITPLANE_FACTORS, NCH, calib_idx=0)
    resp = server.request(body, expect_resp_len=O.D_OUT * 4)
    y = np.frombuffer(resp, dtype="<i4").astype(np.int64)
    live = bool(np.any(y != 0))
    mism = O.compare(y[:NREAL], H[hid]["yref"]) if live else np.arange(NREAL)
    exact = NREAL - mism.size
    ok(exact == NREAL, f"{note}: handle {hid} exact={exact}/{NREAL}" + ("" if live else " ALL-ZERO"))

def dropped_since(mark):
    try: data = open(SRVLOG, "rb").read()
    except FileNotFoundError: return 0, mark
    n = 0
    for line in data[mark:].decode("utf-8", "replace").splitlines():
        if "left ACTIVE: dropped" in line:
            try: n += int(line.split("dropped")[1].split("resident")[0])
            except Exception: pass
    return n, len(data)

# --------------------------------------------------------------------------
print(f"###### CONVEYOR GATE  srv={os.path.basename(SERVER)}  DIN={DIN} ######", flush=True)
try: os.remove(SRVLOG)
except FileNotFoundError: pass
server = pim_linear.PimServer.shared(BENDER, BANK, CALIB, SERVER, extra_env=None)
mark = 0

# setup: H1 on A, H2 on B; both serve (subset serving works both groups)
make_handle(1, 501); make_handle(2, 611)
load(server, 1, GROUP_A)
load(server, 2, GROUP_B)
serve_exact(server, 1, "[setup] group-A subset serve")
serve_exact(server, 2, "[setup] group-B subset serve")

active = {1: GROUP_A, 2: GROUP_B}    # live handle_id -> its (ACTIVE) group banks
gone, survivor = 1, 2                 # demote A first; B survives
next_hid = 3
last_demoted = None
for cyc in range(1, 5):               # 4 boundaries (>=3 required)
    dbanks = active[gone]
    print(f"\n=== cycle {cyc}: demote handle {gone} (banks {dbanks}); keep handle {survivor} ===", flush=True)
    server.set_bank_state([(b, ST_STORAGE) for b in dbanks]); time.sleep(0.25)
    ndrop, mark = dropped_since(mark)
    ok(ndrop == 1, f"(a) demote invalidated EXACTLY 1 resident handle (server log dropped={ndrop})")
    serve_exact(server, survivor, f"[cyc{cyc}] (b) SURVIVOR serves after demote, NO reload")
    last_demoted = gone
    del active[gone]
    # promote the just-demoted group back, load a FRESH handle on it, serve it
    server.set_bank_state([(b, ST_ACTIVE) for b in dbanks])
    hid = next_hid; next_hid += 1
    make_handle(hid, 700 + hid)
    load(server, hid, dbanks)
    serve_exact(server, hid, f"[cyc{cyc}] (c) PROMOTED+reloaded group serves byte-exact")
    active[hid] = dbanks
    gone, survivor = survivor, hid    # roles swap

# destructive ENOENT probe on a known-demoted handle
print(f"\n=== ENOENT probe: serve demoted handle {last_demoted} (must error, not stale-serve) ===", flush=True)
torn = False
try:
    body = O.build_mm3d_request(last_demoted, H[last_demoted]["xbp"], O.BITPLANE_FACTORS, NCH, calib_idx=0)
    server.request(body, expect_resp_len=O.D_OUT * 4)
    print("  (server returned a response -- checking log for unknown-handle)", flush=True)
except Exception as e:
    torn = True
    print(f"  server refused/torn as expected: {type(e).__name__}", flush=True)
time.sleep(0.2)
srv = ""
try: srv = open(SRVLOG, "rb").read().decode("utf-8", "replace")
except Exception: pass
unknown = f"unknown handle {last_demoted}" in srv
ok(torn or unknown, f"(a) demoted handle {last_demoted} is ENOENT, not stale-served "
                    f"[torn={torn} log_unknown={unknown}]")

print(f"\nCONVEYOR_GATE_DONE fails={fails}", flush=True)
sys.exit(1 if fails else 0)
