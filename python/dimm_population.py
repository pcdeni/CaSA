#!/usr/bin/env python3
"""Per-channel DIMM configuration, resolved at run time.

Every tool that drives the card needs the same three fixtures per channel:

    calibration file  +  per-bank pool layout  +  sub-array row window

and the channel's lane role (compute or storage). Both come from ONE file,
`calibration/DIMM_POPULATION.conf`, which is maintained only when the physical
modules change. Nothing else may hardcode them.

WHY IT IS A FILE AND NOT A TABLE IN THE CODE
--------------------------------------------
A fixture set names a specific die. Point a live die at a retired module's
calibration and nothing errors: the majority gates still fire, the readback
still parses, and the answer is quietly wrong on a fraction of cells — well
inside what a correlation-based numerics gate calls a pass. Measured on one
channel, minutes apart: the retired trio gave 265/512 bit-exact at 13.6% worst
relative error and passed the gate; the live trio on the same channel gave
512/512 bit-exact.

So the trio is configuration, the configuration lives in one file, and this
module REFUSES a fixture set belonging to a module that is not in the
population (`PIM_TRIO_ALLOW_RETIRED=1` to replay one deliberately). The
retired fixtures themselves are kept — they are characterization data — they
are just not reachable by accident.

ENVIRONMENT
-----------
    PIM_POPULATION_FILE   population file           (default: ../calibration/DIMM_POPULATION.conf)
    PIM_BN                directory holding fixtures (default: ../calibration)
    PIM_LANE_ROLES        e.g. "c,c,s,s"            (wins over the population file)
Global trio overrides, applying to every channel:
    PIM_TRIO_CALIB  PIM_TRIO_POOL  PIM_TRIO_SUB_START  PIM_TRIO_SUB_END  PIM_TRIO_COLMASK
Per channel N, winning over the global form:
    PIM_TRIO<N>_CALIB  PIM_TRIO<N>_POOL  PIM_TRIO<N>_SUB_START
    PIM_TRIO<N>_SUB_END  PIM_TRIO<N>_COLMASK
Deliberate archival replay:
    PIM_TRIO_ALLOW_RETIRED=1

CLI
---
    python3 dimm_population.py [--sh] [channel ...]

Prints the resolved trio per channel (default 0..3), or shell assignments with
`--sh`. Use it as a dry-run self-check before a session; it touches no hardware.
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

POP_FILE_DEFAULT = os.path.join(_REPO, "calibration", "DIMM_POPULATION.conf")
BN_DEFAULT = os.path.join(_REPO, "calibration")

N_CHANNELS = 4

# Fixtures of modules that are not in the population described by the
# population file. Matched on BASENAME, against the name AS WRITTEN, so a pool
# pattern still carrying its {bank} token matches exactly as the substituted
# form does. Nothing here is deleted: these files stay in calibration/ and
# docs/data/ as the characterization record of the modules they came from.
RETIRED_PATTERNS = (
    r"^calib_dimm0(_scale)?\.txt$",
    r"^calib_dimm[13]\.txt$",
    r"^pool_layout_dimm0_.*\.txt$",
    r"^pool_layout_dimm[13]_.*\.txt$",
    r"^fused_colmask_dimm0_.*\.txt$",
)
RETIRED_WINDOWS = {
    (38400, 39040): "the retired reference module's subarray-61 window",
}


class RetiredFixtureError(RuntimeError):
    pass


def _pop_file():
    return os.environ.get("PIM_POPULATION_FILE", POP_FILE_DEFAULT)


def bn_dir():
    """Directory the fixture files live in."""
    return os.environ.get("PIM_BN", BN_DEFAULT)


_POP_CACHE = {}


def population():
    """Parse the population file (KEY=VALUE, '#' comments). Loud if absent."""
    path = _pop_file()
    if path in _POP_CACHE:
        return _POP_CACHE[path]
    if not os.path.exists(path):
        raise RuntimeError(
            "dimm_population: population file MISSING: %s\n"
            "  It is the single source of truth for the installed modules "
            "(roles + fixture trio)\n  and is maintained only when the DIMMs "
            "change. Point PIM_POPULATION_FILE\n  elsewhere to use another "
            "one." % path)
    pop = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                v = v[1:-1]
            pop[k.strip()] = v
    _POP_CACHE[path] = pop
    return pop


def lane_roles():
    """Role string, one letter per channel, e.g. 'c,c,c,c'.

    An explicit PIM_LANE_ROLES wins — that is for experiments. The server
    generates resident constants only on compute lanes, so a wrong role table
    is not a labelling mistake: it silently re-reads constant material over the
    bus on every operation.
    """
    env = os.environ.get("PIM_LANE_ROLES")
    if env:
        return env
    pop = population()
    out = []
    for n in range(N_CHANNELS):
        role = pop.get("POP_SLOT%d_ROLE" % n, "").strip().lower()
        if role not in ("compute", "storage"):
            raise RuntimeError(
                "dimm_population: POP_SLOT%d_ROLE missing or invalid in %s "
                "(need compute|storage, got %r)" % (n, _pop_file(), role))
        out.append("c" if role == "compute" else "s")
    return ",".join(out)


def _defaults():
    pop = population()
    missing = [k for k in ("POP_TRIO_CALIB", "POP_TRIO_POOL",
                           "POP_TRIO_SUB_START", "POP_TRIO_SUB_END")
               if k not in pop]
    if missing:
        raise RuntimeError("dimm_population: %s missing %s"
                           % (_pop_file(), ", ".join(missing)))
    return {
        "calib": pop["POP_TRIO_CALIB"],
        "pool": pop["POP_TRIO_POOL"],
        "sub_start": int(pop["POP_TRIO_SUB_START"]),
        "sub_end": int(pop["POP_TRIO_SUB_END"]),
        "colmask": pop.get("POP_TRIO_COLMASK") or None,
    }


def _abs(name):
    if name is None:
        return None
    return name if os.path.isabs(name) else os.path.join(bn_dir(), name)


def _is_retired(path):
    if path is None:
        return False
    base = os.path.basename(str(path))
    return any(re.match(p, base) for p in RETIRED_PATTERNS)


def _env(channel, key, default):
    v = os.environ.get("PIM_TRIO%d_%s" % (channel, key))
    if v is None:
        v = os.environ.get("PIM_TRIO_%s" % key)
    return default if v is None or v == "" else v


def trio(channel, check_exists=True):
    """Resolve the (calib, pool, window, colmask) quadruple for one channel.

    Returns absolute paths. Raises RetiredFixtureError if the resolved set
    names a module outside the population, unless PIM_TRIO_ALLOW_RETIRED=1.
    """
    c = int(channel)
    d = _defaults()
    calib = _abs(_env(c, "CALIB", d["calib"]))
    pool = _abs(_env(c, "POOL", d["pool"]))
    colmask_raw = _env(c, "COLMASK", d["colmask"])
    colmask = _abs(colmask_raw) if colmask_raw else None
    sub_start = int(_env(c, "SUB_START", d["sub_start"]))
    sub_end = int(_env(c, "SUB_END", d["sub_end"]))

    allow = os.environ.get("PIM_TRIO_ALLOW_RETIRED", "0") == "1"
    bad = [os.path.basename(str(p))
           for p in (calib, pool, colmask) if _is_retired(p)]
    if (sub_start, sub_end) in RETIRED_WINDOWS:
        bad.append("window [%d,%d) (%s)"
                   % (sub_start, sub_end, RETIRED_WINDOWS[(sub_start, sub_end)]))
    if bad and not allow:
        raise RetiredFixtureError(
            "REFUSING a retired fixture set for channel %d: %s\n"
            "  Those fixtures were measured on a module that is not in the\n"
            "  population described by %s. Running them against a live die\n"
            "  does not error — it computes on a foreign die's calibration.\n"
            "  Set PIM_TRIO_ALLOW_RETIRED=1 only for deliberate replay."
            % (c, ", ".join(bad), _pop_file()))

    if check_exists:
        missing = [] if os.path.exists(calib) else [calib]
        pool0 = pool.replace("{bank}", "0")
        if not os.path.exists(pool0):
            missing.append(pool0)
        if colmask and not os.path.exists(colmask.replace("{bank}", "0")):
            missing.append(colmask)
        if missing:
            raise FileNotFoundError(
                "dimm_population: missing fixture(s) for channel %d: %s\n"
                "  Fixtures are looked up in %s (set PIM_BN to change)."
                % (c, ", ".join(missing), bn_dir()))

    return {"bender": c, "calib": calib, "pool": pool,
            "sub_start": sub_start, "sub_end": sub_end, "colmask": colmask,
            "retired_override": bool(bad)}


def dimm_spec(channel, bank=None):
    """The dict shape pim_linear.pim_substitute expects for one DIMM."""
    t = trio(channel)
    spec = {"bender": t["bender"], "calib": t["calib"],
            "pool_layout": t["pool"],
            "sub_start": t["sub_start"], "sub_end": t["sub_end"]}
    if t["colmask"]:
        spec["fused_colmask"] = t["colmask"]
    if bank is not None:
        spec["bank"] = bank
    return spec


def describe(channel):
    t = trio(channel, check_exists=False)
    return ("channel %d: calib=%s pool=%s window=[%d,%d)%s%s"
            % (t["bender"], os.path.basename(t["calib"]),
               os.path.basename(t["pool"]), t["sub_start"], t["sub_end"],
               "" if not t["colmask"] else
               " colmask=" + os.path.basename(t["colmask"]),
               "  [RETIRED-OVERRIDE]" if t["retired_override"] else ""))


def _main(argv):
    sh = "--sh" in argv
    args = [a for a in argv if not a.startswith("--")]
    channels = [int(a) for a in args] if args else list(range(N_CHANNELS))
    rc = 0
    for c in channels:
        try:
            t = trio(c)
        except (RetiredFixtureError, FileNotFoundError, RuntimeError) as e:
            print("ERROR " + str(e), file=sys.stderr)
            rc = 1
            continue
        if sh:
            print("TRIO_CALIB=%s" % t["calib"])
            print("TRIO_POOL=%s" % t["pool"])
            print("TRIO_SUB_START=%d" % t["sub_start"])
            print("TRIO_SUB_END=%d" % t["sub_end"])
            print("TRIO_COLMASK=%s" % (t["colmask"] or ""))
        else:
            print(describe(c))
    if rc == 0:
        try:
            roles = lane_roles()
        except RuntimeError as e:
            print("ERROR " + str(e), file=sys.stderr)
            return 1
        print(("PIM_LANE_ROLES=%s" if sh else "lane roles: %s") % roles)
    return rc


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
