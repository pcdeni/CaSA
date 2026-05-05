// Algorithmic register allocator for SoftMC programs.
//
// The hand-written `parallel_emit.cpp` packs SiMRA doubleACT across 4 banks
// in pack4 slots, but assumes the caller has already loaded all bar/row
// values into the right register slots. With more concurrent banks (and
// more SiMRA stages chained inside one body), the live-set quickly exceeds
// the 16-slot register file, requiring either:
//   (a) hand-derived re-LI choreography (brittle), or
//   (b) standard register allocation (this).
//
// This module implements **linear-scan with rematerialization-as-spill**.
// Every virtual register here holds a value that is loadable from an
// immediate (`SMC_LI(value, slot)`). Spilling is therefore free — when
// the allocator evicts a vreg it just records the next-use point so a
// fresh `SMC_LI` can be emitted there. No memory traffic, no separate
// spill slot. This makes spills cheap and the allocator simple.
//
// The IR is intentionally narrow:
//   - VRegs are abstract names with associated immediate values (`uint32`).
//   - AbstractInsts are one of:
//       * Pack4: 4 DDR mininsts with vreg-id references in their bar/rar/car
//         fields. Slots may be `NOP_INST` (unused).
//       * Sleep: cycle gap (no DDR command bus activity).
//       * Branch: SMC_BL/JUMP for in-program loops (used by wrRow/rdRow).
//       * Label, End: control-flow markers.
//   - Each Pack4 declares which vregs it USES (read by mininsts) so the
//     allocator can compute liveness.
//
// Output: a concrete `Program` (api/prog.h) ready to send via
// `platform.execute()`, with `SMC_LI` instructions inserted before any
// pack4 whose vregs were spilled.

#pragma once
#include "instruction.h"
#include "prog.h"
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <set>

namespace ra {

// ----------------------- Virtual register table -----------------------

struct VReg {
  std::string name;     // for debug
  uint32_t    value;    // immediate to LI when materialised
  // Filled in by the allocator:
  int         live_first = -1;   // index of first inst that defines this vreg
  int         live_last  = -1;   // index of last inst that uses it
};

// ----------------------- Mininst slot: vreg references --------------

enum SlotKind {
  S_NOP,
  S_ACT,
  S_PRE,
  S_RD,
  S_WR,
};

struct Slot {
  SlotKind kind  = S_NOP;
  int      v_bar = -1;     // vreg id for BAR
  int      v_rar = -1;     // vreg id for RAR (ACT only)
  int      v_car = -1;     // vreg id for CAR (RD/WR only)
  bool     icar  = false;  // post-increment CAR by CASR (WR/RD)
  bool     ibar  = false;  // immediate-bar (rare; we pass via vreg by default)
  bool     irar  = false;  // immediate-rar
  // Convenience constructors:
  static Slot nop()                          { return {}; }
  static Slot act(int bar, int rar)          { Slot s; s.kind=S_ACT; s.v_bar=bar; s.v_rar=rar; return s; }
  static Slot pre(int bar)                   { Slot s; s.kind=S_PRE; s.v_bar=bar; return s; }
  static Slot rd (int bar, int car, bool icar=true)
    { Slot s; s.kind=S_RD; s.v_bar=bar; s.v_car=car; s.icar=icar; return s; }
  static Slot wr (int bar, int car, bool icar=true)
    { Slot s; s.kind=S_WR; s.v_bar=bar; s.v_car=car; s.icar=icar; return s; }
};

// ----------------------- Abstract instructions ----------------------

enum AInstKind {
  A_PACK4,         // 4 mininst slots referring to vregs
  A_SLEEP,         // SMC_SLEEP
  A_LDWD,          // load-word: copy vreg → wdata buffer slot
  A_BR_BL,         // SMC_BL — branch if (vreg_a < vreg_b) → label
  A_JUMP,          // SMC_JUMP → label (unused for now)
  A_LABEL,         // pseudo-instruction; resolves to PC
  A_END,           // SMC_END
  A_DEF,           // declares a vreg defined here (no DDR effect; allocator hint)
};

struct AInst {
  AInstKind kind;
  Slot      slots[4];        // for A_PACK4
  uint32_t  sleep_amt = 0;   // for A_SLEEP
  int       v_a = -1, v_b = -1;  // for A_BR_BL (compare regs)
  int       v_src = -1;      // for A_LDWD (source vreg)
  int       wdata_slot = 0;  // for A_LDWD (wdata buffer slot 0..15)
  std::string label;         // for A_LABEL / A_BR_BL / A_JUMP
  std::vector<int> defs;     // for A_DEF
  // Convenience:
  std::vector<int> uses_vregs() const;  // implemented in .cpp
};

// ----------------------- The abstract program ----------------------

class AbstractProgram {
public:
  // Define a new virtual register.
  int new_vreg(const std::string& name, uint32_t value);

  // Append abstract instructions.
  void emit_pack4(Slot s0, Slot s1, Slot s2, Slot s3);
  void emit_sleep(uint32_t amt);
  void emit_ldwd(int v_src, int wdata_slot);
  void emit_label(const std::string& name);
  void emit_bl(int v_a, int v_b, const std::string& label);
  void emit_end();
  void emit_def(int vreg);  // tells allocator a vreg becomes live here

  // Convenience: emit the SiMRA doubleACT pattern across N banks (1..4)
  // in parallel pack4 slots. Computes the minimum stagger, lays out the
  // ACT/PRE/ACT events, fills NOP slots. The vreg arrays are per-bank.
  void emit_parallel_doubleACT(int t_12, int t_23,
                                const std::vector<int>& v_bar,
                                const std::vector<int>& v_src,
                                const std::vector<int>& v_dst);

  // Public so tests can inspect.
  std::vector<AInst> insts;
  std::vector<VReg>  vregs;
};

// ----------------------- Allocation result --------------------------

struct Allocation {
  // Per AInst, the chosen physical reg id for each (slot.v_bar / v_rar /
  // v_car / v_a / v_b / v_src) reference. Filled by the allocator. For
  // A_PACK4, slot_phys[ainst][0..3].{bar,rar,car} are valid; for A_BR_BL,
  // bl_phys_a/b are valid; for A_LDWD, ldwd_phys_src.
  struct PerAInst {
    int slot_phys_bar[4] = {-1,-1,-1,-1};
    int slot_phys_rar[4] = {-1,-1,-1,-1};
    int slot_phys_car[4] = {-1,-1,-1,-1};
    int bl_phys_a = -1;
    int bl_phys_b = -1;
    int ldwd_phys_src = -1;
    // Reload LIs to insert BEFORE this AInst (vreg-id, phys-slot pairs).
    std::vector<std::pair<int,int>> reloads;
  };
  std::vector<PerAInst> per_inst;
};

// ----------------------- The allocator -----------------------------

// Run linear-scan with rematerialization. Returns the Allocation; throws
// if it can't fit the live-set into n_phys_regs (wouldn't normally
// happen since reloads are unbounded, but the policy may pick badly).
//
// reserved: physical slots that must NOT be touched (caller-provided
// hard-pinned reg IDs — e.g. {} if everything is allocator-managed).
Allocation allocate(const AbstractProgram& prog,
                    int n_phys_regs = 16,
                    const std::set<int>& reserved = {});

// ----------------------- Lowerer ----------------------------------

// Walk the abstract program + allocation, emit a concrete `Program`
// (api/prog.h) by issuing SMC_LI to materialise vregs into their
// allocated slots, then SMC_ACT/PRE/READ/WRITE/SLEEP/BL/END as
// appropriate.
Program lower(const AbstractProgram& prog, const Allocation& alloc);

}  // namespace ra
