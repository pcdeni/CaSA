#include "regalloc.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace ra {

// ============================ AInst introspection ===========================

std::vector<int> AInst::uses_vregs() const {
  std::vector<int> u;
  if (kind == A_PACK4) {
    for (int s = 0; s < 4; s++) {
      const Slot& sl = slots[s];
      if (sl.kind == S_NOP) continue;
      // BAR is read for any non-NOP DDR mininst.
      if (!sl.ibar && sl.v_bar >= 0) u.push_back(sl.v_bar);
      // RAR is read on ACT.
      if (sl.kind == S_ACT && !sl.irar && sl.v_rar >= 0) u.push_back(sl.v_rar);
      // CAR is read on RD/WR.
      if ((sl.kind == S_RD || sl.kind == S_WR) && sl.v_car >= 0) u.push_back(sl.v_car);
    }
  } else if (kind == A_BR_BL) {
    if (v_a >= 0) u.push_back(v_a);
    if (v_b >= 0) u.push_back(v_b);
  } else if (kind == A_LDWD) {
    if (v_src >= 0) u.push_back(v_src);
  }
  // dedupe
  std::sort(u.begin(), u.end());
  u.erase(std::unique(u.begin(), u.end()), u.end());
  return u;
}

// ============================ Builder API ==================================

int AbstractProgram::new_vreg(const std::string& name, uint32_t value) {
  VReg v; v.name = name; v.value = value;
  vregs.push_back(v);
  return (int)vregs.size() - 1;
}

void AbstractProgram::emit_pack4(Slot s0, Slot s1, Slot s2, Slot s3) {
  AInst ai; ai.kind = A_PACK4;
  ai.slots[0] = s0; ai.slots[1] = s1; ai.slots[2] = s2; ai.slots[3] = s3;
  insts.push_back(std::move(ai));
}

void AbstractProgram::emit_sleep(uint32_t amt) {
  AInst ai; ai.kind = A_SLEEP; ai.sleep_amt = amt;
  insts.push_back(std::move(ai));
}

void AbstractProgram::emit_ldwd(int v_src, int wdata_slot) {
  AInst ai; ai.kind = A_LDWD; ai.v_src = v_src; ai.wdata_slot = wdata_slot;
  insts.push_back(std::move(ai));
}

void AbstractProgram::emit_label(const std::string& name) {
  AInst ai; ai.kind = A_LABEL; ai.label = name;
  insts.push_back(std::move(ai));
}

void AbstractProgram::emit_bl(int v_a, int v_b, const std::string& label) {
  AInst ai; ai.kind = A_BR_BL; ai.v_a = v_a; ai.v_b = v_b; ai.label = label;
  insts.push_back(std::move(ai));
}

void AbstractProgram::emit_end() {
  AInst ai; ai.kind = A_END;
  insts.push_back(std::move(ai));
}

void AbstractProgram::emit_def(int vreg) {
  AInst ai; ai.kind = A_DEF; ai.defs.push_back(vreg);
  insts.push_back(std::move(ai));
}

void AbstractProgram::emit_parallel_doubleACT(
    int t_12, int t_23,
    const std::vector<int>& v_bar,
    const std::vector<int>& v_src,
    const std::vector<int>& v_dst)
{
  int n_banks = (int)v_bar.size();
  if (n_banks <= 0) return;
  // Compute the min stagger that keeps banks' events collision-free.
  auto min_shift = [&](){
    if (n_banks == 1) return 0;
    int pat_len = t_12 + t_23 + 3;
    for (int S = 1; S <= pat_len; S++) {
      std::set<int> used; bool ok = true;
      for (int b = 0; b < n_banks; b++) {
        int e0 = b*S, e1 = e0 + t_12 + 1, e2 = e0 + t_12 + t_23 + 2;
        if (used.count(e0)||used.count(e1)||used.count(e2)) { ok=false; break; }
        used.insert(e0); used.insert(e1); used.insert(e2);
      }
      if (ok) return S;
    }
    return pat_len;
  };
  int shift = min_shift();
  int max_pos = (n_banks - 1) * shift + t_12 + t_23 + 2;
  int num_cmd = max_pos + 1;
  if (num_cmd % 4) num_cmd += 4 - (num_cmd % 4);

  // Lay out the slots in a flat array of length num_cmd.
  std::vector<Slot> flat(num_cmd, Slot::nop());
  for (int b = 0; b < n_banks; b++) {
    int e0 = b*shift;
    int e1 = e0 + t_12 + 1;
    int e2 = e0 + t_12 + t_23 + 2;
    flat[e0] = Slot::act(v_bar[b], v_src[b]);
    flat[e1] = Slot::pre(v_bar[b]);
    flat[e2] = Slot::act(v_bar[b], v_dst[b]);
  }
  for (int i = 0; i < num_cmd; i += 4)
    emit_pack4(flat[i], flat[i+1], flat[i+2], flat[i+3]);
}

// ============================ Liveness analysis ============================

// Compute live_first / live_last per vreg by scanning the abstract program.
// def := A_DEF sets live_first; uses are listed by uses_vregs() for each
// inst. live_last := index of last inst that uses the vreg.
static void compute_liveness(AbstractProgram& prog) {
  for (auto& v : prog.vregs) { v.live_first = -1; v.live_last = -1; }
  for (size_t i = 0; i < prog.insts.size(); i++) {
    const AInst& ai = prog.insts[i];
    if (ai.kind == A_DEF) {
      for (int v : ai.defs)
        if (prog.vregs[v].live_first < 0)
          prog.vregs[v].live_first = (int)i;
    }
    for (int v : ai.uses_vregs()) {
      if (prog.vregs[v].live_first < 0)
        prog.vregs[v].live_first = (int)i;  // defensive: implicit def
      prog.vregs[v].live_last = (int)i;
    }
  }
}

// ============================ Linear-scan allocator ========================

// State: which physical slots are free, which vregs are currently active
// (i.e. live and assigned a slot), and (for spilled vregs) at which inst
// index they need to be reloaded next.

struct ActiveEntry { int vreg; int phys_slot; };

Allocation allocate(const AbstractProgram& prog_const,
                    int n_phys_regs,
                    const std::set<int>& reserved)
{
  AbstractProgram prog = prog_const;
  compute_liveness(prog);
  Allocation result;
  result.per_inst.resize(prog.insts.size());

  // Free pool of physical regs.
  std::set<int> free_pool;
  for (int s = 0; s < n_phys_regs; s++)
    if (!reserved.count(s)) free_pool.insert(s);

  // vreg → phys_slot for currently-allocated vregs.
  std::map<int,int> vreg_phys;

  // For each vreg, a queue of next-use indices (for spill heuristic).
  // We pre-compute per-vreg use list.
  std::vector<std::vector<int>> uses_per_vreg(prog.vregs.size());
  for (size_t i = 0; i < prog.insts.size(); i++) {
    for (int v : prog.insts[i].uses_vregs())
      uses_per_vreg[v].push_back((int)i);
  }
  // Each vreg's "next use cursor" — increments as we walk.
  std::vector<size_t> use_cursor(prog.vregs.size(), 0);

  auto next_use = [&](int v) -> int {
    while (use_cursor[v] < uses_per_vreg[v].size()) {
      int u = uses_per_vreg[v][use_cursor[v]];
      return u;
    }
    return INT32_MAX;
  };
  auto advance_cursor = [&](int v, int past) {
    while (use_cursor[v] < uses_per_vreg[v].size() &&
           uses_per_vreg[v][use_cursor[v]] <= past) {
      use_cursor[v]++;
    }
  };

  // Helper: ensure vreg `v` is in some physical slot at instruction `i`.
  // Allocates if not present. May spill an existing vreg to make room.
  auto ensure_alloc = [&](int v, int i) -> int {
    auto it = vreg_phys.find(v);
    if (it != vreg_phys.end()) return it->second;
    int chosen;
    if (!free_pool.empty()) {
      chosen = *free_pool.begin();
      free_pool.erase(free_pool.begin());
    } else {
      // Spill: pick the active vreg whose next use is FURTHEST out.
      int victim = -1;
      int victim_next = -1;
      for (auto& kv : vreg_phys) {
        if (kv.first == v) continue;
        int nu = next_use(kv.first);
        if (nu > victim_next) { victim = kv.first; victim_next = nu; }
      }
      if (victim < 0) {
        fprintf(stderr, "ra::allocate: no victim available — "
                "n_phys_regs=%d reserved=%zu\n", n_phys_regs, reserved.size());
        std::abort();
      }
      chosen = vreg_phys[victim];
      vreg_phys.erase(victim);
      // The victim's NEXT use will need a reload. We don't insert that
      // here — we insert it lazily when we reach the use point.
    }
    vreg_phys[v] = chosen;
    // If `v` is being newly materialized at `i` because of a use (not a
    // def), record a reload at this inst.
    if (prog.vregs[v].live_first != (int)i)
      result.per_inst[i].reloads.emplace_back(v, chosen);
    return chosen;
  };

  // Walk the program, allocating uses and freeing dead vregs.
  for (size_t i = 0; i < prog.insts.size(); i++) {
    const AInst& ai = prog.insts[i];

    // 1. Process DEFS first — they bring vregs to life.
    if (ai.kind == A_DEF) {
      for (int v : ai.defs) {
        // Force allocation now (treat A_DEF as a "use" that materialises).
        if (vreg_phys.find(v) == vreg_phys.end()) {
          (void)ensure_alloc(v, (int)i);
        }
      }
      continue;
    }

    // 2. Process USES — make sure each used vreg has a physical slot.
    auto& pi = result.per_inst[i];
    if (ai.kind == A_PACK4) {
      for (int s = 0; s < 4; s++) {
        const Slot& sl = ai.slots[s];
        if (sl.kind == S_NOP) continue;
        if (!sl.ibar && sl.v_bar >= 0)
          pi.slot_phys_bar[s] = ensure_alloc(sl.v_bar, (int)i);
        if (sl.kind == S_ACT && !sl.irar && sl.v_rar >= 0)
          pi.slot_phys_rar[s] = ensure_alloc(sl.v_rar, (int)i);
        if ((sl.kind == S_RD || sl.kind == S_WR) && sl.v_car >= 0)
          pi.slot_phys_car[s] = ensure_alloc(sl.v_car, (int)i);
      }
    } else if (ai.kind == A_BR_BL) {
      pi.bl_phys_a = ensure_alloc(ai.v_a, (int)i);
      pi.bl_phys_b = ensure_alloc(ai.v_b, (int)i);
    } else if (ai.kind == A_LDWD) {
      pi.ldwd_phys_src = ensure_alloc(ai.v_src, (int)i);
    }

    // 3. Advance use-cursors for any vreg used at this point.
    for (int v : ai.uses_vregs()) {
      advance_cursor(v, (int)i);
    }

    // 4. Free vregs whose live_last has passed.
    std::vector<int> dead;
    for (auto& kv : vreg_phys) {
      if (prog.vregs[kv.first].live_last <= (int)i) dead.push_back(kv.first);
    }
    for (int v : dead) {
      free_pool.insert(vreg_phys[v]);
      vreg_phys.erase(v);
    }
  }

  return result;
}

// ============================ Lowerer ======================================

// Emit a single SMC_LI(value, slot) into the Program.
static void emit_li(Program& p, uint32_t value, int slot) {
  p.add_inst(SMC_LI(value, slot));
}

// Build a Mininst from a Slot + per-slot phys regs.
static Mininst slot_to_mininst(const Slot& sl,
                                int phys_bar, int phys_rar, int phys_car) {
  switch (sl.kind) {
    case S_NOP:   return SMC_NOP();
    case S_ACT:   return SMC_ACT(phys_bar, sl.ibar ? 1 : 0,
                                  phys_rar, sl.irar ? 1 : 0);
    case S_PRE:   return SMC_PRE(phys_bar, sl.ibar ? 1 : 0, /*pall=*/0);
    case S_RD:    return SMC_READ(phys_bar, sl.ibar ? 1 : 0,
                                   phys_car, sl.icar ? 1 : 0,
                                   /*BL4=*/0, /*ap=*/0);
    case S_WR:    return SMC_WRITE(phys_bar, sl.ibar ? 1 : 0,
                                    phys_car, sl.icar ? 1 : 0,
                                    /*BL4=*/0, /*ap=*/0);
  }
  return SMC_NOP();
}

Program lower(const AbstractProgram& prog, const Allocation& alloc) {
  Program p;
  // Initial defs: emit LI for every vreg whose live_first is BEFORE the
  // first inst that uses it (covers vregs declared via A_DEF up-front).
  // Linear-scan's first allocation point is captured as a "reload" for
  // each vreg's first usage point — we handle them together.
  for (size_t i = 0; i < prog.insts.size(); i++) {
    const AInst& ai = prog.insts[i];
    const auto& pi = alloc.per_inst[i];

    // Emit any required reloads BEFORE the inst.
    for (auto& [vreg_id, phys] : pi.reloads) {
      emit_li(p, prog.vregs[vreg_id].value, phys);
    }

    // Also handle the initial def: ensure_alloc() was called on first use
    // OR on A_DEF. For first use without a prior def, the first allocation
    // counts as a "reload" too — already in pi.reloads. For A_DEF, we
    // emit the LI here.
    if (ai.kind == A_DEF) {
      for (int v : ai.defs) {
        // Find the phys slot the allocator picked. Look at the next inst
        // that uses v: its uses_vregs() contains v, and per_inst[that_i]
        // holds the phys. Easier: search forward for first use.
        // Allocator assigns at A_DEF too, so we need to know the phys.
        // The simplest path: look up the slot from the post-allocation
        // state by consulting the next AInst that uses v. To avoid that
        // back-scan, we can require the allocator to record A_DEF
        // assignments in a separate map. For now: scan.
        for (size_t j = i + 1; j < prog.insts.size(); j++) {
          const AInst& aj = prog.insts[j];
          int phys = -1;
          if (aj.kind == A_PACK4) {
            for (int s = 0; s < 4; s++) {
              const Slot& sl = aj.slots[s];
              if (sl.v_bar == v) phys = alloc.per_inst[j].slot_phys_bar[s];
              else if (sl.v_rar == v) phys = alloc.per_inst[j].slot_phys_rar[s];
              else if (sl.v_car == v) phys = alloc.per_inst[j].slot_phys_car[s];
              if (phys >= 0) break;
            }
          } else if (aj.kind == A_LDWD && aj.v_src == v) {
            phys = alloc.per_inst[j].ldwd_phys_src;
          } else if (aj.kind == A_BR_BL) {
            if (aj.v_a == v) phys = alloc.per_inst[j].bl_phys_a;
            else if (aj.v_b == v) phys = alloc.per_inst[j].bl_phys_b;
          }
          if (phys >= 0) { emit_li(p, prog.vregs[v].value, phys); break; }
        }
      }
      continue;  // A_DEF emits no DDR cycle.
    }

    // Emit the actual instruction.
    if (ai.kind == A_PACK4) {
      Mininst m[4];
      for (int s = 0; s < 4; s++) {
        m[s] = slot_to_mininst(ai.slots[s],
                                pi.slot_phys_bar[s],
                                pi.slot_phys_rar[s],
                                pi.slot_phys_car[s]);
      }
      p.add_inst(m[0], m[1], m[2], m[3]);
    } else if (ai.kind == A_SLEEP) {
      p.add_inst(SMC_SLEEP(ai.sleep_amt));
    } else if (ai.kind == A_LDWD) {
      p.add_inst(SMC_LDWD(pi.ldwd_phys_src, ai.wdata_slot));
    } else if (ai.kind == A_LABEL) {
      p.add_label(ai.label);
    } else if (ai.kind == A_BR_BL) {
      p.add_branch(p.BR_TYPE::BL, pi.bl_phys_a, pi.bl_phys_b, ai.label);
    } else if (ai.kind == A_END) {
      p.add_inst(SMC_END());
    }
  }
  return p;
}

}  // namespace ra
