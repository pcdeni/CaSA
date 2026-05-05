#include "pim_sim.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// SoftMC instruction encoding (mirrors api/instruction.h)
namespace {
  constexpr int IS_DDR    = 63;
  constexpr int IS_BR     = 62;
  constexpr int IS_MISC   = 61;
  constexpr int IS_MEM    = 60;
  constexpr int FU_CODE   = 48;
  // EXE field positions
  constexpr int RS1       = 0;
  constexpr int RS2       = 4;
  constexpr int RT        = 20;
  constexpr int IMD1      = 4;   // bits 4..23 (20-bit lower imm)
  constexpr int IMD3      = 24;  // bits 24..47 (upper imm for LI)
  constexpr int BR_TGT    = 8;
  constexpr int J_TGT     = 0;
  constexpr int SLP_AMT   = 0;
  // EXE function codes
  constexpr int FN_ADD    = 0;
  constexpr int FN_ADDI   = 1;
  constexpr int FN_SUB    = 2;
  constexpr int FN_SUBI   = 3;
  constexpr int FN_MV     = 4;
  constexpr int FN_SRC    = 5;
  constexpr int FN_LI     = 6;
  constexpr int FN_LDWD   = 7;
  constexpr int FN_LDPC   = 8;
  // BR function codes
  constexpr int FN_BL     = 0;
  constexpr int FN_BEQ    = 1;
  constexpr int FN_JUMP   = 2;
  constexpr int FN_SLEEP  = 3;
  // DDR mininst (16-bit)
  constexpr int DDR_CMD   = 12;   // bits 12..15 (4-bit command)
  constexpr int DDR_CAR   = 4;    // CAR field (col-addr-reg id)
  constexpr int DDR_BAR   = 0;    // BAR field (bank-addr-reg id)
  constexpr int DDR_RAR   = 4;    // RAR field (row-addr-reg id)
  // DDR command codes (4-bit; bit 3 = 1 marks DDR class — that's how
  // the IS_DDR flag in slot 3 of the packed Inst gets set automatically)
  constexpr int OP_WRITE  = 8;
  constexpr int OP_READ   = 9;
  constexpr int OP_PRE    = 10;
  constexpr int OP_ACT    = 11;
  constexpr int OP_ZQ     = 12;
  constexpr int OP_REF    = 13;
  constexpr int OP_NOP    = 15;
}

SimDramModel::SimDramModel() : phy_ctr(0) {
  for (int b = 0; b < 16; b++) {
    open_row[b] = -1;
    last_act_phy[b] = -1;
    last_pre_phy[b] = -1;
    last_act_row[b] = 0;
    row_buffer[b].assign(8192, 0);
  }
}

void SimDramModel::load_calib(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    fprintf(stderr, "[pim_sim] cannot open calib file %s\n", path.c_str());
    return;
  }
  std::string line;
  int n_loaded = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    CalibEntry c;
    int bank;
    if (!(iss >> c.s_id >> bank >> c.Rfirst >> c.Rsecond)) continue;
    int n_rows = 0;
    uint32_t v;
    while (n_rows < 16 && (iss >> v)) {
      c.open_rows[n_rows++] = v;
    }
    if (n_rows != 16) continue;
    if (bank < 0 || bank >= 16) continue;
    calibs[bank].push_back(c);
    n_loaded++;
  }
  if (verbose) {
    fprintf(stderr, "[pim_sim] loaded %d calib tuples (per-bank counts:",
            n_loaded);
    for (int b = 0; b < 16; b++)
      if (!calibs[b].empty()) fprintf(stderr, " bk%d=%zu", b, calibs[b].size());
    fprintf(stderr, ")\n");
  }
}

void SimDramModel::preload_row(int bank, uint32_t row, const uint8_t* data, size_t n) {
  auto& r = get_row(bank, row);
  std::copy(data, data + std::min<size_t>(n, 8192), r.begin());
}

std::vector<uint8_t>& SimDramModel::get_row(int bank, uint32_t row) {
  auto& m = bank_rows[bank];
  auto it = m.find(row);
  if (it == m.end()) {
    it = m.emplace(row, std::vector<uint8_t>(8192, 0)).first;
  }
  return it->second;
}

const SimDramModel::CalibEntry*
SimDramModel::find_calib_by_open_row(int bank, uint32_t row) const {
  for (const auto& c : calibs[bank])
    for (int i = 0; i < 16; i++)
      if (c.open_rows[i] == row) return &c;
  return nullptr;
}

const SimDramModel::CalibEntry*
SimDramModel::find_calib_by_first_or_second(int bank, uint32_t row) const {
  for (const auto& c : calibs[bank])
    if (c.Rfirst == row || c.Rsecond == row) return &c;
  return nullptr;
}

void SimDramModel::apply_act(int bank, int row) {
  if (open_row[bank] >= 0) {
    // Real DRAM violation, but fail soft: writeback then proceed.
    if (verbose) fprintf(stderr,
        "[pim_sim] WARN: ACT bank=%d row=%d while row=%d still open; auto-PRE\n",
        bank, row, open_row[bank]);
    apply_pre(bank);
  }
  auto& r = get_row(bank, (uint32_t)row);
  std::copy(r.begin(), r.end(), row_buffer[bank].begin());
  open_row[bank] = row;
}

void SimDramModel::apply_pre(int bank) {
  if (open_row[bank] >= 0) {
    auto& r = get_row(bank, (uint32_t)open_row[bank]);
    std::copy(row_buffer[bank].begin(), row_buffer[bank].end(), r.begin());
    open_row[bank] = -1;
  }
}

void SimDramModel::apply_rd(int bank, int col_raw) {
  // SoftMC convention: CAR holds (col_index × 8); each "column" we
  // address is one DDR4 burst = 64 bytes. So col_index = col_raw / 8,
  // valid range 0..127 (8 KB / 64 = 128 columns/row).
  int col = col_raw / 8;
  if (open_row[bank] < 0) {
    if (verbose) fprintf(stderr,
        "[pim_sim] WARN: RD bank=%d col=%d with no open row\n", bank, col);
    response_queue.insert(response_queue.end(), 64, 0);
    return;
  }
  size_t off = (size_t)col * 64;
  if (off + 64 > 8192) {
    fprintf(stderr, "[pim_sim] RD col=%d (raw=%d) out of range\n", col, col_raw);
    response_queue.insert(response_queue.end(), 64, 0);
    return;
  }
  response_queue.insert(response_queue.end(),
                        row_buffer[bank].begin() + off,
                        row_buffer[bank].begin() + off + 64);
}

void SimDramModel::apply_wr(int bank, int col_raw) {
  int col = col_raw / 8;
  if (open_row[bank] < 0) {
    if (verbose) fprintf(stderr,
        "[pim_sim] WARN: WR bank=%d col=%d with no open row\n", bank, col);
    return;
  }
  size_t off = (size_t)col * 64;
  if (off + 64 > 8192) {
    fprintf(stderr, "[pim_sim] WR col=%d (raw=%d) out of range\n", col, col_raw);
    return;
  }
  // Wdata: 16 × 32-bit slots = 64 bytes — copy them in.
  for (int s = 0; s < 16; s++) {
    uint32_t v = wdata_slot[s];
    row_buffer[bank][off + s*4 + 0] = (uint8_t)(v >> 0);
    row_buffer[bank][off + s*4 + 1] = (uint8_t)(v >> 8);
    row_buffer[bank][off + s*4 + 2] = (uint8_t)(v >> 16);
    row_buffer[bank][off + s*4 + 3] = (uint8_t)(v >> 24);
  }
}

void SimDramModel::on_act_check_doubleACT(int bank, int row) {
  // Look for the three EXACT calibrated SiMRA charge-sharing patterns.
  // Anything else is just a normal "open R1, close, open R2" sequence
  // (e.g. the separate ACT-WR-PRE programs that per_column_write_row
  // emits between each WRITE) — we DO NOT apply SiMRA semantics there.
  if (last_act_phy[bank] < 0 || last_pre_phy[bank] < 0) return;
  if (last_pre_phy[bank] <= last_act_phy[bank]) return;
  long long t_12 = last_pre_phy[bank] - last_act_phy[bank] - 1;
  long long t_23 = phy_ctr + /*slot already added*/ 0 - last_pre_phy[bank] - 1;
  uint32_t R1 = last_act_row[bank];
  uint32_t R2 = (uint32_t)row;
  // SiMRA charge-sharing requires the two ACTs to a different rows
  // AND tight controlled timing — the calibrated table is small.
  bool is_simra = false;
  if (R1 != R2) {
    if      (t_12 == 30 && t_23 == 1) { rowclone(bank, R1, R2);   is_simra = true; }
    else if (t_12 == 10 && t_23 == 2) { broadcast(bank, R1);      is_simra = true; }
    else if (t_12 == 0  && t_23 == 0) { maj3_vote(bank, R1);      is_simra = true; }
  }
  if (is_simra) {
    // Reset history so the same ACT R2 doesn't re-trigger.
    last_act_phy[bank] = -1;
    last_pre_phy[bank] = -1;
  }
  // (else): regular ACT-PRE-ACT sequence — apply_act already opened the
  // row into row_buffer; no SiMRA. Leave history advanced to current
  // cycle so the NEXT PRE-ACT pair is detected fresh.
}

void SimDramModel::rowclone(int bank, uint32_t src, uint32_t dst) {
  if (verbose >= 2)
    fprintf(stderr, "[pim_sim] RowClone bank=%d %u→%u\n", bank, src, dst);
  auto& s = get_row(bank, src);
  auto& d = get_row(bank, dst);
  std::copy(s.begin(), s.end(), d.begin());
  // Both rows are now in the row buffer state semantically; the actual
  // open-row tracking already reflects the second ACT (=dst) so the
  // row_buffer[bank] should reflect dst.
  std::copy(d.begin(), d.end(), row_buffer[bank].begin());
  open_row[bank] = (int)dst;
}

void SimDramModel::broadcast(int bank, uint32_t src) {
  if (verbose >= 2) fprintf(stderr, "[pim_sim] broadcast bank=%d src=%u\n", bank, src);
  // Find which calib has src as Rfirst (broadcast goes from Rfirst to
  // its associated 16 open_rows); fall back to "any calib containing
  // src as Rfirst or Rsecond".
  const CalibEntry* c = find_calib_by_first_or_second(bank, src);
  if (!c) {
    fprintf(stderr,
        "[pim_sim] broadcast: no calib has Rfirst/Rsecond=%u for bank=%d\n",
        src, bank);
    abort();
  }
  auto& s = get_row(bank, src);
  for (int i = 0; i < 16; i++) {
    auto& d = get_row(bank, c->open_rows[i]);
    std::copy(s.begin(), s.end(), d.begin());
  }
}

void SimDramModel::maj3_vote(int bank, uint32_t dst) {
  if (verbose >= 2) fprintf(stderr, "[pim_sim] MAJ3 bank=%d → row=%u\n", bank, dst);
  // Find the calib whose Rfirst/Rsecond equals dst (MAJ3 result lands
  // back on Rfirst). Vote across that calib's 16 open_rows.
  const CalibEntry* c = find_calib_by_first_or_second(bank, dst);
  if (!c) {
    fprintf(stderr,
        "[pim_sim] MAJ3: no calib has Rfirst/Rsecond=%u for bank=%d\n",
        dst, bank);
    abort();
  }
  std::vector<uint8_t> result(8192, 0);
  // Cache pointers to the 16 open rows
  std::vector<uint8_t>* rs[16];
  for (int i = 0; i < 16; i++) rs[i] = &get_row(bank, c->open_rows[i]);
  for (int byte_idx = 0; byte_idx < 8192; byte_idx++) {
    uint8_t out_byte = 0;
    for (int bit = 0; bit < 8; bit++) {
      int ones = 0;
      for (int i = 0; i < 16; i++)
        if (((*rs[i])[byte_idx] >> bit) & 1) ones++;
      // 16-input majority: > 8 votes → 1
      if (ones > 8) out_byte |= (1 << bit);
    }
    result[byte_idx] = out_byte;
  }
  auto& d = get_row(bank, dst);
  std::copy(result.begin(), result.end(), d.begin());
  std::copy(result.begin(), result.end(), row_buffer[bank].begin());
  open_row[bank] = (int)dst;
}

void SimDramModel::apply_mininst(int slot, uint16_t m) {
  // Each mininst's PHY position = phy_ctr + slot (where phy_ctr is the
  // current cycle's first slot's position).
  long long my_phy = phy_ctr + slot;
  // Decode 4-bit cmd (bits 12..15)
  int cmd = (m >> DDR_CMD) & 0xF;
  int bar_id = m & 0xF;
  int car_id = (m >> DDR_CAR) & 0xF;
  int rar_id = (m >> DDR_RAR) & 0xF;
  switch (cmd) {
    case OP_NOP:
      // Pure NOP — nothing to apply.
      return;
    case OP_ACT: {
      // The bank field is a register ID; look up the bank value.
      int bank = (int)(reg[bar_id] & 0xF);
      int row  = (int)reg[rar_id];
      // Update phy_ctr's snapshot for this slot: doubleACT detection
      // expects last_act_phy to be THIS slot's position.
      long long save = phy_ctr;
      phy_ctr = my_phy + 1;  // semantic "current time" past this ACT
      apply_act(bank, row);
      // Detect doubleACT pattern AFTER this ACT
      on_act_check_doubleACT(bank, row);
      last_act_phy[bank] = my_phy;
      last_act_row[bank] = (uint32_t)row;
      phy_ctr = save;
      return;
    }
    case OP_PRE: {
      int bank = (int)(reg[bar_id] & 0xF);
      apply_pre(bank);
      last_pre_phy[bank] = my_phy;
      return;
    }
    case OP_READ: {
      int bank = (int)(reg[bar_id] & 0xF);
      int col  = (int)reg[car_id];
      apply_rd(bank, col);
      return;
    }
    case OP_WRITE: {
      int bank = (int)(reg[bar_id] & 0xF);
      int col  = (int)reg[car_id];
      apply_wr(bank, col);
      return;
    }
    case OP_ZQ:
    case OP_REF:
      // Not modelled (no impact on data semantics for our workload).
      return;
    default:
      fprintf(stderr, "[pim_sim] unknown DDR cmd %d at slot %d\n", cmd, slot);
      return;
  }
}

void SimDramModel::apply_ddr_pack4(uint64_t pack4) {
  uint16_t m0 = (uint16_t)(pack4 & 0xFFFF);
  uint16_t m1 = (uint16_t)((pack4 >> 16) & 0xFFFF);
  uint16_t m2 = (uint16_t)((pack4 >> 32) & 0xFFFF);
  uint16_t m3 = (uint16_t)((pack4 >> 48) & 0xFFFF);
  apply_mininst(0, m0);
  apply_mininst(1, m1);
  apply_mininst(2, m2);
  apply_mininst(3, m3);
}

void SimDramModel::execute_inst(uint64_t inst, size_t& pc) {
  // DDR? (bit 63 set = at least one slot is a DDR cmd)
  if (inst & (1ULL << IS_DDR)) {
    apply_ddr_pack4(inst);
    pc++;
    phy_ctr += 4;
    return;
  }
  // BR / EXE / MISC / MEM
  if (inst & (1ULL << IS_BR)) {
    int fcode = (int)((inst >> FU_CODE) & 0x7FF);
    int rs1 = (int)((inst >> RS1) & 0xF);
    int rs2 = (int)((inst >> RS2) & 0xF);
    int tgt = (int)((inst >> BR_TGT) & 0x7FFFF);
    if (fcode == FN_BL) {
      if ((int32_t)reg[rs1] < (int32_t)reg[rs2]) pc = (size_t)tgt;
      else pc++;
    } else if (fcode == FN_BEQ) {
      if (reg[rs1] == reg[rs2]) pc = (size_t)tgt;
      else pc++;
    } else if (fcode == FN_JUMP) {
      int j = (int)((inst >> J_TGT) & 0x7FFFF);
      pc = (size_t)j;
    } else {
      pc++;
    }
    phy_ctr += 4;
    return;
  }
  if (inst & (1ULL << IS_MISC)) {
    int fcode = (int)((inst >> FU_CODE) & 0x7FF);
    if (fcode == FN_SLEEP) {
      uint32_t amt = (uint32_t)(inst & 0xFFFFFF);
      phy_ctr += 4 * (long long)amt;
      pc++;
      return;
    }
    // INFO instruction (fcode==0) — no-op for sim purposes.
    pc++;
    phy_ctr += 4;
    return;
  }
  if (inst == 0) {
    // SMC_END = all-zero instruction.
    pc = (size_t)-1;  // sentinel for "end"
    return;
  }
  // EXE (default)
  int fcode = (int)((inst >> FU_CODE) & 0x7FF);
  int rs1 = (int)((inst >> RS1) & 0xF);
  int rs2 = (int)((inst >> RS2) & 0xF);
  int rt  = (int)((inst >> RT)  & 0xF);
  // SMC_LI's 32-bit immediate is split: low 16 at IMD1 (bit 4..23), high
  // 16 at IMD3 (bit 24..39). For ADDI it's just the 16-bit IMD1.
  uint32_t imd16 = (uint32_t)((inst >> IMD1) & 0xFFFF);
  uint32_t imd_hi = (uint32_t)((inst >> IMD3) & 0xFFFF);
  uint32_t imd32 = imd16 | (imd_hi << 16);
  switch (fcode) {
    case FN_ADD:  reg[rt] = reg[rs1] + reg[rs2]; break;
    case FN_ADDI: reg[rt] = reg[rs1] + imd16;     break;
    case FN_SUB:  reg[rt] = reg[rs1] - reg[rs2]; break;
    case FN_SUBI: reg[rt] = reg[rs1] - imd16;     break;
    case FN_MV:   reg[rt] = reg[rs1];             break;
    case FN_LI:   reg[rt] = imd32;                break;
    case FN_LDWD: {
      // wdata_slot[rt] = reg[rs1]. Here `rt` field actually carries the
      // slot index (per api/instruction.h: SMC_LDWD encodes slot via __RT
      // shift) — but caller pattern is SMC_LDWD(reg, slot) with slot as
      // the 2nd arg, mapped to RT bits.
      int slot = rt;  // 0..15
      if (slot < 0 || slot >= 16) break;
      wdata_slot[slot] = reg[rs1];
      break;
    }
    case FN_SRC: {
      // Shift right circular (1 bit).
      uint32_t v = reg[rs1];
      reg[rt] = (v >> 1) | ((v & 1u) << 31);
      break;
    }
    default:
      // Unknown fcode — silently noop.
      break;
  }
  pc++;
  phy_ctr += 4;
}

void SimDramModel::execute_program() {
  size_t pc = 0;
  size_t guard = 0;
  size_t last_logged_pc = (size_t)-1;
  size_t same_pc_stuck = 0;
  while (pc != (size_t)-1 && pc < imem.size() && guard < 50000000) {
    uint64_t inst = imem[pc];
    if (verbose >= 3 || (guard < 30 && verbose >= 1)) {
      fprintf(stderr, "[pim_sim] @pc=%zu inst=0x%016lx\n", pc, (unsigned long)inst);
    }
    size_t pc_before = pc;
    execute_inst(inst, pc);
    if (pc == pc_before) {
      same_pc_stuck++;
      if (same_pc_stuck > 100) {
        fprintf(stderr, "[pim_sim] STUCK at pc=%zu inst=0x%016lx (guard=%zu)\n",
                pc, (unsigned long)inst, guard);
        return;
      }
    } else {
      same_pc_stuck = 0;
    }
    if (verbose >= 1 && guard >= 100000 && (guard % 100000 == 0)) {
      fprintf(stderr, "[pim_sim] progress: guard=%zu pc=%zu phy_ctr=%lld\n",
              guard, pc, phy_ctr);
    }
    guard++;
  }
  if (guard >= 50000000) {
    fprintf(stderr, "[pim_sim] execute guard hit (50M iterations) — likely infinite loop\n");
  }
  if (verbose >= 1)
    fprintf(stderr, "[pim_sim] program completed in %zu inst-executions\n", guard);
}

int SimDramModel::send_program(const uint8_t* data, size_t bytes) {
  std::lock_guard<std::mutex> lock(mtx);
  // QUEUE the program for deferred execution; the actual run happens
  // inside the next recv_response() call. This mirrors real-hardware
  // ordering — the FPGA wouldn't produce any c2h bytes before the
  // host's c2h reader thread starts pulling — so consumeData()'s
  // initial read sees data appear, drains it, then sees end-of-stream
  // and exits cleanly. Without deferral, send_program runs to
  // completion synchronously and consumeData's first read returns 0
  // (queue still empty when the thread first runs), exits early, and
  // receiveData() blocks forever.
  if (bytes % 32 != 0) {
    fprintf(stderr, "[pim_sim] send_program: size %zu not multiple of 32\n", bytes);
    return 1;
  }
  size_t n_inst = bytes / 32;
  std::vector<uint64_t> prog(n_inst, 0);
  for (size_t i = 0; i < n_inst; i++) {
    std::memcpy(&prog[i], data + i * 32, 8);
  }
  deferred_programs.push_back(std::move(prog));
  return 0;
}

int SimDramModel::recv_response(uint8_t* buf, size_t size) {
  std::lock_guard<std::mutex> lock(mtx);
  // Lazily run any deferred programs to populate the response queue.
  while (response_queue.empty() && !deferred_programs.empty()) {
    imem = std::move(deferred_programs.front());
    deferred_programs.erase(deferred_programs.begin());
    if (verbose >= 1)
      fprintf(stderr, "[pim_sim] running program of %zu instructions\n", imem.size());
    execute_program();
    // Mimic the FPGA's per-program trailing AXI beat. Real consumeData()
    // strips 32 bytes when it sees a partial-chunk read — 32 zeros here.
    response_queue.insert(response_queue.end(), 32, 0);
  }
  size_t avail = response_queue.size();
  size_t n = std::min(size, avail);
  if (n > 0) {
    std::memcpy(buf, response_queue.data(), n);
    response_queue.erase(response_queue.begin(), response_queue.begin() + n);
  }
  return (int)n;
}
