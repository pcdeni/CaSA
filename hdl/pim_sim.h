// In-process behavioral DDR + SiMRA model used as a drop-in for the
// XDMA path when bitnet-proj-server is launched with PIM_BACKEND=sim.
//
// Goal: deterministic, observable PIM execution. The SoftMC program is
// decoded inst-by-inst, the DDR state evolves under standard DDR4
// semantics for ACT/PRE/RD/WR, and the SiMRA charge-sharing patterns
// (doubleACT with characteristic t_12/t_23) are recognised by per-bank
// timing and applied as their PROMISED semantics:
//   t_12=30, t_23=1 → RowClone (deterministic copy)
//   t_12=10, t_23=2 → broadcast to the 16 calibrated open_rows
//   t_12=0,  t_23=0 → MAJ3 majority-bit vote across open_rows
// Any other doubleACT pattern → fail loudly.

#pragma once
#include <cstdint>
#include <cstddef>
#include <map>
#include <vector>
#include <string>

class SimDramModel {
public:
  SimDramModel();
  // Load (bank, s_id, Rfirst, Rsecond, open_rows[16]) tuples per bank
  // from a calib_dimm0.txt file (whitespace-separated, one tuple/line).
  // Used to recognise (Rfirst, Rsecond) row pairs and route RowClone /
  // broadcast / MAJ3 to the right open_rows set.
  void load_calib(const std::string& path);
  // Optional: pre-populate a row with a specific byte pattern (for tests).
  void preload_row(int bank, uint32_t row, const uint8_t* data, size_t n);
  // Receive raw bytes from BoardInterface::sendData. Each Inst occupies
  // 8 bytes within a 32-byte AXI block (matches platform.cpp's pack
  // pattern of `temp_ptr[i*4] = iseq[i]` where temp_ptr is uint64_t*,
  // i.e. the program is stored at byte offsets 0, 32, 64, … of the
  // staging buffer). The first instruction at offset 0 is what fetch
  // would read first. Returns 0 on success.
  int send_program(const uint8_t* data, size_t bytes);
  // Pop bytes from the c2h response queue. Mirrors the
  // BoardInterface::recvData semantics — blocks/returns whatever is
  // available, capped at `size`. Returns bytes copied (≥ 0) or -1.
  int recv_response(uint8_t* buf, size_t size);
  // Verbosity for debug.
  int verbose = 0;

private:
  // ----- DDR state ------------------------------------------------------
  // Sparse storage: rows allocated lazily on first ACT/WR.
  std::map<uint32_t, std::vector<uint8_t>> bank_rows[16];
  // Per-bank open-row state.
  int open_row[16];          // -1 if no row open
  std::vector<uint8_t> row_buffer[16];   // 8 KB each, current open row's contents

  // ----- doubleACT detection (per-bank) --------------------------------
  long long phy_ctr;          // total PHY positions consumed
  long long last_act_phy[16]; // -1 = none
  uint32_t  last_act_row[16];
  long long last_pre_phy[16]; // -1 = none

  // ----- Calibration table ---------------------------------------------
  struct CalibEntry {
    int s_id;
    uint32_t Rfirst;
    uint32_t Rsecond;
    uint32_t open_rows[16];
  };
  std::vector<CalibEntry> calibs[16];

  // ----- Register file ------------------------------------------------
  uint32_t reg[16] = {0};
  // Wide-data buffer (16 × 32-bit slots) used by WRITE instructions.
  uint32_t wdata_slot[16] = {0};

  // ----- c2h response queue --------------------------------------------
  std::vector<uint8_t> response_queue;

  // ----- Program memory (the IMEM the SoftMC pipeline would run) ------
  std::vector<uint64_t> imem;

  // ----- Execution loop ------------------------------------------------
  void execute_program();
  void execute_inst(uint64_t inst, size_t& pc);
  void apply_ddr_pack4(uint64_t pack4);
  void apply_mininst(int slot, uint16_t mininst);
  // Single primitive ops on the DDR state.
  void apply_act(int bank, int row);
  void apply_pre(int bank);
  void apply_rd(int bank, int col);
  void apply_wr(int bank, int col);
  // doubleACT pattern matcher / dispatcher.
  void on_act_check_doubleACT(int bank, int row);
  void rowclone(int bank, uint32_t src_row, uint32_t dst_row);
  void broadcast(int bank, uint32_t src_row);
  void maj3_vote(int bank, uint32_t dst_row);
  // Find the calib whose open_rows[] contain `row`, OR whose (Rfirst,
  // Rsecond) include `row`, for a given bank. nullptr if not found.
  const CalibEntry* find_calib_by_open_row(int bank, uint32_t row) const;
  const CalibEntry* find_calib_by_first_or_second(int bank, uint32_t row) const;
  // Lazy row allocator.
  std::vector<uint8_t>& get_row(int bank, uint32_t row);
};
