// Golden-reference TB (2026-07-25). Drives ONE program in LEGACY
// cadence (send it, then drain), records every c2h byte and every DDR
// command, and writes both to files for a byte-level diff against our
// engine. The same binary builds against either top (ours or pristine)
// via -DTOP_HEADER, so the two runs are identical by construction:
// same program, same DRAM model, same pacing, same recording.
#include TOP_HEADER
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>

double sc_time_stamp() { return 0; }

static TOP_CLASS* top;
static uint64_t t = 0;
static std::vector<uint8_t> c2h;
static FILE* cmdlog;
static long drain_off = 0, dr_budget = 8, dr_pause = 0;

static void tick(){
  if (drain_off <= 0) top->c2h_tready = 1;
  else if (dr_pause > 0) { top->c2h_tready = 0; dr_pause--; }
  else { top->c2h_tready = 1;
         if (top->c2h_tvalid && --dr_budget <= 0) { dr_pause = drain_off; dr_budget = 8; } }
  top->clk = 0; top->eval();
  top->clk = 1; top->eval();
  // DDR command trace: what the PHY sees
  if (top->ddr_act || top->ddr_read || top->ddr_write || top->ddr_pre)
    fprintf(cmdlog, "%lu act=%x rd=%x wr=%x pre=%x\n", (unsigned long)t,
            top->ddr_act, top->ddr_read, top->ddr_write, top->ddr_pre);
  if (top->c2h_tvalid && top->c2h_tready)
    for (int i=0;i<8;i++){ uint32_t w = top->c2h_tdata[i];
      c2h.push_back((uint8_t)w);       c2h.push_back((uint8_t)(w>>8));
      c2h.push_back((uint8_t)(w>>16)); c2h.push_back((uint8_t)(w>>24)); }
  t++;
}

static std::vector<uint64_t> load_hex(const std::string& p){
  std::vector<uint64_t> v; std::ifstream f(p); std::string ln;
  while (std::getline(f, ln)) if (ln.size()>=16)
    v.push_back(strtoull(ln.c_str(), nullptr, 16));
  return v;
}

static bool beat(uint64_t w, bool last, long budget=20000){
  top->h2c_tdata[0] = (uint32_t)(w & 0xFFFFFFFFu);
  top->h2c_tdata[1] = (uint32_t)(w >> 32);
  for (int i=2;i<8;i++) top->h2c_tdata[i] = 0;
  top->h2c_tkeep = 0xFFFFFFFFu;
  top->h2c_tvalid = 1; top->h2c_tlast = last ? 1 : 0;
  long waited = 0;
  while (!top->h2c_tready){ tick(); if (++waited > budget){
      top->h2c_tvalid = 0; top->h2c_tlast = 0; return false; } }
  tick();
  top->h2c_tvalid = 0; top->h2c_tlast = 0;
  return true;
}

int main(int argc, char** argv){
  Verilated::commandArgs(argc, argv);
  const char* hexf = argc>1 ? argv[1] : "s1_read.hex";
  const char* outp = argc>2 ? argv[2] : "c2h.bin";
  const char* cmdp = argc>3 ? argv[3] : "cmd.log";
  int nprog        = argc>4 ? atoi(argv[4]) : 1;
  drain_off        = argc>5 ? atol(argv[5]) : 0;

  cmdlog = fopen(cmdp, "w");
  top = new TOP_CLASS;
  top->rst = 1; top->h2c_tvalid = 0; top->h2c_tlast = 0;
  top->init_calib_complete = 0;
  for (int i=0;i<30;i++) tick();
  top->rst = 0;
  for (int i=0;i<40;i++) tick();
  top->init_calib_complete = 1;
  // settle: let power-on maintenance finish so record 0 is not racing it
  for (int i=0;i<60000;i++) tick();

  auto prog = load_hex(hexf);
  if (prog.empty()){ printf("[ref] no hex %s\n", hexf); return 3; }
  printf("[ref] %s: %zu insts, %d programs, drain_off=%ld\n",
         hexf, prog.size(), nprog, drain_off);

  c2h.clear();
  for (int p = 0; p < nprog; p++){
    for (size_t i=0;i<prog.size();i++)
      if (!beat(prog[i], i+1==prog.size())) { printf("[ref] SEND STALL\n"); return 2; }
    // legacy cadence: drain this program fully before the next
    size_t before = c2h.size(); long quiet = 0;
    for (long i=0;i<400000 && quiet < 3000; i++){
      tick();
      if (c2h.size() != before) { before = c2h.size(); quiet = 0; } else quiet++;
    }
  }
  FILE* f = fopen(outp, "wb");
  fwrite(c2h.data(), 1, c2h.size(), f);
  fclose(f); fclose(cmdlog);
  printf("[ref] c2h bytes=%zu -> %s ; cmds -> %s ; cycles=%lu\n",
         c2h.size(), outp, cmdp, (unsigned long)t);
  return 0;
}
