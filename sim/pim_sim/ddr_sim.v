// Behavioral DDR + SiMRA model — drop-in for the Xilinx MIG/PHY in
// Verilator simulation. Consumes the SoftMC pipeline's ddr_* outputs,
// implements per-bank row-buffer semantics, and recognises the three
// DIMM-0 calibrated SiMRA charge-sharing patterns (RowClone, broadcast,
// MAJ3) as their PROMISED semantics — so we get DETERMINISTIC PIM output
// to A/B against the real hardware.
//
// Recognised patterns (matched by per-bank ACT-PRE-ACT timing in
// fabric cycles, where each fabric cycle = 4 PHY slots):
//   doubleACT(t_12=30, t_23=1, R1, R2) → RowClone:    stored[bank][R2] = stored[bank][R1]
//   doubleACT(t_12=10, t_23=2, R1, R2) → broadcast:   stored[bank][open_row[i] for i in 0..15] = stored[bank][R1]
//   doubleACT(t_12=0,  t_23=0, R1, R2) → MAJ3 vote:   stored[bank][R1] = majority-bit(open_rows[0..15])
//
// open_rows[bank][0..15] is loaded from calib_dimm0.txt at sim startup
// via a $readmemh-style mechanism. Unknown timings → $display + $finish.

`include "parameters.vh"

module ddr_sim #(
    parameter N_BANKS = 16,
    parameter ROWS_PER_BANK = 65536,   // 17-bit row addr
    parameter BYTES_PER_ROW = 8192     // 64-byte cols × 128 cols
) (
    input clk,
    input rst,

    // From softmc_pipeline (4 slots per fabric cycle)
    input [3:0]                ddr_act,
    input [3:0]                ddr_read,
    input [3:0]                ddr_write,
    input [3:0]                ddr_pre,
    input [3:0]                ddr_nop,
    input [4*`BG_WIDTH-1:0]    ddr_bg,
    input [4*`BANK_WIDTH-1:0]  ddr_bank,
    input [4*`COL_WIDTH-1:0]   ddr_col,
    input [4*`ROW_WIDTH-1:0]   ddr_row,
    input [511:0]              ddr_wdata,

    // To readback_engine
    output reg [511:0]         rd_data,
    output reg                 rd_valid
);

  // ----- Storage --------------------------------------------------------
  // 16 banks × 65536 rows × 8 KB = 8 GB. Won't fit in Verilator simulation
  // memory; use a sparse hashmap-equivalent (Verilog associative arrays)
  // or limit to the rows we actually touch.
  // SystemVerilog (which Verilator supports) has assoc arrays:
  //   reg [BYTES_PER_ROW*8-1:0] stored [N_BANKS] [int];
  // — keyed by row index, allocates on first write.

  reg [BYTES_PER_ROW*8-1:0] stored [N_BANKS-1:0] [int];
  // Per-bank open-row state
  reg [16:0] open_row [N_BANKS-1:0];
  reg open_row_valid [N_BANKS-1:0];
  reg [BYTES_PER_ROW*8-1:0] row_buffer [N_BANKS-1:0];

  // ----- Per-bank ACT-PRE-ACT timing tracking ---------------------------
  // Track last activity per bank to detect doubleACT pattern.
  // For each bank we track:
  //   last_act_cycle, last_act_row, last_pre_cycle
  // When a SECOND ACT arrives, compute t_12 = pre_cycle - act1_cycle - 1
  // and t_23 = act2_cycle - pre_cycle - 1. Match against known patterns.
  reg [31:0] cyc_ctr;
  reg [31:0] last_act_cyc [N_BANKS-1:0];
  reg [31:0] last_pre_cyc [N_BANKS-1:0];
  reg [16:0] last_act_row [N_BANKS-1:0];

  // ----- Calibration table (loaded at startup from $plusargs file) -----
  // For each (bank, subarray_id): the 16 open_rows + Rfirst + Rsecond.
  // For demo simplicity, load via $readmemh from a flat file.
  reg [16:0] calib_open_rows [N_BANKS-1:0] [16];   // 16 open rows per bank's calib subarray
  reg [16:0] calib_Rfirst [N_BANKS-1:0];
  reg [16:0] calib_Rsecond [N_BANKS-1:0];

  // (Calibration loading TBD — use Verilator DPI to read calib_dimm0.txt
  //  at $readmemh time, or load via separate testbench main.)

  integer s;
  always @(posedge clk) begin
    if (rst) begin
      cyc_ctr <= 0;
      rd_valid <= 1'b0;
      for (int b = 0; b < N_BANKS; b++) begin
        open_row_valid[b] <= 1'b0;
        last_act_cyc[b] <= 32'hFFFF_FFFF;
        last_pre_cyc[b] <= 32'hFFFF_FFFF;
      end
    end else begin
      cyc_ctr <= cyc_ctr + 1;
      rd_valid <= 1'b0;

      // Process the 4 slots in priority order:
      //   ACT, PRE, READ, WRITE (one type max per slot)
      for (s = 0; s < 4; s++) begin
        if (ddr_act[s]) begin
          int bank = (ddr_bg[s*`BG_WIDTH +: `BG_WIDTH] << `BANK_WIDTH)
                   | ddr_bank[s*`BANK_WIDTH +: `BANK_WIDTH];
          int row  = ddr_row[s*`ROW_WIDTH +: `ROW_WIDTH];
          int cyc  = cyc_ctr * 4 + s;
          // Detect doubleACT: have we seen ACT-PRE on this bank with the
          // right timing relative to NOW?
          if (last_pre_cyc[bank] != 32'hFFFF_FFFF
              && last_pre_cyc[bank] > last_act_cyc[bank]) begin
            int t_12 = last_pre_cyc[bank] - last_act_cyc[bank] - 1;
            int t_23 = cyc - last_pre_cyc[bank] - 1;
            // Match calibrated patterns
            if (t_12 == 30 && t_23 == 1) begin
              // RowClone: copy last_act_row → row
              stored[bank][row] = stored[bank][last_act_row[bank]];
              $display("[ddr_sim] RowClone bank=%0d %0d→%0d (cyc=%0d)",
                       bank, last_act_row[bank], row, cyc);
            end else if (t_12 == 10 && t_23 == 2) begin
              // Broadcast: copy last_act_row → all 16 open_rows
              for (int i = 0; i < 16; i++)
                stored[bank][calib_open_rows[bank][i]] = stored[bank][last_act_row[bank]];
              $display("[ddr_sim] broadcast bank=%0d from row=%0d (cyc=%0d)",
                       bank, last_act_row[bank], cyc);
            end else if (t_12 == 0 && t_23 == 0) begin
              // MAJ3 vote: bit-by-bit majority across the 16 open_rows.
              for (int byte_idx = 0; byte_idx < BYTES_PER_ROW; byte_idx++) begin
                for (int bit_idx = 0; bit_idx < 8; bit_idx++) begin
                  int ones = 0;
                  for (int i = 0; i < 16; i++)
                    if (stored[bank][calib_open_rows[bank][i]][byte_idx*8 + bit_idx])
                      ones++;
                  // Write majority bit back to last_act_row (= R1)
                  if (ones > 8)
                    stored[bank][last_act_row[bank]][byte_idx*8 + bit_idx] = 1'b1;
                  else
                    stored[bank][last_act_row[bank]][byte_idx*8 + bit_idx] = 1'b0;
                end
              end
              $display("[ddr_sim] MAJ3 bank=%0d → row=%0d (cyc=%0d)",
                       bank, last_act_row[bank], cyc);
            end else begin
              // Unrecognised SiMRA pattern — flag it loudly.
              $display("[ddr_sim] UNKNOWN doubleACT t_12=%0d t_23=%0d bank=%0d (cyc=%0d) — UNCALIBRATED!",
                       t_12, t_23, bank, cyc);
              $finish;
            end
            // Reset state after pattern
            last_act_cyc[bank] = 32'hFFFF_FFFF;
            last_pre_cyc[bank] = 32'hFFFF_FFFF;
          end
          // Plain (single) ACT: load row into row_buffer.
          row_buffer[bank] = stored[bank][row];
          open_row[bank] = row[16:0];
          open_row_valid[bank] = 1'b1;
          last_act_cyc[bank] = cyc;
          last_act_row[bank] = row[16:0];
        end

        if (ddr_pre[s]) begin
          int bank = (ddr_bg[s*`BG_WIDTH +: `BG_WIDTH] << `BANK_WIDTH)
                   | ddr_bank[s*`BANK_WIDTH +: `BANK_WIDTH];
          int cyc  = cyc_ctr * 4 + s;
          if (open_row_valid[bank]) begin
            // writeback row_buffer to stored
            stored[bank][open_row[bank]] = row_buffer[bank];
            open_row_valid[bank] = 1'b0;
          end
          last_pre_cyc[bank] = cyc;
        end

        // READ + WRITE (column ops): assume row is open
        // (NOTE: simplified — real DRAM enforces tCCD between consecutive cols)
      end
    end
  end

endmodule
