// Hardware command-sequence engine for DRAM-Bender.
//
// Two emit modes — both bypass fetch/decode/execute and pack 4 DDR
// commands per fabric cycle (the architectural ceiling).
//
//   MODE_PLAIN (op_mode=0)
//     Streams `count` cycles of identical ops on `bank_mask` slots —
//     useful for back-to-back READ / WRITE / ACT / PRE bursts (e.g.
//     reading out N result rows or loading N weights).
//
//   MODE_DOUBLE_ACT (op_mode=1)
//     Emits the SiMRA charge-sharing doubleACT pattern that BitNet PIM's
//     MAJ3 / RowClone / broadcast primitives are built on. Pattern:
//         T0:        ACT  R_first
//         T0..t_12:  NOPs
//         T(t_12+1): PRE
//         T..t_23:   NOPs
//         T(t_12+t_23+2): ACT R_second
//     packed 4-mininst-per-fabric-cycle. Runtime params {t_12, t_23,
//     R_first, R_second, bank_mask, BG, BANK0_id} make this DIMM-portable
//     — the per-DIMM characterization sweep already produces the
//     timing/row pairs (RowClone uses 30/1, broadcast 10/2, MAJ3 0/0
//     on DIMM 0; the engine just consumes whatever the calibration
//     loader hands it).
//
// What makes this faster than a SoftMC loop:
//   * No instruction fetch/decode/execute per cycle — the pattern lives
//     in this module's state machine.
//   * No branch-resolution stall per loop iteration (~5-90 cycles in the
//     existing pipeline).
//   * Every fabric cycle carries 4 commands instead of 1 (the soft loop
//     emits primitive opcodes 1 per cycle, so 75% of slots are idle NOPs).
//
// Opcode interface:
//   start                — pulse high for one cycle to begin the burst
//   op_code [2:0]        — 0=READ, 1=WRITE, 2=PRE, 3=ACT (matches encoding.vh `READ/WRITE/PRE/ACT`)
//   bank_mask[3:0]       — which of the 4 slot lanes (= which bank groups)
//                          should issue commands. Set to 4'b1111 for full
//                          bank-parallel issue.
//   bank0_id [1:0]       — bank ID for slot 0 (other slots get bank0_id+slot)
//   base_col [9:0]       — starting column (used when op = READ/WRITE)
//   base_row [16:0]      — starting row (used when op = ACT)
//   count    [9:0]       — number of fabric cycles to issue commands for.
//                          Total commands emitted = count × popcount(bank_mask).
//   stride   [3:0]       — column / row increment per cycle (typical 1)
//   bl4                  — burst-chop flag (RD/WR only)
//   ap                   — auto-precharge flag (RD/WR only)
//
//   busy                 — high while burst is in flight; pipeline must
//                          NOT drive the ddr_* bus while busy is asserted.
//   done                 — single-cycle pulse when burst completes.

`include "encoding.vh"

module seq_engine(
  input                clk,
  input                rst,

  // opcode / start
  input                start,
  input  [1:0]         op_mode,        // 0 = PLAIN burst, 1 = DOUBLE_ACT
  // ---- PLAIN-mode params (also reused as ACT op for DOUBLE_ACT) ----
  input  [2:0]         op_code,
  input  [3:0]         bank_mask,
  input  [1:0]         bank0_id,
  input  [9:0]         base_col,
  input  [16:0]        base_row,
  input  [9:0]         count,
  input  [3:0]         stride,
  input                bl4,
  input                ap,
  // ---- DOUBLE_ACT-mode params ----
  input  [16:0]        da_r_first,
  input  [16:0]        da_r_second,
  input  [5:0]         da_t12,         // 0..63 NOPs between ACT1 and PRE
  input  [5:0]         da_t23,         // 0..63 NOPs between PRE and ACT2

  // observability
  output reg           busy,
  output reg           done,

  // DDR command bus (matches softmc_pipeline outputs exactly — same widths,
  // same slot semantics, so this can multiplex with execute_stage's ddr_*
  // outputs into ddr_pipeline.v)
  output reg [3:0]     ddr_act,
  output reg [3:0]     ddr_read,
  output reg [3:0]     ddr_write,
  output reg [3:0]     ddr_pre,
  output reg [3:0]     ddr_nop,
  output reg [3:0]     ddr_ap,
  output reg [3:0]     ddr_pall,
  output reg [3:0]     ddr_half_bl,
  output reg [4*2-1:0] ddr_bg,
  output reg [4*2-1:0] ddr_bank,
  output reg [4*10-1:0] ddr_col,
  output reg [4*17-1:0] ddr_row
);

  // Latched opcode params (held for the burst duration).
  reg [1:0]   mode_r;
  reg [2:0]   op_r;
  reg [3:0]   mask_r;
  reg [1:0]   bank0_r;
  reg [9:0]   base_col_r;
  reg [16:0]  base_row_r;
  reg [9:0]   count_r;
  reg [9:0]   ctr_r;          // cycles emitted so far
  reg [3:0]   stride_r;
  reg         bl4_r;
  reg         ap_r;
  // DOUBLE_ACT-mode latches (staggered multi-bank scheme)
  reg [16:0]  da_rf_r;
  reg [16:0]  da_rs_r;
  reg [6:0]   da_t12_r;
  reg [6:0]   da_t23_r;
  reg [7:0]   da_pat_len_r;     // = t_12 + t_23 + 3 (≤ 129)
  reg [2:0]   da_n_banks_r;     // popcount(bank_mask) (1..4)
  reg [9:0]   da_total_cyc_r;   // = pat_len * n_banks / 4 (rounded up)
  reg [9:0]   da_pre_cyc;       // unused (kept for legacy reset block)
  reg [9:0]   da_act2_cyc;      // unused (kept for legacy reset block)

  localparam MODE_PLAIN     = 2'd0;
  localparam MODE_DOUBLE_ACT = 2'd1;

  // Per-slot output for one cycle. Combinational from the latched state +
  // current ctr_r value.
  integer s;
  always @* begin
    ddr_act     = 4'b0;
    ddr_read    = 4'b0;
    ddr_write   = 4'b0;
    ddr_pre     = 4'b0;
    ddr_nop     = 4'b0;
    ddr_ap      = 4'b0;
    ddr_pall    = 4'b0;
    ddr_half_bl = 4'b0;
    ddr_bg      = 8'b0;
    ddr_bank    = 8'b0;
    ddr_col     = 40'b0;
    ddr_row     = 68'b0;
    if (busy) begin
      if (mode_r == MODE_PLAIN) begin
        for (s = 0; s < 4; s = s + 1) begin
          if (mask_r[s]) begin
            ddr_bank[s*2 +: 2] = bank0_r + s[1:0];
            ddr_bg[s*2   +: 2] = 2'b0;
            ddr_col[s*10 +: 10] = base_col_r + (ctr_r * stride_r);
            ddr_row[s*17 +: 17] = base_row_r + (ctr_r * stride_r);
            case (op_r)
              `READ:  begin ddr_read [s] = 1'b1;
                            ddr_ap   [s] = ap_r;
                            ddr_half_bl[s] = bl4_r; end
              `WRITE: begin ddr_write[s] = 1'b1;
                            ddr_ap   [s] = ap_r;
                            ddr_half_bl[s] = bl4_r; end
              `PRE:   ddr_pre[s] = 1'b1;
              `ACT:   ddr_act[s] = 1'b1;
              default: ddr_nop[s] = 1'b1;
            endcase
          end else begin
            ddr_nop[s] = 1'b1;
          end
        end
      end else begin
        // MODE_DOUBLE_ACT — STAGGERED multi-bank emission.
        //
        // Each enabled bank gets its own copy of the SiMRA doubleACT
        // pattern (length = t_12 + t_23 + 3) laid out back-to-back in the
        // command stream. Bank N's pattern occupies absolute positions
        //     [N*pat_len .. (N+1)*pat_len - 1]
        // with active events at offsets 0, t_12+1, t_12+t_23+2 inside its
        // own slice. Per-bank t_12 / t_23 dwell is BIT-EXACT — exactly
        // what the SiMRA `doubleACT()` helper emits — so the SiMRA
        // calibration values keep working. The win is that all 4 banks'
        // patterns share the same 4-slot command bus continuously.
        for (s = 0; s < 4; s = s + 1) begin: dam
          reg [11:0] abs_pos;
          reg [3:0]  bank_idx;
          reg [7:0]  in_bank_pos;
          abs_pos = (ctr_r * 4) + s[2:0];
          // bank_idx = abs_pos / pat_len  ;  in_bank_pos = abs_pos % pat_len
          // Inline divide by sequential subtraction is too slow for synth;
          // use a small static unrolled compare since pat_len ≤ 129 and
          // we only need bank_idx ≤ 3. (Verilator handles this fine.)
          if      (abs_pos < {4'b0, da_pat_len_r})           begin bank_idx = 0; in_bank_pos = abs_pos[7:0]; end
          else if (abs_pos < ({4'b0, da_pat_len_r} << 1))    begin bank_idx = 1; in_bank_pos = abs_pos[7:0] - da_pat_len_r; end
          else if (abs_pos < ({4'b0, da_pat_len_r} * 12'd3)) begin bank_idx = 2; in_bank_pos = abs_pos[7:0] - (da_pat_len_r << 1); end
          else                                                begin bank_idx = 3; in_bank_pos = abs_pos[7:0] - (da_pat_len_r * 8'd3); end

          if (bank_idx[1:0] < da_n_banks_r[1:0] || da_n_banks_r == 3'd4) begin
            ddr_bank[s*2 +: 2] = bank0_r + bank_idx[1:0];
            ddr_bg[s*2   +: 2] = 2'b0;
            if (in_bank_pos == 8'd0) begin
              ddr_act[s] = 1'b1;
              ddr_row[s*17 +: 17] = da_rf_r;
            end else if (in_bank_pos == {1'b0, da_t12_r} + 8'd1) begin
              ddr_pre[s] = 1'b1;
            end else if (in_bank_pos == {1'b0, da_t12_r} + {1'b0, da_t23_r} + 8'd2) begin
              ddr_act[s] = 1'b1;
              ddr_row[s*17 +: 17] = da_rs_r;
            end else begin
              ddr_nop[s] = 1'b1;
            end
          end else begin
            ddr_nop[s] = 1'b1;
          end
        end
      end
    end
  end

  // Helper: ceil(x / 4), x ≤ 1023
  function [9:0] ceil_div4;
    input [9:0] x;
    ceil_div4 = (x + 10'd3) >> 2;
  endfunction

  always @(posedge clk) begin
    if (rst) begin
      busy       <= 1'b0;
      done       <= 1'b0;
      mode_r     <= 2'b0;
      op_r       <= 3'b0;
      mask_r     <= 4'b0;
      bank0_r    <= 2'b0;
      base_col_r <= 10'b0;
      base_row_r <= 17'b0;
      count_r    <= 10'b0;
      ctr_r      <= 10'b0;
      stride_r   <= 4'b0;
      bl4_r      <= 1'b0;
      ap_r       <= 1'b0;
      da_rf_r        <= 17'b0;
      da_rs_r        <= 17'b0;
      da_t12_r       <= 7'b0;
      da_t23_r       <= 7'b0;
      da_pat_len_r   <= 8'd3;
      da_n_banks_r   <= 3'd1;
      da_total_cyc_r <= 10'd1;
      da_pre_cyc     <= 10'b0;
      da_act2_cyc    <= 10'b0;
    end else begin
      done <= 1'b0;
      if (start && !busy) begin
        mode_r     <= op_mode;
        op_r       <= op_code;
        mask_r     <= bank_mask;
        bank0_r    <= bank0_id;
        base_col_r <= base_col;
        base_row_r <= base_row;
        stride_r   <= stride;
        bl4_r      <= bl4;
        ap_r       <= ap;
        ctr_r      <= 10'b0;
        if (op_mode == MODE_DOUBLE_ACT) begin
          da_rf_r      <= da_r_first;
          da_rs_r      <= da_r_second;
          da_t12_r     <= da_t12[5:0] | 7'b0;
          da_t23_r     <= da_t23[5:0] | 7'b0;
          da_pat_len_r <= 8'd3 + {2'b0, da_t12} + {2'b0, da_t23};
          // popcount(bank_mask) — only 4 bits.
          da_n_banks_r <= {2'b0, bank_mask[0]} + {2'b0, bank_mask[1]}
                        + {2'b0, bank_mask[2]} + {2'b0, bank_mask[3]};
          // total_cyc = ceil((pat_len * n_banks) / 4)
          da_total_cyc_r <= ((( {2'b0, 8'd3 + {2'b0, da_t12} + {2'b0, da_t23}} )
                              * ({7'b0, bank_mask[0]} + {7'b0, bank_mask[1]}
                                + {7'b0, bank_mask[2]} + {7'b0, bank_mask[3]}))
                            + 10'd3) >> 2;
          count_r        <= 10'd0;
          busy           <= 1'b1;
        end else begin
          count_r        <= count;
          busy           <= (count != 10'b0);
        end
      end else if (busy) begin
        if ((mode_r == MODE_PLAIN     && ctr_r + 10'd1 == count_r) ||
            (mode_r == MODE_DOUBLE_ACT && ctr_r + 10'd1 == da_total_cyc_r)) begin
          busy <= 1'b0;
          done <= 1'b1;
          ctr_r <= 10'b0;
        end else begin
          ctr_r <= ctr_r + 10'd1;
        end
      end
    end
  end
endmodule
