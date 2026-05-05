// Accumulates a 16-bit popcount stream across N reads, returning the running
// total on a single-cycle `drain` pulse. Used by readback_engine when
// running in the new ACCUM mode: instead of streaming per-read popcounts
// (16b each, packed into 32B AXI beats by diff_shift_reg) over c2h, the
// FPGA aggregates and emits ONE total per matmul.
//
// Bandwidth math for BitNet's typical MAJ3 matmul shape (~4096 reads of 16b
// popcounts = 8 KiB c2h per matmul today): one 4 B accumulator drain
// collapses the c2h volume by 2048× per matmul, eliminating the readback
// FIFO back-pressure on fetch and the per-matmul DMA round-trip.
//
// `ACCUM_WIDTH` defaults to 32 b (handles N up to 65 536 reads × 16 b max).
// The drain pulse semantics: on the cycle `drain` is asserted, the output
// becomes (current sum + in_value if in_valid that cycle) and accum_valid
// rises for one cycle; the internal sum resets to zero on the next cycle.

module popcount_accum #(
    parameter ACCUM_WIDTH = 32
)(
  input                              clk,
  input                              rst,

  // Input stream — width matches the 16-bit pop_count_value in
  // readback_engine.v.
  input      [15:0]                  in_value,
  input                              in_valid,

  // Single-cycle pulse: emit the running sum, reset internal counter.
  input                              drain,

  // Output (registered, held until next drain pulse).
  output reg [ACCUM_WIDTH-1:0]       accum_out,
  output reg                         accum_valid
);

  reg [ACCUM_WIDTH-1:0] sum_r;

  always @(posedge clk) begin
    if (rst) begin
      sum_r       <= {ACCUM_WIDTH{1'b0}};
      accum_out   <= {ACCUM_WIDTH{1'b0}};
      accum_valid <= 1'b0;
    end else begin
      if (drain) begin
        // Capture the *current* in-flight value as well so a drain that
        // coincides with a valid input doesn't lose that sample.
        accum_out   <= sum_r + (in_valid ? {{(ACCUM_WIDTH-16){1'b0}}, in_value}
                                         : {ACCUM_WIDTH{1'b0}});
        accum_valid <= 1'b1;
        sum_r       <= {ACCUM_WIDTH{1'b0}};
      end else begin
        accum_valid <= 1'b0;
        if (in_valid)
          sum_r <= sum_r + {{(ACCUM_WIDTH-16){1'b0}}, in_value};
      end
    end
  end

endmodule
