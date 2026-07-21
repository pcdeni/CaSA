`include "parameters.vh"

// Process data coming from DRAM before sending it to the host.
module readback_engine(
  
  // common signals
  input     clk,
  input     rst,
  
  // other control signals
  input         flush,
  input         read_seq_incoming, // next few instructions will read from DRAM
  input [11:0]  incoming_reads,    // how many reads next few instructions will issue
  output[11:0]  buffer_space,      // remaining buffer size
  input         switch_mode,       // legacy toggle (racy control word, kept for compat)
  input         set_mode_read,     // build4: idempotent SET-READ control word
  input         set_mode_diff,     // build4: idempotent SET-DIFF control word
  input         set_mode_segpop,   // build7: idempotent SET-SEGPOP control word
  
  // DRAM <-> engine if
  input [511:0] rd_data,
  input         rd_valid,
  
  input         per_rd_init,
  input         per_zq_init,
  input         per_ref_init,
  
  // engine <-> regfile if
  input [511:0] ddr_wdata, // to compare read data against
  
  // readback <-> XDMA if
  output [`XDMA_AXI_DATA_WIDTH-1:0]   c2h_tdata_0,  
  output                              c2h_tlast_0,
  output                              c2h_tvalid_0,
  input                               c2h_tready_0,
  output [`XDMA_AXI_DATA_WIDTH/8-1:0] c2h_tkeep_0
  
  );
  
  
  localparam READ_MODE   = 2'd0;
  localparam DIFF_MODE   = 2'd1;
  localparam SEG_POP_MODE = 2'd2;   // build7: per-32b-segment popcount readout
  reg [1:0] mode_r, mode_ns; // READ / DIFF-accum / SEG_POP
  
  reg rd_valid_r;
  reg ignore_read_r, ignore_read_ns;

  // ------------------------------------------------------------------
  // build3 (2026-07-18): accumulator drain-race fix.
  //
  // Old scheme: a single ignore_flush_r flag, set by any maintenance
  // init (per_rd/zq/ref), cleared (and the flush EATEN) by the next
  // flush. Two hazards, both observed on silicon (RESULT.md addendum
  // 20, boot-phase-dependent drain delivery):
  //   1. flush = frontend_ready is softmc_fin delayed 32 cycles and can
  //      be asserted for >1 cycle. The old level-keyed logic then
  //      cleared the flag on cycle 1 and PROCESSED the same flush on
  //      cycle 2 (spurious trailer/drain), and re-drained the
  //      accumulator every remaining flush cycle (spurious zero
  //      chunks).
  //   2. A single flag cannot count >1 outstanding maintenance event,
  //      and any accounting imbalance makes it eat a USER flush: no
  //      proc_flush -> no trailer, and no accum drain -> lost chunk.
  //
  // New scheme:
  //   - flush_edge: all flush consumption keys on the rising edge, so
  //     flush width no longer matters.
  //   - ignore_flush_ctr: 4-bit SATURATING event counter. +1 per
  //     maintenance init event edge, -1 per eaten flush edge.
  //   - POPCOUNT_ACCUM_MODE + DIFF_MODE only: accum_armed_r (>=1
  //     accumulated sample since last drain) identifies USER flushes
  //     independently of the counter, because maintenance reads never
  //     reach the accumulator (diff_valid gates on ~ignore_read_r).
  //     An armed flush is always delivered, so stale maintenance
  //     accounting (e.g. a maintenance program killed by user_rst
  //     after its init but before its own flush) can never eat a user
  //     result again. Unarmed flushes follow counter accounting, so
  //     healthy maintenance flushes stay eaten and produce neither
  //     chunk nor trailer.
  //
  // build4 (2026-07-20): DRAIN-CAPTURE TIMING fix (RESULT.md addendum
  // 20c — the one remaining blocker after build3).
  //
  // Silicon evidence: totals correct only when programs are seconds
  // apart; back-to-back/batched programs capture zeros; combined long
  // programs fail ~always (collision probability ∝ program length).
  //
  // Root cause (RTL-provable): flush = frontend_ready = softmc_fin
  // delayed 32 cycles, and softmc_fin fires when the END word is
  // FETCHED (fetch_stage.v: softmc_end = is_end && valid_in) — while
  // previously fetched READ commands are still queued in decode/exe/
  // ddr_pipeline and their DATA returns through the PHY even later.
  // The 32-cycle delay is a settle heuristic, not a bound: whenever
  // the fin→last-rd_valid gap exceeds it (deep DDR-pipeline queue on
  // long programs, pacing differences when batched), build3 drained a
  // PARTIAL or EMPTY accumulator at flush_edge, and the tail reads
  // then leaked into the NEXT program's total. Note popcount_accum
  // already registers accum_out at drain, so a plain holding register
  // latched at flush_EDGE would capture the same wrong instant — the
  // capture must move to the program's own end-of-reads.
  //
  // Fix: defer the capture until the program's reads are DONE.
  //   - rd_outstanding_r: announced-but-not-returned read counter fed
  //     by the exact signals buffer_space accounting already uses
  //     (read_seq_incoming pulses with incoming_reads = the fetched
  //     SMC_INFO packet's read count) minus one per rd_valid return.
  //     Maintenance per_rd reads return WITHOUT an announcement; the
  //     floor-at-zero handles them (they can only arrive when nothing
  //     user-announced is outstanding, because the frontend serializes
  //     programs and the PHY returns data in CAS order).
  //   - flush_proc (build3 accounting, unchanged) no longer drains
  //     directly in accum mode: it sets capture_pending_r.
  //   - The drain (capture) fires on the first cycle the read path is
  //     QUIET: rd_outstanding==0 and no in-flight sample anywhere in
  //     the rd_valid → diff_valid → pop_count_valid pipe. When the
  //     path is already quiet at flush_edge (the paced/solo case) the
  //     capture fires the SAME cycle — bit-identical to build3 timing.
  //   - The trailer beat waits for the capture (proc_flush_r &&
  //     ~capture_pending_r), so chunk-then-trailer framing holds even
  //     when the capture is deferred.
  //   - Safety valve: if a capture stays pending for 4096 cycles
  //     (~14 µs — far beyond any legitimate read tail) it force-fires
  //     so the c2h stream keeps its framing even if read returns are
  //     lost; the totals of such a wedged window are best-effort.
  //   - Exactly one capture per processed flush: capture_pending_r is
  //     set by flush_proc, cleared by the single-cycle capture_fire,
  //     and accum_armed_r clears at capture_fire (samples landing
  //     between flush_edge and capture belong to the current program
  //     and are folded into its total).
  //   READ_MODE is untouched: capture_pending_r can only be set in
  //   DIFF mode; in READ_MODE flush_proc drives proc_flush directly
  //   as in build3 and the FIFO-write path ignores the accum entirely.
  //
  // build4 also hardens the mode switch: set_mode_read/set_mode_diff
  // are idempotent level-set requests decoded by the frontend from two
  // NEW control words (bits INSTR_WIDTH+5/+6; +4 is taken by the
  // HBM_BENDER temp-read word). The legacy INSTR_WIDTH+1 toggle is
  // kept for compat. Trailer magic: build4 = 0xDBC0DE02,
  // build5 = 0xDBC0DE03 (FWFT-safe trailer framing),
  // build6 = 0xDBC0DE04 (buffer_space conservation in DIFF mode) so the host can
  // identify the image.
  // ------------------------------------------------------------------
  reg        flush_r;
  wire       flush_edge = flush && !flush_r;
  reg        per_rd_init_r, per_zq_init_r, per_ref_init_r;
  wire       maint_any  = per_rd_init || per_zq_init || per_ref_init;
  reg        maint_any_r;
  wire       maint_edge = maint_any && !maint_any_r;
  reg [3:0]  ignore_flush_ctr_r, ignore_flush_ctr_ns;
  reg        flush_eaten; // comb: this flush_edge consumed by maintenance accounting
`ifdef POPCOUNT_ACCUM_MODE
  reg        accum_armed_r; // >=1 DIFF sample accumulated since last drain
  wire       flush_proc = flush_edge && ((mode_r == DIFF_MODE && accum_armed_r)
                                         || ignore_flush_ctr_r == 4'd0);
  // build4: reads-outstanding accounting + deferred capture.
  reg [15:0] rd_outstanding_r, rd_outstanding_ns;
  reg        capture_pending_r, capture_pending_ns;
  reg [11:0] capture_age_r;   // pending-age safety valve (saturates -> force)
  wire       capture_force;
  wire       capture_fire;
  // build5 (2026-07-20): FWFT-fill-latency-safe trailer framing (RESULT.md
  // addendum 20e). The accum drain (dsr_valid) writes ONE 512b word = TWO
  // 256b c2h beats into rbf. The real Xilinx rdback_fifo (First_Word_Fall_
  // Through) reports `empty` HIGH for several cycles after wr_en before the
  // word falls through to `dout` — so the build3/4 trailer_beat, gated only
  // on rbf_empty, fired INSIDE that window, emitting the trailer BEFORE its
  // own chunk surfaced. On silicon that offset every accum message by one
  // program (trailer of prog i delivered with the chunk of prog i-1),
  // deadlocking the host drain after ~7 programs and wedging c2h. The
  // 0-latency behavioral sim FIFO hid it (build4 passed 53/53). Fix: count
  // the chunk's two c2h beats OUT of the FIFO before allowing the trailer.
  reg [1:0]  chunk_beats_r, chunk_beats_ns;
`else
  wire       flush_proc = flush_edge && (ignore_flush_ctr_r == 4'd0);
`endif
  // build3 debug counters, exposed on the trailer beat (unchanged
  // layout in build4; only the magic word is bumped):
  // {cnt_accum_write, cnt_drain, cnt_flush_eaten, cnt_flush_edge,
  //  cnt_ref_init, cnt_zq_init, cnt_rd_init, 32'hDBC0DE04}
  (*KEEP = "TRUE"*) reg [31:0] cnt_rd_init;
  (*KEEP = "TRUE"*) reg [31:0] cnt_zq_init;
  (*KEEP = "TRUE"*) reg [31:0] cnt_ref_init;
  (*KEEP = "TRUE"*) reg [31:0] cnt_flush_edge;
  (*KEEP = "TRUE"*) reg [31:0] cnt_flush_eaten;
  (*KEEP = "TRUE"*) reg [31:0] cnt_drain;
  (*KEEP = "TRUE"*) reg [31:0] cnt_accum_write;

  // Popcount computation part
  reg[511:0] read_diff;
  reg        diff_valid;
  always @(posedge clk) begin
    if(rst) begin
      read_diff <= 512'bX;
      diff_valid <= `LOW;
    end
    read_diff <= rd_valid ? rd_data ^ ddr_wdata : read_diff;
    diff_valid <= rd_valid && ~ignore_read_r && mode_r == DIFF_MODE ? `HIGH : `LOW;
  end

  genvar pcs; // popcount modules

  wire[2:0] pc_out [127:0];
  reg[3:0] pc_out_l2 [63:0];
  reg[4:0] pc_out_l3 [31:0];
  reg[5:0] pc_out_l4 [15:0];
  reg[6:0] pc_out_l5 [7:0];
  reg[7:0] pc_out_l6 [3:0];
  reg[8:0] pc_out_l7 [1:0];
  reg[15:0] pop_count_value;
  reg       pop_count_valid;
  
  generate
    for(pcs = 0 ; pcs < 128 ; pcs = pcs + 1) begin: gen_pcs
      pop_count4 pci
      (
        .in(read_diff[pcs*4 +: 4]),
        .out(pc_out[pcs])
      );
    end
  endgenerate

  integer l1, l2, l3, l4, l5, l6;
  always @* begin
    for(l1 = 0 ; l1 < 64 ; l1 = l1+1)
      pc_out_l2[l1] = pc_out[2*l1] + pc_out[2*l1+1];
    for(l2 = 0 ; l2 < 32 ; l2 = l2+1)
      pc_out_l3[l2] = pc_out_l2[2*l2] + pc_out_l2[2*l2+1];
    for(l3 = 0 ; l3 < 16 ; l3 = l3+1)
      pc_out_l4[l3] = pc_out_l3[2*l3] + pc_out_l3[2*l3+1];
    for(l4 = 0 ; l4 < 8 ; l4 = l4+1)
      pc_out_l5[l4] = pc_out_l4[2*l4] + pc_out_l4[2*l4+1];
    for(l5 = 0 ; l5 < 4 ; l5 = l5+1)
      pc_out_l6[l5] = pc_out_l5[2*l5] + pc_out_l5[2*l5+1];
    for(l6 = 0 ; l6 < 2 ; l6 = l6+1)
      pc_out_l7[l6] = pc_out_l6[2*l6] + pc_out_l6[2*l6+1];
  end
  
  always @(posedge clk) begin
    if(rst) begin
      pop_count_value <= 16'bX;
      pop_count_valid <= `LOW;
    end
    else begin
      pop_count_value <= diff_valid ? pc_out_l7[0] + pc_out_l7[1] : pop_count_value;
      pop_count_valid <= diff_valid ? `HIGH : `LOW;
    end
  end

  // ================= build7: SEG_POP per-segment readout =================
  // pc_out_l4[15:0] are the 16 per-32b-segment popcounts of read_diff
  // (= rd_data ^ ddr_wdata; the server writes ddr_wdata=0, so these are
  // popcounts of the raw product row). Pack each to a byte, assemble 4
  // read-beats' 16 bytes each into one 512b FIFO word (2048 B per 128-beat
  // row = 4x collapse vs the 8 KB raw row). Framing mirrors READ_MODE; the
  // DIFF accum/capture path stays dormant (all its gates are
  // mode==DIFF_MODE). seg_beat_valid aligns with read_diff, exactly as
  // diff_valid does (one cycle after rd_valid).
  reg         seg_beat_valid;
  reg [511:0] seg_sr;          // 4 beats x 128b, LSB beat first
  reg [1:0]   seg_cnt;         // 0..3 beats accumulated
  reg [511:0] seg_word;        // assembled FIFO word
  reg         seg_word_valid;  // one-cycle pulse: seg_word ready for the FIFO
  reg [127:0] seg_beat_bytes;
  integer sgi;
  always @* begin
    for(sgi = 0; sgi < 16; sgi = sgi + 1)
      seg_beat_bytes[sgi*8 +: 8] = {2'b0, pc_out_l4[sgi]};  // 6b popcount -> byte
  end
  always @(posedge clk) begin
    if(rst) begin
      seg_beat_valid <= `LOW;
      seg_sr         <= 512'b0;
      seg_cnt        <= 2'd0;
      seg_word       <= 512'b0;
      seg_word_valid <= `LOW;
    end
    else begin
      // align to read_diff, same as diff_valid (line above): a beat is a
      // SEG_POP user read whose per-segment popcounts are now on pc_out_l4.
      seg_beat_valid <= rd_valid && ~ignore_read_r && (mode_r == SEG_POP_MODE);
      seg_word_valid <= `LOW;
      if(seg_beat_valid) begin
        // shift the new beat's 16 bytes into the high lane; beat 0 ends in
        // bits [127:0] after 4 shifts (natural: segment g = beat*16+lane).
        seg_sr <= {seg_beat_bytes, seg_sr[511:128]};
        if(seg_cnt == 2'd3) begin
          seg_word       <= {seg_beat_bytes, seg_sr[511:128]};
          seg_word_valid <= `HIGH;
          seg_cnt        <= 2'd0;
        end
        else
          seg_cnt <= seg_cnt + 2'd1;
      end
    end
  end
  // ============== end build7 SEG_POP =====================================
  
  // Popcount-output staging:
  //   default: stream per-read 16b popcounts via diff_shift_reg → 512b chunks.
  //   POPCOUNT_ACCUM_MODE: accumulate the entire matmul into one 32b sum,
  //     emit ONE 512b chunk per program flush. For BitNet's typical MAJ3
  //     (4096 reads → 8 KB c2h today) this drops c2h volume by ~2000×,
  //     killing the readback-FIFO back-pressure on fetch and the per-MAJ3
  //     DMA round-trip cost.
  wire[511:0] dsr_out;
  wire        dsr_valid;
`ifdef POPCOUNT_ACCUM_MODE
  // build4: the read path is QUIET when nothing announced is
  // outstanding and no sample is in flight anywhere in the 3-stage
  // rd_valid -> diff_valid -> pop_count_valid pipe (the last sample is
  // folded into the accumulator on pop_count_valid's cycle, which a
  // same-cycle drain also captures via popcount_accum's in-flight
  // term).
  wire rd_path_quiet = (rd_outstanding_r == 16'd0) && ~rd_valid
                       && ~diff_valid && ~pop_count_valid;
  assign capture_force = capture_pending_r && (&capture_age_r);
  // The capture (accumulator drain) fires on the first quiet cycle at
  // or after a processed flush edge. Paced/solo programs are quiet AT
  // the flush edge, so the capture fires that same cycle — build3
  // timing exactly. Only when reads are still in flight (the silicon
  // race) does the capture defer to the end-of-reads edge.
  wire [31:0] accum_total;
  wire        accum_valid;
  assign capture_fire = (mode_r == DIFF_MODE)
                        && (capture_pending_r || flush_proc)
                        && (rd_path_quiet || capture_force);
  popcount_accum #(.ACCUM_WIDTH(32)) pca(
    .clk(clk),
    .rst(rst),
    .in_value(pop_count_value),
    .in_valid(pop_count_valid),
    // build4: capture at the program's end-of-reads (deferred past the
    // flush edge while announced reads are outstanding). build3 drained
    // directly at flush_proc, which sampled a partial/empty accumulator
    // whenever the frontend's fin+32 flush outran the DDR read tail.
    .drain(capture_fire),
    .accum_out(accum_total),
    .accum_valid(accum_valid)
  );
  assign dsr_out   = {480'b0, accum_total};
  assign dsr_valid = accum_valid;
`else
  diff_shift_reg dsr(
    .clk(clk),
    .rst(rst),

    .in(pop_count_value),
    .in_valid(pop_count_valid),

    .flush(flush_proc), // build3: edge-derived one-cycle pulse

    .out(dsr_out),
    .out_valid(dsr_valid)
  );
`endif
  // End popcount computation part
  
  // Count up to 1024 32-byte transfers  
  reg[9:0] xctr_r;

  reg tlast; // indicating c2h's last transfer
  
  // We read DQ_WIDTH*DQ_BURST (512 as of now) bits
  // from DRAM, and have to pipe 256 bit partitions of
  // it to the PCI. We may read data each cycle from 
  // DRAM and have to buffer some of those. 
  wire rbf_empty, rbf_rd_valid, fifo_almost_full, fifo_valid;
  (*KEEP = "TRUE"*) wire rbf_full;
  (*KEEP = "TRUE"*) reg [19:0] dbg_rd_ctr;
  // build3: FIFO output now lands on rbf_dout so the trailer beat can
  // mux the debug counters onto c2h_tdata_0 (data beats untouched).
  wire [`XDMA_AXI_DATA_WIDTH-1:0] rbf_dout;
  rdback_fifo rbf(
    .full(rbf_full),
    .prog_full(fifo_almost_full),
    .empty(rbf_empty),
    .wr_en(mode_r == READ_MODE    ? rd_valid && ~ignore_read_r :
           mode_r == SEG_POP_MODE ? seg_word_valid : dsr_valid),
    // shuffle data because fifo outputs them on wrong order. SEG_POP uses
    // the same 256b half-swap as READ so the host unpacks segments in the
    // natural order byte[g]=popcount(segment g), g=beat*16+lane.
    .din(mode_r == READ_MODE    ? {rd_data[255:0],rd_data[511:256]} :
         mode_r == SEG_POP_MODE ? {seg_word[255:0],seg_word[511:256]} :
                                  {dsr_out[255:0],dsr_out[511:256]}),
    .rd_en(c2h_tready_0),
    .dout(rbf_dout),
    .valid(fifo_valid),
    .clk(clk),
    .srst(rst)
  );
  
  reg proc_flush_ns, proc_flush_r;
  // we count the remaining space in terms of 
  // AXI transactions
  // e.g. 1024 reads will take up 2048 
  reg [11:0] buffer_space_ns, buffer_space_r;
  
  always @* begin
    tlast = `LOW;
    ignore_read_ns = ignore_read_r;
    ignore_flush_ctr_ns = ignore_flush_ctr_r;
    flush_eaten = `LOW;
    buffer_space_ns = buffer_space_r;
    if(per_rd_init || per_zq_init || per_ref_init) begin
      ignore_read_ns = per_rd_init;
    end
    if(rd_valid_r)
      ignore_read_ns = `LOW;
    proc_flush_ns = proc_flush_r;
    // build3: single decision per flush RISING EDGE. Processed flushes
    // set proc_flush (trailer framing); eaten flushes only decrement
    // the maintenance accounting below.
    if(flush_edge) begin
      if(flush_proc)
        proc_flush_ns = `HIGH;
      else
        flush_eaten = `HIGH;
    end
    // build3: saturating maintenance-event counter. +1 per event edge,
    // -1 per eaten flush edge; a same-cycle pair cancels out. (A
    // processed-because-armed flush does NOT decrement: the pending
    // maintenance flush it left behind must still be eaten.)
    if(maint_edge && !flush_eaten) begin
      if(ignore_flush_ctr_r != 4'hf)
        ignore_flush_ctr_ns = ignore_flush_ctr_r + 4'd1;
    end
    else if(!maint_edge && flush_eaten)
      ignore_flush_ctr_ns = ignore_flush_ctr_r - 4'd1;
`ifdef POPCOUNT_ACCUM_MODE
    // build4: announced-reads outstanding counter. +incoming_reads per
    // read_seq_incoming pulse (the fetched SMC_INFO packet, the same
    // signal buffer_space accounting consumes), -1 per rd_valid
    // return. Floor at zero absorbs un-announced maintenance (per_rd)
    // returns, which by frontend serialization + in-order PHY returns
    // can only arrive when nothing announced is outstanding. 16 bits
    // cannot overflow in practice (announcements are buffer_space-gated
    // in fetch_stage; legitimate outstanding depth is O(buffer_space)).
    rd_outstanding_ns = rd_outstanding_r;
    if(read_seq_incoming && rd_valid)
      rd_outstanding_ns = rd_outstanding_r + {4'd0, incoming_reads} - 16'd1;
    else if(read_seq_incoming)
      rd_outstanding_ns = rd_outstanding_r + {4'd0, incoming_reads};
    else if(rd_valid && rd_outstanding_r != 16'd0)
      rd_outstanding_ns = rd_outstanding_r - 16'd1;
    // build4: capture-pending flag — set by a processed DIFF flush,
    // cleared by the single-cycle capture. In the common paced case
    // flush_proc and capture_fire coincide (quiet at the flush edge,
    // capture_pending_r still 0): that is ONE flush -> ONE capture,
    // pending stays clear. Only if a NEW processed flush coincides
    // with an OLD pending capture firing (capture_pending_r == 1;
    // unreachable in practice since flushes are separated by whole
    // program executions) re-arm, preserving the 1 processed-flush :
    // 1 chunk framing the host counts on.
    capture_pending_ns = capture_pending_r;
    if(capture_fire)
      capture_pending_ns = (flush_proc && mode_r == DIFF_MODE
                            && capture_pending_r) ? `HIGH : `LOW;
    else if(flush_proc && mode_r == DIFF_MODE)
      capture_pending_ns = `HIGH;
    // build5: arm the chunk-beat counter at the drain (2 c2h beats per 512b
    // accum chunk), decrement as each chunk beat leaves the FIFO. The
    // trailer waits for this to reach 0 (chunk fully emitted), which is
    // robust to the FWFT fall-through latency that rbf_empty is not.
    chunk_beats_ns = chunk_beats_r;
    if(dsr_valid)
      chunk_beats_ns = 2'd2;
    else if(chunk_beats_r != 2'd0 && fifo_valid && c2h_tready_0)
      chunk_beats_ns = chunk_beats_r - 2'd1;
`endif
    mode_ns = mode_r;
    // build7: legacy toggle is a 2-state READ<->DIFF flip (never produces
    // the SEG_POP state, so old callers are unchanged and no invalid mode
    // 3 is reachable). SEG_POP is entered only via its SET word.
    if(switch_mode)
      mode_ns = (mode_r == READ_MODE) ? DIFF_MODE : READ_MODE;
    // build4/7: idempotent SET words (frontend control words). They win
    // over the legacy toggle if ever co-asserted; last-writer among the
    // three SETs wins (the host never co-asserts them).
    if(set_mode_read)
      mode_ns = READ_MODE;
    if(set_mode_diff)
      mode_ns = DIFF_MODE;
    if(set_mode_segpop)
      mode_ns = SEG_POP_MODE;
    if(&xctr_r && (c2h_tready_0 && c2h_tvalid_0)) begin
      tlast = `HIGH;
    end
    // Send what's remaining in the fifo
    // to host with a random length transfer
    // (tlast is not based on the counter value)
    // build4 (accum mode): the trailer additionally waits for a
    // deferred capture — its chunk must enter the FIFO before the
    // trailer closes the message. capture_pending_r is never set in
    // READ_MODE, so READ_MODE framing is bit-identical to build3.
    if(proc_flush_r) begin
`ifdef POPCOUNT_ACCUM_MODE
      // build5: also wait for the drained chunk's two c2h beats to leave
      // the FIFO (chunk_beats_r==0) before closing the message with tlast.
      if(c2h_tready_0 && rbf_empty && ~dsr_valid && ~capture_pending_r
         && chunk_beats_r == 2'd0) begin
`else
      if(c2h_tready_0 && rbf_empty && ~dsr_valid) begin
`endif
        tlast = `HIGH;
        proc_flush_ns = `LOW;
      end
      else
        proc_flush_ns = `HIGH;
    end
    // build6 (2026-07-21): buffer_space CONSERVATION in DIFF mode.
    // The stock accounting debits 2 units per announced read and credits
    // ONLY actual accepted c2h transfers (+1 each). In DIFF mode almost
    // no read reaches c2h (accum: one 2-transfer chunk per program;
    // streaming: one per 32 reads), so ~2*reads-2 units leak per program.
    // pre_decode's need_flush then loops forever on a budget that
    // flushing can never restore, fetch stalls, and h2c sends fail after
    // floor(2048/254) = 8 programs of 128 reads — the silicon
    // 8-then-wedge (2026-07-20/21, NUM_COLS-independent because rdRow
    // always announces the full row). Fix: credit +2 the cycle a beat is
    // consumed by the DIFF path (diff_valid — maintenance reads are
    // excluded by its ~ignore_read_r gate), and stop crediting DIFF-mode
    // c2h transfers (chunk and trailer beats are not part of the
    // announced-read budget). READ_MODE accounting is unchanged. The
    // three deltas compose additively so coincident events stay exact.
    if(read_seq_incoming)
      buffer_space_ns = buffer_space_ns - (incoming_reads << 1);
    if(mode_r == READ_MODE && c2h_tvalid_0 && c2h_tready_0
       && ~(proc_flush_r && rbf_empty && ~dsr_valid))
      buffer_space_ns = buffer_space_ns + 12'd1;
    if(mode_r == DIFF_MODE && diff_valid)
      buffer_space_ns = buffer_space_ns + 12'd2;
    // build7: SEG_POP conservation. Each user read beat debits 2 (above)
    // but only 1 c2h word leaves per 4 beats (4x collapse) — the same leak
    // class build6 fixed for DIFF. Credit +2 the cycle a beat is consumed
    // by the SEG_POP path (seg_beat_valid: maintenance reads excluded by
    // its ~ignore_read_r gate) and do NOT credit SEG_POP c2h. Net zero, so
    // pre_decode's need_flush budget is conserved over unbounded SEG_POP
    // sessions. READ/DIFF accounting untouched.
    if(mode_r == SEG_POP_MODE && seg_beat_valid)
      buffer_space_ns = buffer_space_ns + 12'd2;
  end
  
  always @(posedge clk) begin
    if(rst) begin
      dbg_rd_ctr <= 20'b0;
      xctr_r <= 15'b0;
      proc_flush_r <= `LOW;
      mode_r <= READ_MODE;
      ignore_read_r <= 1'b0;
      rd_valid_r <= 1'b0;
      buffer_space_r <= 12'd2048;
      // build3
      flush_r <= 1'b0;
      per_rd_init_r <= 1'b0;
      per_zq_init_r <= 1'b0;
      per_ref_init_r <= 1'b0;
      maint_any_r <= 1'b0;
      ignore_flush_ctr_r <= 4'd0;
`ifdef POPCOUNT_ACCUM_MODE
      accum_armed_r <= 1'b0;
      // build4
      rd_outstanding_r  <= 16'd0;
      capture_pending_r <= 1'b0;
      capture_age_r     <= 12'd0;
      chunk_beats_r     <= 2'd0;   // build5
`endif
      cnt_rd_init     <= 32'd0;
      cnt_zq_init     <= 32'd0;
      cnt_ref_init    <= 32'd0;
      cnt_flush_edge  <= 32'd0;
      cnt_flush_eaten <= 32'd0;
      cnt_drain       <= 32'd0;
      cnt_accum_write <= 32'd0;
    end
    else begin
      if(rd_valid && ~ignore_read_r && ~rbf_full)
        dbg_rd_ctr <= dbg_rd_ctr + 1'b1;
      else
        dbg_rd_ctr <= dbg_rd_ctr;
      buffer_space_r <= buffer_space_ns;
      mode_r <= mode_ns;
      rd_valid_r <= rd_valid;
      ignore_read_r <= ignore_read_ns;
      // build3: edge-detect registers + event counter + debug counters
      flush_r <= flush;
      per_rd_init_r <= per_rd_init;
      per_zq_init_r <= per_zq_init;
      per_ref_init_r <= per_ref_init;
      maint_any_r <= maint_any;
      ignore_flush_ctr_r <= ignore_flush_ctr_ns;
`ifdef POPCOUNT_ACCUM_MODE
      // Armed = the accumulator holds >=1 user DIFF sample. build4:
      // cleared at the CAPTURE (not the flush edge) — samples landing
      // between the flush edge and the deferred capture belong to the
      // current program and are folded into its drained total.
      if(capture_fire)
        accum_armed_r <= 1'b0;
      else if(pop_count_valid)
        accum_armed_r <= 1'b1;
      // build4: outstanding-reads / pending-capture state + the
      // pending-age safety valve (counts only while a capture is
      // pending; saturation forces the capture so c2h framing
      // survives even a lost read return).
      rd_outstanding_r  <= rd_outstanding_ns;
      capture_pending_r <= capture_pending_ns;
      chunk_beats_r     <= chunk_beats_ns;   // build5
      if(!capture_pending_ns)
        capture_age_r <= 12'd0;
      else if(~&capture_age_r)
        capture_age_r <= capture_age_r + 12'd1;
`endif
      if(per_rd_init && !per_rd_init_r)
        cnt_rd_init <= cnt_rd_init + 32'd1;
      if(per_zq_init && !per_zq_init_r)
        cnt_zq_init <= cnt_zq_init + 32'd1;
      if(per_ref_init && !per_ref_init_r)
        cnt_ref_init <= cnt_ref_init + 32'd1;
      if(flush_edge)
        cnt_flush_edge <= cnt_flush_edge + 32'd1;
      if(flush_eaten)
        cnt_flush_eaten <= cnt_flush_eaten + 32'd1;
`ifdef POPCOUNT_ACCUM_MODE
      // build4: cnt_drain counts actual accumulator captures (the
      // deferred-fire pulses), not flush edges.
      if(capture_fire)
        cnt_drain <= cnt_drain + 32'd1;
`else
      if(flush_proc)
        cnt_drain <= cnt_drain + 32'd1;
`endif
      if(dsr_valid && mode_r != READ_MODE && ~rbf_full)
        cnt_accum_write <= cnt_accum_write + 32'd1;
      if(proc_flush_r && tlast)
        xctr_r <= 15'b0;
      else if(c2h_tready_0 && c2h_tvalid_0) begin
        xctr_r <= xctr_r + 1;
      end
      proc_flush_r <= proc_flush_ns;
    end
  end
  
  // build3: the trailer beat (the pre-existing tvalid override) now
  // carries the debug counters instead of stale FIFO output. Data
  // beats (fifo_valid path) are bit-identical to before; the host
  // recognizes an instrumented trailer by its magic in [31:0]
  // (build5 = 0xDBC0DE04). build4: in accum mode the trailer also
  // waits out a deferred capture, so chunk-then-trailer order holds.
`ifdef POPCOUNT_ACCUM_MODE
  // build5: trailer only after the chunk's two c2h beats have left (FWFT-
  // fill-latency-safe), not merely on rbf_empty.
  wire trailer_beat = proc_flush_r && rbf_empty && ~dsr_valid
                      && ~capture_pending_r && chunk_beats_r == 2'd0;
`else
  wire trailer_beat = proc_flush_r && rbf_empty && ~dsr_valid;
`endif

  assign c2h_tkeep_0  = {(`XDMA_AXI_DATA_WIDTH/8){1'b1}};
  assign c2h_tlast_0  = tlast;
  assign c2h_tvalid_0 = trailer_beat ? `HIGH : fifo_valid;
  assign c2h_tdata_0  = trailer_beat ?
      {cnt_accum_write, cnt_drain, cnt_flush_eaten, cnt_flush_edge,
       cnt_ref_init, cnt_zq_init, cnt_rd_init, 32'hDBC0DE05} : rbf_dout;

  assign buffer_space = buffer_space_r >> 1;
  
endmodule
