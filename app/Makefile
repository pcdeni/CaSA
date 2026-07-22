SHARED_SRCS := ../util.cpp parallel_emit.cpp regalloc.cpp \
               $(wildcard ../../../api/*.c) $(wildcard ../../../api/*.cpp)
SHARED_OBJS := ${SHARED_SRCS:.cpp=.o}
SHARED_OBJS := ${SHARED_OBJS:.c=.o}
INCLUDE_DIRS := ../../../api ../../../../boost-lib
CPPFLAGS += -g -std=c++17 -pthread -O3 -mavx2
CPPFLAGS += $(foreach d,$(INCLUDE_DIRS),-I$(d))
# boost usage is header-only on this tower; -lboost_filesystem does not
# resolve AND a failed link deletes the existing binary (2026-07-19 trap,
# see RUNBOOK). Default empty; override with make BOOST=-lboost_filesystem
# on hosts that need it.
BOOST =

CC = g++

.PHONY: all clean distclean

BINARIES := maj3-smoke-exe matvec-smoke-exe columnwise-smoke-exe dense-matvec-smoke-exe broadcast-verify-exe dense-matvec-bcast-exe multibank-bcast-exe bitnet-e2e-v1-exe bitnet-e2e-v2-exe bitnet-e2e-v3-exe bitnet-real-exe bitnet-proj-exe bitnet-proj-server rowclone-smoke-exe persistent-smoke-exe rowclone-preserve-exe multibank-disturb-exe write-disturb-exe spread-test-exe

all: $(BINARIES)

multibank-bcast-exe: test_multibank_bcast.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

bitnet-e2e-v1-exe: test_bitnet_e2e_v1.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

bitnet-e2e-v2-exe: test_bitnet_e2e_v2.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

bitnet-e2e-v3-exe: test_bitnet_e2e_v3.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

bitnet-real-exe: test_bitnet_real.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

bitnet-proj-exe: test_bitnet_proj.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

bitnet-proj-server: test_bitnet_server.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

rowclone-smoke-exe: test_rowclone_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

rowclone-preserve-exe: test_rowclone_preserve.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

multibank-disturb-exe: test_multibank_disturb.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

write-disturb-exe: test_write_disturb.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

persistent-smoke-exe: test_persistent_weights_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)


maj3-smoke-exe: test_maj3_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

matvec-smoke-exe: test_matvec_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

popcount-hw-exe: test_popcount_hw.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

segpop-hw-exe: test_segpop_hw.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

accxbp-hw-exe: test_accxbp_hw.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

columnwise-smoke-exe: test_columnwise_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

dense-matvec-smoke-exe: test_dense_matvec_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

broadcast-verify-exe: test_broadcast_verify.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

dense-matvec-bcast-exe: test_dense_matvec_bcast.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

spread-test-exe: test_spread.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

fault-sweep-exe: test_fault_sweep.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-maj5-exe: test_mvdram_maj5.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-adder-exe: test_mvdram_adder.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-gemv-exe: test_mvdram_gemv.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-popcount-exe: test_mvdram_popcount.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

fanout-open-rows-exe: test_fanout_to_open_rows.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

clean:
	@- $(RM) $(BINARIES)
	@- $(RM) test_maj3_smoke.o test_matvec_smoke.o test_columnwise_smoke.o test_dense_matvec_smoke.o test_broadcast_verify.o test_dense_matvec_bcast.o test_multibank_bcast.o test_bitnet_e2e_v1.o test_bitnet_e2e_v2.o test_bitnet_e2e_v3.o test_bitnet_real.o test_bitnet_proj.o test_bitnet_server.o

distclean: clean

mvdram-indram-exe: test_mvdram_indram.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-gemvn-exe: test_mvdram_gemv_n.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-rowsel-exe: test_mvdram_rowsel.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-loaddiag-exe: test_mvdram_loaddiag.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-triple-exe: test_mvdram_triple.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-twolevel-exe: test_mvdram_twolevel.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-chain-exe: test_mvdram_chain.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-fanout-exe: test_mvdram_fanout.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-fulladder-exe: test_mvdram_fulladder.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-plan-exe: test_mvdram_plan.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-gemv-inplace-exe: test_mvdram_gemv_inplace.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-compute-rows-exe: test_mvdram_compute_rows.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-rcrand-exe: test_rowclone_random.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

recv-timeout-test-exe: test_recv_timeout.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-fastpath-ab-exe: test_mvdram_fastpath_ab.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

selection-timing-exe: test_selection_timing.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

frac-sweep-exe: test_frac_sweep.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

frac-maj5-exe: test_frac_maj5.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

mvdram-adder2-exe: test_mvdram_adder2.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

majx-screen-exe: test_majx_screen.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

fused-maj-exe: test_fused_maj.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

fused-colmask-exe: test_fused_colmask.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

fcdram-not-exe: test_fcdram_not.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

adder-chain-exe: test_adder_chain.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

popcount-indram-exe: test_popcount_indram.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

packed-maj-exe: test_packed_maj.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

gemv-encoded-exe: test_gemv_encoded.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

resident-tile-exe: test_resident_tile.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

m3-scratch-ab-exe: test_m3_scratch_ab.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

m3-gate2-exe: test_m3_gate2.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

m3-alloc-test: m3_alloc_test.cpp m3_alloc.h
	$(CC) $(CPPFLAGS) $< -o $@

stream-hw-exe: test_stream_hw.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

# Standalone host-side unit test of the streaming record parser (no
# FPGA, no SoftMC objects): validates payload extraction across payload
# sizes x chunk boundaries. Run before trusting any streaming change.
record-parser-test: record_parser_test.cpp
	$(CC) $(CPPFLAGS) $< -o $@
