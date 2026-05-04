SHARED_SRCS := ../util.cpp $(wildcard ../../../api/*.c) $(wildcard ../../../api/*.cpp)
SHARED_OBJS := ${SHARED_SRCS:.cpp=.o}
SHARED_OBJS := ${SHARED_OBJS:.c=.o}
INCLUDE_DIRS := ../../../api ../../../../boost-lib
CPPFLAGS += -g -std=c++17 -pthread -O3 -mavx2
CPPFLAGS += $(foreach d,$(INCLUDE_DIRS),-I$(d))
BOOST = -lboost_filesystem

CC = g++

.PHONY: all clean distclean

BINARIES := maj3-smoke-exe matvec-smoke-exe columnwise-smoke-exe dense-matvec-smoke-exe broadcast-verify-exe dense-matvec-bcast-exe multibank-bcast-exe bitnet-e2e-v1-exe bitnet-e2e-v2-exe bitnet-e2e-v3-exe bitnet-real-exe bitnet-proj-exe bitnet-proj-server rowclone-smoke-exe persistent-smoke-exe

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

persistent-smoke-exe: test_persistent_weights_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)


maj3-smoke-exe: test_maj3_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

matvec-smoke-exe: test_matvec_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

columnwise-smoke-exe: test_columnwise_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

dense-matvec-smoke-exe: test_dense_matvec_smoke.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

broadcast-verify-exe: test_broadcast_verify.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

dense-matvec-bcast-exe: test_dense_matvec_bcast.o $(SHARED_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

clean:
	@- $(RM) $(BINARIES)
	@- $(RM) test_maj3_smoke.o test_matvec_smoke.o test_columnwise_smoke.o test_dense_matvec_smoke.o test_broadcast_verify.o test_dense_matvec_bcast.o test_multibank_bcast.o test_bitnet_e2e_v1.o test_bitnet_e2e_v2.o test_bitnet_e2e_v3.o test_bitnet_real.o test_bitnet_proj.o test_bitnet_server.o

distclean: clean
