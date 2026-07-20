# Lane-2 MVDRAM GeMV server (phase 1) — mvdram-repro tree, NOT BitNet/.
# Links against the prebuilt DSN_AE_APPS/api objects, mirroring how
# BitNet/Makefile finds ../../../api (see LANE2_GEMV_SERVER.md).
BENDER   := /home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources
APPS     := $(BENDER)/apps/DSN_AE_APPS
API      := $(BENDER)/api

CC       := g++
CPPFLAGS += -g -std=c++17 -pthread -O3 -mavx2 -I$(API) -I$(APPS)
BOOST    :=   # boost usage is header-only (spsc_queue); nothing to link

API_OBJS := $(API)/board.o $(API)/instruction.o $(API)/pim_sim.o \
            $(API)/platform.o $(API)/prog.o
UTIL_OBJ := $(APPS)/util.o

.PHONY: all clean
all: lane2-gemv-server

lane2-gemv-server: lane2_gemv_server.o $(UTIL_OBJ) $(API_OBJS)
	$(CC) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(BOOST)

lane2_gemv_server.o: lane2_gemv_server.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

# the shared objects are maintained by the DSN_AE_APPS builds; rebuild there
# if missing (make -C $(APPS)/BitNet maj3-smoke-exe builds util.o + api/*.o)
clean:
	rm -f lane2_gemv_server.o lane2-gemv-server
