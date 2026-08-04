#!/usr/bin/env bash
# Seam-crash REPRODUCTION on a chosen server binary, persist-OFF config
# (exactly the recorded engine_ab persistOFF run). No timeout wrapper.
# Usage: repro.sh <server_binary_path> <tag>
set -u
BN=/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet
L=/home/deni/Claude/seamfix_2026_08_04/logs
SRV="${1:-$BN/bitnet-proj-server}"
TAG="${2:-baseline}"
cd "$BN"
export PIM_SERVER_PATH="$SRV"
export PIM_SUB_START=45312 PIM_SUB_END=45952
export PIM_POOL_LIST_FILE="$BN/pool_layout_dimm2_cloneok_bank{bank}.txt"
export BITSTREAM_IMEM=8192 PIM_RECV_TIMEOUT_MS=15000
export PIM_USE_LOAD_WEIGHTS=1 PIM_DESC_SERVE=1
export PIM_DESC_XBATCH=1 PIM_DESC_XBATCH_CLIENT=1
export PIM_1BIT_SINGLE=0 PIM_FUSED_COSET=1 PIM_VOTE_FULL=0
export PIM_DESC_PERSIST=0                       # persist OFF (promotion-independent repro)
unset PIM_B49_ORDTOL PIM_DESC_SESSREUSE PIM_DESC_BANKGEN PIM_DESC_B_GUARD
CLIENT="$L/${TAG}_client.log"
SERVER="$L/${TAG}_server.log"
echo "=== seam repro ($TAG) srv_md5=$(md5sum "$SRV"|cut -c1-8) start $(date -Is) ===" | tee "$CLIENT"
python3 -u /home/deni/bitnet_weights/run_bitnet_pim.py \
  --model bonsai_1bit --bender 2 --bank 0,1,2,3 \
  --calib "$BN/calib_dimm2.txt" \
  --pool-layout "$BN/pool_layout_dimm2_cloneok_bank{bank}.txt" \
  --prompt "What is the capital of France?" --max-tokens 8 2>&1 | tee -a "$CLIENT"
rc=${PIPESTATUS[0]}
echo "=== seam repro ($TAG) exit=$rc $(date -Is) ===" | tee -a "$CLIENT"
cp -f /tmp/pim_server_b2_0_1_2_3.log "$SERVER" 2>/dev/null
echo "SEAM_REPRO_${TAG}_DONE rc=$rc"
