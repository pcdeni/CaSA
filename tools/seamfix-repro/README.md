# Seam-crash reproduction

Reproduction scripts for the request/response seam failure that the canonical
server's `stream_on()` guard fixes: under `PIM_DESC_SERVE`, forcing `PIM_STREAM`
off (unless `PIM_STREAM_FORCE=1`) removes a c2h-framing seam between the
descriptor-serve path and the streaming path. The fix is in
`app/test_bitnet_server.cpp`; these scripts drive the published client
(`python/run_bitnet_pim.py`) against a chosen server binary to reproduce the
failure on a pre-fix binary and confirm a fixed binary runs clean.

- **`repro.sh <server_binary> <tag>`** — the full recorded config (Bonsai-1bit,
  DIMM 2, banks 0-3, 8 tokens, persist OFF).
- **`repro_short.sh <server_binary> <tag>`** — a 2-token short variant.

Both take the server binary path as `$1` (default `bitnet-proj-server`) and drive
`run_bitnet_pim.py` with the production DIMM-2 calib + clone-ok pools. No
`timeout` wrapper (a SIGTERM mid-transfer wedges the DMA engine); SIGTERM-and-wait
between runs.

> Paths in these scripts are absolute to the development rig
> (`/home/deni/Claude/...`, `/home/deni/bitnet_weights/...`); adjust `BN` and the
> `run_bitnet_pim.py` path for another host.
