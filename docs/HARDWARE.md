# Hardware

## Reference setup

| Component | What we run on |
|---|---|
| FPGA card | Xilinx BCU1525 (Virtex UltraScale+ VU9P), one PCIe Gen3 x16 endpoint |
| FPGA bitstream | DRAM-Bender's `BCU1525_QUAD` build — four independent DIMM channels |
| Memory | 4 × SK hynix `HMA41GU6AFR8N-TF` (dual-rank UDIMM, H5AN4G8NAFR 4 Gb ×8 chips), DDR4 at 1333 MT/s, one module per channel |
| Host | Linux workstation, PCIe Gen3 x16 to the card |
| Driver | Xilinx XDMA kernel module (`/dev/xdma{0..N}_h2c_M`, `/dev/xdma{0..N}_c2h_M`) |
| DRAM-Bender API | `SoftMCPlatform`, `Program`, `doubleACT`, `wrRow`, `rdRow` (from `DRAM-Bender/sources/api/`) |

The four modules are the same part number from different manufacturing
weeks. Which module sits in which socket, and what role each channel is
given, are recorded in `calibration/DIMM_POPULATION.conf` — the one file to
edit when the modules change. Everything else reads it at run time
(`python/dimm_population.py`); nothing hardcodes a fixture name, a subarray
window or a channel role.

## Address geometry of these parts

| Quantity | Value | Where it comes from |
|---|---|---|
| Row | 8192 B | the atomic read / write / RowClone unit |
| Banks per channel | 16 | all 16 usable; the shipped pools cover banks 0–3 |
| Addressable rows per bank | 32,768 (2^15) | the parts have 15 row address bits, A0..A14 |
| Addressable capacity | **4 GiB per DIMM** | 16 × 32,768 × 8 KiB |
| Sense-amp segment | 640 rows | RowClone / tuple locality domain |
| Predecoder block | 1024 rows | co-activation scope |

Two limits are worth stating plainly, because they cost capacity:

- **Row bit 15 dies inside the part.** Our controller stack carries 17 row
  bits and 17 address pins, but the chip decodes 15. Rows `r` and `r + 2^15`
  are the *same silicon* — confirmed by alias on two dies, with no divergent
  row over hundreds of thousands of compared bits. Any table of subarray
  windows must be audited modulo 2^15 or two "different" windows will
  silently be one, and the second write clobbers the first.
- **The second rank is not reachable.** The command encoding decodes a rank
  bit, but it is dropped before the pins, so a rank-1 write lands on the
  rank-0 row. The modules are dual-rank and the board family routes
  chip-select for both ranks; the gap is in the controller, not the hardware.

## Why this hardware

- **DRAM-Bender** is the only open-source FPGA platform that lets you issue
  DDR commands at sub-spec timings, which is what makes charge-sharing
  operations possible at all. Off-the-shelf memory controllers refuse to
  issue them.
- **DDR4 1333 MT/s** is well inside DRAM-Bender's tested envelope and gives
  reliable charge-sharing behaviour on several DIMM vendors.
- **The QUAD bitstream** exposes four independent channels behind one PCIe
  endpoint, which is what makes multi-DIMM residency and a storage tier
  possible without a second card.

## What you do not need

- Custom DRAM cells. Everything here runs on stock DDR4 modules.
- A modified host CPU or memory controller, beyond what DRAM-Bender already
  provides on the FPGA.
- A GPU at inference time. Training is a separate, GPU-bound problem; once
  the model is trained, running it needs no GPU.

## Setup

1. **Build and flash the DRAM-Bender bitstream.** Follow
   [DRAM-Bender's upstream README](https://github.com/CMU-SAFARI/DRAM-Bender)
   for your card variant.
2. **Verify XDMA exposure.**
   ```
   ls /dev/xdma*
   # h2c_0..h2c_N and c2h_0..c2h_N
   lspci -d 10ee: -vvv | grep DLActive
   # DLActive+ means the PCIe link is up
   ```
   `DLActive-` means the link is wedged. A full card power cycle is
   generally required; a host reboot alone has not been sufficient here.
   Find the card by vendor (`lspci -nn -d 10ee:`), never by a remembered
   bus address — when the card fails to enumerate the bus renumbers and
   something else takes the slot you were expecting.
3. **Insert DIMMs.** Any DDR4 modules of the right voltage. Yield of
   MAJ3-perfect tuples varies enormously by part: we have seen a candidate
   rate around 38 % on some SK hynix, and exactly zero on other parts. Plan
   for characterization time, not install-and-go.
4. **Build the C++ apps.** See `app/README.md`.
5. **Describe the population.** Edit `calibration/DIMM_POPULATION.conf`, then
   run `python3 python/dimm_population.py` as a dry check — it prints the
   trio and lane role it resolved for each channel and errors loudly on a
   missing fixture.
6. **Calibrate.** See `docs/CALIBRATION.md`, then `docs/CALIBRATION_TRANSFER.md`
   for what does *not* have to be re-measured.
7. **Run.** See the top-level `README.md` and `python/README.md`.

## Bringing a channel into production

Screening a channel is not one test, and the order matters. What we measure
before trusting a channel:

1. **RowClone integrity** — `rowclone-smoke-exe` across all `t_23` values,
   expecting `PERFECT_CLONE` on each.
2. **Byte-lane map** — a per-byte-lane comparison over a full row, expecting
   every lane counter at zero.
3. **Read/write screen** over the working window, expecting zero mismatching
   rows.
4. **Numerics** — the matmul oracle against a CPU reference.
5. **The same lane and clone checks again, after sustained model traffic.**

Step 5 is not redundant. A channel can pass every one of steps 1–4 and then
latch a byte lane under sustained load; the numerics oracle stays blind to
it (its correlation is ~1.0 on a channel with a latched lane). Certification
that stops before sustained traffic certifies nothing about sustained
traffic.

Across our four channels, steps 1–4 come out identical: the same 3,969-edge
co-activation fault set (same checksum on every channel), zero mismatches on
the read/write screen, `PERFECT_CLONE` on every `t_23`, and 515–516 of 624
pool rows clone-ok. That uniformity is what makes one calibration serve four
channels — see `docs/CALIBRATION_TRANSFER.md`.

## Temperature

Charge-sharing yield is temperature-sensitive. Our characterization runs sit
at a stable ~50 °C die temperature, and the numbers in the calibration files
are valid within roughly ±5 °C of that. Cooler is generally better for
retention; hotter pushes marginal cells toward the flip threshold and calls
for re-characterization at the operating temperature.
