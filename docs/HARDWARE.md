# Hardware

## Reference setup (what we built and tested on)

| Component | What we used |
|---|---|
| FPGA | Xilinx Alveo U200 / BCU1525 (Virtex UltraScale+ VU9P) |
| FPGA bitstream | DRAM-Bender's `BCU1525_QUAD` build (4 DIMM slots) |
| Memory | DDR4 1333 MT/s DIMMs in the FPGA's slots |
| Host | Linux workstation with PCIe Gen3 x16 to the FPGA |
| Driver | Xilinx XDMA kernel module (`/dev/xdma{0..N}_h2c_M`, `/dev/xdma{0..N}_c2h_M`) |
| DRAM-Bender API | `SoftMCPlatform`, `Program`, `doubleACT`, `wrRow`, `rdRow` (from `DRAM-Bender/sources/api/`) |

## Why this hardware

- **DRAM-Bender** is the only open-source FPGA platform that lets
  you issue DDR commands at sub-spec timings, which is what makes
  charge-sharing operations possible. Off-the-shelf memory
  controllers refuse to issue these and would have to be modified.
- **DDR4 1333 MT/s** is well within DRAM-Bender's tested envelope
  and gives us reliable charge-sharing behavior on multiple DIMM
  vendors.
- **The BCU1525_QUAD bitstream** exposes 4 independent DIMM
  channels, giving us the option to scale across DIMMs once the
  per-DIMM characterization work is done.

## What you do not need (despite often being suggested)

- Custom DRAM cells. Everything in this work runs on stock DDR4
  modules.
- A modified host CPU or memory controller, beyond what DRAM-Bender
  already provides on the FPGA.
- A GPU at inference time (training is a separate, GPU-bound
  problem; once the model is trained, we don't need a GPU to run
  it).

## Setup checklist

1. **Build and flash the DRAM-Bender bitstream.** Follow
   [DRAM-Bender's upstream README](https://github.com/CMU-SAFARI/DRAM-Bender)
   — the BCU1525 (or your variant) project files, Vivado build,
   bitstream loading. We did not modify the RTL; the upstream build
   is sufficient.
2. **Verify XDMA exposure.** After flashing and rebooting:
   ```
   ls /dev/xdma*
   # should list h2c_0..h2c_N and c2h_0..c2h_N
   lspci -d 10ee: -vvv | grep DLActive
   # should show DLActive+ (PCIe link up)
   ```
   If `DLActive-` appears, the PCIe link is wedged — a full FPGA
   power-cycle is generally required to recover. Host reboot alone
   has not been sufficient on our hardware.
3. **Insert DIMM(s).** Any DDR4 modules of the right voltage. Some
   chips will have very high MAJ3-perfect tuple yield (we have seen
   ~38 % candidate rate on Hynix); others will give zero. Plan for
   characterization time, not just install-and-go.
4. **Build the C++ apps.** See `app/README.md`.
5. **Calibrate.** See `docs/CALIBRATION.md`. The shipped
   `calibration/calib_dimm0.txt` is for our reference DIMM only —
   it will not work bit-exact on a different chip even of the same
   part number.
6. **Run.** See the top-level `README.md` and `python/README.md`.

## Temperature

Charge-sharing yield is temperature-sensitive. Our characterization
runs were at a stable ~50 °C die temperature. Numbers in the
calibration files and projections are valid only within ~5 °C of
that. Cooler is generally better for retention; hotter pushes
marginal cells closer to the flip threshold and would require
re-characterization at the operating temperature.
