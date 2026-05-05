#include <string>

#ifndef BOARD_H
#define BOARD_H

class SimDramModel;  // forward decl — defined in pim_sim.h, allocated when iface_type==SIM

/** This class defines how the host
 * interfaces with the board.
 */
class BoardInterface{
  // 32 bytes per instruction (256-bit AXI packet) × IMEM depth (8192 after the 11→13 IMEM_ADDR_WIDTH bump).
  const uint SEND_BUF_SIZE = 32*8192;
  // Kernel xdma can occasionally return slightly more than the requested
  // size in a single read (observed up to ~33 KB on a 32-KB request when
  // FPGA-side residual data accumulates ahead of the read). Pad the
  // host-side buffer accordingly so we always have somewhere to put it.
  const uint RECV_BUF_SIZE = 1024*64;
public:
  enum class IFACE {
      XDMA = 0,
      // In-process behavioral DDR + SiMRA model. Selected by
      // PIM_BACKEND=sim env var (handled in SoftMCPlatform::init()).
      // Calib path comes from PIM_SIM_CALIB env (or first arg of the
      // platform's existing calib path).
      SIM  = 1
  };
  // dimm_select picks the XDMA channel pair on multi-bender bitstreams
  // (e.g. BCU1525_QUAD): N → /dev/xdma0_h2c_N + /dev/xdma0_c2h_N.
  // Default 0 = single-bender behaviour.
  BoardInterface(IFACE, int dimm_select = 0);
  ~BoardInterface();
  int init();
  int sendData(void* data, const uint size);
  int recvData(void* buf , const uint size);
private:
  IFACE iface_type;
  int dimm_select;
  SimDramModel* sim_model = nullptr;  // owned, lazily allocated when iface_type==SIM
  // XDMA related constructs
  int to_card;
  int from_card;
  void* send_buf;
  void* recv_buf;
  int xdma_send(void* data, const uint size);
  int xdma_recv(void* buf,  const uint size);
  // end XDMA related constructs
};

#endif
