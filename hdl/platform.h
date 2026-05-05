#include "board.h"
#include "prog.h"
#include <thread>
#include <boost/lockfree/spsc_queue.hpp>

#ifdef PYSMC
#include "ext/pybind11/include/pybind11/pybind11.h"
namespace py = pybind11;
#endif

//ERROR CODES
#define SOFTMC_SUCCESS 0
#define SOFTMC_ERR -1
#define SOFTMC_NO_PLATFORM -2
#define SOFTMC_ERR_OPEN_FPGA -3
#define SOFTMC_NO_SUCH_FPGA -4

class SoftMCPlatform{
  // Host-side AXI staging buffer — sized to the post-bump (IMEM=8192)
  // bitstream. Buffer can be wider than the FPGA's actual IMEM; the runtime
  // check below uses BITSTREAM_IMEM_INSTS, the deployed bitstream's true cap.
  #define INSTR_BUF_SIZE 32*8192
  // Real IMEM depth in the *currently deployed* bitstream. Bump to 8192
  // ONLY after re-synthesizing instr_blk_mem with ADDR=13/depth=8192;
  // otherwise the FPGA silently truncates and SoftMC deadlocks.
  #define BITSTREAM_IMEM_INSTS 2048
  #define API_BUF_SIZE 1024*1024*2
  public:
    SoftMCPlatform();
    SoftMCPlatform(bool);
    // Open against a specific DRAM-Bender instance (XDMA channel pair) on
    // multi-bender bitstreams like BCU1525_QUAD. dimm_select N → channel N.
    SoftMCPlatform(int dimm_select);
    ~SoftMCPlatform();
    /**
     * Initializes the whole platform
     * @return SOFTMC_SUCCESS on sucessful initialization
     */
    int init();
    /**
     * Resets SoftMC logic, won't reset PCI-E endpoint or the PHY interface
     */
    void reset_fpga();
    /**
     * Sends SoftMC program to the FPGA board over PCI-E
     * @param prog reference to the program object to send
     */
    void execute(Program & prog);
    /**
     * Pop exactly `size` bytes of program payload from the c2h drain
     * queue (which the receiver thread spawned by execute() fills in the
     * background). Blocks until that many bytes are available. Returns
     * the byte count on success; size must be a positive multiple of 4.
     *
     * The 32-byte per-program trailing AXI-Stream beat is stripped by
     * the receiver thread, so apps just see the row payload(s) the SoftMC
     * program emitted. For multi-row programs, call receiveData(buf, 8192)
     * once per row in the loop the program emits them.
     */
    int receiveData(void* dst_buf, int size);

    #ifdef PYSMC
    int py_receiveData(int num_words);
    #endif

    /**
     * Compare data with a given data pattern (repeating bytes)
     * and return number of bitflips in 8KB of data
     * @param comp_pattern one byte data pattern to compare the read data
     */ 
    int count_bitflips_in_row(unsigned char comp_pattern);

    /**
     * Turn auto-refresh on-off
     * @param on true to turn aref on false otherwise
     */
    void set_aref(bool on);

    /**
     * Used along with Program::dumpRegisters to read register content
     */
    void readRegisterDump();

  private:
    bool is_dummy;
    int dimm_select;

    BoardInterface *iface;
    void* instr_buf;

    // c2h drain thread + bounded queue. The thread is spawned by execute()
    // BEFORE sendData and runs until the FPGA emits its end-of-program TLAST
    // (kernel returns a partial read). It reads 32-KB chunks from c2h and
    // pushes the row payload (stripping the 32-B per-program trailer) into
    // api_recv_buf. Required because the FPGA back-pressures h2c when its
    // c2h FIFO fills, so without concurrent c2h drain, sendData of any
    // program whose output exceeds the c2h FIFO blocks indefinitely (and
    // the kernel's 10 s xdma timeout fires). 2 MB queue is overkill but
    // keeps the producer from ever stalling on push.
    std::thread receiver;
    boost::lockfree::spsc_queue<int> api_recv_buf{API_BUF_SIZE/4};
    void* xdma_recv_buf;
    void consumeData();

  #ifdef PYSMC
  public:
    uint8_t* py_data_buffer = nullptr;
    py::memoryview get_buffer_memoryview();
  #endif
};
