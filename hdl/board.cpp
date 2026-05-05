#include "board.h"
#include "pim_sim.h"
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string.h>

BoardInterface::BoardInterface(IFACE iface_type, int dimm_select)
{
  this -> iface_type = iface_type;
  this -> dimm_select = dimm_select;
}
BoardInterface::~BoardInterface()
{
  if (iface_type == IFACE::SIM) {
    delete sim_model;
    return;
  }
  free(send_buf);
  free(recv_buf);
  close(to_card);
  close(from_card);
}

int BoardInterface::init()
{
  switch(iface_type)
  {
    case IFACE::SIM:
    {
      sim_model = new SimDramModel();
      const char* calib = getenv("PIM_SIM_CALIB");
      if (calib && *calib) sim_model->load_calib(calib);
      const char* v = getenv("PIM_SIM_VERBOSE");
      if (v) sim_model->verbose = atoi(v);
      std::cerr << "[pim_sim] BoardInterface initialised in SIM mode "
                << "(calib=" << (calib ? calib : "<none>") << ")" << std::endl;
      return 0;
    }
    case IFACE::XDMA:
    {
      char to_path[64], from_path[64];
      std::snprintf(to_path,   sizeof(to_path), "/dev/xdma0_h2c_%d", dimm_select);
      std::snprintf(from_path, sizeof(from_path), "/dev/xdma0_c2h_%d", dimm_select);

      int fpga_fd = open(to_path, O_RDWR);
      if(fpga_fd<0)
      {
        std::cerr << "Open " << to_path << " (to card) failed!" << std::endl;
        return 1;
      }
      else
        std::cout << "Opened " << to_path <<  " -> " << fpga_fd << std::endl;
      to_card = fpga_fd;
      fpga_fd = open(from_path, O_RDWR);
      if(fpga_fd<0)
      {
        std::cerr << "Open " << from_path << " (to host) failed!" << std::endl;
        return 1;
      }
      else
        std::cout << "Opened " << from_path <<  " -> " << fpga_fd << std::endl;
      from_card = fpga_fd;
      // allocate page size aligned X page size regions to our buffers
      if (posix_memalign((void **)&send_buf, 4096 /*alignment */ , SEND_BUF_SIZE + (4096-(SEND_BUF_SIZE % 4096))) != 0)
      {
        std::cerr << "Send buffer allocation failed!" << std::endl;
        return 1;
      }
      if (posix_memalign((void **)&recv_buf, 4096 /*alignment */ , RECV_BUF_SIZE + (4096-(RECV_BUF_SIZE % 4096))))
      {
        std::cerr << "Receive buffer allocation failed!" << std::endl;
        return 1;
      }
      if( (!send_buf) || (!recv_buf) )
      {
        std::cerr << "Buffers cannot be allocated!" << std::endl;
        return 1;
      }
      return 0;
    }
    default:
      std::cerr << "Unknown iface_type!" << std::endl;
      return 1;
  }
}

int BoardInterface::sendData(void* data, const uint size)
{
  switch(iface_type)
  {
    case IFACE::SIM:
      return sim_model->send_program((const uint8_t*)data, size);
    case IFACE::XDMA:
      return xdma_send(data,size);
      break;
    default:
      std::cerr << "Unknown iface_type!" << std::endl;
      return 1;
  }
}

int BoardInterface::recvData(void* buf, const uint size)
{
  switch(iface_type)
  {
    case IFACE::SIM:
      return sim_model->recv_response((uint8_t*)buf, size);
    case IFACE::XDMA:
      return xdma_recv(buf,size);
      break;
    default:
      std::cerr << "Unknown iface_type!" << std::endl;
      return 1;
  }
}

int BoardInterface::xdma_send(void* data, const uint size)
{
  memcpy((char*)send_buf, (char*)data, size);

  // Loop on the remainder so partial writes (which kernel xdma can
  // legitimately do for large h2c transfers) are handled correctly.
  // Original code passed `size` to every write() and trusted an assert
  // that rc == size || rc == 0; that asserts out for any partial write,
  // and even in the lucky equal-size case it would silently re-send the
  // same bytes. Surface real errors as non-zero return so the caller
  // can reset_fpga and skip.
  char *buf = (char*) send_buf;
  uint count = 0;
  while (count < size) {
    ssize_t rc = write(to_card, buf + count, size - count);
    if (rc < 0) {
      std::cerr << "xdma_send write failed: " << strerror(errno)
                << " (rc=" << rc << ", count=" << count << "/" << size << ")"
                << std::endl;
      return 1;
    }
    if (rc == 0) {
      std::cerr << "xdma_send wrote 0 bytes — engine likely in shutdown"
                << std::endl;
      return 1;
    }
    count += (uint)rc;
  }
  return 0;
}

int BoardInterface::xdma_recv(void* buf, const uint size)
{
  assert(size <= RECV_BUF_SIZE && "given read size is too large");

  ssize_t rc = read(from_card, (char*) recv_buf, size);
  if (rc < 0) {
    std::cerr << "xdma_recv read() failed: " << strerror(errno)
              << " (rc=" << rc << ")" << std::endl;
    return -1;
  }
  if (rc > 0) memcpy(buf, recv_buf, rc);
  return (int)rc;
}
