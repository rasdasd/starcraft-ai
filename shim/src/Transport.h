#pragma once
// Length-prefixed message transport. Currently TCP (localhost by default); the interface is
// deliberately tiny so a shared-memory or named-pipe implementation can be dropped in later.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace shim {

class Transport {
public:
  virtual ~Transport() = default;

  // Block until a peer is connected. Returns false on fatal error.
  virtual bool waitForPeer() = 0;
  virtual bool isConnected() const = 0;
  virtual void closePeer() = 0;

  // Send one message (uint32 LE length prefix + payload). Returns false on disconnect.
  virtual bool send(const uint8_t* data, size_t size) = 0;
  // Receive one message into `out` (payload only). Returns false on disconnect / error.
  virtual bool recv(std::vector<uint8_t>& out) = 0;

  virtual std::string describe() const = 0;
};

// TCP server: binds host:port, accepts a single bot connection at a time.
std::unique_ptr<Transport> makeTcpServer(const std::string& host, uint16_t port);

}  // namespace shim
