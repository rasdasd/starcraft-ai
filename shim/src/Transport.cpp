#include "Transport.h"

#include <cstring>
#include <iostream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
static const sock_t kInvalidSock = INVALID_SOCKET;
static void closeSock(sock_t s) { closesocket(s); }
static int lastSockError() { return WSAGetLastError(); }
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sock_t;
static const sock_t kInvalidSock = -1;
static void closeSock(sock_t s) { ::close(s); }
static int lastSockError() { return errno; }
#endif

namespace shim {
namespace {

struct WinsockInit {
  WinsockInit() {
#ifdef _WIN32
    WSADATA d;
    WSAStartup(MAKEWORD(2, 2), &d);
#endif
  }
  ~WinsockInit() {
#ifdef _WIN32
    WSACleanup();
#endif
  }
};

class TcpServer final : public Transport {
public:
  TcpServer(std::string host, uint16_t port) : host_(std::move(host)), port_(port) {}

  ~TcpServer() override {
    closePeer();
    if (listen_ != kInvalidSock) closeSock(listen_);
  }

  bool waitForPeer() override {
    if (peer_ != kInvalidSock) return true;
    if (listen_ == kInvalidSock && !bind()) return false;
    std::cout << "[shim] waiting for bot on " << host_ << ":" << port_ << " ..." << std::endl;
    sockaddr_in addr{};
#ifdef _WIN32
    int len = sizeof(addr);
#else
    socklen_t len = sizeof(addr);
#endif
    sock_t s = ::accept(listen_, reinterpret_cast<sockaddr*>(&addr), &len);
    if (s == kInvalidSock) {
      std::cerr << "[shim] accept failed: " << lastSockError() << std::endl;
      return false;
    }
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
    peer_ = s;
    char ip[64] = {0};
    inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
    std::cout << "[shim] bot connected from " << ip << ":" << ntohs(addr.sin_port) << std::endl;
    return true;
  }

  bool isConnected() const override { return peer_ != kInvalidSock; }

  void closePeer() override {
    if (peer_ != kInvalidSock) {
      closeSock(peer_);
      peer_ = kInvalidSock;
    }
  }

  bool send(const uint8_t* data, size_t size) override {
    if (peer_ == kInvalidSock) return false;
    uint8_t hdr[4] = {uint8_t(size), uint8_t(size >> 8), uint8_t(size >> 16), uint8_t(size >> 24)};
    return sendAll(hdr, 4) && sendAll(data, size);
  }

  bool recv(std::vector<uint8_t>& out) override {
    if (peer_ == kInvalidSock) return false;
    uint8_t hdr[4];
    if (!recvAll(hdr, 4)) return false;
    uint32_t n = uint32_t(hdr[0]) | (uint32_t(hdr[1]) << 8) | (uint32_t(hdr[2]) << 16) | (uint32_t(hdr[3]) << 24);
    if (n > (256u << 20)) {
      std::cerr << "[shim] message too large: " << n << std::endl;
      closePeer();
      return false;
    }
    out.resize(n);
    return n == 0 || recvAll(out.data(), n);
  }

  std::string describe() const override { return "tcp://" + host_ + ":" + std::to_string(port_); }

private:
  bool bind() {
    static WinsockInit ws;
    listen_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_ == kInvalidSock) {
      std::cerr << "[shim] socket() failed: " << lastSockError() << std::endl;
      return false;
    }
    int one = 1;
#ifdef _WIN32
    setsockopt(listen_, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&one), sizeof(one));
#else
    setsockopt(listen_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    if (inet_pton(AF_INET, host_.c_str(), &addr.sin_addr) != 1) {
      std::cerr << "[shim] bad bind address: " << host_ << std::endl;
      return false;
    }
    if (::bind(listen_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      std::cerr << "[shim] bind(" << host_ << ":" << port_ << ") failed: " << lastSockError() << std::endl;
      closeSock(listen_);
      listen_ = kInvalidSock;
      return false;
    }
    if (::listen(listen_, 1) != 0) {
      std::cerr << "[shim] listen failed: " << lastSockError() << std::endl;
      closeSock(listen_);
      listen_ = kInvalidSock;
      return false;
    }
    return true;
  }

  bool sendAll(const uint8_t* p, size_t n) {
    while (n > 0) {
      int chunk = int(n > (1u << 30) ? (1u << 30) : n);
#ifdef _WIN32
      int w = ::send(peer_, reinterpret_cast<const char*>(p), chunk, 0);
#else
      ssize_t w = ::send(peer_, p, chunk, MSG_NOSIGNAL);
#endif
      if (w <= 0) {
        std::cerr << "[shim] send failed: " << lastSockError() << std::endl;
        closePeer();
        return false;
      }
      p += w;
      n -= size_t(w);
    }
    return true;
  }

  bool recvAll(uint8_t* p, size_t n) {
    while (n > 0) {
      int chunk = int(n > (1u << 30) ? (1u << 30) : n);
#ifdef _WIN32
      int r = ::recv(peer_, reinterpret_cast<char*>(p), chunk, 0);
#else
      ssize_t r = ::recv(peer_, p, chunk, 0);
#endif
      if (r <= 0) {
        if (r < 0) std::cerr << "[shim] recv failed: " << lastSockError() << std::endl;
        else std::cout << "[shim] bot disconnected" << std::endl;
        closePeer();
        return false;
      }
      p += r;
      n -= size_t(r);
    }
    return true;
  }

  std::string host_;
  uint16_t port_;
  sock_t listen_ = kInvalidSock;
  sock_t peer_ = kInvalidSock;
};

}  // namespace

std::unique_ptr<Transport> makeTcpServer(const std::string& host, uint16_t port) {
  return std::make_unique<TcpServer>(host, port);
}

}  // namespace shim
