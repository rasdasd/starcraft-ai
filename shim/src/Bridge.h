#pragma once
// Bridge: owns the bot connection and runs the per-frame lockstep exchange.
// Backend-agnostic; driven either by the client-mode main loop (main_client.cpp) or by an
// AIModule (main_module.cpp).

#include <BWAPI.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CommandApplier.h"
#include "Serializer.h"
#include "Transport.h"
#include "flatbuffers/flatbuffers.h"

namespace shim {

struct BridgeOptions {
  std::string host = "127.0.0.1";
  uint16_t port = 8765;
  int statsEvery = 500;   // print timing stats every N bot decisions (0 = never)
  bool verbose = false;
  bool autoResume = true; // unpause single-player games that paused on focus loss

  // Fills from BWBOT_HOST / BWBOT_PORT / BWBOT_STATS_EVERY / BWBOT_VERBOSE.
  static BridgeOptions fromEnv();
  // Applies --host/--port/--stats-every/--verbose on top of *this.
  void parseArgs(int argc, char** argv);
};

class Bridge {
public:
  explicit Bridge(BridgeOptions opt);
  ~Bridge();

  // BWAPI lifecycle. Call from MatchStart / MatchFrame / MatchEnd.
  void onStart();
  void onFrame();
  void onEnd(bool isWinner);
  // Any other event (unit events, text, nuke, player left): buffered into the next Frame.
  void onEvent(const BWAPI::Event& e);

  bool botConnected() const { return transport_ && transport_->isConnected(); }

private:
  bool ensureConnected();   // accept + Hello/ClientConfig handshake + GameStart
  bool handshake();
  void applyConfig(const bw::ClientConfig& cfg, bool atStart);
  bool exchangeFrame();     // Frame -> Commands
  bool sendBuffer();
  void printStats();

  BridgeOptions opt_;
  std::unique_ptr<Transport> transport_;
  Serializer serializer_;
  CommandApplier applier_;
  flatbuffers::FlatBufferBuilder fbb_{1 << 20};
  std::vector<uint8_t> rx_;

  std::vector<BWAPI::Event> pendingEvents_;
  SerializeOptions serOpt_;
  int frameSkip_ = 1;
  bool inGame_ = false;
  bool gameStartSent_ = false;

  // timing
  int lastRoundtripUs_ = 0;
  struct Acc { double sum = 0; int max = 0; int n = 0; void add(int v) { sum += v; if (v > max) max = v; ++n; } void reset() { *this = Acc{}; } };
  Acc serAcc_, rttAcc_, applyAcc_, bytesAcc_;
  int decisions_ = 0;
  int unitsLast_ = 0;
  int pausedUpdates_ = 0;
  std::chrono::steady_clock::time_point statsT0_;
};

}  // namespace shim
