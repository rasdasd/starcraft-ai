#pragma once
// BWAPI game state -> FlatBuffers messages (bw.fbs).

#include <BWAPI.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "bw_generated.h"
#include "flatbuffers/flatbuffers.h"

namespace shim {

struct SerializeOptions {
  bool includeBullets = true;
  bool includeTiles = true;
  bool includePlayers = true;
};

class Serializer {
public:
  // Static per-match data. `frameSkip` is echoed so the bot knows the effective cadence.
  void buildGameStart(flatbuffers::FlatBufferBuilder& fbb, int frameSkip);

  // Per-frame snapshot. `events` are the BWAPI events accumulated since the last Frame.
  void buildFrame(flatbuffers::FlatBufferBuilder& fbb, const std::vector<BWAPI::Event>& events,
                  const SerializeOptions& opt, int serializeUs, int lastRoundtripUs);

  void buildGameEnd(flatbuffers::FlatBufferBuilder& fbb, bool isWinner);

  void buildHello(flatbuffers::FlatBufferBuilder& fbb);

  void reset() { lastHp_.clear(); }

private:
  bw::UnitState unitState(BWAPI::Unit u, const BWAPI::Playerset& players);
  flatbuffers::Offset<bw::PlayerState> playerState(flatbuffers::FlatBufferBuilder& fbb, BWAPI::Player p, bool full);
  flatbuffers::Offset<bw::PlayerInfo> playerInfo(flatbuffers::FlatBufferBuilder& fbb, BWAPI::Player p);

  std::unordered_map<int, int> lastHp_;
};

inline int idOf(BWAPI::Unit u) { return u ? u->getID() : -1; }
inline int idOf(BWAPI::Player p) { return p ? p->getID() : -1; }

}  // namespace shim
