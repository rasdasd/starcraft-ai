#pragma once
// Applies a bw::Commands message to the live BWAPI game.

#include <BWAPI.h>

#include <cstdint>
#include <vector>

#include "bw_generated.h"

namespace shim {

struct ApplyStats {
  int unitCommands = 0;
  int unitCommandsFailed = 0;
  int gameCommands = 0;
  int draws = 0;
};

class CommandApplier {
public:
  ApplyStats apply(const bw::Commands& cmds);

  // Re-issue the most recently received draw commands (BWAPI draws last one frame only, so
  // between bot decisions, when frame_skip > 1, the previous overlay is kept on screen).
  void redrawLast();

  void clear() { lastDraws_.clear(); }

private:
  bool applyUnitCommand(const bw::UnitCommand& c);
  void applyGameCommand(const bw::GameCommand& c);
  void applyDraw(const bw::DrawCommandT& d);

  std::vector<bw::DrawCommandT> lastDraws_;
};

}  // namespace shim
