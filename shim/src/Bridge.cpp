#include "Bridge.h"
#include "MapAnalysis.h"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace shim {

using namespace BWAPI;
using clock_t_ = std::chrono::steady_clock;

static int usSince(clock_t_::time_point t0) {
  return int(std::chrono::duration_cast<std::chrono::microseconds>(clock_t_::now() - t0).count());
}

// ---------------------------------------------------------------------------

BridgeOptions BridgeOptions::fromEnv() {
  BridgeOptions o;
  if (const char* v = std::getenv("BWBOT_HOST")) o.host = v;
  if (const char* v = std::getenv("BWBOT_PORT")) o.port = uint16_t(std::atoi(v));
  if (const char* v = std::getenv("BWBOT_STATS_EVERY")) o.statsEvery = std::atoi(v);
  if (const char* v = std::getenv("BWBOT_VERBOSE")) o.verbose = std::atoi(v) != 0;
  if (const char* v = std::getenv("BWBOT_AUTO_RESUME")) o.autoResume = std::atoi(v) != 0;
  return o;
}

void BridgeOptions::parseArgs(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--host") host = next();
    else if (a == "--port") port = uint16_t(std::atoi(next()));
    else if (a == "--stats-every") statsEvery = std::atoi(next());
    else if (a == "--verbose" || a == "-v") verbose = true;
    else if (a == "--no-auto-resume") autoResume = false;
    else if (a == "--help" || a == "-h") {
      std::cout << "shim [--host 127.0.0.1] [--port 8765] [--stats-every N] [--verbose] [--no-auto-resume]\n"
                   "env: BWBOT_HOST BWBOT_PORT BWBOT_STATS_EVERY BWBOT_VERBOSE BWBOT_AUTO_RESUME\n";
      std::exit(0);
    }
  }
}

// ---------------------------------------------------------------------------

Bridge::Bridge(BridgeOptions opt) : opt_(std::move(opt)) {
  transport_ = makeTcpServer(opt_.host, opt_.port);
  std::cout << "[shim] " << SHIM_BACKEND_NAME << " v" << SHIM_VERSION << " listening at " << transport_->describe()
            << std::endl;
}

Bridge::~Bridge() = default;

bool Bridge::sendBuffer() {
  bool ok = transport_->send(fbb_.GetBufferPointer(), fbb_.GetSize());
  bytesAcc_.add(int(fbb_.GetSize()));
  return ok;
}

// ---------------------------------------------------------------------------

void Bridge::applyConfig(const bw::ClientConfig& cfg, bool atStart) {
  frameSkip_ = cfg.frame_skip() < 1 ? 1 : cfg.frame_skip();
  serOpt_.includeBullets = cfg.include_bullets();
  serOpt_.includeTiles = cfg.include_tiles();
  serOpt_.includePlayers = cfg.include_players();
  if (atStart) {
    if (cfg.complete_map_information()) Broodwar->enableFlag(Flag::CompleteMapInformation);
    if (cfg.user_input()) Broodwar->enableFlag(Flag::UserInput);
  } else if (cfg.complete_map_information() && !Broodwar->isFlagEnabled(Flag::CompleteMapInformation)) {
    std::cout << "[shim] warning: CompleteMapInformation can only be enabled at match start" << std::endl;
  }
  if (cfg.local_speed() >= 0) Broodwar->setLocalSpeed(cfg.local_speed());
  Broodwar->setGUI(cfg.gui());
  std::cout << "[shim] config: frame_skip=" << frameSkip_ << " bullets=" << serOpt_.includeBullets
            << " tiles=" << serOpt_.includeTiles << " local_speed=" << cfg.local_speed() << " gui=" << cfg.gui()
            << " cmi=" << cfg.complete_map_information() << std::endl;
}

bool Bridge::handshake() {
  fbb_.Clear();
  serializer_.buildHello(fbb_);
  if (!sendBuffer()) return false;

  // Expect a ClientConfig (bot may skip it and go straight to waiting for GameStart, so a
  // missing/invalid message is not fatal; but the wire is lockstep so we do require one).
  if (!transport_->recv(rx_)) return false;
  flatbuffers::Verifier v(rx_.data(), rx_.size());
  if (!bw::VerifyEnvelopeBuffer(v)) {
    std::cerr << "[shim] handshake: invalid ClientConfig buffer" << std::endl;
    transport_->closePeer();
    return false;
  }
  auto* env = bw::GetEnvelope(rx_.data());
  if (env->msg_type() == bw::Message::ClientConfig) {
    applyConfig(*env->msg_as_ClientConfig(), /*atStart=*/!gameStartSent_ && Broodwar->getFrameCount() == 0);
  } else {
    std::cerr << "[shim] handshake: expected ClientConfig, got " << bw::EnumNameMessage(env->msg_type()) << std::endl;
  }

  fbb_.Clear();
  serializer_.buildGameStart(fbb_, frameSkip_);
  if (!sendBuffer()) return false;
  gameStartSent_ = true;
  std::cout << "[shim] sent GameStart (" << fbb_.GetSize() / 1024 << " KB) map=" << Broodwar->mapName()
            << " self=" << (Broodwar->self() ? Broodwar->self()->getName() : "?") << std::endl;
  return true;
}

bool Bridge::ensureConnected() {
  if (transport_->isConnected()) return true;
  while (true) {
    if (!transport_->waitForPeer()) return false;
    if (handshake()) return true;
    std::cerr << "[shim] handshake failed; waiting for another bot" << std::endl;
  }
}

// ---------------------------------------------------------------------------

void Bridge::onStart() {
  inGame_ = true;
  gameStartSent_ = false;
  pendingEvents_.clear();
  serializer_.reset();
  applier_.clear();
  serAcc_.reset(); rttAcc_.reset(); applyAcc_.reset(); bytesAcc_.reset();
  decisions_ = 0;
  statsT0_ = clock_t_::now();
  std::cout << "[shim] match start: " << Broodwar->mapFileName() << " frame=" << Broodwar->getFrameCount() << std::endl;
  initMapAnalysis();
  if (Broodwar->isReplay()) {
    std::cout << "[shim] replay mode: streaming frames, commands ignored by the game" << std::endl;
  }
  // If a bot is already connected (previous game with auto_restart), send the new GameStart.
  if (transport_->isConnected()) {
    if (!handshake()) transport_->closePeer();
  }
  ensureConnected();
}

void Bridge::onEvent(const Event& e) {
  switch (e.getType()) {
    case EventType::MatchStart: onStart(); break;
    case EventType::MatchFrame: onFrame(); break;
    case EventType::MatchEnd: onEnd(e.isWinner()); break;
    case EventType::MenuFrame: break;
    default: pendingEvents_.push_back(e); break;
  }
}

void Bridge::onFrame() {
  if (!inGame_) return;
  const int frame = Broodwar->getFrameCount();
  // Single-player StarCraft pauses itself when its window loses focus (BWAPI keeps calling
  // onFrame while paused, with a frozen frame count). Unpause so unattended runs progress.
  if (opt_.autoResume && Broodwar->isPaused() && !Broodwar->isMultiplayer()) {
    if (++pausedUpdates_ % 50 == 1) {
      std::cout << "[shim] game is paused at frame " << frame << "; resuming" << std::endl;
      Broodwar->resumeGame();
    }
  } else {
    pausedUpdates_ = 0;
  }
  if (frame % frameSkip_ != 0) {
    applier_.redrawLast();
    return;
  }
  if (!ensureConnected()) return;
  if (!exchangeFrame()) {
    // Bot went away mid-game: block until it (or another) reconnects; the game stalls
    // meanwhile in client mode, or runs on with idle units in module mode.
    std::cerr << "[shim] lost bot at frame " << frame << std::endl;
  }
  if (opt_.statsEvery > 0 && decisions_ > 0 && decisions_ % opt_.statsEvery == 0) printStats();
}

static std::vector<bw::PlacementResultT> answerPlacement(const bw::Commands& cmds) {
  std::vector<bw::PlacementResultT> out;
  auto* qs = cmds.placement_queries();
  if (!qs) return out;
  out.reserve(qs->size());
  for (const bw::PlacementQuery* q : *qs) {
    bw::PlacementResultT r;
    r.id = q->id();
    BWAPI::UnitType type(q->unit_type());
    BWAPI::TilePosition desired(q->tile_x(), q->tile_y());
    if (q->kind() == 1) {
      int range = q->max_range() > 0 ? q->max_range() : 64;
      BWAPI::TilePosition tp = Broodwar->getBuildLocation(type, desired, range, false);
      r.ok = tp.isValid();
      r.tile_x = tp.x;
      r.tile_y = tp.y;
    } else {
      r.ok = Broodwar->canBuildHere(desired, type, nullptr, true);
      r.tile_x = q->tile_x();
      r.tile_y = q->tile_y();
    }
    out.push_back(std::move(r));
  }
  return out;
}

bool Bridge::exchangeFrame() {
  auto t0 = clock_t_::now();
  fbb_.Clear();
  serializer_.buildFrame(fbb_, pendingEvents_, serOpt_, serAcc_.n ? int(serAcc_.sum / serAcc_.n) : 0, lastRoundtripUs_);
  pendingEvents_.clear();
  unitsLast_ = int(Broodwar->getAllUnits().size());
  int serUs = usSince(t0);
  serAcc_.add(serUs);

  auto t1 = clock_t_::now();
  if (!sendBuffer()) return false;

  // Wait for Commands (a ClientConfig may be interleaved to change cadence at runtime).
  while (true) {
    if (!transport_->recv(rx_)) return false;
    flatbuffers::Verifier v(rx_.data(), rx_.size());
    if (!bw::VerifyEnvelopeBuffer(v)) {
      std::cerr << "[shim] invalid message from bot (" << rx_.size() << " bytes); dropping connection" << std::endl;
      transport_->closePeer();
      return false;
    }
    auto* env = bw::GetEnvelope(rx_.data());
    if (env->msg_type() == bw::Message::Commands) {
      lastRoundtripUs_ = usSince(t1);
      rttAcc_.add(lastRoundtripUs_);
      auto t2 = clock_t_::now();
      const bw::Commands* cmds = env->msg_as_Commands();
      ApplyStats st = applier_.apply(*cmds);
      serializer_.setPlacementResults(answerPlacement(*cmds));
      applyAcc_.add(usSince(t2));
      if (opt_.verbose && st.unitCommandsFailed)
        std::cout << "[shim] frame " << Broodwar->getFrameCount() << ": " << st.unitCommandsFailed << "/"
                  << st.unitCommands << " unit commands rejected" << std::endl;
      ++decisions_;
      return true;
    }
    if (env->msg_type() == bw::Message::ClientConfig) {
      applyConfig(*env->msg_as_ClientConfig(), false);
      continue;
    }
    std::cerr << "[shim] unexpected message " << bw::EnumNameMessage(env->msg_type()) << "; ignoring" << std::endl;
  }
}

void Bridge::onEnd(bool isWinner) {
  if (!inGame_) return;
  inGame_ = false;
  std::cout << "[shim] match end: " << (isWinner ? "WIN" : "LOSS/DRAW") << " at frame " << Broodwar->getFrameCount()
            << std::endl;
  printStats();
  if (transport_->isConnected()) {
    fbb_.Clear();
    serializer_.buildGameEnd(fbb_, isWinner);
    sendBuffer();
  }
  pendingEvents_.clear();
}

void Bridge::printStats() {
  if (decisions_ == 0) return;
  double wall = std::chrono::duration<double>(clock_t_::now() - statsT0_).count();
  auto ms = [](double us) { return us / 1000.0; };
  std::cout << std::fixed << std::setprecision(2) << "[shim] frame " << Broodwar->getFrameCount() << " decisions="
            << decisions_ << " units=" << unitsLast_ << " | serialize avg " << ms(serAcc_.sum / serAcc_.n) << "ms max "
            << ms(serAcc_.max) << " | rtt avg " << ms(rttAcc_.sum / std::max(1, rttAcc_.n)) << "ms max "
            << ms(rttAcc_.max) << " | apply avg " << ms(applyAcc_.sum / std::max(1, applyAcc_.n)) << "ms | "
            << (bytesAcc_.sum / std::max(1, bytesAcc_.n) / 1024.0) << " KB/msg | " << (decisions_ / std::max(1e-6, wall))
            << " decisions/s" << std::endl;
}

}  // namespace shim
