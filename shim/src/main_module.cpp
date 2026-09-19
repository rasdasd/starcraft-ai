// AI-module entry point. Loaded in-process by the game host:
//   * OpenBW BWAPILauncher (Linux/WSL): bwapi.ini `ai = /path/libshim_module.so` or env
//     BWAPI_CONFIG_AI__AI=... (OpenBW's fork has no client mode, so this is the only option there).
//   * StarCraft 1.16.1 + BWAPI 4.4.0 (Windows): bwapi-data/AI/shim_module.dll.
// Configuration via env: BWBOT_HOST, BWBOT_PORT, BWBOT_STATS_EVERY, BWBOT_VERBOSE.

#include <BWAPI.h>

#include <exception>
#include <iostream>
#include <memory>

#include "Bridge.h"

#ifdef _WIN32
#include <Windows.h>
#define SHIM_EXPORT __declspec(dllexport)
#else
#define SHIM_EXPORT __attribute__((visibility("default")))
#endif

namespace {

class ShimModule final : public BWAPI::AIModule {
public:
  ShimModule() : bridge_(shim::BridgeOptions::fromEnv()) {}

  // A throwing BWAPI stub or transport error must not unwind into the game host.
  void onStart() override { guard("onStart", [&] { bridge_.onStart(); }); }
  void onFrame() override { guard("onFrame", [&] { bridge_.onFrame(); }); }
  void onEnd(bool isWinner) override { guard("onEnd", [&] { bridge_.onEnd(isWinner); }); }

  void onSendText(std::string text) override { push(BWAPI::EventType::SendText, ev().setText(text.c_str())); }
  void onReceiveText(BWAPI::Player player, std::string text) override {
    push(BWAPI::EventType::ReceiveText, ev().setPlayer(player).setText(text.c_str()));
  }
  void onPlayerLeft(BWAPI::Player player) override { push(BWAPI::EventType::PlayerLeft, ev().setPlayer(player)); }
  void onNukeDetect(BWAPI::Position target) override { push(BWAPI::EventType::NukeDetect, ev().setPosition(target)); }
  void onUnitDiscover(BWAPI::Unit u) override { push(BWAPI::EventType::UnitDiscover, ev().setUnit(u)); }
  void onUnitEvade(BWAPI::Unit u) override { push(BWAPI::EventType::UnitEvade, ev().setUnit(u)); }
  void onUnitShow(BWAPI::Unit u) override { push(BWAPI::EventType::UnitShow, ev().setUnit(u)); }
  void onUnitHide(BWAPI::Unit u) override { push(BWAPI::EventType::UnitHide, ev().setUnit(u)); }
  void onUnitCreate(BWAPI::Unit u) override { push(BWAPI::EventType::UnitCreate, ev().setUnit(u)); }
  void onUnitDestroy(BWAPI::Unit u) override { push(BWAPI::EventType::UnitDestroy, ev().setUnit(u)); }
  void onUnitMorph(BWAPI::Unit u) override { push(BWAPI::EventType::UnitMorph, ev().setUnit(u)); }
  void onUnitRenegade(BWAPI::Unit u) override { push(BWAPI::EventType::UnitRenegade, ev().setUnit(u)); }
  void onSaveGame(std::string name) override { push(BWAPI::EventType::SaveGame, ev().setText(name.c_str())); }
  void onUnitComplete(BWAPI::Unit u) override { push(BWAPI::EventType::UnitComplete, ev().setUnit(u)); }

private:
  template <class F>
  void guard(const char* what, F&& f) {
    try {
      f();
    } catch (const std::exception& e) {
      std::cerr << "[shim] " << what << " failed: " << e.what() << std::endl;
    }
  }
  BWAPI::Event ev() { return BWAPI::Event(); }
  void push(BWAPI::EventType::Enum type, BWAPI::Event e) { bridge_.onEvent(e.setType(type)); }

  shim::Bridge bridge_;
};

}  // namespace

extern "C" SHIM_EXPORT void gameInit(BWAPI::Game* game) { BWAPI::BroodwarPtr = game; }
extern "C" SHIM_EXPORT BWAPI::AIModule* newAIModule() { return new ShimModule(); }

#ifdef _WIN32
BOOL APIENTRY DllMain(HANDLE, DWORD, LPVOID) { return TRUE; }
#endif
