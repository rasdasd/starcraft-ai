// AI-module entry point. Loaded in-process by the game host:
//   * OpenBW BWAPILauncher (Linux/WSL): bwapi.ini `ai = /path/libshim_module.so` or env
//     BWAPI_CONFIG_AI__AI=... (OpenBW's fork has no client mode, so this is the only option there).
//   * StarCraft 1.16.1 + BWAPI 4.4.0 (Windows): bwapi-data/AI/shim_module.dll
//     (competitions rename this to BotName.dll).
//
// Competition pack: the module listens on TCP and, unless BWBOT_NO_SPAWN is set, starts a
// 64-bit frozen bot next to the DLL (`bot.exe` or `bot/bot.exe`, override BWBOT_BOT).
// AIIDE run_proxy.bat can start the bot instead; then set BWBOT_NO_SPAWN=1.
//
// Configuration via env: BWBOT_HOST, BWBOT_PORT, BWBOT_STATS_EVERY, BWBOT_VERBOSE,
// BWBOT_BOT, BWBOT_BOT_SPEC, BWBOT_NO_SPAWN.

#include <BWAPI.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "Bridge.h"

#ifdef _WIN32
#include <Windows.h>
#define SHIM_EXPORT __declspec(dllexport)
static HMODULE g_self = nullptr;
#else
#include <unistd.h>
#define SHIM_EXPORT __attribute__((visibility("default")))
#endif

namespace {

void spawnBotIfNeeded() {
  if (std::getenv("BWBOT_NO_SPAWN")) return;
  const char* port = std::getenv("BWBOT_PORT");
  if (!port || !*port) port = "8765";
  const char* spec = std::getenv("BWBOT_BOT_SPEC");
#ifdef _WIN32
  char dll[MAX_PATH] = {};
  if (!g_self || !GetModuleFileNameA(g_self, dll, MAX_PATH)) return;
  std::string dir(dll);
  auto slash = dir.find_last_of("\\/");
  if (slash != std::string::npos) dir.resize(slash);
  const char* env = std::getenv("BWBOT_BOT");
  std::string exe = (env && *env) ? env : (dir + "\\bot.exe");
  if ((!env || !*env) && GetFileAttributesA(exe.c_str()) == INVALID_FILE_ATTRIBUTES)
    exe = dir + "\\bot\\bot.exe";
  if (GetFileAttributesA(exe.c_str()) == INVALID_FILE_ATTRIBUTES) return;
  std::string cmd = "\"" + exe + "\"";
  if (spec && *spec) {
    cmd += " ";
    cmd += spec;
  }
  cmd += " --host 127.0.0.1 --port ";
  cmd += port;
  STARTUPINFOA si{};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  std::string cwd = dir;
  if (CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, cwd.c_str(), &si, &pi)) {
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    std::cout << "[shim] spawned " << exe << std::endl;
  } else {
    std::cerr << "[shim] spawn failed (" << GetLastError() << "): " << exe << std::endl;
  }
#else
  const char* exe = std::getenv("BWBOT_BOT");
  if (!exe || !*exe) return;
  pid_t pid = fork();
  if (pid == 0) {
    execl(exe, exe, spec && *spec ? spec : "mybot", "--host", "127.0.0.1", "--port", port,
          static_cast<char*>(nullptr));
    _exit(127);
  }
  if (pid > 0) std::cout << "[shim] spawned " << exe << " pid=" << pid << std::endl;
#endif
}

class ShimModule final : public BWAPI::AIModule {
public:
  ShimModule() : bridge_(shim::BridgeOptions::fromEnv()) { spawnBotIfNeeded(); }

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
BOOL APIENTRY DllMain(HINSTANCE h, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) g_self = h;
  return TRUE;
}
#endif
