// Client-mode entry point (Windows, BWAPI 4.4.0 + StarCraft 1.16.1).
// Runs as a separate 32-bit process next to StarCraft, connected via BWAPI's shared memory,
// and bridges the game to the bot process over TCP.

#include <BWAPI.h>
#include <BWAPI/Client.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "Bridge.h"

int main(int argc, char** argv) {
  shim::BridgeOptions opt = shim::BridgeOptions::fromEnv();
  opt.parseArgs(argc, argv);
  shim::Bridge bridge(opt);

  std::cout << "[shim] connecting to BWAPI (start StarCraft with BWAPI injected)..." << std::endl;
  while (true) {
    while (!BWAPI::BWAPIClient.connect()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    std::cout << "[shim] connected to BWAPI, waiting for a match" << std::endl;

    while (BWAPI::BWAPIClient.isConnected()) {
      // Dispatch every event, including MatchStart/MatchFrame/MatchEnd, to the bridge.
      for (const BWAPI::Event& e : BWAPI::Broodwar->getEvents()) bridge.onEvent(e);
      BWAPI::BWAPIClient.update();
    }
    std::cout << "[shim] BWAPI disconnected (StarCraft closed?); reconnecting..." << std::endl;
  }
  return 0;
}
