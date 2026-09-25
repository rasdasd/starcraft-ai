#pragma once
// BWEM initialize + FlatBuffers snapshot for GameStart.

#include "bw_generated.h"
#include "flatbuffers/flatbuffers.h"

namespace shim {

// Re-runs BWEM on the current map. Safe to call every match start.
void initMapAnalysis();

struct MapAnalysisOffsets {
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<bw::MapArea>>> areas = 0;
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<bw::MapBase>>> bases = 0;
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<bw::MapChoke>>> chokes = 0;
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<bw::StartBase>>> startBases = 0;
  int selfMainId = -1;
  int selfNaturalId = -1;
};

// Build vectors (must be called before GameStartBuilder starts). Empty if BWEM failed.
MapAnalysisOffsets buildMapAnalysis(flatbuffers::FlatBufferBuilder& fbb);

}  // namespace shim
