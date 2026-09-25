#include "MapAnalysis.h"

#include <BWAPI.h>
#include <bwem.h>

#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

namespace shim {
namespace {

using namespace BWAPI;
namespace fb = flatbuffers;

int pathLength(const BWEM::Map& theMap, Position a, Position b) {
  int len = -1;
  theMap.GetPath(a, b, &len);
  return len;
}

int nearestBaseId(const std::vector<const BWEM::Base*>& bases,
                  const std::unordered_map<const BWEM::Base*, int>& ids,
                  const BWEM::Map& theMap, Position from, const BWEM::Base* skip) {
  int best = -1;
  int bestPath = std::numeric_limits<int>::max();
  int bestEuclid = std::numeric_limits<int>::max();
  for (const BWEM::Base* b : bases) {
    if (b == skip) continue;
    int id = ids.at(b);
    int plen = pathLength(theMap, from, b->Center());
    int euclid = from.getApproxDistance(b->Center());
    if (plen >= 0 && plen < bestPath) {
      bestPath = plen;
      best = id;
    } else if (bestPath == std::numeric_limits<int>::max() && euclid < bestEuclid) {
      bestEuclid = euclid;
      best = id;
    }
  }
  return best;
}

}  // namespace

void initMapAnalysis() {
  if (!BroodwarPtr) return;
  try {
    BWEM::Map& theMap = BWEM::Map::Instance();
    theMap.Initialize(BroodwarPtr);
    if (!theMap.FindBasesForStartingLocations()) {
      std::cout << "[shim] BWEM: FindBasesForStartingLocations did not assign every start" << std::endl;
    }
    std::cout << "[shim] BWEM: " << theMap.Areas().size() << " areas, " << theMap.BaseCount()
              << " bases, " << theMap.ChokePointCount() << " chokes" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[shim] BWEM Initialize failed: " << e.what() << std::endl;
  }
}

MapAnalysisOffsets buildMapAnalysis(fb::FlatBufferBuilder& fbb) {
  MapAnalysisOffsets out;
  BWEM::Map& theMap = BWEM::Map::Instance();
  if (!theMap.Initialized()) return out;

  std::vector<const BWEM::Base*> bases;
  std::unordered_map<const BWEM::Base*, int> baseIds;
  for (const BWEM::Area& area : theMap.Areas()) {
    for (const BWEM::Base& base : area.Bases()) {
      baseIds[&base] = int(bases.size());
      bases.push_back(&base);
    }
  }

  std::unordered_map<const BWEM::ChokePoint*, int> chokeIds;
  std::vector<const BWEM::ChokePoint*> chokes;
  for (const BWEM::Area& area : theMap.Areas()) {
    for (const BWEM::ChokePoint* cp : area.ChokePoints()) {
      if (chokeIds.find(cp) != chokeIds.end()) continue;
      chokeIds[cp] = int(chokes.size());
      chokes.push_back(cp);
    }
  }

  std::vector<fb::Offset<bw::MapArea>> areaOffs;
  areaOffs.reserve(theMap.Areas().size());
  for (const BWEM::Area& area : theMap.Areas()) {
    bw::MapAreaBuilder b(fbb);
    b.add_id(area.Id());
    b.add_top_x(area.Top().x);
    b.add_top_y(area.Top().y);
    b.add_left(area.TopLeft().x);
    b.add_top(area.TopLeft().y);
    b.add_right(area.BottomRight().x);
    b.add_bottom(area.BottomRight().y);
    areaOffs.push_back(b.Finish());
  }
  out.areas = fbb.CreateVector(areaOffs);

  std::vector<fb::Offset<bw::MapBase>> baseOffs;
  baseOffs.reserve(bases.size());
  for (const BWEM::Base* base : bases) {
    bw::MapBaseBuilder b(fbb);
    b.add_id(baseIds[base]);
    b.add_area_id(base->GetArea() ? base->GetArea()->Id() : -1);
    b.add_tile_x(base->Location().x);
    b.add_tile_y(base->Location().y);
    b.add_center_x(base->Center().x);
    b.add_center_y(base->Center().y);
    b.add_minerals(int(base->Minerals().size()));
    b.add_geysers(int(base->Geysers().size()));
    b.add_starting(base->Starting());
    baseOffs.push_back(b.Finish());
  }
  out.bases = fbb.CreateVector(baseOffs);

  std::vector<fb::Offset<bw::MapChoke>> chokeOffs;
  chokeOffs.reserve(chokes.size());
  for (const BWEM::ChokePoint* cp : chokes) {
    Position c(cp->Center());
    Position e1(cp->Pos(BWEM::ChokePoint::end1));
    Position e2(cp->Pos(BWEM::ChokePoint::end2));
    const BWEM::Area* a = cp->GetAreas().first;
    const BWEM::Area* bArea = cp->GetAreas().second;
    bw::MapChokeBuilder b(fbb);
    b.add_id(chokeIds[cp]);
    b.add_area_a(a ? a->Id() : -1);
    b.add_area_b(bArea ? bArea->Id() : -1);
    b.add_center_x(c.x);
    b.add_center_y(c.y);
    b.add_width(e1.getApproxDistance(e2));
    b.add_blocking(cp->Blocked() || cp->IsPseudo());
    chokeOffs.push_back(b.Finish());
  }
  out.chokes = fbb.CreateVector(chokeOffs);

  std::vector<fb::Offset<bw::StartBase>> startOffs;
  for (const TilePosition& sl : theMap.StartingLocations()) {
    const BWEM::Base* assigned = nullptr;
    int bestD = std::numeric_limits<int>::max();
    for (const BWEM::Base* base : bases) {
      int d = sl.getApproxDistance(base->Location());
      if (d < bestD) {
        bestD = d;
        assigned = base;
      }
    }
    int baseId = assigned ? baseIds[assigned] : -1;
    int naturalId = assigned ? nearestBaseId(bases, baseIds, theMap, assigned->Center(), assigned) : -1;
    bw::StartBaseBuilder b(fbb);
    b.add_tile_x(sl.x);
    b.add_tile_y(sl.y);
    b.add_base_id(baseId);
    b.add_natural_id(naturalId);
    startOffs.push_back(b.Finish());
    if (Broodwar->self() && sl == Broodwar->self()->getStartLocation()) {
      out.selfMainId = baseId;
      out.selfNaturalId = naturalId;
    }
  }
  out.startBases = fbb.CreateVector(startOffs);
  return out;
}

}  // namespace shim
