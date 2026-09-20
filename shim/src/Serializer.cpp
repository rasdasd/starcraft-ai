#include "Serializer.h"

#include <algorithm>
#include <string>

namespace shim {

using namespace BWAPI;
namespace fb = flatbuffers;

namespace {

template <class F>
uint32_t bit(F cond, bw::UnitFlag flag) {
  return cond ? static_cast<uint32_t>(flag) : 0u;
}

uint32_t unitTypeFlags(const UnitType& t) {
  using F = bw::UnitTypeFlag;
  uint32_t f = 0;
  auto set = [&](bool c, F flag) { if (c) f |= static_cast<uint32_t>(flag); };
  set(t.isBuilding(), F::Building);
  set(t.isWorker(), F::Worker);
  set(t.isFlyer(), F::Flyer);
  set(t.isOrganic(), F::Organic);
  set(t.isMechanical(), F::Mechanical);
  set(t.isRobotic(), F::Robotic);
  set(t.isDetector(), F::Detector);
  set(t.isResourceContainer(), F::ResourceContainer);
  set(t.isResourceDepot(), F::ResourceDepot);
  set(t.isRefinery(), F::Refinery);
  set(t.isAddon(), F::Addon);
  set(t.isFlyingBuilding(), F::FlyingBuilding);
  set(t.isSpell(), F::Spell);
  set(t.isInvincible(), F::Invincible);
  set(t.isBurrowable(), F::Burrowable);
  set(t.isCloakable(), F::Cloakable);
  set(t.isHero(), F::Hero);
  set(t.isPowerup(), F::Powerup);
  set(t.isBeacon(), F::Beacon);
  set(t.isCritter(), F::Critter);
  set(t.isNeutral(), F::Neutral);
  set(t.canProduce(), F::CanProduce);
  set(t.canAttack(), F::CanAttack);
  set(t.canMove(), F::CanMove);
  set(t.regeneratesHP(), F::RegeneratesHP);
  set(t.isMineralField(), F::IsMineralField);
  set(t.isSpecialBuilding(), F::IsSpecialBuilding);
  set(t.producesLarva(), F::ProducesLarva);
  set(t.requiresPsi(), F::RequiresPsi);
  set(t.requiresCreep(), F::RequiresCreep);
  set(t.isTwoUnitsInOneEgg(), F::TwoUnitsInOneEgg);
  return f;
}

}  // namespace

// ---------------------------------------------------------------------------

void Serializer::buildHello(fb::FlatBufferBuilder& fbb) {
  int clientVersion = 0, revision = 0;
#ifndef SHIM_OPENBW
  clientVersion = BroodwarPtr ? Broodwar->getClientVersion() : 0;
  revision = BroodwarPtr ? Broodwar->getRevision() : 0;
#endif
  auto hello = bw::CreateHelloDirect(fbb, /*protocol_version=*/1, SHIM_VERSION, SHIM_BACKEND_NAME, clientVersion, revision);
  auto env = bw::CreateEnvelope(fbb, bw::Message::Hello, hello.Union());
  fbb.Finish(env, bw::EnvelopeIdentifier());
}

// ---------------------------------------------------------------------------

fb::Offset<bw::PlayerInfo> Serializer::playerInfo(fb::FlatBufferBuilder& fbb, Player p) {
  Player self = Broodwar->self();
  TilePosition sl = p->getStartLocation();
  bw::Pos start(sl.isValid() ? sl.x : -1, sl.isValid() ? sl.y : -1);
  auto name = fbb.CreateString(p->getName());
  bw::PlayerInfoBuilder b(fbb);
  b.add_id(p->getID());
  b.add_name(name);
  b.add_race(p->getRace().getID());
  b.add_type(p->getType().getID());
  b.add_force(p->getForce() ? p->getForce()->getID() : -1);
  b.add_is_self(p == self);
  b.add_is_enemy(self && self->isEnemy(p));
  b.add_is_ally(self && self != p && self->isAlly(p));
  b.add_is_neutral(p->isNeutral());
  b.add_is_observer(p->isObserver());
  b.add_start_location(&start);
  b.add_color(p->getColor().getID());
  return b.Finish();
}

void Serializer::buildGameStart(fb::FlatBufferBuilder& fbb, int frameSkip) {
  Game* g = BroodwarPtr;
  const int w = g->mapWidth(), h = g->mapHeight();

  // --- tile maps -----------------------------------------------------------
  std::vector<uint8_t> ground(size_t(w) * h), buildable(size_t(w) * h);
  std::vector<uint16_t> region(size_t(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      size_t i = size_t(y) * w + x;
      ground[i] = uint8_t(g->getGroundHeight(x, y));
      buildable[i] = g->isBuildable(x, y, false) ? 1 : 0;
      Region r = g->getRegionAt(x * 32 + 16, y * 32 + 16);
      region[i] = uint16_t(r ? r->getID() : 0);
    }
  const int ww = w * 4, wh = h * 4;
  std::vector<uint8_t> walkable(size_t(ww) * wh);
  for (int y = 0; y < wh; ++y)
    for (int x = 0; x < ww; ++x) walkable[size_t(y) * ww + x] = g->isWalkable(x, y) ? 1 : 0;

  auto groundOff = fbb.CreateVector(ground);
  auto buildableOff = fbb.CreateVector(buildable);
  auto walkableOff = fbb.CreateVector(walkable);
  auto regionOff = fbb.CreateVector(region);

  std::vector<bw::Pos> starts;
  for (const TilePosition& tp : g->getStartLocations()) starts.emplace_back(tp.x, tp.y);
  auto startsOff = fbb.CreateVectorOfStructs(starts);

  std::vector<fb::Offset<bw::PlayerInfo>> players;
  for (Player p : g->getPlayers()) players.push_back(playerInfo(fbb, p));
  auto playersOff = fbb.CreateVector(players);

  // --- type tables ---------------------------------------------------------
  std::vector<fb::Offset<bw::UnitTypeInfo>> unitTypes;
  for (const UnitType& t : UnitTypes::allUnitTypes()) {
    std::vector<int> req;
    for (const auto& kv : t.requiredUnits()) req.push_back(kv.first.getID());
    auto name = fbb.CreateString(t.getName());
    auto reqOff = fbb.CreateVector(req);
    bw::UnitTypeInfoBuilder b(fbb);
    b.add_id(t.getID());
    b.add_name(name);
    b.add_race(t.getRace().getID());
    b.add_mineral_price(t.mineralPrice());
    b.add_gas_price(t.gasPrice());
    b.add_build_time(t.buildTime());
    b.add_supply_required(t.supplyRequired());
    b.add_supply_provided(t.supplyProvided());
    b.add_max_hit_points(t.maxHitPoints());
    b.add_max_shields(t.maxShields());
    b.add_max_energy(t.maxEnergy());
    b.add_armor(t.armor());
    b.add_ground_weapon(t.groundWeapon().getID());
    b.add_air_weapon(t.airWeapon().getID());
    b.add_max_ground_hits(t.maxGroundHits());
    b.add_max_air_hits(t.maxAirHits());
    b.add_sight_range(t.sightRange());
    b.add_seek_range(t.seekRange());
    b.add_top_speed(float(t.topSpeed()));
    b.add_acceleration(t.acceleration());
    b.add_tile_width(t.tileWidth());
    b.add_tile_height(t.tileHeight());
    b.add_dimension_left(t.dimensionLeft());
    b.add_dimension_up(t.dimensionUp());
    b.add_dimension_right(t.dimensionRight());
    b.add_dimension_down(t.dimensionDown());
    b.add_size(t.size().getID());
    b.add_what_builds(t.whatBuilds().first.getID());
    b.add_what_builds_count(t.whatBuilds().second);
    b.add_required_units(reqOff);
    b.add_required_tech(t.requiredTech().getID());
    b.add_space_required(t.spaceRequired());
    b.add_space_provided(t.spaceProvided());
    b.add_flags(unitTypeFlags(t));
    unitTypes.push_back(b.Finish());
  }
  auto unitTypesOff = fbb.CreateVector(unitTypes);

  std::vector<fb::Offset<bw::WeaponTypeInfo>> weaponTypes;
  for (const WeaponType& t : WeaponTypes::allWeaponTypes()) {
    auto name = fbb.CreateString(t.getName());
    bw::WeaponTypeInfoBuilder b(fbb);
    b.add_id(t.getID());
    b.add_name(name);
    b.add_tech(t.getTech().getID());
    b.add_what_uses(t.whatUses().getID());
    b.add_damage_amount(t.damageAmount());
    b.add_damage_bonus(t.damageBonus());
    b.add_damage_cooldown(t.damageCooldown());
    b.add_damage_factor(t.damageFactor());
    b.add_upgrade_type(t.upgradeType().getID());
    b.add_damage_type(t.damageType().getID());
    b.add_explosion_type(t.explosionType().getID());
    b.add_min_range(t.minRange());
    b.add_max_range(t.maxRange());
    b.add_inner_splash_radius(t.innerSplashRadius());
    b.add_median_splash_radius(t.medianSplashRadius());
    b.add_outer_splash_radius(t.outerSplashRadius());
    b.add_targets_air(t.targetsAir());
    b.add_targets_ground(t.targetsGround());
    weaponTypes.push_back(b.Finish());
  }
  auto weaponTypesOff = fbb.CreateVector(weaponTypes);

  std::vector<fb::Offset<bw::UpgradeTypeInfo>> upgradeTypes;
  for (const UpgradeType& t : UpgradeTypes::allUpgradeTypes()) {
    auto name = fbb.CreateString(t.getName());
    bw::UpgradeTypeInfoBuilder b(fbb);
    b.add_id(t.getID());
    b.add_name(name);
    b.add_race(t.getRace().getID());
    b.add_mineral_price(t.mineralPrice());
    b.add_mineral_price_factor(t.mineralPriceFactor());
    b.add_gas_price(t.gasPrice());
    b.add_gas_price_factor(t.gasPriceFactor());
    b.add_upgrade_time(t.upgradeTime());
    b.add_upgrade_time_factor(t.upgradeTimeFactor());
    b.add_max_repeats(t.maxRepeats());
    b.add_what_upgrades(t.whatUpgrades().getID());
    upgradeTypes.push_back(b.Finish());
  }
  auto upgradeTypesOff = fbb.CreateVector(upgradeTypes);

  std::vector<fb::Offset<bw::TechTypeInfo>> techTypes;
  for (const TechType& t : TechTypes::allTechTypes()) {
    auto name = fbb.CreateString(t.getName());
    bw::TechTypeInfoBuilder b(fbb);
    b.add_id(t.getID());
    b.add_name(name);
    b.add_race(t.getRace().getID());
    b.add_mineral_price(t.mineralPrice());
    b.add_gas_price(t.gasPrice());
    b.add_research_time(t.researchTime());
    b.add_energy_cost(t.energyCost());
    b.add_what_researches(t.whatResearches().getID());
    b.add_weapon(t.getWeapon().getID());
    b.add_targets_unit(t.targetsUnit());
    b.add_targets_position(t.targetsPosition());
    techTypes.push_back(b.Finish());
  }
  auto techTypesOff = fbb.CreateVector(techTypes);

  auto mapName = fbb.CreateString(g->mapName());
  auto mapFile = fbb.CreateString(g->mapFileName());
  auto mapPath = fbb.CreateString(g->mapPathName());
  auto mapHash = fbb.CreateString(g->mapHash());

  unsigned seed = 0;
#ifndef SHIM_OPENBW
  seed = g->getRandomSeed();
#endif

  bw::GameStartBuilder b(fbb);
  b.add_map_name(mapName);
  b.add_map_file_name(mapFile);
  b.add_map_path_name(mapPath);
  b.add_map_hash(mapHash);
  b.add_map_width(w);
  b.add_map_height(h);
  b.add_ground_height(groundOff);
  b.add_buildable(buildableOff);
  b.add_walkable(walkableOff);
  b.add_region_id(regionOff);
  b.add_start_locations(startsOff);
  b.add_players(playersOff);
  b.add_self_id(idOf(g->self()));
  b.add_enemy_id(idOf(g->enemy()));
  b.add_neutral_id(idOf(g->neutral()));
  b.add_game_type(g->getGameType().getID());
  b.add_latency_frames(g->getLatencyFrames());
  b.add_random_seed(seed);
  b.add_is_replay(g->isReplay());
  b.add_unit_types(unitTypesOff);
  b.add_weapon_types(weaponTypesOff);
  b.add_upgrade_types(upgradeTypesOff);
  b.add_tech_types(techTypesOff);
  b.add_frame_skip(frameSkip);
  auto gs = b.Finish();

  auto env = bw::CreateEnvelope(fbb, bw::Message::GameStart, gs.Union());
  fbb.Finish(env, bw::EnvelopeIdentifier());
}

// ---------------------------------------------------------------------------

bw::UnitState Serializer::unitState(Unit u, const Playerset& players) {
  using F = bw::UnitFlag;
  uint32_t flags = 0;
  flags |= bit(u->exists(), F::Exists);
  flags |= bit(u->isCompleted(), F::Completed);
  flags |= bit(u->isIdle(), F::Idle);
  flags |= bit(u->isMoving(), F::Moving);
  flags |= bit(u->isAttacking(), F::Attacking);
  flags |= bit(u->isAttackFrame(), F::AttackFrame);
  flags |= bit(u->isStartingAttack(), F::StartingAttack);
  flags |= bit(u->isGatheringMinerals() || u->isGatheringGas(), F::Gathering);
  flags |= bit(u->isBeingGathered(), F::BeingGathered);
  flags |= bit(u->isConstructing(), F::Constructing);
  flags |= bit(u->isTraining(), F::Training);
  flags |= bit(u->isMorphing(), F::Morphing);
  flags |= bit(u->isBurrowed(), F::Burrowed);
  flags |= bit(u->isCloaked(), F::Cloaked);
  flags |= bit(u->isDetected(), F::Detected);
  flags |= bit(u->isLifted(), F::Lifted);
  flags |= bit(u->isSieged(), F::Sieged);
  flags |= bit(u->isAccelerating(), F::Accelerating);
  flags |= bit(u->isBraking(), F::Braking);
  flags |= bit(u->isBlind(), F::Blind);
  flags |= bit(u->isHallucination(), F::Hallucination);
  flags |= bit(u->isInterruptible(), F::Interruptible);
  flags |= bit(u->isInvincible(), F::Invincible);
  flags |= bit(u->isParasited(), F::Parasited);
  flags |= bit(u->isSelected(), F::Selected);
  flags |= bit(u->isStuck(), F::Stuck);
  flags |= bit(u->isUnderStorm(), F::UnderStorm);
  flags |= bit(u->isUnderDarkSwarm(), F::UnderDarkSwarm);
  flags |= bit(u->isUnderDisruptionWeb(), F::UnderDWeb);
  flags |= bit(u->isPowered(), F::Powered);
  flags |= bit(u->hasNuke(), F::HasNuke);
  flags |= bit(u->isUnderAttack(), F::RecentlyAttacked);

  uint32_t vis = 0;
  for (Player p : players) {
    int pid = p->getID();
    if (pid >= 0 && pid < 32 && u->isVisible(p)) vis |= (1u << pid);
  }

  const int id = u->getID();
  const int hp = u->getHitPoints();
  int lastHp = hp;
  auto it = lastHp_.find(id);
  if (it != lastHp_.end()) {
    lastHp = it->second;
    it->second = hp;
  } else {
    lastHp_.emplace(id, hp);
  }

  const UnitType::list queue = u->getTrainingQueue();
  int q[5] = {-1, -1, -1, -1, -1};
  for (size_t i = 0; i < queue.size() && i < 5; ++i) q[i] = queue[i].getID();

  Position pos = u->getPosition();
  Position otp = u->getOrderTargetPosition();
  Position tp = u->getTargetPosition();
  Position rp = u->getRallyPosition();
  int carry = u->isCarryingGas() ? 1 : (u->isCarryingMinerals() ? 2 : 0);

  return bw::UnitState(
      id, u->getType().getID(), idOf(u->getPlayer()), pos.x, pos.y,
      hp, u->getShields(), u->getEnergy(), u->getResources(), u->getResourceGroup(),
      flags, vis,
      u->getOrder().getID(), idOf(u->getOrderTarget()), otp.x, otp.y, u->getSecondaryOrder().getID(),
      idOf(u->getTarget()), tp.x, tp.y,
      u->getBuildType().getID(), idOf(u->getBuildUnit()), u->getRemainingBuildTime(), u->getRemainingTrainTime(),
      int(queue.size()), q[0], q[1], q[2], q[3], q[4],
      u->getTech().getID(), u->getUpgrade().getID(), u->getRemainingResearchTime(), u->getRemainingUpgradeTime(),
      u->getGroundWeaponCooldown(), u->getAirWeaponCooldown(), u->getSpellCooldown(),
      float(u->getAngle()), float(u->getVelocityX()), float(u->getVelocityY()),
      u->getKillCount(), u->getAcidSporeCount(), u->getInterceptorCount(), u->getScarabCount(), u->getSpiderMineCount(),
      u->getDefenseMatrixPoints(), u->getDefenseMatrixTimer(), u->getEnsnareTimer(), u->getIrradiateTimer(),
      u->getLockdownTimer(), u->getMaelstromTimer(), u->getOrderTimer(), u->getPlagueTimer(), u->getRemoveTimer(),
      u->getStasisTimer(), u->getStimTimer(),
      idOf(u->getAddon()), idOf(u->getTransport()), idOf(u->getCarrier()), idOf(u->getHatchery()),
      idOf(u->getNydusExit()), idOf(u->getPowerUp()),
      rp.x, rp.y, idOf(u->getRallyUnit()),
      carry, idOf(u->getLastAttackingPlayer()), lastHp, u->getReplayID());
}

fb::Offset<bw::PlayerState> Serializer::playerState(fb::FlatBufferBuilder& fbb, Player p, bool full) {
  fb::Offset<fb::Vector<int32_t>> allOff, completedOff, deadOff, killedOff, upgradeOff;
  fb::Offset<fb::Vector<uint8_t>> researchedOff, researchingOff, upgradingOff;

  if (full) {
    const int nU = UnitTypes::Enum::MAX, nUp = UpgradeTypes::Enum::MAX, nT = TechTypes::Enum::MAX;
    std::vector<int32_t> all(nU), completed(nU), dead(nU), killed(nU), upg(nUp);
    std::vector<uint8_t> researched(nT), researching(nT), upgrading(nUp);
    for (int i = 0; i < nU; ++i) {
      UnitType t(i);
      all[i] = p->allUnitCount(t);
      completed[i] = p->completedUnitCount(t);
      dead[i] = p->deadUnitCount(t);
      killed[i] = p->killedUnitCount(t);
    }
    for (int i = 0; i < nUp; ++i) {
      UpgradeType t(i);
      upg[i] = p->getUpgradeLevel(t);
      upgrading[i] = p->isUpgrading(t) ? 1 : 0;
    }
    for (int i = 0; i < nT; ++i) {
      TechType t(i);
      researched[i] = p->hasResearched(t) ? 1 : 0;
      researching[i] = p->isResearching(t) ? 1 : 0;
    }
    allOff = fbb.CreateVector(all);
    completedOff = fbb.CreateVector(completed);
    deadOff = fbb.CreateVector(dead);
    killedOff = fbb.CreateVector(killed);
    upgradeOff = fbb.CreateVector(upg);
    researchedOff = fbb.CreateVector(researched);
    researchingOff = fbb.CreateVector(researching);
    upgradingOff = fbb.CreateVector(upgrading);
  }

  int32_t usedByRace[3] = {p->supplyUsed(Races::Zerg), p->supplyUsed(Races::Terran), p->supplyUsed(Races::Protoss)};
  int32_t totalByRace[3] = {p->supplyTotal(Races::Zerg), p->supplyTotal(Races::Terran), p->supplyTotal(Races::Protoss)};
  auto usedOff = fbb.CreateVector(usedByRace, 3);
  auto totalOff = fbb.CreateVector(totalByRace, 3);

  bw::PlayerStateBuilder b(fbb);
  b.add_id(p->getID());
  b.add_minerals(p->minerals());
  b.add_gas(p->gas());
  b.add_gathered_minerals(p->gatheredMinerals());
  b.add_gathered_gas(p->gatheredGas());
  b.add_supply_used(p->supplyUsed());
  b.add_supply_total(p->supplyTotal());
  b.add_supply_used_by_race(usedOff);
  b.add_supply_total_by_race(totalOff);
  b.add_is_victorious(p->isVictorious());
  b.add_is_defeated(p->isDefeated());
  b.add_left_game(p->leftGame());
  if (full) {
    b.add_all_unit_count(allOff);
    b.add_completed_unit_count(completedOff);
    b.add_dead_unit_count(deadOff);
    b.add_killed_unit_count(killedOff);
    b.add_upgrade_level(upgradeOff);
    b.add_has_researched(researchedOff);
    b.add_is_researching(researchingOff);
    b.add_is_upgrading(upgradingOff);
  }
  b.add_unit_score(p->getUnitScore());
  b.add_kill_score(p->getKillScore());
  b.add_building_score(p->getBuildingScore());
  b.add_razing_score(p->getRazingScore());
  b.add_custom_score(p->getCustomScore());
  return b.Finish();
}

void Serializer::buildFrame(fb::FlatBufferBuilder& fbb, const std::vector<Event>& events,
                            const SerializeOptions& opt, int serializeUs, int lastRoundtripUs) {
  Game* g = BroodwarPtr;
  const Playerset& players = g->getPlayers();
  Player self = g->self();

  // --- units ---------------------------------------------------------------
  std::vector<bw::UnitState> units;
  units.reserve(g->getAllUnits().size());
  for (Unit u : g->getAllUnits()) units.push_back(unitState(u, players));
  auto unitsOff = fbb.CreateVectorOfStructs(units);

  // --- players -------------------------------------------------------------
  fb::Offset<fb::Vector<fb::Offset<bw::PlayerState>>> playersOff;
  if (opt.includePlayers) {
    std::vector<fb::Offset<bw::PlayerState>> ps;
    for (Player p : players) {
      if (p->isNeutral()) continue;
      ps.push_back(playerState(fbb, p, p == self));
    }
    playersOff = fbb.CreateVector(ps);
  }

  // --- bullets -------------------------------------------------------------
  fb::Offset<fb::Vector<const bw::BulletState*>> bulletsOff;
  if (opt.includeBullets) {
    std::vector<bw::BulletState> bullets;
    for (Bullet bl : g->getBullets()) {
      Position p = bl->getPosition(), tp = bl->getTargetPosition();
      int ev = (bl->exists() ? 1 : 0) | (bl->isVisible(self) ? 2 : 0);
      bullets.emplace_back(bl->getID(), bl->getType().getID(), idOf(bl->getPlayer()), idOf(bl->getSource()),
                           idOf(bl->getTarget()), p.x, p.y, tp.x, tp.y, bl->getRemoveTimer(), float(bl->getAngle()),
                           float(bl->getVelocityX()), float(bl->getVelocityY()), ev);
    }
    bulletsOff = fbb.CreateVectorOfStructs(bullets);
  }

  // --- events --------------------------------------------------------------
  std::vector<fb::Offset<bw::Event>> evs;
  for (const Event& e : events) {
    fb::Offset<fb::String> text;
    if (!e.getText().empty()) text = fbb.CreateString(e.getText());
    bw::Pos pos(e.getPosition().x, e.getPosition().y);
    bw::EventBuilder b(fbb);
    b.add_type(e.getType());
    b.add_unit(idOf(e.getUnit()));
    b.add_player(idOf(e.getPlayer()));
    b.add_is_winner(e.isWinner());
    if (!text.IsNull()) b.add_text(text);
    b.add_position(&pos);
    evs.push_back(b.Finish());
  }
  auto evsOff = fbb.CreateVector(evs);

  // --- tiles ---------------------------------------------------------------
  fb::Offset<fb::Vector<uint8_t>> tilesOff;
  if (opt.includeTiles) {
    const int w = g->mapWidth(), h = g->mapHeight();
    std::vector<uint8_t> tiles(size_t(w) * h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        uint8_t t = 0;
        if (g->isVisible(x, y)) t |= uint8_t(bw::TileFlag::Visible);
        if (g->isExplored(x, y)) t |= uint8_t(bw::TileFlag::Explored);
        if (g->hasCreep(x, y)) t |= uint8_t(bw::TileFlag::Creep);
        tiles[size_t(y) * w + x] = t;
      }
    tilesOff = fbb.CreateVector(tiles);
  }

  std::vector<bw::Pos> nukes;
  for (const Position& p : g->getNukeDots()) nukes.emplace_back(p.x, p.y);
  auto nukesOff = fbb.CreateVectorOfStructs(nukes);

  bw::FrameBuilder b(fbb);
  b.add_frame_count(g->getFrameCount());
  b.add_elapsed_time(g->elapsedTime());
  b.add_fps(g->getFPS());
  b.add_average_fps(float(g->getAverageFPS()));
  b.add_latency_frames(g->getLatencyFrames());
  b.add_remaining_latency_frames(g->getRemainingLatencyFrames());
  b.add_is_paused(g->isPaused());
#ifdef SHIM_OPENBW
  b.add_countdown_timer(0);  // OpenBW's fork throws "countdownTimer?" (unimplemented)
#else
  b.add_countdown_timer(g->countdownTimer());
#endif
  b.add_self_id(idOf(self));
  if (!playersOff.IsNull()) b.add_players(playersOff);
  b.add_units(unitsOff);
  if (!bulletsOff.IsNull()) b.add_bullets(bulletsOff);
  b.add_events(evsOff);
  if (!tilesOff.IsNull()) b.add_tiles(tilesOff);
  b.add_nuke_dots(nukesOff);
  b.add_serialize_us(serializeUs);
  b.add_last_roundtrip_us(lastRoundtripUs);
  auto frame = b.Finish();

  auto env = bw::CreateEnvelope(fbb, bw::Message::Frame, frame.Union());
  fbb.Finish(env, bw::EnvelopeIdentifier());
}

// ---------------------------------------------------------------------------

void Serializer::buildGameEnd(fb::FlatBufferBuilder& fbb, bool isWinner) {
  Game* g = BroodwarPtr;
  std::vector<fb::Offset<bw::PlayerInfo>> players;
  for (Player p : g->getPlayers()) players.push_back(playerInfo(fbb, p));
  auto playersOff = fbb.CreateVector(players);
  bw::GameEndBuilder b(fbb);
  b.add_frame_count(g->getFrameCount());
  b.add_is_winner(isWinner);
  b.add_self_id(idOf(g->self()));
  b.add_players(playersOff);
  auto ge = b.Finish();
  auto env = bw::CreateEnvelope(fbb, bw::Message::GameEnd, ge.Union());
  fbb.Finish(env, bw::EnvelopeIdentifier());
}

}  // namespace shim
