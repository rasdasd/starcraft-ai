#include "CommandApplier.h"

#include <iostream>

namespace shim {

using namespace BWAPI;

namespace {

// Command types whose `extra` field is the shift-queue flag in BWAPI::UnitCommand.
bool extraIsShiftFlag(int type) {
  using T = UnitCommandTypes::Enum::Enum;
  switch (type) {
    case T::Attack_Move: case T::Attack_Unit: case T::Move: case T::Patrol: case T::Hold_Position:
    case T::Stop: case T::Follow: case T::Gather: case T::Return_Cargo: case T::Repair: case T::Load:
    case T::Unload_All: case T::Unload_All_Position: case T::Right_Click_Position: case T::Right_Click_Unit:
    case T::Use_Tech_Position: case T::Use_Tech_Unit:
      return true;
    default:
      return false;
  }
}

CoordinateType::Enum ctypeOf(int v) {
  switch (v) {
    case 1: return CoordinateType::Screen;
    case 3: return CoordinateType::Mouse;
    default: return CoordinateType::Map;
  }
}

}  // namespace

ApplyStats CommandApplier::apply(const bw::Commands& cmds) {
  ApplyStats st;
  if (auto* gcs = cmds.game_commands()) {
    for (const bw::GameCommand* c : *gcs) {
      try {
        applyGameCommand(*c);
      } catch (const std::exception& e) {
        // OpenBW's fork throws for a few unimplemented calls (e.g. replay-only ones).
        std::cerr << "[shim] game command " << bw::EnumNameGameCommandType(c->type()) << " failed: " << e.what()
                  << std::endl;
      }
      ++st.gameCommands;
    }
  }
  if (auto* ucs = cmds.unit_commands()) {
    for (const bw::UnitCommand* c : *ucs) {
      ++st.unitCommands;
      if (!applyUnitCommand(*c)) ++st.unitCommandsFailed;
    }
  }
  lastDraws_.clear();
  if (auto* ds = cmds.draws()) {
    lastDraws_.reserve(ds->size());
    for (const bw::DrawCommand* d : *ds) {
      bw::DrawCommandT t;
      d->UnPackTo(&t);
      applyDraw(t);
      lastDraws_.push_back(std::move(t));
      ++st.draws;
    }
  }
  return st;
}

void CommandApplier::redrawLast() {
  for (const auto& d : lastDraws_) applyDraw(d);
}

bool CommandApplier::applyUnitCommand(const bw::UnitCommand& c) {
  Unit unit = Broodwar->getUnit(c.unit());
  if (!unit) return false;
  Unit target = c.target() >= 0 ? Broodwar->getUnit(c.target()) : nullptr;
  int extra = c.extra();
  if (c.queued() && extraIsShiftFlag(c.type()) && extra == 0) extra = 1;
  if (c.type() < 0 || c.type() >= UnitCommandTypes::Enum::MAX) return false;
  UnitCommand cmd(unit, UnitCommandType(c.type()), target, c.x(), c.y(), extra);
  return unit->issueCommand(cmd);
}

void CommandApplier::applyGameCommand(const bw::GameCommand& c) {
  Game* g = BroodwarPtr;
  const char* text = c.text() ? c.text()->c_str() : "";
  switch (c.type()) {
    case bw::GameCommandType::SetLocalSpeed: g->setLocalSpeed(c.value()); break;
    case bw::GameCommandType::SetFrameSkip: g->setFrameSkip(c.value()); break;
    case bw::GameCommandType::SetGUI: g->setGUI(c.value() != 0); break;
    case bw::GameCommandType::SendText: g->sendText("%s", text); break;
    case bw::GameCommandType::Printf: g->printf("%s", text); break;
    case bw::GameCommandType::LeaveGame: g->leaveGame(); break;
    case bw::GameCommandType::RestartGame: g->restartGame(); break;
    case bw::GameCommandType::PauseGame: g->pauseGame(); break;
    case bw::GameCommandType::ResumeGame: g->resumeGame(); break;
    case bw::GameCommandType::EnableFlag: g->enableFlag(c.value()); break;
    case bw::GameCommandType::SetLatCom: g->setLatCom(c.value() != 0); break;
    case bw::GameCommandType::SetCommandOptimizationLevel: g->setCommandOptimizationLevel(c.value()); break;
    case bw::GameCommandType::SetRevealAll: g->setRevealAll(c.value() != 0); break;
    case bw::GameCommandType::SetVision: {
      if (Player p = g->getPlayer(c.value())) g->setVision(p, c.x() != 0);
      break;
    }
    case bw::GameCommandType::SetAlliance: {
      if (Player p = g->getPlayer(c.value())) g->setAlliance(p, c.x() != 0, c.y() != 0);
      break;
    }
    case bw::GameCommandType::PingMinimap: g->pingMinimap(c.x(), c.y()); break;
    case bw::GameCommandType::SetScreenPosition: g->setScreenPosition(c.x(), c.y()); break;
    default: break;
  }
}

void CommandApplier::applyDraw(const bw::DrawCommandT& d) {
  Game* g = BroodwarPtr;
  CoordinateType::Enum ct = ctypeOf(d.ctype);
  Color color(d.color);
  switch (d.shape) {
    case bw::DrawShape::Text: {
      g->setTextSize(static_cast<Text::Size::Enum>(d.text_size));
      g->drawText(ct, d.x1, d.y1, "%s", d.text.c_str());
      break;
    }
    case bw::DrawShape::Line: g->drawLine(ct, d.x1, d.y1, d.x2, d.y2, color); break;
    case bw::DrawShape::Box: g->drawBox(ct, d.x1, d.y1, d.x2, d.y2, color, d.is_solid); break;
    case bw::DrawShape::Circle: g->drawCircle(ct, d.x1, d.y1, d.radius, color, d.is_solid); break;
    case bw::DrawShape::Ellipse: g->drawEllipse(ct, d.x1, d.y1, d.x3, d.y3, color, d.is_solid); break;
    case bw::DrawShape::Dot: g->drawDot(ct, d.x1, d.y1, color); break;
    case bw::DrawShape::Triangle: g->drawTriangle(ct, d.x1, d.y1, d.x2, d.y2, d.x3, d.y3, color, d.is_solid); break;
    default: break;
  }
}

}  // namespace shim
