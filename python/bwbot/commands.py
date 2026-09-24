"""Command builder: accumulates unit/game/draw commands for one frame and serializes them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import flatbuffers
import numpy as np

from .enums import Color, TextSize, UnitCommandType
from .generated import bw

UnitRef = Union[int, np.void, np.integer]


def _uid(u: UnitRef) -> int:
    """Accept a unit id or a UNIT_DTYPE row."""
    if isinstance(u, np.void):
        return int(u["id"])
    return int(u)


@dataclass
class _UnitCmd:
    unit: int
    type: int
    target: int = -1
    x: int = 0
    y: int = 0
    extra: int = 0
    queued: bool = False


@dataclass
class _GameCmd:
    type: int
    value: int = 0
    x: int = 0
    y: int = 0
    text: Optional[str] = None


@dataclass
class _Draw:
    shape: int
    ctype: int = 2
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    x3: int = 0
    y3: int = 0
    radius: int = 0
    color: int = int(Color.Green)
    is_solid: bool = False
    text: Optional[str] = None
    text_size: int = int(TextSize.Default)


@dataclass
class _PlaceQuery:
    id: int
    kind: int
    unit_type: int
    tile_x: int
    tile_y: int
    max_range: int = 64


@dataclass
class Actions:
    """Per-frame command sink handed to `Bot.on_frame`.

    Positions are pixels unless the BWAPI command takes tile positions (build, land, place_cop),
    which take tile coordinates. `queued=True` shift-queues where BWAPI supports it.
    """

    frame_count: int = 0
    unit_cmds: list[_UnitCmd] = field(default_factory=list)
    game_cmds: list[_GameCmd] = field(default_factory=list)
    draws: list[_Draw] = field(default_factory=list)
    place_qs: list[_PlaceQuery] = field(default_factory=list)

    # ------------------------------------------------------------------ unit
    def command(self, unit: UnitRef, ctype: int, target: Optional[UnitRef] = None, x: int = 0, y: int = 0,
                extra: int = 0, queued: bool = False) -> None:
        self.unit_cmds.append(_UnitCmd(_uid(unit), int(ctype), _uid(target) if target is not None else -1,
                                       int(x), int(y), int(extra), bool(queued)))

    def move(self, unit: UnitRef, x: int, y: int, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Move, x=x, y=y, queued=queued)

    def attack_move(self, unit: UnitRef, x: int, y: int, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Attack_Move, x=x, y=y, queued=queued)

    def attack(self, unit: UnitRef, target: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Attack_Unit, target=target, queued=queued)

    def patrol(self, unit: UnitRef, x: int, y: int, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Patrol, x=x, y=y, queued=queued)

    def hold(self, unit: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Hold_Position, queued=queued)

    def stop(self, unit: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Stop, queued=queued)

    def follow(self, unit: UnitRef, target: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Follow, target=target, queued=queued)

    def gather(self, unit: UnitRef, target: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Gather, target=target, queued=queued)

    def return_cargo(self, unit: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Return_Cargo, queued=queued)

    def repair(self, unit: UnitRef, target: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Repair, target=target, queued=queued)

    def right_click(self, unit: UnitRef, target: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Right_Click_Unit, target=target, queued=queued)

    def right_click_pos(self, unit: UnitRef, x: int, y: int, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Right_Click_Position, x=x, y=y, queued=queued)

    def train(self, unit: UnitRef, unit_type: int) -> None:
        self.command(unit, UnitCommandType.Train, extra=int(unit_type))

    def build(self, unit: UnitRef, unit_type: int, tile_x: int, tile_y: int) -> None:
        """Build `unit_type` with top-left at tile (tile_x, tile_y)."""
        self.command(unit, UnitCommandType.Build, x=tile_x, y=tile_y, extra=int(unit_type))

    def build_addon(self, unit: UnitRef, unit_type: int) -> None:
        self.command(unit, UnitCommandType.Build_Addon, extra=int(unit_type))

    def morph(self, unit: UnitRef, unit_type: int) -> None:
        self.command(unit, UnitCommandType.Morph, extra=int(unit_type))

    def research(self, unit: UnitRef, tech: int) -> None:
        self.command(unit, UnitCommandType.Research, extra=int(tech))

    def upgrade(self, unit: UnitRef, upgrade: int) -> None:
        self.command(unit, UnitCommandType.Upgrade, extra=int(upgrade))

    def set_rally(self, unit: UnitRef, x: int, y: int) -> None:
        self.command(unit, UnitCommandType.Set_Rally_Position, x=x, y=y)

    def set_rally_unit(self, unit: UnitRef, target: UnitRef) -> None:
        self.command(unit, UnitCommandType.Set_Rally_Unit, target=target)

    def burrow(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Burrow)

    def unburrow(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Unburrow)

    def cloak(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Cloak)

    def decloak(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Decloak)

    def siege(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Siege)

    def unsiege(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Unsiege)

    def lift(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Lift)

    def land(self, unit: UnitRef, tile_x: int, tile_y: int) -> None:
        self.command(unit, UnitCommandType.Land, x=tile_x, y=tile_y)

    def load(self, unit: UnitRef, target: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Load, target=target, queued=queued)

    def unload(self, unit: UnitRef, target: UnitRef) -> None:
        self.command(unit, UnitCommandType.Unload, target=target)

    def unload_all(self, unit: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Unload_All, queued=queued)

    def unload_all_pos(self, unit: UnitRef, x: int, y: int, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Unload_All_Position, x=x, y=y, queued=queued)

    def halt_construction(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Halt_Construction)

    def cancel_construction(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Cancel_Construction)

    def cancel_addon(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Cancel_Addon)

    def cancel_train(self, unit: UnitRef, slot: int = -2) -> None:
        if slot == -2:
            self.command(unit, UnitCommandType.Cancel_Train)
        else:
            self.command(unit, UnitCommandType.Cancel_Train_Slot, extra=slot)

    def cancel_morph(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Cancel_Morph)

    def cancel_research(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Cancel_Research)

    def cancel_upgrade(self, unit: UnitRef) -> None:
        self.command(unit, UnitCommandType.Cancel_Upgrade)

    def use_tech(self, unit: UnitRef, tech: int) -> None:
        self.command(unit, UnitCommandType.Use_Tech, extra=int(tech))

    def use_tech_pos(self, unit: UnitRef, tech: int, x: int, y: int, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Use_Tech_Position, x=x, y=y, extra=int(tech), queued=queued)

    def use_tech_unit(self, unit: UnitRef, tech: int, target: UnitRef, queued: bool = False) -> None:
        self.command(unit, UnitCommandType.Use_Tech_Unit, target=target, extra=int(tech), queued=queued)

    def place_cop(self, unit: UnitRef, tile_x: int, tile_y: int) -> None:
        self.command(unit, UnitCommandType.Place_COP, x=tile_x, y=tile_y)

    def can_build_here(self, req_id: int, unit_type: int, tile_x: int, tile_y: int) -> None:
        """Ask the shim to run Broodwar->canBuildHere. Answer is on the next Frame."""
        self.place_qs.append(_PlaceQuery(int(req_id), 0, int(unit_type), int(tile_x), int(tile_y)))

    def get_build_location(self, req_id: int, unit_type: int, near_tile: tuple[int, int],
                           max_range: int = 64) -> None:
        """Ask the shim to run Broodwar->getBuildLocation. Answer is on the next Frame."""
        self.place_qs.append(_PlaceQuery(int(req_id), 1, int(unit_type), int(near_tile[0]),
                                         int(near_tile[1]), int(max_range)))

    # ------------------------------------------------------------------ game
    def set_local_speed(self, ms_per_frame: int) -> None:
        """0 = fastest, 42 = normal 'fastest' human speed, -1 = default."""
        self.game_cmds.append(_GameCmd(bw.GameCommandType.SetLocalSpeed, ms_per_frame))

    def set_frame_skip(self, n: int) -> None:
        """Render only every n-th frame (game GUI), speeds up local games."""
        self.game_cmds.append(_GameCmd(bw.GameCommandType.SetFrameSkip, n))

    def set_gui(self, enabled: bool) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.SetGUI, int(enabled)))

    def send_text(self, text: str) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.SendText, text=text))

    def printf(self, text: str) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.Printf, text=text))

    def leave_game(self) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.LeaveGame))

    def restart_game(self) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.RestartGame))

    def pause_game(self) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.PauseGame))

    def resume_game(self) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.ResumeGame))

    def set_lat_com(self, enabled: bool) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.SetLatCom, int(enabled)))

    def set_command_optimization(self, level: int) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.SetCommandOptimizationLevel, level))

    def set_reveal_all(self, reveal: bool = True) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.SetRevealAll, int(reveal)))

    def ping_minimap(self, x: int, y: int) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.PingMinimap, x=x, y=y))

    def set_screen_position(self, x: int, y: int) -> None:
        self.game_cmds.append(_GameCmd(bw.GameCommandType.SetScreenPosition, x=x, y=y))

    # ------------------------------------------------------------------ draw (ctype: 1 screen, 2 map)
    def draw_text(self, x: int, y: int, text: str, ctype: int = 2, size: int = int(TextSize.Default)) -> None:
        self.draws.append(_Draw(bw.DrawShape.Text, ctype, x, y, text=text, text_size=size))

    def draw_text_screen(self, x: int, y: int, text: str, size: int = int(TextSize.Default)) -> None:
        self.draw_text(x, y, text, ctype=1, size=size)

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, color: int = Color.Green, ctype: int = 2) -> None:
        self.draws.append(_Draw(bw.DrawShape.Line, ctype, x1, y1, x2, y2, color=int(color)))

    def draw_box(self, left: int, top: int, right: int, bottom: int, color: int = Color.Green, solid: bool = False,
                 ctype: int = 2) -> None:
        self.draws.append(_Draw(bw.DrawShape.Box, ctype, left, top, right, bottom, color=int(color), is_solid=solid))

    def draw_circle(self, x: int, y: int, radius: int, color: int = Color.Green, solid: bool = False,
                    ctype: int = 2) -> None:
        self.draws.append(_Draw(bw.DrawShape.Circle, ctype, x, y, radius=radius, color=int(color), is_solid=solid))

    def draw_dot(self, x: int, y: int, color: int = Color.Green, ctype: int = 2) -> None:
        self.draws.append(_Draw(bw.DrawShape.Dot, ctype, x, y, color=int(color)))

    def draw_tile_box(self, tile_x: int, tile_y: int, tiles_w: int = 1, tiles_h: int = 1, color: int = Color.Green,
                      solid: bool = False) -> None:
        self.draw_box(tile_x * 32, tile_y * 32, (tile_x + tiles_w) * 32, (tile_y + tiles_h) * 32, color, solid)

    # ------------------------------------------------------------------ serialize
    def __len__(self) -> int:
        return len(self.unit_cmds) + len(self.game_cmds) + len(self.draws) + len(self.place_qs)

    def clear(self) -> None:
        self.unit_cmds.clear()
        self.game_cmds.clear()
        self.draws.clear()
        self.place_qs.clear()

    def to_bytes(self) -> bytearray:
        b = flatbuffers.Builder(256 + 48 * len(self))

        ucs = []
        for c in self.unit_cmds:
            bw.UnitCommandStart(b)
            bw.UnitCommandAddUnit(b, c.unit)
            bw.UnitCommandAddType(b, c.type)
            bw.UnitCommandAddTarget(b, c.target)
            bw.UnitCommandAddX(b, c.x)
            bw.UnitCommandAddY(b, c.y)
            bw.UnitCommandAddExtra(b, c.extra)
            bw.UnitCommandAddQueued(b, c.queued)
            ucs.append(bw.UnitCommandEnd(b))
        bw.CommandsStartUnitCommandsVector(b, len(ucs))
        for off in reversed(ucs):
            b.PrependUOffsetTRelative(off)
        ucs_vec = b.EndVector()

        gcs = []
        for c in self.game_cmds:
            text = b.CreateString(c.text) if c.text is not None else None
            bw.GameCommandStart(b)
            bw.GameCommandAddType(b, c.type)
            bw.GameCommandAddValue(b, c.value)
            bw.GameCommandAddX(b, c.x)
            bw.GameCommandAddY(b, c.y)
            if text is not None:
                bw.GameCommandAddText(b, text)
            gcs.append(bw.GameCommandEnd(b))
        bw.CommandsStartGameCommandsVector(b, len(gcs))
        for off in reversed(gcs):
            b.PrependUOffsetTRelative(off)
        gcs_vec = b.EndVector()

        ds = []
        for d in self.draws:
            text = b.CreateString(d.text) if d.text is not None else None
            bw.DrawCommandStart(b)
            bw.DrawCommandAddShape(b, d.shape)
            bw.DrawCommandAddCtype(b, d.ctype)
            bw.DrawCommandAddX1(b, d.x1)
            bw.DrawCommandAddY1(b, d.y1)
            bw.DrawCommandAddX2(b, d.x2)
            bw.DrawCommandAddY2(b, d.y2)
            bw.DrawCommandAddX3(b, d.x3)
            bw.DrawCommandAddY3(b, d.y3)
            bw.DrawCommandAddRadius(b, d.radius)
            bw.DrawCommandAddColor(b, d.color)
            bw.DrawCommandAddIsSolid(b, d.is_solid)
            if text is not None:
                bw.DrawCommandAddText(b, text)
            bw.DrawCommandAddTextSize(b, d.text_size)
            ds.append(bw.DrawCommandEnd(b))
        bw.CommandsStartDrawsVector(b, len(ds))
        for off in reversed(ds):
            b.PrependUOffsetTRelative(off)
        ds_vec = b.EndVector()

        pqs = []
        for q in self.place_qs:
            bw.PlacementQueryStart(b)
            bw.PlacementQueryAddId(b, q.id)
            bw.PlacementQueryAddKind(b, q.kind)
            bw.PlacementQueryAddUnitType(b, q.unit_type)
            bw.PlacementQueryAddTileX(b, q.tile_x)
            bw.PlacementQueryAddTileY(b, q.tile_y)
            bw.PlacementQueryAddMaxRange(b, q.max_range)
            pqs.append(bw.PlacementQueryEnd(b))
        bw.CommandsStartPlacementQueriesVector(b, len(pqs))
        for off in reversed(pqs):
            b.PrependUOffsetTRelative(off)
        pq_vec = b.EndVector()

        bw.CommandsStart(b)
        bw.CommandsAddFrameCount(b, self.frame_count)
        bw.CommandsAddUnitCommands(b, ucs_vec)
        bw.CommandsAddGameCommands(b, gcs_vec)
        bw.CommandsAddDraws(b, ds_vec)
        bw.CommandsAddPlacementQueries(b, pq_vec)
        cmds = bw.CommandsEnd(b)

        bw.EnvelopeStart(b)
        bw.EnvelopeAddMsgType(b, bw.Message.Commands)
        bw.EnvelopeAddMsg(b, cmds)
        env = bw.EnvelopeEnd(b)
        b.Finish(env, b"BWB1")
        return b.Output()
