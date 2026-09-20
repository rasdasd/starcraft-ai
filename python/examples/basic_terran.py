"""Minimal Terran bot: saturate minerals, keep building SCVs and supply depots, build barracks,
train marines, attack-move once enough marines are gathered. Exists to validate the pipeline
end to end; not meant to win.

Run (after the shim and StarCraft are up):
    python -m bwbot.run examples.basic_terran
"""
from __future__ import annotations

import logging
import random

import numpy as np

from bwbot import Actions, Bot, ClientConfig, EventType, GameInfo, Observation, UnitFlag, UnitType
from bwbot.enums import Order
from bwbot.observation import UnitTypeFlag

log = logging.getLogger("basic_terran")

SCV = UnitType.Terran_SCV
CC = UnitType.Terran_Command_Center
DEPOT = UnitType.Terran_Supply_Depot
RAX = UnitType.Terran_Barracks
MARINE = UnitType.Terran_Marine
REFINERY = UnitType.Terran_Refinery


class BasicTerran(Bot):
    config = ClientConfig(frame_skip=2, local_speed=0, include_tiles=True, user_input=True)

    def on_start(self, game: GameInfo) -> None:
        self.attacking = False
        self.reserved_minerals = 0
        self.last_build_frame = -1000
        self.builder_id = -1
        self.enemy_start = None
        me = game.self_player
        others = [tuple(s) for s in game.start_locations.tolist() if tuple(s) != tuple(me.start_location)]
        if others:
            self.enemy_start = others[0]
        log.info("map %s, me=%s (%s) start=%s enemy_start_guess=%s", game.map_name, me.name,
                 game.self_race.name, me.start_location, self.enemy_start)

    def on_frame(self, obs: Observation, act: Actions) -> None:
        for e in obs.iter_events(EventType.UnitCreate, EventType.UnitComplete, EventType.UnitDestroy):
            u = obs.unit(e.unit)
            if u is not None and int(u["player"]) == obs.self_id:
                log.info("f%d %s %s #%d", obs.frame_count, e.type.name, obs.game.type_name(int(u["type"])), e.unit)
        mine = obs.my_units
        if len(mine) == 0:
            return
        completed = obs.completed(mine)
        ccs = completed[completed["type"] == CC]
        scvs = completed[completed["type"] == SCV]
        raxes = completed[completed["type"] == RAX]
        marines = completed[completed["type"] == MARINE]

        minerals = obs.minerals - self.reserved_minerals
        supply_left = obs.supply_total - obs.supply_used
        frame = obs.frame_count

        # --- workers: send idle SCVs to the nearest mineral patch ----------------
        fields = obs.minerals_fields
        for scv in obs.idle(scvs):
            if int(scv["id"]) == self.builder_id:
                continue
            patch = obs.nearest(fields, scv["x"], scv["y"])
            if patch is not None:
                act.gather(scv, patch)

        # --- production ------------------------------------------------------
        for cc in ccs:
            if cc["train_queue_count"] == 0 and minerals >= 50 and len(scvs) + obs.count(SCV) - len(scvs) < 22:
                act.train(cc, SCV)
                minerals -= 50
        for rax in raxes:
            if rax["train_queue_count"] == 0 and minerals >= 50 and supply_left >= 1:
                act.train(rax, MARINE)
                minerals -= 50

        # --- construction ----------------------------------------------------
        want_depot = supply_left <= 3 and obs.count(DEPOT) == obs.count(DEPOT, completed_only=True)
        want_rax = obs.count(RAX) < 2 and obs.count(DEPOT, completed_only=True) >= 1
        building = None
        if want_depot and minerals >= 100:
            building = DEPOT
        elif want_rax and minerals >= 150:
            building = RAX
        if building is not None and frame - self.last_build_frame > 24 * 6 and len(ccs) and len(scvs):
            cc = ccs[0]
            tile = self._find_build_tile(obs, building, int(cc["x"]) // 32, int(cc["y"]) // 32)
            if tile is not None:
                builder = self._pick_builder(obs, scvs, tile)
                if builder is not None:
                    act.build(builder, building, tile[0], tile[1])
                    act.draw_tile_box(tile[0], tile[1], int(obs.game.unit_types["tile_width"][building]),
                                      int(obs.game.unit_types["tile_height"][building]))
                    self.builder_id = int(builder["id"])
                    self.last_build_frame = frame

        # --- army ----------------------------------------------------------
        if len(marines) >= 20:
            self.attacking = True
        if self.attacking and len(marines) < 6:
            self.attacking = False
        if self.attacking and self.enemy_start is not None:
            tx, ty = self.enemy_start[0] * 32 + 64, self.enemy_start[1] * 32 + 48
            enemies = obs.enemy_units
            for m in marines:
                if m["flags"] & UnitFlag.Idle:
                    tgt = obs.nearest(enemies, m["x"], m["y"]) if len(enemies) else None
                    if tgt is not None:
                        act.attack_move(m, int(tgt["x"]), int(tgt["y"]))
                    else:
                        act.attack_move(m, tx, ty)
        elif len(raxes):
            rally = (int(raxes[0]["x"]) + 96, int(raxes[0]["y"]))
            for m in obs.idle(marines):
                if abs(int(m["x"]) - rally[0]) + abs(int(m["y"]) - rally[1]) > 160:
                    act.move(m, rally[0] + random.randint(-40, 40), rally[1] + random.randint(-40, 40))

        # --- HUD -----------------------------------------------------------
        act.draw_text_screen(10, 10, f"frame {frame}  scv {len(scvs)}  marines {len(marines)}  "
                                     f"supply {obs.supply_used}/{obs.supply_total}  attack={self.attacking}")
        act.draw_text_screen(10, 22, f"rtt {obs.last_roundtrip_us / 1000:.1f}ms  ser {obs.serialize_us / 1000:.1f}ms  "
                                     f"units {len(obs.units)}")

    # ------------------------------------------------------------------ helpers
    def _pick_builder(self, obs: Observation, scvs: np.ndarray, tile: tuple[int, int]):
        free = scvs[(scvs["flags"] & (UnitFlag.Constructing | UnitFlag.Gathering)) != UnitFlag.Constructing]
        free = free[(free["carry_resource_type"] == 0) & (free["order"] != Order.ConstructingBuilding)]
        if len(free) == 0:
            free = scvs
        return obs.nearest(free, tile[0] * 32, tile[1] * 32)

    def _find_build_tile(self, obs: Observation, building: int, cx: int, cy: int):
        """Spiral out from (cx, cy) for a buildable, unoccupied rectangle (very rough placement)."""
        g = obs.game
        tw = int(g.unit_types["tile_width"][building])
        th = int(g.unit_types["tile_height"][building])
        occupied = self._occupancy(obs)
        for r in range(3, 16):
            cands = []
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    cands.append((cx + dx, cy + dy))
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    cands.append((cx + dx, cy + dy))
            random.shuffle(cands)
            for tx, ty in cands:
                if tx < 1 or ty < 1 or tx + tw + 1 >= g.map_width or ty + th + 1 >= g.map_height:
                    continue
                # 1-tile margin so units can path around and addons fit for barracks
                sl = (slice(ty - 1, ty + th + 1), slice(tx - 1, tx + tw + 1 + (2 if building == RAX else 0)))
                if g.buildable[sl].all() and not occupied[sl].any():
                    return tx, ty
        return None

    def _occupancy(self, obs: Observation) -> np.ndarray:
        g = obs.game
        occ = np.zeros((g.map_height, g.map_width), dtype=bool)
        types = np.clip(obs.units["type"], 0, len(g.unit_types) - 1)
        is_bld = (g.unit_types["flags"][types] & (UnitTypeFlag.Building | UnitTypeFlag.ResourceContainer)) != 0
        for u, t in zip(obs.units[is_bld], types[is_bld]):
            tw, th = int(g.unit_types["tile_width"][t]), int(g.unit_types["tile_height"][t])
            tx = int(u["x"]) // 32 - tw // 2
            ty = int(u["y"]) // 32 - th // 2
            # keep minerals lines clear: pad resources by 3 tiles
            pad = 3 if (g.unit_types["flags"][t] & UnitTypeFlag.ResourceContainer) else 0
            occ[max(0, ty - pad):ty + th + pad, max(0, tx - pad):tx + tw + pad] = True
        return occ

    def on_end(self, is_winner: bool) -> None:
        log.info("game over: %s", "WIN" if is_winner else "LOSS")


BOT = BasicTerran
