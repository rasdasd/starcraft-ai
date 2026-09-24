"""Game loop: drives a `Bot` against a shim connection."""
from __future__ import annotations

import logging
import time
from typing import Optional

from .apm import ApmMeter
from .bot import Bot
from .client import ShimClient
from .commands import Actions
from .protocol import Disconnected

log = logging.getLogger("bwbot.runner")


def run(bot: Bot, host: str = "127.0.0.1", port: int = 8765, games: Optional[int] = None,
        connect_timeout: Optional[float] = None, reconnect: bool = True, max_frames: Optional[int] = None,
        apm_hud: Optional[tuple[int, int]] = (10, 300)) -> None:
    """Play games until `games` matches are done (None = forever) or the shim goes away.

    Reconnects to the shim if it disappears (e.g. StarCraft restarted) when `reconnect` is True.
    `max_frames` makes the bot leave the game after that many frames (useful on OpenBW, which has
    no built-in opponent so games never end by themselves).
    `apm_hud` is the screen position of the runner's APM line (None = don't draw it).
    """
    bot.apm = ApmMeter()
    played = 0
    while games is None or played < games:
        client = ShimClient(host, port)
        try:
            client.connect(timeout=connect_timeout)
            while games is None or played < games:
                _play_one(bot, client, max_frames, apm_hud)
                played += 1
        except Disconnected:
            log.warning("shim disconnected")
            if not reconnect:
                return
            time.sleep(1.0)
        except KeyboardInterrupt:
            log.info("interrupted")
            return
        finally:
            client.close()


def _play_one(bot: Bot, client: ShimClient, max_frames: Optional[int] = None,
              apm_hud: Optional[tuple[int, int]] = None) -> None:
    game = client.wait_for_game(bot.config)
    bot.game = game
    apm = bot.apm
    apm.reset()
    bot.on_start(game)

    act = Actions()
    frames = 0
    t0 = time.perf_counter()
    think_total = 0.0
    think_max = 0.0
    leaving = False
    while True:
        obs = client.next_frame()
        if obs is None:
            break
        act.clear()
        act.frame_count = obs.frame_count
        t1 = time.perf_counter()
        try:
            bot.on_frame(obs, act)
        except Exception:  # keep the game alive; a bot bug should not stall StarCraft
            log.exception("on_frame raised at frame %d", obs.frame_count)
        budget = getattr(bot, "apm_budget", None)
        allowed = apm.commands_allowed(obs.frame_count, budget)
        if allowed is not None and len(act.unit_cmds) > allowed:
            act.unit_cmds = act.unit_cmds[:allowed]
        if max_frames is not None and obs.frame_count >= max_frames and not leaving:
            log.info("max_frames=%d reached; leaving game", max_frames)
            act.leave_game()
            leaving = True
        dt = time.perf_counter() - t1
        think_total += dt
        think_max = max(think_max, dt)
        apm.record(obs.frame_count, len(act.unit_cmds))
        if apm_hud is not None:
            cap = f" / {budget:.0f}" if budget else ""
            act.draw_text_screen(apm_hud[0], apm_hud[1],
                                 f"APM {apm.current:.0f}{cap}  (avg {apm.average:.0f}, game {obs.game_apm}, "
                                 f"{apm.last} cmds this frame)")
        client.send_commands(act.to_bytes())
        frames += 1
        if frames % 500 == 0:
            wall = time.perf_counter() - t0
            log.info("frame %d%s | %d decisions | think avg %.2fms max %.2fms | shim rtt %.2fms ser %.2fms | "
                     "%.1f dec/s | APM %.0f (avg %.0f, game %d)",
                     obs.frame_count, " (paused)" if obs.is_paused else "", frames, 1000 * think_total / frames,
                     1000 * think_max, obs.last_roundtrip_us / 1000, obs.serialize_us / 1000, frames / max(wall, 1e-6),
                     apm.current, apm.average, obs.game_apm)
            think_max = 0.0

    won = client.last_result
    log.info("game over: %s after %d frames (%d decisions, %d actions, avg APM %.0f)", "WIN" if won else "LOSS",
             client.last_end_frame, frames, apm.total, apm.average)
    try:
        bot.on_end(won)
    except Exception:
        log.exception("on_end raised")
