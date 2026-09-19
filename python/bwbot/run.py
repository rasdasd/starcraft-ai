"""CLI: python -m bwbot.run examples.basic_terran:BasicTerran [--host H] [--port P] [--frame-skip N] ...

The bot spec is `module.path:ClassName`; if the class is omitted, the module must expose a
`Bot` subclass named `BOT` or exactly one Bot subclass.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import logging
import os
import sys

from .bot import Bot
from .runner import run


def load_bot(spec: str) -> Bot:
    mod_name, _, cls_name = spec.partition(":")
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(mod_name)
    if cls_name:
        cls = getattr(mod, cls_name)
    elif hasattr(mod, "BOT"):
        cls = getattr(mod, "BOT")
    else:
        cands = [c for _, c in inspect.getmembers(mod, inspect.isclass)
                 if issubclass(c, Bot) and c is not Bot and c.__module__ == mod.__name__]
        if len(cands) != 1:
            raise SystemExit(f"{mod_name}: expected exactly one Bot subclass, found {[c.__name__ for c in cands]}")
        cls = cands[0]
    return cls() if inspect.isclass(cls) else cls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bwbot", description="Run a bwbot Bot against the shim.")
    ap.add_argument("bot", help="module.path[:ClassName], e.g. examples.basic_terran:BasicTerran")
    ap.add_argument("--host", default=os.environ.get("BWBOT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("BWBOT_PORT", "8765")))
    ap.add_argument("--games", type=int, default=None, help="stop after N games (default: forever)")
    ap.add_argument("--frame-skip", type=int, default=None, help="override bot.config.frame_skip")
    ap.add_argument("--speed", type=int, default=None, help="override bot.config.local_speed (0 = fastest)")
    ap.add_argument("--no-gui", action="store_true", help="disable game rendering (OpenBW / speed runs)")
    ap.add_argument("--no-tiles", action="store_true", help="skip per-frame tile visibility map")
    ap.add_argument("--no-bullets", action="store_true")
    ap.add_argument("--cheat-map-info", action="store_true", help="enable CompleteMapInformation (training)")
    ap.add_argument("--connect-timeout", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="leave the game after N frames (OpenBW has no AI opponent, so games never end)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")

    bot = load_bot(args.bot)
    bot.config = cfg = dataclasses.replace(bot.config)  # don't mutate the class-level default
    if args.frame_skip is not None:
        cfg.frame_skip = args.frame_skip
    if args.speed is not None:
        cfg.local_speed = args.speed
    if args.no_gui:
        cfg.gui = False
    if args.no_tiles:
        cfg.include_tiles = False
    if args.no_bullets:
        cfg.include_bullets = False
    if args.cheat_map_info:
        cfg.complete_map_information = True

    run(bot, host=args.host, port=args.port, games=args.games, connect_timeout=args.connect_timeout,
        max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
