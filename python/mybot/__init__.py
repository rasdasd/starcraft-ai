"""mybot - your bot lives here.

Run it against a visible game from the repo root:

    python run.py --bot mybot                  # Windows or WSL
    python run.py --bot mybot --speed 42       # human speed instead of max speed

or directly (shim + StarCraft already running):

    cd python && .venv\\Scripts\\python -m bwbot.run mybot

Layout:

    state.py        perceive(obs) -> State
    policy.py       Policy.decide(State) -> Intents     (strategy; swap for ML later)
    opening.py      supply-gated build lists (data, not if-trees)
    production.py   queue + train / addon / upgrade
    buildings.py    construction state machine
    workers.py      mineral / gas / build / repair / scout jobs
    information.py  fog memory for enemy buildings
    scout.py        one SCV to the other start
    combat.py       defend / push / hunt-air (home includes the BWEM natural)
    opponent.py     race / timings / opening guess (the picture a policy may learn from)
    learned.py      teacher + opponent prior, or a fitted LinearPolicy
    logger.py / train.py   log decisions, fit models/policy.npz
    macro.py        reserved-tile placer
    bot.py          MyBot ticks managers each decision

`bwbot.run` looks for a module attribute named BOT, so this package is runnable as-is.
"""
from .bot import MyBot

BOT = MyBot

__all__ = ["MyBot", "BOT"]
