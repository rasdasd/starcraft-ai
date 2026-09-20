"""mybot - your bot lives here.

Run it against a visible game from the repo root:

    python run.py --bot mybot                  # Windows or WSL
    python run.py --bot mybot --speed 42       # human speed instead of max speed

or directly (shim + StarCraft already running):

    cd python && .venv\\Scripts\\python -m bwbot.run mybot

Layout - three layers so the decision-making part can be swapped without touching the rest:

    state.py    perceive(obs)  -> State        what the agent sees   (features)
    policy.py   Policy.decide(State) -> Intents what the agent wants  (deterministic now, ML later)
    bot.py      MyBot executes Intents          how it happens        (BWAPI commands via `act`)
    macro.py    building placement / builder selection helpers used by bot.py

`bwbot.run` looks for a module attribute named BOT, so this package is runnable as-is.
"""
from .bot import MyBot

BOT = MyBot

__all__ = ["MyBot", "BOT"]
