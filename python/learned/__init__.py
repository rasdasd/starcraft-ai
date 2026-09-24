"""learned - MyBot + LearnedPolicy (Goliath teacher, opponent model, optional .npz).

    python run.py --bot learned
    python -m mybot.train --logs logs --out models/policy.npz
"""
from __future__ import annotations

from goliath.policy import GoliathPolicy
from mybot.bot import MyBot
from mybot.learned import LearnedPolicy


class Learned(MyBot):
    def __init__(self) -> None:
        super().__init__(policy=LearnedPolicy(teacher=GoliathPolicy()))


BOT = Learned

__all__ = ["Learned", "BOT"]
