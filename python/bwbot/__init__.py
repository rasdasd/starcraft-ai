"""bwbot - Brood War bot framework.

A 64-bit Python process that plays StarCraft: Brood War through the C++ BWAPI shim
(see ../shim). Game state arrives as FlatBuffers `Frame` messages exposed here as numpy
arrays; the bot answers with `Commands`.

Typical use::

    from bwbot import Bot, run

    class MyBot(Bot):
        def on_frame(self, obs, act):
            for scv in obs.my_units_of_type(UnitType.Terran_SCV):
                ...

    run(MyBot())
"""

__version__ = "0.1.0"

from .bot import Bot, ClientConfig  # noqa: E402
from .commands import Actions  # noqa: E402
from .enums import (  # noqa: E402
    Color,
    EventType,
    Order,
    Race,
    TechType,
    UnitCommandType,
    UnitType,
    UpgradeType,
    WeaponType,
)
from .observation import GameInfo, Observation, UnitFlag, UNIT_DTYPE  # noqa: E402
from .runner import run  # noqa: E402

__all__ = [
    "Bot",
    "ClientConfig",
    "Actions",
    "GameInfo",
    "Observation",
    "UnitFlag",
    "UNIT_DTYPE",
    "run",
    "UnitType",
    "Order",
    "Race",
    "TechType",
    "UpgradeType",
    "WeaponType",
    "UnitCommandType",
    "EventType",
    "Color",
]
