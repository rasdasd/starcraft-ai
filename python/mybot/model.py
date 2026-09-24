"""Numpy linear policy: features -> tactic + next building.

No torch at inference. `train.py` fits the same matrices with softmax SGD and writes
an `.npz` the frozen competition bot can load from `bwapi-data/read/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from bwbot import UnitType

from .features import FEATURE_DIM
from .tactics import Tactic

BUILD_VOCAB = (
    0,
    int(UnitType.Terran_Supply_Depot),
    int(UnitType.Terran_Barracks),
    int(UnitType.Terran_Refinery),
    int(UnitType.Terran_Factory),
    int(UnitType.Terran_Armory),
    int(UnitType.Terran_Bunker),
)

N_TACTIC = len(Tactic)
N_BUILD = len(BUILD_VOCAB)
BUILD_INDEX = {t: i for i, t in enumerate(BUILD_VOCAB)}


def encode_build(unit_type: Optional[int]) -> int:
    if unit_type is None:
        return 0
    return BUILD_INDEX.get(int(unit_type), 0)


class LinearPolicy:
    def __init__(self, tactic_w: np.ndarray, tactic_b: np.ndarray,
                 build_w: np.ndarray, build_b: np.ndarray) -> None:
        self.tactic_w = np.asarray(tactic_w, dtype=np.float32)
        self.tactic_b = np.asarray(tactic_b, dtype=np.float32)
        self.build_w = np.asarray(build_w, dtype=np.float32)
        self.build_b = np.asarray(build_b, dtype=np.float32)
        if self.tactic_w.shape != (N_TACTIC, FEATURE_DIM):
            raise ValueError(f"tactic_w {self.tactic_w.shape} != {(N_TACTIC, FEATURE_DIM)}")
        if self.build_w.shape != (N_BUILD, FEATURE_DIM):
            raise ValueError(f"build_w {self.build_w.shape} != {(N_BUILD, FEATURE_DIM)}")

    @classmethod
    def zeros(cls) -> "LinearPolicy":
        return cls(
            np.zeros((N_TACTIC, FEATURE_DIM), np.float32),
            np.zeros(N_TACTIC, np.float32),
            np.zeros((N_BUILD, FEATURE_DIM), np.float32),
            np.zeros(N_BUILD, np.float32),
        )

    def tactic_logits(self, x: np.ndarray) -> np.ndarray:
        return self.tactic_w @ x + self.tactic_b

    def build_logits(self, x: np.ndarray) -> np.ndarray:
        return self.build_w @ x + self.build_b

    def predict(self, x: np.ndarray) -> tuple[Tactic, int]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        tactic = Tactic(int(np.argmax(self.tactic_logits(x))))
        build = BUILD_VOCAB[int(np.argmax(self.build_logits(x)))]
        return tactic, build

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, tactic_w=self.tactic_w, tactic_b=self.tactic_b,
                 build_w=self.build_w, build_b=self.build_b, feature_dim=FEATURE_DIM)

    @classmethod
    def load(cls, path: Path) -> "LinearPolicy":
        data = np.load(path)
        return cls(data["tactic_w"], data["tactic_b"], data["build_w"], data["build_b"])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-9, None)


def fit(X: np.ndarray, y_tactic: np.ndarray, y_build: np.ndarray,
        epochs: int = 40, lr: float = 0.2) -> LinearPolicy:
    """Softmax SGD. Tiny on purpose: a few thousand logged decisions is a game night."""
    n, d = X.shape
    assert d == FEATURE_DIM
    W_t = np.zeros((N_TACTIC, d), np.float32)
    b_t = np.zeros(N_TACTIC, np.float32)
    W_b = np.zeros((N_BUILD, d), np.float32)
    b_b = np.zeros(N_BUILD, np.float32)
    Yt = np.eye(N_TACTIC, dtype=np.float32)[y_tactic]
    Yb = np.eye(N_BUILD, dtype=np.float32)[y_build]
    for _ in range(epochs):
        pt = softmax(X @ W_t.T + b_t)
        err_t = (pt - Yt) / n
        W_t -= lr * (err_t.T @ X)
        b_t -= lr * err_t.sum(axis=0)
        pb = softmax(X @ W_b.T + b_b)
        err_b = (pb - Yb) / n
        W_b -= lr * (err_b.T @ X)
        b_b -= lr * err_b.sum(axis=0)
    return LinearPolicy(W_t, b_t, W_b, b_b)
