"""Numpy trainers (no torch needed): linear and small MLPs with Adam, minibatches, L2, early stopping.

`fit(X, y, task=..., hidden=())` returns a `LinearModel` (hidden=()) or `MLPModel`.
Targets: binary -> y in {0,1} (or soft probabilities); softmax -> class indices or (n, k) soft
targets; regression -> (n,) or (n, k). Sample weights are supported (e.g. to down-weight early
decisions in long games).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .models import FeatureSpec, LinearModel, MLPModel, Model, _act_out

log = logging.getLogger("blackboard.train")


@dataclass
class FitReport:
    train_loss: float
    val_loss: float
    val_metric: float          # accuracy (binary/softmax) or R^2 (regression)
    epochs: int
    n_train: int
    n_val: int

    def __str__(self) -> str:
        return (f"train {self.train_loss:.4f} val {self.val_loss:.4f} metric {self.val_metric:.3f} "
                f"epochs {self.epochs} n {self.n_train}/{self.n_val}")


def _targets(y: np.ndarray, task: str, k: Optional[int]) -> np.ndarray:
    y = np.asarray(y)
    if task == "softmax":
        if y.ndim == 1:
            k = int(k or (y.max() + 1))
            return np.eye(k, dtype=np.float32)[y.astype(int)]
        return y.astype(np.float32)
    if y.ndim == 1:
        return y.reshape(-1, 1).astype(np.float32)
    return y.astype(np.float32)


def _loss(p: np.ndarray, Y: np.ndarray, task: str, w: np.ndarray) -> float:
    eps = 1e-7
    if task == "binary":
        l = -(Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)).sum(axis=1)
    elif task == "softmax":
        l = -(Y * np.log(p + eps)).sum(axis=1)
    else:
        l = ((p - Y) ** 2).sum(axis=1)
    return float((l * w).sum() / max(w.sum(), 1e-9))


def _metric(p: np.ndarray, Y: np.ndarray, task: str) -> float:
    if len(Y) == 0:
        return float("nan")
    if task == "binary":
        return float(((p > 0.5) == (Y > 0.5)).mean())
    if task == "softmax":
        return float((p.argmax(1) == Y.argmax(1)).mean())
    ss = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ((p - Y) ** 2).sum() / max(ss, 1e-9))


def fit(X: np.ndarray, y: np.ndarray, task: str = "binary", hidden: Sequence[int] = (), labels: Sequence[str] = (),
        spec: Optional[FeatureSpec] = None, weights: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None,
        epochs: int = 200, lr: float = 1e-2, l2: float = 1e-4, batch: int = 256, val_frac: float = 0.2,
        patience: int = 20, seed: int = 0, n_classes: Optional[int] = None, meta: Optional[dict] = None) -> \
        tuple[Model, FitReport]:
    """`groups` (e.g. game ids) keeps all rows of a game on the same side of the validation split."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    if spec is not None and X.shape[1] != spec.dim:
        raise ValueError(f"X has {X.shape[1]} columns, spec {spec.name} has {spec.dim}")
    Y = _targets(y, task, n_classes)
    n, d = X.shape
    k = Y.shape[1]
    w = np.ones(n, np.float32) if weights is None else np.asarray(weights, np.float32)

    if groups is not None:
        ug = np.unique(groups)
        rng.shuffle(ug)
        val_g = set(ug[: int(round(len(ug) * val_frac))].tolist()) if len(ug) > 1 else set()
        vmask = np.array([g in val_g for g in groups])
    else:
        vmask = rng.random(n) < val_frac if n >= 10 else np.zeros(n, bool)
    Xt, Yt, wt = X[~vmask], Y[~vmask], w[~vmask]
    Xv, Yv, wv = X[vmask], Y[vmask], w[vmask]

    mu = Xt.mean(0) if len(Xt) else np.zeros(d, np.float32)
    sd = Xt.std(0) + 1e-6 if len(Xt) else np.ones(d, np.float32)
    sd[sd < 1e-5] = 1.0
    Zt = (Xt - mu) / sd
    Zv = (Xv - mu) / sd

    sizes = [d, *hidden, k]
    Ws = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a) * (0.1 if i == len(sizes) - 2 else 1.0)).astype(np.float32)
          for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
    bs = [np.zeros(b, np.float32) for b in sizes[1:]]
    if task == "binary" and not hidden:
        pos = float((Yt * wt[:, None]).sum() / max(wt.sum(), 1e-9)) if len(Yt) else 0.5
        pos = min(max(pos, 1e-3), 1 - 1e-3)
        bs[-1][:] = np.log(pos / (1 - pos))
    mW = [np.zeros_like(a) for a in Ws]
    vW = [np.zeros_like(a) for a in Ws]
    mb = [np.zeros_like(a) for a in bs]
    vb = [np.zeros_like(a) for a in bs]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0

    def forward(Z):
        hs = [Z]
        h = Z
        for W, b in zip(Ws[:-1], bs[:-1]):
            h = np.maximum(h @ W + b, 0.0)
            hs.append(h)
        return hs, _act_out(h @ Ws[-1] + bs[-1], task)

    best = (np.inf, None, 0)
    ep = 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Zt))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            hs, p = forward(Zt[idx])
            bw_ = wt[idx][:, None] / max(wt[idx].sum(), 1e-9)
            g = (p - Yt[idx]) * bw_ * (2.0 if task == "regression" else 1.0)
            step += 1
            for i in reversed(range(len(Ws))):
                gW = hs[i].T @ g + l2 * Ws[i]
                gb = g.sum(0)
                if i > 0:
                    g = (g @ Ws[i].T) * (hs[i] > 0)
                for P, G, M, V in ((Ws[i], gW, mW[i], vW[i]), (bs[i], gb, mb[i], vb[i])):
                    M *= b1
                    M += (1 - b1) * G
                    V *= b2
                    V += (1 - b2) * G * G
                    P -= lr * (M / (1 - b1 ** step)) / (np.sqrt(V / (1 - b2 ** step)) + eps)
        if len(Zv):
            vl = _loss(forward(Zv)[1], Yv, task, wv)
        else:
            vl = _loss(forward(Zt)[1], Yt, task, wt)
        if vl < best[0] - 1e-5:
            best = (vl, ([a.copy() for a in Ws], [a.copy() for a in bs]), ep)
        elif ep - best[2] >= patience:
            break
    if best[1] is not None:
        Ws, bs = best[1]
    kw = dict(task=task, mu=mu, sd=sd, labels=list(labels), spec=spec, meta=dict(meta or {}))
    model: Model = LinearModel(Ws[0], bs[0], **kw) if not hidden else MLPModel(Ws, bs, **kw)
    pt = model.predict(Xt) if len(Xt) else np.zeros((0, k))
    pv = model.predict(Xv) if len(Xv) else np.zeros((0, k))
    pt2 = pt.reshape(len(Xt), -1)
    pv2 = pv.reshape(len(Xv), -1)
    rep = FitReport(_loss(pt2, Yt, task, wt) if len(Xt) else float("nan"),
                    _loss(pv2, Yv, task, wv) if len(Xv) else float("nan"),
                    _metric(pv2, Yv, task), ep, len(Xt), len(Xv))
    model.meta.update(train_loss=rep.train_loss, val_loss=rep.val_loss, val_metric=rep.val_metric,
                      n_train=rep.n_train, n_val=rep.n_val, epochs=rep.epochs)
    log.info("fit %s %s: %s", task, "linear" if not hidden else f"mlp{list(hidden)}", rep)
    return model, rep
