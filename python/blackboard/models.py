"""Numpy-only model backends and the `.npz` format the frozen bot loads.

Every model file stores: `kind` (linear | mlp), `task` (binary | softmax | regression), weights,
input standardization (`mu`, `sd`), output labels, and the feature spec it was trained on
(`feature_set`, `feature_version`, `feature_hash`). `load_model(path, spec)` refuses a model whose
feature spec does not match the running code, so a component can fall back to its scripted teacher
instead of acting on garbage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class FeatureMismatch(ValueError):
    pass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    version: int
    names: tuple[str, ...]

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def hash(self) -> str:
        return hashlib.sha1("\n".join(self.names).encode()).hexdigest()[:12]

    def check(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.dim:
            raise FeatureMismatch(f"{self.name} v{self.version}: got {x.shape[-1]} features, want {self.dim}")
        return x


def _act_out(z: np.ndarray, task: str) -> np.ndarray:
    if task == "binary":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    if task == "softmax":
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)
    return z


class Model:
    """Base: standardize -> forward -> output activation."""

    kind = "base"

    def __init__(self, task: str, mu: np.ndarray, sd: np.ndarray, labels: Sequence[str] = (),
                 spec: Optional[FeatureSpec] = None, meta: Optional[dict] = None) -> None:
        if task not in ("binary", "softmax", "regression"):
            raise ValueError(task)
        self.task = task
        self.mu = np.asarray(mu, np.float32)
        self.sd = np.asarray(sd, np.float32)
        self.labels = list(labels)
        self.spec = spec
        self.meta = dict(meta or {})

    def logits(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (d,) or (n, d). Returns probabilities (binary: (n,) or scalar) / values."""
        x = np.asarray(x, np.float32)
        single = x.ndim == 1
        X = (x.reshape(-1, x.shape[-1]) - self.mu) / self.sd
        out = _act_out(self.logits(X), self.task)
        if self.task == "binary" and out.ndim == 2 and out.shape[1] == 1:
            out = out[:, 0]
        return out[0] if single else out

    def _arrays(self) -> dict:
        raise NotImplementedError

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = self.spec
        np.savez(path, kind=self.kind, task=self.task, mu=self.mu, sd=self.sd,
                 labels=np.array(json.dumps(self.labels)), meta=np.array(json.dumps(self.meta)),
                 feature_set=spec.name if spec else "", feature_version=spec.version if spec else -1,
                 feature_hash=spec.hash if spec else "", feature_names=np.array(json.dumps(list(spec.names) if spec else [])),
                 **self._arrays())
        return path

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for v in self._arrays().values():
            h.update(np.ascontiguousarray(v).tobytes())
        return h.hexdigest()[:12]


class LinearModel(Model):
    kind = "linear"

    def __init__(self, W: np.ndarray, b: np.ndarray, **kw) -> None:
        super().__init__(**kw)
        self.W = np.asarray(W, np.float32)     # (d, k)
        self.b = np.asarray(b, np.float32)     # (k,)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def _arrays(self) -> dict:
        return {"W0": self.W, "b0": self.b}


class MLPModel(Model):
    kind = "mlp"

    def __init__(self, Ws: list[np.ndarray], bs: list[np.ndarray], **kw) -> None:
        super().__init__(**kw)
        self.Ws = [np.asarray(w, np.float32) for w in Ws]
        self.bs = [np.asarray(b, np.float32) for b in bs]

    def hidden(self, X: np.ndarray) -> list[np.ndarray]:
        hs = [X]
        h = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = np.maximum(h @ W + b, 0.0)
            hs.append(h)
        return hs

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.hidden(X)[-1] @ self.Ws[-1] + self.bs[-1]

    def _arrays(self) -> dict:
        out = {}
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            out[f"W{i}"], out[f"b{i}"] = W, b
        return out


def load_model(path: Path, spec: Optional[FeatureSpec] = None) -> Model:
    data = np.load(Path(path), allow_pickle=False)
    kind = str(data["kind"])
    fs = str(data["feature_set"])
    fv = int(data["feature_version"])
    fh = str(data["feature_hash"])
    names = tuple(json.loads(str(data["feature_names"])))
    mspec = FeatureSpec(fs, fv, names) if fs else None
    if spec is not None:
        if mspec is None or (fs, fv, fh) != (spec.name, spec.version, spec.hash):
            raise FeatureMismatch(f"{path}: model features {fs} v{fv} ({fh}) != code {spec.name} v{spec.version} "
                                  f"({spec.hash})")
    kw = dict(task=str(data["task"]), mu=data["mu"], sd=data["sd"], labels=json.loads(str(data["labels"])),
              spec=mspec, meta=json.loads(str(data["meta"])))
    n = sum(1 for k in data.files if k.startswith("W"))
    Ws = [data[f"W{i}"] for i in range(n)]
    bs = [data[f"b{i}"] for i in range(n)]
    if kind == "linear":
        return LinearModel(Ws[0], bs[0], **kw)
    if kind == "mlp":
        return MLPModel(Ws, bs, **kw)
    raise ValueError(f"{path}: unknown model kind {kind!r}")


def try_load(path: Optional[str | Path], spec: Optional[FeatureSpec] = None, search: Sequence[Path] = ()) -> \
        tuple[Optional[Model], str]:
    """(model, message). Looks at `path`, then `search` dirs joined with the file name."""
    if not path:
        return None, "no model configured"
    cands = [Path(path)] + [Path(d) / Path(path).name for d in search]
    for p in cands:
        if p.is_file():
            try:
                return load_model(p, spec), f"loaded {p}"
            except (FeatureMismatch, KeyError, ValueError, OSError) as e:
                return None, f"rejected {p}: {e}"
    return None, f"not found: {path}"


def model_search_dirs() -> list[Path]:
    """Where models are found: bwapi-data/read (tournament read folder), the competition pack's
    AI/models (next to the frozen bot.exe), then python/models in the repo."""
    import sys
    here = Path(__file__).resolve().parent
    dirs = [Path("bwapi-data") / "read", Path("bwapi-data") / "AI" / "models"]
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent.parent / "models")
    dirs.append(here.parent / "models")
    return dirs
