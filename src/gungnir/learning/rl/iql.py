"""Implicit Q-Learning (IQL, Kostrikov et al. 2021) — conservative offline RL.

DQN can over-value actions the logged data never took (out-of-distribution
actions), which is dangerous when you can't explore live to correct it. IQL never
queries the Q-function at unseen actions: it learns a state value V via *expectile*
regression toward the dataset's own Q-values, bootstraps Q from V, and extracts a
policy by advantage-weighted regression (AWR). That makes it a far safer fit for
trading than vanilla DQN, at the cost of more moving parts.

Pure NumPy, discrete actions {flat, long, short}, to match the project's ethos.
Advisory/offline only — not wired into live execution.

Update (per minibatch of (s, a, r, s', done)):
  V:   expectile_τ( min(Q1_target, Q2_target)(s,a) − V(s) )
  Q_i: ( r + γ(1−done)·V(s') − Q_i(s,a) )²            for i = 1, 2
  π:   maximize  w·logπ(a|s),  w = clip(exp(β·(Q_target(s,a) − V(s))), ·, w_max)
  targets: Polyak soft update of Q1/Q2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import ACTIONS, STATE_DIM
from .offline import Transition, _Adam, _ACTION_LABEL


class _MLP:
    """One hidden tanh layer with manual backprop + Adam."""

    def __init__(self, din, dout, hidden=64, lr=3e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((din, hidden)) * np.sqrt(1.0 / din)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, dout)) * np.sqrt(1.0 / hidden)
        self.b2 = np.zeros(dout)
        self._opt = {n: _Adam(getattr(self, n).shape, lr) for n in ("W1", "b1", "W2", "b2")}

    def forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2, (X, H)

    def backward(self, dO, cache):
        X, H = cache
        dW2, db2 = H.T @ dO, dO.sum(0)
        dZ1 = (dO @ self.W2.T) * (1.0 - H * H)
        dW1, db1 = X.T @ dZ1, dZ1.sum(0)
        for n, g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
            self._opt[n].step(getattr(self, n), g)

    def polyak_to(self, tgt: "_MLP", tau: float):
        for n in ("W1", "b1", "W2", "b2"):
            getattr(tgt, n)[...] = (1 - tau) * getattr(tgt, n) + tau * getattr(self, n)

    def copy_to(self, tgt: "_MLP"):
        for n in ("W1", "b1", "W2", "b2"):
            getattr(tgt, n)[...] = getattr(self, n)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class IQL:
    def __init__(self, state_dim=STATE_DIM, n_actions=len(ACTIONS), hidden=64,
                 lr=3e-4, gamma=0.99, tau=0.7, beta=3.0, polyak=0.01, w_max=100.0, seed=0):
        self.n_actions = n_actions
        self.gamma, self.expectile, self.beta = gamma, tau, beta
        self.polyak, self.w_max = polyak, w_max
        self.q1 = _MLP(state_dim, n_actions, hidden, lr, seed)
        self.q2 = _MLP(state_dim, n_actions, hidden, lr, seed + 1)
        self.q1t = _MLP(state_dim, n_actions, hidden, lr, seed)
        self.q1.copy_to(self.q1t)
        self.q2t = _MLP(state_dim, n_actions, hidden, lr, seed + 1)
        self.q2.copy_to(self.q2t)
        self.v = _MLP(state_dim, 1, hidden, lr, seed + 2)
        self.pi = _MLP(state_dim, n_actions, hidden, lr, seed + 3)

    def act(self, state: np.ndarray, greedy: bool = True) -> int:
        logits, _ = self.pi.forward(state.reshape(1, -1))
        return int(np.argmax(logits[0]))

    def _qt_sa(self, s, a):
        q1t, _ = self.q1t.forward(s)
        q2t, _ = self.q2t.forward(s)
        idx = np.arange(len(a))
        return np.minimum(q1t[idx, a], q2t[idx, a])

    def train(self, data: list[Transition], epochs=40, batch=64, seed=0) -> dict:
        if not data:
            return {"steps": 0, "transitions": 0}
        rng = np.random.default_rng(seed)
        S = np.stack([t.s for t in data])
        A = np.array([t.a for t in data])
        R = np.array([t.r for t in data])
        S2 = np.stack([t.s2 for t in data])
        D = np.array([t.done for t in data], dtype=np.float64)
        n = len(data)
        steps = 0
        for _ in range(epochs):
            for b in np.array_split(rng.permutation(n), max(1, n // batch)):
                s, a, r, s2, d = S[b], A[b], R[b], S2[b], D[b]
                B = len(b)
                idx = np.arange(B)
                qt = self._qt_sa(s, a)                          # min target Q(s,a)

                # V: expectile regression toward qt.
                vpred, vcache = self.v.forward(s)
                vpred = vpred[:, 0]
                u = qt - vpred
                w = np.where(u > 0, self.expectile, 1.0 - self.expectile)
                self.v.backward((-(2.0 * w * u) / B).reshape(-1, 1), vcache)

                # Q1,Q2: TD target r + γ(1−d)V(s').
                v2, _ = self.v.forward(s2)
                y = r + self.gamma * (1.0 - d) * v2[:, 0]
                for q in (self.q1, self.q2):
                    qpred, qcache = q.forward(s)
                    dO = np.zeros_like(qpred)
                    dO[idx, a] = (2.0 * (qpred[idx, a] - y)) / B
                    q.backward(dO, qcache)

                # Policy: advantage-weighted regression.
                adv = qt - vpred
                weight = np.clip(np.exp(self.beta * adv), 0.0, self.w_max)
                logits, pcache = self.pi.forward(s)
                probs = _softmax(logits)
                onehot = np.zeros_like(probs)
                onehot[idx, a] = 1.0
                self.pi.backward((weight.reshape(-1, 1) * (probs - onehot)) / B, pcache)

                self.q1.polyak_to(self.q1t, self.polyak)
                self.q2.polyak_to(self.q2t, self.polyak)
                steps += 1
        return {"steps": steps, "transitions": n}

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **{f"pi_{n}": getattr(self.pi, n) for n in ("W1", "b1", "W2", "b2")})

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            d = np.load(p)
            if d["pi_W1"].shape != self.pi.W1.shape:
                return False
            for n in ("W1", "b1", "W2", "b2"):
                getattr(self.pi, n)[...] = d[f"pi_{n}"]
            return True
        except (OSError, KeyError, ValueError):
            return False


def recommend(agent: IQL, features, position=0.0, holding=0, unrealized=0.0):
    """Advisory action from an IQL policy (mirrors offline.recommend)."""
    from .env import build_state
    candle = features.candles[-1] if getattr(features, "candles", None) else None
    s = build_state(features, candle, position, holding, unrealized,
                    getattr(features, "symbol", "") or "")
    a = agent.act(s)
    return a, _ACTION_LABEL[a]
