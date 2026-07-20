"""Offline value-based RL over the trading MDP (`env.TradingEnv`).

Markets give you abundant *logged* history but make live exploration expensive and
dangerous, so we learn **offline**: roll a behavior policy through a replay of real
(or synthetic) candles to collect transitions, then fit a Q-function on that fixed
dataset — no live trial-and-error.

The learner is a small NumPy **Double-DQN** with experience replay:
  target  y = r + γ·(1−done)·Q_target(s', argmax_a Q_online(s', a))
Double-DQN decouples action selection (online net) from evaluation (target net),
which curbs the value over-estimation that makes vanilla DQN brittle on noisy
financial rewards. This is the safe, well-understood first cut; a conservative
offline algorithm (IQL/CQL) is the planned upgrade — its job is to further resist
over-valuing actions the dataset never took.

Deliberately **advisory/offline only**: nothing here is wired into live execution.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .env import ACTIONS, STATE_DIM, TradingEnv


class _Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, 1e-8
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, p, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class QNet:
    """One hidden layer (tanh) → Q-values over the discrete action set."""

    def __init__(self, state_dim=STATE_DIM, hidden=64, n_actions=len(ACTIONS), lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((state_dim, hidden)) * np.sqrt(1.0 / state_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, n_actions)) * np.sqrt(1.0 / hidden)
        self.b2 = np.zeros(n_actions)
        self._opt = {n: _Adam(getattr(self, n).shape, lr)
                     for n in ("W1", "b1", "W2", "b2")}

    def q(self, states: np.ndarray):
        H = np.tanh(states @ self.W1 + self.b1)
        return H @ self.W2 + self.b2, H

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.q(states)[0]

    def update(self, states, actions, targets) -> float:
        B = states.shape[0]
        Q, H = self.q(states)
        qsa = Q[np.arange(B), actions]
        td = qsa - targets
        dQ = np.zeros_like(Q)
        dQ[np.arange(B), actions] = td / B
        dW2 = H.T @ dQ
        db2 = dQ.sum(0)
        dH = dQ @ self.W2.T
        dZ1 = dH * (1.0 - H * H)
        dW1 = states.T @ dZ1
        db1 = dZ1.sum(0)
        for n, g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
            self._opt[n].step(getattr(self, n), g)
        return float(np.mean(td * td))

    def copy_weights_to(self, other: "QNet") -> None:
        other.W1, other.b1 = self.W1.copy(), self.b1.copy()
        other.W2, other.b2 = self.W2.copy(), self.b2.copy()


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


def collect_transitions(env: TradingEnv, epsilon: float = 1.0, seed: int = 0,
                        policy: "OfflineDQN | None" = None) -> list[Transition]:
    """Roll one pass over the series with an epsilon-greedy behavior policy and log
    every transition. epsilon=1.0 → uniform-random exploration (pure offline data)."""
    rng = random.Random(seed)
    out: list[Transition] = []
    s = env.reset()
    while not env.done:
        if policy is not None and rng.random() > epsilon:
            a = policy.act(s, greedy=True)
        else:
            a = rng.choice(ACTIONS)
        s2, r, done, _ = env.step(a)
        out.append(Transition(s, a, r, s2, done))
        s = s2
    return out


@dataclass
class OfflineDQN:
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    seed: int = 0
    online: QNet = field(init=False)
    target: QNet = field(init=False)

    def __post_init__(self):
        self.online = QNet(hidden=self.hidden, lr=self.lr, seed=self.seed)
        self.target = QNet(hidden=self.hidden, lr=self.lr, seed=self.seed)
        self.online.copy_weights_to(self.target)

    def act(self, state: np.ndarray, greedy: bool = True) -> int:
        q = self.online.predict(state.reshape(1, -1))[0]
        return int(np.argmax(q))

    def train(self, data: list[Transition], epochs: int = 30, batch: int = 64,
              target_sync: int = 50, seed: int = 0) -> dict:
        """Fit on a fixed transition dataset (Double-DQN targets)."""
        if not data:
            return {"loss": 0.0, "steps": 0}
        rng = random.Random(seed)
        S = np.stack([t.s for t in data])
        A = np.array([t.a for t in data], dtype=np.int64)
        R = np.array([t.r for t in data], dtype=np.float64)
        S2 = np.stack([t.s2 for t in data])
        D = np.array([t.done for t in data], dtype=np.float64)
        n = len(data)
        steps = 0
        last_loss = 0.0
        for _ in range(epochs):
            idx = list(range(n))
            rng.shuffle(idx)
            for k in range(0, n, batch):
                b = idx[k:k + batch]
                s, a, r, s2, d = S[b], A[b], R[b], S2[b], D[b]
                # Double-DQN: select with online, evaluate with target.
                a2 = np.argmax(self.online.predict(s2), axis=1)
                q2 = self.target.predict(s2)[np.arange(len(b)), a2]
                y = r + self.gamma * (1.0 - d) * q2
                last_loss = self.online.update(s, a, y)
                steps += 1
                if steps % target_sync == 0:
                    self.online.copy_weights_to(self.target)
        return {"loss": last_loss, "steps": steps, "transitions": n}

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        o = self.online
        np.savez(p, W1=o.W1, b1=o.b1, W2=o.W2, b2=o.b2, gamma=self.gamma)

    def load(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            d = np.load(p)
            o = self.online
            if o.W1.shape != d["W1"].shape:
                return False
            o.W1, o.b1, o.W2, o.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
            o.copy_weights_to(self.target)
            return True
        except (OSError, KeyError, ValueError):
            return False


def train_offline(candles, feats, cost=None, *, hidden=64, epochs=30,
                  exploration_passes=3, seed=0) -> tuple[OfflineDQN, dict]:
    """Build the MDP, collect random-exploration transitions over several passes,
    and fit a Double-DQN. Returns (agent, diagnostics)."""
    env = TradingEnv(candles, feats, cost=cost)
    data: list[Transition] = []
    for p in range(exploration_passes):
        data += collect_transitions(env, epsilon=1.0, seed=seed + p)
    agent = OfflineDQN(hidden=hidden, seed=seed)
    diag = agent.train(data, epochs=epochs, seed=seed)
    return agent, diag


_ACTION_LABEL = {0: "FLAT", 1: "LONG", 2: "SHORT"}


def recommend(agent, features, position: float = 0.0, holding: int = 0,
              unrealized: float = 0.0) -> tuple[int, str]:
    """Advisory: what target position would the offline policy take, given the
    current market features and (optional) position context? Returns (action,
    label). Pure + side-effect free so the live loop can log it without risk."""
    from .env import build_state
    candle = features.candles[-1] if getattr(features, "candles", None) else None
    s = build_state(features, candle, position, holding, unrealized,
                    getattr(features, "symbol", "") or "")
    a = agent.act(s, greedy=True)
    return a, _ACTION_LABEL[a]


def evaluate_policy(env: TradingEnv, agent: OfflineDQN) -> dict:
    """Greedy roll-out: cumulative step return (net of costs) and action mix."""
    s = env.reset()
    total = 0.0
    counts = {a: 0 for a in ACTIONS}
    while not env.done:
        a = agent.act(s, greedy=True)
        counts[a] += 1
        s, _, _, info = env.step(a)
        total += info["step_return"]
    return {"total_return": total, "action_counts": counts}
