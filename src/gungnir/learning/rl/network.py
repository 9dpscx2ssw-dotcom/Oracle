"""A tiny advantage actor-critic, implemented in pure NumPy.

Why NumPy and not PyTorch: the agent runs in a 1 GB homelab container and the
project's whole ethos is "stay light." The decision here is a 2-way choice
(take / skip) over a 14-dim state, so the network is tiny — a single hidden layer
is plenty. Hand-rolling the forward/backward pass keeps the dependency footprint
at the numpy we already ship, makes the maths inspectable, and trains in
microseconds per step.

Architecture::

    state ─▶ Linear(D→H) ─▶ tanh ─┬─▶ Linear(H→2) ─▶ softmax   (actor: π(a|s))
                                   └─▶ Linear(H→1)             (critic: V(s))

Learning is one-step advantage actor-critic (A2C). Each decision is graded the
moment its trade closes, so we treat it as a contextual bandit: the value target
is just the (optionally bootstrapped) reward, and ``advantage = target - V(s)``
drives both heads. An entropy bonus keeps the policy from collapsing to a single
action before it has seen enough outcomes. Adam gives stable steps on the small,
noisy minibatches a trading loop produces.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _Adam:
    """Per-parameter Adam state. Minimal, just what the update loop needs."""

    def __init__(self, shape: tuple[int, ...], lr: float, b1: float = 0.9, b2: float = 0.999):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, 1e-8
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, param: np.ndarray, grad: np.ndarray) -> None:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ActorCritic:
    """Single-hidden-layer actor-critic over a discrete {skip=0, take=1} action."""

    N_ACTIONS = 2

    def __init__(
        self,
        state_dim: int,
        hidden: int = 32,
        lr: float = 3e-3,
        gamma: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        seed: int = 0,
    ):
        self.state_dim = state_dim
        self.hidden = hidden
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        rng = np.random.default_rng(seed)
        # Xavier-ish init keeps early activations in tanh's linear range.
        self.W1 = rng.normal(0, 1 / np.sqrt(state_dim), (hidden, state_dim))
        self.b1 = np.zeros(hidden)
        self.Wa = rng.normal(0, 1 / np.sqrt(hidden), (self.N_ACTIONS, hidden))
        self.ba = np.zeros(self.N_ACTIONS)
        self.Wv = rng.normal(0, 1 / np.sqrt(hidden), (1, hidden))
        self.bv = np.zeros(1)

        self._opt = {
            "W1": _Adam(self.W1.shape, lr), "b1": _Adam(self.b1.shape, lr),
            "Wa": _Adam(self.Wa.shape, lr), "ba": _Adam(self.ba.shape, lr),
            "Wv": _Adam(self.Wv.shape, lr), "bv": _Adam(self.bv.shape, lr),
        }

    # ── inference ────────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (hidden activations, action probs, state values) for a batch."""
        H = np.tanh(X @ self.W1.T + self.b1)          # (B, hidden)
        probs = _softmax(H @ self.Wa.T + self.ba)     # (B, 2)
        values = (H @ self.Wv.T + self.bv).ravel()    # (B,)
        return H, probs, values

    def predict(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Single-state convenience: returns (action_probs (2,), value scalar)."""
        H, probs, values = self.forward(state.reshape(1, -1))
        return probs[0], float(values[0])

    # ── learning ─────────────────────────────────────────────────────────────
    def update(
        self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray
    ) -> dict[str, float]:
        """One A2C gradient step on a minibatch.

        ``targets`` is the value target G for each transition (reward, already
        bootstrapped by the caller when the next state is non-terminal). Returns a
        small dict of scalar diagnostics.
        """
        B = states.shape[0]
        H, probs, values = self.forward(states)
        advantage = targets - values                       # (B,)

        # ── critic: 0.5 * (V - G)^2 ──
        dvalues = self.value_coef * (values - targets) / B    # (B,)

        # ── actor: -logπ(a|s) * adv  - entropy_coef * H(π) ──
        onehot = np.zeros_like(probs)
        onehot[np.arange(B), actions] = 1.0
        adv_d = advantage.reshape(-1, 1)                      # detached advantage
        # policy-gradient term + entropy gradient (encourages exploration)
        logp = np.log(np.clip(probs, 1e-8, 1.0))
        ent = -(probs * logp).sum(axis=1, keepdims=True)      # (B,1)
        dlogits = (probs - onehot) * adv_d
        dlogits += self.entropy_coef * probs * (logp + ent)
        dlogits /= B                                          # (B, 2)

        # ── backprop into the shared hidden layer ──
        dH = dlogits @ self.Wa + dvalues.reshape(-1, 1) @ self.Wv   # (B, hidden)
        dZ1 = dH * (1.0 - H * H)                                     # tanh'

        grads = {
            "Wa": dlogits.T @ H, "ba": dlogits.sum(axis=0),
            "Wv": (dvalues.reshape(-1, 1)).T @ H, "bv": np.array([dvalues.sum()]),
            "W1": dZ1.T @ states, "b1": dZ1.sum(axis=0),
        }
        for name in ("W1", "b1", "Wa", "ba", "Wv", "bv"):
            self._opt[name].step(getattr(self, name), grads[name])

        return {
            "value_loss": float(np.mean(advantage**2)),
            "entropy": float(np.mean(ent)),
            "mean_value": float(np.mean(values)),
            "mean_advantage": float(np.mean(advantage)),
        }

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Atomic save with a .bak of the previous good file (audit F-13: a
        crash mid-savez corrupted the policy, and load() silently started
        fresh — accumulated learning gone without an error)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / (p.name + ".tmp.npz")
        np.savez(
            tmp, W1=self.W1, b1=self.b1, Wa=self.Wa, ba=self.ba, Wv=self.Wv, bv=self.bv,
            state_dim=self.state_dim, hidden=self.hidden,
        )
        if p.exists():
            try:
                import shutil
                shutil.copy2(p, p.parent / (p.name + ".bak"))
            except OSError:
                pass
        tmp.replace(p)

    def _load_file(self, p: Path) -> bool:
        d = np.load(p)
        if int(d["state_dim"]) != self.state_dim or int(d["hidden"]) != self.hidden:
            return False  # architecture changed; start fresh rather than crash
        self.W1, self.b1 = d["W1"], d["b1"]
        self.Wa, self.ba = d["Wa"], d["ba"]
        self.Wv, self.bv = d["Wv"], d["bv"]
        return True

    def load(self, path: str | Path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            return self._load_file(p)
        except (OSError, KeyError, ValueError) as e:
            # Corrupt main file: fall back to the last good backup, loudly.
            bak = p.parent / (p.name + ".bak")
            log.error("RL policy %s failed to load (%s); trying backup %s", p, e, bak)
            if bak.exists():
                try:
                    return self._load_file(bak)
                except (OSError, KeyError, ValueError) as e2:
                    log.error("RL policy backup also failed to load: %s", e2)
            return False
