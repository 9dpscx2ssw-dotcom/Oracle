"""The RL policy the agent actually talks to.

This wraps the actor-critic with everything a trading loop needs around it:

  • a **decision** API — given a strategy signal and its market context, return
    take/skip plus a confidence the agent can log and gate on;
  • a **warm-up** so the policy gathers experience (taking everything) before it
    starts blocking signals — a cold, untrained net must not starve itself;
  • a **replay buffer** so each update sees a small batch of past decisions, not
    just the latest noisy one;
  • **counterfactual rewards**: a *taken* trade is graded by its realized PnL; a
    *skipped* trade is shadow-filled learning-only, so we can grade the skip by
    the PnL it avoided. Both train the same policy, in the same units;
  • **persistence + a snapshot** for the dashboard's Learning / RL tab.

Reward convention (everything normalized by the trade's dollar risk, then clipped):

    take  →  reward = +pnl_norm     (you earned it)
    skip  →  reward = -pnl_norm     (you avoided it: skipping a loser scores +)

so the policy learns to take signals whose expected normalized PnL is positive
and skip the rest.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from ...data.models import Signal, Trade
from ...features.feature_store import FeatureSet
from . import state as state_mod
from .network import ActorCritic

log = logging.getLogger(__name__)

SKIP, TAKE = 0, 1


@dataclass
class Decision:
    """The policy's verdict on one signal."""

    take: bool
    confidence: float          # P(take) under the current policy, 0..1
    value: float               # critic's estimate of the state's value
    action: int                # SKIP or TAKE (what we'll actually do, post-explore)
    state: np.ndarray          # the encoded state, stashed on the trade for learning
    explored: bool = False     # True if epsilon-greedy overrode the greedy action


class RLPolicy:
    def __init__(
        self,
        *,
        hidden: int = 32,
        lr: float = 3e-3,
        gamma: float = 0.95,
        entropy_coef: float = 0.01,
        buffer_size: int = 5000,
        batch_size: int = 32,
        warmup_trades: int = 40,
        take_threshold: float = 0.5,
        epsilon_start: float = 0.30,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        reward_clip: float = 3.0,
        seed: int = 0,
    ):
        self.net = ActorCritic(
            state_mod.STATE_DIM, hidden=hidden, lr=lr, gamma=gamma,
            entropy_coef=entropy_coef, seed=seed,
        )
        self.batch_size = batch_size
        self.warmup_trades = warmup_trades
        self.take_threshold = take_threshold
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.reward_clip = reward_clip
        self._rng = random.Random(seed)

        self.buffer: deque[tuple[np.ndarray, int, float]] = deque(maxlen=buffer_size)
        # Per-sample decay for recency-weighted replay: chosen so a transition
        # half a buffer old is ~1/e as likely to be drawn as the newest one.
        self._replay_decay = float(np.exp(-2.0 / max(buffer_size, 2)))
        # Calibration: (P(take), won?) pairs for TAKEN trades. Confidence-scaled
        # sizing rests on these probabilities meaning something.
        self.calibration: deque[tuple[float, bool]] = deque(maxlen=200)
        # Telemetry for the dashboard.
        self.updates = 0
        self.cumulative_reward = 0.0
        self.action_counts = {"take": 0, "skip": 0}
        self.reward_history: deque[float] = deque(maxlen=200)
        # Rolling window of recent decisions so a collapse to all-skip (or
        # all-take) is visible while it happens, not only in lifetime totals.
        self.recent_actions: deque[int] = deque(maxlen=100)
        self.last_value = 0.0
        self.last_p_take = 0.0

    # ── decision ───────────────────────────────────────────────────────────────
    @property
    def warming_up(self) -> bool:
        return self.updates < self.warmup_trades

    def decide(
        self, signal: Signal, features: FeatureSet, portfolio_heat: float = 0.0,
        *, explore: bool = True, regime: str | None = None,
    ) -> Decision:
        """Encode the signal's context and choose take/skip.

        ``explore=False`` disables epsilon-greedy for this decision — the agent
        passes it for LIVE signals so random exploration only ever spends
        shadow-book money.
        """
        s = state_mod.encode(signal, features, portfolio_heat, regime=regime)
        probs, value = self.net.predict(s)
        p_take = float(probs[TAKE])
        if not np.isfinite(p_take):    # numerical guard: degrade to "take"
            p_take, value = 1.0, 0.0
        self.last_p_take, self.last_value = p_take, float(value)

        # Greedy action from the policy, then optional exploration / warm-up.
        action = TAKE if p_take >= self.take_threshold else SKIP
        explored = False
        if self.warming_up:
            action = TAKE                       # observe everything while cold
        elif explore and self._rng.random() < self.epsilon:
            action = self._rng.choice((SKIP, TAKE))
            explored = True

        self.recent_actions.append(action)
        return Decision(
            take=(action == TAKE), confidence=p_take, value=float(value),
            action=action, state=s, explored=explored,
        )

    @property
    def recent_take_rate(self) -> float | None:
        """Fraction of the last ≤100 decisions that were TAKE (None if too few)."""
        if len(self.recent_actions) < 20:
            return None
        return sum(1 for a in self.recent_actions if a == TAKE) / len(self.recent_actions)

    # ── learning ───────────────────────────────────────────────────────────────
    def reward_for(self, action: int, pnl: float, risk_amount: float) -> float:
        """Normalize a closed trade's PnL into a clipped reward for `action`."""
        pnl_norm = pnl / risk_amount if risk_amount else pnl
        pnl_norm = float(np.clip(pnl_norm, -self.reward_clip, self.reward_clip))
        return pnl_norm if action == TAKE else -pnl_norm

    def learn(self, state: np.ndarray, action: int, reward: float) -> None:
        """Record one graded decision and take a minibatch gradient step."""
        self.buffer.append((np.asarray(state, dtype=np.float64), int(action), float(reward)))
        self.cumulative_reward += reward
        self.reward_history.append(float(reward))
        self.action_counts["take" if action == TAKE else "skip"] += 1

        if len(self.buffer) < self.batch_size:
            return
        # Recency-weighted replay: markets are non-stationary, so a uniform
        # sample over 5,000 mixed-regime transitions trains on conditions that
        # no longer exist. Newest transitions are ~e× likelier to be sampled
        # than ones half a buffer older.
        n = len(self.buffer)
        weights = [self._replay_decay ** (n - 1 - i) for i in range(n)]
        idxs = self._rng.choices(range(n), weights=weights, k=self.batch_size)
        batch = [self.buffer[i] for i in idxs]
        states = np.stack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float64)
        # One-step (terminal) targets: each decision is graded by its own outcome.
        self.net.update(states, actions, rewards)
        self.updates += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn_from_trade(self, trade: Trade) -> bool:
        """Grade a just-closed trade if it carries an RL decision. Returns True if learned."""
        ctx = trade.context or {}
        if "rl_state" not in ctx or "rl_action" not in ctx or trade.pnl is None:
            return False
        state = np.asarray(ctx["rl_state"], dtype=np.float64)
        # A policy resize (state-dim change) leaves old stamped states in open
        # trades/buffers — skip them rather than crash the grade.
        if state.shape[0] != state_mod.STATE_DIM:
            return False
        action = int(ctx["rl_action"])
        risk = float(ctx.get("rl_risk", 0.0))
        reward = self.reward_for(action, float(trade.pnl), risk)
        # Calibration sample: did the stamped P(take) predict a winner?
        p = ctx.get("rl_p")
        if action == TAKE and p is not None:
            self.calibration.append((float(p), float(trade.pnl) > 0))
        self.learn(state, action, reward)
        return True

    @property
    def brier_score(self) -> float | None:
        """Mean squared error of P(take) vs realized win over recent takes.

        0.25 ≈ uninformative; meaningfully below 0.25 = calibrated confidence
        (what confidence-scaled sizing implicitly assumes)."""
        if len(self.calibration) < 20:
            return None
        return float(np.mean([(p - (1.0 if won else 0.0)) ** 2
                              for p, won in self.calibration]))

    @staticmethod
    def stamp(trade: Trade, decision: Decision, risk_amount: float) -> None:
        """Attach a decision to an open trade so its close can be graded later."""
        trade.context = {
            **(trade.context or {}),
            "rl_state": decision.state.tolist(),
            "rl_action": decision.action,
            "rl_risk": float(risk_amount),
            "rl_p": round(decision.confidence, 4),
        }

    # ── persistence + telemetry ─────────────────────────────────────────────────
    @staticmethod
    def _state_path(path: str):
        from pathlib import Path
        return Path(path).with_suffix(".state.npz")

    def save(self, path: str) -> None:
        try:
            self.net.save(path)
        except OSError as e:
            log.warning("RL policy save failed (%s): %s", path, e)
            return
        # Persist training + telemetry state alongside the weights so a restart
        # resumes where it left off (epsilon, counters, reward history, buffer)
        # instead of re-warming from zero.
        try:
            buf = list(self.buffer)
            if buf:
                buf_states = np.stack([b[0] for b in buf])
                buf_actions = np.array([b[1] for b in buf], dtype=np.int64)
                buf_rewards = np.array([b[2] for b in buf], dtype=np.float64)
            else:
                buf_states = np.zeros((0, state_mod.STATE_DIM), dtype=np.float64)
                buf_actions = np.zeros((0,), dtype=np.int64)
                buf_rewards = np.zeros((0,), dtype=np.float64)
            # Atomic (tmp + rename) so a crash mid-write can't corrupt the
            # sidecar (audit F-13).
            sp = self._state_path(path)
            tmp = sp.parent / (sp.name + ".tmp.npz")
            np.savez(
                tmp,
                epsilon=self.epsilon, updates=self.updates,
                cumulative_reward=self.cumulative_reward,
                take=self.action_counts.get("take", 0), skip=self.action_counts.get("skip", 0),
                reward_history=np.array(list(self.reward_history), dtype=np.float64),
                last_p_take=self.last_p_take, last_value=self.last_value,
                buf_states=buf_states, buf_actions=buf_actions, buf_rewards=buf_rewards,
            )
            tmp.replace(sp)
        except OSError as e:
            log.warning("RL state save failed: %s", e)

    def load(self, path: str) -> bool:
        if not self.net.load(path):
            return False
        # Weights restored; now best-effort restore the training state. If the
        # sidecar is missing/corrupt we keep the loaded weights and reset counters.
        sp = self._state_path(path)
        try:
            if sp.exists():
                d = np.load(sp, allow_pickle=False)
                self.epsilon = float(d["epsilon"])
                self.updates = int(d["updates"])
                self.cumulative_reward = float(d["cumulative_reward"])
                self.action_counts = {"take": int(d["take"]), "skip": int(d["skip"])}
                self.reward_history = deque(
                    (float(x) for x in d["reward_history"]), maxlen=self.reward_history.maxlen)
                self.last_p_take = float(d["last_p_take"])
                self.last_value = float(d["last_value"])
                self.buffer.clear()
                ba, br, bs = d["buf_actions"], d["buf_rewards"], d["buf_states"]
                # A state-dim change (encoder upgrade) makes old transitions
                # unstackable with new ones — drop them, keep the counters.
                if bs.ndim == 2 and bs.shape[1] == state_mod.STATE_DIM:
                    for i in range(len(ba)):
                        self.buffer.append(
                            (np.asarray(bs[i], dtype=np.float64), int(ba[i]), float(br[i])))
                elif len(ba):
                    log.warning("RL replay buffer discarded: stored state dim %s "
                                "!= current %d (encoder upgraded)",
                                bs.shape[1:], state_mod.STATE_DIM)
        except (OSError, KeyError, ValueError) as e:
            log.warning("RL state load failed (weights kept, counters reset): %s", e)
        return True

    def snapshot(self) -> dict:
        """Compact, JSON-safe view for the dashboard's Learning / RL tab."""
        rh = list(self.reward_history)
        avg = sum(rh) / len(rh) if rh else 0.0
        return {
            "enabled": True,
            "mode": "warmup" if self.warming_up else "active",
            "updates": self.updates,
            "cumulative_reward": round(self.cumulative_reward, 3),
            "avg_reward": round(avg, 4),
            "epsilon": round(self.epsilon, 3),
            "states_learned": len(self.buffer),
            "warmup_remaining": max(0, self.warmup_trades - self.updates),
            "action_counts": dict(self.action_counts),
            "reward_history": [round(r, 3) for r in rh[-60:]],
            "last_p_take": round(self.last_p_take, 3),
            "last_value": round(self.last_value, 3),
            "recent_take_rate": (round(self.recent_take_rate, 3)
                                 if self.recent_take_rate is not None else None),
            "brier": (round(self.brier_score, 4)
                      if self.brier_score is not None else None),
            "calibration_n": len(self.calibration),
        }
