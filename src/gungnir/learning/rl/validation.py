"""Leakage-aware validation + off-policy evaluation for the offline learners.

Two tools, because backtest P&L on the training window means nothing:

1. **Purged walk-forward** — train on an earlier window, evaluate greedily on a
   strictly *later* window, with an embargo gap between them. Features are
   recomputed per slice (not sliced from a full-series build) so cumulative
   indicators like VWAP can't leak across the boundary. This is the practical
   out-of-sample test that should gate any promotion.

2. **Weighted Importance Sampling (WIS)** — a formal off-policy estimate of a
   target policy's value from data logged under a known behavior policy. Our
   offline data is collected with a *uniform-random* behavior (prob 1/3 per
   action), so the importance ratios are exact. Use a stochastic target (e.g. the
   IQL softmax) so ratios don't collapse to zero.
"""

from __future__ import annotations

import numpy as np


def walk_forward_boundaries(n: int, warmup: int = 60, n_splits: int = 4,
                            embargo: int = 20) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Expanding-window train ranges with a later, embargoed test range each."""
    usable = n - warmup
    fold = usable // (n_splits + 1)
    splits = []
    for k in range(1, n_splits + 1):
        tr_end = warmup + fold * k
        te_start = tr_end + embargo
        te_end = min(n, te_start + fold)
        if te_start < te_end and tr_end > warmup:
            splits.append(((0, tr_end), (te_start, te_end)))
    return splits


def run_walk_forward(candles, cost=None, algo: str = "dqn", n_splits: int = 4,
                     embargo: int = 20, warmup: int = 60, epochs: int = 30,
                     passes: int = 3, seed: int = 0) -> dict:
    """Train/evaluate across purged walk-forward folds. Returns per-fold OOS
    greedy roll-out results + the mean out-of-sample return."""
    from ...features.feature_store import build_kraken_series
    from .env import TradingEnv
    from .offline import collect_transitions, evaluate_policy, train_offline
    from .iql import IQL

    results = []
    for (a, b), (c, d) in walk_forward_boundaries(len(candles), warmup, n_splits, embargo):
        tr, te = candles[a:b], candles[c:d]
        if len(tr) < warmup + 20 or len(te) < warmup + 5:
            continue
        trf = build_kraken_series("wf", tr)       # recompute per slice → no leakage
        tef = build_kraken_series("wf", te)
        if algo == "iql":
            env = TradingEnv(tr, trf, cost=cost, warmup=warmup)
            data = []
            for p in range(passes):
                data += collect_transitions(env, epsilon=1.0, seed=seed + p)
            agent = IQL(seed=seed)
            agent.train(data, epochs=epochs, seed=seed)
        else:
            agent, _ = train_offline(tr, trf, cost=cost, epochs=epochs,
                                     exploration_passes=passes, seed=seed)
        res = evaluate_policy(TradingEnv(te, tef, cost=cost, warmup=warmup), agent)
        results.append({"train": [a, b], "test": [c, d], **res})

    mean_oos = float(np.mean([r["total_return"] for r in results])) if results else 0.0
    return {"folds": results, "n_folds": len(results), "mean_oos_return": mean_oos}


def wis_ope(episodes, target_probs_fn, behavior_prob: float = 1.0 / 3.0) -> float:
    """Weighted Importance Sampling estimate of a target policy's mean return.

    episodes: list of trajectories, each a list of (state, action, reward).
    target_probs_fn(state) -> action-probability vector for the target policy.
    behavior_prob: per-action probability of the (uniform-random) behavior policy.
    """
    num = den = 0.0
    for ep in episodes:
        rho, G = 1.0, 0.0
        for (s, a, r) in ep:
            rho *= float(target_probs_fn(s)[a]) / behavior_prob
            G += r
        num += rho * G
        den += rho
    return num / den if den else 0.0


def collect_episodes(env, length: int = 50, n_episodes: int = 20,
                     seed: int = 0) -> list[list[tuple]]:
    """Chunk uniform-random roll-outs into fixed-length episodes of (s, a, r)
    for off-policy evaluation."""
    import random

    from .env import ACTIONS
    rng = random.Random(seed)
    episodes, ep = [], []
    s = env.reset()
    while not env.done and len(episodes) < n_episodes:
        a = rng.choice(ACTIONS)
        s2, _, _, info = env.step(a)
        ep.append((s, a, info["step_return"]))
        if len(ep) >= length:
            episodes.append(ep)
            ep = []
        s = s2
    if ep:
        episodes.append(ep)
    return episodes
