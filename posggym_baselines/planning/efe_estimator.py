"""Particle-EFE estimator for the bach EFEPlanner.

Estimates the per-tree-node EFE ``G = -(utility + info_gain)`` from K calls
to ``model.step(state, joint_action)``. Uses state-form info gain via
particle bucketing over hashable observations and next-states.

See docs/superpowers/specs/2026-05-18-bach-skateboard-design.md §4 for the
math and the K=32 / state-form / uniform-continuation decisions.

This module is pure Python; no JAX, no numpy required for the math
(``math.log`` suffices since the entropies sum over at most K terms).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import posggym.model as M


@dataclass(frozen=True)
class EFEEstimate:
    """Result of one particle-EFE estimate at a tree-node expansion.

    Attributes:
        g: ``-(utility_hat + info_gain_hat)``. Smaller is better. This is
            the value ``EFEPlanner._simulate`` accumulates into
            ``action_node.value`` (replacing intmcp.py's ``ego_return``
            Welford-mean of returns).
        utility_hat: Mean ego reward across the K samples (treats reward
            as ``log P(o|C^f)`` up to scaling, per spec §4).
        info_gain_hat: State-form expected information gain.
        sample_next_state: One of the K next_state samples, drawn
            uniformly at random by ``estimate_efe.rng``. The
            tree-traversal uses this for the recursive
            ``_simulate(next_hps, child, depth+1)`` call.
        sample_joint_obs: Corresponding joint observation for
            ``sample_next_state``.
        sample_ego_reward: Corresponding ego reward for
            ``sample_next_state``.
        sample_done: Whether the sampled JointTimestep ended the episode.
    """

    g: float
    utility_hat: float
    info_gain_hat: float
    sample_next_state: Any
    sample_joint_obs: Dict[str, Any]
    sample_ego_reward: float
    sample_done: bool


def estimate_efe(
    model: M.POSGModel,
    state: Any,
    joint_action: Dict[str, M.ActType],
    *,
    ego_agent_id: str,
    K: int,
    rng: random.Random,
) -> EFEEstimate:
    """Particle-EFE estimator. See spec §4 for the math.

    Args:
        model: ``posggym.POSGModel``; only ``model.step()`` is called.
        state: Current physical state (passed to ``model.step``).
        joint_action: Per-agent action dict (passed to ``model.step``).
        ego_agent_id: Which agent's reward/obs is the "ego" perspective.
        K: Number of samples per estimate. Must be >= 2; default 32 per
            spec §4 (Miller-Madow bias ~0.1 nats at K=32 vs ~0.3 nats at
            K=12).
        rng: ``random.Random`` for sampling the continuation.

    Returns:
        ``EFEEstimate``; see class docstring.

    Raises:
        ValueError: If ``K < 2`` (info_gain estimator degenerates at K=1).
    """
    if K < 2:
        raise ValueError(f"K must be >= 2 (got {K})")

    # Step 1: K calls to model.step
    timesteps = [model.step(state, joint_action) for _ in range(K)]

    # Step 2: utility_hat = mean ego reward
    ego_rewards = [ts.rewards[ego_agent_id] for ts in timesteps]
    utility_hat = sum(ego_rewards) / K

    # Step 3: state-form info_gain via particle bucketing
    next_states = [ts.state for ts in timesteps]
    ego_obs = [ts.observations[ego_agent_id] for ts in timesteps]

    # H[Q(s'|u)]: entropy of empirical next-state distribution.
    state_counts = Counter(_hashable(s) for s in next_states)
    H_qs_next = sum(-(c / K) * math.log(c / K) for c in state_counts.values())

    # E_{Q(o|u)}[ H[Q(s'|o,u)] ]: weighted avg conditional entropy per obs bucket.
    obs_to_state_counter: Dict[Any, Counter] = defaultdict(Counter)
    for s, o in zip(next_states, ego_obs):
        obs_to_state_counter[_hashable(o)][_hashable(s)] += 1

    weighted_cond_entropy = 0.0
    for sub_state_counter in obs_to_state_counter.values():
        bucket_size = sum(sub_state_counter.values())
        if bucket_size <= 1:
            # Singleton bucket -> p=1 -> -1*log(1) = 0 contribution.
            continue
        weight = bucket_size / K
        H_qs_given_o = sum(
            -(c / bucket_size) * math.log(c / bucket_size)
            for c in sub_state_counter.values()
        )
        weighted_cond_entropy += weight * H_qs_given_o

    info_gain_hat = H_qs_next - weighted_cond_entropy
    g = -(utility_hat + info_gain_hat)

    # Step 5: pick continuation uniformly at random.
    k_star = rng.randrange(K)
    chosen = timesteps[k_star]
    sample_done = (
        chosen.terminations.get(ego_agent_id, False)
        or chosen.truncations.get(ego_agent_id, False)
        or chosen.all_done
    )

    return EFEEstimate(
        g=g,
        utility_hat=utility_hat,
        info_gain_hat=info_gain_hat,
        sample_next_state=chosen.state,
        sample_joint_obs=chosen.observations,
        sample_ego_reward=chosen.rewards[ego_agent_id],
        sample_done=sample_done,
    )


def _hashable(x: Any) -> Any:
    """Make x usable as a dict / Counter key.

    Tuples, ints, strings, frozensets pass through. Numpy arrays / lists
    are converted to a hashable tuple-of-tuples form. For Driving-v1 /
    LBF / PE the states and obs are already tuples, so this is a no-op
    fast path in practice.
    """
    try:
        hash(x)
        return x
    except TypeError:
        if hasattr(x, "flat"):
            return tuple(x.flat)
        if isinstance(x, list):
            return tuple(_hashable(e) for e in x)
        return tuple(x)
