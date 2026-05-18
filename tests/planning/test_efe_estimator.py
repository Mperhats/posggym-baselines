"""Tests for the particle-EFE estimator.

See docs/superpowers/specs/2026-05-18-bach-skateboard-design.md §4 for the
state-form info gain math and K=32 / uniform-continuation decisions.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from posggym_baselines.planning.efe_estimator import EFEEstimate, estimate_efe


@dataclass
class _MockTimestep:
    state: Any
    observations: Dict[str, Any]
    rewards: Dict[str, float]
    terminations: Dict[str, bool]
    truncations: Dict[str, bool]
    all_done: bool


def _make_mock_model(step_sequence):
    """Build a model whose .step(state, action) yields step_sequence in order."""
    model = MagicMock()
    it = iter(step_sequence)
    model.step.side_effect = lambda state, action: next(it)
    return model


def test_estimate_efe_deterministic_env_zero_info_gain():
    """All K samples identical -> info_gain == 0, utility == reward."""
    same_ts = _MockTimestep(
        state="s1",
        observations={"0": "o_a", "1": "o_b"},
        rewards={"0": 1.0, "1": -1.0},
        terminations={"0": False, "1": False},
        truncations={"0": False, "1": False},
        all_done=False,
    )
    model = _make_mock_model([same_ts] * 12)
    result = estimate_efe(
        model,
        state="s0",
        joint_action={"0": 0, "1": 0},
        ego_agent_id="0",
        K=12,
        rng=random.Random(0),
    )
    assert result.utility_hat == pytest.approx(1.0)
    assert result.info_gain_hat == pytest.approx(0.0, abs=1e-10)
    assert result.g == pytest.approx(-1.0)
    assert result.sample_next_state == "s1"
    assert result.sample_joint_obs == {"0": "o_a", "1": "o_b"}
    assert result.sample_ego_reward == 1.0
    assert result.sample_done is False


def test_estimate_efe_max_info_gain_all_unique_obs():
    """K samples all different obs+state -> H[Q(s'|u)] = log K, weighted_cond_entropy = 0."""
    K = 8
    timesteps = [
        _MockTimestep(
            state=f"s_{i}",
            observations={"0": f"o_{i}", "1": "irrelevant"},
            rewards={"0": 0.0, "1": 0.0},
            terminations={"0": False, "1": False},
            truncations={"0": False, "1": False},
            all_done=False,
        )
        for i in range(K)
    ]
    model = _make_mock_model(timesteps)
    result = estimate_efe(
        model,
        state="s0",
        joint_action={"0": 0, "1": 0},
        ego_agent_id="0",
        K=K,
        rng=random.Random(0),
    )
    assert result.utility_hat == 0.0
    assert result.info_gain_hat == pytest.approx(math.log(K), abs=1e-10)
    assert result.g == pytest.approx(-math.log(K))


def test_estimate_efe_mixed_buckets():
    """Two obs buckets of equal size: each bucket has uniform sub-state distribution.

    6 samples: 3 -> (o_a, s_a/s_b/s_c), 3 -> (o_b, s_d/s_e/s_f).
    H[Q(s'|u)] = log 6 (all 6 next-states unique).
    E_o[H[Q(s'|o,u)]] = 0.5 * log 3 + 0.5 * log 3 = log 3.
    info_gain = log 6 - log 3 = log 2.
    """
    timesteps = [
        _MockTimestep(
            state="s_a", observations={"0": "o_a"},
            rewards={"0": 2.0}, terminations={"0": False},
            truncations={"0": False}, all_done=False,
        ),
        _MockTimestep(
            state="s_b", observations={"0": "o_a"},
            rewards={"0": 2.0}, terminations={"0": False},
            truncations={"0": False}, all_done=False,
        ),
        _MockTimestep(
            state="s_c", observations={"0": "o_a"},
            rewards={"0": 2.0}, terminations={"0": False},
            truncations={"0": False}, all_done=False,
        ),
        _MockTimestep(
            state="s_d", observations={"0": "o_b"},
            rewards={"0": 4.0}, terminations={"0": False},
            truncations={"0": False}, all_done=False,
        ),
        _MockTimestep(
            state="s_e", observations={"0": "o_b"},
            rewards={"0": 4.0}, terminations={"0": False},
            truncations={"0": False}, all_done=False,
        ),
        _MockTimestep(
            state="s_f", observations={"0": "o_b"},
            rewards={"0": 4.0}, terminations={"0": False},
            truncations={"0": False}, all_done=False,
        ),
    ]
    model = _make_mock_model(timesteps)
    result = estimate_efe(
        model,
        state="s0",
        joint_action={"0": 0, "1": 0},
        ego_agent_id="0",
        K=6,
        rng=random.Random(0),
    )
    assert result.utility_hat == pytest.approx(3.0)
    assert result.info_gain_hat == pytest.approx(math.log(2.0), abs=1e-10)
    assert result.g == pytest.approx(-(3.0 + math.log(2.0)))


def test_estimate_efe_done_propagates_from_chosen_sample():
    """sample_done reflects the chosen continuation, not the aggregate."""
    timesteps = [
        _MockTimestep(
            state=f"s_{i}", observations={"0": f"o_{i}"},
            rewards={"0": 0.0}, terminations={"0": i == 0},  # only ts 0 is done
            truncations={"0": False}, all_done=(i == 0),
        )
        for i in range(4)
    ]
    model = _make_mock_model(timesteps)
    # rng will pick index 0 first (with seed 0, Random(0).randrange(4) = 3 on cpython,
    # but we just verify *some* k_star is picked and sample_done matches that index)
    result = estimate_efe(
        model,
        state="s0",
        joint_action={"0": 0},
        ego_agent_id="0",
        K=4,
        rng=random.Random(0),
    )
    chosen_idx = int(result.sample_next_state.split("_")[1])
    assert result.sample_done == (chosen_idx == 0)


def test_estimate_efe_rejects_K_below_2():
    model = MagicMock()
    with pytest.raises(ValueError, match="K must be >= 2"):
        estimate_efe(
            model,
            state="s0",
            joint_action={"0": 0},
            ego_agent_id="0",
            K=1,
            rng=random.Random(0),
        )
    model.step.assert_not_called()
