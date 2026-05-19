"""Tests for TypePartnerPolicy: Dirichlet posterior over a finite type set.

See docs/superpowers/specs/2026-05-18-bach-skateboard-design.md §5.
"""

from typing import Dict
from unittest.mock import MagicMock

import pytest
from posggym_baselines.planning.type_partner import TypePartnerPolicy


def _make_mock_policy(name: str, action_distribution: Dict[int, float]):
    """Build a mock posggym.agents.Policy with a fixed action distribution.

    Mirrors the real Policy interface: get_pi(state) returns a Distribution
    object whose `.probs` is the action -> prob dict (see how
    OtherAgentMixturePolicy.get_pi consumes it in other_policy.py).
    """
    p = MagicMock()
    p.get_initial_state.return_value = f"state_of_{name}"
    p.get_next_state.side_effect = lambda action, obs, state: f"{state}_advanced"
    dist = MagicMock()
    dist.probs = dict(action_distribution)
    p.get_pi.return_value = dist
    return p


@pytest.fixture
def two_type_partner(monkeypatch):
    p_a = _make_mock_policy("A", {0: 0.7, 1: 0.3})
    p_b = _make_mock_policy("B", {0: 0.2, 1: 0.8})

    def fake_make(policy_id, model, agent_id):
        return {"type_A": p_a, "type_B": p_b}[policy_id]

    import posggym.agents
    monkeypatch.setattr(posggym.agents, "make", fake_make)

    model = MagicMock()
    return TypePartnerPolicy(
        model=model,
        agent_id="1",
        type_policy_ids=["type_A", "type_B"],
        dirichlet_prior=1.0,
    ), p_a, p_b


def test_initial_posterior_uniform(two_type_partner):
    partner, _, _ = two_type_partner
    posterior = partner._posterior()
    assert posterior == {"type_A": 0.5, "type_B": 0.5}


def test_sample_initial_state_returns_per_type_states(two_type_partner):
    partner, _, _ = two_type_partner
    state = partner.sample_initial_state()
    assert state == {"type_A": "state_of_A", "type_B": "state_of_B"}


def test_get_pi_is_dirichlet_weighted_mixture(two_type_partner):
    partner, _, _ = two_type_partner
    state = partner.sample_initial_state()
    pi = partner.get_pi(state)
    # Equal weights: pi[0] = 0.5 * 0.7 + 0.5 * 0.2 = 0.45
    #                pi[1] = 0.5 * 0.3 + 0.5 * 0.8 = 0.55
    assert pi[0] == pytest.approx(0.45)
    assert pi[1] == pytest.approx(0.55)


def test_update_posterior_shifts_toward_compatible_type(two_type_partner):
    """Partner takes action 1 (B more likely to do that): posterior shifts to B."""
    partner, _, _ = two_type_partner
    prev_state = partner.sample_initial_state()
    # observed action = 1; pi_A(1)=0.3, pi_B(1)=0.8
    partner.update_posterior(observed_partner_action=1, prev_state=prev_state)
    posterior = partner._posterior()
    # dirichlet_alpha[type_A] = 1.0 + 0.3 = 1.3
    # dirichlet_alpha[type_B] = 1.0 + 0.8 = 1.8
    # posterior[type_A] = 1.3 / 3.1, posterior[type_B] = 1.8 / 3.1
    assert posterior["type_B"] > posterior["type_A"]
    assert posterior["type_A"] == pytest.approx(1.3 / 3.1)
    assert posterior["type_B"] == pytest.approx(1.8 / 3.1)


def test_get_next_state_advances_all_wrapped_policies(two_type_partner):
    partner, p_a, p_b = two_type_partner
    state = partner.sample_initial_state()
    new_state = partner.get_next_state(action=0, obs="some_obs", state=state)
    assert new_state == {
        "type_A": "state_of_A_advanced",
        "type_B": "state_of_B_advanced",
    }
    p_a.get_next_state.assert_called_once()
    p_b.get_next_state.assert_called_once()


def test_sample_action_uses_mixture_distribution(two_type_partner):
    """sample_action draws from the type-mixture get_pi(); over many samples
    the empirical distribution should converge to the mixture."""
    import random

    random.seed(0)
    partner, _, _ = two_type_partner
    state = partner.sample_initial_state()

    counts = {0: 0, 1: 0}
    for _ in range(2000):
        a = partner.sample_action(state)
        counts[a] += 1

    # Expected: ~45% action 0, ~55% action 1. Allow 3% absolute slack.
    assert abs(counts[0] / 2000 - 0.45) < 0.03
    assert abs(counts[1] / 2000 - 0.55) < 0.03


def test_reset_restores_uniform_prior(two_type_partner):
    """After observed actions shift the posterior, reset() restores the prior."""
    partner, _, _ = two_type_partner
    prev_state = partner.sample_initial_state()
    for _ in range(5):
        partner.update_posterior(observed_partner_action=1, prev_state=prev_state)
    # Posterior is now skewed toward B
    posterior_before = partner._posterior()
    assert posterior_before["type_B"] > 0.6
    # Reset
    partner.reset()
    posterior_after = partner._posterior()
    assert posterior_after == {"type_A": 0.5, "type_B": 0.5}


def test_rejects_empty_type_set():
    with pytest.raises(ValueError, match="type_policy_ids must be non-empty"):
        TypePartnerPolicy(
            model=MagicMock(),
            agent_id="1",
            type_policy_ids=[],
            dirichlet_prior=1.0,
        )


def test_rejects_non_positive_dirichlet_prior(monkeypatch):
    monkeypatch.setattr(
        "posggym.agents.make", lambda *args, **kwargs: MagicMock()
    )
    with pytest.raises(ValueError, match="dirichlet_prior must be > 0"):
        TypePartnerPolicy(
            model=MagicMock(),
            agent_id="1",
            type_policy_ids=["type_A"],
            dirichlet_prior=0.0,
        )
