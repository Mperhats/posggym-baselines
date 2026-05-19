"""Integration tests for EFEPlanner -- the four surgical changes from intmcp.py.

These tests validate the EFEPlanner class structure and the depth-k chain
construction. They do NOT run real env steps (smoke tests do that).
"""

from unittest.mock import MagicMock

import gymnasium as gym
import pytest
from posggym_baselines.planning.config import EFEConfig
from posggym_baselines.planning.efe import EFEPlanner
from posggym_baselines.planning.type_partner import TypePartnerPolicy


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.possible_agents = ["0", "1"]
    model.action_spaces = {
        "0": gym.spaces.Discrete(5),
        "1": gym.spaces.Discrete(5),
    }
    model.spec = None
    return model


@pytest.fixture
def fake_posggym_make(monkeypatch):
    """Patch posggym.agents.make to avoid downloading real policy weights."""
    def _fake_make(policy_id, model, agent_id):
        p = MagicMock()
        p.get_initial_state.return_value = {}
        p.get_next_state.return_value = {}
        dist = MagicMock()
        dist.probs = {i: 1.0 / model.action_spaces[agent_id].n for i in range(5)}
        p.get_pi.return_value = dist
        return p

    import posggym.agents
    monkeypatch.setattr(posggym.agents, "make", _fake_make)


def test_efe_planner_initialize_depth0_slots_type_partner(fake_model, fake_posggym_make):
    """At nesting_level=0, the other agent slot is a TypePartnerPolicy."""
    config = EFEConfig(
        discount=0.95, search_time_limit=0.5, c=1.4, truncated=False,
        seed=0, efe_K=4,
    )
    planner = EFEPlanner.initialize(
        model=fake_model,
        ego_agent_id="0",
        config=config,
        nesting_level=0,
        search_policies=None,
    )
    assert planner.nesting_level == 0
    assert "1" in planner.other_agent_policies
    assert isinstance(planner.other_agent_policies["1"], TypePartnerPolicy)
    # 5-element default Driving-v1 type set
    assert len(planner.other_agent_policies["1"].policies) == 5


def test_efe_planner_initialize_depth2_chains_correctly(fake_model, fake_posggym_make):
    """At nesting_level=2: ego -> EFEPlanner(k=1) -> EFEPlanner(k=0) -> TypePartnerPolicy."""
    config = EFEConfig(
        discount=0.95, search_time_limit=0.5, c=1.4, truncated=False,
        seed=0, efe_K=4,
    )
    planner = EFEPlanner.initialize(
        model=fake_model,
        ego_agent_id="0",
        config=config,
        nesting_level=2,
        search_policies=None,
    )
    assert planner.nesting_level == 2
    level_1 = planner.other_agent_policies["1"]
    assert isinstance(level_1, EFEPlanner)
    assert level_1.nesting_level == 1
    level_0 = level_1.other_agent_policies["0"]
    assert isinstance(level_0, EFEPlanner)
    assert level_0.nesting_level == 0
    bottom = level_0.other_agent_policies["1"]
    assert isinstance(bottom, TypePartnerPolicy)


def test_argmin_g_action_selection_picks_smallest_value(fake_model, fake_posggym_make):
    """argmin_g_action_selection picks the action_node with the smallest .value (= best G)."""
    config = EFEConfig(
        discount=0.95, search_time_limit=0.5, c=1.4, truncated=False,
        seed=0, efe_K=4,
    )
    planner = EFEPlanner.initialize(
        model=fake_model, ego_agent_id="0",
        config=config, nesting_level=0, search_policies=None,
    )

    obs_node = MagicMock()
    obs_node.children = [object(), object(), object()]  # non-empty
    child_nodes = [
        MagicMock(action=0, value=-2.5),  # best (smallest G)
        MagicMock(action=1, value=1.0),
        MagicMock(action=2, value=-0.5),
    ]
    obs_node.get_child_nodes.return_value = child_nodes

    result = planner.argmin_g_action_selection(obs_node)
    assert result == 0


def test_softmax_action_selection_respects_warmup(fake_model, fake_posggym_make):
    """During warmup (visits < min_visits_per_action), any low-visit action wins."""
    config = EFEConfig(
        discount=0.95, search_time_limit=0.5, c=1.4, truncated=False,
        seed=0, efe_K=4, min_visits_per_action=3,
    )
    planner = EFEPlanner.initialize(
        model=fake_model, ego_agent_id="0",
        config=config, nesting_level=0, search_policies=None,
    )

    obs_node = MagicMock()
    obs_node.visits = 5
    child_nodes = [
        MagicMock(action=0, visits=5, value=-2.5),
        MagicMock(action=1, visits=1, value=10.0),  # below warmup threshold
        MagicMock(action=2, visits=5, value=-0.5),
    ]
    obs_node.get_child_nodes.return_value = child_nodes
    result = planner.softmax_action_selection(obs_node)
    assert result == 1  # warmup wins despite high G
