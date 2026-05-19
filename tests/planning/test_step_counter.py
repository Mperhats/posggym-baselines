"""Tests for StepCountingModel: counts step() calls on a posggym.POSGModel proxy.

See docs/superpowers/specs/2026-05-18-bach-skateboard-design.md §6 (Change 2).
"""

from unittest.mock import MagicMock

import pytest
from posggym_baselines.planning.step_counter import StepCountingModel


def test_step_counter_increments_on_step():
    inner = MagicMock()
    inner.step.return_value = "sentinel_ts"
    wrapped = StepCountingModel(inner)
    assert wrapped.counter == 0
    out = wrapped.step("state", {"0": 1})
    assert out == "sentinel_ts"
    assert wrapped.counter == 1
    wrapped.step("state2", {"0": 2})
    wrapped.step("state3", {"0": 3})
    assert wrapped.counter == 3


def test_step_counter_reset():
    inner = MagicMock()
    inner.step.return_value = None
    wrapped = StepCountingModel(inner)
    for _ in range(5):
        wrapped.step("s", {"0": 0})
    assert wrapped.counter == 5
    wrapped.reset_counter()
    assert wrapped.counter == 0


def test_step_counter_passes_through_attributes():
    inner = MagicMock()
    inner.possible_agents = ["0", "1"]
    inner.discount = 0.95
    inner.action_spaces = {"0": "space0", "1": "space1"}
    wrapped = StepCountingModel(inner)
    assert wrapped.possible_agents == ["0", "1"]
    assert wrapped.discount == 0.95
    assert wrapped.action_spaces == {"0": "space0", "1": "space1"}


def test_step_counter_passes_through_methods():
    inner = MagicMock()
    inner.sample_initial_state.return_value = "init_state"
    wrapped = StepCountingModel(inner)
    assert wrapped.sample_initial_state() == "init_state"
    inner.sample_initial_state.assert_called_once_with()
