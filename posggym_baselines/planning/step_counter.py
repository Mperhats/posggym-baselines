"""StepCountingModel: thin proxy around posggym.POSGModel that counts step() calls.

Used by the bach skateboard harness to measure matched-step() budgets across
planners (since MCTSConfig has no simulation-count knob -- search_time_limit
is wall-clock). Per spec §6 Change 2.

StepBudgetModel is the strict-budget variant that raises StepBudgetExhausted
when the per-decision budget is exceeded. Used by smoke 5 / smoke 6 (the
matched-step() benchmarks vs INTMCP/POTMMCP).
"""

from __future__ import annotations

from typing import Any, Dict

import posggym.model as M


class StepBudgetExhausted(Exception):
    """Raised by StepBudgetModel.step when the per-decision budget is hit.

    Planner get_action loops catch this and return their best-found action
    so far, mimicking a wall-clock budget exhaustion. The exception is part
    of the harness's matched-step() benchmarking contract.
    """


class StepCountingModel:
    """Proxy: counts step() calls; passes everything else through to inner.

    Usage:
        env = posggym.make(...)
        wrapped = StepCountingModel(env.model)
        planner = SomePlanner(model=wrapped, ...)
        # ... run an episode ...
        n_steps = wrapped.counter
        wrapped.reset_counter()  # reset between episodes
    """

    def __init__(self, inner: M.POSGModel) -> None:
        self._inner = inner
        self.counter = 0

    def step(self, state: Any, joint_action: Dict[str, M.ActType]) -> Any:
        self.counter += 1
        return self._inner.step(state, joint_action)

    def reset_counter(self) -> None:
        self.counter = 0

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal attribute lookup fails, so
        # self.counter / self.step / self.reset_counter / self._inner are
        # found via the instance dict before this method is consulted.
        return getattr(self._inner, name)


class StepBudgetModel(StepCountingModel):
    """Like StepCountingModel but raises StepBudgetExhausted when the
    per-decision budget is exceeded. Used to enforce matched-step()
    comparisons across planners that all loop on wall-clock budgets.

    Usage:
        wrapped = StepBudgetModel(env.model, budget_per_decision=20_000)
        planner = SomePlanner(model=wrapped, ...)
        for env_step in range(50):
            wrapped.reset_decision()           # fresh budget per decision
            try:
                action = planner.step(obs)
            except StepBudgetExhausted:
                # Planner should already have caught this internally; if not,
                # fall back to a random action.
                action = ...
    """

    def __init__(self, inner: M.POSGModel, budget_per_decision: int) -> None:
        super().__init__(inner)
        self.budget_per_decision = budget_per_decision
        self._this_decision_count = 0

    def reset_decision(self) -> None:
        self._this_decision_count = 0

    def step(self, state: Any, joint_action: Dict[str, M.ActType]) -> Any:
        if self._this_decision_count >= self.budget_per_decision:
            raise StepBudgetExhausted(
                f"per-decision budget {self.budget_per_decision} exceeded"
            )
        self._this_decision_count += 1
        return super().step(state, joint_action)
