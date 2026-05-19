"""StepCountingModel: thin proxy around posggym.POSGModel that counts step() calls.

Used by the bach skateboard harness to measure matched-step() budgets across
planners (since MCTSConfig has no simulation-count knob -- search_time_limit
is wall-clock). Per spec §6 Change 2.
"""

from __future__ import annotations

from typing import Any, Dict

import posggym.model as M


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
