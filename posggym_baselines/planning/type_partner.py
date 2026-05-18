"""TypePartnerPolicy: Dirichlet posterior over a finite type set.

Slotted at the bottom of EFEPlanner's depth-k chain (when nesting_level
recursion bottoms at 0). Wraps N posggym.agents.Policy instances and
returns the type-weighted mixture as the partner action distribution.
The Dirichlet posterior updates from observed partner actions on real
env steps (called from EFEPlanner.update).

See docs/superpowers/specs/2026-05-18-bach-skateboard-design.md §5 for
the design rationale and the Driving-v1 type registry.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import posggym.agents
import posggym.model as M
from posggym.agents.policy import PolicyState

from posggym_baselines.planning.other_policy import OtherAgentPolicy


class TypePartnerPolicy(OtherAgentPolicy):
    """Dirichlet posterior over a finite type set of posggym.agents policies.

    The marginal action distribution is the type-weighted mixture. The
    Dirichlet posterior is updated by EFEPlanner.update on observed
    partner actions.

    PolicyState shape for this class: a Dict[str, PolicyState] mapping
    type_policy_id -> the wrapped policy's PolicyState. Each wrapped
    policy advances its own state in parallel.

    Attributes:
        policies: Dict[policy_id, posggym.agents.Policy] -- wrapped types.
        dirichlet_alpha: Dict[policy_id, float] -- current pseudocounts.
    """

    def __init__(
        self,
        model: M.POSGModel,
        agent_id: str,
        type_policy_ids: List[str],
        dirichlet_prior: float = 1.0,
    ):
        super().__init__(model, agent_id)
        if not type_policy_ids:
            raise ValueError("type_policy_ids must be non-empty")
        if dirichlet_prior <= 0:
            raise ValueError(
                f"dirichlet_prior must be > 0 (got {dirichlet_prior})"
            )
        self.policies: Dict[str, posggym.agents.Policy] = {
            pid: posggym.agents.make(pid, model, agent_id)
            for pid in type_policy_ids
        }
        self.dirichlet_alpha: Dict[str, float] = {
            pid: dirichlet_prior for pid in type_policy_ids
        }

    def sample_initial_state(self) -> PolicyState:
        return {pid: p.get_initial_state() for pid, p in self.policies.items()}

    def get_next_state(
        self,
        action: Optional[M.ActType],
        obs: M.ObsType,
        state: PolicyState,
    ) -> PolicyState:
        return {
            pid: self.policies[pid].get_next_state(action, obs, state[pid])
            for pid in self.policies
        }

    def get_pi(self, state: PolicyState) -> Dict[M.ActType, float]:
        """Type-weighted mixture over wrapped policies' action distributions.

        Each wrapped policy.get_pi(state[pid]) returns a Distribution whose
        .probs is an action -> prob dict (per posggym.agents convention,
        mirrored by OtherAgentMixturePolicy.get_pi in other_policy.py:207).
        """
        weights = self._posterior()
        marginal: Dict[M.ActType, float] = defaultdict(float)
        for pid, w in weights.items():
            pi_pid = self.policies[pid].get_pi(state[pid]).probs
            for a, p in pi_pid.items():
                marginal[a] += w * p
        return dict(marginal)

    def sample_action(self, state: PolicyState) -> M.ActType:
        pi = self.get_pi(state)
        actions = list(pi.keys())
        weights = list(pi.values())
        return random.choices(actions, weights=weights, k=1)[0]

    def update_posterior(
        self,
        observed_partner_action: M.ActType,
        prev_state: PolicyState,
    ) -> None:
        """Bayesian Dirichlet update on the observed partner action.

        For each type pid, add the type's likelihood of the observed action
        to dirichlet_alpha[pid]. Called once per real env step from
        EFEPlanner.update, after the real partner action is observed.
        """
        for pid in self.dirichlet_alpha:
            pi = self.policies[pid].get_pi(prev_state[pid]).probs
            self.dirichlet_alpha[pid] += pi.get(observed_partner_action, 0.0)

    def _posterior(self) -> Dict[str, float]:
        total = sum(self.dirichlet_alpha.values())
        return {pid: a / total for pid, a in self.dirichlet_alpha.items()}

    def close(self) -> None:
        for p in self.policies.values():
            if hasattr(p, "close"):
                p.close()
