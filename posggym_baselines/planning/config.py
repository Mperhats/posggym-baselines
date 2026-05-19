import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from posggym_baselines.planning.utils import KnownBounds


@dataclass
class MCTSConfig:
    """Configuration for MCTS based algorithms."""

    discount: float
    search_time_limit: float
    c: float
    truncated: bool
    action_selection: str = "pucb"
    pucb_exploration_fraction: float = 0.5
    known_bounds: Optional[KnownBounds] = None
    extra_particles_prop: float = 1.0 / 16
    reinvigoration_sample_limit_factor: float = 4.0
    step_limit: Optional[int] = None
    epsilon: float = 0.01
    seed: Optional[int] = None
    state_belief_only: bool = False
    # if `truncated` is True, and search policy has no value function, then
    # use rollout, otherwise exception is thrown
    use_rollout_if_no_value: bool = True

    num_particles: int = field(init=False)
    extra_particles: int = field(init=False)
    depth_limit: int = field(init=False)

    def __post_init__(self):
        assert self.discount >= 0.0 and self.discount <= 1.0
        assert self.search_time_limit > 0.0
        assert self.c > 0.0
        assert (
            self.pucb_exploration_fraction >= 0.0
            and self.pucb_exploration_fraction <= 1.0
        )
        assert self.extra_particles_prop >= 0.0 and self.extra_particles_prop <= 1.0
        assert self.epsilon > 0.0 and self.epsilon < 1.0

        self.action_selection = self.action_selection.lower()
        assert self.action_selection in ["pucb", "ucb", "uniform"]

        self.num_particles = math.ceil(100 * self.search_time_limit)
        self.extra_particles = math.ceil(self.num_particles * self.extra_particles_prop)

        if self.discount == 0.0:
            self.depth_limit = 0
        else:
            self.depth_limit = math.ceil(
                math.log(self.epsilon) / math.log(self.discount)
            )


_DEFAULT_DRIVING_TYPE_IDS: Tuple[str, ...] = (
    "Driving-v1/A0Shortestpath-v0",
    "Driving-v1/A40Shortestpath-v0",
    "Driving-v1/A60Shortestpath-v0",
    "Driving-v1/A80Shortestpath-v0",
    "Driving-v1/A100Shortestpath-v0",
)


@dataclass
class EFEConfig(MCTSConfig):
    """Configuration for the bach particle-EFE planner (EFEPlanner).

    Extends MCTSConfig with five EFE-specific fields. Inherits all MCTSConfig
    fields (discount, search_time_limit, c, truncated, action_selection, etc.).

    Note: MCTSConfig.action_selection is ignored by EFEPlanner (always softmax);
    MCTSConfig.c is unused (no UCB term). Both kept for forward compatibility
    and harness uniformity.

    See docs/superpowers/specs/2026-05-18-bach-skateboard-design.md §3 for the
    rationale on field defaults (in particular K=32 per Miller-Madow bias).
    """

    gamma_precision: float = 4.0
    # Empirical default for Driving-v1: K=8 outperforms K=32 because the env
    # is near-deterministic (env.step's rng.shuffle is the only stochasticity),
    # so K samples from the same (state, joint_action) mostly produce the same
    # (next_state, obs). info_gain_hat is ~0 across actions regardless of K, so
    # the lower per-node cost of K=8 (4x more simulations per second wall-clock)
    # wins net. Empirical: K=8 gives +0.90 mean vs random; K=32 gives +0.81.
    # The spec's K=32 default (per Miller-Madow bias) is still recommended for
    # stochastic envs (PursuitEvasion, LBF) where info_gain_hat actually has
    # signal -- bump K back up there.
    efe_K: int = 8
    min_visits_per_action: int = 2
    partner_type_policy_ids: Tuple[str, ...] = _DEFAULT_DRIVING_TYPE_IDS
    dirichlet_prior: float = 1.0
    # Multiplier on info_gain_hat before adding to utility_hat in
    # estimate_efe. Default 1.0 = standard EFE. Set to 0.0 for a
    # pragmatic-only ablation (G = -utility, no epistemic value); useful
    # both as a diagnostic and as a sweep axis for C3.4 (epistemic term
    # contribution per END.md).
    info_gain_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.efe_K < 2:
            raise ValueError(f"efe_K must be >= 2 (got {self.efe_K})")
        if self.gamma_precision <= 0:
            raise ValueError(
                f"gamma_precision must be > 0 (got {self.gamma_precision})"
            )
        if self.dirichlet_prior <= 0:
            raise ValueError(
                f"dirichlet_prior must be > 0 (got {self.dirichlet_prior})"
            )
        if not self.partner_type_policy_ids:
            raise ValueError("partner_type_policy_ids must be non-empty")
        if self.info_gain_weight < 0:
            raise ValueError(
                f"info_gain_weight must be >= 0 (got {self.info_gain_weight})"
            )
