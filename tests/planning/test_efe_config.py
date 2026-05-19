"""Tests for EFEConfig (extends MCTSConfig with EFE-specific fields).

See docs/superpowers/specs/2026-05-18-bach-skateboard-design.md §3.
"""

import pytest
from posggym_baselines.planning.config import EFEConfig


def test_efe_config_defaults():
    cfg = EFEConfig(discount=0.95, search_time_limit=1.0, c=1.4, truncated=False)
    assert cfg.gamma_precision == 4.0
    # Empirical default: K=8 wins on Driving-v1 (see config.py docstring).
    # Stochastic envs (PE, LBF) should override to K=32 per Miller-Madow.
    assert cfg.efe_K == 8
    assert cfg.min_visits_per_action == 2
    assert cfg.dirichlet_prior == 1.0
    assert cfg.partner_type_policy_ids == (
        "Driving-v1/A0Shortestpath-v0",
        "Driving-v1/A40Shortestpath-v0",
        "Driving-v1/A60Shortestpath-v0",
        "Driving-v1/A80Shortestpath-v0",
        "Driving-v1/A100Shortestpath-v0",
    )


def test_efe_config_inherits_mctsconfig_fields():
    cfg = EFEConfig(discount=0.95, search_time_limit=2.0, c=1.4, truncated=False)
    # Inherited fields from MCTSConfig
    assert cfg.discount == 0.95
    assert cfg.search_time_limit == 2.0
    # MCTSConfig.__post_init__ derives num_particles = ceil(100 * search_time_limit).
    assert cfg.num_particles == 200


def test_efe_config_rejects_K_below_2():
    with pytest.raises(ValueError, match="efe_K must be >= 2"):
        EFEConfig(
            discount=0.95,
            search_time_limit=1.0,
            c=1.4,
            truncated=False,
            efe_K=1,
        )


def test_efe_config_rejects_non_positive_gamma_precision():
    with pytest.raises(ValueError, match="gamma_precision must be > 0"):
        EFEConfig(
            discount=0.95,
            search_time_limit=1.0,
            c=1.4,
            truncated=False,
            gamma_precision=-1.0,
        )


def test_efe_config_rejects_non_positive_dirichlet_prior():
    with pytest.raises(ValueError, match="dirichlet_prior must be > 0"):
        EFEConfig(
            discount=0.95,
            search_time_limit=1.0,
            c=1.4,
            truncated=False,
            dirichlet_prior=0.0,
        )


def test_efe_config_rejects_empty_type_set():
    with pytest.raises(ValueError, match="partner_type_policy_ids must be non-empty"):
        EFEConfig(
            discount=0.95,
            search_time_limit=1.0,
            c=1.4,
            truncated=False,
            partner_type_policy_ids=(),
        )
