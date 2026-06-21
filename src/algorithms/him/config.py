"""Config dataclasses for the HIM algorithm.

These subclass the mjlab RSL-RL config dataclasses and only set/extend the
fields needed to select the HIM classes via ``class_name`` and to pass the
HIM-specific ``estimator_vel_group`` through to ``HIMPPO``.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class HIMPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO algorithm cfg that selects HIMPPO and names the HIM obs groups."""

  class_name: str = "src.algorithms.him.him_ppo.HIMPPO"
  estimator_vel_group: str = "estimator_vel"
  estimator_history_group: str = "proprio_history"
  estimator_current_group: str = "proprio_current"
