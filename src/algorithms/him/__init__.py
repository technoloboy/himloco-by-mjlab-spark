"""HIM (Hybrid Internal Model) algorithm package.

Exposes the estimator, actor model, PPO variant, storage, and config dataclasses.
All classes are resolvable by dotted ``class_name`` strings from task configs, so
the installed ``rsl_rl`` / ``mjlab`` packages remain unmodified.
"""

from .estimator import HIMEstimator
from .him_actor import HIMActorModel, HIMActorModelCfg
from .him_ppo import HIMPPO
from .him_storage import HIMRolloutStorage
from .config import HIMPpoAlgorithmCfg

__all__ = [
  "HIMEstimator",
  "HIMActorModel",
  "HIMActorModelCfg",
  "HIMPPO",
  "HIMRolloutStorage",
  "HIMPpoAlgorithmCfg",
]
