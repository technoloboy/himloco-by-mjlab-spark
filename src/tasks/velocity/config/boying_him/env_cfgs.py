"""Boying HIM velocity environment configurations.

Mirrors HIMLoco Go1 obs layout with scale parity.

Single-step proprio dim S depends on MJLAB_PHASE_ENABLED:
  - phase OFF (default): S = 47
      ang_vel(3,×0.25) + gravity(3) + command(3,scaled[2,2,0.25])
    + joint_pos(12) + joint_vel(12,×0.05) + actions(12) + lin_vel(3,×2.0) = 47
  - phase ON: S = 49 (add phase(2))

History T = 6. proprio_history = T × S, proprio_current = S, estimator_vel = 3.
"""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.config.boying.env_cfgs import (
  boying_flat_env_cfg,
  boying_rough_env_cfg,
)

HISTORY_LENGTH = 6
PHASE_ENABLED = os.environ.get("MJLAB_PHASE_ENABLED", "0") == "1"


def _proprio_terms() -> dict[str, ObservationTermCfg]:
  """Boying proprioceptive single-step terms aligned to HIMLoco Go1 obs_scales.

  Noise values are final (noise_level × noise_scale × obs_scale):
    ang_vel: 1.0 × 0.2 × 0.25 = 0.05
    gravity: 1.0 × 0.05 × 1.0 = 0.05
    dof_pos: 1.0 × 0.01 × 1.0 = 0.01
    dof_vel: 1.0 × 1.5  × 0.05 = 0.075
    lin_vel: 1.0 × 0.1  × 2.0  = 0.2
  """
  terms: dict[str, ObservationTermCfg] = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      scale=0.25,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      scale=1.0,
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands_scaled,
      params={"command_name": "twist", "scale": (2.0, 2.0, 0.25)},
    ),
  }
  if PHASE_ENABLED:
    terms["phase"] = ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.5, "command_name": "twist"},
    )
  terms.update({
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      scale=1.0,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.075, n_max=0.075),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=2.0,
    ),
  })
  return terms


def _him_observations(cfg: ManagerBasedRlEnvCfg, play: bool) -> dict:
  """Build the four HIM observation groups, preserving the base critic group."""
  critic_group = cfg.observations["critic"]
  return {
    "proprio_history": ObservationGroupCfg(
      terms=_proprio_terms(),
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "proprio_current": ObservationGroupCfg(
      terms=_proprio_terms(),
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
    "estimator_vel": ObservationGroupCfg(
      terms={
        "base_lin_vel": ObservationTermCfg(
          func=mdp.builtin_sensor,
          params={"sensor_name": "robot/imu_lin_vel"},
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
    "critic": critic_group,
  }


def boying_him_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Boying flat terrain HIM velocity configuration."""
  cfg = boying_flat_env_cfg(play=play)
  cfg.observations = _him_observations(cfg, play)
  return cfg


def boying_him_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Boying rough terrain HIM velocity configuration.

  The actor remains blind (no height scan). The critic retains privileged
  observations including height_scan from the base rough config.
  """
  cfg = boying_rough_env_cfg(play=play)
  cfg.observations = _him_observations(cfg, play)
  return cfg
