"""Unitree G1 HIM (Hybrid Internal Model) velocity environment configuration.

Reuses the existing G1 *flat* velocity environment (pure proprioception, no
height scan) and restructures its observation groups for the HIM algorithm:

- ``proprio_history``: flattened 6-frame history of the proprioceptive terms
  (estimator encoder input). enable_corruption=True (noisy, as on hardware).
- ``proprio_current``: the SAME proprioceptive terms, single frame, clean
  (estimator target network input + the policy's current frame).
- ``estimator_vel`` : true base linear velocity (clean), the estimator's
  velocity-regression target.
- ``critic``        : privileged critic observations (unchanged from base).

The actor's MLP input is built inside HIMActorModel as
``[proprio_current(S) | est_vel(3) | latent(L)]``.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import src.tasks.velocity.mdp as mdp
from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

HISTORY_LENGTH = 6


def _proprio_terms() -> dict[str, ObservationTermCfg]:
  """Proprioceptive single-step terms (IMU + joint encoders + command + phase).

  Mirrors HIMLoco's actor observation: angular velocity, projected gravity,
  velocity command, gait phase, joint positions/velocities, and last action.
  Noise scales match the base velocity task's actor terms.
  """
  return {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def unitree_g1_him_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Preserve the base critic group (privileged obs); rebuild actor-side groups.
  critic_group = cfg.observations["critic"]

  proprio_history = ObservationGroupCfg(
    terms=_proprio_terms(),
    concatenate_terms=True,
    enable_corruption=not play,
    history_length=HISTORY_LENGTH,
    flatten_history_dim=True,
  )
  proprio_current = ObservationGroupCfg(
    terms=_proprio_terms(),
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
    flatten_history_dim=True,
  )
  estimator_vel = ObservationGroupCfg(
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
  )

  cfg.observations = {
    "proprio_history": proprio_history,
    "proprio_current": proprio_current,
    "estimator_vel": estimator_vel,
    "critic": critic_group,
  }
  return cfg


def unitree_g1_him_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough-terrain HIM variant.

  HIM is a blind (proprioception-only) policy, so we keep the same blind actor
  groups even on rough terrain; only the critic retains privileged terms
  (including height_scan, which the base rough critic provides).
  """
  from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

  cfg = unitree_g1_rough_env_cfg(play=play)
  critic_group = cfg.observations["critic"]

  cfg.observations = {
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
  return cfg
