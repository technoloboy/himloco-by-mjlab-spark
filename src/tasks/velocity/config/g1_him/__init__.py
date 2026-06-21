from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import HIMVelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_him_flat_env_cfg,
  unitree_g1_him_rough_env_cfg,
)
from .rl_cfg import unitree_g1_him_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-HIM-Flat",
  env_cfg=unitree_g1_him_flat_env_cfg(),
  play_env_cfg=unitree_g1_him_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_him_ppo_runner_cfg(),
  runner_cls=HIMVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-HIM-Rough",
  env_cfg=unitree_g1_him_rough_env_cfg(),
  play_env_cfg=unitree_g1_him_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_him_ppo_runner_cfg(),
  runner_cls=HIMVelocityOnPolicyRunner,
)
