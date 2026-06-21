from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import HIMVelocityOnPolicyRunner

from .env_cfgs import (
  boying_him_flat_env_cfg,
  boying_him_rough_env_cfg,
)
from .rl_cfg import boying_him_ppo_runner_cfg

register_mjlab_task(
  task_id="Boying-HIM-Flat",
  env_cfg=boying_him_flat_env_cfg(),
  play_env_cfg=boying_him_flat_env_cfg(play=True),
  rl_cfg=boying_him_ppo_runner_cfg(),
  runner_cls=HIMVelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Boying-HIM-Rough",
  env_cfg=boying_him_rough_env_cfg(),
  play_env_cfg=boying_him_rough_env_cfg(play=True),
  rl_cfg=boying_him_ppo_runner_cfg(),
  runner_cls=HIMVelocityOnPolicyRunner,
)
