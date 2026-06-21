"""RL configuration for the Unitree G1 HIM velocity task.

Same PPO hyperparameters as the base G1 velocity task, but:
- the actor is the HIM actor model (estimator-augmented MLP),
- the algorithm is HIMPPO (PPO + estimator update),
- obs_groups route the right observation groups to actor/critic.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from src.algorithms.him import HIMActorModelCfg, HIMPpoAlgorithmCfg


def unitree_g1_him_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=HIMActorModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,  # HIM consumes raw history.
      history_length=6,
      history_group="proprio_history",
      current_group="proprio_current",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=HIMPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      estimator_vel_group="estimator_vel",
      estimator_history_group="proprio_history",
      estimator_current_group="proprio_current",
    ),
    obs_groups={
      "actor": ("proprio_history",),
      "critic": ("critic",),
    },
    experiment_name="g1_him_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
