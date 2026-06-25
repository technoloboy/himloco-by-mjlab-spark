"""HIM PPO: standard rsl_rl PPO + an auxiliary state-estimator update.

The estimator lives inside the HIM actor model (``self.actor.estimator``) and is
optimized independently (its own Adam, its own loss). Because the estimator's
output is detached inside the actor, the PPO graph and the estimator graph never
share gradients; the only coupling is the (optional) adaptive learning rate,
which we forward to the estimator to mirror HIMLoco's behavior.

Data flow for the estimator supervision (per transition t):
  - encoder input  : flattened proprio history at t     -> observations[history_group]
  - target frame   : single proprio frame at t+1        -> next_observations[current_group]
  - velocity target: true base lin vel at t+1           -> next_observations[vel_group]
  - validity mask  : transitions that did NOT terminate -> (1 - dones)

We run the estimator pass BEFORE ``super().update()`` because the base update
clears the rollout storage at the end.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from .him_storage import HIMRolloutStorage


class HIMPPO(PPO):
  def __init__(
    self,
    *args,
    estimator_vel_group: str = "estimator_vel",
    estimator_history_group: str = "proprio_history",
    estimator_current_group: str = "proprio_current",
    **kwargs,
  ) -> None:
    self.estimator_vel_group = estimator_vel_group
    self.estimator_history_group = estimator_history_group
    self.estimator_current_group = estimator_current_group
    super().__init__(*args, **kwargs)
    # The estimator is owned by the actor model.
    self.estimator = self.actor.estimator

  # Rollout ---------------------------------------------------------------

  def process_env_step(
    self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
  ) -> None:
    # ``obs`` here is s_{t+1}. Record it as the transition's next observation.
    self.transition.next_observations = obs
    super().process_env_step(obs, rewards, dones, extras)

  # Update ----------------------------------------------------------------

  def _update_estimator(self) -> tuple[float, float]:
    history_group = self.estimator_history_group
    current_group = self.estimator_current_group
    vel_group = self.estimator_vel_group

    mean_est, mean_swap, n = 0.0, 0.0, 0
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
      obs_history = batch.observations[history_group]
      next_obs = batch.next_observations
      next_frame = next_obs[current_group]
      next_vel = next_obs[vel_group]
      valid_mask = (1.0 - batch.dones.float()).view(-1)

      est_loss, swap_loss = self.estimator.update(
        obs_history=obs_history,
        next_obs_frame=next_frame,
        next_vel=next_vel,
        valid_mask=valid_mask,
        lr=self.learning_rate,
      )
      mean_est += est_loss
      mean_swap += swap_loss
      n += 1

    if n > 0:
      mean_est /= n
      mean_swap /= n
    return mean_est, mean_swap

  def compute_returns(self, obs) -> None:
    super().compute_returns(obs)
    # Clamp returns to prevent a single high-bootstrap episode from corrupting
    # the critic's EmpiricalNormalization via a catastrophic value loss spike.
    # Normal steady-state return ~ 4-5; absolute worst-case ~ 12.
    # ±50 was too loose (spike returns were ~18, within range and not clamped).
    # ±10 catches the observed spikes while preserving all legitimate returns.
    self.storage.returns.clamp_(-10.0, 10.0)

  def update(self) -> dict[str, float]:
    est_loss, swap_loss = self._update_estimator()
    loss_dict = super().update()
    loss_dict["estimation"] = est_loss
    loss_dict["swap"] = swap_loss
    return loss_dict

  # Save / load -----------------------------------------------------------

  def save(self) -> dict:
    saved = super().save()
    saved["estimator_optimizer_state_dict"] = self.estimator.optimizer.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    out = super().load(loaded_dict, load_cfg, strict)
    # Estimator weights ride along inside actor_state_dict (estimator is a
    # submodule of the actor). Only its optimizer needs explicit restore.
    if load_cfg is None or load_cfg.get("optimizer", True):
      if "estimator_optimizer_state_dict" in loaded_dict:
        self.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
    return out

  # Construction ----------------------------------------------------------

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "HIMPPO":
    """Mirror PPO.construct_algorithm but use HIMRolloutStorage.

    Resolves actor/critic/algorithm classes via ``class_name`` exactly like the
    base, so the HIM actor model and HIMPPO algorithm are selected from config.
    """
    alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))

    default_sets = ["actor", "critic"]
    if cfg["algorithm"].get("rnd_cfg") is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns
    critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
    print(f"Critic Model: {critic}")

    storage = HIMRolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )

    alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
    return alg
