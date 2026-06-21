"""HIM actor model: an MLPModel whose input is augmented by a state estimator.

mjlab flattens an observation group's history **term-major**
(``[A_t0,A_t1,...,A_t{H-1}, B_t0,...]``), and rsl_rl's MLPModel only accepts 2D
observation groups. So we cannot slice "the newest single-step frame" out of a
flattened history group. Instead we use THREE cooperating observation groups:

- ``history_group``  : flattened proprio history ``[B, H*S]`` -> estimator encoder
  (the encoder is a dense MLP, so the term-major ordering is irrelevant as long
  as it is consistent).
- ``current_group``  : single proprio frame ``[B, S]`` -> the policy's current
  observation (concatenated with the estimator outputs).
- (velocity target lives in a separate group consumed by the algorithm, not the
  model.)

The policy MLP consumes ``[ current_frame(S) | est_vel(3) | latent(L) ]``,
mirroring HIMLoco's HIMActorCritic.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mjlab.rl import RslRlModelCfg
from rsl_rl.models import MLPModel

from .estimator import HIMEstimator

_DEFAULT_ESTIMATOR: dict = {
  "enc_hidden_dims": (128, 64, 16),
  "tar_hidden_dims": (128, 64),
  "activation": "elu",
  "learning_rate": 1.0e-3,
  "max_grad_norm": 10.0,
  "num_prototype": 32,
  "temperature": 3.0,
}


class HIMActorModel(MLPModel):
  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims=(512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    history_length: int = 6,
    history_group: str = "proprio_history",
    current_group: str = "proprio_current",
    estimator: dict | None = None,
  ) -> None:
    self._him_history_length = int(history_length)
    self._him_history_group = history_group
    self._him_current_group = current_group
    self._him_estimator_cfg = {**_DEFAULT_ESTIMATOR, **(estimator or {})}
    self._him_latent_dim = int(self._him_estimator_cfg["enc_hidden_dims"][-1])

    # Single-step dim from the dedicated current-frame group.
    if current_group not in obs:
      raise KeyError(
        f"HIMActorModel current_group '{current_group}' not in observations "
        f"{list(obs.keys())}."
      )
    self.num_one_step_obs = int(obs[current_group].shape[-1])

    if obs_normalization:
      print("[HIMActorModel] obs_normalization forced to False (HIM uses raw history).")
      obs_normalization = False

    # MLPModel sizes its head from self._get_latent_dim(); the params above are
    # already set so that call resolves correctly.
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    # self.obs_dim is the history group dim (obs_groups["actor"] == [history_group]).
    expected = self.num_one_step_obs * self._him_history_length
    if self.obs_dim != expected:
      raise ValueError(
        f"History group '{history_group}' dim ({self.obs_dim}) != "
        f"num_one_step_obs ({self.num_one_step_obs}) * history_length "
        f"({self._him_history_length}) = {expected}. Ensure obs_groups['actor'] "
        f"== ['{history_group}'] and that group uses history_length="
        f"{self._him_history_length}, flatten_history_dim=True."
      )

    self.estimator = HIMEstimator(
      temporal_steps=self._him_history_length,
      num_one_step_obs=self.num_one_step_obs,
      **self._him_estimator_cfg,
    )

  def _get_latent_dim(self) -> int:
    return self.num_one_step_obs + 3 + self._him_latent_dim

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    history = obs[self._him_history_group]
    current = obs[self._him_current_group]
    vel, latent = self.estimator(history)  # detached inside estimator
    return torch.cat((current, vel, latent), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:  # normalization disabled
    return

  def as_jit(self) -> nn.Module:
    return _HIMExportModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    del verbose
    return _HIMExportModel(self)


class _HIMExportModel(nn.Module):
  """Deploy module. Input = concat([current_frame(S), history(H*S)]).

  forward: split -> encoder(history) -> [current | vel | latent] -> mlp -> action.
  The deploy code must provide both the newest single-step frame and the
  flattened history (term-major, same as training) concatenated in that order.
  """

  is_recurrent: bool = False

  def __init__(self, model: HIMActorModel) -> None:
    super().__init__()
    self.encoder = copy.deepcopy(model.estimator.encoder)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.num_one_step_obs = model.num_one_step_obs
    self.latent_dim = model._him_latent_dim
    self.history_dim = model.obs_dim
    self.input_size = model.num_one_step_obs + model.obs_dim  # current + history

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    S = self.num_one_step_obs
    current = x[..., :S]
    history = x[..., S:]
    parts = self.encoder(history)
    vel = parts[..., :3]
    z = F.normalize(parts[..., 3 : 3 + self.latent_dim], dim=-1, p=2.0)
    out = self.mlp(torch.cat((current, vel, z), dim=-1))
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass


@dataclass
class HIMActorModelCfg(RslRlModelCfg):
  """Config for the HIM actor model."""

  class_name: str = "src.algorithms.him.him_actor.HIMActorModel"
  history_length: int = 6
  history_group: str = "proprio_history"
  current_group: str = "proprio_current"
  estimator: dict = field(default_factory=lambda: dict(_DEFAULT_ESTIMATOR))
