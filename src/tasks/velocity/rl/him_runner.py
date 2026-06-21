"""On-policy runner for HIM velocity tasks.

The stock ``VelocityOnPolicyRunner`` (and ``get_base_metadata``) assume a single
flattened ``"actor"`` observation group. The HIM policy instead needs both the
current single-step frame and the flattened history at deploy time, and its
observation groups are named ``proprio_current`` / ``proprio_history``. This
runner therefore:

- exports the HIM-aware ONNX (input = concat[current(S), history(H*S)]), and
- builds metadata whose ``observation_names`` reflect the HIM input layout,
  avoiding the hardcoded ``active_terms["actor"]`` lookup in the base helper.
"""

from __future__ import annotations

import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class HIMVelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def _him_metadata(self, run_name: str) -> dict:
    env = self.env.unwrapped
    # Reuse the base metadata but override observation_names, since HIM does not
    # have an "actor" group (its actor input is assembled inside the model).
    obs_mgr = env.observation_manager
    # Temporarily synthesize an "actor" entry for the base helper if missing.
    active = obs_mgr.active_terms
    had_actor = "actor" in active
    if not had_actor:
      # proprio_current holds the canonical single-step term order.
      active["actor"] = active.get("proprio_current", [])
    try:
      metadata = get_base_metadata(env, run_name)
    finally:
      if not had_actor:
        active.pop("actor", None)
    metadata["him_history_length"] = self.alg.actor._him_history_length
    metadata["him_num_one_step_obs"] = self.alg.actor.num_one_step_obs
    metadata["him_latent_dim"] = self.alg.actor._him_latent_dim
    metadata["him_input_layout"] = "concat[current(S), history(H*S)]"
    return metadata

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )
    onnx_path = os.path.join(policy_path, filename)
    metadata = self._him_metadata(run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
