"""Rollout storage variant that also records next-step observations.

The HIM estimator is trained to predict the *successor* state's latent and the
true next-step base linear velocity. The stock ``rsl_rl.storage.RolloutStorage``
only records observations at step ``t`` (and ``dones``), so we extend it to also
record the observation at ``t+1`` for every transition.

We deliberately store the *full* next observation TensorDict (all groups). This
keeps the estimator decoupled from any particular group layout: the algorithm
selects whichever groups it needs (e.g. the next single-step proprio frame and
the true next base linear velocity) at update time.
"""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class HIMRolloutStorage(RolloutStorage):
  """RolloutStorage + per-transition next observations."""

  class Transition(RolloutStorage.Transition):
    def __init__(self) -> None:
      super().__init__()
      self.next_observations: TensorDict | None = None

    def clear(self) -> None:
      self.__init__()

  def __init__(
    self,
    training_type: str,
    num_envs: int,
    num_transitions_per_env: int,
    obs: TensorDict,
    actions_shape,
    device: str = "cpu",
  ) -> None:
    super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
    # Mirror the observation buffer layout for next observations.
    self.next_observations = TensorDict(
      {
        key: torch.zeros(num_transitions_per_env, *value.shape, device=device)
        for key, value in obs.items()
      },
      batch_size=[num_transitions_per_env, num_envs],
      device=self.device,
    )

  def add_transition(self, transition: "HIMRolloutStorage.Transition") -> None:
    # ``step`` is incremented inside super().add_transition(); capture it first.
    step = self.step
    if transition.next_observations is not None:
      self.next_observations[step].copy_(transition.next_observations)
    super().add_transition(transition)

  def mini_batch_generator(
    self, num_mini_batches: int, num_epochs: int = 8
  ) -> Generator[RolloutStorage.Batch, None, None]:
    """Same shuffled mini-batches as the base, plus next-obs and dones.

    We replicate the base index shuffling and attach ``next_observations`` and
    ``dones`` onto each yielded ``Batch`` as dynamic attributes so HIMPPO can
    supervise the estimator. The base ``Batch`` class has no slots, so attaching
    attributes is safe.
    """
    if self.training_type != "rl":
      raise ValueError("This function is only available for reinforcement learning training.")

    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

    observations = self.observations.flatten(0, 1)
    next_observations = self.next_observations.flatten(0, 1)
    dones = self.dones.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)

    for _ in range(num_epochs):
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        stop = (i + 1) * mini_batch_size
        batch_idx = indices[start:stop]

        batch = RolloutStorage.Batch(
          observations=observations[batch_idx],
          actions=actions[batch_idx],
          values=values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_actions_log_prob[batch_idx],
          old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
          dones=dones[batch_idx],
        )
        # Attach HIM-specific extras.
        batch.next_observations = next_observations[batch_idx]
        yield batch
