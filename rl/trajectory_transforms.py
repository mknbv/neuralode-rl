""" Trajectory transformations. """
import numpy as np


class GAE:
  """ Generalized Advantage Estimator.

  See [Schulman et al., 2016](https://arxiv.org/abs/1506.02438)
  """
  def __init__(self, policy, gamma=0.99, lambda_=0.95, normalize=None,
               epsilon=1e-8):
    self.policy = policy
    self.gamma = gamma
    self.lambda_ = lambda_
    self.normalize = normalize
    self.epsilon = epsilon

  def __call__(self, trajectory):
    """ Applies the advantage estimator to a given trajectory.

    Returns:
      a tuple of (advantages, value_targets).
    """
    if "advantages" in trajectory:
      raise ValueError("trajectory cannot contain 'advantages'")
    if "value_targets" in trajectory:
      raise ValueError("trajectory cannot contain 'value_targets'")

    rewards = trajectory["rewards"]
    resets = trajectory["resets"]
    values = trajectory["values"]

    # Values might have an additional last dimension of size 1 as outputs of
    # dense layers. Need to adjust shapes of rewards and resets accordingly.
    if (not (0 <= values.ndim - rewards.ndim <= 1)
        or values.ndim == rewards.ndim + 1 and values.shape[-1] != 1):
      raise ValueError(
          f"tarajectory['values'] of shape {trajectory['values'].shape} "
          "must have the same number of dimensions as"
          f"trajectory['rewards'] which has shape {rewards.shape} "
          "or have last dimension of size 1")
    if values.ndim == rewards.ndim + 1:
      values = np.squeeze(values, -1)

    gae = np.zeros_like(values, dtype=np.float32)
    gae[-1] = rewards[-1] - values[-1]
    observation = trajectory["state"]["latest_observation"]
    state = trajectory["state"].get("policy_state", None)
    last_value = self.policy.act(observation, state=state,
                                 update_state=False)["values"]
    if np.asarray(resets[-1]).ndim < last_value.ndim:
      last_value = np.squeeze(last_value, -1)
    gae[-1] += (1 - resets[-1]) * self.gamma * last_value

    for i in range(gae.shape[0] - 1, 0, -1):
      not_reset = 1 - resets[i - 1]
      next_values = values[i]
      delta = (rewards[i - 1]
               + not_reset * self.gamma * next_values
               - values[i - 1])
      gae[i - 1] = delta + not_reset * self.gamma * self.lambda_ * gae[i]
    value_targets = gae + values
    value_targets = value_targets[
        (...,) + (None,) * (trajectory["values"].ndim - value_targets.ndim)]

    if self.normalize or self.normalize is None and gae.size > 1:
      gae = (gae - gae.mean()) / (gae.std() + self.epsilon)

    trajectory["advantages"] = gae
    trajectory["value_targets"] = value_targets
    return gae, value_targets


class MergeTimeBatch:
  """ Merges first two axes typically representing time and env batch. """
  def __call__(self, trajectory):
    assert trajectory["resets"].ndim == 2, trajectory["resets"].shape
    for key, val in filter(lambda kv: isinstance(kv[1], np.ndarray),
                           trajectory.items()):
      trajectory[key] = np.reshape(val, (-1, *val.shape[2:]))


class NormalizeAdvantages:
  """ Normalizes advantages. """
  def __init__(self, epsilon=1e-8):
    self.epsilon = epsilon

  def __call__(self, trajectory):
    advantages = trajectory["advantages"]
    trajectory["advantages"] = ((advantages - advantages.mean())
                                / (advantages.std() + self.epsilon))


class Take:
  """ Keepds data only from specified indices. """
  def __init__(self, indices, axis=1):
    self.indices = indices
    self.axis = axis

  def __call__(self, trajectory):
    for key, val in filter(lambda kv: kv[0] != "state", trajectory.items()):
      trajectory[key] = np.take(val, self.indices, axis=self.axis)
