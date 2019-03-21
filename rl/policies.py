""" Reinforcement learning policies. """
from abc import ABC, abstractmethod

import tensorflow as tf


class Policy(ABC):
  """ RL policy (typically wraps a keras model).  """
  def is_recurrent(self): # pylint: disable=no-self-use
    return False

  def get_state(self): # pylint: disable=no-self-use
    """ Returns current policy state. """
    return None

  def reset(self): # pylint: disable=no-self-use
    """ Resets the state. """

  @abstractmethod
  def act(self, inputs, state=None, update_state=True, training=False):
    """ Returns `dict` of all the outputs of the policy.

    If `training=False`, then inputs can be a batch of observations
    or a `dict` containing `observations` key. Otherwise,
    `inputs` should be a trajectory dictionary with all keys
    necessary to recompute outputs for training.
    """

class ActorCriticPolicy(Policy):
  """ Actor critic policy with discrete number of actions. """
  def __init__(self, model, distribution=tf.distributions.Categorical):
    self.model = model
    self.distribution = distribution

  def act(self, inputs, state=None, update_state=True, training=False):
    # TODO: support recurrent policies.
    _ = update_state
    if state is not None:
      raise NotImplementedError()
    if training:
      observations = inputs["observations"]
    else:
      observations = inputs

    expand_dims = self.model.input.shape.ndims - observations.ndim
    observations = observations[(None,) * expand_dims]
    *distribution_inputs, values = self.model(observations)
    squeeze_dims = tuple(range(expand_dims))
    if squeeze_dims:
      distribution_inputs = [tf.squeeze(inputs, squeeze_dims)
                             for inputs in distribution_inputs]
      values = tf.squeeze(values, squeeze_dims)

    distribution = self.distribution(*distribution_inputs)
    if training:
      return {"distribution": distribution, "values": values}
    actions = distribution.sample()
    log_prob = distribution.log_prob(actions)
    return {"actions": actions.numpy(),
            "log_prob": log_prob.numpy(),
            "values": values.numpy()}
