"""
Implements Proximal Policy Optimization algorithm.
"""
import tensorflow as tf

from .base import BaseAlgorithm


def r_squared(targets, predictions):
  """ Coefficient of determination. """
  target_variance = tf.nn.moments(targets, tuple(range(targets.ndim)))[1]
  return 1. - tf.reduce_mean(tf.square(targets - predictions)) / target_variance


class PPO(BaseAlgorithm):
  """ Proximal Policy Optimization algorithm.

  See [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
  """
  def __init__(self,
               policy,
               optimizer=None,
               cliprange=0.2,
               value_loss_coef=0.25,
               entropy_coef=0.01,
               max_grad_norm=0.5,
               step_var=None):
    super().__init__(model=policy.model, optimizer=optimizer, step_var=step_var)
    self.policy = policy
    self.cliprange = cliprange
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.max_grad_norm = max_grad_norm

  def policy_loss(self, trajectory, act=None):
    """ Compute policy loss (including entropy regularization). """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    if "advantages" not in trajectory:
      raise ValueError("trajectory does not contain 'advantages'")

    old_log_prob = trajectory["log_prob"]
    advantages = trajectory["advantages"]
    actions = trajectory["actions"]

    log_prob = act["distribution"].log_prob(actions)
    if log_prob.shape != old_log_prob.shape:
      raise ValueError("trajectory has mismatched shapes: "
                       f"log_prob.shape={log_prob.shape} "
                       f"old_log_prob.shape={old_log_prob.shape}")
    if log_prob.shape != advantages.shape:
      raise ValueError("trajectory has mismatched shapes: "
                       f"log_prob.shape={log_prob.shape} "
                       f"advantages.shape={advantages.shape}")

    ratio = tf.exp(log_prob - old_log_prob)
    policy_loss = -ratio * advantages
    if self.cliprange is not None:
      ratio_clipped = tf.clip_by_value(ratio, 1. - self.cliprange,
                                       1. + self.cliprange)
      policy_loss_clipped = -ratio_clipped * advantages
      policy_loss = tf.maximum(policy_loss, policy_loss_clipped)

    policy_loss = tf.reduce_mean(policy_loss)
    entropy = tf.reduce_mean(act["distribution"].entropy())
    tf.contrib.summary.scalar("ppo/advantages", tf.reduce_mean(advantages),
                              step=self.step_var)
    tf.contrib.summary.scalar("ppo/policy_loss", policy_loss,
                              step=self.step_var)
    tf.contrib.summary.scalar("ppo/entropy", entropy, step=self.step_var)
    return policy_loss - self.entropy_coef * entropy

  def value_loss(self, trajectory, act=None):
    """ Computes value loss. """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    if "value_targets" not in trajectory:
      raise ValueError("trajectory does not contain 'value_targets'")

    value_targets = trajectory["value_targets"]
    old_value_preds = trajectory["values"]
    values = act["values"]

    if values.shape != value_targets.shape:
      raise ValueError("trajectory has mismatched shapes "
                       f"values.shape={values.shape} "
                       f"value_targets.shape={value_targets.shape}")

    value_loss = tf.square(values - value_targets)
    if self.cliprange is not None:
      values_clipped = old_value_preds + tf.clip_by_value(
          values - old_value_preds, -self.cliprange, self.cliprange)
      value_loss_clipped = tf.square(values_clipped - value_targets)
      value_loss = tf.maximum(value_loss, value_loss_clipped)

    value_loss = tf.reduce_mean(value_loss)
    tf.contrib.summary.scalar("ppo/value_loss", value_loss, step=self.step_var)
    tf.contrib.summary.scalar("ppo/value_targets",
                              tf.reduce_mean(value_targets),
                              step=self.step_var)
    tf.contrib.summary.scalar("ppo/value_preds", tf.reduce_mean(values),
                              step=self.step_var)
    tf.contrib.summary.scalar(
        "ppo/r_squared",
        r_squared(tf.convert_to_tensor(value_targets), values),
        step=self.step_var)
    value_loss = tf.reduce_mean(value_loss)
    return value_loss

  def loss(self, data):
    """ Returns ppo loss for given data (trajectory dict). """
    act = self.policy.act(data, training=True)
    policy_loss = self.policy_loss(data, act)
    value_loss = self.value_loss(data, act)
    loss = policy_loss + self.value_loss_coef * value_loss
    tf.contrib.summary.scalar("ppo/loss", loss, step=self.step_var)
    return loss

  def preprocess_gradients(self, gradients):
    if self.max_grad_norm is None:
      grad_norm = tf.linalg.global_norm(gradients)
    else:
      gradients, grad_norm = tf.clip_by_global_norm(gradients,
                                                    self.max_grad_norm)
    tf.contrib.summary.scalar("ppo/grad_norm", grad_norm, step=self.step_var)
    return gradients
