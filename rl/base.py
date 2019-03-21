"""
Defines base classes.
"""
from abc import ABC, abstractmethod
import re

import tensorflow as tf
from rl.train import StepVariable


class BaseRunner(ABC):
  def __init__(self, step_var=None):
    if step_var is None:
      step_var = StepVariable(f"{camel2snake(self.__class__.__name__)}_step",
                              tf.train.get_or_create_global_step())
    self.step_var = step_var

  """ Gives access to task-specific data. """
  @abstractmethod
  def get_next(self):
    """ Returns next data object """

  def __iter__(self):
    while True:
      yield self.get_next()


def camel2snake(name):
  """ Converts camel case to snake case. """
  sub = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', sub).lower()


class BaseAlgorithm(ABC):
  """ Base algorithm. """
  def __init__(self, model, optimizer=None, step_var=None):
    self.model = model
    self.optimizer = optimizer or self.model.optimizer
    if step_var is None:
      step_var = StepVariable(f"{camel2snake(self.__class__.__name__)}_step")
    self.step_var = step_var

  @abstractmethod
  def loss(self, data):
    """ Computes the loss given inputs and target values. """

  def preprocess_gradients(self, gradients):
    """ Applies gradient preprocessing. """
    # pylint: disable=no-self-use
    return gradients

  def step(self, data):
    """ Performs single training step of the algorithm. """
    with tf.GradientTape() as tape:
      loss = self.loss(data)
    gradients = self.preprocess_gradients(
        tape.gradient(loss, self.model.variables))
    self.optimizer.apply_gradients(zip(gradients, self.model.variables))
    if getattr(self.step_var, "auto_update", True):
      self.step_var.assign_add(1)
    return loss


class KerasAlgorithm(BaseAlgorithm):
  """ Algorithm wrapper for tf.keras.Model. """
  def __init__(self, model, optimizer=None,
               name="keras_algorithm", step_var=None):
    super().__init__(model, optimizer, step_var)
    self.name = name

  def loss(self, data):
    loss = self.model.loss(data['y'], self.model(data['x']))
    if self.name is not None:
      tf.contrib.summary.scalar(f"{self.name}/loss", loss, step=self.step_var)
    return loss
