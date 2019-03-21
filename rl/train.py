""" Helpers for training models. """
import tensorflow as tf


class StepVariable:
  """ Wrapper for scalar non-trainable tf.Variable used for step count. """
  def __init__(self, name, value=0, auto_update=True):
    if not isinstance(value, (int, tf.Variable)):
      raise TypeError("value must be of type int or tf.Variable, "
                      f"but has type {type(value)}")
    if isinstance(value, tf.Variable):
      if value.shape != tuple() or value.dtype != tf.int64:
        raise ValueError("if of type tf.Variable, value must be scalar "
                         f"and have dtype tf.int64, got shape {value.shape}"
                         f" and dtype {value.dtype} instead")
      self.variable = value
    else:
      self.variable = tf.Variable(value, dtype=tf.int64,
                                  trainable=False, name=name)
    self.auto_update = auto_update
    self.anneals = []

  def __int__(self):
    return int(self.variable)

  def convert_to_tensor(self, dtype=None, name=None, as_ref=None):
    """ Converts the step variable to tf.Tensor. """
    _ = name
    if dtype == self.variable.dtype:
      return self.variable if as_ref else self.variable.value()
    return NotImplemented

  def assign_add(self, delta):
    """ Updates the step variable by incrementing it by delta. """
    newstep = self.variable.assign_add(delta)
    for var, fun in self.anneals:
      var.assign(fun(var, self.variable))
      tf.contrib.summary.scalar(f"train/{var.name[:var.name.rfind(':')]}",
                                var, step=self)
    return newstep

  def add_annealing_variable(self, variable, function):
    """ Adds variable that will be annealed after changes in the step. """
    self.anneals.append((variable, function))


tf.register_tensor_conversion_function(
    StepVariable,
    lambda value, *args, **kwargs: value.convert_to_tensor(*args, **kwargs)
)


def linear_anneal(name, start_value, nsteps, step_var, end_value=0.):
  """ Returns variable that will be linearly annealed. """
  if not isinstance(step_var, StepVariable):
    raise TypeError("step_var must be an instance of StepVariable, "
                    f"got {type(step_var)} instead")

  var = tf.Variable(start_value, trainable=False, name=name)
  step_var.add_annealing_variable(
      var,
      lambda var, step: end_value + start_value * (
          1. - tf.to_float(step) / tf.to_float(nsteps)))
  return var
