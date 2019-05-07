""" Defines simple neural ODE model. """
from functools import partial
from math import sqrt
import tensorflow as tf
from odeint import odeint


class MLP(tf.keras.Sequential):
  """ Simple MLP model. """
  def __init__(self,
               output_units,
               num_layers=3,
               hidden_units=64,
               activation=tf.nn.tanh):
    super().__init__([
        tf.keras.layers.Dense(
            units=hidden_units if i < num_layers else output_units,
            activation=activation if i < num_layers else None,
            kernel_initializer=(
                tf.initializers.orthogonal(sqrt(2) if i < num_layers else 1)),
            bias_initializer=tf.initializers.zeros(),
        ) for i in range(1, num_layers + 1)
    ])


class RoboschoolMLP(tf.keras.Sequential):
  """ Roboschool MLP. """
  def __init__(self, output_units, activation=tf.nn.relu):
    def init(output=False):
      scale = 1 if output else sqrt(2)
      return dict(kernel_initializer=tf.initializers.orthogonal(scale),
                  bias_initializer=tf.initializers.zeros())
    super().__init__([
        tf.keras.layers.Dense(256, activation=activation, **init()),
        tf.keras.layers.Dense(128, activation=activation, **init()),
        tf.keras.layers.Dense(output_units, **init(True))
    ])


class ODEModel(tf.keras.Model):
  """ ODE model that wraps state, dynamics and output models. """
  def __init__(self, state, dynamics, outputs,
               time=(0., 1.), rtol=1e-3, atol=1e-3):
    super().__init__()
    self.state = state
    self.dynamics = dynamics
    self.outputs = outputs
    self.time = tf.cast(tf.convert_to_tensor(time), tf.float32)
    self.odeint = partial(odeint, rtol=rtol, atol=atol)

  def call(self, inputs, training=True, mask=None):
    _ = training, mask

    def dynamics(inputs, time):
      time = tf.cast([[time]], tf.float32)
      inputs = tf.concat([inputs, tf.tile(time, [inputs.shape[0], 1])], -1)
      return self.dynamics(inputs)

    state = self.state(inputs)
    hidden = self.odeint(dynamics, state, self.time)[-1]
    out = self.outputs(hidden)
    return out


class ODEMLP(ODEModel):
  """ Basic MLP model with ode. """
  # pylint: disable=too-many-arguments
  def __init__(self, output_units, hidden_units=64,
               num_state_layers=1, num_dynamics_layers=1, num_output_layers=1,
               time=(0., 1.), rtol=1e-3, atol=1e-3):

    def make_sequential(num_layers, **layer_kws):
      return tf.keras.Sequential(
          [tf.keras.layers.Dense(**layer_kws) for _ in range(num_layers)])

    layer_kws = dict(
        units=hidden_units,
        activation=tf.nn.tanh,
        kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
        bias_initializer=tf.initializers.zeros())

    state = make_sequential(num_state_layers, **layer_kws)
    dynamics = make_sequential(num_dynamics_layers, **layer_kws)

    layer_kws.update(units=output_units, activation=None,
                     kernel_initializer=tf.initializers.orthogonal(1))
    output = make_sequential(num_output_layers, **layer_kws)
    super().__init__(state, dynamics, output, time=time, rtol=rtol, atol=atol)


class ContinuousActorCriticModel(tf.keras.Model):
  """ Adds variance variable to policy and value models to create new model. """
  def __init__(self, input_shape, action_dim, policy, value, logstd=None):
    super().__init__()
    self.input_tensor = tf.keras.layers.Input(input_shape)
    self.policy = policy
    self.value = value
    if logstd is not None:
      if tf.shape(logstd) != [action_dim]:
        raise ValueError(f"logstd has wrong shape {tf.shape(logstd)}, ",
                         f"expected 1-d tensor of size action_dim={action_dim}")
      self.logstd = logstd
    else:
      self.logstd = tf.Variable(tf.zeros(action_dim), trainable=True,
                                name="logstd")

  @property
  def input(self):
    return self.input_tensor

  def call(self, inputs, training=True, mask=None):
    _ = training, mask
    inputs = tf.cast(inputs, tf.float32)
    batch_size = tf.shape(inputs)[0]
    logstd = tf.tile(self.logstd[None], [batch_size, 1])
    return self.policy(inputs), tf.exp(logstd), self.value(inputs)
