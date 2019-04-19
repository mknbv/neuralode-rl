""" Defines simple neural ODE model. """
from functools import partial
from math import sqrt
import tensorflow as tf
from odeint import odeint


class MLP(tf.keras.Sequential):
  """ Simple MLP model. """
  def __init__(self,
               output_units,
               nlayers=3,
               hidden_units=64,
               activation=tf.nn.tanh):
    super().__init__([
        tf.keras.layers.Dense(
            units=hidden_units if i < nlayers else output_units,
            activation=activation if i < nlayers else None,
            kernel_initializer=(
                tf.initializers.orthogonal(sqrt(2) if i < nlayers else 1)),
            bias_initializer=tf.initializers.zeros(),
        ) for i in range(1, nlayers + 1)
    ])


class ODEMLP(tf.keras.Model):
  """ Basic MLP model with ode. """
  def __init__(self, output_units, hidden_units=64,
               time=(0., 1.), rtol=1e-3, atol=1e-3):
    super().__init__()
    self.time = tf.cast(tf.convert_to_tensor(time), tf.float32)
    layer_kws = {"units": hidden_units,
                 "activation": tf.nn.tanh,
                 "kernel_initializer": tf.initializers.orthogonal(sqrt(2)),
                 "bias_initializer": tf.initializers.zeros()}
    self.state = tf.keras.layers.Dense(**layer_kws)
    self.dynamics = tf.keras.layers.Dense(**layer_kws)
    self.odeint = partial(odeint, rtol=rtol, atol=atol)
    layer_kws.update(units=output_units, activation=None,
                     kernel_initializer=tf.initializers.orthogonal(1))
    self.out = tf.keras.layers.Dense(**layer_kws)

  def call(self, inputs, training=True, mask=None):
    _ = training, mask

    def dynamics(inputs, time):
      time = tf.cast([[time]], tf.float32)
      inputs = tf.concat([inputs, tf.tile(time, [inputs.shape[0], 1])], -1)
      return self.dynamics(inputs)

    state = self.state(inputs)
    hidden = self.odeint(dynamics, state, self.time)[-1]
    out = self.out(hidden)
    return out


class ODEMujocoModel(tf.keras.Model):
  """ Mujoco model with ODE layer(s). """
  def __init__(self, input_shape, nactions, ode_policy=True, ode_value=False,
               rtol=1e-3, atol=1e-3):
    super().__init__()
    self.input_layer = tf.keras.layers.Input(input_shape)
    if ode_policy:
      self.policy = ODEMLP(nactions, rtol=rtol, atol=atol)
    else:
      self.policy = MLP(nactions)
    if ode_value:
      self.values = ODEMLP(1, rtol=rtol, atol=atol)
    else:
      self.values = MLP(1)
    self.logstd = tf.Variable(tf.zeros(nactions), trainable=True, name="logstd")

  @property
  def input(self):
    return self.input_layer

  def call(self, inputs): # pylint: disable=arguments-differ
    inputs = tf.cast(inputs, tf.float32)
    batch_size = tf.shape(inputs)[0]
    logstd = tf.tile(self.logstd[None], [batch_size, 1])
    return self.policy(inputs), tf.exp(logstd), self.values(inputs)
