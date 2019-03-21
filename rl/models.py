""" Keras models for reinforcement learning problems. """
from math import sqrt
import tensorflow as tf


def maybe_rescale(inputs, ubyte_rescale=None):
  """ Rescales inputs to [0, 1] if they are in uint8 and flag not False. """
  if ubyte_rescale and inputs.dtype != tf.uint8.as_numpy_dtype:
    raise ValueError("ubyte_rescale was set to True but "
                     f"inputs.dtype is {inputs.dtype}")
  if (ubyte_rescale or
      ubyte_rescale is None and inputs.dtype == tf.uint8.as_numpy_dtype):
    return tf.cast(inputs, tf.float32) / 255.
  return inputs


class NatureDQNBase(tf.keras.models.Sequential):
  """ Hidden layers of the Nature DQN model. """
  def __init__(self,
               input_shape=(84, 84, 4),
               ubyte_rescale=None,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    super().__init__([
        tf.keras.layers.Lambda(
            lambda inputs: maybe_rescale(inputs, ubyte_rescale),
            input_shape=input_shape,
            # Specifying output_shape is necessary to prevent redundant call
            # that will determine the output shape by calling the function
            # with tf.float32 type tensor
            output_shape=input_shape
        ),
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        ),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        ),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        ),
    ])


class NatureDQNModel(tf.keras.Model):
  """ Nature DQN model with possibly several outputs. """
  def __init__(self,
               output_units,
               input_shape=(84, 84, 4),
               ubyte_rescale=None,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    base = NatureDQNBase(input_shape, ubyte_rescale,
                         kernel_initializer, bias_initializer)
    inputs = tf.keras.layers.Input(input_shape)
    base_outputs = base(inputs)
    if isinstance(output_units, (list, tuple)):
      outputs = [
          tf.keras.layers.Dense(
              units=units,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(base_outputs)
          for units in output_units
      ]
    else:
      outputs = tf.keras.layers.Dense(
          units=output_units,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer)(base_outputs)
    super().__init__(inputs=inputs, outputs=outputs)
