import argparse
from math import sqrt
import gym
import neuralode
import numpy as np
import rl
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tf.enable_eager_execution()
tf.set_random_seed(0)


def get_parser(parser=None):
  if parser is None:
    parser = argparse.ArgumentParser()
  parser.add_argument("--env-id", required=True)
  parser.add_argument("--logdir", required=True)
  parser.add_argument("--log-period", type=int, default=1)
  parser.add_argument("--nenvs", type=int, default=1)
  parser.add_argument("--policy-odeint", action="store_true")
  parser.add_argument("--value-odeint", action="store_true")
  parser.add_argument("--tol", type=float, default=1e-3)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--entropy-coef", type=float, default=0.)
  parser.add_argument("--cliprange", type=float, default=0.2)
  parser.add_argument("--num-train-steps", type=float, default=1e6)
  parser.add_argument("--num-runner-steps", type=int, default=2048)
  parser.add_argument("--num-epochs", type=int, default=10)
  parser.add_argument("--num-minibatches", type=int, default=32)
  return parser


class Path(tf.keras.Model):
  def __init__(self,
               output_units,
               odeint=False,
               time=(0., 1.),
               rtol=1e-3,
               atol=1e-3,
               **kwargs):
    super().__init__()
    self.odeint = odeint
    self.time = tf.convert_to_tensor(time)
    self.rtol = rtol
    self.atol = atol
    self.state = tf.keras.layers.Dense(
        units=64,
        activation=tf.nn.tanh,
        kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
        bias_initializer=tf.initializers.zeros())
    self.dynamics = tf.keras.layers.Dense(
        units=64,
        activation=tf.nn.tanh,
        kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
        bias_initializer=tf.initializers.zeros())
    self.outputs = tf.keras.layers.Dense(
        units=output_units,
        kernel_initializer=tf.initializers.orthogonal(1),
        bias_initializer=tf.initializers.zeros())

  def call(self, inputs): # pylint: disable=arguments-differ
    state = self.state(inputs)
    if self.odeint:
      def dynamics(state, t):
        t = tf.convert_to_tensor([[tf.cast(t, tf.float32)]])
        t = tf.tile(t, [state.shape[0], 1])
        return self.dynamics(tf.concat([state, t], -1))

      state = neuralode.odeint(dynamics, state, self.time,
                               rtol=self.rtol, atol=self.atol)[-1]
    else:
      state = self.dynamics(state)
    return self.outputs(state)



class MLPModel(tf.keras.Model):
  def __init__(self,
               input_shape,
               action_shape,
               odeint=(False, False),
               time=(0., 1.),
               rtol=1e-3,
               atol=1e-3):
    super().__init__()
    self._input_shape = input_shape
    self.logits = Path(np.prod(action_shape), odeint=odeint[0], time=time,
                       rtol=rtol, atol=atol)
    self.values = Path(1, odeint=odeint[1], time=time, rtol=rtol, atol=atol)
    self.logstd = tf.Variable(np.zeros(np.prod(action_shape)), dtype=tf.float32,
                              trainable=True, name="logstd")

  @property
  def input(self):
    return tf.keras.layers.Input(self._input_shape)

  def call(self, inputs): # pylint: disable=arguments-differ
    inputs = tf.cast(inputs, tf.float32)
    return self.logits(inputs), tf.exp(self.logstd), self.values(inputs)


def main():
  args = get_parser().parse_args()
  def make_env(seed=0):
    env = gym.make(args.env_id)
    env.seed(seed)
    return env

  make_funcs = [lambda seed=i: make_env(seed) for i in range(args.nenvs)]
  env = rl.env.Normalize(rl.env.Summaries(
      rl.env.ParallelEnvBatch(make_funcs), prefix=args.env_id))
  policy = rl.ActorCriticPolicy(
      MLPModel(env.observation_space.shape,
               env.action_space.shape,
               odeint=(args.policy_odeint, args.value_odeint),
               rtol=args.tol, atol=args.tol),
      distribution=tfp.distributions.MultivariateNormalDiag)
  runner = rl.make_ppo_runner(env, policy,
                              num_runner_steps=args.num_runner_steps,
                              num_epochs=args.num_epochs,
                              num_minibatches=args.num_minibatches)

  lr = rl.train.linear_anneal("lr", args.lr, args.num_train_steps,
                              runner.step_var)
  optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-5)
  ppo = rl.PPO(policy, optimizer,
               entropy_coef=args.entropy_coef,
               cliprange=args.cliprange)

  writer = tf.contrib.summary.create_file_writer(args.logdir)
  writer.set_as_default()

  pbar = tqdm(total=args.num_train_steps)
  with tf.contrib.summary.record_summaries_every_n_global_steps(
      args.log_period):
    while int(runner.step_var) < args.num_train_steps:
      pbar.update(int(runner.step_var) - pbar.n)
      trajectory = runner.get_next()
      ppo.step(trajectory)


if __name__ == "__main__":
  main()
