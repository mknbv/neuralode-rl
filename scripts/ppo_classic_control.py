import argparse
from math import sqrt
import gym
import neuralode
import rl
import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()
tf.set_random_seed(0)


def get_parser(parser=None):
  if parser is None:
    parser = argparse.ArgumentParser()
  parser.add_argument("--env-id", required=True)
  parser.add_argument("--logdir", required=True)
  parser.add_argument("--log-period", type=int, default=1)
  parser.add_argument("--nenvs", type=int, default=4)
  parser.add_argument("--odeint", action="store_true")
  parser.add_argument("--tol", type=float, default=1e-4)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--num-train-steps", type=float, default=30e3)
  parser.add_argument("--num-runner-steps", type=int, default=128)
  parser.add_argument("--num-epochs", type=int, default=4)
  parser.add_argument("--num-minibatches", type=int, default=2)
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
    self.state = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh,
                                       **kwargs)
    self.dynamics = tf.keras.layers.Dense(units=64, activation=tf.nn.tanh,
                                          **kwargs)
    self.outputs = tf.keras.layers.Dense(units=output_units, **kwargs)

  def call(self, inputs):
    state = inputs
    if self.odeint:
      def dynamics(state, t):
        t = tf.convert_to_tensor([[tf.cast(t, tf.float32)]])
        t = tf.tile(t, [state.shape[0], 1])
        return self.dynamics(self.state(tf.concat([state, t], -1)))

      state = neuralode.odeint(dynamics, state, self.time,
                               rtol=self.rtol, atol=self.atol)[-1]
    else:
      state = self.dynamics(self.state(state))
    return self.outputs(state)



class CartPoleModel(tf.keras.Model):
  def __init__(self,
               odeint=False,
               time=(0., 1.),
               rtol=1e-3,
               atol=1e-3,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    super().__init__()
    init = {"kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer}
    self.logits = Path(2, odeint=odeint, time=time,
                       rtol=rtol, atol=atol, **init)
    self.values = Path(1, odeint=odeint, time=time,
                       rtol=rtol, atol=atol, **init)

  @property
  def input(self):
    return tf.keras.layers.Input((4,))

  def call(self, inputs):
    inputs = tf.cast(inputs, tf.float32)
    return self.logits(inputs), self.values(inputs)


def main():
  args = get_parser().parse_args()
  def make_env(seed=0):
    env = gym.make(args.env_id)
    env.seed(seed)
    return env

  make_funcs = [lambda seed=i: make_env(seed) for i in range(args.nenvs)]
  env = rl.env.Summaries(rl.env.ParallelEnvBatch(make_funcs),
                         running_mean_size=10,
                         prefix=args.env_id)
  policy = rl.ActorCriticPolicy(
      CartPoleModel(odeint=args.odeint, rtol=args.tol, atol=args.tol))
  runner = rl.make_ppo_runner(env, policy,
                              num_runner_steps=args.num_runner_steps,
                              num_epochs=args.num_epochs,
                              num_minibatches=args.num_minibatches)

  optimizer = tf.train.AdamOptimizer(args.lr)
  ppo = rl.PPO(policy, optimizer)

  writer = tf.contrib.summary.create_file_writer(args.logdir)
  writer.set_as_default()

  pbar = tqdm(total=args.num_train_steps)
  with tf.contrib.summary.record_summaries_every_n_global_steps(
      args.log_period):
    while int(runner.step_var) < args.num_train_steps:
      pbar.update(int(runner.step_var) - pbar.n)
      trajectory = runner.get_next()
      loss = ppo.step(trajectory)


if __name__ == "__main__":
  main()
