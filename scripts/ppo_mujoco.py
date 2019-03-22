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
      rl.MujocoModel(env.observation_space.shape,
                     np.prod(env.action_space.shape), 1),
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
