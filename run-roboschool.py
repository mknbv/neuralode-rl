""" Script to run roboschool experiment on a single env. """
# pylint: disable=invalid-name
from functools import partial
import roboschool # pylint: disable=unused-import
import tensorflow as tf
import derl
from neuralode_model import (ContinuousActorCriticModel,
                             ODEMLP, RoboschoolMLP)
tf.enable_eager_execution()


def add_neorl_args(base_parser):
  """ Adds neuralode-rl arguments to a give base parser. """
  base_parser.add_argument("--seed", type=int, default=0)
  base_parser.add_argument("--hidden-units", type=int, default=192)
  base_parser.add_argument("--num-state-layers", type=int, default=1)
  base_parser.add_argument("--num-dynamics-layers", type=int, default=1)
  base_parser.add_argument("--num-output-layers", type=int, default=1)
  base_parser.add_argument("--ode-policy", action="store_true")
  base_parser.add_argument("--ode-value", action="store_true")
  base_parser.add_argument("--tol", type=float, default=1e-3)
  return base_parser


def make_mlp_class(use_ode, args):
  """ Returns (partial) MLP class with args from args set. """
  if use_ode:
    return partial(ODEMLP, hidden_units=args.hidden_units,
                   num_state_layers=args.num_state_layers,
                   num_dynamics_layers=args.num_dynamics_layers,
                   num_output_layers=args.num_dynamics_layers,
                   rtol=args.tol, atol=args.tol)
  return RoboschoolMLP


def get_defaults(env_id):
  """ Returns dict of default arguments. """
  return {
      "num-train-steps": 100e6 if "Flagrun" in env_id else 50e6,
      "nenvs": 128 if "Flagrun" in env_id else 32,
      "num-runner-steps": 512,
      "num-epochs": 15,
      "num-minibatches": 16 if "Flagrun" in env_id else 4,
      "lr": 3e-4,
      "entropy-coef": 0.,
  }


def get_args():
  """ Returns arguments. """
  parser = derl.get_simple_parser()
  args, unknown_args = parser.parse_known_args()
  parser = derl.get_defaults_parser(get_defaults(args.env_id))
  parser = add_neorl_args(parser)
  args = parser.parse_args(unknown_args, args)
  return derl.log_args(args)


def make_env(env_id, nenvs, seed):
  """ Creates env instance. """
  env = derl.env.make(env_id, nenvs, seed)
  env = derl.env.Summaries(env, prefix=env_id)
  env = derl.env.Normalize(env)
  return env


def make_logstd(action_dim, nsteps, step_var, start_value=-0.7, end_value=-1.6):
  """ Creates and returns logstd variable that is automatically annealed. """
  start_tensor = tf.constant([start_value] * action_dim)
  end_tensor = tf.constant([end_value] * action_dim)
  return derl.train.linear_anneal("logstd", start_tensor, nsteps, step_var,
                                  end_tensor)


def main():
  """ Enterance point. """
  args = get_args()
  env = make_env(args.env_id, args.nenvs, args.seed)

  policy = make_mlp_class(args.ode_policy, args)(env.action_space.shape[0])
  value = make_mlp_class(args.ode_value, args)(1)
  model = ContinuousActorCriticModel(env.observation_space.shape,
                                     env.action_space.shape[0],
                                     policy, value)

  learner = derl.PPOLearner.from_env_args(env, args, model=model)
  model.logstd = make_logstd(env.action_space.shape[0],
                             args.num_train_steps, learner.runner.step_var)
  learner.learn(args.num_train_steps, args.logdir, args.log_period)


if __name__ == "__main__":
  main()
