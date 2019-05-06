""" Script to run muojco experiment on a single env. """
from functools import partial
import tensorflow as tf
import derl
from neuralode_model import ContinuousActorCriticModel, ODEMLP, MLP
tf.enable_eager_execution()


def get_parser(base_parser):
  """ Adds neuralode-rl arguments to a give base parser. """
  base_parser.add_argument("--seed", type=int, default=0)
  base_parser.add_argument("--hidden-units", type=int, default=64)
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
  return partial(MLP, hidden_units=args.hidden_units,
                 num_layers=(args.num_state_layers
                             + args.num_dynamics_layers
                             + args.num_output_layers))


def main():
  """ Eneterance point. """
  parser = get_parser(derl.get_parser(derl.PPOLearner.get_defaults("mujoco")))
  args = derl.log_args(parser.parse_args())

  env = derl.env.make(args.env_id)
  env.seed(args.seed)
  policy = make_mlp_class(args.ode_policy, args)(env.action_space.shape[0])
  value = make_mlp_class(args.ode_value, args)(1)
  model = ContinuousActorCriticModel(env.observation_space.shape,
                                     env.action_space.shape[0],
                                     policy, value)

  learner = derl.PPOLearner.from_env_args(env, args, model=model)
  learner.learn(args.num_train_steps, args.logdir, args.log_period)


if __name__ == "__main__":
  main()
