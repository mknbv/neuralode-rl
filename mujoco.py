import tensorflow as tf
import derl
from neuralode_model import ODEMujocoModel

tf.enable_eager_execution()


def get_parser(base_parser):
  base_parser.add_argument("--seed", type=int, default=0)
  base_parser.add_argument("--ode-policy", action="store_true")
  base_parser.add_argument("--ode-value", action="store_true")
  base_parser.add_argument("--tol", type=float, default=1e-3)
  return base_parser


def main():
  parser = get_parser(derl.get_parser(defaults=derl.PPOLearner.mujoco_defaults))
  args = parser.parse_args()

  env = derl.env.make(args.env_id)
  env.seed(args.seed)
  model = ODEMujocoModel(env.observation_space.shape,
                         env.action_space.shape[0],
                         ode_policy=args.ode_policy,
                         ode_value=args.ode_value,
                         rtol=args.tol, atol=args.tol)

  learner = derl.PPOLearner.from_env_args(env, args, model=model)
  learner.learn(args.num_train_steps, args.logdir, args.log_period)


if __name__ == "__main__":
  main()
