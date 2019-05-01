""" Script to add multiple experiments to task spooler. """
import argparse
import os
import re


def get_default_envs():
  """ Returns list of default envs. """
  with open("mujoco-envs.txt") as envsfile:
    envs = list(map(str.rstrip, envsfile))
  return envs


def get_parser():
  """ Returns parser. """
  parser = argparse.ArgumentParser()
  parser.add_argument("--env-ids", nargs='+', default=get_default_envs())
  parser.add_argument("--logdir-prefix", required=True)
  parser.add_argument("--seed", nargs='+', type=int,
                      default=[0, 123, 250, 500, 1024])
  parser.add_argument("--dry", action="store_true")
  return parser


def env_id_to_logdir_name(env_id):
  """ Converts env-id to name of logdir """
  return '-'.join(
      map(str.lower,
          re.findall('[A-Z][^A-Z]*', ''.join(env_id.split('-')[:-1]))))


def main():
  """ Adds multiple runs to task spooler. """
  args, unknown_args = get_parser().parse_known_args()
  unknown_args = ' '.join(unknown_args)

  for env_id in args.env_ids:
    for i, seed in enumerate(args.seed):
      logdir_name = env_id_to_logdir_name(env_id) + f".{i:02d}"
      logdir = os.path.join(args.logdir_prefix, logdir_name)
      if os.path.isdir(logdir):
        raise ValueError(f"directory {logdir} exists")

      cmd = ("tsp zsh -c "
             "'CUDA_VISIBLE_DEVICES= python mujoco.py "
             f"--env-id {env_id} --seed {seed} "
             f"--logdir {logdir} {unknown_args}'")
      print(cmd)
      if not args.dry:
        os.system(cmd)


if __name__ == "__main__":
  main()
