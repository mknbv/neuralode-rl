""" Script to add multiple experiments to task spooler. """
import argparse
from itertools import chain
import os
import re


def get_envs(env_type="mujoco"):
  """ Returns list of default envs. """
  return {
      "mujoco": [
          "HalfCheetah-v3",
          "Hopper-v3",
          "InvertedDoublePendulum-v2",
          "InvertedPendulum-v2",
          "Reacher-v2",
          "Swimmer-v3",
          "Walker2d-v3",
      ],
      "roboschool": [
          "RoboschoolHumanoid-v1",
          "RoboschoolHumanoidFlagrun-v1",
          "RoboschoolHumanoidFlagrunHarder-v1",
      ]
  }[env_type]


def get_parser():
  """ Returns parser. """
  parser = argparse.ArgumentParser()
  parser.add_argument("--env-ids", nargs='+', default=["mujoco", "roboschool"])
  parser.add_argument("--logdir-prefix", required=True)
  parser.add_argument("--seed", nargs='+', type=int, default=None)
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

  envs = list(chain.from_iterable(
      get_envs(env) if env in ["mujoco", "roboschool"] else (env,)
      for env in args.env_ids
  ))
  for env_id in envs:
    seed = args.seed
    if seed is None:
      seed = [0, 123, 250, 500, 1024][
          :3 if env_id.startswith("Roboschool") else None]

    for i, seed in enumerate(seed):
      logdir_name = env_id_to_logdir_name(env_id) + f".{i:02d}"
      logdir = os.path.join(args.logdir_prefix, logdir_name)
      if os.path.isdir(logdir):
        raise ValueError(f"directory {logdir} exists")

      scriptname = ("run-roboschool.py" if env_id.startswith("Roboschool")
                    else "run-mujoco.py")
      cmd = ("tsp zsh -c "
             f"'CUDA_VISIBLE_DEVICES= python {scriptname} "
             f"--env-id {env_id} --seed {seed} "
             f"--logdir {logdir} {unknown_args}'")
      print(cmd)
      if not args.dry:
        os.system(cmd)


if __name__ == "__main__":
  main()
