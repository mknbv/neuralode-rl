""" RL env runner """
from collections import defaultdict
from gym.spaces import Box
import numpy as np

from .base import BaseRunner
from .env.env_batch import SpaceBatch
from .trajectory_transforms import (
    GAE, MergeTimeBatch, NormalizeAdvantages, AdvantagesToActionVector)


def nenvs(env):
  """ Returns number of envs in env batch or None if single env. """
  env = getattr(env, "unwrapped", env)
  return getattr(env, "nenvs", None)


class EnvRunner(BaseRunner):
  """ Reinforcement learning runner in an environment with given policy """
  def __init__(self, env, policy, nsteps,
               cutoff=None, transforms=None, step_var=None):
    super().__init__(step_var)
    self.env = env
    self.policy = policy
    self.nsteps = nsteps
    self.cutoff = cutoff
    self.transforms = transforms or []
    self.state = {"latest_observation": self.env.reset()}

  @property
  def nenvs(self):
    """ Returns number of batched envs or `None` if env is not batched """
    return nenvs(self.env)

  def reset(self):
    """ Resets env and runner states. """
    self.state["latest_observation"] = self.env.reset()
    self.policy.reset()

  def get_next(self):
    """ Runs the agent in the environment.  """
    trajectory = defaultdict(list, {"actions": []})
    observations = []
    rewards = []
    resets = []
    self.state["env_steps"] = self.nsteps
    if self.policy.is_recurrent():
      self.state["policy_state"] = self.policy.get_state()

    for i in range(self.nsteps):
      observations.append(self.state["latest_observation"])
      act = self.policy.act(self.state["latest_observation"])
      if "actions" not in act:
        raise ValueError("result of policy.act must contain 'actions' "
                         f"but has keys {list(act.keys())}")
      for key, val in act.items():
        trajectory[key].append(val)

      obs, rew, done, _ = self.env.step(trajectory["actions"][-1])
      self.state["latest_observation"] = obs
      rewards.append(rew)
      resets.append(done)
      self.step_var.assign_add(self.nenvs or 1)

      # Only reset if the env is not batched. Batched envs should auto-reset.
      if not self.nenvs and np.all(done):
        self.state["env_steps"] = i + 1
        self.state["latest_observation"] = self.env.reset()
        if self.cutoff or (self.cutoff is None and self.policy.is_recurrent()):
          break

    trajectory.update(observations=observations, rewards=rewards, resets=resets)
    for key, val in trajectory.items():
      trajectory[key] = np.asarray(val)
    trajectory["state"] = self.state

    for transform in self.transforms:
      transform(trajectory)
    return trajectory


class TrajectorySampler(BaseRunner):
  """ Samples parts of trajectory for specified number of epochs. """
  def __init__(self, runner, num_epochs=4, num_minibatches=4,
               shuffle_before_epoch=True, transforms=None):
    super().__init__(runner.step_var)
    self.runner = runner
    self.num_epochs = num_epochs
    self.num_minibatches = num_minibatches
    self.shuffle_before_epoch = shuffle_before_epoch
    self.transforms = transforms or []
    self.minibatch_count = 0
    self.epoch_count = 0
    self.trajectory = None

  @property
  def nenvs(self):
    """ Number of envs in batch or None for unbatched env. """
    return self.runner.nenvs

  def trajectory_is_stale(self):
    """ True iff new trajectory should be generated for sub-sampling. """
    return self.epoch_count >= self.num_epochs

  def shuffle_trajectory(self):
    """ Reshuffles trajectory along the first dimension. """
    sample_size = self.trajectory["observations"].shape[0]
    indices = np.random.permutation(sample_size)
    for key, val in filter(lambda kv: isinstance(kv[1], np.ndarray),
                           self.trajectory.items()):
      self.trajectory[key] = val[indices]

  def get_next(self):
    if self.trajectory is None or self.trajectory_is_stale():
      self.epoch_count = self.minibatch_count = 0
      self.trajectory = self.runner.get_next()
      if self.shuffle_before_epoch:
        self.shuffle_trajectory()

    sample_size = self.trajectory["observations"].shape[0]
    mbsize = sample_size // self.num_minibatches
    start = self.minibatch_count * mbsize
    indices = np.arange(start, min(start + mbsize, sample_size))
    minibatch = {key: val[indices] for key, val in self.trajectory.items()
                 if isinstance(val, np.ndarray)}

    self.minibatch_count += 1
    if self.minibatch_count == self.num_minibatches:
      self.minibatch_count = 0
      self.epoch_count += 1
      if self.shuffle_before_epoch and not self.trajectory_is_stale():
        self.shuffle_trajectory()

    for transform in self.transforms:
      transform(minibatch)
    return minibatch


def make_ppo_runner(env, policy, num_runner_steps, gamma=0.99, lambda_=0.95,
                    num_epochs=4, num_minibatches=4):
  """ Returns env runner for PPO """
  transforms = [GAE(policy, gamma=gamma, lambda_=lambda_, normalize=False)]
  if not policy.is_recurrent() and nenvs(env):
    transforms.append(MergeTimeBatch())
  runner = EnvRunner(env, policy, num_runner_steps, transforms=transforms)
  runner = TrajectorySampler(runner, num_epochs=num_epochs,
                             num_minibatches=num_minibatches,
                             transforms=[NormalizeAdvantages()])
  return runner
