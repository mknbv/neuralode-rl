# Adapted from https://github.com/openai/baselines
import gym
import numpy as np

class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  def __init__(self, eps=1e-4, shape=()):
    self.mean = np.zeros(shape, 'float64')
    self.var = np.ones(shape, 'float64')
    self.count = eps

  def update(self, x):
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    batch_count = x.shape[0]
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(self, batch_mean, batch_var, batch_count):
    self.mean, self.var, self.count = update_mean_var_count_from_moments(
        self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
  delta = batch_mean - mean
  tot_count = count + batch_count

  new_mean = mean + delta * batch_count / tot_count
  m_a = var * count
  m_b = batch_var * batch_count
  M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
  new_var = M2 / tot_count
  new_count = tot_count

  return new_mean, new_var, new_count


class Normalize(gym.Wrapper):
  """
  A vectorized wrapper that normalizes the observations
  and returns from an environment.
  """

  def __init__(self, env, obs=True, ret=True,
               clipobs=10., cliprew=10., gamma=0.99, eps=1e-8):
    super().__init__(env)
    self.obs_rms = (RunningMeanStd(shape=self.observation_space.shape)
                    if obs else None)
    self.ret_rms = RunningMeanStd(shape=()) if ret else None
    self.clipob = clipobs
    self.cliprew = cliprew
    self.ret = np.zeros(getattr(self.env.unwrapped, "nenvs", 1))
    self.gamma = gamma
    self.eps = eps

  def observation(self, obs):
    if not self.obs_rms:
      return obs
    self.obs_rms.update(obs)
    obs = (obs - self.obs_rms.mean) / (self.obs_rms.var + self.eps)
    obs = np.clip(obs, -self.clipob, self.clipob)
    return obs

  def step(self, action): # pylint: disable=method-hidden
    obs, rews, resets, info = self.env.step(action)
    self.ret = self.ret * self.gamma + rews
    obs = self.observation(obs)
    if self.ret_rms:
      self.ret_rms.update(self.ret)
      rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.eps),
                     -self.cliprew, self.cliprew)
    self.ret[resets] = 0.
    return obs, rews, resets, info

  def reset(self, **kwargs): # pylint: disable=method-hidden
    self.ret = np.zeros(self.env.unwrapped.nenvs)
    obs = self.env.reset(**kwargs)
    return self.observation(obs)
