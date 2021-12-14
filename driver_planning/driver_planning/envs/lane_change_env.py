from typing import Optional
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class LaneChangeEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    pass

  def step(self, action):
    pass

  def reset(self):
    super().reset()
    pass

  def render(self, mode='human'):
    pass

  def close(self):
    pass