from typing import Optional
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from enum import Enum

class Acceleration(Enum):
  NEUTRAL = 0
  SLOW_DOWN = 1
  ACCELERATE = 2

class Lateral(Enum):
  STRAIGHT = 0
  LEFT = 1
  RIGHT = 2

class CarFollowingEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}
  # (Acceleration, Lateral)
  action_space = spaces.Tuple([spaces.Discrete(3), spaces.Discrete(3)])

  def __init__(self):
    self.viewer = None


  def step(self, action):
    return None, 0, False, {} # observation, reward, done, info


  def reset(self):
    pass


  def render(self, mode='human'):
    screen_width = 600
    screen_height = 400

    if self.viewer is None:
      from gym.envs.classic_control import rendering

      self.viewer = rendering.Viewer(screen_width, screen_height)

    return self.viewer.render(return_rgb_array=mode == "rgb_array")


  def close(self):
    pass