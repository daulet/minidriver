from enum import Enum
import enum
import itertools
import random
from typing import Optional

import gym
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from gym.utils import seeding

from .car import Car

class Acceleration(Enum):
  NEUTRAL = 0
  SLOW_DOWN = 1
  ACCELERATE = 2

class Lateral(Enum):
  STRAIGHT = 0
  LEFT = 1
  RIGHT = 2

CAR_WIDTH = 25
CAR_HEIGHT = 40

LANE_WIDTH = 35
LANE_LINE_WIDTH = 4

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 500

class CarFollowingEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}
  # (Acceleration, Lateral)
  action_space = spaces.Tuple([spaces.Discrete(3), spaces.Discrete(3)])

  def __init__(self):
    self.viewer = None


  def step(self, action):
    for car in self.agents:
      car.step()

    return None, 0, False, {} # observation, reward, done, info


  def reset(self, seed):
    random.seed(seed)

    self.num_lanes = random.randint(3, 5)

    self.agents = []
    for idx in range(random.randint(2, 6)):
      # TODO no collision checks
      self.agents.append(
        Car(idx,
          random.randint(CAR_WIDTH, SCREEN_WIDTH - CAR_WIDTH),
          random.randint(CAR_HEIGHT, SCREEN_HEIGHT - CAR_HEIGHT),
          random.randint(1, 3)))


  def render(self, mode='human'):
    
    if self.viewer is None:
      # Viewer shows only quarter of requrested view, i.e. middle point is top right corner
      self.viewer = rendering.Viewer(2*SCREEN_WIDTH, 2*SCREEN_HEIGHT)

      for lane_id in range(self.num_lanes+1):
        lane_x = (2*SCREEN_WIDTH - self.num_lanes * LANE_WIDTH)/ 2 + lane_id * LANE_WIDTH
        lane_line = rendering.FilledPolygon([
          (lane_x-LANE_LINE_WIDTH/2, 0), (lane_x-LANE_LINE_WIDTH/2, SCREEN_HEIGHT),
          (lane_x+LANE_LINE_WIDTH/2, 2*SCREEN_HEIGHT), (lane_x+LANE_LINE_WIDTH/2, 0)])
        if lane_id in [0, self.num_lanes]: # edge lanes
          lane_line.set_color(1, 1, 0)
        else:
          lane_line.set_color(0, 0, 0)
        self.viewer.add_geom(lane_line)

      self.car_trans = []
      for agent in self.agents:
        trans = self._render_car(agent)
        self.car_trans.append(trans)
      
    for agent, trans in zip(self.agents, self.car_trans):
      trans.set_translation(agent.x, agent.y)

    return self.viewer.render(return_rgb_array=mode == "rgb_array")


  def _render_car(self, car):
    l, r = car.x - CAR_WIDTH / 2,   car.x + CAR_WIDTH / 2
    t, b = car.y + CAR_HEIGHT / 2,  car.y - CAR_HEIGHT / 2
    cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    trans = rendering.Transform()
    cart.add_attr(trans)
    if car.id == 0: # ID 0 is Ego
      cart.set_color(0, 1, 0)
    else:
      cart.set_color(0, 0, 1)    
    self.viewer.add_geom(cart)
    return trans


  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None