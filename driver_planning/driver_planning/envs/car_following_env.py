from enum import Enum
import math
import random
import time
from typing import Optional

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pygame

from .car import MAX_SPEED, Car
from .colors import bcolors

CAR_WIDTH = 25
CAR_HEIGHT = 40

LANE_WIDTH = 35
LANE_LINE_WIDTH = 4

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 500
FPS = 30

class CarFollowingEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}
  # (Acceleration, Lateral)
  action_space = spaces.MultiDiscrete([3, 3])
  observation_space = spaces.Dict({
    "goal":       spaces.Box(low=0,       high=np.inf, shape=(2,), dtype=np.float32),
    "boundaries": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "dynamic":    spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
    "speed":      spaces.Discrete(MAX_SPEED+1),
  })

  def __init__(self, debug=False):
    self.surface = None
    if debug:
      self._print = print
    else:
      self._print = lambda *args: None


  def _state(self):
    ego = self.agents[0]
    distances = []
    for i in range(1, len(self.agents)):
      car = self.agents[i]
      distances.append((ego.x - car.x, ego.y - car.y, ego.speed - car.speed))

    gx, gy = self.goal
    lb = self._lane_left_boundary(0)
    rb = self._lane_left_boundary(self.num_lanes)
    return {
      "goal":       (abs(ego.x-gx), abs(ego.y-gy)), # distance to goal;
      "boundaries": (ego.x-lb, rb-ego.x),           # (left, right) distance to road boundaries;
      # TODO fix shape of this box to allow multiple distances
      "dynamic":    distances[0],                      # distances to dynamic agents; 
      "speed":      ego.speed,
    }


  def step(self, action):
    self.steps += 1

    accel, lat = action
    done = False

    #
    # Action
    #
    ego = self.agents[0]
    ego.update(accel, lat)

    #
    # Update
    #
    for car in self.agents:
      car.step()

    if ego.y < 0 or ego.y > SCREEN_HEIGHT:
      done = True

    #
    # Observation
    #
    states = self._state()

    #
    # Reward
    # hitting a car: -1.0
    # complete stop: -0.5
    # going off road: -0.5
    # any movement: 0.001
    # reaching a goal: 1.0
    reward = 0
    ego_rect = self._car_rect(ego)

    for i in range(1, len(self.agents)):
      agent_rect = self._car_rect(self.agents[i])
      if ego_rect.colliderect(agent_rect):
        reward = -1
        done = True
        break

    if not done:
      if ego.x < self._lane_left_boundary(0) or ego.x > self._lane_left_boundary(self.num_lanes):
        reward = -0.5
      # if not driving in a lane
      elif self._current_lane(ego.x-CAR_WIDTH/2) != self._current_lane(ego.x+CAR_WIDTH/2):
        reward = -0.01
      elif ego.speed == 0:
        reward = -0.5
      elif ego_rect.collidepoint(*self.goal):
        reward = 1
        done = True
      else:
        reward = 0.001 # incentivize movement

    # limit ego that just stops
    if self.steps == SCREEN_HEIGHT:
      done = True
    self.rewards += reward
    if done:
      if reward == 1:
        self._print(f"{bcolors.OKGREEN}[SUCCESS]\tsteps: {self.steps},\treward: {self.rewards:7.3f}{bcolors.ENDC}")
      else:
        self._print(f"{bcolors.FAIL}[FAILED]\tsteps: {self.steps},\treward: {self.rewards:7.3f}{bcolors.ENDC}")

    return states, reward , done, {} # observation, reward, done, info


  def reset(self, seed=None):
    if seed is None:
      seed = time.time()
    random.seed(seed)

    self.steps = 0
    self.rewards = 0

    self.num_lanes = random.randint(3, 5)
    ego_lane = random.randint(0, self.num_lanes-1)

    self.goal = ((self._lane_left_boundary(ego_lane) + self._lane_left_boundary(ego_lane+1))/2, 0)

    ego = Car(0,
            (self._lane_left_boundary(ego_lane) + self._lane_left_boundary(ego_lane+1))/2,
            random.randint(SCREEN_HEIGHT/2, SCREEN_HEIGHT - CAR_HEIGHT),
            5, # faster so it can catch up to car ahead if no action
          )
    self.agents = [ego]
    
    time_till_offscreen = ego.y // ego.speed

    for idx in range(1, 2):
      speed = random.randint(1, MAX_SPEED)
      away_from_ego = CAR_HEIGHT
      for s in range(ego.speed-speed+1):
        away_from_ego += s
      self.agents.append(
        Car(idx,
          (self._lane_left_boundary(ego_lane) + self._lane_left_boundary(ego_lane+1))/2,
          random.randint(time_till_offscreen+1, ego.y-away_from_ego), # ahead of ego
          speed,
        )
      )
    return self._state()


  def render(self, mode='human', fps=FPS):
    
    if self.surface is None:
      pygame.init()
      pygame.display.set_caption("Car Following")
      self.surface = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    
    self.surface.fill((255, 255, 255))
    for lane_id in range(self.num_lanes+1):
      lane_x = self._lane_left_boundary(lane_id)
      color = (0, 0, 0)
      if lane_id in [0, self.num_lanes]: # edge lanes
        color = (255, 255, 0)
      pygame.draw.line(self.surface, color, (lane_x, 0), (lane_x, SCREEN_HEIGHT), LANE_LINE_WIDTH)

    for agent in self.agents:
      self._render_car(agent)

    pygame.display.update()
    pygame.time.Clock().tick(fps)


  def _lane_left_boundary(self, lane_id):
    return (SCREEN_WIDTH - self.num_lanes * LANE_WIDTH)/ 2 + lane_id * LANE_WIDTH

  def _current_lane(self, x):
    return int((x - self._lane_left_boundary(0)) / LANE_WIDTH)

  def _car_rect(self, car):
    rect = pygame.Rect(0, 00, CAR_WIDTH, CAR_HEIGHT)
    rect.center = (car.x, car.y)
    return rect

  def _render_car(self, car):
    rect = self._car_rect(car)
    color = (0, 0, 255)
    if car.id == 0: # ID 0 is Ego
      color = (0, 255, 0)
    pygame.draw.rect(self.surface, color, rect)


  def close(self):
    if self.surface:
      self.surface = None
      pygame.quit()
