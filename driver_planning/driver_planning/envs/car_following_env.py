from enum import Enum
import random
from typing import Optional

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame

from .car import Car

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
  action_space = spaces.Tuple([spaces.Discrete(3), spaces.Discrete(3)])

  def __init__(self):
    self.surface = None


  def step(self, action):
    accel, _ = action # TODO lateral not supported yet
    done = False

    #
    # Action
    #
    ego = self.agents[0]
    ego.update(accel, 0)

    #
    # Update
    #
    for car in self.agents:
      car.step()

    if ego.y < 0 or ego.y > SCREEN_HEIGHT:
      done = True

    #
    # Reward:
    #
    reward = 0
    # hitting a car -10**6
    ego_rect = self._car_rect(ego)
    for i in range(1, len(self.agents)):
      agent_rect = self._car_rect(self.agents[i])
      if ego_rect.colliderect(agent_rect):
        reward -= 1e6
        done = True
        break

    return None, reward , done, {} # observation, reward, done, info


  def reset(self, seed):
    random.seed(seed)

    self.num_lanes = random.randint(3, 5)

    ego_lane = random.randint(0, self.num_lanes-1)
    ego = Car(0,
            (self._lane_left_boundary(ego_lane) + self._lane_left_boundary(ego_lane+1))/2,
            random.randint(SCREEN_HEIGHT/2, SCREEN_HEIGHT - CAR_HEIGHT),
            5, # faster so it can catch up to car ahead if no action
          )
    self.agents = [ego]
    
    time_till_offscreen = ego.y // ego.speed

    for idx in range(1, 2):
      self.agents.append(
        Car(idx,
          (self._lane_left_boundary(ego_lane) + self._lane_left_boundary(ego_lane+1))/2,
          random.randint(time_till_offscreen+1, ego.y), # ahead of ego
          1, # slow so ego would catch it if no action
        )
      )


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
