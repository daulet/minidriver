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
EXTERNAL_LANE_LINE_WIDTH = 4
LANE_LINE_WIDTH = 2
LANE_COLOR = (255, 255, 255)

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 500
FPS = 30

OBSERVATION_UNDEFINED = 1e4

class CarFollowingEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}
  # (Acceleration, Lateral)
  action_space = spaces.MultiDiscrete([3, 3])
  observed_agents = 4
  observation_space = spaces.Dict({
    "goal":       spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "boundaries": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
    "dynamic":    spaces.Box(low=-np.inf, high=np.inf, shape=(observed_agents,3,), dtype=np.float32),
    "speed":      spaces.Discrete(MAX_SPEED+1),
  })
  undefined_agent = (OBSERVATION_UNDEFINED, OBSERVATION_UNDEFINED, OBSERVATION_UNDEFINED)
  agent_colors = [
    (50,  200, 0  ), # green
    (100, 200, 255), # blue    
    (255, 100, 100), # red
    (200, 200, 0  ), # yellow
    (60,  60,  60 ), # black
    (200, 0,   150), # purple
  ]

  def __init__(self, controllers=None, debug=False):
    self.surface = None
    if controllers is None:
      controllers = []
    self.provided_controllers = controllers
    if debug:
      self._print = print
    else:
      self._print = lambda *args: None


  def _state(self, agent_id):
    ego = self.agents[agent_id]
    distances = []
    for i in range(len(self.agents)):
      if i == agent_id:
        continue
      car = self.agents[i]
      distances.append((ego.x - car.x, ego.y - car.y, ego.speed - car.speed))

    # self.agents includes ego, hence the -1
    for i in range(len(self.agents)-1, CarFollowingEnv.observed_agents):
      distances.append(CarFollowingEnv.undefined_agent)

    assert self.goals[agent_id] is not None, "Can't generate observation for an agent without controller"
    gx, gy = self.goals[agent_id]
    lb = self._lane_left_boundary(0)
    rb = self._lane_left_boundary(self.num_lanes)
    return {
      "goal":       (ego.x-gx, ego.y-gy), # distance to goal;
      "boundaries": (ego.x-lb, rb-ego.x),           # (left, right) distance to road boundaries;
      "dynamic":    distances, # distances to dynamic agents; 
      "speed":      ego.speed,
    }


  def step(self, action):
    self.steps += 1
    prev_act = self.prev_act
    self.prev_act = action

    done = False

    actions = [None] * len(self.agents)
    actions[0] = action
    for idx, controller in enumerate(self.controllers):
      if controller is None:
        continue
      obs = self._state(agent_id=idx)
      actions[idx] = controller.act(obs)

    #
    # Action
    #
    for idx, act in enumerate(actions):
      if act is None:
        continue
      self.agents[idx].update(*act)

    #
    # Update
    #
    for car in self.agents:
      car.step()

    #
    # Rewards
    #
    ego_id = 0
    ego = self.agents[ego_id]
    if ego.x < 0 or ego.x > SCREEN_WIDTH:
      done = True
    if ego.y < 0 or ego.y > SCREEN_HEIGHT:
      # TODO might need to be revisited with multiple controls
      done = True

    #
    # Rewards:
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
        self._collided = True
        reward = -1
        done = True
        break

    if not done:
      if ego.x < self._lane_left_boundary(0) or ego.x > self._lane_left_boundary(self.num_lanes):
        reward = -0.5
      elif ego.speed == 0:
        reward = -0.5
      # if not driving in a lane
      elif self._current_lane(ego.x-CAR_WIDTH/2) != self._current_lane(ego.x+CAR_WIDTH/2):
        reward = -0.001 # has to be lower than changing direction
      elif ego_rect.collidepoint(*self.goals[ego_id]):
        self._goal_reached = True
        reward = 1
        # TODO might need to be revisited with multiple goals
        done = True
      elif prev_act is not None and not np.array_equal(prev_act, action):
        reward = 0
      else:
        reward = 0.001 # incentivize movement

    # limit ego that just stops
    if self.steps == 2*SCREEN_HEIGHT:
      done = True
    self.rewards += reward
    if done:
      if self._collided:
        self._print(f"{bcolors.FAIL}[FAILED]\tsteps: {self.steps},\treward: {self.rewards:7.3f}{bcolors.ENDC}")
      elif self._goal_reached:
        self._print(f"{bcolors.OKGREEN}[SUCCESS]\tsteps: {self.steps},\treward: {self.rewards:7.3f}{bcolors.ENDC}")
      else:
        self._print(f"{bcolors.WARNING}[NO CRASH]\tsteps: {self.steps},\treward: {self.rewards:7.3f}{bcolors.ENDC}")
    return self._state(agent_id=ego_id), reward , done, {} # observation, reward, done, info


  def reset(self, seed=None):
    if seed is None:
      seed = time.time()
    random.seed(seed)

    self._goal_reached = False
    self._collided = False
    self.steps = 0
    self.rewards = 0
    self.prev_act = None

    self.num_lanes = self._num_lanes()

    ego_lane = random.randint(0, self.num_lanes-1)
    ego = Car(0,
            (self._lane_left_boundary(ego_lane) + self._lane_left_boundary(ego_lane+1))/2,
            random.randint(SCREEN_HEIGHT/2, SCREEN_HEIGHT - CAR_HEIGHT),
            random.randint(1, MAX_SPEED),
          )
    self.agents = [ego]
    self.controllers = [None]
    self.goals = [self._goal_position(ego)]
    
    for idx in range(1, 1+self._num_agents()):      
      if len(self.provided_controllers) <= idx-1:
        self.controllers.append(None)
      else:
        self.controllers.append(self.provided_controllers[idx-1])

      x, y, speed = self._agent_position(self.goals[0], self.agents[0])
      agent = Car(idx, x, y, speed)
      self.agents.append(agent)

      if self.controllers[idx] is None:
        self.goals.append(None)
      else:
        self.controllers[idx].reset()
        self.goals.append(self._goal_position(agent))

    assert len(self.agents) == len(self.goals) == len(self.controllers)
    return self._state(agent_id=0)


  def render(self, mode='human', fps=FPS):
    
    if self.surface is None:
      pygame.init()
      pygame.display.set_caption(self._title())
      self.surface = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    
    pygame.event.get()

    self.surface.fill((150, 150, 150))
    for lane_id in range(self.num_lanes+1):
      lane_x = self._lane_left_boundary(lane_id)
      lane_width =  LANE_LINE_WIDTH
      if lane_id in [0, self.num_lanes]: # edge lanes
        lane_width = EXTERNAL_LANE_LINE_WIDTH
      pygame.draw.line(self.surface, LANE_COLOR, (lane_x, 0), (lane_x, SCREEN_HEIGHT), lane_width)

    for idx, goal in enumerate(self.goals):
      if goal is None:
        continue
      goal_rect = pygame.Rect(0, 0, CAR_WIDTH, CAR_HEIGHT)
      goal_rect.center = goal
      pygame.draw.rect(self.surface, CarFollowingEnv.agent_colors[idx], goal_rect, width=3)

    for agent in self.agents:
      self._render_car(agent)

    pygame.display.update()
    pygame.time.Clock().tick(fps)


  def _lane_left_boundary(self, lane_id):
    return (SCREEN_WIDTH - self.num_lanes * LANE_WIDTH)/ 2 + lane_id * LANE_WIDTH

  def _current_lane(self, x):
    return math.floor((x - self._lane_left_boundary(0)) / LANE_WIDTH)

  def _car_rect(self, car):
    rect = pygame.Rect(0, 0, CAR_WIDTH, CAR_HEIGHT)
    rect.center = (car.x, car.y)
    return rect

  def _render_car(self, agent):
    assert agent.id < len(self.agent_colors), "Add more agent colors"
    rect = self._car_rect(agent)
    color = CarFollowingEnv.agent_colors[agent.id]
    pygame.draw.rect(self.surface, color, rect)
    pygame.draw.rect(self.surface, (0,0,0), rect, width=2)


  def close(self):
    if self.surface:
      self.surface = None
      pygame.quit()

  def _title(self):
    return "Car Following"

  def _num_lanes(self):
    return random.randint(1, 5)

  def _num_agents(self):
    return 1

  def _goal_position(self, agent):
    agent_lane = self._current_lane(agent.x)
    return (self._lane_left_boundary(agent_lane) + self._lane_left_boundary(agent_lane+1))/2, 0

  def _agent_position(self, goal, ego):
    speed = random.randint(1, MAX_SPEED)

    time_till_offscreen = ego.y // MAX_SPEED
    away_from_ego = CAR_HEIGHT
    for s in range(ego.speed-speed+1):
      away_from_ego += s

    ego_lane = self._current_lane(ego.x)

    return (
      (self._lane_left_boundary(ego_lane) + self._lane_left_boundary(ego_lane+1))/2,
      random.randint(time_till_offscreen+1, ego.y-away_from_ego), # ahead of ego
      speed,
    )