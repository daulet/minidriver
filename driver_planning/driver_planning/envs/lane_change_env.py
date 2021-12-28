import random
from typing import Optional

from gym import error, spaces, utils
from gym.utils import seeding

from .car_following_env import CarFollowingEnv
from .car_following_env import CAR_HEIGHT, MAX_SPEED

class LaneChangeEnv(CarFollowingEnv):

  def _title(self):
    return "Lane Changing"

  def _num_lanes(self):
    return random.randint(2, 5)

  def _goal_position(self, ego):
    ego_lane = self._current_lane(ego.x)
    adj_lanes = set([ego_lane-1, ego_lane+1]) - set([-1, self.num_lanes])
    goal_lane = random.choice(list(adj_lanes))
    return (self._lane_left_boundary(goal_lane) + self._lane_left_boundary(goal_lane+1))/2, 0

  def _agent_position(self, goal, ego):
    speed = random.randint(1, MAX_SPEED)

    gx, _ = goal
    agent_lane = self._current_lane(gx)

    return (
      (self._lane_left_boundary(agent_lane) + self._lane_left_boundary(agent_lane+1))/2,
      random.randint(ego.y - 2*CAR_HEIGHT, ego.y),
      speed,
    )