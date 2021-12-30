import random

from .car_following_env import CarFollowingEnv
from .car_following_env import CAR_HEIGHT, MAX_SPEED, SCREEN_HEIGHT

class ArenaEnv(CarFollowingEnv):

    # TODO handle no controllers
    def __init__(self, controllers=None, **kwargs):
        super().__init__(**kwargs)
        self.controllers = controllers

    def _title(self):
        return "Arena"

    def _num_lanes(self):
        return random.randint(2, 5)

    def _goal_position(self, ego):
        # Pick a random lane
        goal_lane = random.randint(0, self.num_lanes-1)
        return (self._lane_left_boundary(goal_lane) + self._lane_left_boundary(goal_lane+1))/2, 0

    # TODO: pass in [(agent, goal)], avoid spawning on top of other agents
    def _agent_position(self, goal, ego):
        speed = random.randint(1, MAX_SPEED)
        agent_lane = random.randint(0, self.num_lanes-1)

        return (
            (self._lane_left_boundary(agent_lane) + self._lane_left_boundary(agent_lane+1))/2,
            random.randint(SCREEN_HEIGHT/2, SCREEN_HEIGHT - CAR_HEIGHT),
            speed,
        )