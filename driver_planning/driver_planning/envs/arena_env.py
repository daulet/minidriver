import random

from .car_following_env import CarFollowingEnv
from .car_following_env import CAR_HEIGHT, MAX_SPEED, SCREEN_HEIGHT

class ArenaEnv(CarFollowingEnv):

    def _title(self):
        return "Arena"

    def _num_lanes(self):
        return random.randint(2, 5)

    def _goal_position(self, ego):
        # Pick a random lane
        goal_lane = random.randint(0, self.num_lanes-1)
        return (self._lane_left_boundary(goal_lane) + self._lane_left_boundary(goal_lane+1))/2, 0

    # TODO: pass in [(agent, goal)]
    def _agent_position(self, goal, ego):
        speed = random.randint(1, MAX_SPEED)
        agent_lane = random.randint(0, self.num_lanes-1)

        if agent_lane == self._current_lane(ego.x):
            away_from_ego = CAR_HEIGHT
            for s in range(abs(ego.speed-speed)+1):
                away_from_ego += s
            if ego.y >= away_from_ego+SCREEN_HEIGHT/2:
                pos_y = random.randint(SCREEN_HEIGHT/2, ego.y-away_from_ego)
            else:
                pos_y = random.randint(ego.y+away_from_ego, SCREEN_HEIGHT-CAR_HEIGHT)
        else:
            pos_y = random.randint(SCREEN_HEIGHT/2, SCREEN_HEIGHT-CAR_HEIGHT)
        return (
            (self._lane_left_boundary(agent_lane) + self._lane_left_boundary(agent_lane+1))/2,
            pos_y,
            speed,
        )