import time

import gym
import pytest

from .car import Acceleration, Lateral

class FakeController(object):
    def act(self, obs):
        return Acceleration.NEUTRAL, Lateral.STRAIGHT

def gen_seed():
    # this will get logged when test fails, allowing easy repro
    return time.time()

@pytest.mark.parametrize('seed', [gen_seed()])
def test_consistency(seed):
    controllers = [FakeController()]
    env = gym.make('driver_planning:arena-v0', controllers=controllers)
    env.reset(seed = seed)

    prev_obs = None
    for _ in range(10):
        if prev_obs is not None:
            oth_obs = env._state(agent_id=1)

            x1, y1, s1 = prev_obs['dynamic']
            x2, y2, s2 = oth_obs['dynamic']
            assert x1+x2==0
            assert y1+y2==0
            assert s1+s2==0

            assert prev_obs['speed'] - oth_obs['speed'] == s1 
            assert oth_obs['speed'] - prev_obs['speed'] == s2

        prev_obs, _, _, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))

def test_recorded_observations():
    controllers = [FakeController()]
    env = gym.make('driver_planning:arena-v0', controllers=controllers)
    env.reset(seed = 0)

    actual_observations = []
    total = 0
    for _ in range(10):
        obs, reward, _, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        actual_observations.append(obs)
        total += reward
        # print(f"{obs},")

    assert actual_observations == [
        {'goal': (-35.0, 257), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -189, -1), 'speed': 3},
        {'goal': (-35.0, 254), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -188, -1), 'speed': 3},
        {'goal': (-35.0, 251), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -187, -1), 'speed': 3},
        {'goal': (-35.0, 248), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -186, -1), 'speed': 3},
        {'goal': (-35.0, 245), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -185, -1), 'speed': 3},
        {'goal': (-35.0, 242), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -184, -1), 'speed': 3},
        {'goal': (-35.0, 239), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -183, -1), 'speed': 3},
        {'goal': (-35.0, 236), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -182, -1), 'speed': 3},
        {'goal': (-35.0, 233), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -181, -1), 'speed': 3},
        {'goal': (-35.0, 230), 'boundaries': (122.5, 52.5), 'dynamic': (0.0, -180, -1), 'speed': 3},
    ]

    assert total > 0.01

    env.close()