import time

from driver_planning.envs.arena_env import ArenaEnv
from driver_planning.envs.car_following_env import CarFollowingEnv
import gym
import pytest

from .car import Acceleration, Car, Lateral

class FakeController(object):
    def reset(self):
        pass

    def act(self, obs):
        return ArenaEnv.action_space.sample()

def gen_seed():
    # this will get logged when test fails, allowing easy repro
    return time.time()

def test_consistency():
    seed = 2 # fixed seed to guarantee presence of agent
    controllers = [FakeController() for _ in range(CarFollowingEnv.observed_agents)]
    env = gym.make('driver_planning:arena-v0', controllers=controllers)
    env.action_space.seed(seed = seed)
    env.reset(seed = seed)

    for idx in range(len(env.agents)):
        env.reset(seed = seed)
        
        prev_obs = None
        for _ in range(10):
            if prev_obs is not None:
                oth_obs = env._state(agent_id=idx)

                # observation for idx's agent from ego perspective
                x1, y1, s1 = prev_obs['dynamic'][idx]
                # observation for ego from idx's perspective
                x2, y2, s2 = oth_obs['dynamic'][0]
                assert x1+x2==0
                assert y1+y2==0
                assert s1+s2==0

                assert prev_obs['speed'] - oth_obs['speed'] == s1
                assert oth_obs['speed'] - prev_obs['speed'] == s2

        prev_obs, _, _, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))

@pytest.mark.parametrize('seed', [gen_seed()])
def test_reset(seed):
    env = gym.make('driver_planning:arena-v0', controllers=[])
    env.action_space.seed(seed=int(seed))
    
    env.reset(seed=seed)
    c1, g1 = env.controllers, env.goals
    env.reset(seed=seed)
    c2, g2 = env.controllers, env.goals
    assert c1 == c2
    assert g1 == g2

    env._goal_reached = True
    env._collided = True
    env.reset(seed=seed)
    assert not env._goal_reached
    assert not env._collided

def test_spawn_point():
    seed = 2 # fixed seed to guarantee presence of agent
    controllers = [FakeController()]
    env = gym.make('driver_planning:arena-v0', controllers=controllers)
    env.action_space.seed(seed=int(seed))
    env.reset(seed = seed)

    assert len(env.agents) >= 2
    r1, r2 = env._car_rect(env.agents[0]), env._car_rect(env.agents[1])
    assert not r1.colliderect(r2)

def test_recorded_observations():
    controllers = [FakeController()]
    env = gym.make('driver_planning:arena-v0', controllers=controllers)
    env.action_space.seed(seed=0)
    env.reset(seed = 0)

    actual_observations = []
    total = 0
    for _ in range(10):
        obs, reward, _, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        actual_observations.append(obs)
        total += reward
        # print(f"{obs},")

    assert actual_observations == [
        {'goal': (-35.0, 257), 'boundaries': (122.5, 52.5), 'dynamic': [(30.0, -111, -1), (70.0, -117, -2), (35.0, -26, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                               
        {'goal': (-35.0, 254), 'boundaries': (122.5, 52.5), 'dynamic': [(30.0, -111, 0), (70.0, -115, -2), (35.0, -27, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                                
        {'goal': (-35.0, 251), 'boundaries': (122.5, 52.5), 'dynamic': [(25.0, -111, 0), (70.0, -113, -2), (35.0, -28, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                                
        {'goal': (-35.0, 248), 'boundaries': (122.5, 52.5), 'dynamic': [(20.0, -112, 1), (70.0, -111, -2), (35.0, -29, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                                
        {'goal': (-35.0, 245), 'boundaries': (122.5, 52.5), 'dynamic': [(25.0, -113, 1), (70.0, -109, -2), (35.0, -30, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                                
        {'goal': (-35.0, 242), 'boundaries': (122.5, 52.5), 'dynamic': [(20.0, -114, 1), (70.0, -107, -2), (35.0, -31, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                                
        {'goal': (-35.0, 239), 'boundaries': (122.5, 52.5), 'dynamic': [(25.0, -116, 2), (70.0, -105, -2), (35.0, -32, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                                
        {'goal': (-35.0, 236), 'boundaries': (122.5, 52.5), 'dynamic': [(20.0, -117, 1), (70.0, -103, -2), (35.0, -33, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                                
        {'goal': (-35.0, 233), 'boundaries': (122.5, 52.5), 'dynamic': [(15.0, -119, 2), (70.0, -101, -2), (35.0, -34, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3},                                                
        {'goal': (-35.0, 230), 'boundaries': (122.5, 52.5), 'dynamic': [(10.0, -121, 2), (70.0, -99, -2), (35.0, -35, 1), (10000.0, 10000.0, 10000.0)], 'speed': 3}, 

    ]

    assert total > 0.01

    env.close()