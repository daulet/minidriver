import time

import gym
import pytest

from .car import MAX_SPEED, Acceleration, Lateral

def gen_seed():
    # this will get logged when test fails, allowing easy repro
    return time.time()

@pytest.mark.parametrize('seed', [gen_seed()])
def test_no_collision(seed):
    env = gym.make('driver_planning:lane-changing-v0')
    env.reset(seed = seed)

    total, done = 0, False
    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
        env.render(fps=1000)
    env.close()
    # won't reach the goal, but won't crash either
    assert 0 < total < 1

@pytest.mark.parametrize('seed', [gen_seed()])
def test_no_collision_no_render(seed):
    env = gym.make('driver_planning:lane-changing-v0')
    env.reset(seed = seed)

    total, done = 0, False
    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
    env.close()
    # won't reach the goal, but won't crash either
    assert 0 < total < 1

def test_recorded_observations():
    env = gym.make('driver_planning:lane-changing-v0')
    env.reset(seed = 0)

    actual_observations = []
    total = 0
    for _ in range(10):
        obs, reward, _, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        actual_observations.append(obs)
        total += reward
        # print(f"{obs},")

    assert actual_observations == [
        {'goal': (-35.0, 257), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 4, -1), 'speed': 3}, 
        {'goal': (-35.0, 254), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 5, -1), 'speed': 3}, 
        {'goal': (-35.0, 251), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 6, -1), 'speed': 3}, 
        {'goal': (-35.0, 248), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 7, -1), 'speed': 3}, 
        {'goal': (-35.0, 245), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 8, -1), 'speed': 3}, 
        {'goal': (-35.0, 242), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 9, -1), 'speed': 3}, 
        {'goal': (-35.0, 239), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 10, -1), 'speed': 3},
        {'goal': (-35.0, 236), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 11, -1), 'speed': 3},
        {'goal': (-35.0, 233), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 12, -1), 'speed': 3},
        {'goal': (-35.0, 230), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, 13, -1), 'speed': 3},
    ]

    assert total > 0.01

    env.close()