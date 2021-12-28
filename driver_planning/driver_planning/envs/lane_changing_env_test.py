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
        {'goal': (-35.0, 255), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 250), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 245), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 240), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 235), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 230), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 225), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 220), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 215), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
        {'goal': (-35.0, 210), 'boundaries': (122.5, 52.5), 'dynamic': (-35.0, -44, 0), 'speed': 5},
    ]

    assert total > 0.01

    env.close()