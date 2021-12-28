import time

import gym
import pytest

from .car import MAX_SPEED, Acceleration, Lateral

def gen_seed():
    # this will get logged when test fails, allowing easy repro
    return time.time()

# Assert that if actor takes no action (change in acceleration or lateral)
# than ego hits the other car
@pytest.mark.parametrize('seed', [gen_seed()])
def test_collision(seed):
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = seed)
    # slow down lead car so we can run into it
    env.agents[1].speed = 1

    total, done = 0, False
    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
        env.render(fps=1000)
    env.close()

    assert reward == -1

@pytest.mark.parametrize('seed', [gen_seed()])
def test_collision_no_render(seed):
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = seed)
     # slow down lead car so we can run into it
    env.agents[1].speed = 1

    total, done = 0, False
    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
    env.close()

    assert reward == -1

@pytest.mark.parametrize('seed', [gen_seed()])
def test_slow_achieves_goal(seed):
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = seed)

    total, done = 0, False

    # ensure ego speed == 1
    for i in range(MAX_SPEED):
        _, reward, _, _ = env.step((Acceleration.SLOW_DOWN, Lateral.STRAIGHT))
        total += reward
    _, reward, _, _ = env.step((Acceleration.ACCELERATE, Lateral.STRAIGHT))
    total += reward

    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
        
    env.close()

    assert total > 0

@pytest.mark.parametrize('seed', [gen_seed()])
def test_stopping_is_punished(seed):
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = seed)
    
    for i in range(MAX_SPEED):
        _, reward, _, _ = env.step((Acceleration.SLOW_DOWN, Lateral.STRAIGHT))
        
    env.close()

    assert reward == -0.5

@pytest.mark.parametrize('seed', [gen_seed()])
def test_off_lane_is_punished(seed):
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = seed)
    
    for _ in range(2):
        _, reward, _, _ = env.step((Acceleration.NEUTRAL, Lateral.LEFT))
    assert reward == -0.001

    for _ in range(4):
        _, reward, _, _ = env.step((Acceleration.NEUTRAL, Lateral.RIGHT))
    assert reward == -0.001

    env.close()

def test_recorded_observations():
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = 0)

    actual_observations = []
    total = 0
    for _ in range(10):
        obs, reward, _, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        actual_observations.append(obs)
        total += reward
        # print(f"{obs},")

    assert actual_observations == [
        {'goal': (0.0, 255), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 75, 2), 'speed': 5},
        {'goal': (0.0, 250), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 73, 2), 'speed': 5},
        {'goal': (0.0, 245), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 71, 2), 'speed': 5},
        {'goal': (0.0, 240), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 69, 2), 'speed': 5},
        {'goal': (0.0, 235), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 67, 2), 'speed': 5},
        {'goal': (0.0, 230), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 65, 2), 'speed': 5},
        {'goal': (0.0, 225), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 63, 2), 'speed': 5},
        {'goal': (0.0, 220), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 61, 2), 'speed': 5},
        {'goal': (0.0, 215), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 59, 2), 'speed': 5},
        {'goal': (0.0, 210), 'boundaries': (122.5, 17.5), 'dynamic': (0.0, 57, 2), 'speed': 5}
    ]

    assert total > 0.01

    env.close()
