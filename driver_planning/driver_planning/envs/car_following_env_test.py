import time

import gym
import driver_planning

from .car import MAX_SPEED, Acceleration, Lateral

# Assert that if actor takes no action (change in acceleration or lateral)
# than ego hits the other car
def test_collision():
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = time.time())
    # slow down lead car so we can run into it
    env.agents[1].speed = 1

    total, done = 0, False
    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
        env.render(fps=1000)
    env.close()

    assert reward == -1e9

def test_collision_no_render():
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = time.time())
     # slow down lead car so we can run into it
    env.agents[1].speed = 1

    total, done = 0, False
    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
    env.close()

    assert reward == -1e9

def test_slow_achieves_goal():
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = time.time())
    # ensure ego speed == 1
    # TODO make sure this doesn't happen in training?
    env.agents[0].speed = 1

    total, done = 0, False
    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
        
    env.close()

    assert total > 0

def test_stopping_is_punished():
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = time.time())
    
    for i in range(MAX_SPEED):
        _, reward, _, _ = env.step((Acceleration.SLOW_DOWN, Lateral.STRAIGHT))
        
    env.close()

    assert reward == -1e5

def test_recorded_observations():
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = 0)

    actual_observations = []
    total, done = 0, False
    for _ in range(10):
        obs, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        actual_observations.append(obs)
        total += reward

    assert actual_observations == [
        {'dynamic': (0.0,
                        139,
                        2),
            'goal': (0.0,
                    255),
            'speed': 5},
        {'dynamic': (0.0,
                        137,
                        2),
            'goal': (0.0,
                    250),
            'speed': 5},
        {'dynamic': (0.0,
                        135,
                        2),
            'goal': (0.0,
                    245),
            'speed': 5},
        {'dynamic': (0.0,
                        133,
                        2),
            'goal': (0.0,
                    240),
            'speed': 5},
        {'dynamic': (0.0,
                        131,
                        2),
            'goal': (0.0,
                    235),
            'speed': 5},
        {'dynamic': (0.0,
                        129,
                        2),
            'goal': (0.0,
                    230),
            'speed': 5},
        {'dynamic': (0.0,
                        127,
                        2),
            'goal': (0.0,
                    225),
            'speed': 5},
        {'dynamic': (0.0,
                        125,
                        2),
            'goal': (0.0,
                    220),
            'speed': 5},
        {'dynamic': (0.0,
                        123,
                        2),
            'goal': (0.0,
                    215),
            'speed': 5},
        {'dynamic': (0.0,
                        121,
                        2),
            'goal': (0.0,
                    210),
            'speed': 5},
    ]

        
    env.close()