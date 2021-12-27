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
        env.render(fps=240)
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
        env.render(fps=240)
        
    env.close()

    assert total > 0