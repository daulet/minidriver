import time

import gym
import driver_planning

from .car import Acceleration, Lateral

def test_collision_if_no_reaction():
    env = gym.make('driver_planning:car-following-v0')
    env.reset(seed = time.time())

    total, done = 0, False
    while not done:
        _, reward, done, _ = env.step((Acceleration.NEUTRAL, Lateral.STRAIGHT))
        total += reward
        env.render(fps=240)
    env.close()

    assert reward == -1e6
