import time

import gym
import driver_planning

env = gym.make('driver_planning:car-following-v0')
env.reset(seed = time.time())

total, done = 0, False
while not done:
    # take a random action
    _, reward, done, _ = env.step(env.action_space.sample())
    total += reward
    env.render()
print("Reward:", total)
input("Press Enter to continue...")
env.close()
