import gym
import driver_planning

env = gym.make('driver_planning:car-following-v0')

env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
