import gym

env = gym.make('driver_planning:car-following-v0')

env.reset()
for _ in range(10):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
