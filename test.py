import time

import gym
from stable_baselines3 import PPO

env = gym.make("driver_planning:car-following-v0")
model = PPO.load("models/ppo_carfollowing_10000")

obs = env.reset(seed=time.time())
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset(seed=time.time())

env.close()