import gym
from stable_baselines3 import PPO

env = gym.make("driver_planning:car-following-v0")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("models/ppo_carfollowing_10000")
