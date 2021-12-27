import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("driver_planning:car-following-v0", n_envs=8)

model = PPO(
  "MultiInputPolicy",
  env,
  verbose=1,
  policy_kwargs={"net_arch":[16, 8, 8]})
model.learn(total_timesteps=400000)
model.save("models/ppo_carfollowing_10000")
