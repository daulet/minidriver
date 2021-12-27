import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from test import test

ITERATION = 1e5
arch = [16, 16, 16]

env = make_vec_env("driver_planning:car-following-v0", n_envs=8)
model = PPO(
  "MultiInputPolicy",
  env,
  verbose=1,
  policy_kwargs={"net_arch":arch})

timesteps = 0
while True:
  model.learn(total_timesteps=ITERATION)
  timesteps+=ITERATION
  model.save(f"models/carfollowing_ppo_{'-'.join(map(str, arch))}_{timesteps}")
  test(model)
