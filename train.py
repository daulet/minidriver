from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from test import test

ITERATION = 1e5
arch = [32, 32, 32]

env = make_vec_env("driver_planning:car-following-v0", n_envs=8)
model = PPO(
  "MultiInputPolicy",
  env,
  verbose=1,
  policy_kwargs={"net_arch":arch},
  tensorboard_log=f"./tensorboard/carfollowing_ppo_{'-'.join(map(str, arch))}/",
)

timesteps = 0
while True:
  model.learn(total_timesteps=ITERATION, reset_num_timesteps=False)
  timesteps+=ITERATION
  model.save(f"./checkpoints/carfollowing_ppo_{'-'.join(map(str, arch))}_{timesteps}")
  test(model)
