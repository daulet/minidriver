import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from test import test

ITERATION = 1e5
arch = [32, 32, 32]
batch_size = 256
seed = int(time.time())

env = make_vec_env("driver_planning:car-following-v0", n_envs=16, seed=seed)
model = PPO(
  "MultiInputPolicy",
  env,
  verbose=1,
  batch_size=batch_size,
  policy_kwargs={"net_arch":arch},
  seed=seed,
  tensorboard_log=f"./tensorboard/carfollowing_ppo_arch{'-'.join(map(str, arch))}_batch{batch_size}/",
)

timesteps = 0
while True:
  model.learn(total_timesteps=ITERATION, reset_num_timesteps=False)
  timesteps+=ITERATION
  model.save(f"./checkpoints/carfollowing_ppo_arch{'-'.join(map(str, arch))}_batch{batch_size}_{timesteps}")
  test(model)
