import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from test import test

ITERATION = 1e5
arch = [64, 64, 64]
batch_size = 256
learning_rate = 1e-3
seed = int(time.time())

env_name = "car-following"
full_env_name = f"driver_planning:{env_name}-v0"

env = make_vec_env(full_env_name, n_envs=16, seed=seed)
model = PPO(
  "MultiInputPolicy",
  env,
  verbose=1,
  policy_kwargs={"net_arch":arch},
  batch_size=batch_size,
  learning_rate=learning_rate,
  seed=seed,
  tensorboard_log=f"./tensorboard/{env_name}_ppo_arch{'-'.join(map(str, arch))}_batch{batch_size}_lr{learning_rate}/",
)

timesteps = 0
while True:
  model.learn(total_timesteps=ITERATION, reset_num_timesteps=False)
  timesteps+=ITERATION
  model.save(f"./checkpoints/{env_name}_ppo_arch{'-'.join(map(str, arch))}_batch{batch_size}_lr{learning_rate}_{timesteps}")
  test(full_env_name, model)
