import io
import sys
import time
from driver_planning.controller import Controller

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from test import test

def train(env_name):
  ITERATION = 10**5
  LIMIT = 10**7
  arch = [64, 64, 64]
  batch_size = 256
  gamma = 0.99
  learning_rate = 1e-3
  seed = int(time.time())
  self_play = 0.2

  full_env_name = f"driver_planning:{env_name}-v0"

  controller = Controller(None, self_play=self_play)
  env = make_vec_env(full_env_name, n_envs=16, seed=seed, env_kwargs={'controllers':[controller]})
  model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    policy_kwargs={"net_arch":arch},
    batch_size=batch_size,
    gamma=gamma,
    learning_rate=learning_rate,
    seed=seed,
    tensorboard_log=f"./tensorboard/{env_name}_ppo/",
  )

  for timesteps in range(ITERATION, LIMIT+1, ITERATION):
    if self_play > 0.0:
      with io.BytesIO() as bytes:
        with io.BufferedRandom(bytes) as buffer:
          print("Cloning model for self-play...")
          model.save(buffer)
          clone = PPO.load(buffer)
          controller.update(clone)

    print(f"Training model for {ITERATION} timesteps ({timesteps} total)")
    model.learn(
      total_timesteps=ITERATION,
      reset_num_timesteps=False,
      tb_log_name=f"arch{'-'.join(map(str, arch))}_batch{batch_size}_g{gamma}_lr{learning_rate}_sp{self_play}",
    )

    path = f"./checkpoints/{env_name}_ppo_arch{'-'.join(map(str, arch))}_batch{batch_size}_g{gamma}_lr{learning_rate}_sp{self_play}_{timesteps}"
    print(f"Saving to {path}")
    model.save(path)
    test(full_env_name, model, render=False, rounds=20)


if __name__ == "__main__":
  args = sys.argv[1:]
  assert len(args) > 0, "Usage: python train.py <env-name>"
  env_name = args[0]
  train(env_name)
