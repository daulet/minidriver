import os
import sys
import time

import gym
from stable_baselines3 import PPO

def main(env_name, path):
  model = get_model(path)
  print("Testing model...")
  test(env_name, model, rounds=100) 


def get_model(path):
  print("Loading model from", path)
  return PPO.load(path)


def test(env_name, model, rounds=10, controllers=None):
  env = gym.make(env_name, debug=True, controllers=controllers)
  
  wins = 0
  for i in range(rounds):
    obs, rew = env.reset(seed=time.time()), 0
    done = False
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      rew += reward
      env.render(fps=240)
    wins += rew > 0.5
  print(f"Wins: {wins}/{rounds}")
  env.close()


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    assert paths, "No files in " + path
    return max(paths, key=os.path.getctime)


if __name__ == "__main__":
  args = sys.argv[1:]
  assert len(args) > 0, "Usage: python test.py <env-name> [<path-to-model>]"
  env_name = args[0]
  path = len(args) > 1 and args[1] or newest("checkpoints")
  main(f"driver_planning:{env_name}-v0", path)
