import os
import sys
import time

import gym
from stable_baselines3 import PPO

def main(path):
  print("Loading model from", path)
  model = PPO.load(path)
  print("Testing model...")
  test(model) 


def test(model):
  env = gym.make("driver_planning:car-following-v0", debug=True)
  
  rounds, wins = 10, 0
  for i in range(rounds):
    obs, rew = env.reset(seed=time.time()), 0
    done = False
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      rew += reward
      env.render(fps=240)
    wins += rew > 0
    print("reward:", rew)
  print(f"Wins: {wins}/{rounds}")
  env.close()


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


if __name__ == "__main__":
  args = sys.argv[1:]
  path = len(args) > 0 and args[0] or newest("models")
  main(path)
