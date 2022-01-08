import argparse
import os
import sys
import time

import gym
from stable_baselines3 import PPO


def get_model(path):
  print("Loading model from", path)
  return PPO.load(path)


def test(env_name, model, controllers=None, fps=240, render=False, rounds=10):
  env = gym.make(env_name, debug=True, controllers=controllers)
  
  failures, neutral, successes = 0, 0, 0
  for _ in range(rounds):
    obs, rew = env.reset(seed=time.time()), 0
    done = False
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      rew += reward
      if render:
        env.render(fps=fps)
    if env._collided:
      failures += 1
    elif env._goal_reached:
      successes += 1
    else:
      neutral += 1
  print(f"Success/Neutral/Failures: {successes}/{neutral}/{failures}")
  env.close()


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    assert paths, "No files in " + path
    return max(paths, key=os.path.getctime)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate model in specified environment.')
  parser.add_argument('env_name', type=str, help='Environment name')
  parser.add_argument('--fps', type=int, default=240, help='Frames per second, if --render is enabled.')
  parser.add_argument('--headless', default=False, action='store_true', help='Skip rendering.')
  parser.add_argument('--path', type=str, default=None, help='Path to model.')
  parser.add_argument('--rounds', type=int, default=100, help='Number of rounds to test.')
  args = parser.parse_args()

  if args.path is None:
    args.path = newest("checkpoints")

  model = get_model(args.path)
  print("Testing model...")
  test(f"driver_planning:{args.env_name}-v0", model, fps=args.fps, render=not args.headless, rounds=args.rounds) 
