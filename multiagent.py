import argparse

from driver_planning.controller import Controller
from stable_baselines3 import PPO

from test import get_model, newest, test


def main(env_name, path1, path2, fps, render, rounds):
  model1, model2 = get_model(path1), get_model(path2)
  print("Testing model...")
  test(env_name, model1, controllers=[Controller(model2)], fps=fps, render=render, rounds=rounds) 


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate two models at the same time in specified environment.')
  parser.add_argument('env_name', type=str, help='Environment name')
  parser.add_argument('--fps', type=int, default=240, help='Frames per second, if --render is enabled.')
  parser.add_argument('--headless', default=False, action='store_true', help='Skip rendering.')
  parser.add_argument('--path1', type=str, default=None, help='Path to model that acts as the main model.')
  parser.add_argument('--path2', type=str, default=None, help='Path to model that controls the agent.')
  parser.add_argument('--rounds', type=int, default=100, help='Number of rounds to test.')
  args = parser.parse_args()

  if args.path1 is None:
    args.path1 = newest("checkpoints")
  if args.path2 is None:
    args.path2 = newest("checkpoints")

  main(f"driver_planning:{args.env_name}-v0", args.path1, args.path2, args.fps, not args.headless, args.rounds)
