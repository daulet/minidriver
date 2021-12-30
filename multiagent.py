import sys

from driver_planning.controller import Controller
from stable_baselines3 import PPO

from test import get_model, newest, test


def main(env_name, path1, path2):
  model1, model2 = get_model(path1), get_model(path2)
  print("Testing model...")
  test(env_name, model1, rounds=100, controllers=[Controller(model2)]) 


if __name__ == "__main__":
  args = sys.argv[1:]
  assert len(args) > 0, "Usage: python test.py <env-name> [<path-to-model>] [<path-to-model>]"
  env_name = args[0]
  path1 = len(args) > 1 and args[1] or newest("checkpoints")
  path2 = len(args) > 2 and args[2] or newest("checkpoints")
  main(f"driver_planning:{env_name}-v0", path1, path2)
