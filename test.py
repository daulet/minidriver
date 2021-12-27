import time

import gym
from stable_baselines3 import PPO

def main():
  model = PPO.load("models/ppo_carfollowing_10000")
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


if __name__ == "__main__":
  main()
