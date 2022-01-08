import argparse
import time

import gym

def drive(env_name, rounds):
    env = gym.make(env_name)

    for _ in range(rounds):
        env.reset(seed = time.time())

        total, done = 0, False
        while not done:
            # take a random action
            _, reward, done, _ = env.step(env.action_space.sample())
            total += reward
            env.render()
        print("Reward:", total)
    
    env.close()
    input("Press Enter to continue...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drive randomly in specified environment.')
    parser.add_argument('env_name', type=str, help='Environment name')
    parser.add_argument('--rounds', type=int, default=100, help='Number of rounds to drive.')
    args = parser.parse_args()

    drive(f"driver_planning:{args.env_name}-v0", args.rounds)
