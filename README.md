This is a minimal environment to demonstrate that even a small network can learn to "drive" in reinforcement learning setting with simple and intuitive rewards. For example, lane boundaries are not known to the agent, it is learned from negative reward for driving between lanes (number of lanes changes, so lane position can't be easily memorized).

## Evolution

Initially model learned to be passive: let the lead car disappear from the view and then accelerate towards the goal.

![Sandbagging example](./assets/00%20sandbagging.gif)

To avoid such unnatural behavior, we've added a small negative reward for being stationary, after which model learned to follow the lead car, sometimes tailgating.

![Following example](./assets/01%20following.gif)

While preparing for the next environment, we've added ability to move ego laterally, after which the model learned to overtake the lead car to reach the goal faster.

![Overtaking example](./assets/02%20overtaking.gif)

## Getting Started

Setup environment
```
conda env create -f environment.yml
conda activate minidriver
pip install -e ./driver_planning
```

To try one of the pretrained models:
```
python test.py arena --path models/stable --fps=30 --rounds=10
```

Currently there are three environments: *car-following* (lead car between ego and goal), *lane-changing* (reaching a goal requires a lane change) and *arena* (arbitrary number of agents and goal position).