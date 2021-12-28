## Install
```
pip install -r requirements.txt
pip install -e ./driver_planning
```

## Test
```
pytest
```

## Evolution

1. First trained model learned to let the lead car to get further away and then speed towards the goal, while not hitting the lead car.

To reproduce disable lateral movement in environment:
```
python test.py models/carfollowing_ppo_sandbagging
```

2. To avoid such sporadic behavior, we've added a negative reward for completely stopping, after which model learned to follow the lead car.

To reproduce disable lateral movement in environment:
```
python test.py models/carfollowing_ppo_following
```

3. When preparing for the next task, we've added ability to move laterally, after which the model learned to overtake the lead car to reach the goal faster.

```
python test.py models/carfollowing_ppo_overtaking
```