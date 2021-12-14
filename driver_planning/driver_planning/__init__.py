from gym.envs.registration import register

register(
    id='car-following-v0',
    entry_point='driver_planning.envs:CarFollowingEnv',
)
register(
    id='lane-change-v0',
    entry_point='driver_planning.envs:LaneChangeEnv',
)
register(
    id='turn-right-v0',
    entry_point='driver_planning.envs:RightTurnEnv',
)
register(
    id='turn-left-v0',
    entry_point='driver_planning.envs:TurnLeftEnv',
)