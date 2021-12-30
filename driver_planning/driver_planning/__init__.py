from gym.envs.registration import register

register(
    id='arena-v0',
    entry_point="driver_planning.envs:ArenaEnv",
)
register(
    id='car-following-v0',
    entry_point='driver_planning.envs:CarFollowingEnv',
)
register(
    id='lane-changing-v0',
    entry_point='driver_planning.envs:LaneChangingEnv',
)
register(
    id='turn-right-v0',
    entry_point='driver_planning.envs:RightTurnEnv',
)
register(
    id='turn-left-v0',
    entry_point='driver_planning.envs:TurnLeftEnv',
)