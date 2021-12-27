from enum import Enum

class Acceleration(Enum):
  NEUTRAL = 0
  SLOW_DOWN = 1
  ACCELERATE = 2

class Lateral(Enum):
  STRAIGHT = 0
  LEFT = 1
  RIGHT = 2

LATERAL_STEP=5
MAX_SPEED=5

class Car(object):

    def __init__(self, id, x, y, speed) -> None:
        super().__init__()
        self.id = id
        self.x = x
        self.y = y
        self.speed = speed

        self.accel_inc = {
          Acceleration.NEUTRAL: 0,
          Acceleration.SLOW_DOWN: -1,
          Acceleration.ACCELERATE: 1
        }
        self.lateral_inc = {
          Lateral.STRAIGHT: 0,
          Lateral.LEFT: -LATERAL_STEP,
          Lateral.RIGHT: LATERAL_STEP,
        }


    def step(self):
        self.y -= self.speed
        return self.x, self.y


    def update(self, accel, lat):
        self.speed += self.accel_inc[Acceleration(accel)]
        self.speed = min(self.speed, MAX_SPEED)
        self.speed = max(self.speed, 0)

        self.x += self.lateral_inc[Lateral(lat)]
