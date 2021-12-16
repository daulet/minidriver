from enum import Enum

class Acceleration(Enum):
  NEUTRAL = 0
  SLOW_DOWN = 1
  ACCELERATE = 2

class Lateral(Enum):
  STRAIGHT = 0
  LEFT = 1
  RIGHT = 2

class Car(object):

    def __init__(self, id, x, y, speed) -> None:
        super().__init__()
        self.id = id
        self.x = x
        self.y = y
        self.speed = speed

        self.increment = {
            Acceleration.NEUTRAL: 0,
            Acceleration.SLOW_DOWN: -1,
            Acceleration.ACCELERATE: 1
        }

    def step(self):
        self.y += self.speed
        return self.x, self.y

    def update(self, accel, lat):
        assert lat == 0 # TODO lateral not supported yet

        self.speed += self.increment[Acceleration(accel)]
