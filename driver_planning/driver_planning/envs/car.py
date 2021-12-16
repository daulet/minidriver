
class Car(object):

    def __init__(self, id, x, y, speed) -> None:
        super().__init__()
        self.id = id
        self.x = x
        self.y = y
        self.speed = speed

    def step(self):
        self.y += self.speed
        return self.x, self.y