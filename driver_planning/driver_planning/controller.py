import random
import threading
import torch

from driver_planning.envs.car import Acceleration, Lateral

class Controller(object):

    def __init__(self, model, self_play=1.0) -> None:
        super().__init__()
        self.mutex = threading.Lock()
        self.model = model
        self.self_play = self_play

    def reset(self):
        # :return: True if the controller will use model to act,
        #          False if the controller will use fixed actions
        self._controlled = random.random() < self.self_play
        self.act = self._act_model if self._controlled else self._act_fixed
        return self._controlled

    def update(self, model):
        self.mutex.acquire()
        self.model = model
        self.mutex.release()

    def act(self, obs):
        assert False, "Call reset() on controller first"

    def _act_fixed(self, obs):
        return (Acceleration.NEUTRAL, Lateral.STRAIGHT)

    def _act_model(self, obs):
        with torch.no_grad():
            action, _ = self.model.predict(obs, deterministic=True)
            return action
