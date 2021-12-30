

class Controller(object):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action