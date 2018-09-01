import numpy as np

from PreDQN.estimator import Estimator


class OracleEstimator(Estimator):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        v = np.sign(state[1])
        i = 0 if v <= 0 else 2
        a = [0, 0, 0]
        a[i] = 1
        return a

    def predict_batch(self, batch_state):
        return [self.predict(s) for s in batch_state]

    def update(self, batch_state, batch_action, batch_target):
        pass

    def copy_params(self):
        pass

    def save(self, directory):
        pass

    def load(self, directory):
        pass
