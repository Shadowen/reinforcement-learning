from abc import abstractmethod


class Estimator:
    """
    Value Function approximator.
    """

    @abstractmethod
    def predict(self, state):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for

        Returns
            This returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        pass

    @abstractmethod
    def predict_batch(self, batch_state):
        """
        Makes value function predictions.

        Args:
            batch_state: batch of states to make predictions for

        Returns
            This returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        pass

    @abstractmethod
    def update(self, batch_state, batch_action, batch_target):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        pass
