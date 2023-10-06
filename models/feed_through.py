import numpy as np

from models import Model


class FeedThrough(Model):
    """
    Feed through model to be used in an empty observer.

    Not quite sure what this would be good for ^^
    """

    def __init__(self):
        super().__init__({}, {})

        self.name = 'feed_through'
        self.n_states = 0

    def __call__(self, x: np.ndarray, u: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Just feeds through the given state without modification

        Parameters
        ----------
        x : np.ndarray
            current state to be returned
        u : np.ndarray
           current input, is ignored
        t : np.ndarray
            current time, is ignored

        Returns
        -------
        np.ndarray
            given state x
        """
        return x

    def plot_process(*args) -> None:
        raise NotImplemented("No plotter for feed-through model")

    def __str__(self):
        description = f"Simple feed through model\n"

        return description

    def __repr__(self):
        description = f"FeedThrough()"

        return description


if __name__ == '__main__':
    pass
