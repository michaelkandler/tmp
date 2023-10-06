from observers import Observer
from utils.logger import CustomLogger
my_logger = CustomLogger()


class EmptyObserver(Observer):
    """
    Empty instance for an observer. Requires x and u, returns x.

    Used if exact state-measurement is available
    """

    def __init__(self) -> None:
        super().__init__()

        self.name = "feed_through"

        my_logger.debug(f"{self.name}")

    def __call__(self, x, u):
        """
        Feed through observer, just return given x and u

        Parameters
        ----------
        x : np.ndarray
            current state
        u : np.ndarray
            current input

        Returns
        -------
        (np.ndarray, np.ndarray)
            Current state and input
        """
        return x, u

    def __str__(self):
        description = f"Empty observer\n" \
                      f"  - model:{self.model}"

        return description

    def __repr__(self):
        description = f"EmptyObserver()"

        return description


if __name__ == '__main__':
    raise NotImplemented
