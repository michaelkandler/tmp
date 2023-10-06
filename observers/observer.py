import numpy as np

# type hinting
from numeric.noise import Noise

from utils.logger import CustomLogger

my_logger = CustomLogger()


class Observer:
    """
    Abstract class to define an observer using a given model.

    Will be used to estimate state x̂ by using a model and an input y
    """

    def __init__(self, *args) -> None:
        """
        Parameters
        ----------
        args: any

        """
        self.name = "observer"

        # model to simulate
        self.model = None

        # Set initial values
        self.x0 = None

        # generate measurement noise
        self.noise_generator = None

        my_logger.debug(f"creating observer...")

    def __call__(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Estimate x by using y and u, will return x̂

        Parameters
        ----------
        y : np.ndarray
            Measurement of any kind 
        u : np.ndarray       
            Input
        Returns
        -------
        np.ndarray
            Estimated state x̂
        """
        raise NotImplemented('Abstract class only')

    def set_initial_guess(self, x0: np.ndarray) -> None:
        """
        Setting the initial guess for the state observer

        Parameters
        ----------
        x0 : np.ndarray
            initial guess for state observer

        Returns
        -------
        None
        """
        my_logger.debug(f"setting observer initial value {x0}")
        self.x0 = x0

    def set_noise_generator(self, generator: Noise) -> None:
        """
        Set noise generator to get noise output values.

        Parameters
        ----------
        generator : Noise
            Full configured noise generator, can be instantiated in module numeric.noise

        Returns
        -------
        None
        """
        my_logger.debug(f"setting noise to {generator} in controller {self.__repr__()}")
        self.noise_generator = generator

    def __getitem__(self, item):
        if item == "name":
            return self.name
        elif item == "model" or item == 0:
            return self.model
        elif item == "x0" or item == 1:
            return self.x0
        else:
            raise IndexError("Invalid index. Choose 'model' (0) for used model or 'x0' (1) for initial value")

    def __str__(self):
        description = f"Abstract observer\n" \
                      f"-----------\n" \
                      f"  - model: {self.model}" \
                      f"  - x0: {self.x0}"

        return description

    def __repr__(self):
        description = f"Observer()"

        return description


if __name__ == '__main__':
    raise NotImplemented
