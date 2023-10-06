"""
Base class for any type of controller
"""
import numpy as np

from numeric import Noise

from utils.logger import CustomLogger

my_logger = CustomLogger()


class Controller:
    """
    Abstract class to define a controller.

    Can be a state, a feed-forward, or an input-output controller.

    Attributes
    ----------
    name : str
      The name of the controller.

    Methods
    -------
    __init__(*args)
      Initializes the controller with the given parameters.

    __call__(x, t)
      Return the output corresponding to the time t and state/output x.

    __getitem__(item)
      Gets the specified attribute of the controller.

    __str__()
      Returns a human-readable description of the controller.

    __repr__()
      Returns a string representation of the controller.
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        *args : any
            Parameters to define the controller.
        """

        self.name = "abstract_controller"

        self.noise_generator = None

        my_logger.debug("creating controller...")

    # == builder options ===

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
        if not isinstance(generator, Noise):
            raise ValueError("Incompatible noise generator. Use a Noise object from numeric.noise")
        my_logger.debug(f"setting noise to {generator} in controller {self.__repr__()}")
        self.noise_generator = generator

    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Return the output corresponding to the time t and state/output x.

        Parameters
        ----------
        x : np.ndarray
            State x or measurement y, will not be used for feedforward controllers.
        t : np.ndarray
            Current time.

        Returns
        -------
        np.ndarray
            Controller output u.
        """
        raise NotImplemented('Abstract class only')

    # == miscellaneous ===

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item : str|int
            - 'name': Name of controller.

        Returns
        -------
        str
            The name of the controller.
        """
        if item == "name":
            return self.name
        else:
            raise IndexError("Invalid index")

    def __str__(self):
        description = f"Abstract controller\n" \
                      f"-----------\n"

        return description

    def __repr__(self):
        representation = f"Controller(name='{self.name}')"

        return representation


if __name__ == '__main__':
    example_controller = Controller()

    print(repr(example_controller))
