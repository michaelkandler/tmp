import numpy as np

# type hinting
from numeric.noise import Noise

from utils.logger import CustomLogger

my_logger = CustomLogger()


class Model:
    """
    Represents any dynamic model by implementing the differential equation in the form dx/dt=f(x,u,t).

    Used for simulation
    """

    def __init__(self, model_params: dict | None, input_params: dict | None):
        """
        Parameters
        ----------
        model_params : dict
            Dictionary of parameters inherent to the model

        input_params : dict
            Dictionary of the input parameters such as concentration of inflowing fluids
        """

        self.name = "abstract_model"

        # number of states an input of the model
        self.n_states = None
        self.n_inputs = None

        # parameterize model
        self.model_params = model_params
        self.input_params = input_params

        self.content = None

        self.noise_generator = None

        my_logger.debug(f"creating model...")
        my_logger.debug(f"model parameters: {self.model_params}, input parameters: {self.input_params}")

    def change_params(self, model_params=None, input_params=None) -> None:
        """
        Change model- and input parameters. Kept unchanged if not given.

        Parameters
        ----------
        model_params : dict
            Model parameters to replace

        input_params : dict
            Input parameters to replace

        Returns
        -------
        None
        """

        self.model_params = self.model_params if model_params is None else model_params
        self.input_params = self.input_params if input_params is None else input_params

        if model_params is not None:
            my_logger.debug(f"model parameters changed to: {self.model_params}")
        if input_params is not None:
            my_logger.debug(f"input parameters changed to: {self.input_params}")

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

    def stationary_point(self, *args: any) -> any:
        """
        Calculate stationary point of model with given inputs

        Parameters
        ----------
        args : any
            Input parameters necessary to calculate the stationer points

        Returns
        -------
        any
            State-space vector of stationary point
        """
        raise NotImplemented('Abstract model only')

    def homogenous_solution(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Analytical solution of the autonomous system with a given initial state

        Parameters
        ----------
        x0 : np.ndarray
            Initial state

        t : np.ndarray
            Time grid to calculate solution on

        Returns
        -------
        np.ndarray
            State trajectory along the given time grid
        """
        raise NotImplemented('Abstract model only')

    def __call__(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Calculate the function f(x, u, t) describing the dynamic model

        Parameters
        ----------
        x : np.ndarray
            Current state of the system
        u : np.ndarray
            Current input into the system
        t : np.ndarray
            Current time

        Returns
        -------
        np.ndarray
            Left hand side of the differential equation describing the model
        """
        raise NotImplemented("Abstract model only")

    def plot_process(self, *args, **kwargs) -> None:
        """
        Plot the process modeled with given data

        Parameters
        ----------
        args : any

        Returns
        -------
        None
        """
        raise NotImplemented("Abstract class only")

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'mod_par' or 0:   model parameters
            - 'in_par' or 1:    input parameters
            - str of used dicts

        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        elif item == "mod_par" or item == 0:
            return self.model_params
        elif item == "in_par" or item == 1:
            return self.input_params
        elif item in self.model_params.keys():
            return self.model_params[item]
        elif item in self.input_params.keys():
            return self.model_params[item]
        elif item == "shape" or item == 2:
            return self.n_states, self.n_inputs
        else:
            raise IndexError("Invalid index. Choose 'mod_par' (alternatively 0) or 'in_par' (alternatively 1) "
                             "to get parameters of model")

    def __str__(self):
        description = f"Model: {self.name}\n" \
                      f"  - model_params: {self.model_params}\n" \
                      f"  - input_params: {self.input_params}\n" \
                      f"  - model_shape: ({self.n_states}, {self.n_inputs})"

        return description

    def __repr__(self):
        description = (f"Model(name='{self.name}', model_params={self.model_params}, input_params={self.input_params}, "
                       f"model_shape=({self.n_states}, {self.n_inputs}))")

        return description


if __name__ == '__main__':
    example_model = Model({'mod_param': None}, {'input_param': None})
    print(repr(example_model))
