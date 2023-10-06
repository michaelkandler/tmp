import numpy as np

from models import Model
from utils.logger import CustomLogger

logger = CustomLogger()


class Integrator(Model):
    """
    Integrator for testing of functionalities
    """
    def __init__(self, model_params: dict, input_params: dict) -> None:
        """
        Simple integrator for testing

        Parameters
        ----------
        model_params: dict
            integration constant for model
        input_params: dict
            not used, set to None
        """

        super().__init__(model_params, input_params)

        self.name = "integrator"
        self.n_states = 1
        self.n_inputs = 1

        logger.debug(f"model name: {self.name}, model shape: ({self.n_states}, {self.n_inputs})")

    def __call__(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Implement a simple input integration (f(x)=u)

        Parameters
        ----------
        x : np.ndarray
            current state
        u : np.ndarray
            current input
        t : float
            current time

        Returns
        -------
        np.ndarray
            Left hand side of differential equation
        """
        dxdt = self.model_params['k_i'] * u
        return dxdt


if __name__ == '__main__':
    example_model = Integrator({'k_i': 1}, {'input_param': None})
    print(example_model)
