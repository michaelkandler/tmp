"""
This representation of the controlled plant will be further developed into:

SimulatedControlledPlant etc.
"""

from typing import Tuple, Any, List

import numpy as np

from controllers import Controller
from observers import Observer, EmptyObserver
from models import Model
from utils.logger import CustomLogger

my_logger = CustomLogger()

class ControlledPlant:
    """
    A class representing a controlled plant, which can be a model or a physical reactor with an implemented controller
    and observer.

    This class allows users to set the plant, controller, and observer for the controlled system. The system can then be
    run through the __call__ method, which performs the control process using the given controller and observer.

    Parameters
    ----------
    None

    Attributes
    ----------
    plant : Model or Any
        The plant to be controlled, can be either a Model instance or any other interface the controller can use.
    observer : Observer
        The observer used for state estimation.
    controller : Controller
        The controller used for control action calculation.
    x0 : np.ndarray or None
        The initial state vector for the integration process. None if not set.
    t : list[float]
        List of time values used for tracking inputs and state estimations.
    u : list[np.ndarray]
        List of control inputs applied to the plant during the control process.
    x_hat : list[np.ndarray]
        List of estimated state vectors calculated by the observer during the control process.
    track_hist : bool
        If True, tracking of inputs and state estimations is enabled.
    n_states : int
        Number of states in the plant (system).
    n_inputs : int
        Number of control inputs for the plant (system).

    Methods
    -------
    __call__(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        Runs the controlled plant for the given state vector and time.

    set_plant(plant: Any, model_params: dict = None, input_params: Any = None) -> None:
        Set the plant to be used by the controlled plant.

    set_controller(controller: Controller) -> None:
        Set the controller to be used by the controlled plant.

    set_observer(observer: Observer) -> None:
        Set an observer to be used by the controlled plant.

    set_initial_value(x0: np.ndarray) -> None:
        Set the initial values for the integration in the controlled plant.

    track_history(track: bool) -> None:
        Enable or disable tracking of input and state estimation.

    return_history(t: np.ndarray, reset_history: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        Return the interpolated control history over a given time vector, optionally resetting it.

    reset_system(x0: np.ndarray = None) -> None:
        Reset the controlled plant's history and optionally set a new initial state.

    __getitem__(item) -> Any:
        Get specific attributes of the controlled plant.

    __repr__() -> str:
        Returns a string representation of the controlled plant.

    """

    def __init__(self):
        my_logger.debug(f"creating {ControlledPlant.__name__}")

        # plant to control (can also be the model for simulation)
        self.plant = None
        self.controller = None
        self.observer = EmptyObserver()

        self.n_states, self.n_inputs = 5, 4

        # initial state value
        self.x0 = None

        # tracking of inputs and times, used for interpolation later
        self.t = []
        self.u = []
        self.x_hat = []

        # current state of the plant
        self.x = None

        # history can be deactivated to avert memory issues
        self.track_hist = True

    def __call__(self, x: np.ndarray | None, t: np.ndarray) -> np.ndarray:
        """
        Callable of the controlled plant. Runs the process

        Parameters
        ----------
        x : np.ndarray | None
            The current state vector give None if not available
        t : np.ndarray
            The current time.

        Returns
        -------
        np.ndarray
            The estimated state vector.
        """
        if self.controller is None:
            my_logger.error("no controller set, aborting...")
            raise AttributeError("No controller set. Use set_controller() methode")
        if self.observer is None:
            my_logger.error("no observer set, aborting...")
            raise AttributeError("No observer set. Use set_observer() methode")
        if self.plant is None:
            my_logger.error("no plant set, aborting...")
            raise AttributeError("No controller set. Use set_plant() methode")
        # calculate us for the plant
        u = self.controller(x, t)

        # get values from plant (measurements or simulation with model)
        dxdt = self.plant(x, u, t)

        # calculate state with observer
        x_hat, u = self.observer(x, u)

        # append to history if tracking is activated
        if self.track_hist:
            self.u.append(u)
            self.t.append(t)
            self.x_hat.append(x_hat)

        return dxdt

    # === building options ===

    def set_plant(self, plant: Any, model_params: dict = None, input_params: Any = None) -> None:
        """
        Set the plant to be controlled

        Parameters
        ----------
        plant : Any
            Plant to be used, can be any kind of interface that the controller can use
        model_params : dict, optional
            Parameters of the calibrated model, by default None.
        input_params : Any, optional
            Parameters for the model input/output, by default None.
        """
        my_logger.debug(f"setting plant {repr(plant)} to {ControlledPlant.__name__}")
        self.plant = plant if isinstance(plant, Model) else plant(model_params, input_params)
        self.n_states, self.n_inputs = self.plant.n_states, self.plant.n_inputs

    def set_controller(self, controller: Controller) -> None:
        """
        Set the controller to be used by the controlled plant.

        Parameters
        ----------
        controller : Controller
            The controller to be used.
        """
        if not isinstance(controller, Controller):
            raise ValueError(f"{controller} is no instance of {Controller.__name__}")
        my_logger.debug(f"providing {repr(controller)} to {ControlledPlant.__name__}")
        self.controller = controller

    def set_observer(self, observer: Observer) -> None:
        """
        Set an observer to be used by the controlled plant.

        Parameters
        ----------
        observer : Observer
            The observer to be used.
        """
        if not isinstance(observer, Observer):
            raise ValueError(f"{observer} is no instance of {Observer.__name__}")
        my_logger.debug(f"providing {repr(observer)} to {repr(ControlledPlant.__name__)}")
        self.observer = observer

    def set_initial_value(self, x0: np.ndarray) -> None:
        """
        Set the initial values for the controlled plant and its observer.

        Parameters
        ----------
        x0 : numpy.ndarray
           Initial state vector
        """
        my_logger.debug(f"setting {ControlledPlant.__name__} initial value to {x0}")
        self.x0 = x0
        self.observer.set_initial_guess(x0)

    def track_history(self, track: bool):
        """
        Enable or disable tracking of input and state estimation.

        Parameters
        ----------
        track : bool
           Whether to enable tracking.
        """
        my_logger.debug(f"history tracking: {'enabled' if track else 'disabled'}")
        self.track_hist = track

    # === access ===

    def return_history(self, t: np.ndarray, reset_history=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the interpolated control history over a given time vector, optionally resetting it.

        Parameters
        ----------
        t : np.ndarray
           Time vector to interpolate values over
        reset_history : bool, optional
           Whether to reset control history. Default is True.

        Returns
        -------
        tuple of np.ndarray
           Input history with corresponding time vector `(t, u)`.
        """
        # interpolate and return
        my_logger.debug("interpolating and returning history...")
        t, u, x_hat = self._interpolate_history(t)

        if reset_history:
            self.reset_system()

        return t, u

    def reset_system(self, x0: np.ndarray = None) -> None:
        """
        Reset controller to initial state.
        """
        my_logger.debug("resetting controlled plant history...")
        self.t = []
        self.u = []
        self.x_hat = []

    # === helper functions ===

    def _interpolate_history(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate input history over a given time vector and return a tuple containing the interpolated values and
        the corresponding time vector `(t, u)`.

        Parameters
        ----------
        t : np.ndarray
           Time vector to perform the interpolation on.

        Returns
        -------
        tuple of np.ndarray
           Interpolated input and its corresponding time vector, as a tuple of two numpy arrays `(t, u)`.
        """
        # don't interpolate if no input was given
        if self.u is None:
            my_logger.info('No input data available.')
            return t, np.asarray([]), np.asarray([])

        my_logger.debug("doing interpolation...")

        # convert to numpy
        u_sim = np.asarray(self.u)
        x_hat_sim = np.asarray(self.x_hat)
        t_sim = np.asarray(self.t)

        # sort numpy arrays
        t_sim_arg = np.argsort(t_sim)
        t_sim, u_sim, x_hat_sim = t_sim[t_sim_arg], u_sim[t_sim_arg], x_hat_sim[t_sim_arg]

        # instantiate output vector
        u_res = np.empty((t.shape[0], 0))
        x_hat_res = np.empty((t.shape[0], 0))

        # interpolation
        for i in range(u_sim.shape[1]):
            u_int = np.expand_dims(np.interp(t, t_sim, u_sim[:, i]), axis=1)
            u_res = np.append(u_res, u_int, axis=1)

        # interpolation
        for i in range(x_hat_sim.shape[1]):
            x_hat_int = np.expand_dims(np.interp(t, t_sim, x_hat_sim[:, i]), axis=1)
            x_hat_res = np.append(x_hat_res, x_hat_int, axis=1)

        return t, u_res, x_hat_res

    # === miscellaneous ===

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'plant' or 0: plant
            - 'x0' or 1: initial state
            - 'u' or 2: recorded input with corresponding time as tuple (t, u)
            - 'x_hat' or 3: recorded estimated states with corresponding time as tuple (t, x_hat)

        Returns
        -------
        str
            Requested item
        """
        if item == 'model' or item == 0:
            return self.plant
        elif item == 'x0' or item == 1:
            return self.x0
        elif item == 'u' or item == 2:
            return self.t, self.u
        elif item == 'x_hat' or item == 3:
            return self.t, self.x_hat
        else:
            raise IndexError("Invalid index. Choose 'model' (0), 'x0' (1), 'u' (2) or 'x_hat' (3) to get access data")

    def __str__(self):
        description = f"Controlled plant\n" \
                      f"----------------------\n" \
                      f"Initial state: {self.x0}\n" \
                      f"-----------\n" \
                      f"Plant: {self.plant}\n" \
                      f"-----------\n" \
                      f"In-/Out-shape: {self.n_states, self.n_inputs}\n" \
                      f"-----------\n" \
                      f"Observer: {self.observer}\n" \
                      f"-----------\n" \
                      f"Controller: {self.controller}" \

        return description

    def __repr__(self):
        representation = (f"ControlledPlant(x0={self.x0}, plant={repr(self.plant)}, "
                          f"observer={repr(self.observer)}, controller={repr(self.controller)})")

        return representation


if __name__ == '__main__':
    help(ControlledPlant)
