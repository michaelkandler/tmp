"""
This module defines optimization of the input trajectory on a fixed time-grid.

The constraints, cost functions and minimizers can be defined separately
"""

from time import time
from datetime import timedelta

import numpy as np

from processes import Optimization

# for type hinting
from optimization.cost_functions import DynamicError, NormOptError
from optimization.constraints import DynamicConstraint, NormOptConstraints
from controlled_plants import ControlledPlant
from controllers import FeedForward, ZeroOrderHold

from utils.logger import CustomLogger

my_logger = CustomLogger()


class UOptimization(Optimization):
    """
    Optimize the trajectory of a plant to fit an endpoint using the norm

    d
    """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            name of the proces
        """
        super().__init__(name)

        # set discretization interval and initialize optimization grid
        self.t_k = None

        # optimized in flow
        self.u_star = None

        # order of input discretization and number of used pumps
        self.zero_order = None
        self.used_pumps = None

    # === building options ===

    def set_controlled_plant(self, controlled_plant: ControlledPlant) -> None:
        """
        Set the controlled plant used for the target simulation to the process.

        Some checks are made

        Parameters
        ----------
        controlled_plant : ControlledPlant
            Fully configured controlled plant

        Returns
        -------
        None
        """
        # check if plant is configured correctly
        if controlled_plant.x0 is None:
            my_logger.exception(f"controlled plant has no initial value set, aborting...")
            raise AttributeError("Controlled plant has no initial value, use controlled_plant.set_initial_value()")
        if not isinstance(controlled_plant.controller, FeedForward):
            my_logger.exception(f"controller is not of type FeedForward, aborting...")
            raise AttributeError(f"Controller in controlled_plant has to be of type FeedForward")

        my_logger.debug(f"setting {repr(controlled_plant)} to optimization process")
        self.controlled_plant = controlled_plant

        # disable history tracking to avoid memory overflow
        self.controlled_plant.track_history(False)

        # set flag if zero order hold is used
        self.zero_order = isinstance(controlled_plant.controller, ZeroOrderHold)

    def set_cost_func(self, error_func: DynamicError) -> None:
        """
        Set the cost function for the minimizer.

        Parameters
        ----------
        error_func : DynamicConstraint
            The cost function.
        """
        if not isinstance(error_func, DynamicError):
            my_logger.exception("error function not of type DynamicError, aborting...")
            raise ValueError("Error function must be of type 'DynamicError' in optimization package")

        my_logger.debug(f"setting error function to {repr(error_func)}")
        self.error_func = error_func

    def set_constraint_func(self, constraint_func: DynamicConstraint) -> None:
        """
        Set the constraint function for the optimization. The constraint function will act as the non-linear constraint
        g(z) =(<) 0 if a __call__ function is implemented and create box- and linear constraints if available in
        process

        Parameters
        ----------
        constraint_func : DynamicConstraint
            Instance of DynamicConstraint class

        Returns
        -------
        None
        """
        if not isinstance(constraint_func, DynamicConstraint):
            my_logger.exception("constraint function is no instance of DynamicConstraint, aborting...")
            raise ValueError("Constraint function is no instance of DynamicConstraints from optimization module")

        my_logger.debug(f"setting constraint_function: {constraint_func}")
        self.constraint_func = constraint_func

    def set_bounds(self, pump_limits: list) -> None:
        """
        Set pump limits
        
        Parameters
        ----------
        pump_limits : list
            Pump limits in form [lim1, lim2, lim3, lim4], lim_i is None if unused pump else [lim_low, lim_high]

        Returns
        -------
        None
        """
        self.bounds = pump_limits

        self.used_pumps = sum([0 if p is None else 1 for p in pump_limits])

    def set_weight(self, weights: tuple | list) -> None:
        """
        Set weights for optimization

        Parameters
        ----------
        weights : tuple or list
            Weights for optimization given as tuple in form (k_end, k_i, k_reg). Every weight can either be a scalar 
            or a vector with shape of target variable (eg. (5,) for end state |x_N-x_star|)
        scale_weights : bool
            Scale weights relative to the magnitude of the error

        Returns
        -------
        None
        """
        # check for number of weight arrays
        if len(weights) != 3:
            my_logger.exception(f"weights shape is {len(weights)} not 3, aborting...")
            raise ValueError(f"weights tuple shape has to be 3 not {len(weights)}")

        k_end, k_i, k_reg = weights

        # check if weights have correct shape  
        if len(k_end) != 5 and len(k_end) != 1:
            my_logger.exception(f"k_end has shape {len(k_end)} not 5, aborting...")
            raise ValueError(f"Shape of k_end is {len(k_end)}, has to be 5")
        if False:  # len(k_i) != 5 and len(k_i) != 1:
            my_logger.exception(f"k_i has shape {len(k_i)} not 5, aborting...")
            raise ValueError(f"Shape of k_i is {len(k_i)}, has to be 5")
        if len(k_reg) != 4 and len(k_reg) != 1:
            my_logger.exception(f"k_reg has shape {len(k_reg)} not 5, aborting...")
            raise ValueError(f"Shape of k_reg is {len(k_reg)}, has to be 4")

        my_logger.debug(f"setting optimization weights to: {weights}")
        self.weights = weights

    def set_t(self, t_end: float, n_step: int, mode='lin') -> None:
        """
        Create a time vector [0, t_end]. With given spacing mode. Default is linearly spaces

        Parameters
        ----------
        t_end : float
            final time
        n_step : int
            number of steps
        mode : str
            spacing mode, default is linear

        Returns
        -------
        None
        """
        if mode == "lin":
            self.t_k = np.linspace(0, t_end, n_step)
        elif mode == "geom":
            self.t_k = np.geomspace(1, t_end + 1, n_step) - 1
        elif mode == "geom_rev":
            t_k = np.geomspace(1, t_end + 1, n_step) - 1
            self.t_k = t_k[::-1]
        else:
            raise ValueError("Invalid spacing mode for set_t(). Only linear available at this point")
        self.t = self.t_k

    # === run optimizations ===
    def get_constraints(self) -> tuple:
        """
        Set constraints for optimization based on the pump limits and concentration limitations.

        Returns
        -------
        tuple
            box constraints and linear constraints
        """
        if self.bounds is None:
            raise AttributeError("No pump limits chose. Use set_pump_limits() to set")

        if len(self.bounds) != self.controlled_plant.plant.n_inputs:
            raise AttributeError(
                "Pump limits must have same size as plant inputs. Set limit to None if input is not used "
                "...")
        # get constraints from constraints-function
        self.constraint_func.set_time(self.t_k)
        self.constraint_func.set_bounds(self.bounds)
        bounds_z = self.constraint_func.create_box_constraints(zero_order=self.zero_order)

        # get proper constraint object to pass to optimizer
        constraint_box = self.minimizer.build_box_constraints(bounds_z)

        return constraint_box, None

    def target(self, z: np.ndarray) -> float:
        """
        Defines optimization target by using given minimizer and using scipy's odeint to perform the underlying
        integration

        Parameters
        ----------
        z: np.ndarray
            Optimization vector: flattened input sequence

        Returns
        -------
        float
            cost value for given vector
        """
        # create u from optimization vector
        u = self._create_full_u(z)
        self.controlled_plant.controller.update_u(u, self.t_k)

        # do integration and reshape x
        x = self.integrator(self.controlled_plant, self.controlled_plant.x0, self.t_k)
        x = x if self.noise_generator is None else self.noise_generator(x)
        x = np.ravel(x)

        # append zeros if first order hold is used
        z = np.hstack((z, [0] * self.used_pumps)) if self.zero_order else z

        # calculate and return error function
        return self.error_func(np.hstack((x, z)))

    def run_optimization(self) -> None:
        """
        Run optimization with given configuration

        Returns
        -------
        None
        """
        # check if process is configured correctly
        if self.controlled_plant is None:
            my_logger.exception("no controlled plant set, aborting...")
            raise AttributeError("No controlled_plant chosen. Use 'self.set_controlled_plant()' to set.")
        if self.controlled_plant.plant is None:
            my_logger.exception("no model set, aborting...")
            raise AttributeError("No model ist chosen. Use controlled_plant_instance.set_plant() to set.")
        if self.controlled_plant.controller is None:
            my_logger.exception("no controller set, aborting...")
            raise AttributeError("No controller chosen. Use controlled_plant_instance.set_controller() to set.")
        if not isinstance(self.controlled_plant.controller, FeedForward):
            my_logger.error("only feedforward controllers can be used, aborting...")
            raise AttributeError("Controller is not a feedforward controller, choose ZeroOrderHold, or FirstOrderHold")
        if self.t_k is None:
            my_logger.exception("no time-vector set, aborting...")
            raise AttributeError("No time-vector set. Use self.set_t() to set")
        if self.z_0 is None:
            my_logger.exception("no initial guess provided, aborting...")
            raise AttributeError("No initial guess provided. Use self.set_start_point() to set.")
        if len(self.z_0) != ((len(self.t_k) - self.zero_order) * self.used_pumps):
            my_logger.exception(f"inital guess has wrong shape, should be "
                                f"{(len(self.t_k) - self.zero_order) * self.used_pumps}, but is {len(self.z_0)}"
                                f", aborting...")
            raise AttributeError(f"inital guess has wrong shape, should be "
                                 f"{(len(self.t_k) - self.zero_order) * self.used_pumps}, but is {len(self.z_0)}"
                                 f", aborting...")
        if self.minimizer is None:
            my_logger.exception("no minimizer chosen, aborting...")
            raise AttributeError("No minimizer chosen. Use self.set_optimizer() to set.")
        if self.constraint_func is None:
            my_logger.exception(f"constraints not set")
            raise AttributeError("Constraints not set")

        # configure error function
        self.error_func.adjust_time(self.t_k)
        self.error_func.adjust_set_point(self.x_star)
        self.error_func.adjust_weights(self.weights)

        # get box constraints
        constraint_box, _ = self.get_constraints()

        # run optimization
        start_time = time()
        my_logger.info(f"starting optimization with {self.minimizer}...")
        sol = self.minimizer(self.target, self.z_0, bounds=constraint_box)
        end_time = time() - start_time
        my_logger.info(f"Optimization finished. Success: {sol.success}, duration: {timedelta(seconds=(end_time))}")

        # get results
        self.z_star = sol.x
        self.u_star = self._create_full_u(self.z_star)

        # log run
        self.success = sol.success
        self.sol = sol
        self.T_opt = end_time

        # rerun with optimized parameters to update self.x
        self.t, self.u = self.t_k, self.u_star
        self.controlled_plant.controller.update_u(self.u, self.t_k)
        self.run_process()

    def plot_results(self, reference=None, save_path=None, title=None):
        """
        Simulate process with optimized parameters and plot results

        Parameters
        ----------
        save_path: string|None
            path to save plot to
        reference: np.ndarray|None
            No effect
        title : str
            optional title for plot

        Returns
        -------
        None
        """
        # interpolate u along finer time-grid and rerun simulation
        u_or, t_or = (self._create_full_u(self.z_star),
                      np.linspace(0, self.t[-1], int(self.z_star.shape[0] / self.used_pumps)))
        self.u, self.t = self.controlled_plant.controller.interpolate_u()
        self.controlled_plant.controller.update_u(self.u, self.t)
        self.run_process()

        # set set-point as reference value
        reference = (np.asarray(self.t[-1]), self.x_star[None, :])

        file_title = f"{self.name}_{time()}" if title is None else title
        self.controlled_plant.plant.plot_process(self.t, self.x, u=self.u,
                                                 save_path=save_path, title=file_title,
                                                 ref=reference, u_points=(t_or, u_or))

    # === helper functions ===

    def _create_full_u(self, z) -> np.ndarray:
        """
        Create u vector (in m*t-form) from flattened z_star vector return it and set it to self

        Parameters
        ----------
        z : np.ndarray
            optimization vector, flattened input sequence

        Returns
        -------
        np.ndarray
            input array u with shape t*m
        """
        # get not None indexes and there amount
        not_none_indexes = [i for i, v in enumerate(self.bounds) if v is not None]
        num_not_none = len(not_none_indexes)

        # create full u
        z = np.hstack((z, np.zeros(self.used_pumps))) if self.zero_order else z
        u = np.tile(np.asarray([0., 0., 0., 0.]), int(z.shape[0] / num_not_none))

        for i, j in enumerate(not_none_indexes):
            u[j::4] = z[i::num_not_none]

        u = np.reshape(u, (-1, self.controlled_plant.plant.n_inputs))

        return u

    # === miscellaneous ===

    def __str__(self):
        old_str = super().__str__()
        new_str = f"{old_str.strip()}\n" \
                  f"  - u*: {self.u}\n" \
                  f"-----------\n" \
                  f"Number grid points: {len(self.t_k)}\n" \
                  f"-----------\n" \
                  f"T: {self.t[-1]} h"
        return new_str

    def __repr__(self):
        representation = (
            f"UOptimization(error_func={repr(self.error_func)}, constraint_func={repr(self.constraint_func)}, "
            f"bounds={self.bounds}, x_star={self.x_star}, "
            f"minimizer={repr(self.minimizer)}, z_0={self.z_0}, weights={self.weights}, "
            f"t_k={self.t_k})")

        return representation


class NormOptimization(UOptimization):

    def set_cost_func(self, error_func: NormOptError) -> None:
        """
        Set the cost function for the minimizer.

        Parameters
        ----------
        error_func : NormOptError
            The cost function.
        """
        if not isinstance(error_func, NormOptError):
            my_logger.exception("error function not of type NormOptError, aborting...")
            raise ValueError("Error function must be of type 'NormOptError' in optimization package")

        my_logger.debug(f"setting error function to {repr(error_func)}")
        self.error_func = error_func

    def set_constraint_func(self, constraint_func: NormOptConstraints) -> None:
        """
        Set the constraint function for the optimization. The constraint function will act as the non-linear constraint
        g(z) =(<) 0 if a __call__ function is implemented and create box- and linear constraints if available in
        process

        Parameters
        ----------
        constraint_func : NormOptConstraints
            Instance of NormOptConstraints class

        Returns
        -------
        None
        """
        if not isinstance(constraint_func, NormOptConstraints):
            my_logger.exception("constraint function is no instance of NormOptConstraints, aborting...")
            raise ValueError("Constraint function is no instance of NormOptConstraints from optimization module")

        my_logger.debug(f"setting constraint_function: {constraint_func}")
        self.constraint_func = constraint_func


if __name__ == '__main__':
    # imports for simulation
    from controlled_plants import ControlledPlant
    from models import INA
    from controllers import ZeroOrderHold, FirstOrderHold
    from numeric import SciPy

    # imports for optimization
    from optimization import NormOptError, NormOptConstraints
    from optimization import TrustReg, DualAnnealing, SHOG, SLSQP

    example_opt = NormOptimization("example")

    # === simulation configuration ===

    # create controlled plant
    example_controlled_plant = ControlledPlant()

    # configure plant
    model_parameters = {'an': 1.3343, 'bn': -8.6824, 'aa': 12.0465, 'ba': -16.45, 'n': 2}  # J3
    input_parameters = {'cSR1': 11.5, 'cDR1': 4., 'cSR2': 0., 'cDR2': 0., 'cSR3': 0., 'cDR3': 0.}
    example_model = INA(model_parameters, input_parameters)
    example_controlled_plant.set_plant(example_model)

    # configure controller
    example_controller = FirstOrderHold(None, None)
    example_controlled_plant.set_controller(example_controller)

    # set initial value
    x0 = np.asarray([0, 0, 0, 0, .8])
    example_controlled_plant.set_initial_value(x0)

    # put it all together
    example_opt.set_controlled_plant(example_controlled_plant)

    # set integration method
    example_integrator = SciPy()
    example_opt.set_integrator(example_integrator)

    # === optimization configuration ===

    # set set-point
    x_star = np.asarray([0.11525703, 0.32221684, 0.08651147, 0.19054012, 1.])
    example_opt.set_x_star(x_star)

    # time-grid configuration
    n_steps = 2
    example_opt.set_t(12, n_steps, mode="lin")

    # set constraints
    bounds = [[0., .1], None, [0., .1], None]
    example_opt.set_bounds(bounds)
    example_opt.set_constraint_func(NormOptConstraints())

    # set error_function
    example_opt.set_cost_func(NormOptError())

    # set weights
    k_end = np.asarray([1., 1., 1., 1., 1e0]) * 1e1
    k_i = np.asarray([0])
    k_reg = np.asarray([1., 1., 1., 1.]) * 0e-1
    norm_weights = (k_end, k_i, k_reg)
    example_opt.set_weight(norm_weights)

    # set optimization algorithm
    example_opt_alg = SLSQP()
    example_opt.set_minimizer(example_opt_alg)

    # create initial guess
    from utils.input_utils import create_constant_z0

    z0 = create_constant_z0(np.asarray([0.001, 0.00, 0.1, 0.1]), n_steps, bounds, zero_order=False)
    example_opt.set_initial_guess(z0)

    # run and plot everything
    example_opt.run_optimization()
    example_opt.plot_results()

    print(example_opt)
