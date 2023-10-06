from time import time
from datetime import timedelta

import numpy as np

# for type hints
from processes import Optimization
from controlled_plants import ControlledPlant
from controllers import ZeroOrderHold, FeedForward

from optimization.constraints import UTOptConstraints, TimeNormOptConstraints, FedBatchConstraints
from optimization.cost_functions import UTOptError, TimeNormError, FedBatchError

from utils.logger import CustomLogger

my_logger = CustomLogger()


class UTOptimization(Optimization):
    """
    Optimize the input trajectory as well as the process time

    Can b
    """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            Name of optimization
        """
        super().__init__(name)

        # set normalized time-vector
        self.tau = None

        # unraveled optimization results
        self.u_star = None
        self.T_star = None

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

        my_logger.debug(f"setting {controlled_plant} to optimization process")
        self.controlled_plant = controlled_plant

        # disable history tracking to avoid memory overflow
        self.controlled_plant.track_history(False)

        # set flag if zero order hold is used
        self.zero_order = isinstance(controlled_plant.controller, ZeroOrderHold)

    def set_cost_func(self, error_func: UTOptError) -> None:
        """
        Set the cost function for the minimizer.

        Parameters
        ----------
        error_func : UTOptError
            The cost function.
        """
        if not isinstance(error_func, UTOptError):
            my_logger.exception("error function not of type UTOptError, aborting...")
            raise ValueError("Error function must be of type 'UTOptError' in optimization package")

        my_logger.debug(f"setting error function to {error_func}")
        self.error_func = error_func

    def set_constraint_func(self, constraint_func: UTOptConstraints) -> None:
        """
        Set the constraint function for the optimization. The constraint function will act as the non-linear constraint
        g(z) =(<) 0 if a __call__ function is implemented and create box- and linear constraints if available in
        process

        Parameters
        ----------
        constraint_func : UTOptConstraints
            Instance of DynamicConstraint class

        Returns
        -------
        None
        """
        if not isinstance(constraint_func, UTOptConstraints):
            my_logger.exception("constraint function is no instance of UTOptConstraints, aborting...")
            raise ValueError("Constraint function is no instance of UTOptConstraints from optimization module")

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
        if len(pump_limits) != 4:
            my_logger.exception(f"{len(pump_limits)} pump limits are provided not 4, aborting...")
            raise ValueError(f"Four pump limits must be given, but {len(pump_limits)} are provided")

        my_logger.debug(f"pump limits set to {pump_limits}")
        self.bounds = pump_limits

        # get numbers of used pumps
        self.used_pumps = sum([0 if p is None else 1 for p in pump_limits])

    def set_initial_guess(self, z0: np.ndarray) -> None:
        """
        Set initial guess for input trajectory

        Parameters
        ----------
        z0 : np.ndarray
            initial guess for form (m*n_steps) in F order appending an initial guess for time. shape : [m*n_steps, 1]

        Returns
        -------
        None
        """

        if self.bounds is None:
            raise AttributeError("Pump limits not set")

        num_nones = sum([0 if p is None else 1 for p in self.bounds])

        if (len(z0) % num_nones) == 0:
            raise ValueError("Initial guess must contain initial guess for end time T")

        self.z_0 = z0

    def set_weight(self, weights: tuple | list) -> None:
        """
        Set optimization weight for optimization

        Parameters
        ----------
        weights : tuple or list
            Weights for optimization

        Returns
        -------
        None
        """
        my_logger.debug(f"setting weights to {weights}")
        self.weights = weights

    def set_tau(self, n_step: int, mode='lin') -> None:
        """
        Create a normalized time vector [0, 1]. With given spacing mode. Default is linearly spaces

        Parameters
        ----------
        n_step : int
            number of steps
        mode : str
            spacing mode, default is linear

        Returns
        -------
        None
        """
        if mode == "lin":
            self.tau = np.linspace(0, 1, n_step)
        elif mode == "geom":
            self.tau = np.geomspace(1, 1 + 1, n_step) - 1
        elif mode == "geom_rev":
            t_k = np.geomspace(1, 1 + 1, n_step) - 1
            self.tau = t_k[::-1]
        else:
            raise ValueError("Invalid spacing mode for set_t(). Only linear available at this point")

    # === run optimization ===
    def get_constraints(self) -> tuple:
        """
        Set constraints for optimization based on the pump limits and concentration limitations.

        Returns
        -------

        """

        if len(self.bounds) != self.controlled_plant.plant.n_inputs:
            raise ValueError("Pump limits must have same size as plant inputs. Set limit to None if input is not used "
                             "...")

        self.constraint_func.set_time(self.tau)
        self.constraint_func.set_bounds(self.bounds)
        bounds_z = self.constraint_func.create_box_constraints(zero_order=self.zero_order)

        lower_bound = np.append(bounds_z[0], 2)
        upper_bound = np.append(bounds_z[1], 48)
        bounds_z = (lower_bound, upper_bound)

        constraint_box = self.minimizer.build_box_constraints(bounds_z)
        constraint_lin = None
        constraint_non_lin = self.minimizer.build_non_lin_constraints(func=self.constraint_func, lb=(self.x_star[-1]-self.controlled_plant.x0[-1]),ub=(self.x_star[-1]-self.controlled_plant.x0[-1]))

        return constraint_box, constraint_lin, constraint_non_lin

    def target(self, z: np.ndarray) -> float:
        """
        Defines optimization target by using given minimizer and using the provided integrator to perform the underlying
        integration

        Parameters
        ----------
        z: np.ndarray
            Optimization vector: flattened input sequence

        Returns
        -------
        float
            Cost value of z_star
        """

        T = z[-1]
        u = self._create_full_u(z[:-1])

        t = self.tau * T

        self.controlled_plant.controller.update_u(u, t)

        # do integration and reshape x
        x = self.integrator(self.controlled_plant, self.controlled_plant.x0, t)
        x = x if self.noise_generator is None else self.noise_generator(x)
        x = np.ravel(x)

        self.error_func.adjust_time(t)

        # add last (unused) u if zero order hold
        z = np.hstack((u.flatten(), T))

        return self.error_func(np.hstack((x, z)))

    def run_optimization(self) -> None:
        """
        Run optimization of full configured object

        Returns
        -------
        None
        """
        # set error function and constraints
        self.error_func.adjust_time(self.tau * self.z_0[-1])
        self.error_func.adjust_set_point(self.x_star)
        self.error_func.adjust_weights(self.weights)

        box_constraints, _, non_lin_constraints = self.get_constraints()

        start_time = time()
        my_logger.info(f"starting optimization with {self.minimizer}...")
        sol = self.minimizer(self.target, self.z_0, bounds=box_constraints, constraints=(non_lin_constraints), options={"maxiter": 1e5})
        end_time = time() - start_time
        my_logger.info(f"Optimization finished. Success: {sol.success}, duration: {timedelta(seconds=(end_time))}")

        # log minimizer results
        self.z_star = sol.x
        self.u_star, self.T_star = self.z_star[:-1], self.z_star[-1]

        self.success = sol.success
        self.sol = sol
        self.T_opt = end_time

        # Set interpolated input for plant and run to get results
        self.u, self.t = self.controlled_plant.controller.interpolate_u()
        self.controlled_plant.controller.update_u(self.u, self.t)
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

        Returns
        -------
        None
        """
        # get optimized inputs
        u_or, t_or = self._create_full_u(self.z_star[:-1]), self.tau * self.z_star[-1]

        # set set-point as reference
        reference = (np.asarray(self.t[-1]), self.x_star[None, :]) if reference is None else reference

        file_title = f"{self.name}_{time()}" if title is None else title
        self.controlled_plant.plant.plot_process(self.t, self.x, self.u, title=file_title, save_path=save_path,
                                                 ref=reference, u_points=(t_or, u_or))

    # === helper functions ===

    def _create_full_u(self, z) -> np.ndarray:
        """
        Create u vector (in m*t-form) from flattened z_star vector

        Parameters
        ----------
        z

        Returns
        -------

        """
        not_none_indexes = [i for i, v in enumerate(self.bounds) if v is not None]
        num_not_none = len(not_none_indexes)

        # just a safety feature so z_star can be provided with and without T
        z = z if z.shape[0] % num_not_none == 0 else z[:-1]
        z = np.hstack((z, [0] * num_not_none)) if self.zero_order else z

        u = np.tile(np.asarray([0., 0., 0., 0.]), int(z.shape[0] / num_not_none))

        for i, j in enumerate(not_none_indexes):
            u[j::4] = z[i::num_not_none]

        u = np.reshape(u, (-1, self.controlled_plant.plant.n_inputs))

        return u

    def __str__(self):
        old_str = super().__str__()
        new_str = f"{old_str.strip()}\n" \
                  f"  - u*: {self.u_star} l/h\n" \
                  f"  - T*: {str(self.z_star[-1])} h\n" \
                  f"-----------\n" \
                  f"Number grid points: {len(self.tau)}\n"
        return new_str


class NormTimeOptimization(UTOptimization):

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            Name of optimization
        """
        super().__init__(name)

    # === builder options ===

    def set_cost_func(self, error_func: TimeNormError) -> None:
        """
        Set the cost function for the minimizer.

        Parameters
        ----------
        error_func : TimeNormError
            The cost function.
        """
        if not isinstance(error_func, TimeNormError):
            my_logger.exception("error function not of type NormOptError, aborting...")
            raise ValueError("Error function must be of type 'NormOptError' in optimization package")

        my_logger.debug(f"setting error function to {error_func}")
        self.error_func = error_func

    def set_constraint_func(self, constraint_func: TimeNormOptConstraints) -> None:
        """
        Set the constraint function for the optimization. The constraint function will act as the non-linear constraint
        g(z) =(<) 0 if a __call__ function is implemented and create box- and linear constraints if available in
        process

        Parameters
        ----------
        constraint_func : TimeNormOptConstraints
            Instance of TimeNormOptConstraints class

        Returns
        -------
        None
        """
        if not isinstance(constraint_func, TimeNormOptConstraints):
            my_logger.exception("constraint function is no instance of TimeNormOptConstraints, aborting...")
            raise ValueError("Constraint function is no instance of TimeNormOptConstraints from optimization module")

        my_logger.debug(f"setting constraint_function: {constraint_func}")
        self.constraint_func = constraint_func


class FedBatchOptimization(UTOptimization):

    def __init__(self, name: str):
        super().__init__(name)

    # === building options ===

    def set_cost_func(self, error_func: FedBatchError) -> None:
        """
        Set the cost function for the minimizer.

        Parameters
        ----------
        error_func : FedBatchError
            The cost function.
        """
        if not isinstance(error_func, FedBatchError):
            my_logger.exception("error function not of type FedBatchError, aborting...")
            raise ValueError("Error function must be of type 'FedBatchError' in optimization package")

        my_logger.debug(f"setting error function to {repr(error_func)}")
        self.error_func = error_func

    def set_constraint_func(self, constraint_func: FedBatchConstraints) -> None:
        """
        Set the constraint function for the optimization. The constraint function will act as the non-linear constraint
        g(z) =(<) 0 if a __call__ function is implemented and create box- and linear constraints if available in
        process

        Parameters
        ----------
        constraint_func : FedBatchConstraints
            Instance of FedBatchConstraints class

        Returns
        -------
        None
        """
        if not isinstance(constraint_func, FedBatchConstraints):
            my_logger.exception("constraint function is no instance of FedBatchConstraints, aborting...")
            raise ValueError("Constraint function is no instance of FedBatchConstraints from optimization module")

        my_logger.debug(f"setting constraint_function: {constraint_func}")
        self.constraint_func = constraint_func

    def set_bounds(self, pump_limits: list) -> None:
        """
        Set pump limits, check if bleed is 0 ∀ t first. Does not make sense in Fed-batch
        
        Parameters
        ----------
        pump_limits : list
            Pump limits in form [lim1, lim2, lim3, None], lim_i is None if unused pump else [lim_low, lim_high]

        Returns
        -------
        None
        """
        # check if bleed is None
        if pump_limits[-1] is not None:
            my_logger.exception(f"bleed is not zero for fed-batch, abroting...")
            raise ValueError(f"Bleed must be 0 ∀ t. Set pump_limits[-1] = None")

        # set pump limits
        super().set_bounds(pump_limits)

    # === visualization ===

    def plot_results(self, noisy_values=False, reference=None, save_path=None, title=None, plot_dynamics=False):
        """
        Simulate process with optimized parameters and plot results

        Parameters
        ----------
        save_path: string|None
            path to save plot to
        noisy_values: Bool
            No effect
        reference: np.ndarray|None
            No effect

        Returns
        -------
        None
        """
        # get optimized inputs
        u_or, t_or = self._create_full_u(self.z_star[:-1]), self.tau * self.z_star[-1]

        # set Volume end point as reference
        ref_x = np.asarray([[None, None, None, None, self.x_star[-1]]])
        reference = (np.asarray(self.t[-1]), ref_x) if reference is None else reference

        file_title = f"{self.name}_{time()}" if title is None else title
        self.controlled_plant.plant.plot_process(self.t, self.x, self.u, title=file_title, save_path=save_path,
                                                 ref=reference, u_points=(t_or, u_or), plot_dynamics=plot_dynamics)

if __name__ == '__main__':
    from controlled_plants import ControlledPlant
    from models import INA
    from controllers import ZeroOrderHold, FirstOrderHold

    from optimization import TimeNormError

    from numeric import SciPy
    from optimization import TrustReg, DualAnnealing, NelderMead, SHOG

    example_opt = NormTimeOptimization("example")

    #### simulation configuration ####

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

    #### optimization configuration ####

    # set set-point
    x_star = np.asarray([0.11525703, 0.32221684, 0.08651147, 0.19054012, 1.])
    example_opt.set_x_star(x_star)

    # time-grid configuration
    n_steps = 5
    example_opt.set_tau(n_steps, mode="lin")

    # set pump limits
    pump_limits = [[0., .1], [0., .1], [0., .1], None]
    example_opt.set_bounds(pump_limits)

    # create initial guess
    from utils.input_utils import create_constant_z0

    z0_part = np.asarray([0.05, 0.05, 0.05, 0.0])
    T0 = 12
    z0 = create_constant_z0(z0_part, n_steps, pump_limits, T_start=T0, zero_order=False)
    example_opt.set_initial_guess(z0)

    # set weights
    k_end = np.asarray([1., 2., 2., 1., 10.]) * 1e4
    k_i = np.asarray([0])
    k_t = np.asarray([1]) * 750e-3
    k_reg = np.asarray([1., 1., 1., 1.]) * 0e-9
    time_norm_weights = (k_end, k_i, k_t, k_reg)
    example_opt.set_weight(time_norm_weights)

    # set optimization algorithm
    example_opt_alg = TrustReg()
    # mif plot_
    # from optimization.opt_callbacks import LogSteps
    # example_opt_alg.set_callback(LogSteps())
    example_opt.set_minimizer(example_opt_alg)

    # set error_function
    example_opt.set_cost_func(TimeNormError())

    # set constraint function
    example_opt.set_constraint_func(TimeNormOptConstraints())

    example_opt.run_optimization()
    example_opt.plot_results()
    # example_opt.minimizer.callback.plot_f()

    print(example_opt)
