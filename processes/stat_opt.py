from time import time
from datetime import timedelta

import os

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# fixed packages
from processes import Optimization
from controllers import ChemoStat

# type hinting
from optimization.cost_functions import StationaryError
from optimization.constraints import StationaryConstraint
from controlled_plants import ControlledPlant

from utils.logger import CustomLogger

my_logger = CustomLogger()


class StationaryPoint(Optimization):
    def __init__(self, name: str):
        super().__init__(name)

        self.name = name

        # set simulation time
        self.set_t(24, 30)

        # automatically chosen if not given
        self.z_0 = True

        # optimization variables parametrized
        self.tau = None
        self.S_in = None

        # normalize problem
        self.V = 1.

    # === builder options ===

    def set_volume(self, V: float) -> None:
        """
        Setting the Volume of the process, will be normalized to one if not given otherwise

        Parameters
        ----------
        V : float
            Volume to be used in optimization

        Returns
        -------
        None
        """
        my_logger.debug(f"setting stationary Volume to: {V}")

        self.V = V

    def set_controlled_plant(self, controlled_plant: ControlledPlant) -> None:
        """
        Set the controlled plant used for the target simulation to the process.

        Some checks are made and controller is set to Chemostat controller

        Parameters
        ----------
        controlled_plant : ControlledPlant
            Fully configured controlled plant

        Returns
        -------
        None
        """
        my_logger.debug(f"setting controller to chemostat")

        # set controller to chemostat
        controlled_plant.set_controller(ChemoStat(None))

        my_logger.debug(f"setting {repr(controlled_plant)} to optimization process")
        self.controlled_plant = controlled_plant

    def set_cost_func(self, error_func: StationaryError) -> None:
        """
        Set the cost function for the minimizer.

        Parameters
        ----------
        error_func : StationaryError
            The cost function.
        """
        if not isinstance(error_func, StationaryError):
            my_logger.exception("error function not of type StationaryError, aborting...")
            raise ValueError("Error function must be of type 'StationaryError' in optimization package")

        my_logger.debug(f"setting error function to {error_func}")
        self.error_func = error_func
        self.error_func.adjust_V(self.V)

    def set_constraint_func(self, constraint_func: StationaryConstraint) -> None:
        """
        Set the constraint function for the optimization. The constraint function will act as the non-linear constraint
        g(z) =(<) 0 if a __call__ function is implemented and create box- and linear constraints if available in
        process

        Parameters
        ----------
        constraint_func : StationaryConstraint
            Instance of StationaryConstraint class

        Returns
        -------
        None
        """
        if not isinstance(constraint_func, StationaryConstraint):
            my_logger.exception("constraint functino is no instance of StationaryConstraint, aborting...")
            raise ValueError("Constraint functino is no instance of StationaryConstraint from optimization module")

        my_logger.debug(f"setting constraint_function: {constraint_func}")
        self.constraint_func = constraint_func

    def set_bounds(self, bounds: list) -> None:
        """
        Set pump limits

        Parameters
        ----------
        bounds : list
            Pump limits in form [lim1, lim2, lim3, lim4], lim_i is None if unused pump else [lim_low, lim_high]

        Returns
        -------
        None
        """
        num_pumps = sum([1 for b in bounds[:-1] if b is not None])
        if num_pumps < 1:
            raise ValueError("In flow must be at least one")

        my_logger.debug(f"setting pump limits to {bounds}")
        if bounds[-1] is None:
            my_logger.info(f"Bleed pump limit cant be None, limit will be ignored")
        self.bounds = bounds

    def set_initial_guess(self, z0: np.ndarray) -> None:
        """
        Set the starting point for optimization.

        Parameters
        ----------
        z0 : np.ndarray
            The starting point for optimization.
        """
        my_logger.debug(f"setting initial guess to:{z0}")
        my_logger.info(f"initial guess manually set, drop this if z0 should be chosen automatically")
        self.z_0 = z0

    def set_weights(self, weights: tuple) -> None:
        """
        Set the weights for the minimization algorithm

        Parameters
        ----------
        weights : tuple
            Tuple of np.ndarray of weights (k_tau, k_Sin)

        Returns
        -------
        None
        """
        if len(weights) != 2:
            raise ValueError("Number of weights must be 2")

        self.weights = weights

    def set_t(self, t: float, t_samp: float):
        """
        Set the simulation time for the optimization process.

        Will be called with default values during instantiation process

        Parameters
        ----------
        t : float
            end-time in hours
        t_samp : float
            sampling time in seconds

        Returns
        -------
        None
        """
        my_logger.debug(f"setting simulation time to t={t} with sampling time t_samp={t_samp}")

        self.t = np.arange(0, t + t_samp / 60 / 60, t_samp / 60 / 60)

    # === run optimization ===

    def get_constraints(self) -> tuple:
        """
        Set constraints for optimization based on the pump limits and concentration limitations.

        Returns
        -------
        tuple
            box constraints and linear constraints (const_box, const_lin)
        """
        if self.bounds is None:
            raise AttributeError("No pump limits chose. Use set_pump_limits() to set")

        # create box constraint vectors
        self.constraint_func.set_bounds(self.bounds)
        bounds_z = self.constraint_func.create_box_constraints()

        # get proper constraint object to pass to optimizer
        constraint_box = self.minimizer.build_box_constraints(bounds_z)

        # no linear constraints are used
        constraint_lin = None

        return constraint_box, constraint_lin

    def target(self, z: np.ndarray) -> float:
        """
        Calculates the actual optimization target with static point calculation.

        Parameters
        ----------
        z : np.ndarray
            Optimization vector.

        Returns
        -------
        np.ndarray
            Static scalar error.
        """
        # vectorize if list
        z = np.asarray(z)

        # get D and S inflow
        S_in = self._get_S(z)
        D_in = self._get_D(z)

        # calculate flow time
        tau = np.power(np.sum(z, axis=0), -1) * self.V

        # calculation of steady-states for given inputs
        stat_state = self.controlled_plant.plant.stationary_point(S_in, D_in, tau, V=self.V)
        I, N, A, _, _ = stat_state

        # build vector for cost function
        z_int = np.asarray([I, N, A, tau, S_in])

        return self.error_func(z_int)

    def run_optimization(self) -> None:
        """
        Runs the optimization process.

        Returns
        ----------
        None
        """
        # set contraints and error function
        box_constraints, _ = self.get_constraints()
        self.error_func.adjust_weights(self.weights)

        bounds = [b for b in self.bounds[:-1] if b is not None]
        self.z_0 = np.mean(np.asarray(bounds), axis=1)
        box_constraints.keep_feasible = [True] * self.z_0.shape[0]

        start_time = time()
        my_logger.info(f"starting stationary optimization using: {self.minimizer}")
        sol = minimize(self.target, self.z_0, method=self.minimizer, bounds=box_constraints)
        end_time = time() - start_time
        my_logger.info(f"Optimization finished. Success: {sol.success}, duration: {timedelta(seconds=(end_time))}")

        # save optimal z_star
        self.S_in = self._get_S(sol.x)
        self.tau = np.power(np.sum(sol.x), -1) * self.V
        self.z_star = sol.x

        # log run
        self.sol = sol
        self.success = sol.success
        self.T_opt = end_time

        self.controlled_plant.set_controller(ChemoStat(self.z_star))
        self.run_process()

    def run_process(self) -> None:
        """
        Runs the process with the specified configuration.

        Returns
        -------
        None
        """
        if self.z_star is None:
            raise AttributeError("run optimization first")

        D = self._get_D(self.z_star)
        self.controlled_plant.x0 = self.controlled_plant.plant.stationary_point(self.S_in, D, self.tau, V=self.V)

        u_in, i = [], 0
        for b in self.bounds[:-1]:
            if b is not None:
                u_in.append(self.z_star[i])
                i += 1
            else:
                u_in.append(0)

        self.controlled_plant.controller = ChemoStat(np.asarray([*u_in, np.sum(u_in)]))

        super().run_process()

    # === plotting utilities ===

    def plot_cost_func(self, opt_point=True, save_path=None, save_name=None) -> None:
        """
        Plots the cost function over the entire pump-limit space

        Parameters
        ----------
        opt_point : array_like, optional
            Coordinates of the optimal point to be plotted. Default is None.
        save_path : str
            path to save plot to
        save_name : str
            name for file to save

        Returns
        -------
        None
        """
        num_pumps = sum([1 for p in self.bounds[:-1] if p is not None])
        if num_pumps == 1:
            plt_obj = self._plot_cost_one_pump(opt_point=opt_point)
        elif num_pumps == 2:
            plt_obj = self._plot_cost_two_pumps(opt_point=opt_point)
        elif num_pumps == 3:
            plt_obj = self._plot_cost_three_pumps(opt_point=opt_point)
        else:
            my_logger.info(f"Nothing plotted number of pumps is: {num_pumps}")
            return None

        if save_path is None:
            plt_obj.show()
        else:
            # get path
            save_path = os.path.abspath(save_path)
            # get name
            save_name = time() if save_path is not None else save_name
            save_name = f"{save_name}_cost.png"
            # get full path
            full_path = os.path.join(save_path, save_name)

            plt.savefig(full_path)
            plt.close()

    def _plot_cost_one_pump(self, opt_point=True, grid_density=100) -> plt:
        """
        Plot cost function if one pump is active

        Parameters
        ----------
        opt_point : bool
            Plot optimal point on graph if True
        grid_density : int
            Density of grid used for plotting

        Returns
        -------
        plt
            matplotlib object ready to plot/save
        """
        #
        bound = [b for b in self.bounds if b is not None]
        u = np.linspace(bound[0][0], bound[0][1], grid_density)

        ind_pumps = [i for i, p in enumerate(self.bounds[:-1]) if p is not None]

        u_label = self._get_labels(ind_pumps[0])

        error = self.target(u[None, :])

        # Create the surface plot
        plt.plot(u, error)

        if opt_point:
            opt_error = self.target(self.z_star)
            plt.scatter(self.z_star, opt_error, color="red")

        plt.xlabel(u_label)
        plt.ylabel("error")

        return plt

    def _plot_cost_two_pumps(self, opt_point=True, grid_density=100) -> plt:
        """
        Plot cost function if two pumps is active

        Parameters
        ----------
        opt_point : bool
            Plot optimal point on graph if True
        grid_density : int
            Density of grid used for plotting

        Returns
        -------
        plt
            matplotlib object ready to plot/save
        """
        # get active pumps
        bound = [b for b in self.bounds if b is not None]

        # get grid of pump rates
        u_1 = np.linspace(bound[0][0], bound[0][1], grid_density)
        u_2 = np.linspace(bound[1][0], bound[1][1], grid_density)
        u = np.meshgrid(u_1, u_2)

        error = self.target(u)

        # Create the surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(*u, error, cmap='viridis')

        # plot optimal point on plane
        if opt_point:
            opt_error = self.target(self.z_star)
            ax.scatter(*self.z_star, opt_error, color='red')

        # get labels for graph
        ind_pumps = [i for i, p in enumerate(self.bounds[:-1]) if p is not None]
        u_1_label = self._get_labels(ind_pumps[0])
        u_2_label = self._get_labels(ind_pumps[1])

        # add labels and title
        ax.set_xlabel(u_1_label)
        ax.set_ylabel(u_2_label)
        ax.set_zlabel("cost")
        ax.set_title('Surface Plot Example')

        return plt

    def _plot_cost_three_pumps(self, opt_point=True, grid_density=10) -> plt:
        """
        Plot cost function if three pumps are active

        Parameters
        ----------
        opt_point : bool
            Plot optimal point on graph if True
        grid_density : int
            Density of grid used for plotting

        Returns
        -------
        plt
            matplotlib object ready to plot/save
        """
        u_1 = np.linspace(self.bounds[0][0], self.bounds[0][1], grid_density)
        u_2 = np.linspace(self.bounds[1][0], self.bounds[1][1], grid_density)
        u_3 = np.linspace(self.bounds[2][0], self.bounds[2][1], grid_density)
        u = np.meshgrid(u_1, u_2, u_3)

        u_1_label = self._get_labels(0)
        u_2_label = self._get_labels(1)
        u_3_label = self._get_labels(2)

        error = self.target(u)

        # Create the surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*u, cmap='viridis', c=error)

        if opt_point:
            ax.scatter(*self.z_star, color='green')

        # Add labels and title
        ax.set_xlabel(u_1_label)
        ax.set_ylabel(u_2_label)
        ax.set_zlabel(u_3_label)
        ax.set_title('Surface Plot Example')

        return plt

    # === helper functions ===

    def _get_D(self, z: np.ndarray) -> float:
        """
        Calculated D in tank from in-flows and input concentrations

        Parameters
        ----------
        z : np.ndarray
            Optimization vector of inflow

        Returns
        -------
        float
            Concentration of D in tank
        """
        z = np.asarray(z)

        # get contribution from pumps
        D_in1 = self.controlled_plant.plant.input_params['cDR1']
        D_in2 = self.controlled_plant.plant.input_params['cDR2']
        D_in3 = self.controlled_plant.plant.input_params['cDR3']
        D_in = [D_in1, D_in2, D_in3]

        # remove concentration for deactivated pumps
        D_in = [S for S, p in zip(D_in, self.bounds) if p is not None]
        D_in = np.asarray(D_in)

        new_ax = [np.newaxis] * (len(z.shape) - 1)

        # get average
        D_in_av = np.sum(D_in[:, *new_ax] * z, axis=0) / np.sum(z, axis=0)

        return D_in_av

    def _get_S(self, z: np.ndarray) -> float:
        """
        Calculated S in tank from in-flows and input concentrations

        Parameters
        ----------
        z : np.ndarray
            Optimization vector of inflow

        Returns
        -------
        float
            Concentration of S in tank
        """
        z = np.asarray(z)
        # get contribution from pumps
        S_in1 = self.controlled_plant.plant.input_params['cSR1']
        S_in2 = self.controlled_plant.plant.input_params['cSR2']
        S_in3 = self.controlled_plant.plant.input_params['cSR3']
        S_in = [S_in1, S_in2, S_in3]

        # get all used pumps
        S_in = [S for S, p in zip(S_in, self.bounds) if p is not None]
        S_in = np.asarray(S_in)

        # create inflow-vector and average over z
        new_ax = [np.newaxis] * (len(z.shape) - 1)
        S_in_av = np.sum(S_in[:, *new_ax] * z, axis=0) / np.sum(z, axis=0)

        return S_in_av

    def _get_labels(self, ind: int) -> str:
        """
        Get labels for cost function

        Parameters
        ----------
        ind : int
            index of pump in question

        Returns
        -------
        str
            label to use in plot
        """
        # create labels for S
        label_S = ((self.controlled_plant.plant.input_params['cSR1'], "S_in 1"),
                   (self.controlled_plant.plant.input_params['cSR2'], "S_in 2"),
                   (self.controlled_plant.plant.input_params['cSR3'], "S_in 3"))

        # get labels for D
        label_D = ((self.controlled_plant.plant.input_params['cDR1'], "D_in 1"),
                   (self.controlled_plant.plant.input_params['cDR2'], "D_in 2"),
                   (self.controlled_plant.plant.input_params['cDR3'], "D_in 3"))

        label_str = (f"{label_S[ind][1]}: {label_S[ind][0]} g/l\n"
                     f"{label_D[ind][1]}: {label_D[ind][0]} mol/l")

        return label_str

    def __str__(self):
        old_str = super().__str__()
        new_str = f"{old_str}" \
                  f"  - Sᵢₙ: {self.S_in} g/l\n" \
                  f"  - 𝜏: {self.tau} h\n" \
                  f"-----------\n"

        return new_str

    def __repr__(self):
        representation_string = f"StationaryPoint()"


if __name__ == '__main__':
    from models import INA
    from controlled_plants import ControlledPlant
    from optimization.constraints import StationaryConstraint
    from optimization.cost_functions import StationaryError
    from numeric import SciPy
    from optimization.minimizer import TrustReg

    tmp_opt = StationaryPoint("example")

    # === simulation configuration ===

    # instantiate controlled plant
    example_controlled_plant = ControlledPlant()

    # create model
    mod_pars = {'an': 1.3343, 'bn': -8.6824, 'aa': 12.0465, 'ba': -16.45, 'n': 2}  # J3
    inp_pars = {'cSR1': 11, 'cDR1': 4, 'cSR2': 0., 'cDR2': 4., 'cSR3': 0., 'cDR3': 0.}
    example_model = INA(mod_pars, inp_pars)

    # set model to plant and set plant -> controller would be ignored
    example_controlled_plant.set_plant(example_model)

    # x0 is not needed it would just be overwritten

    # set controlled plant
    tmp_opt.set_controlled_plant(example_controlled_plant)

    # set integration method
    tmp_opt.set_integrator(SciPy())

    # === optimization configuration ===

    # set pump limits
    bounds = [[0.005, 0.1], None, [0.005, 0.1], [0.005, 0.1]]
    tmp_opt.set_bounds(bounds)

    # set weights
    weights = np.asarray([1e1, 1e6])
    tmp_opt.set_weights(weights)

    # set minimizer, cost and constraint
    tmp_opt.set_minimizer(TrustReg())
    tmp_opt.set_cost_func(StationaryError())
    tmp_opt.set_constraint_func(StationaryConstraint())

    # actual optimization
    tmp = tmp_opt.run_optimization()

    # plot stuff
    # tmp_opt.plot_results()
    tmp_opt.plot_cost_func()

    # print(tmp_opt)

    # print results
    print(f"success: {tmp_opt.sol.success}")
    print(f"z_star: {tmp_opt.sol.x}")
    print(f"    S_in: {tmp_opt.S_in} g/l")
    print(f"    𝛕: {tmp_opt.tau:.3} h")
    print(f"------------------------")
    print(f"Einsatz: {500 * np.sum(tmp_opt.x[-1, :3])}€")
    print(f"Gewinn: {1e6 * tmp_opt.x[1, -1] / (np.sum(tmp_opt.z_star) / tmp_opt.x[1, -1])}€")