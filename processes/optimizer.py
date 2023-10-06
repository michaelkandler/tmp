import numpy as np
import datetime

from processes import Simulation

# type hinting
from optimization import Cost
from optimization.minimizer import Minimizer
from optimization.constraints import Constraint

from utils.logger import CustomLogger

my_logger = CustomLogger()


class Optimization(Simulation):
    """
    Implement interface for optimization of a target based on a simulation.


    """

    def __init__(self, name: str):
        super().__init__(name)

        # error functions and its restraints
        self.error_func = None
        self.constraint_func = None

        # set bounds of pump limits
        self.bounds = None

        # set target if necessary
        self.x_star = None

        # set and configure optimization algorithm
        self.minimizer = None
        self.z_0 = None
        self.weights = None

        # optimized variable and optimization report
        self.T_opt = None
        self.z_star = None
        self.success = None
        self.sol = None

    # === building options ===

    def set_cost_func(self, error_func: Cost) -> None:
        """
        Set the cost function for the minimizer.

        Parameters
        ----------
        error_func : Cost
            The cost function.
        """
        if not isinstance(error_func, Cost):
            my_logger.exception("error function not of type cost, aborting...")
            raise ValueError("Error function must be of type 'Cost' in optimization package")

        my_logger.debug(f"setting error function to {error_func}")
        self.error_func = error_func

    def set_constraint_func(self, constraint_func: Constraint) -> None:
        """
        Set the constraint function for the optimization. The constraint function will act as the non-linear constraint
        g(z) =(<) 0 if a __call__ function is implemented and create box- and linear constraints if available in
        process

        Parameters
        ----------
        constraint_func : Constraint
            Instance of Constraint class

        Returns
        -------
        None
        """
        if not isinstance(constraint_func, Constraint):
            my_logger.exception("constraint functino is no instance of Constraint, aborting...")
            raise ValueError("Constraint functino is no instance of Constraint from optimization module")

        my_logger.debug(f"setting constraint_function: {constraint_func}")
        self.constraint_func = constraint_func

    def set_bounds(self, *args) -> None:
        """
        Set the box boundaries for the target vector.

        Parameters
        ----------
        args : tuple
            The box boundaries for the target vector.
        """
        raise NotImplemented

    def set_x_star(self, x_star: np.ndarray) -> None:
        """
        Set set-point for optimization

        Parameters
        ----------
        x_star : np.ndarray
            Point in state-space to reach

        Returns
        -------
        None
        """
        if len(x_star) != 5:
            raise ValueError("Set-point must have shape 5")
        if np.any(x_star < 0.):
            raise ValueError("All set-point values must be positive")

        my_logger.debug(f"setting set-point to: {x_star}")
        self.x_star = x_star

    def set_minimizer(self, minimizer: Minimizer) -> None:
        """
        Setting the minimizer from the standard (Simplex) to the given minimizer

        Parameters
        ----------
        minimizer : str
            name of scipy.minimize minimizer

        Returns
        -------
        None
        """
        if not isinstance(minimizer, Minimizer):
            my_logger.exception("minimizer not of type Minimizer, aborting...")
            raise ValueError("Optimizer not of type Minimizer of optimization package")

        my_logger.debug(f"minimizer set to {self.minimizer}")
        self.minimizer = minimizer

    def set_initial_guess(self, z0: np.ndarray) -> None:
        """
        Set the starting point for optimization.

        Parameters
        ----------
        z0 : np.ndarray
            The starting point for optimization.
        """
        my_logger.debug(f"setting initial guess to:{z0}")
        self.z_0 = z0

    def set_weights(self, weights: tuple) -> None:
        """
        Set the weights for the minimization algorithm

        Parameters
        ----------
        weights : tuple
            Tuple of np.ndarray of weights (eg. (k_end, k_i, k_reg))

        Returns
        -------
        None
        """
        raise NotImplementedError

    # === run optimizations ===

    def get_constraints(self) -> tuple:
        """
        Create linear or box constraints for optimization.

        Returns
        ----------
        None
        """
        raise NotImplemented

    def target(self, z: np.ndarray) -> float:
        """
        Set the actual target for the minimizer.

        Parameters
        ----------
        z : np.ndarray
            The target variable to optimize

        Returns
        ----------
        float
            Cost for the given variable z
        """
        raise NotImplemented

    def run_optimization(self) -> None:
        """
        Run the optimization and save all states and results internally.

        Returns
        ----------
        None
        """
        if self.minimizer is None:
            raise ValueError('Optimizer must be set')
        if self.error_func is None:
            raise ValueError('Error function must be set')
        if self.constraint_func is None:
            my_logger.warning("no constraints set for this problem, consider using self.set_constraint_func")
        if self.z_0 is None:
            raise ValueError('Starting point must be given')

    # === miscellaneous ===

    def __str__(self) -> str:
        """
        Return the string representation of the Optimization object.

        Returns
        -------
        str
            The string representation of the Optimization object.
        """
        creation_time = datetime.datetime.fromtimestamp(self.timestamp)
        readable_creation_time = creation_time.strftime('%Y-%m-%d %H:%M:%S')

        description = f"Optimization '{self.name}' created at {readable_creation_time}\n" \
                      f"-----------------------------------------------------\n" \
                      f"{self.controlled_plant}\n" \
                      f"-----------\n" \
                      f"Optimizer configuration\n" \
                      f"----------------------\n" \
                      f"Minimizer: {self.minimizer}\n" \
                      f"-----------\n" \
                      f"Cost function: {self.error_func}\n" \
                      f"-----------\n" \
                      f"Constraints: {self.constraint_func}\n"\
                      f"-----------\n" \
                      f"Weights: {self.weights}\n" \
                      f"-----------\n" \
                      f"Bounds: {self.bounds}\n" \
                      f"-----------\n" \
                      f"Initial guess: {list(self.z_0)}\n" \
                      f"-----------\n" \
                      f"Set-value: {self.x_star}\n" \
                      f"-----------\n" \
                      f"z*: {self.z_star}\n" \
                      f"-----------\n"

        return description
    
    def __repr__(self):
        representation = (f"Optimization(controlled_plant={repr(self.controlled_plant)}, "
                          f"error_func={repr(self.error_func)}, "
                          f"constraint_func={repr(self.constraint_func)}, "
                          f"bounds={self.bounds}, x_star={self.x_star}, "
                          f"minimizer={repr(self.minimizer)}, z_0={self.z_0}, weights={self.weights})")

        return representation


if __name__ == '__main__':
    pass
