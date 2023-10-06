import numpy as np

from scipy.optimize import minimize
import scipy.optimize as optimize
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

from utils.logger import CustomLogger

my_logger = CustomLogger()


class Minimizer:
    """
    Wrapper to keep arbitrary minimizers in a consistent format

    Will mostly be used with scipy.optimize
    """

    def __init__(self):
        """
        No initialization argument needed for any instance
        """
        self.name = "minimizer"

        # callback function
        self.callback = None

    def __call__(self, func: callable, x0: np.ndarray, args=(), method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None) -> any:
        """
        Perform the actual optimization

        Parameters
        ----------
        func
        x0
        args
        method
        jac
        hess
        hessp
        bounds
        constraints
        tol
        callback
        options

        Returns
        -------

        """
        raise NotImplementedError

    def set_callback(self, callback_fun: callable) -> None:
        """
        Set a callback function for the minimizer, can be used to analyse the optimization but will slow down the
        algorithm.

        Parameters
        ----------
        callback_fun : callable with the correct signature TODO: check signature

        Returns
        -------
        None
        """
        self.callback = callback_fun
        my_logger.warning(f"callback function {callback_fun} set. My affect performance and cause memory issues.")

    def build_box_constraints(self, box_constraints: tuple) -> any:
        """
        Build box_constraints compatible with minimizer

        Parameters
        ----------
        box_constraints : tuple
            Tuple of box constraints (lower, upper)

        Returns
        -------
        any
            whatever the minimizer needs
        """
        if len(box_constraints) != 2:
            raise ValueError("Lower and upper bounds must be given")
        if len(box_constraints[0]) != len(box_constraints[1]):
            raise ValueError("Upper and lower bounds must have same shape")

    def build_lin_constraints(self, A: np.ndarray = None, lb: np.ndarray = None, ub: np.ndarray = None) -> any:
        """
        Build linear constraint object for specific

        Parameters
        ----------
        A : np.ndarray
            Linear constraint matrix of form m x n, whereas m is the number of constraints and n is the shape of z
        lb : np.ndarray
            vector of lower bounds of shape m
        ub : np.ndarray
            vector of upper bounds of shape m

        Returns
        -------
        any
            linear constraints suitable for optimizer, None if not given
        """
        raise NotImplementedError("Abstract class only")

    def build_non_lin_constraints(self, func: callable, lb=None, ub=None) -> any:
        """
        Create non-linear constraints g(z)=0 for the specific optimizer

        Parameters
        ----------
        func : callable
            vector function of shape m
        lb : np.ndarray
            vector of shape m given for g(z) >= lb, 0 if not given
        ub : np.ndarray
            vector of shape m given for g(z) <= ub, 0 if not given

        Returns
        -------
        any
            Non-linear constraint function
        """
        raise NotImplementedError

    def __repr__(self):
        description = "Minimizer()"

        return description

    def __str__(self):
        description = "generic minimizer"

        return description


class ScipyMinimizers(Minimizer):
    """
    Parent for every Scipy minimizer
    """

    def __init__(self):
        super().__init__()

        self.name = "scipy_optimizer"

    def build_box_constraints(self, box_constraints: tuple) -> Bounds:
        """
        Build box constraints for scipy-minimizers

        Parameters
        ----------
        box_constraints

        Returns
        -------

        """
        # check plausibility
        super().build_box_constraints(box_constraints)

        return Bounds(lb=box_constraints[0], ub=box_constraints[1])

    def build_lin_constraints(self, A: np.ndarray = None, lb: np.ndarray = None, ub: np.ndarray = None) -> any:
        """
        Build linear constraint object for specific

        Parameters
        ----------
        A : np.ndarray
            Linear constraint matrix of form m x n, whereas m is the number of constraints and n is the shape of z
        lb : np.ndarray
            vector of lower bounds of shape m
        ub : np.ndarray
            vector of upper bounds of shape m

        Returns
        -------
        any
            linear constraints suitable for optimizer, None if not given
        """
        if A is None:
            return None
        if lb is None and ub is None:
            raise ValueError("lower or upper bounds have to be given")
        if lb is not None and ub is not None and len(ub) != len(lb):
            raise ValueError("Lower and upper bounds must have same shape")

        return LinearConstraint(A, lb=lb, ub=ub)

    def build_non_lin_constraints(self, func: callable, lb=None, ub=None) -> any:
        """
        Create non-linear constraints g(z)=0 for the specific optimizer

        Parameters
        ----------
        func : callable
            vector function of shape m
        lb : np.ndarray
            vector of shape m given for g(z) >= lb, 0 if not given
        ub : np.ndarray
            vector of shape m given for g(z) <= ub, 0 if not given

        Returns
        -------
        any
            Non-linear constraint function
        """
        lb = 0 if lb is None else lb
        ub = 0 if ub is None else ub

        jac = func.grad
        hess = func.hess
        try:
            jac(np.asarray([1]))
        except NotImplementedError:
            jac = None
        try:
            hess(np.asarray([1]), np.asarray([1]))
        except NotImplementedError:
            hess = None

        return NonlinearConstraint(func, lb, ub, jac=jac, hess=hess)


######################################################## local #########################################################

class NelderMead(ScipyMinimizers):

    def __init__(self):
        super().__init__()
        self.name = "nelder-mead"

    def __call__(self, func: callable, x0: np.ndarray, args=(), method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None):
        return minimize(func, x0, method="nelder-mead", callback=self.callback, args=args, jac=jac, hess=hess,
                        hessp=hessp,
                        bounds=bounds, constraints=constraints, tol=tol, options=options)

    def __str__(self):
        description = "nelder_mead"

        return description

    def __repr__(self):
        description = "NelderMead()"

        return description


class TrustReg(ScipyMinimizers):

    def __init__(self):
        super().__init__()
        self.name = "nelder-mead"

    def __call__(self, func: callable, x0: np.ndarray, args=(), method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None):
        return minimize(func, x0, method="trust-constr", callback=self.callback, args=args, jac=jac, hess=hess,
                        hessp=hessp,
                        bounds=bounds, constraints=constraints, tol=tol, options=options)

    def __str__(self):
        description = f"trust-region"

        return description

    def __repr__(self):
        description = "TrustReg()"

        return description


class COBYLA(ScipyMinimizers):

    def __init__(self):
        super().__init__()
        self.name = "nelder-mead"

    def __call__(self, func: callable, x0: np.ndarray, args=(), method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None):
        return minimize(func, x0, method="COBYLA", callback=self.callback, args=args, jac=jac, hess=hess, hessp=hessp,
                        bounds=bounds, constraints=constraints, tol=tol, options=options)

    def __str__(self):
        description = f"Constrained Optimization BY Linear Approximation"

        return description

    def __repr__(self):
        description = "COBYLA()"

        return description


class SLSQP(ScipyMinimizers):

    def __init__(self):
        super().__init__()
        self.name = "nelder-mead"

    def __call__(self, func: callable, x0: np.ndarray, args=(), method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None):
        return minimize(func, x0, method="SLSQP", callback=self.callback, args=args, jac=jac, hess=hess, hessp=hessp,
                        bounds=bounds, constraints=constraints, tol=tol, options=options)

    def __str__(self):
        description = f"SLSQP"

        return description

    def __repr__(self):
        description = "SLSQP()"

        return description


####################################################### global #########################################################
# TODO raise exception for unavailable keywords
class DualAnnealing(ScipyMinimizers):

    def __init__(self):
        super().__init__()

        self.name = "dual_annealing"

    def __call__(self, func: callable, x0: np.ndarray, args=(), method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None):
        if constraints != ():
            raise ValueError("Dual annealing only implements box constraints")

        return optimize.dual_annealing(func, x0=x0, bounds=bounds)

    def __str__(self):
        description = f"dual_annealing"

        return description


class DifferentialEvolution(ScipyMinimizers):

    def __init__(self):
        super().__init__()

        self.name = "differential_evolution"

    def __call__(self, func: callable, x0: np.ndarray, args=(), method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None):
        if constraints != ():
            raise ValueError("Differential evolution only implements box constraints")

        return optimize.differential_evolution(func, x0=x0, bounds=bounds)

    def __str__(self):
        description = f"dual_annealing"

        return description


class SHOG(ScipyMinimizers):

    def __init__(self):
        super().__init__()

        self.name = "shog"

    def __call__(self, func: callable, x0: np.ndarray, args=(), method=None, jac=None, hess=None, hessp=None,
                 bounds=None, constraints=(), tol=None, callback=None, options=None):
        if constraints != ():
            raise ValueError("SHOG only implements box constraints")

        return optimize.shgo(func, bounds=bounds)

    def __str__(self):
        description = "shog-minimizer"

        return description
