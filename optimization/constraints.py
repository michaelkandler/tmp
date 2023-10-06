from typing import List, Tuple

import numpy as np
from scipy.linalg import block_diag

from models import Model
from utils.logger import CustomLogger

my_logger = CustomLogger()


class Constraint:
    """
    Generic class of constraints

    Box and linear constraints can be created via calling self.create_linear_constraints() and
    self.create_box_constraints()
    """

    def __init__(self, *args):
        """
        Instantiate a general constrain function in the form g(z_star)=0

        Parameters
        ----------
        args
        """
        self.name = "abstract_constraints"

        self.bounds = None

        self.x_star = None

        my_logger.debug("creating constraints...")

    def __call__(self, z: np.ndarray) -> float:
        """
        Equality constraint call of the form g(z_star)=0

        Parameters
        ----------
        z: np.ndarray
            Optimization vector

        Returns
        -------

        """
        raise NotImplementedError("Abstract class only")

    def set_time(self, t_k: np.ndarray) -> None:
        """
        Set internal time grid

        Parameters
        ----------
        t_k : np.ndarray
            new internal time grid

        Returns
        -------
        None
        """
        raise NotImplemented("Abstract class only")

    def set_bounds(self, bounds: tuple) -> None:
        """
        Set the bounds given by the tuple

        Parameters
        ----------
        bounds : tuple
            tuple of bounds

        Returns
        -------
        None
        """
        raise NotImplemented("Abstract class only")

    def create_linear_constraints(self) -> np.ndarray:
        """
        Return the A-matrix of a linear constraint-function

        Returns
        -------
        np.ndarray
            A-matrix of linear contraint
        """
        raise NotImplementedError

    def create_box_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create box constraints from self.bounds und z_star.shape, to get z_min<=z_star<=z_max

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            tuple of box constraints like: (z_min, z_max)
        """
        raise NotImplementedError("Abstract class only")

    def create_non_linear_constraints(self) -> None:
        """
        Take the set-point x_star and create a non-linear constraint

        Returns
        -------
        None
        """
        raise NotImplementedError("Abstract class only")

    def grad(self, z: np.ndarray) -> np.ndarray:
        """
        Gradient of constraint function g(z_star)

        Parameters
        ----------
        z: np.ndarray
            optimization vector


        Returns
        -------
        np.ndarray
            gradient of constraint violation
        """
        raise NotImplementedError

    def hess(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Return hessian of the constraints g(z_star) -> should be in convention with scipy

        Parameters
        ----------
        z : np.ndarray
            optimization vector

        Returns
        -------
        np.ndarray

        """
        raise NotImplementedError

    def __str__(self):
        description = f"Abstract class of constraint"

        return description

    def __repr__(self):
        return "Constraint()"


########################################################################################################################


class StationaryConstraint(Constraint):
    def __init__(self):
        super().__init__()

        self.name = "stationary constraint"

    def set_bounds(self, bounds: tuple) -> None:
        """
        Set the bounds given by the tuple

        Parameters
        ----------
        bounds : tuple
            tuple of bounds

        Returns
        -------
        None
        """
        my_logger.debug(f"constraint bounds set to: {bounds}")
        self.bounds = bounds[:-1]

    def create_box_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return box constraints in form z_min<z_star<z_max from pump limits and z_star shape

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds for z_star in form (z_min, z_max)
        """
        if self.bounds is None:
            my_logger.exception("no bounds given, aborting...")
            raise ValueError("No bounds available, use self.set_bounds to get")

        # get pump limits from bounds
        z_lower = np.asarray([b[0] for b in self.bounds if b is not None])
        z_upper = np.asarray([b[1] for b in self.bounds if b is not None])

        my_logger.debug("box constraints calculated")

        return z_lower, z_upper


########################################################################################################################


class DynamicConstraint(Constraint):

    def __init__(self):
        super().__init__()

        self.name = "dynamic_constraints_partial"

        # box constraints
        self.bounds = None

        # time grid
        self.t_k = None

        my_logger.debug(f"name: {self.name}")

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """
        Non-linear constraint inactive ∀ z


        Parameters
        ----------
        z : np.ndarray

        Returns
        -------
        np.ndarray
            in this case -1 ∀ t -> always inactive
        """
        return -1

    def set_time(self, t_k: np.ndarray) -> None:
        """
        Set internal time grid

        Parameters
        ----------
        t_k : np.ndarray
            new internal time grid

        Returns
        -------
        None
        """
        my_logger.debug(f"constraint t_k set to: {t_k}")
        self.t_k = t_k

    def set_bounds(self, bounds: tuple) -> None:
        """
        Set the bounds given by the tuple

        Parameters
        ----------
        bounds : tuple
            tuple of bounds

        Returns
        -------
        None
        """
        my_logger.debug(f"constraint bounds set to: {bounds}")
        self.bounds = bounds

    def set_x_star(self, x_star: np.ndarray) -> None:
        self.x_star = x_star

    def create_box_constraints(self, zero_order=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return box constraints in form z_min<z_star<z_max from pump limits and z_star shape

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds for z_star in form (z_min, z_max)
        """
        if self.bounds is None:
            my_logger.exception("no bounds given, aborting...")
            raise ValueError("No bounds available, use self.set_bounds to get")

        # get pump limits from bounds
        lower_bound = np.asarray([b[0] for b in self.bounds if b is not None])
        upper_bound = np.asarray([b[1] for b in self.bounds if b is not None])

        len_z = self.t_k.shape[0] - 1 if zero_order else self.t_k.shape[0]

        # extend bounds to fit z_star.shape
        z_lower = np.tile(lower_bound, len_z)
        z_upper = np.tile(upper_bound, len_z)

        my_logger.debug("box constraints calculated")

        return z_lower, z_upper

    def __str__(self):
        description = (f"Dynamic Constraints\n"
                       f"  - t_k: {self.t_k}\n"
                       f"  - bounds: {self.bounds}")

        return description

    def __repr__(self):
        representation = f"DynamicConstraint(t_k={self.t_k}, bounds={self.bounds})"

        return representation


class UOptConstraints(DynamicConstraint):

    def __init__(self):
        super().__init__()

        self.name = "u_optimal_constraints"

    def __str__(self):
        description = (f"In-flow optimization Constraints\n"
                       f"  - t_k: {self.t_k}\n"
                       f"  - bounds: {self.bounds}")

        return description

    def __repr__(self):
        representation = f"UOptConstraints(t_k={self.t_k}, bounds={self.bounds})"

        return representation


class UTOptConstraints(DynamicConstraint):

    def __init__(self):
        super().__init__()

        self.name = "ut_optimal_constraints"

    def __str__(self):
        description = (f"In-flow-Time-optimal Constraints\n"
                       f"  - t_k: {self.t_k}\n"
                       f"  - bounds: {self.bounds}")

        return description

    def __repr__(self):
        representation = f"UTOptConstraints(t_k={self.t_k}, bounds={self.bounds})"

        return representation


# === norm optimal ===

class NormOptConstraints(UOptConstraints):

    def __init__(self):
        super().__init__()

        self.name = "norm_opt_constraints"

    def __str__(self):
        description = (f"Norm-optimal Constraints\n"
                       f"  - t_k: {self.t_k}\n"
                       f"  - bounds: {self.bounds}")

        return description

    def __repr__(self):
        representation = f"NormOptConstraints(t_k={self.t_k}, bounds={self.bounds})"

        return representation


# === norm time optimal ===

class TimeNormOptConstraints(UTOptConstraints):

    def __init__(self):
        super().__init__()

        self.name = "norm_optimal_constraints"

    def __str__(self):
        description = (f"Norm-Time-optimal Constraints\n"
                       f"  - t_k: {self.t_k}\n"
                       f"  - bounds: {self.bounds}")

        return description

    def __repr__(self):
        representation = f"TimeNormOptConstraints(t_k={self.t_k}, bounds={self.bounds})"

        return representation


# === fed batch


class FedBatchConstraints(UTOptConstraints):

    def __init__(self):
        super().__init__()

        self.name = "fed_batch_constraints"

    def __call__(self, z: np.ndarray) -> float:
        """
        Isoperimetric constraint for end-volume

        Parameters
        ----------
        z : np.ndarray
            optimization vector

        Returns
        -------
        float
            g(z)
        """
        g = np.mean(z[:-1]) * z[-1] * int(len(z) / len(self.t_k))
        return g

    def grad(self, z) -> np.ndarray:
        """

        Parameters
        ----------
        z

        Returns
        -------

        """
        grad = np.ones(z.shape) * z[-1]
        grad[-1] = np.sum(z[:-1])

        return grad.T / (z[:-1].shape[0]) * int(len(z) / len(self.t_k))

    def hess(self, z, v) -> np.ndarray:
        """

        Parameters
        ----------
        z : np.ndarray

        v : np.ndarray

        Returns
        -------

        """

        hess = np.zeros((z.shape[0], z.shape[0]))
        hess[:-1, -1] = 1
        hess[-1, :-1] = 1

        return hess


if __name__ == '__main__':
    example_constraint = Constraint()
    print(example_constraint)
