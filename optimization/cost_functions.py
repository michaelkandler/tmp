"""
define error functions for optimization
"""

import numpy as np

from utils.logger import CustomLogger

my_logger = CustomLogger()


class Cost:
    """
    Implements a scalar cost function for n-dimensional vectors
    """

    def __init__(self, *args) -> None:
        """

        Parameters
        ----------
        args
        """
        self.name = None

        self.weights = None

        my_logger.debug(f"creating cost function...")

    def __call__(self, z: np.ndarray) -> float:
        """
        Calculate cost of given vector z_star

        Parameters
        ----------
        z : np.ndarray
            column optimization vector

        Returns
        -------
        float
            scalar cost
        """
        raise NotImplemented

    def grad(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the cost function

        Parameters
        ----------
        z : np.ndarray
            optimization variable column vector

        Returns
        -------
        np.ndarray
            Gradient of cost function at point z_star
        """
        raise NotImplemented

    def hess(self, z: np.ndarray) -> np.ndarray:
        """
        Calculate the hessian of the cost function

        Parameters
        ----------
        z : np.ndarray
            optimization variable column vector
        Returns
        -------
        np.ndarray
            Hessian of cost function at point z_star
        """
        raise NotImplemented

    def __str__(self) -> str:
        description = f"Interface for error function:\\n" \
                      f"  - weights: {self.weights}"

        return description


########################################################################################################################


class WeightedResidues(Cost):
    """
    Implement cost of weighted residues for model calibration.

    The mean of every state over time is weighted before
    averaging
    """

    def __init__(self, data: np.ndarray, weights=np.array([1.])) -> None:
        """
        Parameters
        ----------
        data : np.ndarray
            experimental data to fit
        weights : np.ndarray
            weights for sum of each state
        """
        super().__init__()

        self.name = "wssr"

        self.adjust_weights(weights)

        # set experimental data
        self.data = data

        my_logger.debug(f"name: {self.name}, weights: {self.weights}, data shape: {self.data.shape}")

    def adjust_weights(self, weights: np.ndarray) -> None:
        """
        Adjust weight for cost function and check plausibility.

        Parameters
        ----------
        weights : np.ndarray
            new weights

        Returns
        -------
        None
        """

        # check plausibility
        if np.any(weights < 0):
            raise ValueError('Only positive weights are reasonable')

        self.weights = weights

        my_logger.debug(f"weights set to {self.weights}")

    def __call__(self, x: np.ndarray) -> float:
        """
        Calculate the weighted sums of residues from current state of system

        Parameters
        ----------
        x : np.ndarray
            simulation data with approximated parameters in non-flattened shape : (n_times, n_states)
        Returns
        -------
        float
            scalar cost
        """
        error = np.mean(((self.data - x) / self.weights) ** 2, axis=0)
        error = np.mean(error)

        return float(error)

    def __str__(self):
        description = f"Error of weighted residues: ∑︀₁ᴹ∑︀₁ᴺxₘₙ²/N/M\n" \
                      f"  - weights: {self.weights}"

        return description

    def __repr(self):
        description = f"WeightedResidues(data={self.data}, weights={self.weights}))"

        return description


########################################################################################################################


class StationaryError(Cost):
    """
    Implements error of INA-modeled process, based on yields and replacement time.

    k_y (N+A+I)² + k_s (N/τ)
    """

    def __init__(self, weights=np.array([1., 1.])):
        """
        Parameters
        ----------
        weights : np.ndarray
            scalar values to weigh: yield (0) and space-time-yield (1)
        """
        super().__init__()

        self.name = "stationary_error"

        self.adjust_weights(weights)

        self.V = 1.

        my_logger.debug(f"name: {self.name}, weights: {self.weights}")

    def adjust_weights(self, weights: np.ndarray) -> None:
        """
        Adjust weight for cost function and check plausibility.

        Parameters
        ----------
        weights : np.ndarray
            scalar values to weigh: yield (0) and space-time-yield (1)

        Returns
        -------
        None
        """

        # check plausibility
        if any(w < 0. for w in weights):
            raise ValueError('Only positive weights are reasonable for yields')
        if len(weights) != 2:
            raise ValueError("Please provide two weights. Yield is first space-time-yield is second")
        self.weights = np.asarray(weights)

        my_logger.debug(f"weights set to {self.weights}")

    def adjust_V(self, V: float) -> None:
        """
        Adjust the Volume of the stationary reactor

        Parameters
        ----------
        V : float
            New Volume

        Returns
        -------
        None
        """
        my_logger.debug(f"setting volume for stationary error to {V}")
        self.V = V


    def __call__(self, z: np.ndarray) -> float:
        """
        Calculate scalar cost for chemostat operation. Combination of yield and space-time-yield

        Parameters
        ----------
        z : np.ndarray
            optimization vector consisting of: [I, N, A, tau]

        Returns
        -------
        float
            scalar error
        """
        # unravel optimization vector
        I, N, A, tau, S_in = z

        # get yield weights
        p_yield, p_space_time_yield = self.weights

        # actual error
        error = p_yield * ((I + A + S_in)/tau) ** 2 - p_space_time_yield * (N * self.V / tau) ** 2

        return error

    def __str__(self):
        description = f"Error of stationary point: k_p (I+A+Sₑ)² + k_s (N/τ)²\n" \
                      f"  - weight yield: {self.weights[0]} g/g\n" \
                      f"  - weight space-time yield: {self.weights[1]} hl/g\n"

        return description

    def __repr__(self):
        representation = f"StationaryError(weights={self.weights})"

        return representation


########################################################################################################################


class DynamicError(Cost):
    """
    Implement error of dynamic optimization on the basis of partial discretization.

    d
    """

    def __init__(self):
        super().__init__()

        self.name = "dynamic_error"

        self.x_star = None
        self.weights = None
        self.t = None

        self.n_states, self.n_inputs = 5, 4

        my_logger.debug(f"{self.name} instantiated with weights: {self.weights}")

    def adjust_set_point(self, x_star: np.ndarray) -> None:
        """
        Adjust set-point for cost function

        Parameters
        ----------
        x_star : np.ndarray
            set-point in state space

        Returns
        -------
        None
        """
        self.x_star = x_star

    def adjust_time(self, time: np.ndarray):
        """
        Adjust time-grid to calculate cost over.

        Parameters
        ----------
        time : np.ndarray
            new time-grid

        Returns
        -------
        None
        """
        self.t = time

        my_logger.debug(f"time grid adjusted: t_0 = {self.t[0]}, t_end = {self.t[-1]}")

    def adjust_weights(self, weights: tuple) -> None:
        """
        Adjust weight for cost function and check plausibility.

        Parameters
        ----------
        weights : np.ndarray
            new weights

        Returns
        -------
        None
        """
        raise NotImplemented

    def _lagrangian_density(self, x_i, u_i, T) -> float:
        """
        Lagrangian-density of cost function. Consist of weighted sum of norm-error from set-point and weighted
        time-inkrement

        Parameters
        ----------
        x_i : np.ndarray
            state at time-step i

        Returns
        -------
        float
            scalar lagrangian error at time-step i
        """
        raise NotImplemented

    def _mayer_part(self, x_end: np.ndarray, T: float) -> float:
        """
        Calculate mayer-/end-part of cost function. Consisting of norm-error to set-point and time-step N.

        Parameters
        ----------
        x_end : np.ndarray
            state at time-step N

        Returns
        -------
        float
            scalar end-term of cost function
        """
        raise NotImplemented

    def _regularisation(self, u: np.ndarray) -> float:
        """
        Term to regularize the input of the plant.

        It is implemented by minimizing the cumulative difference of |u_i - u_i+1|

        Parameters
        ----------
        u : np.ndarray
            plant input

        Returns
        -------
        np.ndarray
            Regularization error
        """
        k_reg = self.weights[2]

        u_tilde = np.roll(u, -1, axis=1) - u
        u_tilde = u_tilde[:, -1] ** 2
        u_tilde = (k_reg * u_tilde.T).T

        return float(np.sum(u_tilde))

    def __call__(self, z: np.ndarray) -> float:
        """
        Calculate cost function using the given optimization vector and lagrangian- and mayer-density

        Parameters
        ----------
        z : np.ndarray
            optimization vector

        Returns
        -------
        float
            scalar dynamic error
        """
        if self.t is None:
            raise ValueError("time grid must be set to calculate error")

        # unravel optimization vector to shape (n_times, n_states)
        x = z[:self.n_states * self.t.shape[0]]
        x = np.reshape(x, (self.n_states, self.t.shape[0]), order='F')

        u = z[self.n_states * self.t.shape[0]:] if z.shape[0] % self.t.shape[0] == 0. else z[
                                                                                           self.n_states * self.t.shape[
                                                                                               0]:-1]

        n_inputs = int(u.shape[0] / self.t.shape[0])
        u = np.reshape(u, (n_inputs, self.t.shape[0]), order='F')
        u = np.vstack((u, np.zeros((self.n_inputs - n_inputs, self.t.shape[0]))))
        u = u[:, :-1]

        T = z[-1]

        # get values at t_k+1
        x_plus = np.roll(x, -1, axis=1)
        t_plus = np.roll(self.t, -1)

        # use trapezoids rule for approximated integration
        trapezoids = 1 / 2 * (t_plus[:-1] - self.t[:-1]) * \
                     (self._lagrangian_density(x[:, :-1], u, T) + self._lagrangian_density(x_plus[:, :-1], u, T))
        lagrange = np.sum(trapezoids, axis=0)

        # calculate error
        mayer = self._mayer_part(x[:, -1], T)
        regularisation = self._regularisation(u)

        return mayer + lagrange + regularisation

    def __str__(self) -> str:
        description = f"Abstract class for dynamic error \n" \
                      f"  - weights end cost: {self.weights}\n" \
                      f"  - time : {self.t}"

        return description


class UOptError(DynamicError):
    """
    Reach the set-point optimally according to the norm of the difference |x_N - x_star|

    d
    """

    def __init__(self):
        super().__init__()

        self.name = "u_opt_error"

    def adjust_weights(self, weights: tuple) -> None:
        """
        Adjust weight for cost function and check plausibility.

        Parameters
        ----------
        weights : np.ndarray
            new weights

        Returns
        -------
        None
        """

        # check plausibility
        if any([np.any(w < 0.) for w in weights]):
            raise ValueError('Only positive weights are reasonable for yields')

        if len(weights) != 3:
            raise ValueError("Number of weights must be 3. Mayer, Lagrange, Regularisation")

        self.weights = weights

        my_logger.debug(f"weights set to {self.weights}")

    def _lagrangian_density(self, x_i, u_i, T) -> float:
        """
        Lagrangian-density of cost function. Consist of weighted sum of norm-error from set-point and weighted
        time-inkrement

        Parameters
        ----------
        x_i : np.ndarray
            state at time-step i
        u_i : np.ndarray
            input vector, not used

        Returns
        -------
        float
            scalar lagrangian error at time-step i
        """
        k_x = self.weights[1]

        diff_x = x_i - self.x_star[:, None]
        weighted_diff_x = (k_x * diff_x.T).T

        return np.linalg.norm(weighted_diff_x, axis=0)

    def _mayer_part(self, x_end: np.ndarray, T: float | None) -> float:
        """
        Calculate mayer-/end-part of cost function. Consisting of norm-error to set-point and time-step N.

        Parameters
        ----------
        x_end : np.ndarray
            state at time-step N
        T : float
            end time, not used

        Returns
        -------
        float
            scalar end-term of cost function
        """
        k_end = self.weights[0]

        weighted_diff = (k_end * (x_end - self.x_star).T).T

        return np.linalg.norm(weighted_diff)

    def __call__(self, z: np.ndarray) -> float:
        """
        Calculate cost function using the given optimization vector and lagrangian- and mayer-density

        Parameters
        ----------
        z : np.ndarray
            optimization vector

        Returns
        -------
        float
            scalar dynamic error
        """
        if self.x_star is None:
            raise ValueError("optimal point must be set")

        return super().__call__(z)

    def __str__(self):
        description = f"Error of minimal arrival norm, J = k_N|x_s - x_n|² + k_n∫|x_s - x(t)|dt\n" \
                      f"  - weight end cost: {self.weights[0]} g/g\n" \
                      f"  - weight continuous cost: {self.weights[1]} hg/g\n" \
                      f"  - weight regularisation cost: {self.weights[2]} h²/l²"

        return description

    def __repr__(self):
        representation = f"NormOptError(weights={self.weights})"

        return representation


class UTOptError(DynamicError):

    def __init__(self):
        super().__init__()

        self.name = "ut_opt_error"

    def adjust_weights(self, weights: tuple) -> None:
        """
        Adjust weight for cost function and check plausibility.

        Parameters
        ----------
        weights : np.ndarray
            new weights

        Returns
        -------
        None
        """

        # check plausibility
        if any([np.any(w < 0.) for w in weights]):
            raise ValueError('Only positive weights are reasonable for yields')

        self.weights = weights

        my_logger.debug(f"weights set to {self.weights}")

    def __call__(self, z: np.ndarray) -> float:
        """
        Calculate cost function using the given optimization vector and lagrangian- and mayer-density

        Parameters
        ----------
        z : np.ndarray
            optimization vector

        Returns
        -------
        float
            scalar dynamic error
        """
        if self.x_star is None:
            raise ValueError("optimal point must be set")

        return super().__call__(z)

    def __repr__(self) -> str:
        description = f"Error of minimal arrival timen" \
                      f"  - weight end cost: {self.weights[0]} g/g\n" \
                      f"  - weight continuous cost: {self.weights[1]} hg/g\n" \
                      f"  - weight time cost: {self.weights[2]} hl/g\n" \
                      f"  - weight regularisation cost: {self.weights[3]} h²/l²"

        return description


# === norm optimal errors ===


class NormOptError(UOptError):

    def __init__(self):
        super().__init__()

        self.name = "norm_opt_error"

    def _lagrangian_density(self, x_i, u_i, T) -> float:
        """
        Lagrangian-density of cost function. Consist of weighted sum of norm-error from set-point and weighted
        time-inkrement

        Parameters
        ----------
        x_i : np.ndarray
            state at time-step i
        u_i : np.ndarray
            input vector, not used

        Returns
        -------
        float
            scalar lagrangian error at time-step i
        """
        k_x = self.weights[1]

        diff_x = x_i - self.x_star[:, None]
        weighted_diff_x = (k_x * diff_x.T).T

        return np.linalg.norm(weighted_diff_x, axis=0)

    def _mayer_part(self, x_end: np.ndarray, T: float | None) -> float:
        """
        Calculate mayer-/end-part of cost function. Consisting of norm-error to set-point and time-step N.

        Parameters
        ----------
        x_end : np.ndarray
            state at time-step N
        T : float
            end time, not used

        Returns
        -------
        float
            scalar end-term of cost function
        """
        k_end = self.weights[0]

        weighted_diff = (k_end * (x_end - self.x_star).T).T

        return np.linalg.norm(weighted_diff)

    def __str__(self):
        description = f"Error of minimal arrival norm, J = k_N|x_s - x_n|² + k_n∫|x_s - x(t)|dt\n" \
                      f"  - weight end cost: {self.weights[0]} g/g\n" \
                      f"  - weight continuous cost: {self.weights[1]} hg/g\n" \
                      f"  - weight regularisation cost: {self.weights[2]} h²/l²"

        return description

    def __repr__(self):
        representation = f"NormOptError(weights={self.weights})"

        return representation


# === norm and time opt errors ===


class TimeNormError(UTOptError):

    def __init__(self):
        super().__init__()

        self.name = "norm_time_opt_error"

    def _lagrangian_density(self, x_i, u_i, T) -> float:
        """
        Lagrangian-density of cost function. Consist of weighted sum of norm-error from set-point and weighted
        time-inkrement

        Parameters
        ----------
        x_i : np.ndarray
            state at time-step i

        Returns
        -------
        float
            scalar lagrangian error at time-step i
        """
        k_x = self.weights[1]

        diff_x = x_i - self.x_star[:, None]
        weighted_diff_x = (k_x * diff_x.T).T

        return np.linalg.norm(weighted_diff_x, axis=0)

    def _mayer_part(self, x_end: np.ndarray, T: float) -> float:
        """
        Calculate mayer-/end-part of cost function. Consisting of norm-error to set-point and time-step N.

        Parameters
        ----------
        x_end : np.ndarray
            state at time-step N

        Returns
        -------
        float
            scalar end-term of cost function
        """
        k_end = self.weights[0]
        k_t = self.weights[2]

        weighted_diff = (k_end * (x_end - self.x_star).T).T

        return np.linalg.norm(weighted_diff)**2 + float(k_t * T ** 2)

    def __repr__(self) -> str:
        description = f"Error of minimal arrival time, , J = k_N|x_s - x_n|² + k_t T² + k_n∫|x_s - x(t)|dt\ \n" \
                      f"  - weight end cost: {self.weights[0]} g/g\n" \
                      f"  - weight continuous cost: {self.weights[1]} hg/g\n" \
                      f"  - weight time cost: {self.weights[2]} hl/g\n" \
                      f"  - weight regularisation cost: {self.weights[3]} h²/l²"

        return description


# === fed-batch error ===

class FedBatchError(UTOptError):

    def __init__(self):
        super().__init__()

        self.name = "fed_batch_error"


class FedBatchErrorYield(FedBatchError):

    def __init__(self):
        super().__init__()

        self.name = "fed_batch_error"

    def _lagrangian_density(self, x_i: np.ndarray, u_i: np.ndarray, T: float) -> float:
        """
        Lagrangian-density of cost function. Consist of weighted sum of norm-error from set-point and weighted
        time-inkrement

        Parameters
        ----------
        x_i : np.ndarray
            state at time-step i

        Returns
        -------
        float
            scalar lagrangian error at time-step i
        """
        return 0

    def _mayer_part(self, x_end: np.ndarray, T: float) -> float:
        """
        Calculate mayer-/end-part of cost function. Consisting of norm-error to set-point and time-step N.

        Parameters
        ----------
        x_end : np.ndarray
            state at time-step N

        Returns
        -------
        float
            scalar end-term of cost function
        """
        # get weights
        k_y, k_sty, _ = self.weights

        # get states and time
        I, N, A, D, V = x_end

        # mayer term space-time yield
        st_yield_end = np.linalg.norm(V * N / T)
        _mayer_st_yield = -(st_yield_end * k_sty)

        # mayer term for yield
        yield_end = np.linalg.norm(N / (I + N + A))
        _mayer_yield_end = -(yield_end * k_y)

        return _mayer_st_yield

    def __str__(self):
        description = f"Error of fed-batch process: J = k_I ∫ u(t)/T dt - k_N (N*V/T)\n" \
                      f"  - weight yield: {self.weights[0]} hl/g\n" \
                      f"  - weight space-time yield: {self.weights[1]} hl/g\n"

        return description

    def __repr__(self):
        representation = f"FedBatchError(weights={self.weights})"

        return representation


class FedBatchErrorFlow(FedBatchError):

    def __init__(self):
        super().__init__()

        self.name = "fed_batch_error"

    def _lagrangian_density(self, x_i: np.ndarray, u_i: np.ndarray, T: float) -> float:
        """
        Lagrangian-density of cost function. Consist of weighted sum of norm-error from set-point and weighted
        time-inkrement

        Parameters
        ----------
        x_i : np.ndarray
            state at time-step i

        Returns
        -------
        float
            scalar lagrangian error at time-step i
        """
        k_u = self.weights[0]

        _lagrange_density = u_i[0] / T
        _lagrange_density = (_lagrange_density * k_u) ** 2

        return _lagrange_density

    def _mayer_part(self, x_end: np.ndarray, T: float) -> float:
        """
        Calculate mayer-/end-part of cost function. Consisting of norm-error to set-point and time-step N.

        Parameters
        ----------
        x_end : np.ndarray
            state at time-step N

        Returns
        -------
        float
            scalar end-term of cost function
        """
        # get weights

        k_sty = self.weights[1]

        # get states and time
        I, N, A, D, V = x_end

        # mayer term space-time yield
        st_yield_end = np.linalg.norm(V * N / T)
        _mayer_st_yield = -(st_yield_end * k_sty) ** 2

        return _mayer_st_yield

    def __str__(self):
        description = f"Error of fed-batch process: J = k_I ∫ u(t)/T dt - k_N (N*V/T)\n" \
                      f"  - weight yield: {self.weights[0]} hl/g\n" \
                      f"  - weight space-time yield: {self.weights[1]} hl/g\n" \
                      f"  - regularization: {self.weights[2]}"

        return description

    def __repr__(self):
        representation = f"FedBatchError(weights={self.weights})"

        return representation


if __name__ == '__main__':
    example_error = StationaryError(np.asarray([1, 1]))

    print(example_error)
