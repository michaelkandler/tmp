"""
Feed forward controllers, after configuration they ignore x and just return an input u(t). The implement different
versions
"""
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator

from controllers import Controller
from utils.logger import CustomLogger

my_logger = CustomLogger()


class FeedForward(Controller):
    """
    Simple feedforward controller

    Just returns input corresponding to time t
    """

    def __init__(self, u: np.ndarray | None, t: np.ndarray | None):
        """
        Parameters
        ----------
        u : np.ndarray
            Out flow at knot-points. Can be given None for initialisation if input will often be changed via
            self.update_u()
        t : np.ndarray
            Time values for corresponding knot-points, can also be left blank
        """
        super().__init__()

        self.name = "feed_forward"

        # time-grid and according outputs
        self.u = u
        self.t = t

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError("Abstract class only")

    # === builder options ===
    def update_u(self, u: np.ndarray, t: np.ndarray) -> None:
        """
        Update the input to feed forward

        Parameters
        ----------
        u : np.ndarray
            input vector every u_k corresponds to [t_k, t_k+1)  [l/h]
        t : np.ndarray
            time-vector corresponding to u (strictly monotonous)[h]

        Returns
        -------
        None
        """
        self.u = u
        self.t = t

    # === access data ===

    def interpolate_u(self, t: np.ndarray) -> np.ndarray:
        """
        Interpolate u according to the order of the feed_forward type

        Parameters
        ----------
        t : np.ndarray
            time vector to interpolate along

        Returns
        -------
        np.ndarray
            interpolated u
        """
        raise NotImplementedError

    # === miscellaneous ===

    def __str__(self):
        description = f"Feedforward abstract controller\n" \
                      f"  - length: {self.t.shape[0] if self.t is not None else None}"

        return description

    def __repr__(self):
        representation = f"FeedForward(t={self.t}, u={self.u})"

        return representation


class ZeroOrderHold(FeedForward):
    """
    Hold input u_k constant for a t ∊ [t_k, t_k+1)

    Use a given input vector with corresponding time vector to control system at given point in time. For t>t_N, u_n
    will be returned.
    """

    def __init__(self, u: np.ndarray | None, t: np.ndarray | None) -> None:
        """
        Parameters
        ----------
        u : np.ndarray
            input vector every u_k corresponds to [t_k, t_k+1)  [l/h]
        t : np.ndarray
            time-vector corresponding to u (strictly monotonous)[h]
        """

        super().__init__(u, t)

        self.name = "zero_order_hold"

        # check for time-vector monotony
        if t is not None and np.any(0 >= (np.roll(t, -1) - t)[0:-1]):
            my_logger.exception("time vector is not strictly monotonous")
            raise ValueError("Time vector must be strictly monotonous")

        # check for input time correspondence
        if u is not None and len(u) != len(t):
            my_logger.exception("length of u is not the same as length of t... "
                                "every input must have a corresponding time")
            raise ValueError("Length of t is not the same as u.")

        self.u = u
        self.t = t

        my_logger.debug(f"setting controller to {ZeroOrderHold.__name__} with t: {self.t} and u: {self.u}")

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Get the zero order hold output value for u(t) with provided t (x is ignored). u(t) = u_k | t ∊ [t_k, t_k+1)

        Parameters
        ----------
        x : np.ndarray
            State vector (not used)
        t : float
            Time for output

        Returns
        -------
        np.ndarray
            Input u(t) at time t
        """
        t_in, u_in = self.t, self.u

        if self.u is None or self.t is None:
            missing_data = 'u' if self.u is None else 't'
            missing_data = "u and t" if self.u is None and self.t is None else missing_data
            my_logger.exception(f"{missing_data} is not given, aborting")
            raise AttributeError(f"{missing_data} is None, give value")

        if t < t_in[0]:
            u_ret = u_in[0]
        elif t >= t_in[-1]:
            u_ret = u_in[-1]
        else:
            idx = np.searchsorted(t_in, t, side="right") - 1
            u_ret = u_in[idx, :]

        # get index of deactivated pumps
        zero_ind = u_ret == 0

        # add noise
        u_ret = u_ret if self.noise_generator is None else self.noise_generator(u_ret)

        # set negative and deactivated pumps to zero
        u_ret[zero_ind] = 0
        u_ret[u_ret < 0] = 0

        return u_ret

    # === helper functions ===

    def interpolate_u(self, num=1e3) -> np.ndarray:
        """
        Interpolate u according to zero order rule. This is used to get a better resolution for plotting the input data
        after a optimization.

        Parameters
        ----------
        num : int
            number of points in new time-grid

        Returns
        -------
        np.ndarray
            interpolated u
        """
        my_logger.debug(f"Interpolating input u to {num} values according to zero order hold")

        # create interpolation function
        u_interpolation = interp1d(self.t, self.u, axis=0, kind='previous')

        # create new time vector
        t = np.linspace(0, self.t[-1], num=int(num))

        return u_interpolation(t), t

    # === miscellaneous ===

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': model controller name
            - 'u' or 0: tuple of time and corresponding input (t, u)

        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        if item == "u" or 0:
            return self.t, self.u
        else:
            raise IndexError("Invalid index")

    def __str__(self):
        description = f"ZeroOrderHold\n" \
                      f"  - length: {self.t.shape[0] if self.t is not None else None}"

        return description

    def __repr__(self):
        description = f"ZeroOrderHold(u={self.u}, t={self.t})"

        return description


class FirstOrderHold(FeedForward):
    """
    Linear interpolate the values u_k and u_k+1 for a t ∊ [t_k, t_k+1).

    Will return u_N for t>=t_N
    """

    def __init__(self, u: np.ndarray | None, t: np.ndarray | None) -> None:
        """
        Parameters
        ----------
        u : np.ndarray
            input vector every u_k corresponds to [t_k, t_k+1)  [l/h]
        t : np.ndarray
            time-vector corresponding to u (strictly monotonous)[h]
        """

        super().__init__(u, t)

        self.name = "first_order_hold"

        # check for time-vector monotony
        if t is not None and np.any(0 >= (np.roll(t, -1) - t)[0:-1]):
            my_logger.exception("time vector is not strictly monotonous")
            raise ValueError("Time vector must be strictly monotonous")

        # check for input time correspondence
        if t is not None and len(u) != len(t):
            my_logger.exception(
                "length of u is not the same as length of t... every input must have a corresponding time")
            raise ValueError("Length of t is not the same as u.")

        self.u = u
        self.t = t

        my_logger.debug(f"setting controller to first order hold with t: {self.t} and u: {self.u}")

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Get First-order hold output at time t. x will be ignored

        Parameters
        ----------
        x : np.ndarray
            State vector (not used)
        t : np.ndarray
            Time for output

        Returns
        -------
        np.ndarray
            Model input at time t
        """
        t_in, u_in = self.t, self.u

        if t < t_in[0]:
            u_ret = u_in[0]

        elif t >= t_in[-1]:
            u_ret = u_in[-1]
        else:
            idx = np.searchsorted(t_in, t, side="right") - 1

            u_k, u_k_plus = u_in[idx, :], u_in[idx + 1, :]
            t_k, t_k_plus = t_in[idx], t_in[idx + 1]
            lin_slope = (u_k_plus - u_k) / (t_k_plus - t_k)

            u_ret = u_k + lin_slope * (t - t_k)

            if np.any(u_ret < 0):
                pass

        # get index of deactivated pumps
        zero_ind = u_ret == 0

        # add noise
        u_ret = u_ret if self.noise_generator is None else self.noise_generator(u_ret)

        # set negative and deactivated pumps to zero
        u_ret[zero_ind] = 0
        u_ret[u_ret < 0] = 0

        return u_ret

    # === helper functions ===

    def interpolate_u(self, num=1e3) -> np.ndarray:
        """
        Interpolate u according to zero order rule

        Mostly used to get a finer grid for plotting. Will not affect the output of the function

        Parameters
        ----------
        num : int
            number of points in new time-grid

        Returns
        -------
        np.ndarray
            interpolated u
        """
        my_logger.debug(f"Interpolating input u to {num} values according to first order hold")

        # create interpolation function
        u_interpolation = interp1d(self.t, self.u, axis=0, kind='linear')

        # creating new time vector
        t = np.linspace(0, self.t[-1], num=int(num))

        u = u_interpolation(t)
        zero_u_ind = u == 0
        u = u if self.noise_generator is None else self.noise_generator(u)

        u[zero_u_ind] = 0
        u[u < 0] = 0

        return u, t

    # === miscellaneous ===

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': model controller name
            - 'u' or 0: tuple of time and corresponding input (t, u)

        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        if item == "u" or 0:
            return self.t, self.u
        else:
            raise IndexError("Invalid index")

    def __str__(self):
        description = f"Provide ZeroOrderHold of given vector\n" \
                      f"  - length: {self.t.shape[0] if self.t is not None else None}"

        return description

    def __repr__(self):
        description = f"FirstOrderHold(u={self.u}, t={self.t})"

        return description


class CubicSplines(FeedForward):

    def __init__(self, u: np.ndarray | None, t: np.ndarray | None):
        """
        Parameters
        ----------
        u : np.ndarray
            Out flow at knot-points. Can be given None for initialisation if input will often be changed via
            self.update_u()
        t : np.ndarray
            Time values for corresponding knot-points, can also be left blank
        """
        super().__init__(u, t)

        self.name = "cubic_splines"

        # functino used for interpolation
        self.interp_splines = None

        if u is not None and t is not None:
            self.update_u(u, t)

    def __call__(self, x: np.ndarray, t: float) -> float:
        """
                Get output according to quadratic spline interpolation

                Parameters
                ----------
                x : np.ndarray
                    Current state (not used)
                t : np.ndarray
                    Current time

                Returns
                -------
                float
                    Output u(t)
                """
        # get output
        u_ret = self.interp_splines(t)

        # get index of deactivated pumps
        zero_ind = u_ret == 0

        # add noise
        u_ret = u_ret if self.noise_generator is None else self.noise_generator(u_ret)

        # set negative and deactivated pumps to zero
        u_ret[zero_ind] = 0
        u_ret[u_ret < 0] = 0

        return u_ret

    def update_u(self, u: np.ndarray, t: np.ndarray) -> None:
        """
                Create the interpolation function with new input data.

                Parameters
                ----------
                u : np.ndarray
                    New input values
                t : np.ndarray
                    New corresponding time steps
                Returns
                -------
                None
                """
        if not any(t < 0):
            self.interp_splines = PchipInterpolator(t, u)
        else:
            t = np.linspace(0, 1)
            self.interp_splines = PchipInterpolator(t, np.ones((t.shape[0], 4)) * u[0])

        # used for plot-interpolation
        self.u = u
        self.t = t

    def interpolate_u(self, num=1e3) -> np.ndarray:
        my_logger.debug(f"Interpolating input u to {num} values according to first order hold")

        # creating new time vector
        t = np.linspace(0, self.t[-1], num=int(num))

        interp_val = self.interp_splines(t)
        interp_val[interp_val < 0] = 0

        return interp_val, t

    # === miscellaneous ===

    def __str__(self):
        description = f"Cubic spline hold\n" \
                      f"  - length: {self.t.shape[0] if self.t is not None else None}"

        return description

    def __repr__(self):
        representation = f"CubicSplines(t={self.t}, u={self.u})"

        return representation


if __name__ == '__main__':
    example_controller = ZeroOrderHold(np.linspace(0, 5, num=5), np.linspace(0, 5, num=5))
    print(repr(example_controller))
