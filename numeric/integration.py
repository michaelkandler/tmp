import numpy as np

from scipy.integrate import odeint

from utils import CustomLogger

my_logger = CustomLogger()


class Integrator:
    """
    Abstract class that implements an integration ∫₀ᵀ f(t) dt evaluated at discrete-time-steps tₖ

    Will implement library or self implemented integrators
    """

    def __init__(self):
        self.name = "integrator"

    def __call__(self, func: callable, x_0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the integrated function 'func' (∫₀ᵀ f(t)) along the time-grid t, with a given initial value f(x) = x₀

        Parameters
        ----------
        func : Callable
            function to integrate
        x_0 : np.ndarray
            inital value for integration
        t : np.ndarray
            time-vector to evaluate integrated function along

        Returns
        -------
        np.ndarray
            array of size (len(t), len(x_0)) xₖ=x(tₖ)
        """
        raise NotImplemented

    def grad(self):
        """
        Gradient of integrated function
        Returns
        -------

        """
        raise NotImplemented

    def hess(self):
        """
        Hessian of integrated function
        Returns
        -------

        """
        raise NotImplemented

    def __str__(self):
        description = f"Abstract integrator"

        return description

    def __repr__(self):
        representation = "Integrator()"

        return representation


# === libraries ===

class SciPy(Integrator):

    def __init__(self):
        super().__init__()

        self.name = "ode_int"

    def __call__(self, func, x_0: np.ndarray, t: np.ndarray):
        return odeint(func, x_0, t)

    def __str__(self):
        description = f"scipy.integrate.odeint"

        return description

    def __repr__(self):
        return f"SciPY"


# ===  self written ===
class IntegratorFixedStepSize(Integrator):

    def __init__(self):
        super().__init__()
        self.name = "fixed_step_size"

        # integration time-step
        self.delta_t = None

    def set_delta_t(self, delta_t: float) -> None:
        """
        Set the time difference for integration

        Parameters
        ----------
        delta_t : float
            interval between time-steps

        Returns
        -------
        None
        """
        my_logger.debug(f"setting delta t for {self.__repr__()} to {delta_t}")
        self.delta_t = delta_t

    def __str__(self):
        description = f"Abstract integrator with fixed step size"

        return description

    def __repr__(self):
        representation = "IntegratorFixedStepSize()"

        return representation

class IntegratorAdaptiveStepSize(Integrator):
    pass

class Euler(IntegratorFixedStepSize):
    """
    Integrate an ODE-System using the euler-method with a fixed step size

    The Default step size is 5 min or 0,083334h.
    """
    def __init__(self):
        super().__init__()
        self.name = "euler"

        # integration time-step
        self.set_delta_t(5 / 60)

    def __call__(self, func: callable, x_0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Integrate an ODE system using the Euler method.

        Parameters
        ----------
        func : callable
            The function that defines the ODE system. It should take two arguments: t (time) and y (state vector).
        x0 : numpy.ndarray
            A numpy array containing the initial values of the state variables.
        t : np.ndarray
            Time grid to perform integration on

        Returns
        -------
        y_values : numpy.ndarray
            Numpy array of state vectors corresponding to each time value.
        """
        # get given time steps
        t_original = np.copy(t)
        t_plus = np.roll(t_original, -1)
        prov_deltas = (t_plus - t_original)[:-1]

        # check if any given time step is smaller than delta t
        if self.delta_t > np.min(prov_deltas):
            my_logger.info(f"delta t not smaller then provided grid, taking smallest time difference of t. "
                           f"changed {self.delta_t*60:.4f}min to {min(prov_deltas)*60:.4f}min")
            self.delta_t = np.min(prov_deltas)

        # create finer t-grid
        num_time_steps = int((t[-1] - t[0]) / self.delta_t)
        t = np.linspace(t[0], t[-1], num_time_steps)

        # instantiate solution vector
        num_dimensions = len(x_0)
        integrated_values = np.zeros((num_time_steps, num_dimensions))
        integrated_values[0] = x_0.copy()

        for i in range(1, num_time_steps):
            step_size = t[i] - t[i - 1]
            current_state = integrated_values[i - 1]
            integrated_values[i] = current_state + step_size * func(current_state, t[i - 1])

        # get t and u value closest to provided time axis
        absolute_diff_t = np.abs(t[:, None] - t_original)
        sorted_indices = np.argsort(absolute_diff_t, axis=0)
        closest_indices = sorted_indices[0, :]

        return integrated_values[closest_indices]

    def __str__(self):
        description = f"Euler integration"

        return description

    def __repr__(self):
        representation = f"Euler(delta_t={self.delta_t})"

        return representation


class Trapezoid(IntegratorFixedStepSize):
    """
    Integrate an ODE-System using the trapezoid rule with a fixed step size

    The Default step size is 5 min or 0,083334h.
    """

    def __init__(self):
        super().__init__()
        self.name = "trapezoid"

        # integration time-step
        self.set_delta_t(5 / 60)

    def __call__(self, func: callable, x_0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Perform numerical integration using the trapezoid rule.

        Parameters:
            func (callable):
                The n-dimensional function to be integrated.
            x_0 (numpy array):
                The starting value for the integration.
            t (numpy array):
                A 1D numpy array representing the time points to integrate over.

        Returns:
            numpy array: The approximate integrated values at each time point.
        """
        # get given time steps
        t_original = np.copy(t)
        t_plus = np.roll(t_original, -1)
        prov_deltas = (t_plus - t_original)[:-1]

        # check if any given time step is smaller than delta t
        if self.delta_t > np.min(prov_deltas):
            my_logger.info(f"delta t not smaller then provided grid, taking smallest time difference of t. "
                           f"changed {self.delta_t*60:.4f}min to {min(prov_deltas)*60:.4f}min")
            self.delta_t = np.min(prov_deltas)

        # create finer t-grid
        num_time_steps = int((t[-1] - t[0]) / self.delta_t) + 1
        t = np.linspace(t[0], t[-1], num_time_steps)

        # instantiate solution vector
        num_dimensions = len(x_0)
        integrated_values = np.zeros((num_time_steps, num_dimensions))
        integrated_values[0] = x_0.copy()

        # trapezoidal integration
        for i in range(1, num_time_steps):
            # get 𝚫t_i and prev integration
            dt = t[i] - t[i - 1]
            current_state = integrated_values[i - 1]

            # calculate the average of current and next state using the trapezoid rule
            k1 = func(current_state, t[i])
            k2 = func(current_state + dt * k1, t[i])
            avg_slope = 0.5 * (k1 + k2)

            # Update the current state using the average slope
            integrated_values[i] = current_state + dt * avg_slope

        # get t and u value closest to provided time axis
        absolute_diff_t = np.abs(t[:, None] - t_original)
        sorted_indices = np.argsort(absolute_diff_t, axis=0)
        closest_indices = sorted_indices[0, :]

        return integrated_values[closest_indices]

    def __str__(self):
        description = f"Trapezoidal integration"

        return description

    def __repr__(self):
        representation = f"Trapezoid(delta_t={self.delta_t})"

        return representation


class RungeKutta(IntegratorFixedStepSize):

    def __init__(self):
        super().__init__()

        self.order = 2

        self.name = f"runge_kutta_{self.order}"

        self.set_delta_t(5/60)

    def set_order(self, order: int) -> None:
        if order > 6:
            raise NotImplemented("Runge-Kutta is only implemented to order 6")
        self.order = order

    def __call__(self, func, x_0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Perform numerical integration using the Runge-Kutta method.

        Parameters:
        ----------
            func : callable
                The n-dimensional function to be integrated.
            x_0 : numpy array
                The starting value for the integration.
            t : numpy array
                A 1D numpy array representing the time points to integrate over.

        Returns:
        ----------
        numpy array
            The integrated values at each time point.
        """
        # get given time steps
        t_original = np.copy(t)
        t_plus = np.roll(t_original, -1)
        prov_deltas = (t_plus - t_original)[:-1]

        # check if any given time step is smaller than delta t
        if self.delta_t > np.min(prov_deltas):
            my_logger.info(f"delta t not smaller then provided grid, taking smallest time difference of t. "
                           f"changed {self.delta_t*60:.4f}min to {min(prov_deltas)*60:.4f}min")
            self.delta_t = np.min(prov_deltas)

        # create finer t-grid
        num_time_steps = int((t[-1] - t[0]) / self.delta_t)
        t = np.linspace(t[0], t[-1], num_time_steps)

        # instantiate solution vector
        num_dimensions = len(x_0)
        integrated_values = np.zeros((num_time_steps, num_dimensions))
        integrated_values[0] = x_0.copy()

        # Define the coefficients for different orders of Runge-Kutta
        if self.order == 1:
            a, b = [0], [1]
        elif self.order == 2:
            a, b = [0, 1], [0.5, 0.5]
        elif self.order == 3:
            a, b = [0, 1, 0], [1 / 6, 2 / 3, 1 / 6]
        elif self.order == 4:
            a, b = [0, 0.5, 0.5, 1], [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        elif self.order == 5:
            a, b = [0, 1 / 4, 3 / 8, 12 / 13, 1], [1 / 24, 0, 3 / 32, 9 / 32, 3 / 24, 1 / 14]
        elif self.order == 6:
            a, b = [0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 1], [1 / 120, 0, 2 / 60, 3 / 40, 2 / 60, 1 / 120, 1 / 90]
        else:
            raise ValueError(f"Invalid order. Use an integer between 1 and 6. Given was {self.order}")

        for i in range(1, num_time_steps):
            current_state = integrated_values[i - 1]
            k = np.zeros((self.order, num_dimensions))

            dt = t[i] - t[i-1]

            for j in range(self.order):
                k[j] = func(current_state + dt * sum(a[k] * b[j] for k in range(j)), t[i])

            integrated_values[i] = current_state + dt * sum(b[j] * k[j] for j in range(self.order))

        # get t and u value closest to provided time axis
        absolute_diff_t = np.abs(t[:, None] - t_original)
        sorted_indices = np.argsort(absolute_diff_t, axis=0)
        closest_indices = sorted_indices[0, :]

        return integrated_values[closest_indices]
