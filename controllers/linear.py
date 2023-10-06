import numpy as np

from controller import Controller


class P(Controller):
    def __init__(self, k_p: float, r: np.ndarray):
        """
        Implements a simple proportional controller with given constant k_p and setpoint r

        Parameters
        ----------
        k_p : float
            proportional part
        r : np.ndarray
            set point
        """
        super().__init__()

        self.name = 'p_controller'

        # controller constant
        self.k_p = k_p

        # setpoint
        self.r = r

        print(f"  - name: {self.name}\n"
              f"  - k_p: {self.k_p}")

    def __call__(self, y: np.ndarray, t: float):
        """
        Return control output proportional to set point error

        Parameters
        ----------
        y : np.ndarray
            Output of plant
        t : float
            Current time

        Returns
        -------
        np.ndarray
            Input for plant to p-regulation
        """
        # calculate error
        e = self.r - y

        return e * self.k_p

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': Name of controller
            - 'k_p' or 0: Kp-constant of controller

        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        elif item == "k_p" or 0:
            return self.k_p
        else:
            raise IndexError("Invalid index")

    def __repr__(self):
        description = f"P-controller\n" \
                      f"-----------\n" \
                      f"  - k_p: {self.k_p}"

        return description


class PI(Controller):

    def __init__(self, k_p: float, k_i: float, r: np.ndarray, anti_windup=False):
        """
        Implements a simple proportional-integral controller with given constant k_p and setpoint r

        Parameters
        ----------
        k_p : float
            proportional part
        k_i : float
            integral part
        r : np.ndarray
            set point
        anti_windup : boo
            enable anti-windup
        """
        super().__init__()

        self.name = "pi_controller"

        # controller constants
        self.k_p = k_p
        self.k_i = k_i

        # setpoint
        self.r = r

        # integrated error
        self.e_int = 0

        self.anti_windup = anti_windup

        print(f"  - name: {self.name}\n"
              f"  - k_p: {self.k_p}\n"
              f"  - k_i: {self.k_i}\n")

    def __call__(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Return control output proportional to set point error and integral of it

        Parameters
        ----------
        y : np.ndarray
            Output of plant
        t : float
            Current time

        Returns
        -------
        np.ndarray
            Input for plant to pi-regulation
        """
        # callable error
        e = self.r - y

        # numeric integration
        self.e_int += e

        # MISSING anti-windup

        return e * self.k_p + self.e_int * self.k_i

    def reset_integral(self):
        """
        Reset the integral part to zero

        Returns
        -------
        None
        """
        self.e_int = 0

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': Name of controller
            - 'k_p' or 0: Kp-constant of controller
            - 'k_i' or 1: Ki-constant of controller

        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        elif item == "k_p" or 0:
            return self.k_p
        elif item == "k_i" or 1:
            return self.k_i
        else:
            raise IndexError("Invalid index")

    def __repr__(self):
        description = f"Abstract controller\n" \
                      f"-----------\n"

        return description


class PID(Controller):
    def __init__(self, k_p: float, k_i: float, k_d: float, r: np.ndarray, t_0=0, anti_windup=False):
        """
        Implements a simple PID controller with given constant k_p, k_i, k_p and setpoint r

        Parameters
        ----------
        k_p : float
            proportional part
        k_i : float
            integral part
        k_d : float
            differential part
        r : np.ndarray
            set point
        anti_windup : boo
            enable anti-windup
        """
        super().__init__()

        self.name = "pid_controller"

        # controller constants
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        # setpoint
        self.r = r

        self.anti_windup = anti_windup

        # integral of error
        self.e_int = 0

        # previous error and time
        self.e_min = 0
        self.t_min = t_0
        print(f"  - name: {self.name}\n"
              f"  - k_p: {self.k_p}\n"
              f"  - k_d: {self.k_d}\n"
              f"  - k_i: {self.k_i}\n")

    def __call__(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Return control output proportional to set point error, differential and integral of it

        Parameters
        ----------
        y : np.ndarray
            Output of plant
        t : float
            Current time

        Returns
        -------
        np.ndarray
            Input for plant to pid-regulation
        """
        # calculate error
        e = self.r - y

        # numeric integration
        self.e_int += e

        # numeric differentiation
        delta_e = e - self.e_min
        delta_t = t - self.t_min
        e_dot = delta_e / delta_t

        # MISSING anti-windup

        return e * self.k_p + self.e_int * self.k_i + e_dot * self.k_d

    def reset_integral(self):
        """
        Reset the integral part to zero

        Returns
        -------
        None
        """
        self.e_int = 0

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': Name of controller
            - 'k_p' or 0: Kp-constant of controller
            - 'k_i' or 1: Ki-constant of controller
            - 'k_d' or 2: Kd-constant of controller

        Returns
        -------
        None
        """

        if item == "name":
            return self.name
        elif item == "k_p" or 0:
            return self.k_p
        elif item == "k_i" or 1:
            return self.k_i
        elif item == "k_d" or 2:
            return self.k_d
        else:
            raise IndexError("Invalid index")

    def __repr__(self):
        description = f"Abstract controller\n" \
                      f"-----------\n"

        return description
