import numpy as np

from controllers import Controller
from utils.logger import CustomLogger

my_logger = CustomLogger()


class ChemoStat(Controller):
    """ 
    Return a set output FRs0 for ∀ t. 
    
    Used to run a plant in chemostat mode.
    """

    def __init__(self, FRs0: np.ndarray) -> None:
        """
        Parameters
        ----------
        FRs0 : np.ndarray
            Constant in-/outflow into the system
        """

        super().__init__()

        self.name = 'chemostat'

        self.FRs0 = FRs0

        my_logger.debug(f"  - name: {self.name}"
                        f"  - Frs0: {self.FRs0}")

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Return a constant input FRs0 ∀ t to operate system in stationary point. All inputs will be ignored

        System must be in stationary state to avoid transient behaviour.

        Parameters
        ----------
        x : np.ndarray
            Current state of the system, not used
        t : np.ndarray
            Current time, not used

        Returns
        -------
        np.ndarray
            Constant system input for any time t. Out-flow will be recalculated if out-flow does not match in-flow
        """

        # Calculate feed
        u = self.FRs0

        if np.sum(u[:-1]) != u[-1]:
            u_out = np.sum(u[:-1])
            my_logger.info(f"no flow equilibrium. Changes out flow to {u_out}")

        return u

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': model controller name
            - 'FR': input that is held constant

        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        if item == "FR" or item == 0:
            return self.FRs0
        else:
            raise IndexError("Invalid index. Choose 'FR' (0) for flow")

    def __str__(self):
        description = f"Chemostat controller:\n" \
                      f"  - u_in: {self.FRs0} L/h"

        return description

    def __repr__(self):
        description = f"ChemoStat(FRs0={self.FRs0})"

        return description


class Batch(Controller):
    """
    Set input to 0 ∀ t.

    Used to run plant in Batch-mode
    """

    def __init__(self, num_in=4) -> None:
        """
        Parameters
        ----------
        num_in : int
            Optional length of zero vector to return. Default is 4
        """

        super().__init__()

        self.name = "batch"

        self.num_in = num_in

        my_logger.debug(f"name: {self.name}")

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Return a zero-vector to operate system in batch process ∀ t. All inputs will be ignored.

        Parameters
        ----------
        x : np.ndarray
            Current state of the system, not used
        t : np.ndarray
            Current time, not used

        Returns
        -------
        np.ndarray
            Zero vector to operate system in batch process
        """
        return np.zeros(self.num_in)

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': model controller name

        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        else:
            my_logger.exception("Invalid index. Choose 'name' for name")
            raise IndexError("Invalid index. Choose 'name' for name")

    def __str__(self):
        description = f"Batch controller:\n" \
                      f"  - u = {[0]*self.num_in}"

        return description

    def __repr__(self):
        description = f"Batch(num_in={self.num_in})"

        return description


class Step(Controller):
    """
    Return a step FRs at set time t.

    Used to simulate a step response for the system
    """

    def __init__(self, t_step: float, FRs: np.ndarray, num_in=4) -> None:
        """
        Parameters
        ----------
        t_step : float
            time of step occurrence [h]
        FRs : np.ndarray
            height of steps         [l/h]
        num_in : int
            Optional, size of u-vector. Default is 4
        """

        super().__init__()

        self.name = "step_response"

        self.t_step = t_step
        self.FRs = FRs

        self.num_in = num_in

        my_logger.debug(f"name: {self.name}, step time: {self.t_step} h,  step height: {self.FRs} l/h")

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Perform a step of height FRs if t >= t_step

        Parameters
        ----------
        x : np.ndarray
           Current state of the system, not used
        t : float
            Current time, step will be activated if t >= t_step

        Returns
        -------
        np.ndarray
            Input either provided by FR if t=>t_step or 0 otherwise
        """
        u = self.FRs if t >= self.t_step else np.zeros(4)

        return u

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': model controller name
            - 't' or 0: step time
            - 'FR' or 1: step height
            - 'num_in' or 2: size of u-vector
        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        if item == "t" or item == 0:
            return self.t_step
        if item == "FR" or item == 1:
            return self.FRs
        if item == "num_in" or item == 2:
            return self.num_in
        else:
            raise IndexError("Invalid index. Choose 't' (0) for step time, 'FR' (1) for step height or "
                             "'num_in' (0) for u-size")

    def __str__(self):
        description = f"Step:\n" \
                      f"  - t_step: {self.t_step} h\n" \
                      f"  - u_step: {self.FRs} L/h\n" \
                      f"  - num_in: {self.num_in}"

        return description

    def __repr__(self):
        description = f"Step(t_step={self.t_step}, u_step={self.FRs}, num_in={self.num_in})"

        return description


class Impulse(Controller):
    """
    Return an impulse of height FR_imp and of length delta_imp at time t_imp

    Used to apply an impulse to a plant
    """

    def __init__(self, t_imp: float, delta_imp: float, FR_imp: np.ndarray) -> None:
        """
        Parameters
        ----------
        t_imp : float
            Time of impulse     [h]
        delta_imp : float
            duration of impulse  [h]
        FR_imp : np.ndarray
            height of impulse    [l/h]
        """

        super().__init__()

        self.name = "impulse_response"

        self.t_imp = t_imp
        self.delta_imp = delta_imp
        self.FR_imp = FR_imp

        my_logger.debug(f"  - name: {self.name}"
                        f"  - delta imp: {self.delta_imp} h"
                        f"  - height imp: {self.FR_imp} l/h")

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Implementing an impulse by returning an output FR_imp if t_imp =< t =< (t_imp + T0)

        Parameters
        ----------
        x : np.ndarray
            Current state of the system, not used
        t : float
            Current time of the system

        Returns
        -------
        np.ndarray
           Give in- and out-flow if t_imp=<t<t_imp+delta_imp
        """
        # Calculate feed
        u = np.zeros(4)
        if self.t_imp < t <= self.t_imp + self.delta_imp:
            u[0] = self.FR_imp

        # Calculate out flow (there is none)
        u = np.hstack((u, 0))

        return u

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': model controller name
            - 't' or 0: impulse start           [h]
            - 'delta_imp' or 1: impulse width   [h]
            - 'FR' or 2: impulse height         [g/l]
        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        elif item == "t" or item == 0:
            return self.t_imp
        elif item == "delta_imp" or item == 1:
            return self.delta_imp
        elif item == "FR" or item == 2:
            return self.FR_imp
        else:
            raise IndexError("Invalid index. Choose 't' (0) for impulse time, "
                             "'T0' (1) for impuls width or 'FR' (2) for impulse height")

    def __str__(self):
        description = f"Impulse response:\n" \
                      f"  - t_imp: {self.t_imp} h\n" \
                      f"  - t_delta: {self.delta_imp} h\n" \
                      f"  - u_imp: {self.FR_imp} L/h\n"

        return description

    def __repr__(self):
        description = f"Impulse(t_imp={self.t_imp}, t_delta={self.delta_imp}, u_imp={self.FR_imp})"

        return description


class Ramp(Controller):
    """
    Return a linear rising signal of slope FRs0 if t => t_ramp

    Used to simulate a ramp response
    """

    def __init__(self, t_ramp: float, FRs0: np.ndarray) -> None:
        """
        Parameters
        ----------
        t_ramp : float
            Start time of ramp  [h]
        FRs0 : np.ndarray
            Gradient of in flow [l/h²]
        """

        super().__init__()

        self.name = "ramp_response"

        self.t_ramp = t_ramp
        self.FRs0 = FRs0

        my_logger.debug(f"name: {self.name}, t_ramp: {self.t_ramp} h, FRs: {self.FRs0} l/h²")

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Return an input that increases linearly with slope FRs0 if t >= t_ramp

        Parameters
        ----------
        x : np.ndarray
           Current state of the system, not used
        t : float
            Current time of the System, ramp starts when t<=t_ramp
        Returns
        -------
        np.ndarray
            Output that starts at zero and increases linearly over time with given slope
        """
        # Calculate feed
        u = np.zeros(4)
        if t >= self.t_ramp:
            u = self.FRs0 * (t - self.t_ramp)

        return u

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item: str|int
            - 'name': model controller name
            - 't' or 0: ramp start [h]
            - 'FRs0' or 1: input gradient [l/h²]
        Returns
        -------
        None
        """
        if item == "name":
            return self.name
        elif item == "t" or item == 0:
            return self.t_ramp
        elif item == "FRs0" or item == 1:
            return self.FRs0
        else:
            raise IndexError("Invalid index. Choose 't' (0) for start time or 'FR' (1) for ramp slope")

    def __str__(self):
        description = f"Impulse response:\n" \
                      f"  - t_ramp: {self.t_ramp} h\n" \
                      f"  - delta_u: {self.FRs0} L/h²"

        return description

    def __repr__(self):
        description = f"Ramp(t_ramp={self.t_ramp}, delta_u={self.FRs0})"

        return description
