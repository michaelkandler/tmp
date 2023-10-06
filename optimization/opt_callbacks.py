"""
Callback functions to be used by the minimizer in use.
"""
import os
from time import time

import numpy as np
import matplotlib.pyplot as plt

from utils.logger import CustomLogger

my_logger = CustomLogger()


class LogSteps(object):
    """
    Log z_i for every iteration step i.

    Will only work with scipy functions

    Log
    """

    def __init__(self):
        self.u = []
        self.f = []

    def __call__(self, inter_steps, sol):
        self.f.append(sol.fun)
        self.u.append(inter_steps)

    def get_u(self, clear_u=False) -> None:
        """
        Access saved inputs

        Parameters
        ----------
        clear_u : Bool
            clear internally saved u is changed to True

        Returns
        -------
        None
        """
        u_arr = np.asarray(self.u)
        if clear_u:
            self._clear_u()

        return u_arr

    def get_f(self, clear_f=False) -> None:
        """
        Access saved error function values

        Parameters
        ----------
        clear_f : Bool
            clear internally saved f is changed to True

        Returns
        -------
        None
        """
        f_arr = np.asarray(self.f)
        if clear_f:
            self._clear_f()

        return f_arr

    def plot_u(self) -> None:
        """
        Some sort of u plotting over time will be developed in the future
        Returns
        -------
        None
        """
        pass

    def plot_f(self, title=None, path=None) -> None:
        """
        Plot the development of the cost function

        Parameters
        ----------
        title : str
            optional title to save figure timestamp is used if not provided
        path : str
            optional path so save figure instead of showing it

        Returns
        -------
        None
        """

        # get data
        f = self.get_f()

        plt.figure()
        plt.plot(f)

        # fancy stuff
        plt.title("Cost function")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")

        # fancy stuff
        plt.grid(True)

        if path is not None:
            # get name and path
            path = os.path.abspath(path)
            title = time() if title is None else title
            title = f"{title}.png"
            full_path = os.path.join(path, title)

            # save it!
            plt.savefig(full_path)
        else:
            plt.show()

    def _clear_u(self) -> None:
        """
        Clear internally saved u

        Returns
        -------
        None
        """
        self.u = []

    def _clear_f(self) -> None:
        """
        Clear internally saved f

        Returns
        -------
        None
        """
        self.f = []

    def __str__(self):
        description = "optimization_logger"

        return description


class MinimizeStopper(object):
    """
    Class that implements a time callback-function to raise warning/exception if too much time has passed
    """

    def __init__(self, max_sec):
        """
        Parameters
        ----------
        max_sec : int
            maximum seconds before exception is raised
        """
        self.max_sec = max_sec
        self.start = time.time()

    def __call__(self):
        """
        Raises exception/warning if max_sec have passed since instantiation

        Returns
        -------
        None
        """
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            my_logger.warning("terminating optimization: time limit reached")
            raise TookTooLong("terminating optimization: time limit reached")


#-------------------------------- Exceptions --------------------------------#

class TookTooLong(Exception):
    """
    Custom warning class to indicate that too much time has passed. Missing something?
    """
    pass
