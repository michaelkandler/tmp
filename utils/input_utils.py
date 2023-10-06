"""
Utilities to prepare any kind of input

For example creating an inflow vector or preparing a constant z_star₀
"""

import numpy as np
from typing import List
import os

import utils


def phases2input_vec(t: np.ndarray, t_samp: float | None, Frs: List[float], phases: List[List[float]]) -> np.ndarray:
    """
    Get a vector of input flow from constant phase of inflow

    Parameters
    ----------
    t : np.ndarray
        Process time or time vector of input
    t_samp : float, optional
        Sampling time, will only be used if time is not given as vector
    Frs : List[float]
        Concentration of inflow
    phases : List[List[float]]
        Time phases at which inflow takes place, index of list matches index of concentrations

    Returns
    -------
    np.ndarray
        Input u as m*time_steps dimensional vector
    """

    # Define constants
    num_feeds = len(Frs)
    if num_feeds != 4:
        raise ValueError("We use a four feed model. Please provide three inputs and one bleed")

    # Create time vector if not given
    if isinstance(t, np.ndarray):
        if t_samp is None:
            raise ValueError("Provide a time vector or sampling time and horizon")
    else:
        if t_samp is None:
            raise ValueError("Provide a time vector or sampling time and horizon")
        # Get time vector
        t_samp = t_samp / 60 / 60
        t = np.arange(0, t + t_samp, t_samp)

    # Instantiate output
    u = np.zeros((t.shape[0], num_feeds))

    for i, F, phase in zip(range(num_feeds), Frs, phases):
        ind_min = np.argmin(np.abs(t - phase[0]))
        ind_max = np.argmin(np.abs(t - phase[-1]))

        u[ind_min: ind_max + 1, i] = F

    return u


def add_phases2xls(path: str, Frs: List[float], phases: List[List[float]], overwrite_file=False, sheet_index=None) -> None:
    """
    Load data from Excel sheet, and write vector input for given phases.

    Parameters
    ----------
    path : str
        Path to load Excel from
    Frs : List[float]
        Concentration of inflow
    phases : List[List[float]]
        Time phases of inflow
    overwrite_file : bool, optional
        Add '_new' to new file name
    sheet_index : int, optional
        Sheet index to add inputs to, first sheet is taken if not given
    """

    # Load dataframe
    pandas_frame = utils.xls2pd(path)

    # Check sheet choice
    if len(pandas_frame[2]) > 1 and sheet_index is not None:
        print("More than one datasheet in file, the first one is taken. Set sheet_index to use another one ...")
    sheet_index = sheet_index if sheet_index is not None else 0
    data_frame = pandas_frame[2][sheet_index]

    # Calculate input vector
    t = data_frame.to_numpy()[:, 0]
    u = phases2input_vec(t, None, Frs, phases)

    # Write input to dataframe
    for u_ind in range(u.shape[1] - 1):
        data_frame[f'u_{u_ind} [L/h]'] = u[:, u_ind]
    data_frame['u_out [L/h]'] = u[:, -1]

    # Replace sheet
    pandas_frame[2][sheet_index] = data_frame

    # Save file under new name if requested
    old_name = os.path.basename(path)
    new_name = f"{old_name.split('.')[0]}_new" if not overwrite_file else old_name.split('.')[0]
    utils.pd2xls(pandas_frame[0], pandas_frame[1], pandas_frame[2], pandas_frame[3], new_name, os.path.dirname(path))


def create_constant_z0(z0: np.ndarray, n_steps: int, lims: List[float], T_start: float | None = None, zero_order=True) -> np.ndarray:
    """
    Create a constant initial input guess from numpy array

    Parameters
    ----------
    z0 : np.ndarray
        Initial input guess to be held constant
    n_steps : int
        Number of discrimination steps (length of vector)
    lims : List[float]
        Pump limits, unused pumps have limit of None
    T_start : float, optional
        Initial guess for process time

    Returns
    -------
    np.ndarray
        Initial guess vector
    """

    # Get number of inputs and list of used inputs
    ind_nn = tuple(i for i in range(len(lims)) if lims[i] is not None)
    n_inp = sum([1 for _ in lims])

    n_steps = n_steps - 1 if zero_order else n_steps

    # Construct initial constant input
    z_start = np.tile(np.asarray(z0), n_steps)
    z_start = z_start.reshape(-1, n_inp)
    z_start = z_start[:, ind_nn]
    start_point = z_start.flatten()
    start_point = np.append(start_point, T_start) if T_start is not None else start_point

    return start_point


if __name__ == '__main__':
    pass
