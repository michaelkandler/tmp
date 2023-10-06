"""
Utilities to postprocess data
"""
import os.path

import numpy as np
import pickle

from typing import List

from processes import Process, StationaryPoint
from utils.logger import CustomLogger

my_logger = CustomLogger()


# === data recalculation ===

def calculate_yields(process: Process | str, dead_time=0) -> tuple:
    """
    Calculate the yield and space-time yield from a given process

    Parameters
    ----------
    process : Process
        Process to calculate yields for, can also be given as path to pickled process

    Returns
    -------
    tuple
        (yield, space-time-yield)
    """
    # unpickle process if necessary
    process = process if isinstance(process, Process) else unpickle_process(process)

    # get run data
    x = process.x
    T = process.t[-1] + dead_time if not isinstance(process, StationaryPoint) else process.tau

    # unpack states
    I, N, A, _, _ = x[-1, :]

    # calculate yields
    if isinstance(process, StationaryPoint):
        S = process.S_in
        y = N / (I + N + A + S)
    else:
        y = N / (I + N + A)
    sty = N / T

    return y, sty


def calculate_cost():
    pass


def calculate_price():
    pass


def calculate_dev_zeros(process: Process | str, axis_ind=(0, 1, 2, 3), considered_zero=2e-3) -> tuple:
    """
    Approximate the zeros of the first and second derivative of the process states.

    Parameters
    ----------
    process : str | Process
        Process or path to pickled process

    axis_ind : list
        states to consider given as their corresponding indexes

    considered_zero : float
        threshold for value to be considered zero
    Returns
    -------
    np.ndarray
        Array with same shape as state-array but all none-zero values are replaced with None.
    """
    # unpickle process if necessary
    process = process if isinstance(process, Process) else unpickle_process(process)

    # unpack data
    x, t = process.x, process.t

    # calculate first and second derivative
    first_dev = np.gradient(x, t, axis=0)
    second_dev = np.gradient(first_dev, t, axis=0)

    # get zeros
    first_zeros = abs(first_dev) < considered_zero
    second_zeros = abs(second_dev) < considered_zero

    # exclude unwanted axis
    mask = np.zeros(x.shape, dtype=bool)
    mask[:, list(axis_ind)] = True
    first_zeros = first_zeros * mask
    second_zeros = second_zeros * mask

    x_first = np.full(x.shape, None, dtype=object)
    x_second = np.full(x.shape, None, dtype=object)

    # remove unwanted axis
    x_first[first_zeros] = x[first_zeros]
    x_second[second_zeros] = x[second_zeros]

    return (t, x_first), (t, x_second)


# === data analysis ===

def check_restraints(lims: tuple, u: np.ndarray) -> bool:
    """
    Function to check if constraints for a given input u and pump limits are kept
    
    Parameters
    ----------
    lims : tuple
        pump limits in shape [[lower1, upper1]...] 
    u : np.ndarray 
        plant input with shape (t, num_lims)

    Returns
    -------
    bool 
        true if constraints are kept else false
    """
    # check if upper limits are kept
    upper_lims = np.asarray([l[1] if l is not None else 0 for l in lims])
    upper_const = np.all(np.less_equal(u, upper_lims))

    # check if lower limits are kept
    lower_lims = np.asarray([l[0] if l is not None else 0 for l in lims])
    lower_const = np.all(np.greater_equal(u, lower_lims))

    return lower_const and upper_const


# === data manipulation ===

def remove_outliers(data: np.ndarray, mask=None, limits=(0, 1)) -> tuple:
    """
    Remove outliers from a dataset and attach mask of removal

    All data that is not in a given bound ((0,1) by default) will be removed. If mask is given all values flagged False
    will be used.

    Parameters
    ----------
    data : np.ndarray
        data du remove outliers from
    mask : np.ndarray | None
        optional mask for outlier removal, outliers will be calculated from limits if not given otherwise
    limits : tuple
        upper and lower limits for outliers. (0 ,1) if not given other

    Returns
    -------
    (np.ndarray, np.ndarray)
        dataset with removed outliers, removal mask with the same shape as the input data and Boolean values according
        to removal (False means data has been removed)
    """
    if limits[0] >= limits[1]:
        my_logger.error(f"lower limit ({limits[0]}) is lager than upper limit ({limits[1]})")
        raise ValueError(f"Lower limit for outlier must be lager than the bigger one. {limits[0]}>{limits[1]}")

    if mask is not None and mask.shape != data.shape:
        raise ValueError("mask must be same shape as data")

    # if mask is not given calculate it
    if mask is None:
        my_logger.debug("outliers are calculated...")
        # get indices of outliers
        mask_non_zero = np.where(data < limits[0])[-1]
        mask_non_high = np.where(data > limits[1])[-1]

        # merge locations
        mask = np.union1d(mask_non_high, mask_non_zero)
        my_logger.info(f"{len(mask)} outliers removed")

    # delete outliers and return data
    data_no_outliers = np.delete(data, mask, axis=-1)
    return data_no_outliers, mask


def sort_and_rearrange(arr1: np.ndarray, arr2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort the first array and rearrange the second array in the same order using NumPy.

    Parameters
    ----------
    arr1 : np.ndarray
        The first input array to be sorted.

    arr2 : np.ndarray
        The second input array to be rearranged according to the sorted order of arr1.

    Returns
    -------
    sorted_arr1 : np.ndarray
        A new sorted NumPy array obtained from sorting arr1.

    sorted_arr2 : np.ndarray
        A new NumPy array obtained by rearranging arr2 to match the sorted order of arr1.
    """
    # sort the first array and get the sorted indices
    sorted_indices = np.argsort(arr1)
    sorted_arr1 = np.sort(arr1)

    # rearrange the second array using the sorted indices
    sorted_arr2 = arr2[sorted_indices]

    return sorted_arr1, sorted_arr2


# === data store and load ===

def unpickle_process(pickel_path: str) -> Process:
    """
    Unpickle a pickled process.

    Parameters
    ----------
    pickel_path : str
        Path to pickled process

    Returns
    -------
    Any or None
        The unpickled process if successful, or None if an error occurs.
    """
    try:
        with open(pickel_path, 'rb') as file:
            process = pickle.load(file)
        return process
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{pickel_path}' not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the pickle object: {e}")


def pickle_process(process: Process, path: str) -> None:
    """
    Pickle a process.

    Parameters
    ----------
    process : Process
        The process to be pickled.
    path : str
        folder to save file to

    Returns
    -------
    None
    """
    name = f"{process.name}.pickle"
    full_path = os.path.join(os.path.abspath(path), name)

    try:
        with open(full_path, 'wb') as pickle_file:
            pickle.dump(process, pickle_file)
    except Exception as e:
        raise Exception(f"An error occurred while pickling the object: {e}")


def unpickle_dir(folder_path: str) -> List[Process]:
    """
    Unpickle all pickle files from a directory and return a list of projects.

    Parameters
    ----------
    folder_path : str
        path to folder with pickle files

    Returns
    -------
    list
        list of unpicled processes
    """
    return_list = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        raise FileNotFoundError

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter the files that end with ".pickle"
    pickle_files = [file for file in files if file.endswith(".pickle")]

    # Print the matching filenames
    if pickle_files:
        for pickle_file in pickle_files:
            unp_prc = unpickle_process(os.path.join(folder_path, pickle_file))
            return_list.append(unp_prc)
    else:
        print("No pickled files found in the folder.")
    
    return return_list


if __name__ == '__main__':
    res = unpickle_dir(
        "/data/stat_opt2")

    pass
