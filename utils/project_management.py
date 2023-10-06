"""
Functions to keep track of the project (mainly cleaning demons)
"""

import os
import shutil
import traceback
import subprocess

from utils import CustomLogger
from utils.pp import log_fail_run

my_logger = CustomLogger()


# === data clearing ===

def _clear_folder(path: str) -> None:
    """
    Clear all recently all files from given folder

    Parameters
    ----------
    path : str (optional)
        path to folder that holds files to clear

    Returns
    -------
    None
    """
    path = os.path.abspath(path)
    for filename in os.listdir(path):
        if filename == "__init__.py":
            continue
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def clear_logs(path=os.path.join("..", "logs")) -> None:
    """
    Clear all log files

    Parameters
    ----------
    path : str (optional)
        path to log folder assumed if not given

    Returns
    -------
    None
    """
    print("deleting log files...")
    _clear_folder(path)


def clear_tmp(path=os.path.join("..", "tmp")) -> None:
    """
    Clear all temporary files

    Parameters
    ----------
    path : str (optional)
        path to tmp folder assumed if not given

    Returns
    -------
    None
    """
    print("deleting temporary files...")
    _clear_folder(path)


def clear_data(path=os.path.join("..", "data")) -> None:
    """
    Clear all simulation data

    Parameters
    ----------
    path : str (optional)
        path to data folder assumed if not given

    Returns
    -------
    None
    """

    path = os.path.abspath(path)
    root, dirs, _ = next(os.walk(path))

    print("deleting simulation data...")

    for d in dirs:
        folder = d
        if folder == "experimental_data":
            continue

        _clear_folder(os.path.join(root, d))


def clear_all() -> None:
    """
    Clear all temporary files in project (simulation, logs, tmp)

    Returns
    -------
    None
    """
    print("clearing project...")
    clear_logs()
    clear_tmp()
    clear_data()
    print("project cleared!")


# === data access ===

def search_logs(log_file: str, search_string: str, log_path=os.path.join("..", 'logs')) -> None:
    """
    Search for lines in a file that contain a given string and write results in a new file.

    Parameters
    ----------
    log_file : str
        path to log file

    search_string : str
        string to search in line

    log_path : str (optional)
        path to folder containing the log files, assumed if not given

    Returns
    -------
    None
    """
    new_name = f"{log_file.split('.')[0]}_filtered.log"

    with open(os.path.join(log_path, log_file), 'r') as f_in:
        with open(os.path.join(log_path, new_name), 'w') as f_out:
            for line in f_in:
                if search_string in line:
                    f_out.write(line)


def find_files_with_extension(directory: str, extension: str) -> list:
    """
    Find files in a directory (excluding subdirectories) with a specific file extension.

    Parameters:
    -----------
    directory : str
        The path to the directory to search for files.

    extension : str
        The file extension to search for (e.g., '.txt', '.csv').

    Returns:
    --------
    list of str
        A list of file paths that match the specified extension in the specified directory.
    """
    # Initialize an empty list to store matching file paths
    matching_files = []

    # Iterate through files in the specified directory
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)

        # Check if the file is a regular file (not a directory) and has the specified extension
        if os.path.isfile(file_path) and file.endswith(extension):
            matching_files.append(file_path)

    return matching_files


# === project run ===

def execute_scripts(file_list):
    """
    Execute multiple Python scripts.

    Parameters
    ----------
    file_list : tuple
        A tuple containing the names of Python script files to execute.

    Raises
    ------
    Exception
        If any exception occurs during execution.

    Returns
    -------
    None
    """
    # iterate over given file list to run consecutively
    for file_name in file_list:
        try:
            subprocess.run(['python', file_name], check=True)
        except Exception as e:
            # get run name and exception traceback
            run_name = my_logger.file_name
            traceback_message = traceback.format_exc()

            # log error and continue
            my_logger.error(f"error executing {run_name} in {file_name}: {e}")
            log_fail_run(run_name, file_name, traceback_message)

            # reset logger file
            my_logger.change_file_handler()


if __name__ == '__main__':
    clear_all()
