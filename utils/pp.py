"""
string conversion tools for
    - pretty printing
    - csv log-creation
"""
import os
import re

# type hints
# from processes import Optimization

import numpy as np


# === pretty print ===


def box(text: str, box_char='*', center=120) -> str:
    """
    returns the given string with surrounding box

    Parameters
    ----------
    text : str
        string to return
    box_char : char
        character to build the box standard is '*'
    center : bool
        center text along python line length (79) if True
    Returns
    -------
    str
    """
    box_str = [f"{box_char * (len(text) + 4)}", f"{box_char} {text} {box_char}", f"{box_char * (len(text) + 4)}"]
    box_str = [b.center(center) if center is not None else b for b in box_str]
    box_str = f"{box_str[0]}\n{box_str[1]}\n{box_str[2]}"

    return box_str


def hl(sep_char='=', num=120) -> str:
    """
    Return a horizontal separating line

    Parameters
    ----------
    sep_char: char
        character to build horizontal line, standard is '-'
    num: int
        length of line, standard is 120 (python line length)

    Returns
    -------
    str
        horizontal line
    """
    return sep_char * num


def round_all_floats(text: str, digits=5) -> str:
    """
    Get a string and round all occurring float values to the given number of digits.

    Parameters
    ----------
    text : str
        text to reformat
    digits : int
        optional, number of digits to round to. default is 5

    Returns
    -------
    str
        reformatted string with rounded values
    """
    if not isinstance(digits, int):
        raise ValueError("Number of digits must be an integer")

    # define group substitution
    def truncate_float(match):
        float_value = float(match.group())
        truncated_value = format(float_value, f".{digits}f")
        return truncated_value

    # sub all groups
    return re.sub(r"\d+\.\d+", truncate_float, text)


def pretty_print(title: str, text: str, round_floats: None | int = None) -> str:
    """
    Create a pretty block with title for given string

    Parameters
    ----------
    title : str
        title for box
    text : str
        actual text
    round_floats : bool


    Returns
    -------
    str
        pretty text
    """
    # round floating point numbers if given
    text = text if round_floats is None else round_all_floats(text, round_floats)

    return box(title) + '\n' + hl() + '\n' + text + '\n' + hl() + '\n\n\n'


# === cvs utils ===

# TODO create csv from optimization object
def log_runs(opt: type | None, failed: str | None,
             file_name="run.csv", file_path=os.path.join("..", "data")) -> None:
    pass


def log_fail_run(run_name, file_name, traceback_message):
    pass


if __name__ == '__main__':
    pass
