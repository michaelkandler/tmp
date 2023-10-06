"""
Conversion tools for different data-types (like dict to list)
"""


def ina_params_l2d(model_par_list: list, input_par_list: list) -> tuple:
    """
    Create model/input-parameters dictionaries from lists in correct order

    Parameters
    ----------
    model_par_list : list
        Model parameters provided at list in same order as dictionary is instantiated
    input_par_list : list
        Input parameters provided at list in same order as dictionary is instantiated

    Returns
    -------
    Tuple
        model- and input parameters dictionaries
    """

    # set ina dict label
    ina_model_labels = ["an", "bn", "aa", "ba", "n"]
    ina_input_labels = ["cSR1", "cDR1", "cSR2", "cDR2", "cSR3", "cDR3"]

    # set aggregation order to 2 if not given
    model_par_list = model_par_list if len(model_par_list) == 5 else model_par_list.append(2)

    # create dicts
    model_parameters = {l: p for (l, p) in zip(ina_model_labels, model_par_list)}
    input_parameters = {l: p for (l, p) in zip(ina_input_labels, input_par_list)}

    return model_parameters, input_parameters
