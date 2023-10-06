import numpy as np

from processes import Analysis

# define standard values
stdx0 = [0., 0., 0., 0., 0.8]
std_mod_params = {'an': 0.776796168, 'bn': -6.320839249, 'aa': 2.180377932, 'ba': -8.553075612, 'n': 2}
std_in_params = {'cSR1': 11 / 4, 'cDR1': 4 / 4, 'cSR2': 0, 'cDR2': 0, 'cSR3': 0, 'cDR3': 0.}

# create and define analysis object
analysis_obj = Analysis(np.asarray(stdx0), std_mod_params, std_in_params)

# analyse step response
x_0 = np.asarray([0., 0., 0., 0., 0.5])
analysis_obj.step_response(11., 4., 0.005, x_0, std_mod_params, 24, silent=True)
analysis_obj.plot_dynamics()

# analyse ramp response
analysis_obj.ramp_response(5., 4., 0.005, x_0, std_mod_params, 24, silent=True)
analysis_obj.plot_dynamics()

# analyse batch behavior
x_0 = np.asarray([0.5, 0., 0., .25, 0.5])
analysis_obj.batch_response(x_0, std_mod_params, 24, silent=True)
analysis_obj.plot_dynamics()

# print example
print(analysis_obj)
