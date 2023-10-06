import numpy as np

from processes import StationaryPoint

from models import INA
from controlled_plants import ControlledPlant
from optimization.constraints import StationaryConstraint
from optimization.cost_functions import StationaryError
from numeric import SciPy
from optimization.minimizer import TrustReg

tmp_opt = StationaryPoint("example")

#############################################################
######################change here############################

bounds = [[0.005, 0.1], None, [0.005, 0.1], [0.005, 0.1]]
weights = np.asarray([1e2, 0e7])

#############################################################
#############################################################

# === simulation configuration ===

# instantiate controlled plant
example_controlled_plant = ControlledPlant()

# create model
mod_pars = {'an': 1.3343, 'bn': -8.6824, 'aa': 12.0465, 'ba': -16.45, 'n': 2}  # J3
inp_pars = {'cSR1': 11, 'cDR1': 4, 'cSR2': 0., 'cDR2': 4., 'cSR3': 0., 'cDR3': 0.}
example_model = INA(mod_pars, inp_pars)

# set model to plant and set plant -> controller would be ignored
example_controlled_plant.set_plant(example_model)

# x0 is not needed it would just be overwritten

# set controlled plant
tmp_opt.set_controlled_plant(example_controlled_plant)

# set integration method
tmp_opt.set_integrator(SciPy())

# === optimization configuration ===

# set pump limits
tmp_opt.set_bounds(bounds)

# set weights
tmp_opt.set_weights(weights)

# set minimizer, cost and constraint
tmp_opt.set_minimizer(TrustReg())
tmp_opt.set_cost_func(StationaryError())
tmp_opt.set_constraint_func(StationaryConstraint())

# actual optimization
tmp = tmp_opt.run_optimization()

# plot stuff
tmp_opt.plot_results()
tmp_opt.plot_cost_func()

# print(tmp_opt)

# print results
print(f"success: {tmp_opt.sol.success}")
print(f"z_star: {tmp_opt.sol.x}")
print(f"    S_in: {tmp_opt.S_in} g/l")
print(f"    𝛕: {tmp_opt.tau:.3} h")
print(f"------------------------")
print(f"Einsatz: {500 * np.sum(tmp_opt.x[-1, :3])}")
print(f"Gewinn: {1e6 * tmp_opt.x[1, -1] / (np.sum(tmp_opt.z_star) / tmp_opt.x[1, -1])}€")
