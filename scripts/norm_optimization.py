import os.path

import numpy as np

from processes import NormOptimization

# imports for simulation
from controlled_plants import ControlledPlant
from models import INA
from controllers import ZeroOrderHold, FirstOrderHold, CubicSplines
from numeric import SciPy, Trapezoid, Euler
from numeric.noise import Gauss, NoNoise

# imports for optimization
from optimization import NormOptError, NormOptConstraints
from optimization import TrustReg, DualAnnealing, SHOG, SLSQP

example_opt = NormOptimization("example")

#########################################################
##################change here############################

# pump limits
bounds = [[0.005, .1], None, [0.005, .1], None]

# weights
k_end = np.asarray([1., 1., 1., 1., 1e0]) * 1e1
k_i = np.asarray([0])
k_reg = np.asarray([1., 1., 1., 1.]) * 0e6
norm_weights = (k_end, k_i, k_reg)

# time-grid
n_steps = 10
T = 12

# hold order
hold = FirstOrderHold

# minimizer
example_opt_alg = SLSQP()

# noise
measurement_noise = NoNoise()  # Gauss(1e-4)
pump_noise = NoNoise()#  Gauss(1e-3, mean=0.00)

#########################################################
#########################################################

# === simulation configuration ===

# create controlled plant
example_controlled_plant = ControlledPlant()

# configure plant
model_parameters = {'an': 1.3343, 'bn': -8.6824, 'aa': 12.0465, 'ba': -16.45, 'n': 2}  # J3
input_parameters = {'cSR1': 11.5, 'cDR1': 4., 'cSR2': 0., 'cDR2': 0., 'cSR3': 0., 'cDR3': 0.}
example_model = INA(model_parameters, input_parameters)
example_controlled_plant.set_plant(example_model)

# configure controller
example_controller = hold(None, None)
example_controlled_plant.set_controller(example_controller)

# set initial value
x0 = np.asarray([0, 0, 0, 0, .8])
example_controlled_plant.set_initial_value(x0)

# put it all together
example_opt.set_controlled_plant(example_controlled_plant)

# set integration method
example_integrator = Trapezoid()
# example_integrator.set_delta_t(0.25)
example_opt.set_integrator(example_integrator)

# === optimization configuration ===

# set set-point
x_star = np.asarray([0.11525703, 0.32221684, 0.08651147, 0.19054012, 1.8])
example_opt.set_x_star(x_star)

# time-grid configuration
example_opt.set_t(T, n_steps, mode="lin")

# set constraints
example_opt.set_bounds(bounds)
example_opt.set_constraint_func(NormOptConstraints())

# set error_function
example_opt.set_cost_func(NormOptError())

# set weights

example_opt.set_weight(norm_weights)

# set optimization algorithm
example_opt.set_minimizer(example_opt_alg)

# create initial guess
from utils.input_utils import create_constant_z0
z0 = create_constant_z0(np.asarray([0.005, 0.00, 0.1, 0.1]), n_steps, bounds, zero_order=isinstance(example_opt, ZeroOrderHold))#, zero_order=isinstance(example_controller, ZeroOrderHold))
example_opt.set_initial_guess(z0)

# run and plot everything
example_opt.run_optimization()
# example_opt.controlled_plant.controller.set_noise_generator(pump_noise)
example_opt.plot_results(save_path=os.path.abspath(f"/home/friedi/Desktop"), title="first guess")

example_opt.set_initial_guess(example_opt.z_star)
example_opt.run_optimization()
example_opt.plot_results(save_path=os.path.abspath(f"/home/friedi/Desktop"), title="second guess")

print(example_opt.sol)
