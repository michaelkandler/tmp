import numpy as np

from processes import NormTimeOptimization

from controlled_plants import ControlledPlant
from models import INA
from controllers import ZeroOrderHold, FirstOrderHold, CubicSplines

from optimization.cost_functions import TimeNormError
from optimization.constraints import TimeNormOptConstraints

from numeric import SciPy, Trapezoid, Euler
from optimization import TrustReg, DualAnnealing, NelderMead, SHOG, SLSQP

#########################################################
##################change here############################

# pump limits
bounds = [[0., .1], None, [0., .1], None]

# weights
k_end = np.asarray([1., 1., 1., 1., 1e1]) * 1e1
k_t = np.asarray([1]) * 1e-2

# time-grid
n_steps = 10
T0 = 100

# hold order
hold = FirstOrderHold

# minimizer
example_opt_alg = SLSQP()

show_f_hist = False

#########################################################
#########################################################
example_opt = NormTimeOptimization("example")

#### simulation configuration ####

# create controlled plant
example_controlled_plant = ControlledPlant()

# configure plant
model_parameters = {'an': 1.3343, 'bn': -8.6824, 'aa': 12.0465, 'ba': -16.45, 'n': 2}  # J3
input_parameters = {'cSR1': 11.5, 'cDR1': 4., 'cSR2': 0., 'cDR2': 5., 'cSR3': 0., 'cDR3': 0.}
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
example_opt.set_integrator(example_integrator)

#### optimization configuration ####

# set set-point
x_star = np.asarray([0.11525703, 0.32221684, 0.08651147, 0.19054012, 1.8])
example_opt.set_x_star(x_star)

# time-grid configuration
example_opt.set_tau(n_steps, mode="lin")

# set pump limits
example_opt.set_bounds(bounds)

# create initial guess
from utils.input_utils import create_constant_z0
z0_part = np.asarray([0.005, 0.00, 0.1, 0.0])
z0 = create_constant_z0(z0_part, n_steps, bounds, T_start=T0, zero_order=isinstance(hold, ZeroOrderHold))
example_opt.set_initial_guess(z0)

# set weights
time_norm_weights = (k_end, 0, k_t, 0)
example_opt.set_weight(time_norm_weights)

# set optimization algorithm
if show_f_hist:
    from optimization.opt_callbacks import LogSteps
    example_opt_alg.set_callback(LogSteps())
example_opt.set_minimizer(example_opt_alg)

# set error_function
example_opt.set_cost_func(TimeNormError())

# set constraint function
example_opt.set_constraint_func(TimeNormOptConstraints())

example_opt.run_optimization()
example_opt.plot_results()

if show_f_hist:
    example_opt.minimizer.callback.plot_f()

print(example_opt.sol)
