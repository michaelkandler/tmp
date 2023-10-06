import numpy as np

from processes import FedBatchOptimization

from controlled_plants import ControlledPlant
from models import INA
from controllers import ZeroOrderHold, FirstOrderHold, CubicSplines

from optimization.cost_functions import FedBatchErrorYield, FedBatchErrorFlow
from optimization.constraints import FedBatchConstraints

from numeric import SciPy, Trapezoid
from optimization import TrustReg, DualAnnealing, NelderMead, SHOG, SLSQP

#########################################################
##################change here############################

# pump limits
bounds = [[0., .3], None, [0., .2], None]

# weights
k_y = 1e2
k_sty = 1e6
k_reg = np.asarray([1., 1., 1., 1.]) * 0e-1
norm_weights = (k_y, k_sty, k_reg)

# time-grid
n_steps = 50
T0 = 24
# hold order
hold = FirstOrderHold

# minimizer
example_opt_alg = TrustReg()

show_f_hist = False

#########################################################
#########################################################
example_opt = FedBatchOptimization("fed_batch")

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
x_star = np.asarray([0.11525703, 0.32221684, 0.08651147, 0.19054012, 1.8 * 2])
example_opt.set_x_star(x_star)

# time-grid configuration
example_opt.set_tau(n_steps, mode="lin")

# set pump limits
example_opt.set_bounds(bounds)

# create initial guess
from utils.input_utils import create_constant_z0
z0_part = np.asarray([.0, .0, 0.1, .0])
z0 = create_constant_z0(z0_part, n_steps, bounds, T_start=T0, zero_order=isinstance(hold, ZeroOrderHold))
example_opt.set_initial_guess(z0)

# set weights
example_opt.set_weight(norm_weights)

# set optimization algorithm
if show_f_hist:
    from optimization.opt_callbacks import LogSteps
    example_opt_alg.set_callback(LogSteps())
example_opt.set_minimizer(example_opt_alg)

# set error_function
example_opt.set_cost_func(FedBatchErrorFlow())

# set constraint function
example_opt.set_constraint_func(FedBatchConstraints())

example_opt.run_optimization()
example_opt.plot_results(save_path="../data/fed_batch", plot_dynamics=True)

if show_f_hist:
    example_opt.minimizer.callback.plot_f()

I, N, A, D, V = example_opt.x[-1]
T = example_opt.t[-1]

y = (N/(I + A + N))
sty = N/T

print(f"yield: {y}")
print(f"st_yield: {sty}")
print(example_opt.sol)