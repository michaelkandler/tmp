import os.path
from time import time

import numpy as np

# fixed packages
from processes import NormOptimization
from controlled_plants import ControlledPlant
from models import INA
from optimization import NormOptError, NormOptConstraints

# type hinting
from controllers import FeedForward, ZeroOrderHold
from numeric import Integrator
from optimization import Minimizer
from numeric import Noise, NoNoise


def run_norm_opt(plant: INA, controller: FeedForward, integrator: Integrator, x0: np.ndarray,
                 set_point: np.ndarray, num_steps: int, T: float, weights: tuple, pump_limits: tuple,
                 minimizer: Minimizer, z0: np.ndarray,
                 pump_noise: Noise = NoNoise(), state_noise: Noise = NoNoise(), x0_noise: Noise = NoNoise(),
                 name=None) -> NormOptimization:
    """
    Wrapper function to generate data from a norm-optimal optimization

    Parameters
    ----------
    plant : INA
        configured plant
    controller : FeedForward
        controller used in plant (zero-order, etc.)
    integrator : Integrator
        integration method used
    x0 : np.ndarray
        inital value for ode
    set_point : np.ndarray
        state-space point to reach
    num_steps : int
        number of discretization-steps
    T : float
        length of time horizont
    weights : tuple
        weights for optimizer (p_end, p_k , p_reg), p_i can be a scalar or a vector of proper size
    pump_limits : tuple
        pump limits in form ([low1, high1], ... [low4, high4]), replace [...] with None if pump is not used
    minimizer : Minimizer
        algorithm to use for minimization
    z0 : np.ndarray
        inital guess for constant flow over entire time-horizont
    x0_noise : Noise
        noise generator for inital value insecurity
    state_noise : Noise
        noise generator for state noise
    pump_noise : Noise
        noise generator for pump noise
    name : str
        optional name for optimization if not given a unix time-stamp will be used

    Returns
    -------

    """
    # create optimization object
    name = str(time()) if name is None else name
    optimizer = NormOptimization(name)

    # === simulation configuration ===

    # create controlled plant
    controlled_plant = ControlledPlant()

    # configure plant
    example_model = plant
    controlled_plant.set_plant(example_model)

    # configure controller
    controller = controller(None, None)
    controlled_plant.set_controller(controller)

    # set initial value
    x0 = x0_noise(x0)
    controlled_plant.set_initial_value(x0)

    # set state noise
    plant.set_noise_generator(state_noise)

    # put it all together
    optimizer.set_controlled_plant(controlled_plant)

    # set integration method
    optimizer.set_integrator(integrator())

    # === optimization configuration ===

    # set set-point
    optimizer.set_x_star(set_point)

    # time-grid configuration
    optimizer.set_t(T, num_steps, mode="lin")

    # set constraints
    optimizer.set_bounds(pump_limits)
    optimizer.set_constraint_func(NormOptConstraints())

    # set error_function
    optimizer.set_cost_func(NormOptError())

    # set weights
    optimizer.set_weight(weights)

    # set optimization algorithm
    optimizer.set_minimizer(minimizer)

    # create initial guess
    from utils.input_utils import create_constant_z0
    z0 = create_constant_z0(z0, num_steps, pump_limits, zero_order=isinstance(controller, ZeroOrderHold))
    optimizer.set_initial_guess(z0)

    # run and plot everything
    optimizer.run_optimization()
    optimizer.controlled_plant.controller.set_noise_generator(pump_noise)

    return optimizer


if __name__ == '__main__':

    from configurations import NormOptimizationConfiguration
    from models import INA
    from utils import pp, pickle_process

    conf = NormOptimizationConfiguration()

    controller = conf.controller

    for m, i in conf.plant_conf:
        plant = INA(m, i)
        for x0 in conf.x0:
            for controller in conf.controller:
                for integrator in conf.integrator:
                    for set_point, z0 in conf.star_conf:
                        for pump_limits in conf.limits:
                            for weights in conf.weights:
                                for mymin in conf.minimizer:
                                    minimizer = mymin()
                                    for T, num_steps in conf.time_conf:
                                        run_opt = run_norm_opt(plant, controller, integrator, x0, set_point, num_steps,
                                                               T, weights, pump_limits, minimizer, z0)
                                        # save figure and results
                                        parent_path = os.path.dirname(os.path.dirname(__file__))
                                        save_path = os.path.abspath(os.path.join(parent_path, "norm_opt"))
                                        if not os.path.exists(save_path):
                                            os.makedirs(save_path)

                                        run_opt.plot_results(save_path=save_path)

                                        with open(os.path.join(save_path, f"{run_opt.name}.txt"), 'w+') as f:
                                            f.write(pp.box("set-up"))
                                            f.write(f"\n{str(run_opt)}\n")
                                            f.write(pp.box("scipy"))
                                            f.write(f"\n{str(run_opt.sol)}\n")

                                        pickle_process(run_opt, save_path)
