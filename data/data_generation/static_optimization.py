# type hinting
import os.path
from time import time
import numpy as np

from controllers import FeedForward
from numeric import Integrator
from optimization import Minimizer

from processes import StationaryPoint
from models import INA
from controlled_plants import ControlledPlant
from optimization import StationaryConstraint, StationaryError


def stat_opt_run(plant: INA, integrator: Integrator, weights: tuple, pump_limits: tuple, minimizer: Minimizer,
                 x_star: np.ndarray,
                 name=None) -> StationaryPoint:

    name = str(time()) if name is None else name
    optimizer = StationaryPoint(name)
    optimizer.set_volume(x_star[-1])

    # === simulation configuration ===

    # instantiate controlled plant
    example_controlled_plant = ControlledPlant()

    # set model to plant and set plant -> controller would be ignored
    example_controlled_plant.set_plant(plant)

    # x0 is not needed it would just be overwritten

    # set controlled plant
    optimizer.set_controlled_plant(example_controlled_plant)

    # set integration method
    optimizer.set_integrator(integrator)

    # === optimization configuration ===

    # set pump limits
    optimizer.set_bounds(pump_limits)

    # set weights
    optimizer.set_weights(weights)

    # set minimizer, cost and constraint
    optimizer.set_minimizer(minimizer)
    optimizer.set_cost_func(StationaryError())
    optimizer.set_constraint_func(StationaryConstraint())

    optimizer.run_optimization()

    return optimizer


if __name__ == '__main__':

    from configurations import StaticOptimizationConfiguration
    conf = StaticOptimizationConfiguration()
    from utils import pp
    from utils.output_utils import pickle_process

    for m, i in conf.plant_conf:
        plant = INA(m, i)
        for integrator in conf.integrator:
            integrator = integrator()
            for weights in conf.weights:
                for pump_limits in conf.limits:
                    for minimizer in conf.minimizer:
                        for x_star, _ in conf.star_conf:
                            minimizer = minimizer()
                            run_opt = stat_opt_run(plant, integrator, weights, pump_limits, minimizer, x_star)

                            # save figure and results
                            parent_path = os.path.dirname(os.path.dirname(__file__))
                            save_path = os.path.abspath(os.path.join(parent_path, "stat_opt"))
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)

                            run_opt.plot_results(save_path=save_path)
                            run_opt.plot_cost_func(save_path=save_path)

                            with open(os.path.join(save_path, f"{run_opt.name}.txt"), 'w+') as f:
                                f.write(pp.box("set-up"))
                                f.write(f"\n{str(run_opt)}\n")
                                f.write(pp.box("scipy"))
                                f.write(f"\n{str(run_opt.sol)}\n")

                            pickle_process(run_opt, save_path)
