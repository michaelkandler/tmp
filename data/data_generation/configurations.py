from dataclasses import dataclass, field
from itertools import product

import numpy as np

# for type hinting
from models import Model

################################################simulation configuration################################################

from models import INA
from controllers.feed_forward import ZeroOrderHold, FirstOrderHold
from numeric.integration import SciPy, Trapezoid

model_parameters = [
    {'an': 1.3343, 'bn': -8.6824, 'aa': 12.0465, 'ba': -16.45, 'n': 2},  # J3
]
input_parameters = [
    {'cSR1': 11.5, 'cDR1': 4., 'cSR2': 0., 'cDR2': 5., 'cSR3': 0., 'cDR3': 0.},
]

# should be unpacked as m, i for range in plant_conf
plant_conf = tuple(product(model_parameters, input_parameters))

# initial values
x0 = [
    np.asarray([0., 0., 0., 0., 0.8]),
    np.asarray([0., 0., 0., 0., 1.]),
    np.asarray([0., 0., 0., 0., 1.2]),
    np.asarray([0., 0., 0., 0., 1.4]),
]

# order of input hold
input_hold = [
    FirstOrderHold,
]

# integration method
integrator = [
    Trapezoid,
]

# should be unpacked as h, i
numeric_params_sim = tuple(product(input_hold, integrator))

################################################optimizer configuration#################################################

from optimization import minimizer

# pump limitations
limits = [
    [[0.005, 0.1], None, [0.01, 0.2], None],
    # [[0.005, .1], None, [0.005, .1], [0.005, .2]],

]

# minimization algorithm
minimizer = [
    minimizer.TrustReg
]

# set point
x_star = [
    np.asarray([0.1277000261857676, 0.35704608930838094, 0.06296104405417025, 0.1905068381037631, 1.5]),
]

# set-point and initial guess
z_0 = [
    np.asarray([.005, 0., .1, 0.]),
]

# unpack as xs, z0
star_conf = tuple(zip(x_star, z_0))

# end time or initial guess for end time and time discretization
T0 = [
    9,
]
n_grid = [
    7,
]

# should be unpacked as t0, ng
time_grid = tuple(product(T0, n_grid))

# === static optimization ===

from optimization.cost_functions import StationaryError
from optimization.constraints import StationaryConstraint

p_y = [
    1

]
p_sty = np.geomspace(5e0, 1e2, num=20)
p_sty = list(p_sty)

# unpack as py, pyst
static_weights = tuple(product(p_y, p_sty))

# cost and constraint
static_cost = [
    StationaryError
]
static_constraints = [
    StationaryConstraint
]

# unpack as cost, const
static_cost_const = tuple(product(static_cost, static_constraints))

# === norm optimization ===

from optimization.cost_functions import NormOptError
from optimization.constraints import NormOptConstraints

# cost
k_end = [
    np.asarray([1., 2., 2., 1., 1.]) * 1e0
    ,
]
k_n = np.asarray([
    0
])
k_reg = [
    np.asarray([1., 1., 1., 1.]) * 1e0
]

# unpack as py, pyst, preg
norm_weights = tuple(zip(k_end, k_n, k_reg))

# cost and constraint
norm_cost = [
    NormOptError
]
norm_constraints = [
    NormOptConstraints
]

# unpack as cost, const
norm_cost_const = tuple(product(norm_cost, norm_constraints))

# === norm-time optimization ===

from optimization.cost_functions import TimeNormError
from optimization.constraints import TimeNormOptConstraints

# end consts
k_end = [
    np.asarray([1., 2., 2., 1., 1.]) * 1e6
    ,
]
k_n = [
    0,
]
k_t = [
    1e1,
]
k_reg = [
    np.asarray([1., 1., 1., 1.]) * 0e0
]

# unpack as py, pyst, pt, preg
norm_time_weights = tuple(zip(k_end, k_n, k_t, k_reg))

# cost and constraint
fed_batch_cost = [
    TimeNormError
]
fed_batch_constraints = [
    TimeNormOptConstraints
]

# unpack as cost, const
norm_time_cost_const = tuple(product(fed_batch_cost, fed_batch_constraints))


# === fed batch optimization ===

from optimization.cost_functions import FedBatchErrorFlow, FedBatchErrorYield
from optimization.constraints import FedBatchConstraints

tmp_vec = np.linspace(0, 10, num=10)
k_u = np.asarray([
    *(tmp_vec * 1),
    *(tmp_vec * 10),
])

k_sty = [
    np.asarray([1.]) * 1e0,
    ] * len(k_u)

k_reg = [
    np.asarray([1., 1., 1., 1.]) * 0e0,
] * len(k_u)

# unpack as py, pyst, pt, preg
fed_batch_weights = tuple(zip(k_u, k_sty, k_reg))

# cost and constraint
fed_batch_cost = [
    FedBatchErrorFlow
]
fed_batch_constraints = [
    FedBatchConstraints
]

# unpack as cost, const
fed_batch_cost_const = tuple(product(fed_batch_cost, fed_batch_constraints))

########################################################################################################################
###################################################actual dataclasses###################################################
########################################################################################################################

@dataclass
class SimulationConfiguration:
    """
    Basic configurations for any simulation
    """
    # used model
    model: Model = INA

    # model parameters
    model_params: tuple = field(default_factory=lambda: model_parameters)
    input_params: tuple = field(default_factory=lambda: input_parameters)

    # get zipped plant conf
    plant_conf: tuple = field(default_factory=lambda: plant_conf)

    # initial state
    x0: tuple = field(default_factory=lambda: x0)

    # used hold
    controller: tuple = field(default_factory=lambda: input_hold)

    # used integration method
    integrator: tuple = field(default_factory=lambda: integrator)


@dataclass
class OptimizationConfiguration(SimulationConfiguration):
    """
    Base configuration for every optimization
    """
    limits: tuple = field(default_factory=lambda: limits)
    minimizer: tuple = field(default_factory=lambda: minimizer)
    star_conf: tuple = field(default_factory=lambda: star_conf)
    time_conf: tuple = field(default_factory=lambda: time_grid)


@dataclass
class StaticOptimizationConfiguration(OptimizationConfiguration):
    """
    Configuration for stationary point optimization
    """
    weights: tuple = field(default_factory=lambda: static_weights)
    cost_const: tuple = field(default_factory=lambda: static_cost_const)


@dataclass
class NormOptimizationConfiguration(OptimizationConfiguration):
    """
    Configuration for norm-optimal optimization
    """
    weights: tuple = field(default_factory=lambda: norm_weights)
    cost_const: tuple = field(default_factory=lambda: norm_cost_const)


@dataclass
class TimeNormOptimizationConfiguration(OptimizationConfiguration):
    """
    Configuration for time-norm-optimal optimization
    """
    weights: tuple = field(default_factory=lambda: norm_time_weights)
    cost_const: tuple = field(default_factory=lambda: norm_time_cost_const)

@dataclass
class FedBatchConfiguration(OptimizationConfiguration):
    """
    Configurations for fed-batch optimization
    """
    weights: tuple = field(default_factory=lambda: fed_batch_weights)
    cost_const: tuple = field(default_factory=lambda: fed_batch_cost_const)


if __name__ == '__main__':
    pass
