import numpy as np
from time import time

from processes.process import Process
from numeric import Integrator

from numeric.noise import Noise, NoNoise
from utils.logger import CustomLogger

my_logger = CustomLogger()


class Simulation(Process):
    """
    General simulation work bench.

    Implements:
        - Creating time-grid (equidistant) for simulation
        - Running simple simulation with given controller and start value
        - Adding measurement noise to done simulation

    """

    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            name of simulation
        """
        super().__init__(name)

        self.sim_name = "generic simulation"

        # sampling and process time
        self.t_samp = None
        self.t = None

        self.integrator = None

        self.noise_generator = NoNoise()

    def set_t(self, t: float, t_samp: float) -> np.ndarray:
        """
        Create an evenly spaced time vector with given end- and sampling-time and assign it to self as simulation time

        Parameters
        ----------
        t : float
            End time of simulation
        t_samp : float
            Sampling time

        Returns
        -------
        np.ndarray
            Evenly spaced time vector with sampling time self.t_samp and end time t
        """
        my_logger.debug(f"setting time from 0 to {t} with sampling time {t_samp}")

        t_samp = t_samp / 60 / 60
        t = np.arange(0, t + t_samp, t_samp)

        self.t = t
        self.t_samp = t_samp

        return t

    def set_noise_generator(self, generator: Noise) -> None:
        """
        Set noise generator to get noise output values.

        Parameters
        ----------
        generator : Noise
            Full configured noise generator, can be instantiated in module numeric.noise

        Returns
        -------
        None
        """
        my_logger.debug(f"setting noise to {generator} in controller {self.__repr__()}")
        self.noise_generator = generator

    def run_process(self) -> None:
        """
        Simulate the process with the simulation_data/configuration held by the controlled plant

        Returns
        -------
        None
        """

        if self.controlled_plant is None:
            my_logger.error("controlled plant not found")
            raise ValueError('Controlled plant must be given!')
        if self.controlled_plant.plant is None:
            my_logger.error("plant not found")
            raise ValueError('Plant of the controlled plant must be given')
        if self.controlled_plant.controller is None:
            my_logger.error("controller not found")
            raise ValueError('Controller of the controlled plant must be given')
        if self.controlled_plant.x0 is None:
            my_logger.error("initial value not found")
            raise ValueError('Initial values must be given')

        my_logger.debug("running simulation")

        # The actual simulation
        self.x = self.integrator(self.controlled_plant, self.controlled_plant.x0, self.t)
        self.x = self.x if self.noise_generator is None else self.noise_generator(self.x)

        _, self.u = self.controlled_plant.return_history(self.t) if self.controlled_plant.track_hist else (None, self.u)

        # just stamp time as simulation name
        self.sim_name = str(time())

    def set_integrator(self, integrator: Integrator) -> None:
        """
        Set integration methode for simulation

        Parameters
        ----------
        integrator : Integrator
            Instance of integrator with signature int(func, x0, t)

        Returns
        -------
        None
        """
        self.integrator = integrator
        my_logger.debug(f"setting integration methode to: {repr(integrator)}")

    def add_noise_generator(self, generator: Noise) -> None:
        """
        Add noise output noise to simulated data

        Parameters
        ----------
        generator : Noise
            Noise generating object

        Returns
        -------
        None
        """
        my_logger.debug(f"setting noise generator: {generator}")
        self.noise_generator = generator

    def plot_results(self, reference=None, save_path=None, title=None) -> None:
        """
        Plot results obtained by simulation

        Parameters
        ----------
        reference : bool
            plot reference values in plot (like measurements)
        save_path : str or None
            save figure to path if given
        title : str or None
            title to save plot under
        Returns
        -------
        None
        """
        my_logger.debug("plotting results...")

        if self.x is None:
            raise ValueError('No data available')

        if title is not None:
            title = title
        else:
            title = self.name + "_" + self.sim_name

        my_logger.debug(f"plotting results...")

        self.controlled_plant.plant.plot_process(self.t, self.x, u=self.u, title=title, ref=reference, save_path=save_path)

    def __str__(self) -> str:
        description = f"Simulation {self.name} created at {self.timestamp}\n" \
                      f"-----------\n" \
                      f"Duration: {self.t[-1]} [h]\n" \
                      f"-----------\n" \
                      f"Sampling time: {self.t_samp} [h]\n" \
                      f"-----------\n" \
                      f"Measurement noise: {self.noise_generator}\n" \
                      f"----------------------\n" \
                      f"{self.controlled_plant}"

        return description

    def __repr__(self):

        representation = (f"Simulation(controlled_plant={repr(self.controlled_plant)}, name={self.name}, "
                          f"t={self.t}, noise_generator={repr(self.noise_generator)})")

        return representation


if __name__ == '__main__':
    from models import INA
    from controllers import *
    from controlled_plants import ControlledPlant
    from numeric.noise import Gauss
    from numeric.integration import SciPy, Trapezoid, Euler, RungeKutta

    # instantiate simulation builder
    example_simulation = Simulation('example simulation')

    #### configure plant and input ####

    # instantiate controlled plant
    example_controlled_plant = ControlledPlant()

    # set up controller and modeled plant
    model_params = {'an': 1.8038, 'bn': -13.6793, 'aa': 58.4058, 'ba': -23.0235, 'n': 2}
    input_params = {'cSR1': 11, 'cDR1': 4, 'cSR2': 0, 'cDR2': 5., 'cSR3': 0, 'cDR3': 0}
    example_plant = INA(model_params, input_params)
    example_controlled_plant.set_plant(example_plant)

    example_controller = Step(0, np.asarray([0.005, 0, 0, 0]))
    example_controlled_plant.set_controller(example_controller)

    # initial value
    x0 = np.asarray([0., 0., 0., 0., 1.])
    example_controlled_plant.set_initial_value(x0)

    # give controlled plant to simulation object
    example_simulation.set_controlled_plant(example_controlled_plant)

    #### configure simulation paramters ####

    # simulation time
    t_sim = 30  # [h]
    t_sp = 60  # [s]
    example_simulation.set_t(t_sim, t_sp)

    # integration methode
    example_simulation.integrator = SciPy()

    # add optional measurement noise
    example_simulation.add_noise_generator(Gauss(.001))

    #### run simulation ####

    # run simulation and plot it
    example_simulation.run_process()
    example_simulation.plot_results()

    print(example_simulation)
