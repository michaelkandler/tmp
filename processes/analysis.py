import os

import numpy as np
import matplotlib.pyplot as plt

from models import INA
from controlled_plants import ControlledPlant
from controllers import Step, Ramp, Batch
from processes import Simulation
from numeric import SciPy

from utils.logger import CustomLogger

my_logger = CustomLogger()


class Analysis(Simulation):
    """
    Standard signal analysis for INA-model. Implements step, ramp and batch response analysis and plotting.

    Just a neat wrapper.
    """

    def __init__(self, std_x0: np.ndarray, std_model_params: dict, std_input_params: dict) -> None:
        """

        Parameters
        ----------
        std_x0 : np.ndarray
            standard initial value for integrator
        std_model_params : dict
            standard model parameters, probably won't change
        std_input_params : dict
            standard input params, will change for step response
        """
        super().__init__('analysis')

        # set initial value
        self.x0 = std_x0

        self.controlled_plant = ControlledPlant()
        self.controlled_plant.set_initial_value(std_x0)
        self.set_t(10, 60)
        self.controlled_plant.set_plant(INA, std_model_params, std_input_params)

        # set standard values for analysis
        self.std_x0 = std_x0
        self.std_model_params = std_model_params
        self.std_input_params = std_input_params
        self.controlled_plant.plant = INA(self.std_model_params, self.std_input_params)

        self.set_integrator(SciPy())

        my_logger.debug(f"setting standard model parameters to: {self.std_model_params}")
        my_logger.debug(f"setting standard input parameters to: {self.std_input_params}")
        my_logger.debug(f"setting standard initial value to: {self.std_x0}")

    def step_response(self, c_s_in: float, c_d_in: float, u_in: float,
                      x0: np.ndarray = None, model_params: dict = None, t_sim: float = None, silent: bool = False
                      ) -> None:
        """
        Simulate and plot step-response with given input flow and concentration.

        Parameters
        ----------
        c_s_in : float
            concentration of in flowing solubilizate
        c_d_in : float
            concentration of in flowing solubilizate
        u_in : float
            flow rate for input
        x0 : np.ndarray, optional
            initial state, std value if `None`, by default None
        model_params : dict, optional
            model parameters, std value if `None`, by default None
        t_sim : float, optional
            simulation time, 10h if `None`, by default None
        silent : bool, optional
            supress plotting if `True`, by default False
        """
        my_logger.debug(f"setting up ramp response controller...")

        # parameters of step input
        input_params = self.controlled_plant.plant.input_params
        input_params['cSR1'] = c_s_in
        input_params['cDR1'] = c_d_in
        self.controlled_plant.plant.change_params(input_params=input_params)

        # set step controller with given parameters
        step_controller = Step(0, FRs=np.asarray([u_in, 0., 0., 0.]))
        self.controlled_plant.controller = step_controller
        self.controlled_plant.track_history(True)
        self.controlled_plant.x0 = x0

        # actual simulation
        self._simulate(x0=x0, model_params=model_params, input_params=input_params, t_sim=t_sim)

        if not silent:
            self.plot_dynamics()

    def ramp_response(self, c_s_in: float, c_d_in: float, d_uin: float,
                      x0: np.ndarray = None, model_params: dict = None, t_sim: float = None, silent: bool = False
                      ) -> None:
        """
        Simulate and plot ramp-response with given input flow and concentration.

        Parameters
        ----------
        c_s_in : float
            concentration of in flowing solubilizate 
        c_d_in : float
            concentration of in flowing solubilizate 
        d_uin : float
            flow rate gradient for input
        x0 : np.ndarray, optional
            initial state, std value if `None`, by default None
        model_params : dict, optional
            model parameters, std value if `None`, by default None
        t_sim : float, optional
            simulation time, 10h if `None`, by default None
        silent : bool, optional
            supress plotting if `True`, by default False
        """
        my_logger.debug(f"setting up step response controller...")

        # parameters of step input
        input_params = self.controlled_plant.plant.input_params
        input_params['cSR1'] = c_s_in
        input_params['cDR1'] = c_d_in

        # set step controller with given parameters
        self.controlled_plant.controller = Ramp(0., FRs0=np.asarray([d_uin, 0., 0., 0.]))
        self.controlled_plant.x0 = x0

        # actual simulation
        self._simulate(x0=x0, model_params=model_params, input_params=input_params, t_sim=t_sim)

        if not silent:
            self.plot_dynamics()

    def batch_response(self, x0: np.ndarray,
                       model_params: dict = None, t_sim: float = None, silent: bool = False) -> None:
        """
        Simulate and plot batch process with given initial state

        Parameters
        ----------
        x0 : np.ndarray
            initial state
        model_params : dict, optional
            model parameters, std value if `None`, by default None
        t_sim : float, optional
            simulation time, 10h if `None`, by default None
        silent : bool, optional
            supress plotting if `True`, by default False
        """
        my_logger.debug(f"setting up batch controller...")

        self.controlled_plant.controller = Batch()
        self.controlled_plant.set_initial_value(x0)

        self._simulate(x0=x0, model_params=model_params, t_sim=t_sim)

        if not silent:
            self.plot_dynamics()

    def _simulate(self, x0: np.ndarray = None, model_params: dict = None, input_params: dict = None,
                  t_sim: float = None) -> None:
        """
        Simulate and plot response with set controller and parameters

        Parameters
        ----------
        x0 : np.ndarray, optional
            initial state, by default None
        model_params : dict, optional
            model parameters, std value if `None`, by default None
        input_params : dict, optional
            input parameters, std value if `None`, by default None
        t_sim : float, optional
            simulation time, 10h if `None`, by default None

        Returns
        ----------
        None
        """
        if self.controlled_plant.plant is None or self.controlled_plant.controller is None:
            raise AssertionError('Model and controller have to be defined')

        my_logger.debug(f"setting initial value to {x0}")
        my_logger.debug(f"setting model parameters to {model_params}")
        my_logger.debug(f"setting input parameters to {input_params}")

        # set given values for initialization
        self.controlled_plant.plant.x0 = self.x0 \
            if x0 is None else x0
        self.controlled_plant.plant.model_params = self.controlled_plant.plant.model_params \
            if model_params is None else model_params
        self.controlled_plant.plant.input_params = self.controlled_plant.plant.input_params \
            if input_params is None else input_params
        self.t = np.arange(0, 10 + self.t_samp, self.t_samp) \
            if t_sim is None else np.arange(0, t_sim + self.t_samp, self.t_samp)

        # actual simulation
        self.run_process()

        # reset standards
        self._reset_standards()

    def _reset_standards(self) -> None:
        """
        Reset to standard values

        Returns
        ----------
        None
        """
        my_logger.debug(f"resetting to standard parameters...")
        my_logger.debug(f"standard initial value: {self.std_model_params}")
        my_logger.debug(f"standard model parameters: {self.std_model_params}")
        my_logger.debug(f"standard inputs parameters:: {self.std_model_params}")

        # retesting standard parameters
        self.controlled_plant.plant.x0 = self.std_x0
        self.controlled_plant.plant.model_params = self.std_model_params
        self.controlled_plant.plant.input_params = self.std_input_params

    def plot_dynamics(self, path=None, title=None) -> None:
        """
        Plot the time constant of the simulation over the entire horizont
        """
        # calculate time constants
        t_batch, t_part_u = self.controlled_plant.plant.calc_time_dynamics(self.x, self.u)

        fig = plt.figure(figsize=(14, 6))  # Sets the figure size

        # plot concentrations
        axes1 = fig.add_axes([0.075, 0.2, 0.7 / 2, 0.7])  # defines the position a dimensions of the first subplot
        axes1.plot(self.t, self.x[:, 0], color="tab:olive",
                   label="folding intermediate c_i")  # Plotting of N_simulated; definition of legend entry for c_i
        axes1.plot(self.t, self.x[:, 1], color="tab:green",
                   label="native protein N")  # Plotting of N_simulated; definition of legend entry for N
        axes1.plot(self.t, self.x[:, 2], color="tab:red",
                   label="aggregated protein A")  # Plotting of A_simulated; definition of legend entry for A
        if self.u is not None:
            axes_u = axes1.twinx()
            axes_u.plot(self.t, self.u, color="tab:grey",
                        label="input")  # plotting the input
            axes_u.set_ylabel("Input flow $u$ [L/h]")

        axes1.set_ylabel("Concentration of I, N and A [g/L]")  # Label at the Y-Axis
        axes1.set_xlabel("Time [h]")  # Label at the X-Axis
        axes1.grid()  # Turns the grid on
        axes1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),  # Shows the legend
                     ncol=2, fancybox=True, shadow=True)

        # plot time constant
        axis2 = fig.add_axes(
            [0.075 * 2 + 0.7 / 2 + 0.08, 0.2, 0.7 / 2,
             0.7])  # defines the position a dimensions of the second subplot
        axis2.plot(self.t, t_part_u, color="tab:purple", label="time constant due to \'batch\' ")
        axis2.plot(self.t, t_batch, color="tab:red", label="time constant due to input")

        axis2.set_ylabel(r"$\tau$ [h]")  # Label at the Y-Axis
        axis2.set_xlabel("Time [h]")  # Label at the X-Axis
        axis2.grid()  # Turns the grid on
        axis2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),  # Shows the legend
                     ncol=1, fancybox=True, shadow=True)

        title = "INA_analysis" if title is None else title

        if path is not None:
            full_path = os.path.join(os.path.abspath(path), f"{title}.png")

            # save plot
            plt.savefig(full_path)
        else:
            plt.show()


if __name__ == '__main__':
    import numpy as np

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
