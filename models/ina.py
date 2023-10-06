import os

import numpy as np
import matplotlib.pyplot as plt

from models import Model
from utils.logger import CustomLogger

my_logger = CustomLogger()


class INA(Model):
    """
    Implementation of INA model (k_I → ∞).
    """

    def __init__(self, model_params: dict, input_params: dict):
        """
        Create INA-model for protein refolding
        
        Parameters
        ----------
        model_params: dict
            Parameters of reaction kinetics:
            - 'an': parameter for natural form
            - 'bn': parameter (exponent) for natural form
            - 'aa': parameter for aggregated form
            - 'ba': parameter (exponent) for aggregated form
            - 'n': reaction order for aggregation
            
        input_params: dict 
            Parameters of tanks used for inflow
            - 'cSR1': concentration of solubilizate in tank 1
            - 'cDR1': concentration of denataureat in tank 1
            - 'cSR2': concentration of solubilizate in tank 1
            - 'cDR2': concentration of denataureat in tank 1
            - 'cSR3': concentration of solubilizate in tank 1
            - 'cDR3': concentration of denataureat in tank 1
        noise : tuple
            parameters for gaussian noise
        """
        super().__init__(model_params, input_params)

        self.name = 'INA'

        self.n_states = 5
        self.n_inputs = 4

        # set units for states
        self.content = ('t [h]', 'c_IL [g/L]', 'c_NL [g/L]', 'c_AL [g/L]', 'c_DL [mol/L]', 'V_L [L]')

        self.t_scale = 1.

        my_logger.debug(f"model name: {self.name}, model shape: ({self.n_states}, {self.n_inputs})")

    def __call__(self, x: np.ndarray, u: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Implementation of differential equations for INA-model. Same as SINA-model but with ki → ∞ ∀ t.
           
        Parameters
        ----------
        x : np.ndarray
            The state x consists of:
            - I : Intermediate  [g/l]
            - A : Aggregate     [g/l]
            - N : Native        [g/l]
            - D : Denataureat     [mol/l]
            - V : Volume        [g/l]  
            
        u : np.ndarray
            The input/output u consists of:
            - F_in_1 : In flow one      [l/h]
            - F_in_2 : In flow two      [l/h]
            - F_in_2 : In flow three    [l/h]
            - F_out  : Out flow         [l/h]
            
        t : np.ndarray 
            Current time (not used)
            
        Returns
        -------
        np.ndarray
            Right hand side of differential equation describing the model
        """
        # Model parameters
        if type(self.model_params) is dict:
            an, bn, aa, ba, n = self.model_params['an'], self.model_params['bn'], \
                                self.model_params['aa'], self.model_params['ba'], \
                                self.model_params['n']
        # enable to unpack lists as well
        else:
            an, bn, aa, ba, n = self.model_params
            
        # Input parameters
        cSR1, cDR1, cSR2, cDR2, cSR3, cDR3 = self.input_params['cSR1'], self.input_params['cDR1'], \
                                             self.input_params['cSR2'], self.input_params['cDR2'], \
                                             self.input_params['cSR3'], self.input_params['cDR3']

        # Input
        FR1, FR2, FR3 = u[:-1]  # In flow
        FRout = u[-1]           # Out flow

        # Definition of the model
        I = x[0]        # State: concentration of folding intermediates         [g/L]
        N = x[1]        # State: concentration of native protein                [g/L]
        A = x[2]        # State: concentration of aggregated protein            [g/L]
        D = x[3]        # State: concentration of denataureat                    [mol/L]
        V = x[4]        # State: volume of the refolding vessel                 [L]

        # Reaction rates
        kn = (an * (1 + D) ** bn)  # reaction rate for refolding
        ka = (aa * (1 + D) ** ba)  # reaction rate for aggregation

        # Actual Model
        # diff = reaction                           + input                                 - dilution                  - output
        dIdt = - (kn * I + ka * I ** n )            + (FR1*cSR1+FR2*cSR2+FR3*cSR3) / V      - (FR1+FR2+FR3) * I / V                 # Derivative I      [g/(L h)]
        dNdt = kn * I                                                                       - (FR1+FR2+FR3) * N / V                 # Derivative N      [g/(L h)]
        dAdt = ka * I ** n                                                                  - (FR1+FR2+FR3) * A / V                 # Derivative A      [g/(L h)]
        dDdt =                                      (FR1*cDR1+FR2*cDR2+FR3*cDR3) / V        - (FR1+FR2+FR3) * D / V                 # Derivative D      [mol/(L h)]
        dVdt =                                      FR1+FR2+FR3                                                         - FRout     # Derivative V      [L/h]

        dxdt = np.asarray([dIdt, dNdt, dAdt, dDdt, dVdt]) * self.t_scale

        dxdt = dxdt if self.noise_generator is None else self.noise_generator(dxdt)

        return dxdt

    def part_x(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Partial derivative of f(x, u, t): δf/δx

        Parameters
        ----------
        x : np.ndarray
            Current state of the system

        u : np.ndarray
            Current input into the system

        t : float
            Current time (not used)

        Returns
        -------
        np.ndarray
            Left hand side of differential equation describing the system partially derived by x
        """
        if type(self.model_params) is dict:
            an, bn, aa, ba, n = self.model_params['an'], self.model_params['bn'], \
                                self.model_params['aa'], self.model_params['ba'], \
                                self.model_params['n']
        else:
            an, bn, aa, ba, n = self.model_params
        # Input parameters
        cSR1, cDR1, cSR2, cDR2, cSR3, cDR3 = self.input_params['cSR1'], self.input_params['cDR1'], \
                                             self.input_params['cSR2'], self.input_params['cDR2'], \
                                             self.input_params['cSR3'], self.input_params['cDR3']

        # Input
        FR1, FR2, FR3 = u[:-1]  # In flow
        FRout = u[-1]  # Out flow

        # Definition of the model
        I = x[0]        # State: concentration of folding intermediates         [g/L]
        N = x[1]        # State: concentration of native protein                [g/L]
        A = x[2]        # State: concentration of aggregated protein            [g/L]
        D = x[3]        # State: concentration of denataureat                    [mol/L]
        V = x[4]        # State: volume of the refolding vessel                 [L]

        zeros = np.zeros(I.shape)

        part_x = np.stack((
            [
            np.asarray([-2*(D + 1)**ba*I*aa - (D + 1)**bn*an - FR1/V - FR2/V - FR3/V,  zeros, zeros,   -(D + 1)**(ba - 1)*I**2*aa*ba - (D + 1)**(bn - 1)*I*an*bn,      FR1*(I - cSR1)/V**2 + FR2*(I - cSR2)/V**2 + FR3*(I - cSR3)/V**2]),
            np.asarray([(D + 1)**bn*an, -FR1/V - FR2/V - FR3/V, zeros, (D + 1)**(bn - 1)*I*an*bn, FR1*N/V**2 + FR2*N/V**2 + FR3*N/V**2]),
            np.asarray([2*(D + 1)**ba*I*aa, zeros, -FR1/V - FR2/V - FR3/V, (D + 1)**(ba - 1)*I**2*aa*ba, A*FR1/V**2 + A*FR2/V**2 + A*FR3/V**2]),
            np.asarray([zeros, zeros, zeros,  -FR1/V - FR2/V - FR3/V, (D - cDR1)*FR1/V**2 + (D - cDR2)*FR2/V**2 + (D - cDR1)*FR3/V**2]),
            np.asarray([zeros, zeros, zeros, zeros, zeros])
            ]
        ))

        return np.rollaxis(part_x, 2)

    def part_u(self, x: np.ndarray, u: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Partial derivative of f(x, u, t): δf/δu

        Parameters
        ----------
        x : np.ndarray
            Current state of the system

        u : np.ndarray
            Current input into the system

        t : float
            Current time (not used)

        Returns
        -------
        np.ndarray
            Left hand side of differential equation describing the system partially derived by u
        """
        # Definition of the model
        I = x[0]        # State: concentration of folding intermediates         [g/L]
        N = x[1]        # State: concentration of native protein                [g/L]
        A = x[2]        # State: concentration of aggregated protein            [g/L]
        D = x[3]        # State: concentration of denataureat                    [mol/L]
        V = x[4]        # State: volume of the refolding vessel                 [L]

        cSR1, cDR1, cSR2, cDR2, cSR3, cDR3 = self.input_params['cSR1'], self.input_params['cDR1'], \
            self.input_params['cSR2'], self.input_params['cDR2'], \
            self.input_params['cSR3'], self.input_params['cDR3']

        zeros = np.zeros(I.shape)
        ones = np.ones(I.shape)

        part_u = np.stack((
            [
            np.asarray([-(I - cSR1) / V, -(I - cSR2) / V, -(I - cSR3) / V, zeros]),
            np.asarray([-N / V, -N / V, -N / V, zeros]),
            np.asarray([-A / V, - A / V, -A / V, zeros]),
            np.asarray([-(D - cDR1) / V, -(D - cDR2) / V, -(D - cDR1) / V, zeros]),
            np.asarray([ones, ones, ones, ones])
            ]

        ))

        return np.rollaxis(part_u, 2)

    def stationary_point(self, S_in: float, D: float, tau: np.ndarray, V=1.) -> np.ndarray:
        """
        Calculate stationary state for given in flow and S_in and D-concentration as well as process time.

        Parameters
        ----------


        Returns
        -------
        np.ndarray:
            State-space vector of stationary point
        """
        a_n, b_n, a_a, b_a, n = self.model_params['an'], self.model_params['bn'], \
                            self.model_params['aa'], self.model_params['ba'], \
                            self.model_params['n']

        k_a = a_a * (D + 1) ** b_a
        k_n = a_n * (D + 1) ** b_n

        I = -1/2*(k_n*tau - np.sqrt(k_n**2*tau**2 + 2*(2*S_in*k_a + k_n)*tau + 1) + 1)/(k_a*tau)
        N = k_n * I * tau
        A = k_a * I ** 2 * tau

        V = np.ones(I.shape)*V

        return np.asarray([I, N, A, D, V])

    def homogenous_solution(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Return analytical calculation of the homogenous solution of the SINA- model (U_in = 0 ∀ t).

        Parameters
        ----------
        x0 : np.ndarray
            Initial state
        t : np.ndarray
            Time grid for calculation

        Returns
        -------
        np.ndarray
            State trajectory along the given time grid
        """
        # unravel states
        I_0 = x0[0]
        N_0 = x0[1]
        A_0 = x0[2]
        D_0 = x0[3]
        V_0 = x0[4]

        # unravel model parameters
        an, bn, aa, ba, n = self.model_params['an'], self.model_params['bn'], \
                            self.model_params['aa'], self.model_params['ba'], \
                            self.model_params['n']

        # calculate
        k_a = aa * (D_0 + 1) ** ba
        k_n = an * (D_0 + 1) ** bn

        C = (1+I_0*k_a/k_n)
        k_tilde = -1/C*k_a/k_n

        I = 1/(C*np.exp(t*k_n) - k_a/k_n)
        N = N_0 + -k_n/k_a*(t*k_n - np.log(np.exp(t*k_n)+k_tilde) + np.log(1+k_tilde))
        A = A_0 + I_0-I-N

        return np.asarray([I, N, A, D_0, V_0])

    def calc_time_dynamics(self, x, u) -> tuple:
        """
        Calculate the current dynamic of the model, due to time constants of linearized system

        Parameters
        ----------
        x : np.ndarray of shape (t_steps, states) or (states)
            given state can be a single column vector or a matrix of column vectors

        u : np.ndarray of shape (t_steps, states) or (states)
            given input can be a single column vector or a matrix of column vectors

        Returns
        -------
        tuple
        - t_int: dynamic based on state of system
        - t_flow: dynamic based on in flow
        """
        # get model parameters
        an, bn, aa, ba, n = self.model_params['an'], self.model_params['bn'], \
                            self.model_params['aa'], self.model_params['ba'], \
                            self.model_params['n']

        # unravel given states
        I = x[:, 0] if len(x.shape) == 2 else x[0]
        D = x[:, 3] if len(x.shape) == 2 else x[3]
        V = x[:, 4] if len(x.shape) == 2 else x[4]

        # calculate time constants
        tau_int = -2*aa * (1 + D) ** ba * I - an * (1 + D) ** bn
        tau_flow = - np.sum(u, axis=1) / V if len(x.shape) == 2 else - sum(u)/V

        return tau_int, tau_flow

    def plot_process(self, t: np.ndarray, x: np.ndarray, u: np.ndarray, ref: tuple = None, u_points: tuple = None,
                     first_order_input=False, plot_dynamics=False,
                     title: str = None, save_path: str = None):
        """
        Plotting the process by using the simulated or measured values

        Parameters
        ----------
        t : np.ndarray
            time grid of simulation/measurement
        x : np.ndarray
            simulated or estimated states of simulation/measurement
        u : np.ndarray
            simulated or used input of simulation/measurement
        ref : np.ndarray
            optional: reference values to plot over, could set-points to reach or measured points
        u_points : tuple

        title : str
            optional, title for the plot
        save_path : str
            optional, if given plot will be saved to the path instead of shown
        first_order_input : bool
            optional: use first order interpolation to plot inputs
        plot_dynamics : bool
            optional: plot linearized time constant

        Returns
        -------
        None
        """
        # separate states
        concentrations = x[:, :3]
        D = x[:, 3]
        V = x[:, 4]

        # separate in and outflow
        u_in = u[:, :-1]
        u_out = u[:, -1]

        # Create a 4x4 grid of subplots
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 15))

        # -------------- plot concentrations --------------
        axs[1, 0].plot(t, concentrations[:, 0], label="I [g/L]", color="y", linewidth=3.)
        axs[1, 0].plot(t, concentrations[:, 1], label="N [g/L]", color="g", linewidth=3.)
        axs[1, 0].plot(t, concentrations[:, 2], label="A [g/L", color="r", linewidth=3.)
        ax_D = axs[1, 0].twinx()
        ax_D.plot(t, D, label="D [mol/l]", color="k", linewidth=3.)

        axs[1, 0].set_title("Concentrations", fontsize=20)
        axs[1, 0].set_xlabel("Time [h]", fontsize=14)
        axs[1, 0].set_ylabel("Concentration [g/L]", fontsize=14)
        ax_D.set_ylabel("D [mol/L]", fontsize=14)

        axs[1, 0].grid(visible=True)

        # -------------- plot flows --------------

        # select u-plotter according to type of input
        u_plotter = axs[0, 0].plot if first_order_input else axs[0, 0].step

        # plot inputs
        u_plotter(t, u_in[:, 0], label="S and D [L/h]", color="c", linewidth=3., where='pre')
        u_plotter(t, u_in[:, 1], label="Only D [L/h]", color="m", linewidth=3., where='pre')
        u_plotter(t, u_in[:, 2], label="Buffer [L/h]", color="black", linewidth=3., where='pre')

        axs[0, 0].set_title("In flow", fontsize=20)
        axs[0, 0].set_ylabel("Flow [L/h]", fontsize=14)
        axs[0, 0].set_xlabel("TIme [h]", fontsize=14)

        axs[0, 0].grid(visible=True)

        # -------------- plot volume --------------

        axs[1, 1].plot(t, V, label="V [L]", color="blue", linewidth=3.)

        axs[1, 1].set_title("Reactor Volume", fontsize=20)
        axs[1, 1].set_xlabel("Time [h]", fontsize=14)
        axs[1, 1].set_ylabel("Volume [L]", fontsize=14)

        axs[1, 1].grid(visible=True)

        # -------------- plot outputs --------------

        # select plotter
        u_plotter = axs[0, 1].plot if first_order_input else axs[0, 1].step
        u_plotter(t, u_out, label="Bleed [l/h]", color="grey", linewidth=3., where='pre')

        axs[0, 1].set_title("Bleed", fontsize=20)
        axs[0, 1].set_xlabel("Time [h]", fontsize=14)
        axs[0, 1].set_ylabel("Flow[L/h]", fontsize=14)

        axs[0, 1].grid(visible=True)

        # Set the title and axis labels for the entire figure
        fig.suptitle('')

        # ------------------ plot u-points ------------------

        if u_points is not None:
            u_ref = u_points[1]
            t_u_ref = u_points[0]

            # plot references
            axs[0, 0].scatter(t_u_ref, u_ref[:, 0], color="c")
            axs[0, 0].scatter(t_u_ref, u_ref[:, 1], color="m")
            axs[0, 0].scatter(t_u_ref, u_ref[:, 2], color="black")
            axs[0, 1].scatter(t_u_ref, u_ref[:, 3], color="grey")

        # -------------- plot reference values --------------

        if ref is not None:

            # unravel reference state h
            I_ref = ref[1][:, 0]
            N_ref = ref[1][:, 1]
            A_ref = ref[1][:, 2]
            D_ref = ref[1][:, 3]
            V_ref = ref[1][:, 4]

            # unravel reference time
            t_ref = ref[0]

            # plot references
            axs[1, 0].scatter(t_ref, I_ref, color="y")
            axs[1, 0].scatter(t_ref, N_ref, color="g")
            axs[1, 0].scatter(t_ref, A_ref, color="r")
            ax_D.scatter(t_ref, D_ref, color="k")

            axs[1, 1].scatter(t_ref, V_ref, color="blue")

        # -------------- plot dynamics --------------
        if plot_dynamics:
            dyn_int, dyn_flow = self.calc_time_dynamics(x, u)

            dyn_axis = axs[0, 0].twinx()

            dyn_axis.plot(t, dyn_int, label="internal dynamics")
            dyn_axis.plot(t, dyn_flow, label="flow dynamics")

            dyn_axis.set_ylabel("Linearized time-constant [1/h]", fontsize=14)

        # Display the figure
        axs[0, 0].legend(loc="upper left")
        axs[1, 0].legend(loc="upper left")
        axs[0, 1].legend()
        axs[1, 1].legend()
        ax_D.legend(loc="lower right")

        title = "refolding_plot.png" if title is None else title
        title = title if title.split()[-1] == "png" else f"{title.split()[0]}.png"
        if save_path is not None:
            plt.savefig(os.path.join(save_path, title))
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    # example parameters
    mod_pars = {'an': 2, 'bn': -6, 'aa': 20, 'ba': -10, 'n': 2}
    inp_pars = {'cSR1': 7, 'cDR1': 4, 'cSR2': 0, 'cDR2': 0, 'cSR3': 0, 'cDR3': 0.}

    # model instantiation
    example_model = INA(mod_pars, inp_pars)
    print(example_model)
