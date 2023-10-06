from typing import Callable
import numpy as np

from observers import Observer


class ParticleFilter(Observer):
    """
    Implements particle filter as state-observer.
    """

    def __init__(self,
        model: Callable,
        x0: np.ndarray,
        n_particles: int,
        initial_value_uncertainty: float | np.ndarray,
        model_parameter_uncertainty: float | np.ndarray,
        measurement_uncertainty: float | np.ndarray) -> None:

        super().__init__(model, x0)

        # Setting of number of particles
        assert isinstance(n_particles, int)
        self.n_particles = n_particles

        self.particles = None

    def __call__(self):
        # Offline:
            # controller calls == FR for simulation
        # Online:
            # controller calls sets setpoints in lucullus
            # FR via softsensor from lucullus == FR for simulation

        # Check for measurement

        # no new meas: --> call controller --> callt model

        # new measurement:
            # - update at current timestamp
            # - get sampling timepoint
            # - get us for timeframe [sampling time, current time]
            # - loop over odeint simulation with saved FRs (us) --> Recalculation
        pass

    def set_uncertainties(self, T: float | np.ndarray, Q: float | np.ndarray, R: float | np.ndarray):
        """
        T: initial_value_uncertainty
        R: model_parameter_uncertainty
        Q: measurement_uncertainty
        """
        # Setting of uncertainties
        if isinstance(T, float):
            T= np.diag([T]*self.n_states)
        if isinstance(R, float):
            R = np.diag([R]*4)        
        if isinstance(Q, float):
            Q = np.diag([Q]*2)
        self.initial_value_uncertainty = T
        self.model_parameter_uncertainty = R
        self.measurement_uncertainty = Q


    def initialize_filter(self):
        particles = np.empty((self.n_particles, self.model.n_states))
        particles = np.random.multivariate_normal(mean=self.x0, cov = self.initial_value_uncertainty)

    def _predict(self):
        raise NotImplemented

    def __repr__(self):
        description = f"Particle Filter\n" \
                      f"-----------\n" \
                      f" - Number of particles: {self.n_particles}\n" \
                      f" - {self.model}"                       
        return description

if __name__ == "__main__":

    pf = ParticleFilter
    print(pf)
    print("done")