```mermaid
classDiagram
direction BT
class node5 {
    std_model_params
    t
    std_input_params
    x0
    std_x0
    controlled_plant
   __init__(self, std_x0: np.ndarray, std_model_params: dict, std_input_params: dict) 
   step_response(self, c_s_in: float, c_d_in: float, u_in: float,
                      x0: np.ndarray = None, model_params: dict = None, t_sim: float = None, silent: bool = False
                      ) 
   ramp_response(self, c_s_in: float, c_d_in: float, d_uin: float,
                      x0: np.ndarray = None, model_params: dict = None, t_sim: float = None, silent: bool = False
                      ) 
   batch_response(self, x0: np.ndarray,
                       model_params: dict = None, t_sim: float = None, silent: bool = False) 
   _simulate(self, x0: np.ndarray = None, model_params: dict = None, input_params: dict = None,
                  t_sim: float = None) 
   _reset_standards(self) 
   plot_dynamics(self, path=None, title=None) 
}
class node8 {
    sol
    solution
    bounds
    optimzer
    model
    error_func
   __init__(self, model_params: dict, input_params: dict) 
   set_minimizer(self, minimizer: str) 
   set_bounds(self, I_max, D_max, tau_max) 
   target(self, x) 
   run_optimization(self) 
   plot_cost_func(self) 
}
class node4 {
    constraint_func
    sol
    success
    bounds
    z_star
    z_0
    error_func
    weights
    minimizer
    x_star
   __init__(self, name: str) 
   set_cost_func(self, error_func: Cost) 
   set_constraint_func(self, constraint_func: Constraint) 
   set_bounds(self, *args) 
   set_x_star(self, x_star: np.ndarray) 
   set_minimizer(self, minimizer: Minimizer) 
   set_initial_guess(self, z0: np.ndarray) 
   set_weights(self, weights: tuple) 
   get_constraints(self) 
   target(self, z: np.ndarray) 
   run_optimization(self) 
   __str__(self) 
   __repr__(self) 
}
class node6 {
    t
    u
    name
    x
    units
    controlled_plant
    t_samp
    timestamp
   __init__(self, name: str) 
   set_controlled_plant(self, controlled_plant: ControlledPlant) 
   plot_results(self, save_path=None, *args, **kwargs) 
   process2df(self) 
   df2process(self, df_model_params: pd.DataFrame, df_input_params: pd.DataFrame,
                   df_names: str, df_data: pd.DataFrame) 
   save_data(self, save_dir: str, save_name=None) 
   load_data(self, path: str) 
   pd2xls(df_model_params: pd.DataFrame, df_input_params: pd.DataFrame, df_data: List[pd.DataFrame],
               df_names: List[str], save_name: str, path: str) 
   xls2pd(path: str) 
   _set_t_samp(self) 
   _create_parameters_dataframe(self) 
   _create_simulation_dataframe(self) 
   __str__(self) 
   __repr__(self) 
}
class node2 {
    sim_name
    t
    u
    x
    x_noise
    integrator
    noise_generator
    t_samp
   __init__(self, name: str) 
   set_t(self, t: float, t_samp: float) 
   run_process(self) 
   set_integrator(self, integrator: Integrator) 
   add_noise_generator(self, generator: Noise) 
   plot_results(self, noisy_values=False, reference=None, save_path=None, title=None) 
   __str__(self) 
   __repr__(self) 
}
class node1 {
    constraint_func
    z_star
    tau
    z_0
    controlled_plant
    weights
    sol
    t
    S_in
    V
    success
    name
    bounds
    error_func
   __init__(self, name: str) 
   set_volume(self, V: float) 
   set_controlled_plant(self, controlled_plant: ControlledPlant) 
   set_cost_func(self, error_func: StationaryError) 
   set_constraint_func(self, constraint_func: StationaryConstraint) 
   set_bounds(self, pump_limits: list) 
   set_initial_guess(self, z0: np.ndarray) 
   set_weights(self, weights: tuple) 
   set_t(self, t: float, t_samp: float) 
   get_constraints(self) 
   target(self, z: np.ndarray) 
   run_optimization(self) 
   run_process(self) 
   plot_cost_func(self, opt_point=True, save_path=None, save_name=None) 
   _plot_cost_one_pump(self, opt_point=True, grid_density=100) 
   _plot_cost_two_pumps(self, opt_point=True, grid_density=100) 
   _plot_cost_three_pumps(self, opt_point=True, grid_density=10) 
   _get_D(self, z: np.ndarray) 
   _get_S(self, z: np.ndarray) 
   _get_labels(self, ind: int) 
   __str__(self) 
}
class node0 {
    constraint_func
    error_func
   set_cost_func(self, error_func: NormOptError) 
   set_constraint_func(self, constraint_func: NormOptConstraints) 
}
class node9 {
    t_k
    constraint_func
    z_star
    u_star
    controlled_plant
    weights
    zero_order
    sol
    t
    u
    used_pumps
    bounds
    error_func
   __init__(self, name: str) 
   set_controlled_plant(self, controlled_plant: ControlledPlant) 
   set_cost_func(self, error_func: DynamicError) 
   set_constraint_func(self, constraint_func: DynamicConstraint) 
   set_bounds(self, pump_limits: list) 
   set_weight(self, weights: tuple | list) 
   set_t(self, t_end: float, n_step: int, mode='lin') 
   get_constraints(self) 
   target(self, z: np.ndarray) 
   run_optimization(self) 
   plot_results(self, noisy_values=False, reference=None, save_path=None, title=None) 
   _create_full_u(self, z) 
   __str__(self) 
   __repr__(self) 
}
class node7
class node3 {
    constraint_func
    z_star
    tau
    z_0
    u_star
    controlled_plant
    weights
    zero_order
    sol
    t
    u
    used_pumps
    bounds
    T_star
    error_func
   __init__(self, name: str) 
   set_controlled_plant(self, controlled_plant: ControlledPlant) 
   set_cost_func(self, error_func: TimeOptError) 
   set_constraint_func(self, constraint_func: DynamicConstraint) 
   set_bounds(self, pump_limits: list) 
   set_initial_guess(self, z0: np.ndarray) 
   set_weight(self, weights: tuple | list) 
   set_tau(self, n_step: int, mode='lin') 
   get_constraints(self) 
   target(self, z: np.ndarray) 
   run_optimization(self) 
   plot_results(self, noisy_values=False, reference=None, save_path=None, title=None) 
   _create_full_u(self, z) 
   __str__(self) 
}

node2  -->  node5 
node4  -->  node8 
node2  -->  node4 
node6  -->  node2 
node4  -->  node1 
node9  -->  node0 
node4  -->  node9 
node3  -->  node7 
node4  -->  node3

```
