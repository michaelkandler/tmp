# Analysis

The analysis object is just a wrapper to quickly do dynamic analysis of the plant with some standard signals

To use it the Model first has to be instantiated with standard parameters and then standard-signals can be used to see the model behavior

## Base configuration

For the Analysis of the INA model only needs a standard parametrisation

```python
from processes import Analysis

stdx0 = [0., 0., 0., 0., 0.8]
std_mod_params = {'an': 0.776796168, 'bn': -6.320839249, 'aa': 2.180377932, 'ba': -8.553075612, 'n': 2}
std_in_params = {'cSR1': 11 / 4, 'cDR1': 4 / 4, 'cSR2': 0, 'cDR2': 0, 'cSR3': 0, 'cDR3': 0.}

analysis_obj = Analysis(np.asarray(stdx0), std_mod_params, std_in_params)
```

## Do analysis

Now the standard-signals can be simulated: 

+ `self.batch_process`
+ `self.step_response`
+ `self.ramp_response`

### Batch

```python

# set model parameters
x_0 = np.asarray([0., 0., 0., 0., 0.5])
mod_pars = {'an': 0.776796168, 'bn': -6.320839249, 'aa': 2.180377932, 'ba': -8.553075612, 'n': 2}

# set simulation time 
t_sim = 24

analysis_obj.batch_process(x_0, mod_pars, t_sim, silent=True)
analysis_obj.plot_dynamics()
```
### Step response

```python
# set model parameters
x_0 = np.asarray([0., 0., 0., 0., 0.5])
mod_pars = {'an': 0.776796168, 'bn': -6.320839249, 'aa': 2.180377932, 'ba': -8.553075612, 'n': 2}

# set input parameters
s_in = 11.
d_in = 4.
u_in = 0.005

# set simulation time 
t_sim = 24

analysis_obj.step_response(s_in, d_in, u_in, x_0, mod_pars, t_sim, silent=True)
analysis_obj.plot_dynamics()
```

### Ramp response

```python

# set model parameters
x_0 = np.asarray([0., 0., 0., 0., 0.5])
mod_pars = {'an': 0.776796168, 'bn': -6.320839249, 'aa': 2.180377932, 'ba': -8.553075612, 'n': 2}

# set input parameters
s_in = 11.
d_in = 4.
u_in = 0.005

# set simulation time 
t_sim = 24

analysis_obj.ramp_response(s_in, d_in, u_in, x_0, mod_pars, t_sim, silent=True)
analysis_obj.plot_dynamics()
```

## Plotting 

Alle the plots will be shown automatically unless `silent=True`, otherwise `self.plot_dynamics` can be called to show the process dynamics

```python
analysis_obj_dymamics()
```

