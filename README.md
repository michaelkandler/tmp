[![](https://static.wixstatic.com/media/5af097_dd3aa2e8860d4837b3bd3e97faf4b8ac~mv2.png/v1/fill/w_180,h_40,al_c,q_85,usm_4.00_1.00_0.00,enc_auto/5af097_dd3aa2e8860d4837b3bd3e97faf4b8ac~mv2.png)](https://www.chasecenter.at/)

# refolding_control

The `refolding_control`-project is seperated in different modules. Every module contains a class that can be
instantiated and configured separately and them be fumbled together using a `project`-object, which acts as a kind of
builder. These `project`-objects implement different tasks and are listed below:

+ `process`: Save and process data (states, input and plant-configuration)
+ `simulation`: Simulates the plant behavior with a given input
+ `optimization`: Optimize input trajectory with given optimization configuration
+ `analysis`: Analyses plant behaviour using standard signals (step, ramp, etc.)

The configuration for these modules are documented in: `documentation/processes_configuration`

## data

Simulation and meassurement data is stored in the following convention

+ axis 0: t 
+ axis 1: Data types eg. I,N,A ...

| input value | meaning                                 	                                                   |
|-------------|---------------------------------------------------------------------------------------------|
| `x`     	   | State as a `ndarray` of shape `(t, 5)`              	                                       |
| `u`    	    | Input as an `ndarray` of shape `(t, 4)` implementing `F1,F1,F1,Fout`                      	 |
| `t`         | Time t as `ndarray` (t, )          	                                                        |

Plant configuration data is stored as dict that are hold by the plant class

| input value 	        | meaning                                       	                                       |
|----------------------|---------------------------------------------------------------------------------------|
| `model_params`     	 | Reaction rate parameters $a_i$ and $b_i$ for every rate given as `dict`             	 |
| `input_params`    	  | Concentration of inflows given as `dict`                      	                       |

## modules

Modules can be configured separately and  