# Process 

The pure process file is just made for loading and saving measurement data.

It is imported by calling

```python
from processes import Process
```


## Name Process

The process has to be instantiated and named using `name`

```python
process = Process(name)
```

## Loading plant

Before handling anything the model or plant has to be defined. It is used to handle the data loaded and to implement plotting utilities.

The plant itself has to be instantiated without any parameters from the module `controlled_plants` if the model is simply loaded.

```python
from controlled_plants import ControlledPlant

this_controlled_plant = ControlledPlant()
this_process.set_controlled_plant(this_controlled_plant)
```

Now the data can be loaded from an Excel spread-sheet located at `"this_path"` if it is formatted correctly: 

```python
this_process.load_data("this_path")
```

Afterwards (or from any child-object) the data can be saved at `"this_path"` via: 

```python
this_process.save_data("this_path")
```