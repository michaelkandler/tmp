import os.path
from time import time
from typing import List, Tuple
import datetime

import pandas as pd
import numpy as np

from controlled_plants import ControlledPlant

from models import INA

from utils.logger import CustomLogger
my_logger = CustomLogger()


class Process:
    """
    Simple generic form of a process. Will be overwritten with simulation, controller, etc.

    Implements:
        - Loading data
        - Saving data
        - Converting data from and to pandas dataframes
        - Plotting data
    """

    def __init__(self, name: str) -> None:
        """
        Parameters
        ----------
        name : str
            name of the simulation
        """

        my_logger.debug("creating process...")

        # convert name to naming convection
        name = name.replace(' ', '_')
        name = name.lower()

        self.name = name
        self.timestamp = time()

        # Process data
        self.t = None
        self.x = None
        self.u = None

        # sampling time
        self.t_samp = None

        self.controlled_plant = None

        self.units = \
            {'model': None,
             'cSR1': '[g/L]', 'cDR1': '[mol/L]',
             'cSR2': '[g/L]', 'cDR2': '[mol/L]',
             'cSR3': '[g/L]', 'cDR3': '[mol/L]',
             'n': '[]', 'm': '[]',
             'an': '[g/mol/h]', 'bn': '[1]',
             'aa': '[g/mol/h]', 'ba': '[1]',
             'ai': '[g/mol/h]', 'bi': '[1]',
             'amf': '[g/mol/h]', 'bmf': '[1]',
             'amr': '[g/mol/h]', 'bmr': '[1]',
             'FRw1': '[mL/h]', 'Feed1': '[h]',
             'FRw2': '[mL/h]', 'Feed2': '[h]',
             'FRw3': '[mL/h]', 'Feed3': '[h]',
             'FRout': '[mL/h]'
             }

    # === builder options ===

    def set_controlled_plant(self, controlled_plant: ControlledPlant) -> None:
        """
        Set the controlled plant to be used by the process.

        Parameters
        ----------
        controlled_plant : ControlledPlant
            Controller to be used

        Returns
        -------

        """
        my_logger.debug("set controller...")
        self.controlled_plant = controlled_plant

    # === utilities ===

    def plot_results(self, save_path=None, *args, **kwargs) -> None:
        """
        Plot results obtained by simulation

        Parameters
        ----------
        save_path: str | None
            Save directory

        Returns
        -------

        """
        if self.controlled_plant.plant is None:
            raise ValueError('No plotter defined in plant')
        if any(i is None or i is [] for i in [self.t, self.x, self.u]):
            raise ValueError('Data must be available to plot')

        self.controlled_plant.plant.plot_process(self.t, self.x, self.u, **kwargs)

    def process2df(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Wrapper for create_model_dataframe and create_simulation_dataframe

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Tuple of pandas dataframes containing
            - model parameters
            - input parameters
            - process data

        """
        df_model_params, df_input_params = self._create_parameters_dataframe()
        df_data = self._create_simulation_dataframe()
        my_logger.debug("building pandas dataframe ...")
        return df_model_params, df_input_params, df_data

    def df2process(self, df_model_params: pd.DataFrame, df_input_params: pd.DataFrame,
                   df_names: str, df_data: pd.DataFrame) -> None:
        """
        Write data from dataframes to process.

        Parameters
        ----------
        df_model_params : pd.dataframe
            Dataframe of model parameters obtained with the according function
        df_input_params : pd.dataframe
            Dataframe of input parameters obtained with the according function
        df_data : pd.dataframe
            Dataframe of simulation/measured data obtained with the according function
        df_names : List[str]
            Dataframe of model parameters obtained with the according function

        Returns
        -------
        None
        """
        my_logger.debug("creating process from dataframe")

        if self.controlled_plant is None:
            my_logger.exception('Model must be defined for data loading')
            raise AttributeError('Model must be defined for data loading')

        model_params_list = df_model_params.values.tolist()
        input_params_list = df_input_params.values.tolist()

        input_params = {param[0]: param[2] for param in input_params_list}

        # check if model params is not empty --> for parametrization
        if model_params_list:
            model_params = {param[0]: param[2] for param in model_params_list}
            model_params.pop('model')
        else:
            my_logger.info('Model parameters not given, standard is kept. Call \'.set_model()\' to overwrite')
            model_params = None

        self.controlled_plant.plant = INA(model_params, input_params)

        ind_u = len(list(df_data.keys())) - sum(['u_' in name for name in list(df_data.keys())])

        data_array = np.asarray(df_data).astype(np.float64)
        self.t = data_array[:, 0]
        self.x = data_array[:, 1:ind_u]
        self.u = data_array[:, ind_u:]

        self.name = df_names

    def save_data(self, save_dir: str, save_name=None) -> None:
        """
        Save obtained data in Excel format

        Parameters
        ----------
        save_dir : str
            Directory to save Excel file to
        save_name : str
            Name for Excel file, self.name will be used if not given

        Returns
        -------

        """
        if save_name is None:
            save_name = self.name

        df_model_params, df_input_params, df_data = self.process2df()

        my_logger.debug(f"saving process {save_name} to {os.path.abspath(save_dir)}...")

        self.pd2xls(df_model_params, df_input_params, [df_data], [self.name], save_name, path=save_dir)

    def load_data(self, path: str) -> None:
        """
        Load simulated or measured data

        :param path:
        Parameters
        ----------
        path : str
            path to excel data

        Returns
        -------
        None
        """
        my_logger.debug(f"loading data from {os.path.abspath(path)}...")

        df_model_params, df_input_params, df_data, df_name = self.xls2pd(path)
        self.df2process(df_model_params, df_input_params, df_data, df_name)
        self._set_t_samp()

    @staticmethod
    def pd2xls(df_model_params: pd.DataFrame, df_input_params: pd.DataFrame, df_data: List[pd.DataFrame],
               df_names: List[str], save_name: str, path: str) -> None:
        """
        Save simulation dataframes in an Excel sheet.

        Parameters
        ----------
        df_model_params : pd.DataFrame
            Dataframe of model parameters.
        df_input_params : pd.DataFrame
            Dataframe of input parameters.
        df_data : List[pd.DataFrame]
            Simulation data to be saved in the spreadsheet.
        df_names : List[str]
            Names for the single spreadsheets.
        save_name : str
            Name of the file.
        path : str
            Path to save the Excel file.

        Returns
        -------
        None
        """
        file_name = save_name + '.xlsx'

        # create saving path os independently
        path = os.path.abspath(path)
        save_path = os.path.join(path, file_name)

        dfs = {name: value for name, value in zip(df_names, df_data)}
        dfs["model_parameters"] = df_model_params
        dfs["input_parameters"] = df_input_params

        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            for sheet_name, df in dfs.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def xls2pd(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame], List[str]]:
        """
        Load xls data into dataframes.

        Parameters
        ----------
        path : str
            Path to Excel file.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame], List[str]]
            A tuple containing the following:
            - pd.DataFrame: dataframe of model parameters.
            - pd.DataFrame: dataframe of input parameters.
            - List[pd.DataFrame]: list of dataframes for simulation/experimental data.
            - List[str]: list of simulation/experiment names.
        """
        sheets_dict = pd.read_excel(path, sheet_name=None)

        if len(sheets_dict) != 3:
            my_logger.warning("too many sheets found in excel sheet, only using the first three ones...")
            sheets_dict = {k: sheets_dict[k] for k in list(sheets_dict)[:3]}

        if 'model_parameters' not in sheets_dict:
            raise FileNotFoundError("no model parameters in excel sheet")

        if 'input_parameters' not in sheets_dict:
            raise FileNotFoundError("no input parameters in excel sheet")

        df_model_params = sheets_dict['model_parameters']
        sheets_dict.pop('model_parameters')

        df_input_params = sheets_dict['input_parameters']
        sheets_dict.pop('input_parameters')

        (df_data, df_names), = sheets_dict.items()

        return df_model_params, df_input_params, df_data, df_names

    # === helper functions ===

    def _set_t_samp(self) -> np.ndarray | None:
        """
        Set the sampling distance of a vector if the is evenly spaced

        Returns
        -------
        np.ndarray | None
            Calculated sampled time vector if successful else None
        """
        t_a = np.roll(self.t, 1)[1:]
        t_b = self.t[1:]

        t_samp = t_b[0] - t_a[0]
        t_diff = t_b - t_a

        if np.all(t_diff == t_samp):
            self.t_samp = t_samp
            my_logger.debug(f"sampling time set to {self.t_samp}")
        else:
            my_logger.info("No sampling time could be found due to non-equidistant time vector")

        return t_samp if np.all(t_diff == t_samp) else None

    def _create_parameters_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates a dataframe with the simulation parameters of the given process. Will return empty dataframe of no
        parameters are given.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Dataframe of model parameters
        """
        my_logger.debug(f"loading parameters from pandas frame...")

        if isinstance(self.controlled_plant.plant, INA):
            model = self.controlled_plant.plant.name
            model_params = self.controlled_plant.plant.model_params
            input_params = self.controlled_plant.plant.input_params
        else:
            model = self.controlled_plant.observer.model.name
            model_params = self.controlled_plant.observer.model.model_params
            input_params = self.controlled_plant.observer.model.input_params

        dic_input_params = {k: [self.units[k], v] for k, v in input_params.items()}
        df_input_params = pd.DataFrame(dic_input_params).transpose()

        dic_model_params = {k: [self.units[k], v] for k, v in model_params.items()}
        dic_model_params['model'] = [None, model]
        df_model_params = pd.DataFrame(dic_model_params).transpose()

        return df_model_params, df_input_params

    def _create_simulation_dataframe(self) -> pd.DataFrame:
        """
        Convert numpy array from simulation in standard pandas frame using the headers defined in data_formats.py.
        Also add concretion of overall protein (PL) after single protein contraction.

        Returns
        -------
        pd.DataFrame
            list pandas dataframes of simulation  data
        """
        my_logger.debug(f"loading data from pandas frame...")

        t = self.t
        x = self.x

        df_header = self.controlled_plant.plant.content

        df_sim_data = {}

        # iterate over every simulation state
        for j, k in enumerate(df_header):

            if j == 0:
                df_sim_data[k] = t
            else:
                df_sim_data[k] = x[:, j - 1]

        for u_ind in range(self.u.shape[1] - 1):
            df_sim_data[f'u_{u_ind} [L/h]'] = self.u[:, u_ind]

        df_sim_data['u_out [L/h]'] = self.u[:, -1]

        return pd.DataFrame(df_sim_data)

    # === miscellaneous ===

    def __str__(self) -> str:
        intended_controlled_plant = '\n'.join([f"\t{line}" for line in str(self.controlled_plant).splitlines()])
        creation_time = datetime.datetime.fromtimestamp(self.timestamp)
        creation_time = creation_time.strftime('%Y-%m-%d %H:%M:%S')

        description = f"\nProcess:\n" \
                      f"  - Name: {self.name}\n" \
                      f"  - created at: {creation_time}\n" \
                      f"  - Duration: {self.t[-1]} [h]\n" \
                      f"  - Sampling time: {self.t_samp} [h]\n" \
                      f"----------------------\n" \
                      f"\t{intended_controlled_plant}"

        return description

    def __repr__(self):
        pass


if __name__ == '__main__':
    # create object
    loading_data = Process('loading_saving_example')

    # build project
    loaded_plant = ControlledPlant()
    loading_data.set_controlled_plant(loaded_plant)

    try:
        # load actual measurement
        loading_data.load_data("/home/friedi/Documents/university/master_thesis/refolding_control/data/experimental_data/exp_h.xlsx")
    except FileNotFoundError:
        input_path = input('Enter file location to load ...')
        loading_data.load_data(input_path)

    # plot the loaded results
    # loading_data.plot_results(plot_dynamics=True)

    # save the same data (with the given model parameters) to a new file
    loading_data.plot_results()

    print(loading_data)
