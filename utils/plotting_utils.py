import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# type hints
from processes import Process, Optimization

from utils.output_utils import unpickle_process, unpickle_dir, calculate_yields
from utils.project_management import find_files_with_extension


def plot_reaction_rates(process: str | Process, save_path=None) -> None:
    """
    Plot the reaction rate of a given process or pickled process with given path

    Parameters
    ----------
    process : Process | str
        process or path to pickled process
    save_path : str
        optional location to save png to instead of showing it

    Returns
    -------
    None
    """
    # load process if string to pickle given
    if isinstance(process, str):
        path = os.path.abspath(process)
        process = unpickle_process(path)

    # get Denataureat
    x = process.x
    D = x[:, 3]

    # get time grid
    t = process.t

    # unpack model parameters
    model_params = process.controlled_plant.plant.model_params
    an, bn, aa, ba, n = model_params['an'], model_params['bn'], \
        model_params['aa'], model_params['ba'], \
        model_params['n']

    # calculate reaction rate
    kn = (an * (1 + D) ** bn)
    ka = (aa * (1 + D) ** ba)

    plt.plot(t, kn, label="k_n")
    plt.plot(t, ka, label="k_a")

    title = "Reaction rates"

    plt.title(title, size=16)
    plt.xlabel("t [h]", size=12)
    plt.ylabel("Reaction-rates [1/h]", size=12)

    plt.legend()

    _plot_or_show(plt, title, save_path)


def plot_dynamics(process: str | Process):
    """
    Plot the linearized dynamics of a pickled process

    Parameters
    ----------
    process : str
        path to process pickle

    Returns
    -------
    None
    """
    if isinstance(process, str):
        path = os.path.abspath(process)
        process = unpickle_process(path)

    x, u = process.x, process.u
    t = process.t

    tau_int, tau_ext = process.controlled_plant.plant.calc_time_dynamics(x, u)

    plt.plot(t, tau_int, label="internal")
    plt.plot(t, tau_ext, label="external")
    plt.plot(t, tau_int + tau_ext, label="sum")

    plt.title("Linearized Dynamics", size=16)
    plt.xlabel("t [h]", size=12)
    plt.ylabel(r"Time-constant $\tau$ [1/h]", size=12)

    plt.grid(visible=True)

    plt.legend()
    plt.show()


def T_violin_plot(processes: str) -> None:
    # unpickle all processes in list
    procs = unpickle_dir(processes)

    # get all optimization durations
    T_opt = [p.t[-1] for p in procs]

    # actual plot
    plt.violinplot(T_opt)

    plt.show()


def T_hist_plot(processes: str) -> None:
    # unpickle all processes in list
    procs = unpickle_dir(processes)

    # get all optimization durations
    T_opt = [p.t[-1] for p in procs]

    # actual plot
    plt.hist(T_opt, bins=25)

    plt.show()


def time_violin_plot(processes: str) -> None:
    """
    Plot a violin plot of time necessary to optimize the processes

    Parameters
    ----------
    processes : str
        path to folder of pickled processes

    Returns
    -------
    None
    """
    # unpickle all processes in list
    procs = unpickle_dir(processes)

    # get all optimization durations
    T_opt = [p.T_opt for p in procs]

    # actual plot
    plt.violinplot(T_opt)

    plt.show()


def time_histogram(processes: str) -> None:
    """
    Plot a histogram of the time necessary to optimize the processes

    Parameters
    ----------
    processes : str
        path to folder of pickled processes

    Returns
    -------
    None
    """
    # unpickle all processes in list
    procs = unpickle_dir(processes)

    T_opt = [p.T_opt for p in procs]

    plt.hist(T_opt)

    folder_name = processes.split('/')[-1]

    plt.title(f"Time for optimization: {folder_name}")
    plt.xlabel("time [s]")
    plt.ylabel("amount")

    plt.show()


def plot_opt_success(processes: str) -> None:
    """
    Plot a bar-plot showing the optimization success of all process-pickles in given directory

    Parameters
    ----------
    processes : str
        path to folder of pickled processes

    Returns
    -------
    None
    """
    procs = unpickle_dir(processes)

    # list to count True and False values of the optimization successes
    successes = [0, 0]

    # get every process and
    for p in procs:
        if p.sol.success:
            successes[0] += 1
        else:
            successes[1] += 1

    plt.bar(("True", "False"), successes, color=("#00FF00", "r"))
    plt.yticks(range(int(0), int(max(successes) + 1), 5))

    plt.title("Success of optimization", size=16)

    plt.show()


def yield_time_scatter(processes: str):
    procs = unpickle_dir(processes)

    yield_list = [calculate_yields(p)[0] for p in procs]
    T_list = [p.t[-1] for p in procs]

    # get set-point and end-point from every process
    x_star = [p.x_star for p in procs]
    x_end = [p.x[-1] for p in procs]

    # numpyize
    x_star = np.asarray(x_star)
    x_end = np.asarray(x_end)

    # difference to plot
    x_diff = x_end - x_star
    diff_norm = np.linalg.norm(x_diff, axis=1)

    plt.title("Transient Time for calculated State")
    plt.xlabel("Yield [g/g]")
    plt.ylabel("Transient time [h]")

    plt.scatter(yield_list, T_list, c=diff_norm)

    cbar = plt.colorbar()
    cbar.set_label('set-point diff')

    plt.show()


def yield_scatter(processes: str, plot_area=False, plot_weight_rel=False, plot_prc_time=False, save_path=None) -> None:
    """
    Scatter plot the yields of all the processes in a given directory

    Parameters
    ----------
    processes : str
        folder with pickle data from processes
    save_path : str
        optional path to save plot to, will just be shown if not given
    plot_area : bool
        plot feasible yield/sty area under scatter plot
    plot_weight_rel : bool
        color-code the weight ratio in scatter plot

    Returns
    -------
    None
    """
    plt.figure(1, (12, 6))

    if plot_prc_time and plot_weight_rel:
        print("only process time is printed")
    # unpickle all processes in list
    procs = unpickle_dir(processes)

    # calculate and get yields for every element in list
    yield_list = [calculate_yields(p) for p in procs]
    yield_arr = np.asarray(yield_list)
    sort_args = np.argsort(yield_arr[:, 0])
    yield_arr = yield_arr[sort_args]

    colors = None
    if plot_weight_rel:
        weights = [p.error_func.weights for p in procs]
        k_u = np.squeeze(np.asarray([w[0] for w in weights]))
        k_phi = np.squeeze(np.asarray([w[1] for w in weights]))

        k_u = k_u[sort_args]
        k_phi = k_phi[sort_args]

        colors = np.log10(k_u / k_phi)

        color_bar_label = "weight-ratio (log10)"

    if plot_prc_time:
        colors = [p.t[-1] for p in procs]
        colors = np.asarray(colors)
        colors = colors[sort_args]

        color_bar_label = "Time [h]"

    plt.scatter(yield_arr[:, 0], yield_arr[:, 1], c=colors, cmap="plasma_r")

    if plot_area:
        T_tup = (0, 2, 4, 6, 8)
        for i, T in enumerate(T_tup):
            # calculate and get yields for every element in list
            yield_list = [calculate_yields(p, dead_time=T) for p in procs]
            yield_arr = np.asarray(yield_list)
            yield_arr = yield_arr[sort_args]

            peak_arg = np.argsort(yield_arr[:, 1])[-1]
            alpha_inc = 1 / len(T_tup) / 2
            plt.fill_between(yield_arr[peak_arg:, 0], yield_arr[peak_arg:, 1], y2=0, alpha=alpha_inc * i + 0.2,
                             color="tab:green", label=f"$T_{{dead}}$={str(T)}h")

    # label the plot
    folder_name = processes.split('/')[-1]
    # plt.title(f"{folder_name}", size=16)
    plt.xlabel(f"Yield [g/g]", size=18)
    plt.ylabel(f"Space-Time-Yield [g/h]", size=18)

    if plot_prc_time or plot_weight_rel:
        cbar = plt.colorbar()
        cbar.set_label(color_bar_label, fontsize=18)

    plt.legend(fontsize=14)

    _plot_or_show(plt, folder_name, save_path)


def plot_diff_histogram(processes: str, used_states=(4,), bins=10, save_path=None) -> None:
    """
    Plot a histogram for the difference between set-point and reached state

    Parameters
    ----------
    processes : str
        path to folder containing pickle data
    used_states :
        index of state to plot difference, Must be a tuple with indexes 0-4. Default is 4 (Volume)
    bins : int
        number of bins in the histogram
    save_path : str
        optional folder path, if given plot will be saved there instead of showing it

    Returns
    -------

    """
    # unpickle all processes in list
    procs = unpickle_dir(processes)

    # get set-point and end-point from every process
    x_star = [p.x_star for p in procs]
    x_end = [p.x[-1] for p in procs]

    # remove unused states
    x_star = [[x_s[u] for u in used_states] for x_s in x_star]
    x_end = [[x_e[u] for u in used_states] for x_e in x_end]

    # numpyize
    x_star = np.asarray(x_star)
    x_end = np.asarray(x_end)

    # difference to plot
    x_diff = x_end - x_star

    # labels for legend
    labels = ['I [g/l]', 'N [g/l]', 'A [g/l]', 'D [mol/l]', 'V [l]']

    # plot the histogram
    if len(x_diff.shape) == 1:
        plt.hist(x_diff.T, bins=bins, label=labels[used_states[0]])
    else:
        for i, u in enumerate(used_states):
            plt.hist(x_diff[:, i].T, bins=bins, label=labels[u])

    # === pretty up ===
    plt.legend()

    title = f"Difference to set-point"
    plt.title(title, size=16)
    plt.xlabel(f"Δx", size=12)
    plt.ylabel(f"amount", size=12)

    _plot_or_show(plt, title, save_path)


def plot_sty_hist(process: str | Process) -> None:
    process = process if isinstance(process, Process) else unpickle_process(process)

    yield_list = []
    sty_list = []
    t_list = list(process.t)

    for i, t in enumerate(t_list):
        x = process.x[i]
        I, N, A, _, _ = x

        y = N / (I + N + A)
        sty = N / t

        yield_list.append(y)
        sty_list.append(sty)

    yield_arr = np.asarray(yield_list)
    sty_arr = np.asarray(sty_list)
    t = process.t

    plt.plot(t, yield_arr, label="yield")
    plt.plot(t, sty_arr, label="sty")

    plt.legend()

    plt.show()


def plot_T(processes: str) -> None:
    processes = unpickle_dir(processes)

    T = [p.t[-1] for p in processes]

    plt.plot(T)
    plt.show()


# === helpers ===

def _plot_or_show(plot: plt, name: str, path: str | None) -> None:
    """
    Save instead of show the plot if a path is not None

    Parameters
    ----------
    plot : plt
        ready to show matplotlib object
    name : str
        title to save under
    path : str | None
        ignored if None, path to save file to otherwise

    Returns
    -------
    None
    """
    if path is None:
        plot.show()
    else:
        path = os.path.abspath(path)
        full_path = os.path.join(path, f"{name}.png")

        plot.savefig(full_path)


if __name__ == '__main__':
    pass
